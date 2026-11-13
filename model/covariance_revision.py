"""Train-fitted correlation coordinates for bounded contextual revision.

The fixed channel and covariance transforms use training observations only.
They expose a second-view measurement and its conditional first-view residual;
they do not claim independent evidence or calibrated Bayesian probabilities.
Python 3.8 / PyTorch 1.9 compatible.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.context_evidence import ContextEvidence


def _within(x, labels):
    means, covariance, counts = [], [], []
    for value in (0, 1):
        selected = x[labels == value]
        if selected.shape[0] < 2:
            raise ValueError("At least two training observations per class are required")
        mean = selected.mean(0)
        centered = selected - mean
        means.append(mean)
        covariance.append(centered.t().mm(centered) / selected.shape[0])
        counts.append(int(selected.shape[0]))
    return means, .5 * (covariance[0] + covariance[1]), counts


def _regularize(covariance, shrinkage):
    diagonal = torch.diag(torch.diag(covariance))
    scale = max(float(torch.diag(covariance).mean()), 1e-8)
    eye = torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device)
    return (1. - shrinkage) * covariance + shrinkage * diagonal + 1e-5 * scale * eye


class CovarianceRevision(ContextEvidence):
    """A fixed train-fitted statistical transform followed by one small head.

    full: both views and conditional whitening.
    no_context: duplicate the second view, including during statistics fitting.
    no_revision: same channels/capacity and marginal scale, without decorrelation.
    second: unchanged second score, without a fitted or trainable correction.
    """
    VARIANTS = ("full", "no_context", "no_revision", "second")
    CHANNEL_BASES = ("shared_fisher", "joint_fisher")

    def __init__(self, feature_dim=60, hidden=8, kappa=2., variant="full",
                 channels=8, shrinkage=.05, channel_basis="shared_fisher"):
        nn.Module.__init__(self)
        for name, value in (("feature_dim", feature_dim), ("hidden", hidden), ("channels", channels)):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(name + " must be a positive integer")
        if channels > feature_dim + 1:
            raise ValueError("channels cannot exceed feature_dim + 1")
        if variant not in self.VARIANTS:
            raise ValueError("Unknown covariance variant: " + str(variant))
        if channel_basis not in self.CHANNEL_BASES:
            raise ValueError("Unknown covariance channel_basis: " + str(channel_basis))
        if channel_basis == "joint_fisher" and channels < min(3, feature_dim + 1):
            raise ValueError("joint_fisher needs at least min(3, feature_dim + 1) channels")
        if not math.isfinite(float(kappa)) or kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        if not math.isfinite(float(shrinkage)) or not 0. < shrinkage <= 1.:
            raise ValueError("shrinkage must be in (0, 1]")
        self.feature_dim, self.hidden, self.channels = int(feature_dim), int(hidden), int(channels)
        self.kappa, self.variant, self.shrinkage = float(kappa), variant, float(shrinkage)
        self.channel_basis = channel_basis
        self.has_context = variant in ("full", "no_revision")
        self.has_revision = variant != "second"
        d, c = self.feature_dim + 1, self.channels
        self.register_buffer("raw_mean", torch.zeros(d))
        self.register_buffer("raw_scale", torch.ones(d))
        self.register_buffer("projection", torch.zeros(d, c))
        self.register_buffer("joint_mean", torch.zeros(2 * c))
        self.register_buffer("joint_transform", torch.eye(2 * c))
        self.register_buffer("within_covariance", torch.eye(2 * c))
        self.register_buffer("regularized_covariance", torch.eye(2 * c))
        self.register_buffer("cross_regression", torch.zeros(c, c))
        self.register_buffer("schur_covariance", torch.eye(c))
        self.register_buffer("class_mean_difference", torch.zeros(2 * c))
        self.register_buffer("class_counts", torch.zeros(2, dtype=torch.long))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))
        self._is_fitted = False
        if self.has_revision:
            self.input = nn.Linear(2 * c, self.hidden)
            self.output = nn.Linear(self.hidden, 1, bias=False)
            nn.init.zeros_(self.output.weight)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        self._is_fitted = bool(self.fitted.detach().cpu().item())

    @torch.no_grad()
    def fit_statistics(self, first_z, second_z, first_h, second_h, labels):
        """Fit once from caller-supplied training data; forward never refits.

        Labels serve only class-conditioned training moments. The legacy basis
        contains a pooled-view Fisher direction and orthogonal variance axes.
        joint_fisher retains the second-only and two joint-template blocks in
        the shared channel span, then fills with variance axes. This preserves
        those fitted *linear score functions*, not invariance of a subsequent
        lower-dimensional covariance shrinkage or final nonlinear classifier.
        All solves use CPU float64, then fixed buffers follow the model dtype.
        """
        if self.variant == "second":
            return {"variant": self.variant, "fitted": False, "reason": "passthrough"}
        def cpu(value):
            return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
        z1, z2, h1, h2 = [cpu(value) for value in (first_z, second_z, first_h, second_h)]
        self._validate(z1, z2, h1, h2)
        y = torch.as_tensor(labels).detach().to(device="cpu").reshape(-1)
        if y.shape != z1.shape or not bool(((y == 0) | (y == 1)).all()):
            raise ValueError("Training labels must have shape [N] and contain only zero/one")
        if any(not bool(torch.isfinite(value).all()) for value in (z1, z2, h1, h2)):
            raise ValueError("Training observations must be finite")
        if self.variant == "no_context":
            z1, h1 = z2, h2
        raw1, raw2 = torch.cat((h1, z1[:, None]), 1), torch.cat((h2, z2[:, None]), 1)
        shared = torch.cat((raw1, raw2), 0)
        shared_means, raw_cov, _ = _within(shared, torch.cat((y, y), 0))
        mean = .5 * (shared_means[0] + shared_means[1])
        scale = torch.diag(raw_cov).clamp_min(1e-8).sqrt()
        standardized_cov = raw_cov / (scale[:, None] * scale[None, :])
        regularized_shared = _regularize(standardized_cov, self.shrinkage)
        difference = (shared_means[1] - shared_means[0]) / scale
        fisher = torch.linalg.solve(regularized_shared, difference)
        _, eigenvectors = torch.linalg.eigh(standardized_cov)
        template_audit = {}
        templates = None
        if self.channel_basis == "joint_fisher":
            # Standardize both observations with exactly the shared coordinates
            # used at deployment; the joint order remains [second, first].
            standardized1 = (raw1 - mean) / scale
            standardized2 = (raw2 - mean) / scale
            raw_joint = torch.cat((standardized2, standardized1), 1)
            raw_means, raw_joint_covariance, _ = _within(raw_joint, y)
            raw_joint_regularized = _regularize(raw_joint_covariance, self.shrinkage)
            raw_delta = raw_means[1] - raw_means[0]
            d = len(scale)
            joint_template = torch.linalg.solve(raw_joint_regularized, raw_delta)
            second_template = torch.linalg.solve(raw_joint_regularized[:d, :d], raw_delta[:d])
            templates = (second_template, joint_template[:d], joint_template[d:])
            candidates = list(templates) + list(eigenvectors.flip(1).t())
        else:
            # Keep the old default order and numerical path unchanged.
            candidates = [fisher] + list(eigenvectors.flip(1).t())
        candidates += list(torch.eye(len(scale), dtype=torch.float64))
        basis = []
        for vector in candidates:
            vector = vector.clone()
            # Reorthogonalization prevents nearly collinear Fisher/PC axes.
            for _ in range(2):
                for axis in basis:
                    vector -= torch.dot(axis, vector) * axis
            norm = vector.norm()
            if float(norm) > 1e-8:
                basis.append(vector / norm)
            if len(basis) == self.channels:
                break
        projection = torch.stack(basis, 1)
        if templates is not None:
            centered2 = standardized2 - raw_means[0][:d]
            centered1 = standardized1 - raw_means[0][d:]
            names = ("second_marginal", "joint_second_block", "joint_first_block")
            for name, template, observed in zip(names, templates, (centered2, centered2, centered1)):
                retained = projection.mv(projection.t().mv(template))
                error = template - retained
                score_error = observed.mv(error)
                template_audit[name] = {
                    "raw_template_norm": float(template.norm()),
                    "retained_template_norm": float(retained.norm()),
                    "relative_reconstruction_error": float(error.norm() / template.norm().clamp_min(1e-12)),
                    "training_score_rmse": float(score_error.square().mean().sqrt()),
                    "training_score_max_abs_error": float(score_error.abs().max()),
                }
        compact1, compact2 = (raw1 - mean).mm(projection / scale[:, None]), (raw2 - mean).mm(projection / scale[:, None])
        # The second view comes first so the block transform is triangular.
        joint = torch.cat((compact2, compact1), 1)
        means, covariance, counts = _within(joint, y)
        regularized = _regularize(covariance, self.shrinkage)
        c = self.channels
        sigma22, sigma12 = regularized[:c, :c], regularized[c:, :c]
        regression = torch.linalg.solve(sigma22, sigma12.t()).t()
        schur = regularized[c:, c:] - regression.mm(sigma12.t())
        schur = .5 * (schur + schur.t())
        eye = torch.eye(c, dtype=torch.float64)
        inv2 = torch.linalg.solve(torch.linalg.cholesky(sigma22), eye)
        invr = torch.linalg.solve(torch.linalg.cholesky(schur), eye)
        transform = torch.zeros(2 * c, 2 * c, dtype=torch.float64)
        transform[:c, :c], transform[c:, c:] = inv2, invr
        transform[c:, :c] = -invr.mm(regression)
        values = {"raw_mean": mean, "raw_scale": scale, "projection": projection,
                  "joint_mean": means[0], "joint_transform": transform,
                  "within_covariance": covariance, "regularized_covariance": regularized,
                  "cross_regression": regression, "schur_covariance": schur,
                  "class_mean_difference": means[1] - means[0],
                  "class_counts": torch.tensor(counts, dtype=torch.long)}
        for name, value in values.items():
            destination = getattr(self, name)
            destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
        self.fitted.fill_(True)
        self._is_fitted = True
        delta = means[1] - means[0]
        second_distance = torch.dot(delta[:c], torch.linalg.solve(sigma22, delta[:c]))
        residual_delta = delta[c:] - regression.mv(delta[:c])
        residual_distance = torch.dot(residual_delta, torch.linalg.solve(schur, residual_delta))
        joint_distance = torch.dot(delta, torch.linalg.solve(regularized, delta))
        whitening_error = (transform.mm(regularized).mm(transform.t()) - torch.eye(2*c, dtype=torch.float64)).abs().max()
        # Diagnostics only: do not change the fitted transform or forward path.
        # A successful Cholesky need not imply a well-conditioned transform;
        # log amplification as well as relative conditioning for new windows.
        regularized_eigenvalues = torch.linalg.eigvalsh(regularized)
        schur_eigenvalues = torch.linalg.eigvalsh(schur)
        transform_spectral_gain = torch.linalg.norm(transform, ord=2)
        return {"fitted": True, "variant": self.variant, "training_pairs": len(y),
                "class_counts": counts, "raw_channels": self.feature_dim + 1,
                "channel_basis": self.channel_basis,
                "channel_basis_template_audit": template_audit,
                "channel_basis_guarantee": "fitted linear template span only; not shrinkage invariance or accuracy",
                "channels_per_view": c, "shrinkage": self.shrinkage,
                "whitening_error_float64": float(whitening_error),
                "minimum_eigenvalue": float(regularized_eigenvalues[0]),
                "regularized_condition_number": float(regularized_eigenvalues[-1] / regularized_eigenvalues[0]),
                "schur_minimum_eigenvalue": float(schur_eigenvalues[0]),
                "schur_condition_number": float(schur_eigenvalues[-1] / schur_eigenvalues[0]),
                "transform_spectral_gain": float(transform_spectral_gain),
                "fisher_second": float(second_distance), "fisher_conditional_increment": float(residual_distance),
                "fisher_joint": float(joint_distance),
                "fisher_decomposition_error": abs(float(joint_distance-second_distance-residual_distance)),
                "coordinate_transform": "diagonal_standardization" if self.variant == "no_revision" else "conditional_whitening",
                "statistics_scope": "caller-supplied training pairs only; frozen during forward",
                "interpretation": "regularized linear decorrelation, not independence or an error-rate guarantee"}

    def coordinates(self, first_z, second_z, first_h, second_h):
        self._validate(first_z, second_z, first_h, second_h)
        if not self._is_fitted:
            raise RuntimeError("CovarianceRevision needs training-only fit_statistics before forward")
        if self.variant == "no_context":
            first_z, first_h = second_z, second_h
        raw1 = torch.cat((first_h, first_z[:, None]), 1)
        raw2 = torch.cat((second_h, second_z[:, None]), 1)
        p = self.projection / self.raw_scale[:, None]
        first = (raw1 - self.raw_mean).matmul(p)
        second = (raw2 - self.raw_mean).matmul(p)
        joint = torch.cat((second, first), 1) - self.joint_mean
        if self.variant == "no_revision":
            # Keep marginal variance scaling matched to the full head. This
            # control removes correlation modeling, not basic standardization.
            return joint / torch.diag(self.regularized_covariance).sqrt()
        return F.linear(joint, self.joint_transform)

    def encode_pair(self, first_z, second_z, first_h, second_h):
        """Fixed training-fitted coordinates suitable for a frozen-feature cache."""
        return self.coordinates(first_z, second_z, first_h, second_h)

    def forward_encoded(self, q, second_z, return_info=False):
        if second_z.dim() != 1 or tuple(q.shape) != (len(second_z), 2*self.channels):
            raise ValueError("Encoded coordinates must have shape [N,2*channels]")
        if self.variant != "second" and not self._is_fitted:
            raise RuntimeError("CovarianceRevision needs training-only fit_statistics before forward")
        if self.variant == "second":
            delta = torch.zeros_like(second_z)
        else:
            delta = self.kappa * torch.tanh(self.output(F.silu(self.input(q))).squeeze(1))
        result = second_z + delta
        if return_info:
            return result, {"alpha_first": torch.zeros_like(second_z), "delta": delta, "consensus": second_z}
        return result

    def forward(self, first_z, second_z, first_h, second_h, return_info=False):
        self._validate(first_z, second_z, first_h, second_h)
        if self.variant == "second":
            delta = torch.zeros_like(second_z)
            result = second_z + delta
            if return_info:
                return result, {"alpha_first": delta, "delta": delta, "consensus": second_z}
            return result
        q = self.encode_pair(first_z, second_z, first_h, second_h)
        return self.forward_encoded(q, second_z, return_info)

    def variance_regularizer(self):
        """Penalize whitened linear gain and downstream rescaling together.

        For regularized unit-covariance coordinates, the first layer's squared
        Frobenius norm is its summed linear response variance. The output term
        prevents evading this penalty by inverse rescaling of the two layers.
        This is not a guarantee about nonlinear tail probabilities.
        """
        if not self.has_revision:
            return self.raw_mean.sum() * 0.
        return self.input.weight.square().mean() + self.output.weight.square().mean()

    def macs_per_pair(self):
        if not self.has_revision:
            return 0
        c, d = self.channels, self.feature_dim + 1
        projection = 2 * d * c
        whitening = 2 * c if self.variant == "no_revision" else 4 * c * c
        return int(projection + whitening + 2*c*self.hidden + self.hidden)
