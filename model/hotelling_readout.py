"""Explicit train-fitted Hotelling scores with two scalar calibration parameters.

The SNN supplies nonlinear task descriptors. Class-conditioned means and
covariances are fitted by the caller on training pairs, then held fixed during
task backpropagation and inference. This is alternating statistical fitting and
feature learning, not differentiation through the estimator or a claim that the
calibrated score is an exact likelihood ratio. Python 3.8 / PyTorch 1.9.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.context_evidence import ContextEvidence


class HotellingReadout(ContextEvidence):
    """Signed joint HO, second-only HO, diagonal observer, or second passthrough.

    The joint order is [second descriptor, first descriptor], where each
    descriptor is [dec1 features, original event logit]. No dimensionality
    reduction or free MLP follows the fitted discriminant direction. Only its
    positive scale and scalar bias are learned. All statistics are checkpointed.
    """
    VARIANTS = ("full", "no_context", "no_revision", "second")

    def __init__(self, feature_dim=12, variant="full", shrinkage=.05):
        nn.Module.__init__(self)
        if isinstance(feature_dim, bool) or int(feature_dim) != feature_dim or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        if variant not in self.VARIANTS:
            raise ValueError("Unknown Hotelling variant: " + str(variant))
        if (isinstance(shrinkage, bool) or not math.isfinite(float(shrinkage))
                or not 0. <= float(shrinkage) <= 1.):
            raise ValueError("shrinkage must be finite and in [0, 1]")
        self.feature_dim, self.variant = int(feature_dim), variant
        self.shrinkage = float(shrinkage)
        self.has_context = variant in ("full", "no_revision")
        self.has_revision = variant != "second"
        d = self.feature_dim + 1
        self.active_dim = 2*d if self.has_context else d
        n = self.active_dim
        self.register_buffer("raw_midpoint", torch.zeros(n))
        self.register_buffer("raw_scale", torch.ones(n))
        self.register_buffer("within_covariance", torch.zeros(n, n))
        self.register_buffer("regularized_covariance", torch.eye(n))
        self.register_buffer("class_mean_difference", torch.zeros(n))
        self.register_buffer("hotelling_template", torch.zeros(n))
        self.register_buffer("normalized_template", torch.zeros(n))
        # Fold standardization and centering into one deployment affine map.
        # The first-observation block is exactly zero for no_context.
        self.register_buffer("raw_weight", torch.zeros(2*d))
        self.register_buffer("score_bias", torch.zeros(()))
        self.register_buffer("score_normalizer", torch.ones(()))
        self.register_buffer("fisher_distance_squared", torch.zeros(()))
        self.register_buffer("second_template", torch.zeros(d))
        self.register_buffer("cross_regression", torch.zeros(d, d))
        self.register_buffer("conditional_template", torch.zeros(d))
        self.register_buffer("schur_covariance", torch.zeros(d, d))
        self.register_buffer("fisher_second", torch.zeros(()))
        self.register_buffer("fisher_conditional", torch.zeros(()))
        self.register_buffer("decomposition_error", torch.zeros(()))
        self.register_buffer("class_counts", torch.zeros(2, dtype=torch.long))
        self.register_buffer("fit_calls", torch.zeros((), dtype=torch.long))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))
        self._is_fitted = False
        if self.has_revision:
            self.scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.))))
            self.bias = nn.Parameter(torch.zeros(()))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        self._is_fitted = bool(self.fitted.detach().cpu().item())

    def calibration_scale(self):
        """Keep the calibrated score monotone even after extreme finite updates."""
        if not self.has_revision:
            return self.score_bias.new_ones(())
        return F.softplus(self.scale_raw).clamp_min(1e-6)

    @torch.no_grad()
    def fit_statistics(self, first_z, second_z, first_h, second_h, labels):
        """Fit from training observations, without changing learned calibration.

        Each class contributes half of the within-class covariance, independently
        of its sample count. Column standardization precedes diagonal shrinkage
        so units of membrane features and logits do not control the ridge.
        All estimation and solves use CPU float64. Caller must refresh after
        changing the backbone and before validating/saving the paired model.
        """
        if self.variant == "second":
            return self.statistics_summary()

        def cpu(value):
            return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)

        z1, z2, h1, h2 = [cpu(v) for v in (first_z, second_z, first_h, second_h)]
        self._validate(z1, z2, h1, h2)
        y = torch.as_tensor(labels).detach().to(device="cpu").reshape(-1)
        if y.shape != z1.shape or not bool(((y == 0) | (y == 1)).all()):
            raise ValueError("Training labels must match [N] and contain only zero/one")
        if any(not bool(torch.isfinite(v).all()) for v in (z1, z2, h1, h2)):
            raise ValueError("Training observations must be finite")
        raw2 = torch.cat((h2, z2[:, None]), 1)
        raw1 = torch.cat((h1, z1[:, None]), 1)
        raw = torch.cat((raw2, raw1), 1) if self.has_context else raw2
        means, covariances, counts = [], [], []
        for value in (0, 1):
            selected = raw[y == value]
            if selected.shape[0] < 2:
                raise ValueError("At least two training observations per class are required")
            mean = selected.mean(0)
            centered = selected - mean
            means.append(mean)
            covariances.append(centered.t().mm(centered) / selected.shape[0])
            counts.append(int(selected.shape[0]))
        raw_cov = .5 * (covariances[0] + covariances[1])
        if any(not bool(torch.isfinite(v).all()) for v in means + [raw_cov]):
            raise ValueError("Training class moments must be finite")
        midpoint = .5 * (means[0] + means[1])
        scale = raw_cov.diag().clamp_min(1e-8).sqrt()
        covariance = raw_cov / (scale[:, None] * scale[None, :])
        covariance = .5 * (covariance + covariance.t())
        diagonal = torch.diag(covariance.diag())
        ridge = 1e-5 * max(float(covariance.diag().mean()), 1e-8)
        metric = (diagonal if self.variant == "no_revision" else
                  (1.-self.shrinkage) * covariance + self.shrinkage * diagonal)
        metric = metric + ridge * torch.eye(self.active_dim, dtype=torch.float64)
        difference = (means[1] - means[0]) / scale
        if not bool(torch.isfinite(metric).all()) or not bool(torch.isfinite(difference).all()):
            raise ValueError("Standardized training moments must be finite")
        try:
            torch.linalg.cholesky(metric)
            template = torch.linalg.solve(metric, difference)
        except RuntimeError as error:
            raise ValueError("Regularized within-class covariance must be positive definite") from error
        distance = torch.dot(difference, template)
        if not bool(torch.isfinite(template).all()) or not math.isfinite(float(distance)) or float(distance) < -1e-10:
            raise ValueError("Hotelling solve produced invalid discriminant statistics")
        distance = distance.clamp_min(0.)
        # Zero class separation has no supported direction. Do not manufacture a
        # huge normalized score by dividing roundoff by a near-zero d-prime.
        normalizer = distance.sqrt().clamp_min(1e-6)
        normalized = template / normalizer if float(distance) > 1e-12 else torch.zeros_like(template)
        d = self.feature_dim + 1
        second = torch.linalg.solve(metric[:d, :d], difference[:d])
        second_distance = torch.dot(difference[:d], second)
        regression = torch.zeros(d, d, dtype=torch.float64)
        residual_template = torch.zeros(d, dtype=torch.float64)
        schur = torch.zeros(d, d, dtype=torch.float64)
        increment = torch.zeros((), dtype=torch.float64)
        if self.has_context:
            regression = torch.linalg.solve(metric[:d, :d], metric[d:, :d].t()).t()
            schur = metric[d:, d:] - regression.mm(metric[:d, d:])
            schur = .5 * (schur + schur.t())
            residual_delta = difference[d:] - regression.mv(difference[:d])
            try:
                torch.linalg.cholesky(schur)
                residual_template = torch.linalg.solve(schur, residual_delta)
            except RuntimeError as error:
                raise ValueError("Conditional within-class covariance must be positive definite") from error
            increment = torch.dot(residual_delta, residual_template)
        raw_weight = torch.zeros(2*d, dtype=torch.float64)
        raw_weight[:self.active_dim] = normalized / scale
        offset = -torch.dot(normalized / scale, midpoint)
        values = {"raw_midpoint": midpoint, "raw_scale": scale,
                  "within_covariance": covariance, "regularized_covariance": metric,
                  "class_mean_difference": difference, "hotelling_template": template,
                  "normalized_template": normalized, "raw_weight": raw_weight,
                  "score_bias": offset, "score_normalizer": normalizer,
                  "fisher_distance_squared": distance, "second_template": second,
                  "cross_regression": regression, "conditional_template": residual_template,
                  "schur_covariance": schur, "fisher_second": second_distance,
                  "fisher_conditional": increment,
                  "decomposition_error": (distance-second_distance-increment).abs(),
                  "class_counts": torch.tensor(counts, dtype=torch.long)}
        if any(not bool(torch.isfinite(v).all()) for v in values.values()):
            raise ValueError("Fitted Hotelling statistics must be finite")
        # Install only after every check succeeds, leaving an earlier valid fit
        # intact on failure. Calibration parameters deliberately remain untouched.
        installed = {}
        for name, value in values.items():
            destination = getattr(self, name)
            converted = value.to(device=destination.device, dtype=destination.dtype)
            if not bool(torch.isfinite(converted).all()):
                raise ValueError("Fitted statistics exceed the deployment dtype's finite range")
            installed[name] = converted
        for name, value in installed.items():
            getattr(self, name).copy_(value)
        self.fit_calls.add_(1)
        self.fitted.fill_(True)
        self._is_fitted = True
        return self.statistics_summary()

    def forward(self, first_z, second_z, first_h, second_h, return_info=False):
        self._validate(first_z, second_z, first_h, second_h)
        if self.variant == "second":
            result, score = second_z, second_z
        else:
            if not self._is_fitted:
                raise RuntimeError("HotellingReadout needs training-only fit_statistics before forward")
            second = torch.cat((second_h, second_z[:, None]), 1)
            if self.has_context:
                first = torch.cat((first_h, first_z[:, None]), 1)
                observed = torch.cat((second, first), 1)
            else:
                observed = second
            # torch 1.9 requires a matrix weight even for a scalar linear score.
            score = F.linear(observed, self.raw_weight[:self.active_dim].unsqueeze(0),
                             self.score_bias.reshape(1)).squeeze(1)
            result = self.calibration_scale() * score + self.bias
        if return_info:
            return result, {"alpha_first": torch.zeros_like(second_z),
                            "consensus": second_z, "delta": result-second_z,
                            "observer_score": score}
        return result

    @torch.no_grad()
    def statistics_summary(self):
        summary = {"fitted": self._is_fitted, "variant": self.variant,
                   "feature_dim": self.feature_dim, "raw_channels_per_view": self.feature_dim+1,
                   "active_channels": self.active_dim, "channel_compression": False,
                   "fit_calls": int(self.fit_calls.detach().cpu()), "shrinkage": self.shrinkage,
                   "statistics_scope": "caller-supplied training pairs; fixed during forward",
                   "normalization": "class midpoint; unit regularized score variance unless zero separation",
                   "coordinate_order": "second, first",
                   "interpretation": "regularized maximum-linear-SNR direction; not a probability or tail guarantee"}
        if not self._is_fitted:
            return summary
        metric = self.regularized_covariance.detach().to(device="cpu", dtype=torch.float64)
        eigenvalues = torch.linalg.eigvalsh(metric)
        summary.update(training_pairs=int(self.class_counts.sum().detach().cpu()),
                       class_counts=self.class_counts.detach().cpu().tolist(),
                       minimum_eigenvalue=float(eigenvalues[0]),
                       regularized_condition_number=float(eigenvalues[-1] / eigenvalues[0]),
                       fisher_joint=float(self.fisher_distance_squared.detach().cpu()),
                       fisher_second=float(self.fisher_second.detach().cpu()),
                       fisher_conditional_increment=float(self.fisher_conditional.detach().cpu()),
                       fisher_decomposition_error=float(self.decomposition_error.detach().cpu()),
                       zero_separation=bool(float(self.fisher_distance_squared.detach().cpu()) <= 1e-12),
                       scale=float(self.calibration_scale().detach().cpu()),
                       bias=float(self.bias.detach().cpu()),
                       raw_template_norm=float(self.raw_weight.norm().detach().cpu()),
                       duplicate_observation_caveat="shrinkage can create apparent Fisher gain without new information")
        return summary

    def macs_per_pair(self):
        """Folded score dot-product only; affine calibration is elementwise."""
        return self.active_dim if self.has_revision else 0
