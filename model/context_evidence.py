"""Lightweight correlated-view consensus and bounded conditional revision.

The two readouts refer to the same events in overlapping windows. They are
correlated predictions, not independent likelihoods. M1 forms a convex
log-opinion pool; M2 learns a bounded correction from their hidden difference.
No event owns a recurrent state, and no revision is fed into the SNN state.
Python 3.8 / PyTorch 1.9 compatible.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ContextEvidence(nn.Module):
    VARIANTS = ("full", "no_context", "no_revision", "second")

    def __init__(self, feature_dim=32, hidden=8, kappa=1.0, variant="full"):
        super().__init__()
        for name, value in (("feature_dim", feature_dim), ("hidden", hidden)):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(name + " must be a positive integer")
        if not math.isfinite(float(kappa)) or float(kappa) <= 0:
            raise ValueError("kappa must be finite and positive")
        if variant not in self.VARIANTS:
            raise ValueError("Unsupported evidence variant: " + str(variant))
        self.feature_dim, self.hidden = int(feature_dim), int(hidden)
        self.kappa, self.variant = float(kappa), variant
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_std", torch.ones(self.feature_dim))
        self.has_context = variant in ("full", "no_revision")
        self.has_revision = variant in ("full", "no_context")
        if self.has_context:
            self.reliability = nn.Sequential(
                nn.Linear(self.feature_dim + 1, self.hidden), nn.SiLU(),
                nn.Linear(self.hidden, 1))
            nn.init.zeros_(self.reliability[-1].weight)
            nn.init.zeros_(self.reliability[-1].bias)
        if self.has_revision:
            self.condition = nn.Linear(self.feature_dim + 3, self.hidden)
            self.difference = nn.Linear(self.feature_dim, self.hidden, bias=False)
            self.output = nn.Linear(self.hidden, 1, bias=False)
            # Only this projection is zero: zeroing both projections would
            # permanently block the difference path's learning signal.
            nn.init.zeros_(self.output.weight)

    @torch.no_grad()
    def set_normalization(self, mean, std):
        """Install shared moments computed from training observations only.

        Caller must pool both appearances into one coordinate system. The
        method never fits moments from a forward call or validation examples.
        """
        mean = torch.as_tensor(mean, device=self.feature_mean.device,
                               dtype=self.feature_mean.dtype)
        std = torch.as_tensor(std, device=self.feature_std.device,
                              dtype=self.feature_std.dtype)
        if mean.shape != self.feature_mean.shape or std.shape != self.feature_std.shape:
            raise ValueError("Training moments must each have shape [feature_dim]")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("Training moments must be finite")
        if not bool((std > 0).all()):
            raise ValueError("Training standard deviations must be strictly positive")
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def _validate(self, first_z, second_z, first_h, second_h):
        if first_z.dim() != 1 or second_z.shape != first_z.shape:
            raise ValueError("Scores must be equally sized one-dimensional logits")
        expected = (first_z.shape[0], self.feature_dim)
        if tuple(first_h.shape) != expected or tuple(second_h.shape) != expected:
            raise ValueError("Hidden readouts must each have shape [N,feature_dim]")
        tensors = (first_z, second_z, first_h, second_h)
        if any(not value.is_floating_point() for value in tensors):
            raise ValueError("Scores and hidden readouts must be floating point")
        if any(value.device != first_z.device or value.dtype != first_z.dtype for value in tensors):
            raise ValueError("All inputs must share a device and dtype")

    def forward(self, first_z, second_z, first_h, second_h, return_info=False):
        self._validate(first_z, second_z, first_h, second_h)
        if not self.has_context and not self.has_revision:
            result = second_z
            alpha = torch.zeros_like(second_z)
            delta = torch.zeros_like(second_z)
            consensus = second_z
        else:
            h1 = (first_h - self.feature_mean) / self.feature_std
            h2 = (second_h - self.feature_mean) / self.feature_std
            if self.has_context:
                # The same reliability function makes consensus invariant to
                # swapping the observations. Shared evidence is not added.
                s1 = self.reliability(torch.cat((h1, first_z[:, None]), dim=1))
                s2 = self.reliability(torch.cat((h2, second_z[:, None]), dim=1))
                alpha = torch.softmax(torch.cat((s1, s2), dim=1), dim=1)[:, 0]
                consensus = second_z + alpha * (first_z - second_z)
            else:
                alpha = torch.zeros_like(second_z)
                consensus = second_z
            if self.has_revision:
                # Common conditioning is identical in both terms, so equal
                # hidden observations give zero residual for any scores.
                common = torch.stack((first_z, second_z, alpha), dim=1)
                baseline = self.condition(torch.cat((h1, common), dim=1))
                change = self.difference(h2 - h1)
                conditional = F.silu(baseline + change) - F.silu(baseline)
                delta = self.kappa * torch.tanh(self.output(conditional).squeeze(-1))
            else:
                delta = torch.zeros_like(second_z)
            result = consensus + delta
        if return_info:
            return result, {"alpha_first": alpha, "delta": delta, "consensus": consensus}
        return result

    def macs_per_pair(self):
        """Linear MACs only; excludes elementwise work, readout and SNN."""
        d, h = self.feature_dim, self.hidden
        consensus = 2 * ((d + 1) * h + h) if self.has_context else 0
        revision = (d + 3) * h + d * h + h if self.has_revision else 0
        return int(consensus + revision)

    def estimated_macs(self, pairs):
        if isinstance(pairs, bool) or int(pairs) != pairs or pairs < 0:
            raise ValueError("pairs must be a nonnegative integer")
        return int(pairs) * self.macs_per_pair()
