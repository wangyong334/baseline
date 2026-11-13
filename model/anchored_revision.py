"""One bounded context update anchored to the latest V5 readout.

The overlapping observations are correlated. This module learns a conditional
correction, not a product of independent likelihoods. Identical observations
give exactly zero update. Python 3.8 / PyTorch 1.9 compatible.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.context_evidence import ContextEvidence


class AnchoredRevision(ContextEvidence):
    """Reuse the paired-event interface without the r2 consensus parameters."""

    def __init__(self, feature_dim=32, hidden=8, kappa=2.0, variant="full"):
        nn.Module.__init__(self)
        for name, value in (("feature_dim", feature_dim), ("hidden", hidden)):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(name + " must be a positive integer")
        if not math.isfinite(float(kappa)) or float(kappa) <= 0:
            raise ValueError("kappa must be finite and positive")
        if variant not in self.VARIANTS:
            raise ValueError("Unknown anchored variant: " + str(variant))
        self.feature_dim, self.hidden = int(feature_dim), int(hidden)
        self.kappa, self.variant = float(kappa), variant
        self.has_context = variant == "full"
        self.has_revision = variant in ("full", "no_context")
        if self.has_revision:
            self.condition = nn.Linear(self.feature_dim + 1, self.hidden)
            self.difference = nn.Linear(self.feature_dim + 1, self.hidden, bias=False)
            self.output = nn.Linear(self.hidden, 1, bias=False)
            nn.init.zeros_(self.output.weight)

    @staticmethod
    def bounded_hidden(value):
        # No train/validation moment fitting or near-zero channel division.
        # RMS magnitude is bounded by one; the logit retains readout strength.
        return value / torch.sqrt(1. + value.square().mean(dim=-1, keepdim=True))

    def set_normalization(self, mean, std):
        raise ValueError("AnchoredRevision uses per-event bounded scaling; fitted channel statistics are not used")

    def forward(self, first_z, second_z, first_h, second_h, return_info=False):
        self._validate(first_z, second_z, first_h, second_h)
        delta = torch.zeros_like(second_z)
        if self.has_revision:
            h2 = self.bounded_hidden(second_h)
            s2 = F.softsign(second_z).unsqueeze(-1)
            anchor = torch.cat((h2, s2), dim=-1)
            if self.has_context:
                first = torch.cat((self.bounded_hidden(first_h),
                                   F.softsign(first_z).unsqueeze(-1)), dim=-1)
                difference = first - anchor
            else:
                # Capacity-matched single-observation control, independently
                # trained. It cannot access the first observation.
                difference = anchor
            common = self.condition(anchor)
            update = F.silu(common + self.difference(difference)) - F.silu(common)
            delta = self.kappa * torch.tanh(self.output(update).squeeze(-1))
        result = second_z + delta
        if return_info:
            return result, {"delta": delta, "consensus": second_z,
                            "alpha_first": torch.zeros_like(second_z)}
        return result

    def macs_per_pair(self):
        return (2 * (self.feature_dim + 1) * self.hidden + self.hidden
                if self.has_revision else 0)

