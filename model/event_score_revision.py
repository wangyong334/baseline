"""Lightweight V14 conditional revision of two logits for the same event.

This is a learned discriminative update, not multiplication of independent
likelihoods. Overlapping observations contain common evidence. The second
logit enters through its difference from the first, while gain/calibration
are conditioned only on the first logit. Backbone readout remains unchanged.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


REVISION_MODES = ("first", "second", "mean", "conditional", "calibration_only")


class EventScoreRevision(nn.Module):
    """34 parameters at hidden=8: r_f = r1 + a(r1)*(r2-r1) + c(r1).

    a=softplus(raw_a)>0. Initialization gives a=0.5,c=0 (mean logits).
    Parameter construction preserves the caller's CPU RNG state, so adding
    this head cannot change a subsequently initialized backbone's seed.
    """

    def __init__(self, mode="conditional", hidden=8):
        super().__init__()
        if mode not in REVISION_MODES:
            raise ValueError("unknown revision mode: %s" % mode)
        if int(hidden) <= 0:
            raise ValueError("hidden must be positive")
        self.mode = mode
        with torch.random.fork_rng(devices=[]):
            self.net = nn.Sequential(nn.Linear(1, int(hidden)), nn.ReLU(), nn.Linear(int(hidden), 2))
            nn.init.zeros_(self.net[-1].weight)
            with torch.no_grad():
                self.net[-1].bias.copy_(torch.tensor([math.log(math.expm1(0.5)), 0.0]))

    def forward(self, r1, r2, return_details=False, mode=None):
        if r1.shape != r2.shape:
            raise ValueError("r1 and r2 must have identical event-aligned shapes")
        mode = self.mode if mode is None else mode
        if mode not in REVISION_MODES:
            raise ValueError("unknown revision mode: %s" % mode)
        if mode in ("conditional", "calibration_only"):
            raw = self.net(r1.unsqueeze(-1))
            a, c = F.softplus(raw[..., 0]), raw[..., 1]
            if mode == "calibration_only":
                a = torch.zeros_like(a)
        else:
            gain = {"first": 0.0, "second": 1.0, "mean": 0.5}[mode]
            a, c = torch.full_like(r1, gain), torch.zeros_like(r1)
        delta = r2 - r1
        if mode == "first":
            fused = r1
        elif mode == "second":
            fused = r2
        elif mode == "mean":
            fused = 0.5 * r1 + 0.5 * r2
        elif mode == "calibration_only":
            fused = r1 + c
        else:
            fused = r1 + a * delta + c
        if return_details:
            return fused, {"a": a, "c": c, "delta": delta, "correction": a * delta + c}
        return fused


class ConcatScoreFusion(nn.Module):
    """Separately trained ordinary concatenation baseline: 2->8->1, 33 params."""

    def __init__(self, hidden=8):
        super().__init__()
        if int(hidden) <= 0:
            raise ValueError("hidden must be positive")
        with torch.random.fork_rng(devices=[]):
            self.net = nn.Sequential(nn.Linear(2, int(hidden)), nn.ReLU(), nn.Linear(int(hidden), 1))

    def forward(self, r1, r2):
        if r1.shape != r2.shape:
            raise ValueError("r1 and r2 must have identical event-aligned shapes")
        return self.net(torch.stack((r1, r2), dim=-1)).squeeze(-1)
