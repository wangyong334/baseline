"""Slot 6 base variant: two per-pixel heads on the last pre-reset membrane potential u7 (1x1 convs, one hidden layer).

    mark   logit that an event at this pixel in this step belongs to a target (zero-wait per-event decision)
    log_g  log target intensity of this step (expected target events per pixel per step), softly capped at
           log_g_max via m - softplus(m - raw); the verifier shifts it along velocity hypotheses as next-step forecast
Initialisation: last layer scaled by 0.1, biases at the priors (logit of mark_prior, log of intensity_prior).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarkIntensityHeads(nn.Module):
    outputs = ("mark", "log_g")

    def __init__(self, in_channels, hidden=16, mark_prior=0.03, intensity_prior=2e-4, log_g_max=8.0):
        super(MarkIntensityHeads, self).__init__()
        if not (0.0 < float(mark_prior) < 1.0 and float(intensity_prior) > 0.0):
            raise ValueError("mark_prior must be in (0, 1) and intensity_prior > 0")
        # The last layer is created first, as in the legacy network (same initialisation per seed).
        last = nn.Conv2d(int(hidden), 2, 1, bias=True)
        self.net = nn.Sequential(nn.Conv2d(int(in_channels), int(hidden), 1, bias=True), nn.ReLU(), last)
        with torch.no_grad():
            last.weight.mul_(0.1)
            last.bias.copy_(torch.tensor([math.log(mark_prior / (1.0 - mark_prior)), math.log(intensity_prior)]))
        self.log_g_max = float(log_g_max)
        self.in_channels, self.hidden = int(in_channels), int(hidden)

    def forward(self, u):
        """u [N,C,H,W] -> {"mark": [N,1,H,W], "log_g": [N,1,H,W]}"""
        out = self.net(u)
        return {"mark": out[:, 0:1], "log_g": self.log_g_max - F.softplus(self.log_g_max - out[:, 1:2])}

    def operations(self, positions):
        """MACs / ACs per step when evaluated at `positions` pixels (dense = whole canvas)."""
        p = float(positions)
        return {"mac": p * (self.in_channels * self.hidden + self.hidden * 2), "ac": p * self.hidden,
                "transcendental": 2.0 * p}
