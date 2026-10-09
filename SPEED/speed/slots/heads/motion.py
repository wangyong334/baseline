"""Slot 6 V4 variant: mark and intensity heads plus a motion head (learned motion, used by transport, loss, verifier).

    mark, log_g  exactly the base heads on u7 (created first: same initialisation per seed as the base variant)
    motion       per pixel (vy, vx) in px/ms and log sigma (isotropic standard deviation of that velocity, px/ms),
                 from u7 and the network input (the input carries the events' timing, which the velocity needs)
The motion head starts at zero velocity and sigma = sigma_init (one pixel per clock step).
"""
import math

import torch
import torch.nn as nn

from speed.slots.heads.mark_intensity import MarkIntensityHeads

LOG_SIGMA_MIN, LOG_SIGMA_MAX = math.log(1e-4), math.log(1e2)          # px/ms, numerical guard only


class MarkIntensityMotionHeads(nn.Module):
    outputs = ("mark", "log_g", "motion")
    takes_input = True

    def __init__(self, in_channels, input_channels, dt_ms, hidden=16, motion_hidden=16, mark_prior=0.03,
                 intensity_prior=2e-4, log_g_max=8.0, sigma_init_px_per_step=1.0):
        super(MarkIntensityMotionHeads, self).__init__()
        self.base = MarkIntensityHeads(in_channels, hidden, mark_prior, intensity_prior, log_g_max)
        last = nn.Conv2d(int(motion_hidden), 3, 1, bias=True)
        self.motion = nn.Sequential(nn.Conv2d(int(in_channels) + int(input_channels), int(motion_hidden), 1, bias=True),
                                    nn.ReLU(), last)
        with torch.no_grad():
            last.weight.mul_(0.1)
            last.bias.copy_(torch.tensor([0.0, 0.0, math.log(float(sigma_init_px_per_step) / float(dt_ms))]))
        self.in_channels, self.input_channels = int(in_channels), int(input_channels)
        self.hidden, self.motion_hidden = int(hidden), int(motion_hidden)

    def forward(self, u, x):
        """u [N,C,H,W] (u7), x [N,Cin,H,W] (network input) -> mark, log_g [N,1,H,W], motion [N,3,H,W]."""
        out = self.base(u)
        m = self.motion(torch.cat([u, x.to(u.dtype)], 1))
        out["motion"] = torch.cat([m[:, :2], m[:, 2:3].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)], 1)
        return out

    def operations(self, positions):
        p = float(positions)
        base = self.base.operations(p)
        c_in = self.in_channels + self.input_channels
        return {"mac": base["mac"] + p * (c_in * self.motion_hidden + self.motion_hidden * 3),
                "ac": base["ac"] + p * self.motion_hidden, "transcendental": base["transcendental"]}
