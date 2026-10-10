"""Slot 6, V4-3: the base heads plus the inputs of the learned verifier.

    mark, log_g  exactly the base heads on u7 (created first: same initialisation per seed; a base checkpoint loads
                 into heads.base); log_g is the target intensity of the current step
    log_mu       background events per pixel for the next step: the front end's log mu0 (input channel) plus a learned
                 correction; reads u7 detached, so the background objective never shapes the trunk
    phi          motion features (non-negative) from u7 and the event features: one 1 x 1 projection + ReLU
                 (motion_hidden > 0 adds a hidden layer); the anchor motion module (network.motion) matches them between
                 consecutive steps over a 4-level pyramid. detach_trunk=True keeps every new objective off the trunk
"""
import torch
import torch.nn as nn

from speed.slots.heads.mark_intensity import MarkIntensityHeads


class V43Heads(nn.Module):
    outputs = ("mark", "log_g", "log_mu", "phi")
    takes_input = True

    def __init__(self, in_channels, input_channels, background_channel, hidden=16, mark_prior=0.03,
                 intensity_prior=2e-4, log_g_max=8.0, motion_channels=16, motion_hidden=32, background_hidden=8,
                 detach_trunk=True):
        super(V43Heads, self).__init__()
        self.base = MarkIntensityHeads(in_channels, hidden, mark_prior, intensity_prior, log_g_max)
        self.in_channels, self.input_channels = int(in_channels), int(input_channels)
        self.background_channel = int(background_channel)
        if self.background_channel != self.input_channels - 1:
            raise ValueError("log_mu0 must be the last input channel")
        self.feature_channels = self.input_channels - 1
        self.hidden, self.motion_channels = int(hidden), int(motion_channels)
        self.motion_hidden, self.background_hidden = int(motion_hidden), int(background_hidden)
        self.detach_trunk = bool(detach_trunk)
        c_in = self.in_channels + self.feature_channels
        if self.motion_hidden > 0:
            self.encoder = nn.Sequential(nn.Conv2d(c_in, self.motion_hidden, 1), nn.ReLU(),
                                         nn.Conv2d(self.motion_hidden, self.motion_channels, 1), nn.ReLU())
        else:
            self.encoder = nn.Sequential(nn.Conv2d(c_in, self.motion_channels, 1), nn.ReLU())
        last = nn.Conv2d(self.background_hidden, 1, 1)
        self.background = nn.Sequential(nn.Conv2d(self.in_channels + self.input_channels, self.background_hidden, 1),
                                        nn.ReLU(), last)
        with torch.no_grad():
            last.weight.mul_(0.1)
            last.bias.zero_()

    def forward(self, u, x):
        """u [N,C,H,W] (u7), x [N,Cin,H,W] (network input) -> mark, log_g, log_mu [N,1,H,W], phi [N,Cm,H,W]."""
        out = self.base(u)
        x = x.to(u.dtype)
        ut = u.detach() if self.detach_trunk else u
        out["phi"] = self.encoder(torch.cat([ut, x[:, :self.feature_channels]], 1))
        c = self.background_channel
        out["log_mu"] = x[:, c:c + 1] + self.background(torch.cat([u.detach(), x], 1))
        return out

    def encoder_macs(self):
        c_in = self.in_channels + self.feature_channels
        if self.motion_hidden > 0:
            return c_in * self.motion_hidden + self.motion_hidden * self.motion_channels
        return c_in * self.motion_channels

    def operations(self, positions):
        """Dense per-pixel cost (as implemented): base heads, motion encoder, background branch."""
        p = float(positions)
        base = self.base.operations(p)
        enc = self.encoder_macs()
        bg = (self.in_channels + self.input_channels) * self.background_hidden + self.background_hidden
        return {"mac": base["mac"] + p * (enc + bg),
                "ac": base["ac"] + p * (self.motion_hidden + self.motion_channels + self.background_hidden + 1),
                "transcendental": base["transcendental"] + p}
