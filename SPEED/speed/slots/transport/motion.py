"""Slot 5 V4 variant: carried membrane states move with the learned motion (memory follows the target).

After each step every layer's state is split by the target probability p = sigmoid(mark): the part p * u is pushed
(bilinear splatting) along the head's velocity for one clock step, the rest stays. Only pixels the network decides are
more likely target than not (p >= gate, 0.5 = the decision boundary of the mark posterior) move: background memory is
never pushed along an untrained velocity, and the work scales with the moving fraction. Layers at stride s use the
p-weighted mean velocity of each s x s cell divided by s, so coarse layers move by fewer cells (coarse-to-fine).
p and the velocity are detached: the transport uses the heads, the gradients through the states stay.
"""
import torch
import torch.nn.functional as F

from speed.core.splat import bilinear_splat


class MotionTransport(object):
    is_identity = False

    def __init__(self, dt_ms, gate=0.5):
        self.dt = float(dt_ms)
        self.gate = float(gate)
        self.active_sum, self.calls = 0.0, 0                               # moving fraction, for the energy count

    def apply(self, states, outputs):
        motion, mark = outputs["motion"].detach(), outputs["mark"].detach()
        p = torch.sigmoid(mark.float())
        p = p * (p >= self.gate).to(p.dtype)
        self.active_sum = self.active_sum + (p > 0).float().mean()     # stays on the device (no sync per step)
        self.calls += 1
        v = motion[:, :2].float() * self.dt                                 # px per step at full resolution
        H, W = int(p.shape[2]), int(p.shape[3])
        out = []
        for st in states:
            if st is None:
                out.append(None)
                continue
            s = H // int(st.shape[2])
            if s * int(st.shape[2]) != H or s * int(st.shape[3]) != W:
                raise ValueError("state resolution must divide the canvas")
            if s == 1:
                ps, vs = p, v
            else:
                ps = F.avg_pool2d(p, s)
                vs = F.avg_pool2d(p * v, s) / ps.clamp(min=1e-6) / s
            moving = ps.to(st.dtype) * st
            out.append(st - moving + bilinear_splat(moving, vs[:, 0:1].to(st.dtype), vs[:, 1:2].to(st.dtype)))
        return out

    def active_fraction(self):
        return float(self.active_sum) / self.calls if self.calls else 1.0

    def operations(self, backbone, height, width):
        """Per step: split, 4-way splat and merge of the moving state elements (measured moving fraction)."""
        sizes = {"enc1": 1, "enc2": 2, "enc3": 4, "enc4": 8, "dec3": 4, "dec2": 2, "dec1": 1}
        elements = sum(block.out_ch * (int(height) // s) * (int(width) // s)
                       for block, s in zip(backbone.blocks(), sizes.values()))
        a = self.active_fraction()
        return {"mac": 6.0 * elements * a, "ac": 8.0 * elements * a + elements,
                "transcendental": float(int(height) * int(width))}
