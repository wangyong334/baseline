"""Slot 6b, V4-3: learned motion as a distribution over fixed motion anchors.

Anchors (per clock step): zero and n_dir directions x speed rings in a geometric series (octave spacing), so every
speed has the same relative resolution and no dataset range is built in. Per queried pixel the module returns
    logits over the anchors = context prior (current-step features at 4 pyramid levels)
                              + gain(level) x correlation(current features at the pixel,
                                                         previous-step features one anchor displacement back)
    residual per anchor     = bounded refinement (log-speed, angle) inside the anchor's cell
Matching the current step against the previous one at every candidate displacement (cost-volume style, E-RAFT /
PWC-Net) gives the module a reach equal to the largest anchor, at the pyramid level that fits each speed.
Classification over anchors plus a residual (MultiPath) avoids the pull of a single regressed velocity towards the
most frequent speed and keeps several modes. The verifier uses the top-K anchors of an event's pixel at its birth as
the event's motion hypotheses and their renormalised probabilities as mixture weights.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LEVELS = (1, 2, 4, 8)                       # pyramid strides; anchors use the finest level with speed / stride <= 2


def sample_cells(maps, frame, y, x, stride):
    """Bilinear features of a pooled map at pixel coordinates (0 outside).
    maps [F,C,h,w]; frame, y, x of one shape S (y, x in full-resolution pixels, float) -> [S..., C]."""
    Fn, C, h, w = (int(n) for n in maps.shape)
    s = float(stride)
    gy, gx = (y - 0.5 * (s - 1.0)) / s, (x - 0.5 * (s - 1.0)) / s
    y0, x0 = torch.floor(gy), torch.floor(gx)
    fy, fx = (gy - y0).unsqueeze(-1), (gx - x0).unsqueeze(-1)
    y0, x0 = y0.long(), x0.long()
    flat = maps.permute(0, 2, 3, 1).reshape(Fn * h * w, C)
    out = 0.0
    for dy, dx, wgt in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        yy, xx = y0 + dy, x0 + dx
        inside = ((yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)).unsqueeze(-1).to(maps.dtype)
        idx = (frame * h + yy.clamp(0, h - 1)) * w + xx.clamp(0, w - 1)
        out = out + flat[idx] * (wgt.to(maps.dtype) * inside)
    return out


class AnchorTable(object):
    """Zero anchor (index 0) and ring r, direction j at index 1 + r * n_dir + j; displacements in px per step."""

    def __init__(self, rings, n_dir):
        self.rings = [float(r) for r in rings]
        self.n_dir = int(n_dir)
        if len(self.rings) < 2 or self.n_dir < 4 or min(self.rings) <= 0:
            raise ValueError("need >= 2 positive rings and >= 4 directions")
        ratios = [b / a for a, b in zip(self.rings[:-1], self.rings[1:])]
        if max(ratios) - min(ratios) > 1e-6 or ratios[0] <= 1.0:
            raise ValueError("rings must form an increasing geometric series")
        self.ratio, self.r0 = ratios[0], self.rings[0]
        self.size = 1 + len(self.rings) * self.n_dir
        self.step_angle = 2.0 * math.pi / self.n_dir
        ring = [-1] + [r for r in range(len(self.rings)) for _ in range(self.n_dir)]
        direc = [0] + [j for _ in range(len(self.rings)) for j in range(self.n_dir)]
        disp = [(0.0, 0.0)] + [(self.rings[r] * math.sin(j * self.step_angle), self.rings[r] * math.cos(j * self.step_angle))
                               for r in range(len(self.rings)) for j in range(self.n_dir)]
        level = [0] + [self._level(self.rings[r]) for r in range(len(self.rings)) for _ in range(self.n_dir)]
        self.ring = torch.tensor(ring, dtype=torch.long)
        self.direction = torch.tensor(direc, dtype=torch.long)
        self.disp = torch.tensor(disp, dtype=torch.float64)
        self.level = torch.tensor(level, dtype=torch.long)

    @staticmethod
    def _level(speed):
        for i, s in enumerate(LEVELS):
            if speed / s <= 2.0:
                return i
        return len(LEVELS) - 1

    def soft_target(self, v):
        """v [n, 2] (dy, dx) px per step -> (soft target [n, A], dominant anchor [n], residual target [n, 2]).
        Linear sharing in ring coordinate (log speed, zero anchor at ring -1) and in angle between neighbours."""
        n, dev, dt = int(v.shape[0]), v.device, v.dtype
        R = len(self.rings)
        r = torch.hypot(v[:, 0], v[:, 1])
        u = torch.log(r.clamp(min=1e-12) / self.r0) / math.log(self.ratio)
        u = u.clamp(min=-1.0, max=float(R - 1))
        phi = torch.remainder(torch.atan2(v[:, 0], v[:, 1]), 2.0 * math.pi)
        jf = phi / self.step_angle
        j0 = torch.floor(jf)
        g = jf - j0
        j0 = j0.long() % self.n_dir
        j1 = (j0 + 1) % self.n_dir
        i0 = torch.floor(u).clamp(max=float(R - 2))
        f = u - i0
        i0 = i0.long()
        target = torch.zeros(n, self.size, device=dev, dtype=dt)
        rows = torch.arange(n, device=dev)
        for ring, wr in ((i0, 1.0 - f), (i0 + 1, f)):
            for jj, wd in ((j0, 1.0 - g), (j1, g)):
                w = wr * wd
                zero = ring < 0
                idx = torch.where(zero, torch.zeros_like(ring), 1 + ring.clamp(min=0) * self.n_dir + jj)
                target = target.index_put((rows, idx), w, accumulate=True)
        dom = target.argmax(1)
        dring = self.ring.to(dev)[dom]
        ddir = self.direction.to(dev)[dom]
        du = (u - dring.to(dt)).clamp(-0.5, 0.5)
        dphi = torch.remainder(phi - ddir.to(dt) * self.step_angle + math.pi, 2.0 * math.pi) - math.pi
        res = torch.stack([du, (dphi / self.step_angle).clamp(-0.5, 0.5)], 1)
        res = torch.where((dring >= 0).unsqueeze(1), res, torch.zeros_like(res))
        return target, dom, res

    def displacement(self, idx, residual):
        """idx [...] anchor indices, residual [..., 2] (log-speed in rings, angle in direction steps) -> [..., 2]."""
        ring = self.ring.to(idx.device)[idx]
        direc = self.direction.to(idx.device)[idx]
        dt = residual.dtype
        speed = self.r0 * torch.pow(torch.tensor(self.ratio, dtype=dt, device=idx.device),
                                    ring.to(dt) + residual[..., 0])
        angle = (direc.to(dt) + residual[..., 1]) * self.step_angle
        disp = torch.stack([speed * torch.sin(angle), speed * torch.cos(angle)], -1)
        return torch.where((ring >= 0).unsqueeze(-1), disp, torch.zeros_like(disp))


class AnchorMotion(nn.Module):
    """Learned motion distribution over anchors at queried pixels (parameters live in the network)."""

    def __init__(self, channels, rings=(0.5, 1, 2, 4, 8, 16, 32, 64), directions=16, hidden=32):
        super(AnchorMotion, self).__init__()
        self.table = AnchorTable(rings, directions)
        self.channels, self.hidden = int(channels), int(hidden)
        A = self.table.size
        self.prior = nn.Sequential(nn.Linear(self.channels * len(LEVELS), self.hidden), nn.ReLU(),
                                   nn.Linear(self.hidden, A))
        self.gain = nn.Parameter(torch.ones(len(LEVELS)))
        last = nn.Linear(self.hidden, 2 * A)
        self.residual = nn.Sequential(nn.Linear(self.channels, self.hidden), nn.ReLU(), last)
        with torch.no_grad():
            self.prior[2].weight.mul_(0.1)
            self.prior[2].bias.zero_()
            last.weight.mul_(0.1)
            last.bias.zero_()
        self.register_buffer("disp", self.table.disp.float(), persistent=False)
        self.register_buffer("level", self.table.level, persistent=False)
        order = torch.argsort(self.table.level * self.table.size + torch.arange(self.table.size))
        self.register_buffer("order", order, persistent=False)                      # anchors grouped by level
        self.register_buffer("inverse", torch.argsort(order), persistent=False)

    @property
    def size(self):
        return self.table.size

    @staticmethod
    def pyramid(phi):
        """phi [F,C,H,W] (H, W multiples of 8) -> list of block means at the LEVELS strides (reshape-mean: deterministic
        backward on CUDA)."""
        Fn, C, H, W = (int(n) for n in phi.shape)
        if H % LEVELS[-1] or W % LEVELS[-1]:
            raise ValueError("the canvas must be a multiple of %d" % LEVELS[-1])
        return [phi if s == 1 else phi.reshape(Fn, C, H // s, s, W // s, s).mean((3, 5)) for s in LEVELS]

    def query(self, cur, prev, frame, y, x):
        """cur, prev: pyramids of the current / previous step; frame, y, x [n] -> logits [n, A], residual [n, A, 2]."""
        n = int(y.shape[0])
        dt = cur[0].dtype
        yf, xf = y.to(dt), x.to(dt)
        ctx = [sample_cells(cur[i], frame, yf, xf, s) for i, s in enumerate(LEVELS)]           # each [n, C]
        logits = self.prior(torch.cat(ctx, 1))
        disp = self.disp.to(dt)
        level = self.level[self.order]
        parts = []
        for i, s in enumerate(LEVELS):
            sel = self.order[level == i]
            if int(sel.numel()) == 0:
                continue
            d = disp[sel]                                                                         # [A_s, 2]
            py = yf.view(n, 1) - d[:, 0].view(1, -1)
            px = xf.view(n, 1) - d[:, 1].view(1, -1)
            fr = frame.view(n, 1).expand(n, int(sel.numel()))
            past = sample_cells(prev[i], fr, py, px, s)                                           # [n, A_s, C]
            parts.append((past * ctx[i].unsqueeze(1)).mean(-1) * self.gain[i].to(dt))
        corr = torch.cat(parts, 1)[:, self.inverse]
        residual = torch.tanh(self.residual(ctx[0])).view(n, self.size, 2) * 0.5
        return logits + corr, residual

    def hypotheses(self, logits, residual, k):
        """Top-k anchors -> displacement per step [k, n, 2], log weights [k, n] (renormalised), anchor indices [n, k]."""
        values, idx = logits.topk(int(k), dim=1)
        logw = F.log_softmax(values, 1)
        rows = torch.arange(int(idx.shape[0]), device=idx.device).view(-1, 1).expand_as(idx)
        disp = self.table.displacement(idx, residual[rows, idx])
        return disp.permute(1, 0, 2).contiguous(), logw.t().contiguous(), idx

    def best(self, logits, residual):
        """Most probable anchor with its residual -> displacement per step [n, 2]."""
        idx = logits.argmax(1)
        return self.table.displacement(idx, residual[torch.arange(int(idx.shape[0]), device=idx.device), idx])

    def operations_per_query(self):
        A, C, h = self.size, self.channels, self.hidden
        L = len(LEVELS)
        return {"mac": (C * L * h + h * A) + A * C * 5 + (C * h + h * 2 * A), "ac": 2 * h + A,
                "transcendental": 2 * A}
