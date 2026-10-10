"""Slot 2 base variant: the V2 evidence front end (no learnable parameters).

Background intensity mu0 (expected events per pixel per step, from past steps only):
    mu = (S + n0 * prior) / (W + n0),  S <- b*S + N,  W <- b*W + 1   for a fast and a slow time constant
    mu0 = max(floor, slow per-pixel estimate, fast estimate averaged over a (2r+1)^2 neighbourhood)
Exact-time moments per time scale tau and polarity (event ages to the end of the step, not binned):
    A = sum exp(-age/tau),  B = sum exp(-age/tau) * age,  A(k) = d A(k-1) + a_new,  B(k) = d (B(k-1) + dt A(k-1)) + b_new
Features, all relative to the background (zero where nothing happened):
    count  log1p(N / (mu0 / 2))                       2 channels
    ratio  log1p(A / (mu0 * tau / (2 dt)))            2K channels  ("how many times more than usual")
    age    min(B / (A + eps) / tau, 3)                2K channels
    dipole ON centroid - OFF centroid in a (2R+1)^2 neighbourhood, / R   2 per dipole scale
    logmu  log mu0 (optional; read by the V4-3 background head only, not by the backbone)   1 channel
encode() also returns mu0 and the per-pixel event count of every step for the verifier (slot 7).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_GROUPS = ("count", "ratio", "age", "dipole", "logmu")


def offset_kernels(radius, dtype=torch.float32):
    k = 2 * int(radius) + 1
    offsets = torch.arange(k, dtype=dtype) - int(radius)
    ones = torch.ones(k, k, dtype=dtype)
    return torch.stack([ones, offsets.view(1, k).expand(k, k), offsets.view(k, 1).expand(k, k)]).unsqueeze(1).contiguous()


class EvidenceFrontEnd(nn.Module):
    def __init__(self, taus_ms, dt_ms, dipole_taus_ms, dipole_radius=3, bg_fast_ms=500.0, bg_slow_ms=5000.0,
                 bg_prior=0.01, bg_prior_steps=2.0, bg_floor=1e-3, bg_smooth_radius=7, features=FEATURE_GROUPS):
        super(EvidenceFrontEnd, self).__init__()
        self.features = tuple(str(f) for f in features)
        if not self.features or any(f not in FEATURE_GROUPS for f in self.features):
            raise ValueError("features must be a non-empty subset of %s" % (FEATURE_GROUPS,))
        self.taus = [float(t) for t in taus_ms]
        self.dt_ms = float(dt_ms)
        self.dipole_index = []
        for tau in dipole_taus_ms:
            if float(tau) not in self.taus:
                raise ValueError("dipole time scale %s is not in taus_ms" % tau)
            self.dipole_index.append(self.taus.index(float(tau)))
        self.dipole_radius = int(dipole_radius)
        self.beta_fast = math.exp(-self.dt_ms / float(bg_fast_ms))
        self.beta_slow = math.exp(-self.dt_ms / float(bg_slow_ms))
        self.bg_prior, self.bg_prior_steps = float(bg_prior), float(bg_prior_steps)
        self.bg_floor, self.bg_smooth_radius = float(bg_floor), int(bg_smooth_radius)
        if min(self.taus) <= 0 or self.bg_floor <= 0 or self.bg_prior <= 0 or self.bg_prior_steps <= 0:
            raise ValueError("time scales, background prior and floor must be positive")
        K = len(self.taus)
        per_channel = torch.tensor([t for t in self.taus for _ in (0, 1)], dtype=torch.float64)
        self.register_buffer("decay", torch.exp(-self.dt_ms / per_channel).view(1, 2 * K, 1, 1), persistent=False)
        self.register_buffer("tau_channel", per_channel.view(1, 2 * K, 1, 1), persistent=False)
        self.register_buffer("kernels", offset_kernels(self.dipole_radius, torch.float64), persistent=False)

    @property
    def n_features(self):
        return len(self.feature_names())

    def feature_names(self):
        names = ["count_pos", "count_neg"] if "count" in self.features else []
        for kind in ("ratio", "age"):
            if kind in self.features:
                for tau in self.taus:
                    names += ["%s%g_pos" % (kind, tau), "%s%g_neg" % (kind, tau)]
        if "dipole" in self.features:
            for j in self.dipole_index:
                names += ["dipole%g_x" % self.taus[j], "dipole%g_y" % self.taus[j]]
        if "logmu" in self.features:
            names.append("log_mu0")
        return names

    def init_state(self, batch, height, width, device, dtype=torch.float32):
        zeros = lambda c: torch.zeros(batch, c, height, width, device=device, dtype=dtype)  # noqa: E731
        K2 = 2 * len(self.taus)
        return {"A": zeros(K2), "B": zeros(K2), "S_fast": zeros(1), "S_slow": zeros(1), "W_fast": 0.0, "W_slow": 0.0,
                "k": 0}

    def background(self, state):
        n0, prior = self.bg_prior_steps, self.bg_prior
        fast = (state["S_fast"] + n0 * prior) / (state["W_fast"] + n0)
        slow = (state["S_slow"] + n0 * prior) / (state["W_slow"] + n0)
        r = self.bg_smooth_radius
        if r > 0:
            fast = F.avg_pool2d(fast, 2 * r + 1, stride=1, padding=r, count_include_pad=False)
        return torch.clamp(torch.maximum(fast, slow), min=self.bg_floor)

    def dipole(self, A_pos, A_neg):
        batch = int(A_pos.shape[0])
        sums = F.conv2d(torch.cat([A_pos, A_neg], 0), self.kernels.to(A_pos.dtype), padding=self.dipole_radius)
        pos, neg = sums[:batch], sums[batch:]
        eps = 1e-6
        both = pos[:, 0:1] * neg[:, 0:1]
        return (both / (both + 1.0)) * (pos[:, 1:3] / (pos[:, 0:1] + eps) - neg[:, 1:3] / (neg[:, 0:1] + eps)) \
            / float(self.dipole_radius)

    def step(self, state, counts, moment_a, moment_b):
        """counts [B,2,H,W], moments [B,2K,H,W] -> (state, features [B,C,H,W], mu0 [B,1,H,W], total [B,1,H,W])"""
        mu0 = self.background(state)
        dt = self.dt_ms
        decay = self.decay.to(counts.dtype)
        A_old, B_old = state["A"], state["B"]
        A = decay * A_old + moment_a
        B = decay * (B_old + dt * A_old) + moment_b
        total = counts.sum(1, keepdim=True)
        new_state = {"A": A, "B": B, "S_fast": self.beta_fast * state["S_fast"] + total,
                     "S_slow": self.beta_slow * state["S_slow"] + total,
                     "W_fast": self.beta_fast * state["W_fast"] + 1.0, "W_slow": self.beta_slow * state["W_slow"] + 1.0,
                     "k": int(state["k"]) + 1}
        tau = self.tau_channel.to(counts.dtype)
        available = {"count": torch.log1p(counts / (0.5 * mu0)), "ratio": torch.log1p(A / (mu0 * tau / (2.0 * dt))),
                     "age": torch.clamp(B / (A + 1e-3) / tau, max=3.0)}
        parts = [available[name] for name in ("count", "ratio", "age") if name in self.features]
        if "dipole" in self.features:
            for j in self.dipole_index:
                parts.append(self.dipole(A[:, 2 * j:2 * j + 1], A[:, 2 * j + 1:2 * j + 2]))
        if "logmu" in self.features:
            parts.append(torch.log(mu0))
        return new_state, torch.cat(parts, 1), mu0, total

    def scatter(self, blk):
        """Per-step polarity counts and moment sums of the new events of a block (same reduction order as V2)."""
        src = blk["source"]
        ev = blk["events"]
        steps, plane, H, W = blk["n_steps"], src.plane, src.height, src.width
        rel, pix, neg = ev["t"], ev["pixel"], ev["negative"]
        ones = torch.ones(int(rel.shape[0]), dtype=src.dtype, device=src.device)
        counts = src.accumulate((rel * 2 + neg) * plane + pix, ones, steps * 2 * plane).view(steps, 1, 2, H, W)
        K = len(self.taus)
        keys, wa, wb = [], [], []
        for j, tau in enumerate(self.taus):
            w = torch.exp(-ev["age_ms"] / tau)
            keys.append((rel * (2 * K) + 2 * j + neg) * plane + pix)
            wa.append(w)
            wb.append(w * ev["age_ms"])
        keys = torch.cat(keys)
        moment_a = src.accumulate(keys, torch.cat(wa), steps * 2 * K * plane).view(steps, 1, 2 * K, H, W)
        moment_b = src.accumulate(keys, torch.cat(wb), steps * 2 * K * plane).view(steps, 1, 2 * K, H, W)
        return counts, moment_a, moment_b

    def encode(self, state, blk):
        """-> (state, inputs [T,B,C,H,W], aux {"mu0": [T,B,1,H,W], "total": [T,B,1,H,W]})"""
        counts, moment_a, moment_b = self.scatter(blk)
        feats, mus, totals = [], [], []
        for t in range(blk["n_steps"]):
            state, f, mu0, total = self.step(state, counts[t], moment_a[t], moment_b[t])
            feats.append(f)
            mus.append(mu0)
            totals.append(total)
        return state, torch.stack(feats), {"mu0": torch.stack(mus), "total": torch.stack(totals)}

    def operations(self, height, width, events_per_step):
        """Operations per step, split into MAC, elementwise and transcendental (direct implementation)."""
        p, e, k = float(int(height) * int(width)), float(events_per_step), len(self.taus)
        r, rad = int(self.bg_smooth_radius), self.dipole_radius
        parts = [("event scatter and moment weights", 2 * k * e, e + 2 * k * e, k * e),
                 ("moment recursion", 6 * k * p, 2 * k * p, 0.0),
                 ("background (two EMAs + box smoothing)", 4 * p, 3 * p + ((2 * r + 1) ** 2 if r > 0 else 0) * p, 0.0)]
        if "count" in self.features:
            parts.append(("count features", 4 * p, 0.0, 2 * p))
        if "ratio" in self.features:
            parts.append(("ratio features", 6 * k * p, 0.0, 2 * k * p))
        if "age" in self.features:
            parts.append(("age features", 4 * k * p, 4 * k * p, 0.0))
        if "logmu" in self.features:
            parts.append(("log background channel", 0.0, 0.0, p))
        if "dipole" in self.features and self.dipole_index:
            scales = len(self.dipole_index)
            parts.append(("polarity dipole", 2 * 3 * (2 * rad + 1) ** 2 * p * scales + 11 * p * scales,
                          8 * p * scales, 0.0))
        return [{"part": name, "mac": float(mac), "elementwise": float(el), "transcendental": float(tr)}
                for name, mac, el, tr in parts]
