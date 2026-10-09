"""Slot 7, V4 step 1: evidence along measured motion; per location two hypotheses, the measured velocity and zero.

Order inside step k (every proposal depends on steps <= k-1 only, so the H0 bound of DriftEvidence holds unchanged):
    1. propose   velocities from the events of steps <= k-1 at the pixels that need one (prediction sources and the
                 pixels of the events born at k-1); no valid fit -> the velocity carried with the prediction -> zero
    2. predict   measured channel: g_{k-1} and the carried tube intensity are pushed along the proposals (largest value
                 wins where sources collide); zero channel: DriftEvidence with velocity (0, 0)
    3. verify    per-step evidence ell of the counts of step k (DriftEvidence.tube_evidence)
    4. remember  the events of step k enter the motion moments (they propose from step k+1 on)
An event born at step j follows a straight path with the velocity proposed at its pixel from the events of steps <= j.

Velocity estimator (A_sel of the V4 analysis E1): weighted least squares of event position on event time over the
events in the (2r+1)^2 box around a pixel, weights w = exp(-age / tau):
    v = cov_w(position, t) / var_w(t)
Among taus x radii the fit with weight sum >= min_weight, non-degenerate variances and the largest motion R^2 wins.
The per-pixel moments (sum w, w a, w a^2, w^2 per tau) decay exactly from step to step (no hard window).
"""
import math

import numpy as np
import torch

from speed.slots.verify.drift_cusum import DriftEvidence

MOMENTS = 4      # per pixel and tau: sum w, w a, w a^2, w^2  (a = age in ms)
FITS = 10        # box sums per tau: S0 S1 S2 Q0 xS0 yS0 xS1 yS1 xxS0 yyS0


class MotionMoments(object):
    def __init__(self, dt_ms, taus_ms, radii_px, min_weight=5.0):
        self.dt = float(dt_ms)
        self.taus = [float(t) for t in taus_ms]
        self.radii = [int(r) for r in radii_px]
        if not self.taus or not self.radii or min(self.taus) <= 0 or min(self.radii) < 1:
            raise ValueError("need positive taus and radii >= 1")
        self.min_weight = float(min_weight)

    def init_state(self, batch, height, width, device):
        return torch.zeros(batch, len(self.taus), MOMENTS, height, width, dtype=torch.float64, device=device)

    def advance(self, m, b, y, x, age_ms):
        """Moves the reference time one step forward, then adds the events of that step (ages to its end)."""
        d = self.dt
        K = len(self.taus)
        lam = torch.tensor([math.exp(-d / t) for t in self.taus], dtype=torch.float64, device=m.device).view(1, K, 1, 1)
        S0, S1, S2, Q0 = m.unbind(2)
        out = torch.stack([lam * S0, lam * (S1 + d * S0), lam * (S2 + (2.0 * d) * S1 + (d * d) * S0),
                           (lam * lam) * Q0], 2)
        n = int(y.shape[0])
        if n:
            H, W = int(out.shape[3]), int(out.shape[4])
            a = age_ms.to(torch.float64).view(1, n)
            tau = torch.tensor(self.taus, dtype=torch.float64, device=m.device).view(K, 1)
            w = torch.exp(-a / tau)
            vals = torch.stack([w, w * a, w * a * a, w * w], 1)                       # [K, MOMENTS, n]
            plane = (b.view(1, 1, n) * K + torch.arange(K, device=m.device).view(K, 1, 1)) * MOMENTS \
                + torch.arange(MOMENTS, device=m.device).view(1, MOMENTS, 1)
            keys = (plane * H + y.view(1, 1, n)) * W + x.view(1, 1, n)
            out.view(-1).index_put_((keys.reshape(-1),), vals.reshape(-1), accumulate=True)
        return out

    def integral(self, m):
        """Integral images [B, K, FITS, H+1, W+1] of the quantities summed over a box."""
        B, K, _, H, W = (int(n) for n in m.shape)
        ys = torch.arange(H, dtype=torch.float64, device=m.device).view(1, 1, H, 1)
        xs = torch.arange(W, dtype=torch.float64, device=m.device).view(1, 1, 1, W)
        S0, S1, S2, Q0 = m.unbind(2)
        maps = torch.stack([S0, S1, S2, Q0, xs * S0, ys * S0, xs * S1, ys * S1, xs * xs * S0, ys * ys * S0], 2)
        ii = torch.zeros(B, K, FITS, H + 1, W + 1, dtype=torch.float64, device=m.device)
        ii[..., 1:, 1:] = maps.cumsum(3).cumsum(4)
        return ii

    def estimate(self, m, b, y, x, with_spread=False):
        """-> velocity [n, 2] (vy, vx) in px/ms, valid [n] (some scale fits), r2 [n] of the winning scale; with
        with_spread also the R^2-weighted second moment [n, 3] (yy, yx, xx) of all valid scales' estimates around the
        chosen one, (px/ms)^2: the disagreement between scales is the estimate's uncertainty."""
        n = int(y.shape[0])
        H, W = int(m.shape[3]), int(m.shape[4])
        ii = self.integral(m)
        fits = []
        for r in self.radii:
            y0, y1 = (y - r).clamp(0, H), (y + r + 1).clamp(0, H)
            x0, x1 = (x - r).clamp(0, W), (x + r + 1).clamp(0, W)
            fits.append(ii[b, :, :, y1, x1] - ii[b, :, :, y0, x1] - ii[b, :, :, y1, x0] + ii[b, :, :, y0, x0])
        s = torch.stack(fits, 2)                                     # [n, K, R, FITS], tau-major as in E1
        S0, S1, S2 = s[..., 0], s[..., 1], s[..., 2]
        safe = torch.where(S0 > 0, S0, torch.ones_like(S0))
        ma, mx, my = S1 / safe, s[..., 4] / safe, s[..., 5] / safe
        ctt = S2 / safe - ma * ma                                    # var of t = -a equals var of a
        cxt = -(s[..., 6] / safe - mx * ma)
        cyt = -(s[..., 7] / safe - my * ma)
        spread = (s[..., 8] / safe - mx * mx) + (s[..., 9] / safe - my * my)
        ok = (S0 >= self.min_weight) & (ctt > 1e-6) & (spread > 1e-9)
        ctt_safe = torch.where(ok, ctt, torch.ones_like(ctt))
        r2 = ((cxt * cxt + cyt * cyt) / (ctt_safe * torch.where(ok, spread, torch.ones_like(spread)))).clamp(0.0, 1.0)
        score = torch.where(ok, r2, torch.full_like(r2, -1.0)).reshape(n, -1)
        best = score.argmax(1)                                       # first maximum = E1's strict ">" over scales
        pick = lambda t: t.reshape(n, -1).gather(1, best.view(n, 1)).view(n)  # noqa: E731
        valid = pick(ok.to(torch.float64)) > 0
        v = torch.stack([pick(cyt / ctt_safe), pick(cxt / ctt_safe)], 1)
        v = torch.where(valid.view(n, 1), v, torch.zeros_like(v))
        best_r2 = torch.where(valid, pick(r2), torch.zeros_like(pick(r2)))
        if not with_spread:
            return v, valid, best_r2
        okf = ok.reshape(n, -1).to(torch.float64)
        w = r2.reshape(n, -1) * okf
        w = torch.where(w.sum(1, keepdim=True) > 0, w, okf)          # all valid fits with R^2 = 0: equal weights
        w = w / w.sum(1, keepdim=True).clamp(min=1e-300)
        dy = (cyt / ctt_safe).reshape(n, -1) - v[:, 0:1]
        dx = (cxt / ctt_safe).reshape(n, -1) - v[:, 1:2]
        cov = torch.stack([(w * dy * dy).sum(1), (w * dy * dx).sum(1), (w * dx * dx).sum(1)], 1)
        return v, valid, best_r2, torch.where(valid.view(n, 1), cov, torch.zeros_like(cov))

    def operations(self, height, width, events_per_step, queries_per_step):
        p, K, R = float(int(height) * int(width)), float(len(self.taus)), float(len(self.radii))
        e, q = float(events_per_step), float(queries_per_step)
        return [("motion moments: decay", 7 * K * p, 0.0, 0.0),
                ("motion moments: new events", 3 * K * e, 4 * K * e, K * e),
                ("motion moments: box sums", 6 * K * p, 2 * FITS * K * p, 0.0),
                ("velocity fits", 20 * K * R * q, (4 * FITS + 6) * K * R * q, 0.0)]


UT_SCALE = 3.0                                    # n + lambda = 3 for n = 2 (Julier & Uhlmann 2004)
UT_WEIGHTS = (1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
UT_POINTS = ((0, 0.0), (0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0))   # (column of L, sign); first entry = centre


def cholesky2(cov):
    """Lower Cholesky factor (l11, l21, l22) of 2x2 covariances [n, 3] = (yy, yx, xx); zero rows stay zero."""
    l11 = cov[:, 0].clamp(min=0.0).sqrt()
    l21 = torch.where(l11 > 0, cov[:, 1] / torch.where(l11 > 0, l11, torch.ones_like(l11)), torch.zeros_like(l11))
    l22 = (cov[:, 2] - l21 * l21).clamp(min=0.0).sqrt()
    return torch.stack([l11, l21, l22], 1)


def sigma_points(centre, chol):
    """centre [n, 2] (vy, vx), chol [n, 3] -> [5, n, 2]: centre and centre +- sqrt(3) x the columns of L."""
    cols = (torch.stack([chol[:, 0], chol[:, 1]], 1), torch.stack([torch.zeros_like(chol[:, 2]), chol[:, 2]], 1))
    k = math.sqrt(UT_SCALE)
    return torch.stack([centre if sign == 0 else centre + (sign * k) * cols[c] for c, sign in UT_POINTS])


class MeasuredEvidence(DriftEvidence):
    """Evidence along measured motion; same evidence and readout interface as V2.

    Hypotheses per location: the measured velocity (one channel, V4-a) or, with cloud, 5 sigma points of N(v, S)
    (unscented transform, n + lambda = 3; S = the disagreement of the estimator's scales + (spacing / sqrt 3)^2 I, so
    that sigma points lie at least cloud_spacing px/step from the centre), plus the zero velocity.
    Mixture weights: V4-a uniform; cloud (1 - w0) x UT weights and w0 for zero, with w0 = 1/2 (equal) or
    1/2 exp(-d^2 / 2), d = Mahalanobis distance of zero under N(v, S) (adaptive). Weights and paths are fixed by
    the past of the event's birth, so every mixture stays a test supermartingale under H0.
    source "moments": the hand-written estimator below; "network": the velocity and sigma of the network's motion head
    at step k-1 (V4: learned motion; the estimator constants are then unused).
    fixed_velocity (vy, vx) px/step replaces the estimator by a constant centre (equivalence tests).
    oracle uses the label velocity of target events as centre at their pixels (reference only).
    """

    def __init__(self, dt_ms, taus_ms=(20.0,), radii_px=(4,), min_weight=5.0, footprint=3, track_decay=0.0,
                 gate_eps=0.0, include_zero=True, fixed_velocity=None, oracle=False, history_steps=16,
                 window_us=50000, cloud=False, cloud_spacing=1.0, zero_weight="equal", source="moments"):
        super(MeasuredEvidence, self).__init__([(0.0, 0.0)], footprint, track_decay, gate_eps)
        self.dt = float(dt_ms)
        if source not in ("moments", "network"):
            raise ValueError("source must be moments or network")
        self.source = source
        self.moments = MotionMoments(dt_ms, taus_ms, radii_px, min_weight)
        self.include_zero = bool(include_zero)
        self.fixed = None if fixed_velocity is None else (float(fixed_velocity[0]), float(fixed_velocity[1]))
        self.oracle = bool(oracle)
        if self.oracle and self.fixed is not None:
            raise ValueError("oracle and fixed_velocity exclude each other")
        self.cloud = bool(cloud)
        self.cloud_spacing = float(cloud_spacing)
        if self.cloud_spacing < 0:
            raise ValueError("cloud_spacing must be >= 0")
        if zero_weight not in ("equal", "adaptive"):
            raise ValueError("zero_weight must be equal or adaptive")
        if zero_weight == "adaptive" and not self.cloud:
            raise ValueError("the adaptive zero weight needs the cloud (it uses its covariance)")
        self.zero_weight = zero_weight
        self.history_steps = int(history_steps)
        self.window_us = int(window_us)
        self._oracle_v = None
        self._hist = {}           # proposals of the current stream by step (shared with the state)

    @property
    def n_moving(self):
        return len(UT_WEIGHTS) if self.cloud else 1

    @property
    def n_hypotheses(self):
        return self.n_moving + (1 if self.include_zero else 0)

    @property
    def weighted(self):
        return self.cloud

    def begin_stream(self, stream):
        if self.oracle:
            from speed.eval.breakdown import target_kinematics
            vx, vy, _ = target_kinematics(stream, self.window_us)
            self._oracle_v = np.stack([vy, vx], 1) * self.dt                         # px/step, nan = unknown

    def init_state(self, batch, height, width, device, dtype=torch.float32):
        V = self.n_hypotheses
        state = {"G": torch.zeros(batch, V, height, width, device=device, dtype=dtype),
                 "ell": torch.zeros(batch, V, height, width, device=device, dtype=dtype), "k": 0,
                 "carried": torch.zeros(batch, self.n_moving, 2, height, width, device=device, dtype=dtype),
                 "hist": {}, "births": None, "queried": 0}
        if self.fixed is None and self.source == "moments":
            state["moments"] = self.moments.init_state(batch, height, width, device)
        self._hist = state["hist"]
        return state

    def _floor_cov(self, n, device, dtype):
        f = (self.cloud_spacing ** 2) / UT_SCALE
        return torch.tensor([f, 0.0, f], dtype=dtype, device=device).view(1, 3).expand(n, 3)

    def _propose(self, state, fresh, motion_prev=None):
        """Proposals (px/step) from the events of steps <= k-1 where needed ->
        (one dense field [B,2,H,W] per moving channel, history entry, number of queried pixels)."""
        G = state["G"]
        B, _, H, W = (int(n) for n in G.shape)
        plane = H * W
        M = self.n_moving
        if self.fixed is not None:
            centre = torch.tensor(self.fixed, dtype=G.dtype, device=G.device).view(1, 2)
            if self.cloud:
                pts = sigma_points(centre, cholesky2(self._floor_cov(1, G.device, G.dtype)))
            else:
                pts = centre.view(1, 1, 2)
            fields = [pts[j].view(1, 2, 1, 1).expand(B, 2, H, W).contiguous() for j in range(M)]
            return fields, None, 0
        fields = [torch.zeros(B, 2, H, W, device=G.device, dtype=G.dtype) for _ in range(M)]
        need = G[:, 0] > 0
        for j in range(1, M):
            need = need | (G[:, j] > 0)
        if fresh is not None:
            need = need | (fresh[:, 0] > 0)
        flat = need.reshape(-1).nonzero().view(-1)
        births = state["births"]
        if births is not None:
            flat = torch.unique(torch.cat([flat, births["flat"]]))
        n = int(flat.numel())
        entry = torch.zeros(B, 5 if self.cloud else 2, H, W, device=G.device, dtype=G.dtype)
        if n == 0:
            return fields, entry, 0
        b, rem = torch.div(flat, plane, rounding_mode="floor"), flat % plane
        y, x = torch.div(rem, W, rounding_mode="floor"), rem % W
        if self.source == "network":
            if motion_prev is None:
                raise ValueError("source=network needs the motion output of the previous step")
            mp = motion_prev.reshape(B, 3, plane)[b, :, rem].to(torch.float64)       # (vy, vx) px/ms, log sigma
            v, valid = mp[:, :2], torch.ones(n, dtype=torch.bool, device=G.device)
            sig2 = torch.exp(2.0 * mp[:, 2])
            est = (v, valid, None, torch.stack([sig2, torch.zeros_like(sig2), sig2], 1))
        else:
            est = self.moments.estimate(state["moments"], b, y, x, with_spread=self.cloud)
            v, valid = est[0], est[1]
        carried = [state["carried"][:, j].reshape(B, 2, plane)[b, :, rem] for j in range(M)]
        centre = torch.where(valid.view(n, 1), (v * self.dt).to(G.dtype), carried[0])
        if births is not None and births.get("oracle") is not None:
            pos = torch.searchsorted(flat, births["oracle_flat"])
            centre[pos] = births["oracle"].to(G.dtype)
            valid = valid.clone()
            # full-shape value: the deterministic CUDA index_put of torch 1.9 cannot broadcast a scalar
            valid[pos] = torch.ones(int(pos.shape[0]), dtype=torch.bool, device=G.device)
        if not self.cloud:
            fields[0].view(B, 2, plane)[b, :, rem] = centre
            entry.view(B, 2, plane)[b, :, rem] = centre
            return fields, entry, n
        cov = torch.where(valid.view(n, 1), est[3] * (self.dt * self.dt), torch.zeros_like(est[3])).to(G.dtype)
        chol = cholesky2(cov + self._floor_cov(n, G.device, G.dtype))
        pts = sigma_points(centre, chol)
        for j in range(M):
            fields[j].view(B, 2, plane)[b, :, rem] = torch.where(valid.view(n, 1), pts[j], carried[j])
        entry.view(B, 5, plane)[b, :, rem] = torch.cat([centre, chol], 1)
        return fields, entry, n

    def _push(self, values, field, k):
        """values [B,1,H,W] >= 0 moved by the phase-rounded displacement of step k along field; on collisions the
        largest value wins (ties: larger source index). -> (moved values, field of the winning sources)"""
        B, _, H, W = (int(n) for n in values.shape)
        plane = H * W
        out = torch.zeros_like(values)
        vel = torch.zeros(B, 2, H, W, device=values.device, dtype=field.dtype)
        flat_values = values.reshape(-1)
        src = (flat_values > 0).nonzero().view(-1)
        if src.numel() == 0:
            return out, vel
        b, rem = torch.div(src, plane, rounding_mode="floor"), src % plane
        y, x = torch.div(rem, W, rounding_mode="floor"), rem % W
        v = field.view(B, 2, plane)[b, :, rem]
        vk = v.to(torch.float64)
        d = (torch.floor(vk * k + 0.5) - torch.floor(vk * (k - 1) + 0.5)).long()
        ny, nx = y + d[:, 0], x + d[:, 1]
        inside = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        dest = (b * H + ny) * W + nx
        dest, val, v = dest[inside], flat_values[src][inside], v[inside]
        if dest.numel() == 0:
            return out, vel
        order = torch.sort(val, stable=True)[1]
        dest, val, v = dest[order], val[order], v[order]
        order = torch.sort(dest, stable=True)[1]
        dest, val, v = dest[order], val[order], v[order]
        last = torch.ones_like(dest, dtype=torch.bool)
        last[:-1] = dest[1:] != dest[:-1]
        dest, val, v = dest[last], val[last], v[last]
        out.view(-1)[dest] = val
        vel.view(B, 2, plane)[torch.div(dest, plane, rounding_mode="floor"), :, dest % plane] = v
        return out, vel

    def step(self, state, counts, mu0, log_g_prev, events=None, motion_prev=None):
        k = int(state["k"])
        g_prev = None if log_g_prev is None else torch.exp(log_g_prev)
        fresh = None if g_prev is None else self._gate(g_prev)
        fields, entry, queried = self._propose(state, fresh, motion_prev) if k > 0 else (None, None, 0)
        hist = state["hist"]
        self._hist = hist
        if entry is not None:
            hist[k - 1] = entry
            for old in [j for j in hist if j < k - self.history_steps]:
                del hist[old]
        G_old = state["G"]
        rho = self.track_decay
        channels, carried = [], []
        for j in range(self.n_moving):
            carried_j = rho * G_old[:, j:j + 1] if rho > 0 else None
            source = carried_j if fresh is None else (fresh if carried_j is None else torch.maximum(fresh, carried_j))
            if source is None:
                source = torch.zeros_like(G_old[:, j:j + 1])
            if fields is None:
                G_j, vel_j = source.clone(), torch.zeros_like(state["carried"][:, j])
            else:
                G_j, vel_j = self._push(source, fields[j], k)
            channels.append(self._gate(G_j))
            carried.append(vel_j)
        if self.include_zero:
            z = self.n_moving
            carried_0 = rho * G_old[:, z:z + 1] if rho > 0 else None
            G_0 = carried_0 if fresh is None else (fresh if carried_0 is None else torch.maximum(fresh, carried_0))
            channels.append(self._gate(G_0 if G_0 is not None else torch.zeros_like(G_old[:, z:z + 1])))
        G = torch.cat(channels, 1)
        new = {"G": G, "ell": self.tube_evidence(counts, mu0, G), "k": k + 1, "carried": torch.stack(carried, 1),
               "hist": hist, "births": None, "queried": queried}
        if self.fixed is None:
            if events is None:
                none = torch.zeros(0, dtype=torch.long, device=G.device)
                events = {"b": none, "y": none, "x": none, "age_ms": torch.zeros(0, device=G.device), "idx": []}
            if self.source == "moments":
                new["moments"] = self.moments.advance(state["moments"], events["b"], events["y"], events["x"],
                                                      events["age_ms"])
            new["births"] = self._births(events, G.shape)
        return new

    def _births(self, events, shape):
        H, W = int(shape[2]), int(shape[3])
        flat = (events["b"] * H + events["y"]) * W + events["x"]
        out = {"flat": torch.unique(flat), "oracle": None}
        if self._oracle_v is not None and int(flat.numel()):
            v = torch.from_numpy(self._oracle_v[np.asarray(events["idx"])]).to(flat.device)
            known = torch.isfinite(v).all(1)
            if bool(known.any()):
                # mean label velocity per pixel (deterministic for pixels with several target events)
                pix, inv = torch.unique(flat[known], return_inverse=True)
                total = torch.zeros(int(pix.numel()), 2, dtype=v.dtype, device=v.device)
                total.index_put_((inv,), v[known], accumulate=True)
                count = torch.zeros(int(pix.numel()), dtype=v.dtype, device=v.device)
                count.index_put_((inv,), torch.ones_like(inv, dtype=v.dtype), accumulate=True)
                out.update(oracle=total / count.view(-1, 1), oracle_flat=pix)
        return out

    def _birth_cloud(self, b, y, x, k_from):
        """Centre [n, 2] and Cholesky factor [n, 3] (zeros without cloud) proposed at each event's birth pixel."""
        n = int(y.shape[0])
        if self.fixed is not None:
            centre = torch.tensor(self.fixed, dtype=torch.float64, device=y.device).view(1, 2).expand(n, 2)
            chol = cholesky2(self._floor_cov(n, y.device, torch.float64)) if self.cloud \
                else torch.zeros(n, 3, dtype=torch.float64, device=y.device)
            return centre, chol
        width = 5 if self.cloud else 2
        values = torch.zeros(n, width, dtype=torch.float64, device=y.device)
        for j in torch.unique(k_from).tolist():
            if j not in self._hist:
                raise KeyError("no velocity proposal kept for step %d (history_steps=%d)" % (j, self.history_steps))
            m = k_from == j
            values[m] = self._hist[j][b[m], :, y[m], x[m]].to(torch.float64)
        if not self.cloud:
            return values, torch.zeros(n, 3, dtype=torch.float64, device=y.device)
        return values[:, :2], values[:, 2:]

    def gather_along_many(self, tensor, k_to, b, y, x, k_from):
        """Values of tensor [B,V,H,W] at step k_to along each event's paths (born at k_from); 0 outside."""
        V, H, W = self.n_hypotheses, int(tensor.shape[2]), int(tensor.shape[3])
        n = int(y.shape[0])
        centre, chol = self._birth_cloud(b, y, x, k_from)
        v = sigma_points(centre, chol) if self.cloud else centre.view(1, n, 2)
        kf = k_from.to(torch.float64).view(1, n, 1)
        delta = (torch.floor(v * float(k_to) + 0.5) - torch.floor(v * kf + 0.5)).long()       # [M, n, 2]
        yy, xx = y.view(1, n) + delta[..., 0], x.view(1, n) + delta[..., 1]
        if self.include_zero:
            yy, xx = torch.cat([yy, y.view(1, n)]), torch.cat([xx, x.view(1, n)])
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        vv = torch.arange(V, device=y.device).view(V, 1).expand(V, n)
        values = tensor[b.view(1, n).expand(V, n), vv, yy.clamp(0, H - 1), xx.clamp(0, W - 1)]
        return torch.where(inside, values, torch.zeros_like(values))

    def log_weights(self, b, y, x, k_from):
        """Log mixture weights [V, n] of each event's hypotheses (None = uniform, the V2 / V4-a readout)."""
        if not self.weighted:
            return None
        n = int(y.shape[0])
        if self.fixed is None and any(j not in self._hist for j in torch.unique(k_from).tolist()):
            return torch.full((self.n_hypotheses, n), -math.log(self.n_hypotheses), dtype=torch.float64,
                              device=y.device)                       # born at the last step: no evidence yet
        centre, chol = self._birth_cloud(b, y, x, k_from)
        if not self.include_zero:
            w0 = torch.zeros(n, dtype=torch.float64, device=y.device)
        elif self.zero_weight == "equal":
            w0 = torch.full((n,), 0.5, dtype=torch.float64, device=y.device)
        else:
            l11, l21, l22 = chol[:, 0], chol[:, 1], chol[:, 2]
            ok = (l11 > 0) & (l22 > 0)
            z1 = centre[:, 0] / torch.where(ok, l11, torch.ones_like(l11))
            z2 = (centre[:, 1] - l21 * z1) / torch.where(ok, l22, torch.ones_like(l22))
            d2 = torch.where(ok, z1 * z1 + z2 * z2, torch.zeros_like(z1))
            w0 = 0.5 * torch.exp(-0.5 * d2)
        rows = [(1.0 - w0) * w for w in UT_WEIGHTS]
        if self.include_zero:
            rows.append(w0)
        return torch.log(torch.stack(rows).clamp(min=1e-300))

    def operations(self, height, width, active_fraction=1.0, events_per_step=0.0, queries_per_step=0.0):
        parts = super(MeasuredEvidence, self).operations(height, width, active_fraction)
        sources = float(int(height) * int(width)) * float(active_fraction)
        sort = (2.0 * math.log2(max(sources, 2.0)) + 6.0) * self.n_moving
        parts.append({"part": "moving-channel push", "mac": 0.0, "elementwise": sort * sources, "transcendental": 0.0})
        if self.fixed is None and self.source == "moments":
            for name, mac, el, tr in self.moments.operations(height, width, events_per_step, queries_per_step):
                parts.append({"part": name, "mac": float(mac), "elementwise": float(el), "transcendental": float(tr)})
            if self.cloud:
                q = float(queries_per_step)
                fits = float(len(self.moments.taus) * len(self.moments.radii))
                parts.append({"part": "cloud covariance and sigma points", "mac": (6 * fits + 20) * q,
                              "elementwise": 10 * q, "transcendental": 3 * q})
        return parts
