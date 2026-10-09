"""Slot 7 base variant: evidence along fixed velocity hypotheses (V2 drift-CUSUM evidence, no learnable parameters).

Background model per pixel and step: H0  N ~ Poisson(mu0);  H1(v)  N ~ Poisson(mu0 + G_v).
Predicted target intensity of tube v (from the past only):
    G_v(x, k) = gate( max( gate(g(x - d_v(k), k-1)), rho * G_v(x - d_v(k), k-1) ) ),   rho = exp(-dt / tau_track)
    g = exp(log_g) of the previous step, d_v(k) = integer displacement of hypothesis v in step k (phase-accumulated),
    gate(z) = z * 1[z >= eps] (sparse execution; eps = 0 disables it).
Per-pixel log-likelihood ratio e_v = N log(1 + G_v/mu0) - G_v, aggregated over the footprint by log-mean-exp
(valid under any spatial correlation: E0[mean exp e] <= 1). The per-step tube evidence ell_v is what readouts use.
TubeReadout: an event of step k reads F(d) = logsumexp_v sum_{m=k+1..k+d} ell_v(path_v(m)) - log V after d steps;
anchored=True only credits positive evidence while the event's chain stays supported (V3 anchored evidence chain).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_ZERO = -30.0
EXP_CLIP = 80.0
SHIFT_CACHE_SIZE = 8
OFFSET_TABLE_MIN = 256


def velocity_grid(axis_values):
    values = [float(v) for v in axis_values]
    if len(set(values)) != len(values):
        raise ValueError("duplicate velocity values: %s" % values)
    return [(vy, vx) for vy in values for vx in values]


def pixel_evidence(counts, mu0, log_g):
    log_r = F.softplus(log_g - torch.log(mu0))
    return counts * log_r - torch.exp(log_g)


class DriftEvidence(nn.Module):
    def __init__(self, velocities, footprint=3, track_decay=0.0, gate_eps=0.0):
        super(DriftEvidence, self).__init__()
        if int(footprint) < 1 or int(footprint) % 2 == 0:
            raise ValueError("footprint must be a positive odd number")
        if not velocities:
            raise ValueError("need at least one velocity hypothesis")
        if not 0.0 <= float(track_decay) < 1.0:
            raise ValueError("track_decay must be in [0, 1)")
        if float(gate_eps) < 0:
            raise ValueError("gate_eps must be >= 0")
        self.velocities = [(float(vy), float(vx)) for vy, vx in velocities]
        self.footprint = int(footprint)
        self.track_decay = float(track_decay)
        self.gate_eps = float(gate_eps)
        self._shift_cache = {}
        self._offset_cache = {}

    @property
    def n_hypotheses(self):
        return len(self.velocities)

    def offsets(self, k):
        return [(int(math.floor(vy * k + 0.5)), int(math.floor(vx * k + 0.5))) for vy, vx in self.velocities]

    def step_shifts(self, k):
        if k <= 0:
            return [(0, 0)] * self.n_hypotheses
        before, after = self.offsets(k - 1), self.offsets(k)
        return [(a[0] - b[0], a[1] - b[1]) for a, b in zip(after, before)]

    def init_state(self, batch, height, width, device, dtype=torch.float32):
        zeros = torch.zeros(batch, self.n_hypotheses, height, width, device=device, dtype=dtype)
        return {"G": zeros, "ell": zeros, "k": 0}

    def _shift_index(self, shifts, height, width, device):
        h, w = int(height), int(width)
        key = (tuple(shifts), h, w, str(device))
        hit = self._shift_cache.get(key)
        if hit is not None:
            return hit
        inside = [(sy, sx) for sy, sx in shifts if abs(sy) < h and abs(sx) < w]
        pad = max([abs(s) for pair in inside for s in pair] + [1])
        wp = w + 2 * pad
        ys = torch.arange(h, device=device, dtype=torch.long).view(h, 1)
        xs = torch.arange(w, device=device, dtype=torch.long).view(1, w)
        rows = []
        for sy, sx in shifts:
            if abs(sy) >= h or abs(sx) >= w:
                rows.append(torch.zeros(h * w, device=device, dtype=torch.long))
            else:
                rows.append(((ys - sy + pad) * wp + (xs - sx + pad)).reshape(-1))
        entry = (pad, torch.stack(rows))
        if len(self._shift_cache) >= SHIFT_CACHE_SIZE:
            self._shift_cache.pop(next(iter(self._shift_cache)))
        self._shift_cache[key] = entry
        return entry

    def _shift_each(self, tensor, shifts, fill=0.0):
        """Channel v of tensor [B,V,H,W] (or one map [B,1,H,W] for every v) moved by shifts[v]; out[y,x] = in[y-sy,x-sx]."""
        batch, channels, h, w = (int(n) for n in tensor.shape)
        V = len(shifts)
        if channels not in (1, V):
            raise ValueError("channels must be 1 or the number of hypotheses")
        if all(pair == (0, 0) for pair in shifts):
            return tensor.expand(batch, V, h, w).clone(memory_format=torch.contiguous_format)
        pad, index = self._shift_index(shifts, h, w, tensor.device)
        padded = F.pad(tensor, (pad, pad, pad, pad), value=float(fill))
        if channels == 1:
            out = padded.reshape(batch, -1)[:, index]
        else:
            out = torch.gather(padded.reshape(batch, V, -1), 2, index.unsqueeze(0).expand(batch, V, h * w))
        return out.view(batch, V, h, w)

    def _gate(self, G):
        if self.gate_eps <= 0:
            return G
        return G * (G >= self.gate_eps).to(G.dtype)

    def predicted_intensity(self, G_old, g_prev, shifts):
        carried = self.track_decay * self._shift_each(G_old, shifts) if self.track_decay > 0 else None
        fresh = None if g_prev is None else self._gate(self._shift_each(g_prev, shifts))
        if fresh is None:
            G = carried if carried is not None else torch.zeros_like(G_old)
        else:
            G = fresh if carried is None else torch.maximum(fresh, carried)
        return self._gate(G)

    def aggregate_evidence(self, e):
        f = self.footprint
        if f == 1:
            return e
        r = f // 2
        padded = F.pad(torch.exp(torch.clamp(e, max=EXP_CLIP)), (r, r, r, r), value=1.0)
        return torch.clamp(torch.log(F.avg_pool2d(padded, f, stride=1)), min=-1e4)

    def tube_evidence(self, counts, mu0, G):
        log_g = torch.log(torch.clamp(G, min=math.exp(LOG_ZERO)))
        return self.aggregate_evidence(pixel_evidence(counts, mu0, log_g))

    def step(self, state, counts, mu0, log_g_prev, events=None):
        """counts, mu0 [B,1,H,W]; log_g_prev [B,1,H,W] of the previous step or None -> new state (G, ell, k).
        events (the step's b, y, x, age_ms, idx) is unused here; MeasuredEvidence needs it."""
        k = int(state["k"])
        shifts = self.step_shifts(k)
        g_prev = None if log_g_prev is None else torch.exp(log_g_prev)
        G = self.predicted_intensity(state["G"], g_prev, shifts)
        return {"G": G, "ell": self.tube_evidence(counts, mu0, G), "k": k + 1}

    def support_field(self, counts):
        present = (counts > 0).to(counts.dtype)
        f = self.footprint
        return present if f == 1 else F.max_pool2d(present, f, stride=1, padding=f // 2)

    def offset_table(self, k_max, device):
        key = str(device)
        table = self._offset_cache.get(key)
        if table is None or int(table.shape[0]) <= int(k_max):
            size = max(int(k_max) + 1, OFFSET_TABLE_MIN, 0 if table is None else 2 * int(table.shape[0]))
            table = torch.tensor([self.offsets(k) for k in range(size)], dtype=torch.long, device=device)
            self._offset_cache[key] = table
        return table

    def gather_along_many(self, tensor, k_to, b, y, x, k_from):
        """Values of tensor [B,V,H,W] at step k_to along each event's tubes (event born at k_from); 0 outside."""
        V, H, W = self.n_hypotheses, int(tensor.shape[2]), int(tensor.shape[3])
        n = int(y.shape[0])
        table = self.offset_table(int(k_to), y.device)
        delta = table[int(k_to)].unsqueeze(0) - table[k_from]
        yy = y.view(1, n) + delta[:, :, 0].t()
        xx = x.view(1, n) + delta[:, :, 1].t()
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        vv = torch.arange(V, device=y.device).view(V, 1).expand(V, n)
        values = tensor[b.view(1, n).expand(V, n), vv, yy.clamp(0, H - 1), xx.clamp(0, W - 1)]
        return torch.where(inside, values, torch.zeros_like(values))

    def operations(self, height, width, active_fraction=1.0, **stats):
        """Operations per step over V hypotheses x canvas (dense unless an active fraction is given)."""
        p, v, f, a = float(int(height) * int(width)), float(self.n_hypotheses), self.footprint, float(active_fraction)
        vp = v * p * a
        parts = [("tube intensity prediction", vp if self.track_decay > 0 else 0.0,
                  (3 if self.track_decay > 0 else 1) * vp, p)]
        if self.gate_eps > 0 or a < 1.0:
            parts.append(("prediction gate", 0.0, 2 * p + 2 * vp, 0.0))
        if a < 1.0:
            parts.append(("sparse bookkeeping", 0.0, (float(f * f) + 6.0) * vp + p, 0.0))
        parts.append(("per-pixel log-likelihood ratio", vp, 2 * vp, 2 * vp + p + vp + vp))
        if f > 1:
            parts.append(("footprint log-mean-exp", 0.0, (float(f * f) + 2) * vp, 2 * vp))
        return [{"part": name, "mac": float(mac), "elementwise": float(el), "transcendental": float(tr)}
                for name, mac, el, tr in parts]


def anchored_accumulate(run, alive, values, support):
    """V3 anchored chain: full evidence while the chain holds, only negative evidence after it broke."""
    run = run + torch.where(alive, values, torch.clamp(values, max=0.0))
    return run, alive & support


class TubeReadout(object):
    """Per-event delayed readout F(d) along the tubes; step() returns [(key, d, F [N], publish_step)] when due."""

    def __init__(self, verifier, delays, anchor=False):
        self.verifier = verifier
        self.delays = sorted(set(int(d) for d in delays))
        if any(d < 1 for d in self.delays):
            raise ValueError("delays must be >= 1")
        self.max_delay = max(self.delays) if self.delays else 0
        self.anchor = bool(anchor)
        self.pending = []
        self.last_k = -1

    def _score(self, run):
        return torch.logsumexp(run, dim=0) - math.log(self.verifier.n_hypotheses)

    def _pending(self, name):
        parts = [entry[name] for entry in self.pending]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def step(self, state, k, b, y, x, key, support=None):
        out = []
        k = int(k)
        if self.anchor and support is None:
            raise ValueError("anchored readout needs the support field of every step")
        if self.pending:
            where = (self._pending("b"), self._pending("y"), self._pending("x"))
            values = self.verifier.gather_along_many(state["ell"], k, *where, self._pending("k_from"))
            if self.anchor:
                V = self.verifier.n_hypotheses
                held = self.verifier.gather_along_many(support.expand(-1, V, -1, -1), k, *where,
                                                       self._pending("k_from")) > 0
            start = 0
            for entry in self.pending:
                n = int(entry["y"].shape[0])
                if self.anchor:
                    entry["run"], entry["alive"] = anchored_accumulate(entry["run"], entry["alive"],
                                                                       values[:, start:start + n],
                                                                       held[:, start:start + n])
                else:
                    entry["run"] = entry["run"] + values[:, start:start + n]
                start += n
                if (k - entry["k"]) in self.delays:
                    out.append((entry["key"], k - entry["k"], self._score(entry["run"]), k))
        self.pending = [e for e in self.pending if k - e["k"] < self.max_delay]
        if self.max_delay > 0:
            entry = {"k": k, "b": b, "y": y, "x": x, "key": key, "k_from": torch.full_like(y, k),
                     "run": state["ell"].new_zeros(self.verifier.n_hypotheses, int(y.shape[0]))}
            if self.anchor:
                entry["alive"] = torch.ones(self.verifier.n_hypotheses, int(y.shape[0]), dtype=torch.bool,
                                            device=y.device)
            self.pending.append(entry)
        self.last_k = k
        return out

    def flush(self):
        out = []
        for entry in self.pending:
            for d in self.delays:
                if d > self.last_k - entry["k"]:
                    out.append((entry["key"], d, self._score(entry["run"]), self.last_k))
        self.pending = []
        return out
