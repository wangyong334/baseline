"""V13 Value-of-Waiting (VoW) unit (11-10): does waiting still pay for this event?

The spiking decision unit (SDU, tools/deco/sdn_v11.py) carries the posterior log-odds U(k) of an event; U(0) starts
from the prior z0 of the BIR network. Waiting pays only if the decision can still change before the deadline D, so
the unit predicts the remaining movement of the membrane, Delta = U(D) - U(k), by three quantiles
    q_lo (alpha), q_50 (median), q_hi (1 - alpha)          q_lo <= q_50 <= q_hi by construction
and waits iff the threshold lies inside the predicted range:
    stop "target"      if U >= tau and U + q_lo >= tau
    stop "background"  if U <  tau and U + q_hi <  tau      else wait.
If the range is calibrated, stopping changes the deadline decision with probability <= alpha, so its expected loss
exceeds that of waiting by at most c * alpha (c = cost of one wrong label). The three outputs name the three factors:
distance m = U - tau, volatility q_hi - q_lo, drift (prior bias) q_50.

Two taps, one rule:
    k = 0 (birth gate, before an SDU exists): the event's past in the event memory, offsets -P .. 0, read through a
          temporal BIR neuron (bounded leaky integration with a learned memory per channel, multi-level report), then
          AvgPool over the windows (persistence) and last - average (onset), plus the margin m0 = z0 - tau.
    k >= 1 (commit): the SDU's own state (margin, arrival margin, last step, largest step, sign flips, k / D,
          attributed evidence mass, effective count; tools/deco/fit_range.py range_features).
Both heads are fitted label-free on held-out trajectories (pinball loss) and widened by split conformal (CQR).
"""
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TRACE = 10
FAR = 256.0


def logit(p):
    return math.log(p / (1.0 - p))


def trace_features(tok, tok_valid, nall, nevid, past, k_near):
    """[n, past + 1, N_TRACE] past of every event in the event memory (window offsets -past .. 0), float32.
    Per window: nearest-neighbour geometry (log mean distance, log distance of the farthest picked neighbour, share of
    neighbour slots filled), the neighbours' prior (mean, max of z0 / 10), the non-background neighbours (share of
    slots filled, mean z0 / 10, log mean distance) and the window activity (log event / non-background totals)."""
    W = past + 1
    t = tok[:, :W].astype(np.float32)
    v = tok_valid[:, :W]
    d = np.sqrt(t[..., 0] ** 2 + t[..., 1] ** 2)
    z = t[..., 5] / 10.0
    vn, ve = v[..., :k_near], v[..., k_near:]
    dn, de = np.where(vn, d[..., :k_near], np.nan), np.where(ve, d[..., k_near:], np.nan)
    zn, ze = np.where(vn, z[..., :k_near], np.nan), np.where(ve, z[..., k_near:], np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # windows without neighbours -> defaults below
        f = [np.log1p(np.nan_to_num(np.nanmean(dn, -1), nan=FAR)),
             np.log1p(np.nan_to_num(np.nanmax(dn, -1), nan=FAR)),
             vn.mean(-1),
             np.nan_to_num(np.nanmean(zn, -1), nan=-2.0),
             np.nan_to_num(np.nanmax(zn, -1), nan=-2.0),
             ve.mean(-1),
             np.nan_to_num(np.nanmean(ze, -1), nan=-2.0),
             np.log1p(np.nan_to_num(np.nanmean(de, -1), nan=FAR)),
             np.log1p(nall[:, :W]),
             np.log1p(nevid[:, :W])]
    return np.stack(f, -1).astype(np.float32)


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


class TemporalBIR(nn.Module):
    """BIR neuron over the event's past windows (same equations as model/lif2d_stream.StreamingTIF2d, 1-D):
        z = W f_w - k,   M_w = min(relu(lam M_{w-1} + (1 - lam) z), L theta),   out_w = theta round(M_w / theta)
    lam = sigmoid(.) per channel is the learned memory (1 / (1 - lam) windows), k = softplus(.) >= 0 the reference."""

    def __init__(self, n_in, channels=16, levels=7):
        super(TemporalBIR, self).__init__()
        self.inp = nn.Linear(n_in, channels)
        self.lam_raw = nn.Parameter(torch.full((channels,), 1.0))          # lam = 0.73 at the start (~3.7 windows)
        self.k_raw = nn.Parameter(torch.full((channels,), -3.0))
        self.log_theta = nn.Parameter(torch.zeros(channels))
        self.levels = int(levels)

    def forward(self, f):                                                  # f [B, W, n_in]
        lam = torch.sigmoid(self.lam_raw)
        k = F.softplus(self.k_raw)
        theta = torch.exp(self.log_theta)
        z = self.inp(f) - k
        M = torch.zeros_like(z[:, 0])
        outs = []
        for w in range(z.shape[1]):
            M = torch.minimum(F.relu(lam * M + (1.0 - lam) * z[:, w]), self.levels * theta)
            outs.append(theta * RoundSTE.apply(M / theta))
        return torch.stack(outs, 1)

    def memory_windows(self):
        return (1.0 / (1.0 - torch.sigmoid(self.lam_raw))).detach().cpu().numpy()


def quantiles(o):
    """o [..., 3] -> (q_lo, q_50, q_hi), ordered by construction."""
    q50 = o[..., 1]
    return q50 - F.softplus(o[..., 0]), q50, q50 + F.softplus(o[..., 2])


class VoWGate(nn.Module):
    """k = 0 tap: temporal BIR over the past windows -> [AvgPool_t, last - avg] -> concat margin -> L-R-L -> 3 quantiles.
    traces=False is the ablation that sees only the margin and polarity."""

    def __init__(self, n_trace=N_TRACE, channels=16, width=64, traces=True):
        super(VoWGate, self).__init__()
        self.traces = bool(traces)
        self.bir = TemporalBIR(n_trace, channels) if traces else None
        n_in = (2 * channels if traces else 0) + 2
        self.mlp = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Linear(width, 3))

    def forward(self, trace, m0, p):
        x = [m0[:, None] / 5.0, p[:, None]]
        if self.traces:
            out = self.bir(trace)
            avg = out.mean(1)
            x = [avg, out[:, -1] - avg] + x
        return quantiles(self.mlp(torch.cat(x, -1)))


class VoWCommit(nn.Module):
    """k >= 1 tap: the SDU state -> L-R-L-R-L -> 3 quantiles."""

    def __init__(self, n_in, width=64):
        super(VoWCommit, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, 3))

    def forward(self, x):
        return quantiles(self.mlp(x))


def pinball(q, y, level):
    e = y - q
    return torch.max(level * e, (level - 1.0) * e).mean()


def vow_loss(qs, y, alpha):
    q_lo, q50, q_hi = qs
    return pinball(q_lo, y, alpha) + pinball(q50, y, 0.5) + pinball(q_hi, y, 1.0 - alpha)


def fit(model, inputs, y, alpha, steps=4000, batch=8192, lr=3e-3, seed=0, device="cpu"):
    """Adam-style training of a VoW head on held-out rows; inputs = list of arrays sharing the first axis."""
    torch.manual_seed(seed)
    model = model.to(device)
    xs = [torch.as_tensor(a, device=device) for a in inputs]
    yt = torch.as_tensor(y.astype(np.float32), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        b = torch.randint(0, len(yt), (min(batch, len(yt)),), device=device)
        loss = vow_loss(model(*[x[b] for x in xs]), yt[b], alpha)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model


def predict(model, inputs, batch=65536, device="cpu"):
    out = [[], [], []]
    with torch.no_grad():
        for i in range(0, len(inputs[0]), batch):
            qs = model(*[torch.as_tensor(a[i:i + batch], device=device) for a in inputs])
            for j in range(3):
                out[j].append(qs[j].cpu().numpy())
    return tuple(np.concatenate(o) for o in out)


def cqr_shift(q_lo, q_hi, y, alpha):
    """Split-conformal widening (CQR): the (1 - alpha)(1 + 1/n) quantiles of how far y falls below q_lo / above q_hi."""
    lev = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / max(len(y), 1)))
    return float(np.quantile(q_lo - y, lev)), float(np.quantile(y - q_hi, lev))


def stop_now(U, tau, q_lo, q_hi):
    """True where the threshold lies outside the predicted range of U(D) (waiting cannot change the decision)."""
    return ((U >= tau) & (U + q_lo >= tau)) | ((U < tau) & (U + q_hi < tau))
