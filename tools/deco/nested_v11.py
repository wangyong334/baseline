"""Nested V11 network (11-08): a clock-driven small network inside an event-driven big network, trained end to end.

Small network (inner): the BIR streaming spiking U-Net (V5). Every tick (window) it updates its membrane state and,
for every event of the tick, emits a prior z0 (calibrated log-odds) AND an event report r_j (a few bounded numbers
taken from the readout's hidden layer): what the dense, history-aware network tells the decision neurons about event j.
Big network (outer): the V11 spiking decision neurons (tools/deco/sdn_v11.py). A neuron is born for every event whose
prior falls in the pending band; it reads the raw events around it (geometry, polarity), the small network's priors
and reports of those events, re-audits its evidence ledger every tick and fires once (thresholds fitted afterwards on
held-out trajectories, tools/deco/fit_range.py).
Joint objective (tools/deco/train_nested.py): small-network BCE on every event (+ its message budget) + decision-neuron
stepwise BCE on the pending events. The neurons' loss reaches the small network through the priors and the reports,
so the small network learns what to report.
Start: the report head has ordinary random weights (reports of order one), the neurons' report columns are zero, so the
untrained nested network computes exactly the staged V11 network while both sides still receive gradients (if both
started at zero, neither would ever move).
Candidates: evaluation uses tools/deco/evidence.candidate_picks (identical to the staged evidence files); training uses
the GPU version below (same rule, ties between equal distances may be broken differently).
"""
import math

import numpy as np
import torch
import torch.nn as nn

INF = float("inf")


class ReportHead(nn.Module):
    """Event reports r_j = tanh(W h_j) from the hidden layer h_j of the small network's readout MLP (captured with a
    forward hook, same event order as the logits)."""

    def __init__(self, small, dim=4):
        super(ReportHead, self).__init__()
        hidden = small.readout[0].out_features
        self.dim = int(dim)
        self.proj = nn.Linear(hidden, self.dim)     # default init: reports of order one from the start (the neurons'
        with torch.no_grad():                       # report columns are zero, so the start is still the staged network)
            self.proj.bias.zero_()
        self._h = None
        small.readout[1].register_forward_hook(self._grab)

    def _grab(self, module, inputs, output):
        self._h = output

    def forward(self):
        if self._h is None:
            raise RuntimeError("the small network has not run since the last report")
        out = torch.tanh(self.proj(self._h))
        self._h = None
        return out


def gpu_picks(x, y, cell, z0_det, pend, cell_of_query, k_near, k_evid, flo):
    """Training-time candidates: for every (pending event, window offset) query, the k_near nearest events of the same
    cell (one window of one recording) and the k_evid nearest non-background events (z0 > flo) not already picked,
    the pending event itself excluded.
    x, y [N] float; cell [N] long (cell id of every event); z0_det [N]; pend [n] long (event indices);
    cell_of_query [n, n_off] long (-1 = no cell). Returns pick [n, n_off, K] long (-1 = none) and ok."""
    n, n_off = cell_of_query.shape
    K = k_near + k_evid
    pick = torch.full((n, n_off, K), -1, dtype=torch.long, device=x.device)
    ok = torch.zeros((n, n_off, K), dtype=torch.bool, device=x.device)
    order = torch.argsort(cell)
    cs = cell[order]
    cells = torch.unique(cell_of_query[cell_of_query >= 0])
    starts = torch.searchsorted(cs, cells)
    ends = torch.searchsorted(cs, cells, right=True)
    qi, qo = torch.nonzero(cell_of_query >= 0, as_tuple=True)
    qcell = cell_of_query[qi, qo]
    for c, a, b in zip(cells.tolist(), starts.tolist(), ends.tolist()):
        sel = qcell == c
        if b <= a or not bool(sel.any()):
            continue
        ev = order[a:b]                                              # events of this cell
        qs_i, qs_o = qi[sel], qo[sel]
        me = pend[qs_i]
        d2 = (x[me][:, None] - x[ev][None, :]) ** 2 + (y[me][:, None] - y[ev][None, :]) ** 2
        d2 = d2.masked_fill(ev[None, :] == me[:, None], INF)
        kn = min(k_near, ev.numel())
        dn, jn = torch.topk(d2, kn, dim=1, largest=False)
        near_ok = torch.isfinite(dn)
        near = ev[jn]
        pick[qs_i, qs_o, :kn] = torch.where(near_ok, near, torch.full_like(near, -1))
        ok[qs_i, qs_o, :kn] = near_ok
        bg = ~(z0_det[ev] > flo)
        d2e = d2.masked_fill(bg[None, :], INF)
        taken = torch.zeros(d2e.shape, dtype=torch.uint8, device=d2e.device)
        taken.scatter_(1, jn, near_ok.to(torch.uint8))
        d2e = d2e.masked_fill(taken.bool(), INF)
        ke = min(k_evid, ev.numel())
        de, je = torch.topk(d2e, ke, dim=1, largest=False)
        evid_ok = torch.isfinite(de)
        evid = ev[je]
        pick[qs_i, qs_o, k_near:k_near + ke] = torch.where(evid_ok, evid, torch.full_like(evid, -1))
        ok[qs_i, qs_o, k_near:k_near + ke] = evid_ok
    return pick, ok


def build_tokens(x, y, t_ms, p, z0, reports, pend, pick, ok, offsets):
    """Token features of the staged evidence files (dx, dy, dt, window offset, polarity, neighbour z0 clipped to
    +-20, rounded through float16 like the files) plus the neighbours' reports. Differentiable in z0 and reports.
    Returns raw [n, T, 6], rep [n, T, M], ok [n, T], dw [n, T], dist2 [n, T] (T = n_off * K)."""
    n, n_off, K = pick.shape
    safe = torch.where(ok, pick, torch.zeros_like(pick))
    me = pend[:, None, None]
    dw = torch.as_tensor(offsets, dtype=torch.float32, device=x.device)[None, :, None].expand(n, n_off, K)
    raw = torch.stack([x[safe] - x[me], y[safe] - y[me], t_ms[safe] - t_ms[me], dw, p[safe],
                       torch.clamp(z0[safe], -20.0, 20.0)], -1)
    raw = raw.half().float()                                         # identical to the float16 evidence files
    okf = ok.to(raw.dtype)[..., None]
    raw = raw * okf
    rep = reports[safe] * okf
    T = n_off * K
    raw = raw.reshape(n, T, 6)
    return raw, rep.reshape(n, T, -1), ok.reshape(n, T), raw[..., 3], raw[..., 0] ** 2 + raw[..., 1] ** 2


class NestedDeCo(nn.Module):
    """Small network + report head + V11 decision neurons, with the staged featurizer statistics as buffers."""

    def __init__(self, small, big, report, feat_mu, feat_sd, z_mu, z_sd, window_ms):
        super(NestedDeCo, self).__init__()
        self.small, self.big, self.report = small, big, report
        self.register_buffer("feat_mu", torch.as_tensor(feat_mu, dtype=torch.float32))
        self.register_buffer("feat_sd", torch.as_tensor(feat_sd, dtype=torch.float32))
        self.z_mu, self.z_sd, self.win_s = float(z_mu), float(z_sd), float(window_ms) / 1000.0

    def decide(self, raw, rep, ok, dw, dist2, z0_pend, p_pend, valid, D):
        """Decision-neuron trajectories U [n, D + 1] from built tokens (FeaturizerV10 scaling of the raw part)."""
        feat = torch.stack([raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3] * self.win_s, raw[..., 4], raw[..., 5]], -1)
        feat = (feat - self.feat_mu) / self.feat_sd
        if self.big.n_extra:
            feat = torch.cat([feat, rep], -1)
        z_in = (z0_pend - self.z_mu) / self.z_sd
        return self.big(feat, ok, dw, valid, z_in, z0_pend, p_pend, D, dist2)
