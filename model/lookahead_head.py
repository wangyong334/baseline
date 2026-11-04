"""B2 control for module three: a learned look-ahead head, the black-box twin of RCL (FINAL_PLAN_20261104.md §4.4).

Same inputs, future budget, commit rules and training protocol as RCL v4.1 with score-only experts
(configs/changepoint_v41_insample.yaml), but none of RCL's model-based parts: no presence state, no label-fitted
dynamics, no occupancy supervision, no presence gate, no mixture of experts.
Per window t (cells of `cell` x `cell` pixels, as in RCL):
    (B1) V_t(c) = sum_{e in c, t} softplus(z0_e + beta)              the marked-Poisson cell map RCL's evidence reads
    (B2) E_t    = enc(V_t)                                            three 3x3 convs, dilation 1 / 2 / 4 (+-7 cells,
                                                                      the reach of RCL's displacement rings 1/2/4/8)
    (B3) F_t    = ConvGRU_f(E_t, F_{t-1})                             causal context (RCL: the forward filter)
    (B4) B_u|t  = ConvGRU_b([E_u, F_u], B_{u+1|t}), u = t .. t - D    learned backward smoother over the last D + 1
                                                                      windows (RCL: the RTS smoother), redone every window
Per event e (window s, cell c), waiting at lag d = t - s (d = 1..D):
    (B5) z_e(d) = lp0_e + w_o . relu(W_s [p, t_local, lp0_e] + W_h B_s|t(c) + w_d d / D) + b_o
         w_o, b_o start at zero, so the head starts exactly at V5 (residual fallback, review 11.28); one model gives
         every lag (multi-lag supervision).
Training: BCE of z_e(d) for every lag (= RCL's L_evt); nothing else. Provisional output = V5's z0, as in RCL.
occupancy_aux = True (the "B2 + occupancy" variant): the same box-occupancy labels RCL's L_pres uses supervise a
1x1-conv presence readout of the backward states at every lag (auxiliary multi-task loss into exist_sum), so the
only remaining difference from RCL is the model-based recursion itself (label-fitted HMM + RTS + gated experts).
Commit: identical to RCL (fixed at D, or adaptive with the tau*-centred bounds; provisional negatives never commit
early). future_stats = past (eval only) reads B_s|s, the state available at arrival.
Interface = model.changepoint_layer.ChangepointLayer (init_state / detach_state / forward_chunk / flush).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.changepoint_layer import (EPS, STATUS_CONFIRM, STATUS_DEADLINE, STATUS_FLUSH, STATUS_REJECT,
                                     ChangepointLayer, _scalar_logit)

LOOKAHEAD_KEYS = ("defer_max", "defer_alpha", "commit", "cell", "lookahead_hidden", "member_hidden", "commit_lag",
                  "future_stats", "occupancy_aux")


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell (Shi et al. 2015 / Ballas et al. 2016), 3x3 kernels, zero padding."""

    def __init__(self, in_ch, hid_ch):
        super(ConvGRUCell, self).__init__()
        self.hid = int(hid_ch)
        self.zr = nn.Conv2d(int(in_ch) + self.hid, 2 * self.hid, 3, padding=1)
        self.n = nn.Conv2d(int(in_ch) + self.hid, self.hid, 3, padding=1)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.zr(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.n(torch.cat([x, r * h], 1)))
        return (1.0 - z) * n + z * h


class LookaheadHead(nn.Module):
    """forward_chunk(events, z0, feat, state, steps, keep, labels, tid, log, keys, trace) -> (z0, state, aux)."""

    def __init__(self, feat_dim, height, width, pos_weight, threshold=0.9, defer_max=10, defer_alpha=0.05,
                 commit="fixed", cell=4, lookahead_hidden=3, member_hidden=16, commit_lag=None,
                 future_stats="smoothed", occupancy_aux=False):
        super(LookaheadHead, self).__init__()
        if commit not in ("fixed", "adaptive"):
            raise ValueError("commit 必须是 fixed 或 adaptive")
        if future_stats not in ("smoothed", "past"):
            raise ValueError("future_stats 必须是 smoothed 或 past")
        if not 0.0 < float(defer_alpha) < 0.5 or int(defer_max) < 1:
            raise ValueError("需要 0 < defer_alpha < 0.5、defer_max >= 1")
        self.cell = int(cell)
        if self.cell < 1 or int(height) % self.cell or int(width) % self.cell:
            raise ValueError("画布 %dx%d 必须能被判决格 %d 整除" % (height, width, self.cell))
        self.hc, self.wc = int(height) // self.cell, int(width) // self.cell
        self.defer_max, self.alpha, self.commit = int(defer_max), float(defer_alpha), str(commit)
        self.commit_lag = self.defer_max if commit_lag is None else int(commit_lag)
        if not 1 <= self.commit_lag <= self.defer_max:
            raise ValueError("需要 1 <= commit_lag <= defer_max")
        self.future_stats = str(future_stats)
        self.log_w = math.log(float(pos_weight))
        self.tau_logit = _scalar_logit(float(threshold)) - self.log_w
        h, m = int(lookahead_hidden), int(member_hidden)
        self.hidden = h
        self.obs_bias = nn.Parameter(torch.zeros(()))
        self.enc = nn.Sequential(nn.Conv2d(1, h, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(h, h, 3, padding=2, dilation=2), nn.ReLU(),
                                 nn.Conv2d(h, h, 3, padding=4, dilation=4), nn.ReLU())
        self.fwd = ConvGRUCell(h, h)
        self.bwd = ConvGRUCell(2 * h, h)
        self.read_s = nn.Linear(3, m)
        self.read_h = nn.Linear(h, m, bias=False)
        self.read_d = nn.Linear(1, m, bias=False)
        self.read_o = nn.Linear(m, 1)
        with torch.no_grad():                        # residual fallback: the head starts at V5
            self.read_o.weight.zero_()
            self.read_o.bias.zero_()
        # interface attributes read by the trainers (same as ChangepointLayer)
        self.context, self.decision_lag, self.defer_revision = "lookahead", 0, "lookahead"
        self.has_delayed, self.collect_calibration = True, False
        self.defer_A = float("inf")
        self.status_names = ("confirmed", "rejected", "unused", "flushed", "deadline")
        # occupancy labels only for the "+ occupancy" variant (the trainer then passes events["occ"])
        self.occupancy_aux = bool(occupancy_aux)
        self.occ_head = nn.Conv2d(h, 1, 1) if self.occupancy_aux else None

    @property
    def occupancy_spec(self):
        return (self.cell, self.hc, self.wc) if self.occupancy_aux else None

    # ------------------------------------------------------------------ pieces
    def commit_bounds(self):
        half = _scalar_logit(1.0 - self.alpha)
        return self.tau_logit - half, self.tau_logit + half

    def init_occupancy_prior(self, rate):
        """Prior-probability bias init of the occupancy readout (Lin et al. 2017): bias = logit(training occupancy
        rate), from the training labels. Without it Adam needs thousands of steps to move the bias from ~0 to the
        ~-7 that a ~0.1% occupancy rate requires. Returns what was set (written to occupancy_prior.json)."""
        rate = min(max(float(rate), 1e-6), 0.5)
        with torch.no_grad():
            self.occ_head.bias.fill_(_scalar_logit(rate))
        return {"occupancy_rate": rate, "occupancy_bias": _scalar_logit(rate)}

    def evidence_summary(self):
        with torch.no_grad():
            return {"lookahead_beta": float(self.obs_bias), "readout_norm": float(self.read_o.weight.norm()),
                    "readout_bias": float(self.read_o.bias)}

    def cell_map(self, z0, b, cy, cx, batch):
        """(B1) [B, 1, hc, wc]: per-cell sum of softplus(z0 + beta); 0 where the window has no event."""
        val = F.softplus(z0.to(self.obs_bias.dtype) + self.obs_bias)
        flat = (b.long() * self.hc + cy.long()) * self.wc + cx.long()
        out = torch.zeros(batch * self.hc * self.wc, device=val.device, dtype=val.dtype)
        if int(flat.shape[0]):
            out = out.index_put((flat,), val, accumulate=True)     # deterministic on CUDA in torch 1.9
        return out.view(batch, 1, self.hc, self.wc)

    def readout(self, hs, hb, lag, lp0):
        """(B5) calibrated logit; hs = W_s [p, t_local, lp0] + b_s computed at arrival."""
        x = hs + self.read_h(hb) + self.read_d((lag.to(lp0.dtype) / float(self.defer_max))[:, None])
        return lp0 + self.read_o(F.relu(x)).squeeze(-1)

    # ------------------------------------------------------------------ state
    def init_state(self, batch, device):
        dtype = self.obs_bias.dtype
        return {"F": torch.zeros(batch, self.hidden, self.hc, self.wc, device=device, dtype=dtype),
                "E": [], "Fbuf": [], "queue": None, "occ": [],
                "age": torch.zeros(batch, dtype=torch.long, device=device)}

    @staticmethod
    def detach_state(state):
        if state is None:
            return None
        q = state["queue"]
        return {"F": state["F"].detach(), "E": [x.detach() for x in state["E"]],
                "Fbuf": [x.detach() for x in state["Fbuf"]],
                "queue": None if q is None else {k: v.detach() for k, v in q.items()},
                "occ": list(state.get("occ", [])), "age": state.get("age")}

    _take = staticmethod(ChangepointLayer._take)
    _cat = staticmethod(ChangepointLayer._cat)
    _pack = staticmethod(ChangepointLayer._pack)

    def _emit(self, out, z_rep, key, lag, status):
        if int(key.shape[0]):
            out.append((z_rep, key, lag, status))

    def _flush_entries(self, q, out):
        if q is None or int(q["b"].shape[0]) == 0:
            return
        self._emit(out, q["zlast"] + self.log_w, q["key"], q["lag"], torch.full_like(q["lag"], STATUS_FLUSH))

    def flush(self, state):
        out = []
        self._flush_entries(state["queue"] if state is not None else None, out)
        return self._pack(out)

    # ------------------------------------------------------------------ chunk
    def forward_chunk(self, events, z0, feat, state, steps, keep=None, labels=None, tid=None, log=None, keys=None,
                      trace=None):
        train = labels is not None
        if state is None:
            batch = int(keep.shape[1]) if keep is not None else (int(events["b"].max()) + 1 if events["b"].numel() else 1)
            state = self.init_state(batch, z0.device)
        batch = int(state["F"].shape[0])
        D = self.defer_max
        zero = z0.new_zeros(())
        aux = {"exist_sum": zero, "exist_n": 0, "pos_sum": zero, "pos_n": 0, "pd_sum": zero, "pd_n": 0,
               "org_sum": zero, "org_n": 0, "mem_sum": zero, "mem_n": 0, "evt_sum": zero, "evt_n": 0,
               "dyn_sum": zero, "dyn_n": 0, "births": 0, "recycled": 0, "live": 0, "slot_windows": 0}
        out = []
        lp0_all = z0 - self.log_w
        lo_commit, hi_commit = self.commit_bounds()
        adaptive = self.commit == "adaptive" and not train
        Fh, bufE, bufF, q = state["F"], list(state["E"]), list(state["Fbuf"]), state["queue"]
        occ, age = list(state.get("occ", [])), state.get("age")
        if age is None:
            age = torch.zeros(batch, dtype=torch.long, device=Fh.device)
        occ_all = events.get("occ") if (train and self.occupancy_aux) else None
        if train and self.occupancy_aux and occ_all is None:
            raise ValueError("occupancy_aux 训练需要 events['occ']（目标轨迹占据）")
        for t in range(int(steps)):
            idx = torch.nonzero(events["t"] == t, as_tuple=False).flatten()
            if keep is not None and bool((keep[t] == 0).any()):
                reset = keep[t] == 0
                if q is not None:
                    gone = reset[q["b"]]
                    self._flush_entries(self._take(q, gone), out)
                    q = self._take(q, ~gone)
                Fh = torch.where(reset[:, None, None, None], torch.zeros_like(Fh), Fh)
                age = torch.where(reset, torch.zeros_like(age), age)
            eb = events["b"][idx].long()
            cy = torch.div(events["y"][idx].long(), self.cell, rounding_mode="floor").clamp(max=self.hc - 1)
            cx = torch.div(events["x"][idx].long(), self.cell, rounding_mode="floor").clamp(max=self.wc - 1)
            # (B1)-(B4): encoder, forward context, backward smoother over the buffer (index -1 = this window)
            E = self.enc(self.cell_map(z0[idx], eb, cy, cx, batch))
            Fh = self.fwd(E, Fh)
            bufE, bufF = (bufE + [E])[-(D + 1):], (bufF + [Fh])[-(D + 1):]
            n = len(bufE)
            smooth, hb = [None] * n, torch.zeros_like(Fh)
            for j in range(n - 1, -1, -1):
                hb = self.bwd(torch.cat([bufE[j], bufF[j]], 1), hb)
                smooth[j] = hb
            age = (age + 1).clamp(max=D + 2)
            if occ_all is not None:                   # auxiliary presence supervision at every lag (as RCL's L_pres)
                occ = (occ + [occ_all[t]])[-(D + 1):]
                tot, cnt = Fh.new_zeros(()), 0
                for lag in range(0, n):
                    valid = (lag < age).to(Fh.dtype)[:, None, None]
                    logit_occ = self.occ_head(smooth[n - 1 - lag]).squeeze(1)
                    tot = tot + (F.binary_cross_entropy_with_logits(logit_occ, occ[n - 1 - lag], reduction="none")
                                 * valid).sum()
                    cnt += int(valid.sum()) * self.hc * self.wc
                aux["exist_sum"] = aux["exist_sum"] + tot
                aux["exist_n"] += cnt
            # waiting events: one more window of future
            if q is not None and int(q["b"].shape[0]):
                q = dict(q, lag=q["lag"] + 1)
                stack = torch.stack(smooth)                                     # [n, B, H, hc, wc]
                j = (n - 1 - q["lag"]).clamp(min=0)
                h_ev = q["h_arr"] if self.future_stats == "past" else stack[j, q["b"], :, q["cy"], q["cx"]]
                z_cal = self.readout(q["hs"], h_ev, q["lag"], q["lp0"])
                if train:
                    aux["evt_sum"] = aux["evt_sum"] + F.binary_cross_entropy_with_logits(z_cal, q["y"], reduction="sum")
                    aux["evt_n"] += int(z_cal.shape[0])
                z_cal = z_cal.detach()
                if trace is not None:
                    trace.append((q["key"], q["lag"], z_cal))
                deadline = q["lag"] >= self.commit_lag
                if adaptive:
                    up = z_cal >= hi_commit
                    down = (q["prov"] > 0.5) & (z_cal <= lo_commit)
                    status = torch.where(up, torch.full_like(q["lag"], STATUS_CONFIRM),
                                         torch.where(down, torch.full_like(q["lag"], STATUS_REJECT),
                                                     torch.full_like(q["lag"], STATUS_DEADLINE)))
                    rel = up | down | deadline
                else:
                    status = torch.full_like(q["lag"], STATUS_DEADLINE)
                    rel = deadline
                if bool(rel.any()):
                    self._emit(out, z_cal[rel] + self.log_w, q["key"][rel], q["lag"][rel], status[rel])
                q = dict(q, zlast=z_cal)
                q = self._take(q, ~rel)
            # this window's events: provisional output = z0, then wait (or commit at arrival in adaptive mode)
            if int(idx.shape[0]):
                lp0 = lp0_all[idx]
                x_s = torch.stack([events["p"][idx].to(lp0.dtype), events["t_local"][idx].to(lp0.dtype), lp0], 1)
                key = keys[idx] if keys is not None else idx
                entry = {"b": eb, "cy": cy, "cx": cx, "lag": torch.zeros_like(eb), "key": key, "lp0": lp0,
                         "hs": self.read_s(x_s), "h_arr": smooth[-1][eb, :, cy, cx],
                         "y": labels[idx].to(z0.dtype) if train else torch.zeros_like(lp0),
                         "prov": (lp0 >= self.tau_logit).to(z0.dtype), "zlast": lp0.detach()}
                if trace is not None:
                    trace.append((entry["key"], entry["lag"], lp0.detach()))
                if adaptive:
                    now = lp0 >= hi_commit
                    if bool(now.any()):
                        self._emit(out, z0[idx][now], entry["key"][now], entry["lag"][now],
                                   torch.full_like(entry["lag"][now], STATUS_CONFIRM))
                    entry = self._take(entry, ~now)
                q = self._cat(q, entry)
            aux["slot_windows"] += batch
        aux.update(self._pack(out))
        return z0, {"F": Fh, "E": bufE, "Fbuf": bufF, "queue": q, "occ": occ, "age": age}, aux


def count_parameters(module):
    return int(sum(p.numel() for p in module.parameters()))


__all__ = ["LookaheadHead", "LOOKAHEAD_KEYS", "ConvGRUCell", "count_parameters", "EPS"]
