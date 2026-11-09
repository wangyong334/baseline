"""DeCo-SNN V12 (11-09): one spiking network trained end to end; every event is decided by a spiking decision neuron.

Small network (inner, clock-driven, 50 ms windows): the BIR streaming spiking U-Net (model/evspsegnet_stream.py). Every
window it updates its states and gives every event of the window its prior z0 (log-odds of "target", read out at the
event's pixel). The prior is the decision neuron's membrane at birth, not a decision of its own.
Big network (outer, event-driven): one V11 spiking decision neuron per event (tools/deco/sdn_v11.py), no birth rule. A
neuron is born with its event and lives D windows. It reads the raw events around it (K nearest + K non-background per
window: the past P windows at birth, one more window per step) with the small network's priors of those events,
re-audits its evidence every window (membrane U = z0 + attributed evidence) and fires once (commit thresholds fitted
afterwards on held-out trajectories, tools/deco/fit_range.py); a confident event fires at step 0.
Interaction (big -> small): after every window the live neurons' belief updates are rasterized into a two-channel
innovation map, log1p of sum relu(sigma(U) - sigma(z0)) and of sum relu(sigma(z0) - sigma(U)) per pixel: what the
decision layer has learned beyond the small network's own priors (the priors are no news to the small network; events
whose evidence confirms their prior add nothing). The map enters the small network's last layer (dec1) at the next
window as an extra input current through a zero-initialised 3x3 convolution, so the untrained loop is the open network.
The map carries no gradient (no backpropagation around the loop through time); the convolution learns through the
next windows' priors.
Token features use fixed physical scales: dx, dy / 32 px; dt / (S windows) and window offset / S with S = max(P, D);
polarity - 1/2; neighbour prior / 5 (log-odds, clipped to +-20). The receptive-field width starts at 32 px.
Streaming engine (Engine): an event store holding the last P + D windows of every stream and the living neurons;
training, export and the runtime measurement run the same per-window code (only the number of streams and the
gradients differ). Training keeps the engine across TBPTT chunks, so the decision layer and the feedback map are those
of deployment.
Ablation switches (DeCoV12): feedback "innov" | "belief" (log1p of sum sigma(U) and sum 1 - sigma(U)) | "none";
birth "none" | "band" (only events with z0 in the pending band get a neuron, as V10 / V11); the neuron options of
sdn_v11 (value, gate, rho, audit).
"""
import math

import torch
import torch.nn as nn

import sdn_v11

PX = 32.0              # px scale of dx, dy and the start of the receptive-field width
ZS = 5.0               # log-odds scale of the priors fed to the neurons
INF = float("inf")


def logit(p):
    return math.log(p / (1.0 - p))


class DeCoV12(nn.Module):
    def __init__(self, small, P=10, D=10, k_near=8, k_evid=8, window_ms=50, pos_weight=30.0, alpha=0.01,
                 filter_alpha=0.01, feedback="innov", birth="none", value="signed", gate="sigmoid", rho="mean",
                 audit="reaudit", dim=64, heads=4, sigma0=PX):
        super(DeCoV12, self).__init__()
        if feedback not in ("innov", "belief", "none") or birth not in ("none", "band"):
            raise ValueError("feedback must be innov | belief | none, birth none | band")
        if small.merged_decoder or small.has_subthreshold_residual or small.invariant is not None or small.has_dual_memory:
            raise ValueError("V12 expects the plain BIR U-Net (concatenating decoder, no residual / invariant / dual memory)")
        self.small = small
        self.P, self.D, self.k_near, self.k_evid, self.window_ms = int(P), int(D), int(k_near), int(k_evid), int(window_ms)
        self.feedback, self.birth = feedback, birth
        self.pos_weight = float(pos_weight)
        self.log_w = math.log(self.pos_weight)
        self.tau = logit(0.9) - self.log_w                     # official threshold on calibrated log-odds
        half = logit(1.0 - float(alpha))
        self.band = (self.tau - half, self.tau + half)
        self.flo = self.tau - logit(1.0 - float(filter_alpha))
        self.span = float(max(self.P, self.D, 1))
        self.fb = None
        if feedback != "none":
            self.fb = nn.Conv2d(2, small.dec1.out_ch, 3, padding=1, bias=False)
            nn.init.zeros_(self.fb.weight)
        self.big = sdn_v11.SpikingDecisionNeuronsV11(self.P + self.D + 1, self.k_near + self.k_evid, sigma0, dim=dim,
                                                     heads=heads, audit=audit, rho=rho, gate=gate, value=value)
        self.offsets = list(range(-self.P, self.D + 1))
        self.register_buffer("off_t", torch.arange(-self.P, self.D + 1, dtype=torch.float32), persistent=False)

    # ------------------------------------------------------------------------------------------- small network
    def pre_dec1(self, x, states, keep=None, collect=False):
        """Layers enc1 .. dec2 and the dec1 upsampling over a chunk of T windows at once (they get no feedback).
        x [T, B, C, H, W]; states: 7 states or None. Returns s1, deep1 ([T, B, ...]), the 6 new states and, with
        collect, the 6 membranes for the message budget."""
        sm = self.small
        prev = states if states is not None else [None] * 7
        steps, batch = int(x.shape[0]), int(x.shape[1])

        def flat(seq):
            return seq.reshape((steps * batch,) + tuple(seq.shape[2:]))

        def unflat(t):
            return t.view((steps, batch) + tuple(t.shape[1:]))

        def run(i, name, inputs):
            return getattr(sm, name).forward_steps(inputs, prev[i], True, collect, keep=keep)[:3]

        s1, n1, u1 = run(0, "enc1", x)
        s2, n2, u2 = run(1, "enc2", s1)
        s3, n3, u3 = run(2, "enc3", s2)
        s4, n4, u4 = run(3, "enc4", s3)
        s5, n5, u5 = run(4, "dec3", torch.cat([unflat(sm.up3(flat(s4))), s3], 2))
        s6, n6, u6 = run(5, "dec2", torch.cat([unflat(sm.up2(flat(s5))), s2], 2))
        deep1 = unflat(sm.up1(flat(s6)))
        return s1, deep1, [n1, n2, n3, n4, n5, n6], ([u1, u2, u3, u4, u5, u6] if collect else None)

    def dec1_window(self, s1_w, deep1_w, state7, fbmap):
        """dec1 for one window [B, ...]; the feedback map of the previous window enters as an extra input current."""
        extra = self.fb(fbmap) if (self.fb is not None and fbmap is not None) else None
        _, n7, u7, _ = self.small.dec1(torch.cat([deep1_w, s1_w], 1), state7, extra_current=extra)
        return u7, n7

    def readout_window(self, u7, deep1_w, ev):
        """Priors z0 (calibrated log-odds) of the events of one window; ev has b, y, x, p, t_local."""
        feat, _, _ = self.small._event_features(u7, None, deep1_w, ev)
        return self.small._readout(feat, ev) - self.log_w

    @staticmethod
    def message_layers_keep(u_pre6):
        """The message budget reads the membranes of the sending layers (enc1 .. dec2) by layer index; dec1 sends
        nothing (its membrane is read by the readout)."""
        return list(u_pre6) + [None]

    # ------------------------------------------------------------------------------------------- big network
    def token_features(self, st, z, eid, pick, ok):
        """Tokens of neurons eid [n] (store indices) from their candidate picks [n, n_off, K]; z [store] = priors (may
        carry gradients). Returns feat [n, T, 6], ok [n, T], window offset dw [n, T], dist2 [n, T] (px^2)."""
        n, n_off, K = pick.shape                     # n_off may be the first windows only (the arrived ones)
        okf = ok.to(st.xf.dtype)
        safe = torch.where(ok, pick, torch.zeros_like(pick))
        me = eid[:, None, None]
        dx = (st.xf[safe] - st.xf[me]) * okf
        dy = (st.yf[safe] - st.yf[me]) * okf
        dt = (st.tm[safe] - st.tm[me]) * okf
        dw = self.off_t[None, :n_off, None] * okf
        feat = torch.stack([dx / PX, dy / PX, dt / (self.span * self.window_ms), dw / self.span,
                            (st.pf[safe] - 0.5) * okf, torch.clamp(z[safe], -20.0, 20.0) * (okf / ZS)], -1)
        T = n_off * K
        return feat.reshape(n, T, 6), ok.reshape(n, T), dw.reshape(n, T), (dx * dx + dy * dy).reshape(n, T)

    def trajectories(self, st, z, eid, pick, ok, run, own_prior_grad=False):
        """Membrane trajectories U [n, D + 1] (training). run [n, D + 1]: the steps that are computed (later steps keep
        the last membrane). Gradients reach the neighbours' priors (tokens) and the neuron's own prior through its
        context c0; the own prior as the membrane's base U = z0 + evidence only with own_prior_grad (otherwise the
        evidence is fitted on top of the prior held fixed, and the prior is trained by its own birth term)."""
        feat, okT, dw, dist2 = self.token_features(st, z, eid, pick, ok)
        z0 = z[eid]
        base = z0 if own_prior_grad else z0.detach()
        return self.big(feat, okT, dw, run, z0 / ZS, base, st.pf[eid], self.D, dist2)

    def step_rows(self, st, eid, pick, ok, z0, c0, c, U, age):
        """One streaming step of the given neurons, each at its own age (no gradients in the callers)."""
        feat, okT, dw, dist2 = self.token_features(st, st.z, eid, pick, ok)
        return self.big.stream_step(feat, okT, dw, dist2, z0, c0, c, U, age)


def feedback_map(kind, B, H, W, b, y, x, U, z0):
    """[B, 2, H, W] feedback map of the given neurons (see the module docstring)."""
    m = torch.zeros((B, 2, H, W), dtype=U.dtype, device=U.device)
    if U.numel():
        pu = torch.sigmoid(U)
        if kind == "belief":
            a, c = pu, 1.0 - pu
        else:
            pz = torch.sigmoid(z0)
            a, c = torch.relu(pu - pz), torch.relu(pz - pu)
        m.index_put_((b, torch.zeros_like(b), y, x), a, accumulate=True)
        m.index_put_((b, torch.ones_like(b), y, x), c, accumulate=True)
    return torch.log1p(m)


# ----------------------------------------------------------------------------------------------------------------------
# streaming engine
# ----------------------------------------------------------------------------------------------------------------------


class EventStore(object):
    """Events of the most recent windows of every stream, sorted by (global window g, stream b); key = g * B + b.
    xf, yf px; tm ms within the recording; pf polarity 0 / 1; tl position inside the window (the readout's input);
    z detached prior (filled when the window is read); lab label (training); b, y, x; g global window; glo, ghi global
    windows of the first / last window of the event's recording item in its stream; rec, fidx, rw recording index,
    index in the recording file, window in the recording (export)."""
    FLOAT = ("xf", "yf", "tm", "pf", "tl", "z", "lab")
    LONG = ("b", "y", "x", "g", "glo", "ghi", "rec", "fidx", "rw")

    def __init__(self, n_streams, device):
        self.B, self.dev = int(n_streams), device
        for k in self.FLOAT:
            setattr(self, k, torch.zeros(0, device=device))
        for k in self.LONG:
            setattr(self, k, torch.zeros(0, dtype=torch.long, device=device))
        self.key = torch.zeros(0, dtype=torch.long, device=device)

    @property
    def n(self):
        return int(self.key.numel())

    def prune(self, gmin):
        """Drop the events of windows < gmin; returns the old -> new index map (-1 = dropped) or None."""
        keep = self.g >= int(gmin)
        if bool(keep.all()):
            return None
        remap = torch.cumsum(keep.long(), 0) - 1
        remap[~keep] = -1
        for k in self.FLOAT + self.LONG + ("key",):
            setattr(self, k, getattr(self, k)[keep])
        return remap

    def append(self, cols):
        """New events (every FLOAT / LONG field; sorted by (g, b), later windows than the stored ones). Returns the
        store index of the first one."""
        base = self.n
        for k in self.FLOAT:
            setattr(self, k, torch.cat([getattr(self, k), cols[k].to(device=self.dev, dtype=torch.float32)]))
        for k in self.LONG:
            setattr(self, k, torch.cat([getattr(self, k), cols[k].to(device=self.dev, dtype=torch.long)]))
        self.key = torch.cat([self.key, cols["g"].to(self.dev).long() * self.B + cols["b"].to(self.dev).long()])
        return base

    def window(self, g):
        """Store range [a, e) of the events of global window g (all streams)."""
        q = torch.tensor([int(g) * self.B, (int(g) + 1) * self.B], dtype=torch.long, device=self.dev)
        a, e = torch.searchsorted(self.key, q).tolist()
        return int(a), int(e)


def gpu_picks(st, q_eid, q_key, k_near, k_evid, flo, block=4096):
    """Candidates of the queries (event q_eid[i], cell key q_key[i] = window * B + stream): the k_near nearest events of
    the cell and the k_evid nearest non-background events (prior > flo) not already picked, the event itself excluded
    (the staged evidence rule, tools/deco/evidence.candidate_picks). Returns pick [Q, K] (store indices, -1 = none), ok.
    One distance matrix per cell (a contiguous store range, columns in arrival order): a query's row, and so the way
    ties between equal pixel distances are broken, does not depend on which other queries are batched with it, so
    training, export and deployment pick the same candidates."""
    Q, K = int(q_eid.numel()), k_near + k_evid
    pick = torch.full((Q, K), -1, dtype=torch.long, device=st.dev)
    ok = torch.zeros((Q, K), dtype=torch.bool, device=st.dev)
    if Q == 0 or st.n == 0:
        return pick, ok
    cells, inv = torch.unique(q_key, return_inverse=True)
    starts = torch.searchsorted(st.key, cells).tolist()
    ends = torch.searchsorted(st.key, cells, right=True).tolist()
    order = torch.argsort(inv)
    bq = torch.searchsorted(inv[order], torch.arange(len(starts) + 1, device=st.dev)).tolist()
    for ci, (a, e) in enumerate(zip(starts, ends)):
        rows_c = order[bq[ci]:bq[ci + 1]]
        if e <= a or rows_c.numel() == 0:
            continue
        xe, ye = st.xf[a:e], st.yf[a:e]
        ar = torch.arange(a, e, device=st.dev)
        bg = ~(st.z[a:e] > flo)
        kn, ke = min(k_near, e - a), min(k_evid, e - a)
        for s in range(0, rows_c.numel(), block):
            r = rows_c[s:s + block]
            me = q_eid[r]
            d2 = (st.xf[me][:, None] - xe[None, :]) ** 2 + (st.yf[me][:, None] - ye[None, :]) ** 2
            d2 = d2.masked_fill(ar[None, :] == me[:, None], INF)
            dn, jn = torch.topk(d2, kn, dim=1, largest=False)
            near_ok = torch.isfinite(dn)
            pick[r, :kn] = torch.where(near_ok, jn + a, torch.full_like(jn, -1))
            ok[r, :kn] = near_ok
            if ke > 0:
                d2e = d2.masked_fill(bg[None, :], INF)
                d2e.scatter_(1, jn, torch.where(near_ok, torch.full_like(dn, INF), d2e.gather(1, jn)))
                de, je = torch.topk(d2e, ke, dim=1, largest=False)
                evid_ok = torch.isfinite(de)
                pick[r, k_near:k_near + ke] = torch.where(evid_ok, je + a, torch.full_like(je, -1))
                ok[r, k_near:k_near + ke] = evid_ok
    return pick, ok


class Engine(object):
    """The streaming decision layer of B streams: event store + living neurons (one per event, or per pending event
    with birth = band). Per window: append / readout (caller) -> birth -> advance -> feedback -> retire.
    Neuron fields: eid store index, gs birth window, ghi last window of its recording item, b stream, lab label (training),
    pick / ok candidates [n, n_off, K] (int32 store indices), c0 / c context, U membrane, z0 prior, act (still computing);
    with keep_traj also traj / mass / neff [n, D + 1] (the export; NaN where no step was taken). Retired neurons are
    collected in self.retired (lists of field dicts).
    gate (deployment mode, None in training): (lo, hi) indecision band in log-odds. A neuron whose membrane leaves the
    band has fired: it stops computing (no more candidates, no more steps) and stays in the feedback map with its frozen
    belief until its deadline. Without a gate every neuron computes all its D + 1 steps.
    stats: neuron steps, tokens audited (every step re-embeds its tokens in this implementation) and tokens that
    arrived (what an embed-once implementation embeds), for the runtime report."""

    def __init__(self, model, n_streams, height, width, device, keep_traj=False, sub_batch=4096, pick_block=4096,
                 gate=None):
        self.m, self.B, self.H, self.W, self.dev = model, int(n_streams), int(height), int(width), device
        self.keep_traj, self.sub, self.pick_block = keep_traj, int(sub_batch), int(pick_block)
        self.gate = None if gate is None else (float(gate[0]), float(gate[1]))
        self.st = EventStore(n_streams, device)
        self.pop = self._empty()
        self.retired = []
        self.stats = {"steps": 0, "tokens": 0, "births": 0, "new_tokens": 0}

    def _empty(self):
        m, dev = self.m, self.dev
        n_off, K, dim, D = len(m.offsets), m.k_near + m.k_evid, m.big.dim, m.D
        long0 = torch.zeros(0, dtype=torch.long, device=dev)
        d = {"eid": long0, "gs": long0.clone(), "ghi": long0.clone(), "b": long0.clone(),
             "lab": torch.zeros(0, device=dev), "pick": torch.zeros((0, n_off, K), dtype=torch.int32, device=dev),
             "ok": torch.zeros((0, n_off, K), dtype=torch.bool, device=dev), "c0": torch.zeros((0, dim), device=dev),
             "c": torch.zeros((0, dim), device=dev), "U": torch.zeros(0, device=dev), "z0": torch.zeros(0, device=dev),
             "act": torch.zeros(0, dtype=torch.bool, device=dev)}
        if self.keep_traj:
            for k in ("traj", "mass", "neff"):
                d[k] = torch.zeros((0, D + 1), device=dev)
        return d

    @property
    def n(self):
        return int(self.pop["eid"].numel())

    def begin_chunk(self, gmin, slack=8):
        """Drop the store's events of windows < gmin (no living neuron reads them; done once the oldest stored
        window is slack windows older, so short chunks do not prune every window) and the retired list."""
        self.retired = []
        if self.st.n == 0 or int(self.st.g[0]) >= int(gmin) - int(slack):
            return
        remap = self.st.prune(gmin)
        if remap is not None and self.n:
            p = self.pop
            pk = p["pick"].long()
            new = torch.where(p["ok"], remap[pk.clamp(min=0)], torch.full_like(pk, -1))
            if bool((new[p["ok"]] < 0).any()) or bool((remap[p["eid"]] < 0).any()):
                raise RuntimeError("store pruned an event a living neuron still reads")
            p["pick"] = new.to(torch.int32)
            p["eid"] = remap[p["eid"]]
        self.retired = []

    def _place(self, rows, cols, eids, keys):
        pick, ok = gpu_picks(self.st, eids, keys, self.m.k_near, self.m.k_evid, self.m.flo, self.pick_block)
        self.pop["pick"][rows, cols] = pick.to(torch.int32)
        self.pop["ok"][rows, cols] = ok

    def birth(self, g, a, e):
        """Neurons for the events of window g (store range [a, e)): all of them, or the pending ones (birth = band)."""
        m, st = self.m, self.st
        if e <= a:
            return
        eid = torch.arange(a, e, device=self.dev)
        z = st.z[a:e]
        if m.birth == "band":
            eid = eid[(z > m.band[0]) & (z < m.band[1])]
            z = st.z[eid]
        n = int(eid.numel())
        if n == 0:
            return
        with torch.no_grad():
            c0 = m.big.stream_start(z / ZS, st.pf[eid])
        rec = {"eid": eid, "gs": torch.full((n,), int(g), dtype=torch.long, device=self.dev), "ghi": st.ghi[eid],
               "b": st.b[eid], "lab": st.lab[eid], "c0": c0, "c": c0.clone(), "U": z.clone(), "z0": z.clone(),
               "pick": torch.full((n, len(m.offsets), m.k_near + m.k_evid), -1, dtype=torch.int32, device=self.dev),
               "ok": torch.zeros((n, len(m.offsets), m.k_near + m.k_evid), dtype=torch.bool, device=self.dev),
               "act": torch.ones(n, dtype=torch.bool, device=self.dev)}
        self.stats["births"] += n
        self.stats["new_tokens"] += n * (m.P + 1) * (m.k_near + m.k_evid)      # embedded once, when they arrive
        if self.keep_traj:
            for k in ("traj", "mass", "neff"):
                rec[k] = torch.full((n, m.D + 1), float("nan"), device=self.dev)
        base = self.n
        self.pop = {k: torch.cat([self.pop[k], rec[k]]) for k in self.pop}
        offs = torch.arange(-m.P, 1, device=self.dev)                       # the past and the birth window
        gw = int(g) + offs[None, :].expand(n, m.P + 1)
        valid = gw >= st.glo[eid][:, None]
        ri, oi = torch.nonzero(valid, as_tuple=True)
        keys = gw[ri, oi] * self.B + st.b[eid][ri]
        self._place(base + ri, oi, eid[ri], keys)

    def live(self, g):
        p = self.pop
        age = int(g) - p["gs"]
        return torch.nonzero((age >= 0) & (age <= self.m.D) & (int(g) <= p["ghi"]), as_tuple=True)[0], age

    def advance(self, g):
        """Window g has been read: every living neuron of age a >= 1 gets its offset-a candidates, then every living
        neuron takes its step (a = 0 for the ones born in g); with a gate only the ones that have not fired, and those
        whose membrane leaves the band fire now. No gradients."""
        m, p = self.m, self.pop
        if self.n == 0:
            return
        rows, age = self.live(g)
        if self.gate is not None:
            rows = rows[p["act"][rows]]
        if rows.numel() == 0:
            return
        old = rows[age[rows] >= 1]
        if old.numel():
            self._place(old, age[old] + m.P, p["eid"][old], int(g) * self.B + p["b"][old])
            self.stats["new_tokens"] += int(old.numel()) * (m.k_near + m.k_evid)
        with torch.no_grad():
            if rows.numel() <= self.sub:             # few neurons (launch-bound): one call, one age per row
                groups = [(rows, age[rows], len(m.offsets))]
            else:                                    # many (compute-bound): per age, arrived windows only
                ages = age[rows]
                groups = [(rows[ages == a], int(a), m.P + 1 + int(a)) for a in torch.unique(ages).tolist()]
            for ra, a, w_ in groups:
                for s in range(0, ra.numel(), self.sub):
                    r = ra[s:s + self.sub]
                    ar = a[s:s + self.sub] if torch.is_tensor(a) else a
                    U, c, mass, neff = m.step_rows(self.st, p["eid"][r], p["pick"][r, :w_].long(), p["ok"][r, :w_],
                                                   p["z0"][r], p["c0"][r], p["c"][r], p["U"][r], ar)
                    p["U"][r], p["c"][r] = U, c
                    self.stats["steps"] += int(r.numel())
                    self.stats["tokens"] += int(r.numel()) * w_ * (m.k_near + m.k_evid)
                    if self.keep_traj:
                        p["traj"][r, ar] = U
                        p["mass"][r, ar] = mass
                        p["neff"][r, ar] = neff
            if self.gate is not None:
                u = p["U"][rows]
                p["act"][rows] = (u > self.gate[0]) & (u < self.gate[1])

    def feedback(self, g):
        """Feedback map [B, 2, H, W] of the living neurons after window g (None without feedback)."""
        m = self.m
        if m.fb is None:
            return None
        with torch.no_grad():
            if self.n == 0:
                return torch.zeros((self.B, 2, self.H, self.W), device=self.dev)
            rows, _ = self.live(g)
            eid = self.pop["eid"][rows]
            return feedback_map(m.feedback, self.B, self.H, self.W, self.st.b[eid], self.st.y[eid], self.st.x[eid],
                                self.pop["U"][rows], self.pop["z0"][rows])

    def retire(self, g):
        """Neurons that took their last step in window g (age D or the end of their recording) leave the population."""
        if self.n == 0:
            return
        p = self.pop
        done = ((int(g) - p["gs"]) >= self.m.D) | (int(g) >= p["ghi"])
        if not bool(done.any()):
            return
        keep = ~done
        self.retired.append({k: v[done] for k, v in p.items()})
        self.pop = {k: v[keep] for k, v in p.items()}

    def chunk_neurons(self):
        """Every neuron that lived during the current chunk: the retired ones and the living ones (field dict)."""
        parts = self.retired + [self.pop]
        return {k: torch.cat([q[k] for q in parts]) for k in self.pop}
