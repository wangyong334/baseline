"""Slot 7, V4-3: tube evidence along learned motion hypotheses (V2's tube test with learned hypotheses and weights).

Each event (step b, pixel q) receives K hypotheses at its birth from the anchor motion module (network.motion): a
displacement per step d_j and a mixture weight w_j (top-K anchors, renormalised; computed from steps <= b only).
At step m = b + s, for every hypothesis j:
    tube position          p_j = round(q + d_j s),   previous position   p'_j = round(q + d_j (s - 1))
    forecast (H1)          G(o) = g_{m-1}(p'_j + o)        target intensity one step earlier along the same tube
    background (H0)        mu_m(p_j + o)                    learned background forecast made at step m - 1
    per-offset evidence    e(o) = N_m(p_j + o) log(1 + G(o) / mu) - G(o)      (0 where p_j + o leaves the canvas;
                                                                                 E_j = 0 once p_j itself has left it)
    position mixture       E_j = log sum_o pi_o exp(e(o))   (3 x 3 offsets, learned pi)
    anchored chain         r_j += alive_j ? E_j : min(E_j, 0);  alive_j &= an event within 1 px of p_j   (V3)
    score                  F = log sum_j w_j exp(r_j)
With the 49 grid velocities, uniform weights and uniform pi this equals V2's drift evidence without track memory and
gate (tests/test_v43.py). All mixtures have weights summing to one, fixed before the counts they weigh, so exp(F) is a
test supermartingale under H0 when mu is the background rate. The loss of slot 9 calls the same functions.
"""
import torch
import torch.nn.functional as F

OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
BACKGROUNDS = ("head", "front")


def presence_support(counts):
    """Anchor support: an event within one pixel of each position."""
    return F.max_pool2d((counts > 0).to(counts.dtype), 3, stride=1, padding=1)


def tube_step(N, f_now, mu, f_mu, g, f_prev, support, y, x, disp, age, run, alive, log_pi):
    """One step of the chain for n events of one age.
    N [F,1,H,W] counts of the step (rows f_now [n]); mu background rate of that step (rows f_mu [n]); g target
    intensity of the step before (rows f_prev [n]); support [F,1,H,W] (rows f_now) or None (no anchor);
    y, x [n] birth pixels; disp [K,n,2] px per step; age >= 1; run, alive [K,n] -> (run, alive)."""
    K, n = int(disp.shape[0]), int(disp.shape[1])
    H, W = int(N.shape[-2]), int(N.shape[-1])
    dt = run.dtype
    dev = y.device
    a = float(age)
    off = torch.tensor(OFFSETS, dtype=torch.long, device=dev)
    yf, xf = y.to(dt).view(1, n), x.to(dt).view(1, n)
    d = disp.to(dt)
    with torch.no_grad():
        py, px = torch.floor(yf + d[..., 0] * a + 0.5).long(), torch.floor(xf + d[..., 1] * a + 0.5).long()
        qy, qx = torch.floor(yf + d[..., 0] * (a - 1.0) + 0.5).long(), torch.floor(xf + d[..., 1] * (a - 1.0) + 0.5).long()
        Y, X = py.unsqueeze(-1) + off[:, 0], px.unsqueeze(-1) + off[:, 1]                       # [K, n, 9]
        Yq, Xq = qy.unsqueeze(-1) + off[:, 0], qx.unsqueeze(-1) + off[:, 1]
        in_now = (Y >= 0) & (Y < H) & (X >= 0) & (X < W)
        inside = (py >= 0) & (py < H) & (px >= 0) & (px < W)
        in_prev = (Yq >= 0) & (Yq < H) & (Xq >= 0) & (Xq < W)
        rows = lambda f: f.view(1, n, 1).expand(K, n, len(OFFSETS))  # noqa: E731
        i_now = (rows(f_now) * H + Y.clamp(0, H - 1)) * W + X.clamp(0, W - 1)
        i_mu = (rows(f_mu) * H + Y.clamp(0, H - 1)) * W + X.clamp(0, W - 1)
        i_prev = (rows(f_prev) * H + Yq.clamp(0, H - 1)) * W + Xq.clamp(0, W - 1)
        counts = N.reshape(-1)[i_now].to(dt) * in_now.to(dt)
    G = g.reshape(-1)[i_prev].to(dt) * in_prev.to(dt)
    rate = mu.reshape(-1)[i_mu].to(dt)
    e = (counts * torch.log1p(G / rate) - G) * in_now.to(dt)
    E = torch.logsumexp(e + log_pi.to(dt).view(1, 1, -1), -1) * inside.to(dt)
    if support is None:
        return run + E, alive
    with torch.no_grad():
        idx = (f_now.view(1, n) * H + py.clamp(0, H - 1)) * W + px.clamp(0, W - 1)
        held = (support.reshape(-1)[idx] > 0) & inside
    return run + torch.where(alive, E, E.clamp(max=0.0)), alive & held


def score(run, logw):
    """run, logw [K,n] -> F [n]."""
    return torch.logsumexp(run + logw.to(run.dtype), 0)


class TubeEvidence(object):
    """Per-step state for the learned readouts: counts, background rate, previous intensity, anchor support, and the
    hypotheses of the step's events (computed here once, from the motion features of this and the previous step)."""
    takes_outputs = True

    def __init__(self, dt_ms, head, motion, hypotheses=5, mu_floor=1e-3, background="head", anchor=True):
        if background not in BACKGROUNDS:
            raise ValueError("background must be one of %s" % (BACKGROUNDS,))
        if int(hypotheses) < 1 or float(mu_floor) <= 0:
            raise ValueError("need hypotheses >= 1 and mu_floor > 0")
        self.dt, self.head, self.motion = float(dt_ms), head, motion
        self.n_hypotheses = min(int(hypotheses), motion.size)
        self.mu_floor, self.background, self.anchor = float(mu_floor), background, bool(anchor)

    def background_rate(self, log_mu, mu0):
        if self.background == "head" and log_mu is not None:
            return self.mu_floor + torch.exp(log_mu)
        return mu0

    def support(self, counts):
        return presence_support(counts) if self.anchor else None

    def hypotheses(self, cur, prev, frame, y, x):
        """Pyramids of the motion features of the birth step and the step before -> (disp [K,n,2], logw [K,n],
        anchor indices [n,K], best displacement [n,2])."""
        logits, residual = self.motion.query(cur, prev, frame, y, x)
        disp, logw, idx = self.motion.hypotheses(logits, residual, self.n_hypotheses)
        return disp, logw, idx, self.motion.best(logits, residual)

    def init_state(self, batch, height, width, device, dtype=torch.float32):
        return {"k": 0, "pyr_prev": None, "hyp": None, "queries": 0}

    def step(self, state, counts, mu0, log_g_prev, events=None, motion_prev=None, outputs_prev=None, outputs=None):
        k = int(state["k"])
        log_mu = None if outputs_prev is None else outputs_prev.get("log_mu")
        new = {"N": counts, "mu": self.background_rate(log_mu, mu0), "support": self.support(counts), "k": k + 1,
               "g_prev": torch.zeros_like(counts) if log_g_prev is None else torch.exp(log_g_prev),
               "pyr_prev": state["pyr_prev"], "hyp": None, "queries": 0}
        if outputs is not None:
            cur = self.motion.pyramid(outputs["phi"])
            prev = state["pyr_prev"] if state["pyr_prev"] is not None else [torch.zeros_like(c) for c in cur]
            n = 0 if events is None else int(events["y"].shape[0])
            if n:
                disp, logw, idx, best = self.hypotheses(cur, prev, events["b"], events["y"], events["x"])
                new["hyp"] = {"disp": disp, "logw": logw, "idx": idx, "best": best}
            new.update(pyr_prev=cur, queries=n)
        return new

    def operations(self, height, width, active_fraction=1.0, events_per_step=0.0, queries_per_step=0.0, **stats):
        """Per step: pyramid pooling of the motion features, anchor scoring of the step's events, background rate and
        anchor support over the canvas (the per-event chains are counted with the readouts)."""
        p = float(int(height) * int(width))
        q = float(queries_per_step)
        c = float(self.motion.channels)
        per = self.motion.operations_per_query()
        parts = [("motion pyramid", 0.0, c * p * (1.0 / 4 + 1.0 / 16 + 1.0 / 64), 0.0),
                 ("anchor motion queries", per["mac"] * q, per["ac"] * q, per["transcendental"] * q)]
        if self.background == "head":
            parts.append(("background rate", 0.0, p, p))
        if self.anchor:
            parts.append(("anchor support", 0.0, 10.0 * p, 0.0))
        return [{"part": name, "mac": float(mac), "elementwise": float(el), "transcendental": float(tr)}
                for name, mac, el, tr in parts]
