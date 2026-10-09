"""Slot 7, V4-2: learned evidence along the learned motion.

The V2 / V3 verifier keeps its structure (Poisson likelihood ratio of a forecast against the background, mixture over
motion hypotheses, anchored evidence chain); every quantity inside now comes from the network or a training objective.
From the heads of step k-1 (causal):
    (1) cloud       5 sigma points of N(v, s^2 I), s^2 = sigma^2 + spacing^2 / 3 (px / step), plus zero velocity;
                    weights (1 - w0) x UT weights for the sigma points, w0 = 1/2 exp(-|v|^2 / (2 s^2)) for zero
    (2) forecast    G_j = splat(g 1[g >= gate], d_j) for the sigma points, G_0 = g 1[g >= gate]
    (3) background  mu = mu_floor + exp(log_mu)  (learned background head; the front end's mu0 without it)
    (4) evidence    e_j = N log(1 + G_j / mu) - G_j                       (N = event count of step k)
    (5) mixture     E_j(x) = log sum_o pi_o exp(e_j(x + o))              (pi = learned position weights, 5 x 5)
An event born at step b at pixel q reads E_j of step m at q + d_j(b) (m - b), bilinearly, for each hypothesis j:
    (6) chain       r_j += alive_j ? E_j : min(E_j, 0),  alive_j &= support at the rounded path position   (V3)
    (7) score       F = log sum_j w_j exp(r_j)
Every mixture (offsets pi, bilinear corners, hypotheses w) has weights summing to at most one and fixed before the
counts it weighs, so exp(F) is a test supermartingale under H0 when mu is the background rate (Ville).
The same functions run in training (slot 9, LearnedEvidenceLoss) and in inference (this verifier, learned readouts).
"""
import math

import torch
import torch.nn.functional as F

from speed.core.splat import sample_bilinear, splat_points

UT_SCALE = 3.0                                      # n + lambda = 3 for n = 2 (Julier & Uhlmann 2004)
UT_WEIGHTS = (1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
UT_DIRECTIONS = ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))   # centre, +-y, +-x (x sqrt(3) s)
N_MOVING = len(UT_WEIGHTS)
N_HYPOTHESES = N_MOVING + 1                         # sigma points first, zero velocity last
E_CLIP = 80.0                                       # exp range guard of the position mixture (float32)
BACKGROUNDS = ("head", "front")


def sigma_cloud(velocity, log_sigma, dt_ms, spacing):
    """(1): velocity [..., 2] (vy, vx px/ms), log_sigma [...] (px/ms) -> displacement per step of the sigma points
    [5, ..., 2] and log mixture weights [6, ...]."""
    centre = velocity * float(dt_ms)
    s2 = torch.exp(2.0 * log_sigma) * float(dt_ms) ** 2 + float(spacing) ** 2 / UT_SCALE
    k = torch.sqrt(UT_SCALE * s2).unsqueeze(-1).unsqueeze(0)
    dirs = torch.tensor(UT_DIRECTIONS, dtype=centre.dtype, device=centre.device)
    pts = centre.unsqueeze(0) + k * dirs.view((N_MOVING,) + (1,) * (centre.dim() - 1) + (2,))
    log_w0 = math.log(0.5) - 0.5 * (centre * centre).sum(-1) / s2
    log_moving = torch.log1p(-torch.exp(log_w0))
    logw = torch.stack([log_moving + math.log(w) for w in UT_WEIGHTS] + [log_w0])
    return pts, logw


def forecast_maps(log_g, motion, dt_ms, spacing, gate):
    """(2): log_g [N,1,H,W], motion [N,3,H,W] of the previous step -> G [N,6,H,W], number of pushed sources.
    Only sources with g >= gate are pushed (sparse execution); the zero channel is the gated g itself."""
    N, _, H, W = (int(n) for n in log_g.shape)
    plane = H * W
    g = torch.exp(log_g)
    keep = g >= float(gate)
    zero = g * keep.to(g.dtype)
    flat = keep.reshape(-1).nonzero().view(-1)
    n = int(flat.numel())
    if n == 0:
        return torch.cat([torch.zeros_like(g)] * N_MOVING + [zero], 1), 0
    frame, rem = torch.div(flat, plane, rounding_mode="floor"), flat % plane
    y, x = torch.div(rem, W, rounding_mode="floor"), rem % W
    m = motion.reshape(N, 3, plane)[frame, :, rem]
    pts, _ = sigma_cloud(m[:, :2], m[:, 2], dt_ms, spacing)
    values = g.reshape(-1)[flat]
    moved = [splat_points(values, frame, y, x, pts[j, :, 0], pts[j, :, 1], (N, H, W)) for j in range(N_MOVING)]
    return torch.cat(moved + [zero], 1), n


def log_likelihood_ratio(counts, mu, G):
    """(4): counts, mu [N,1,H,W], G [N,V,H,W] -> e [N,V,H,W]."""
    return counts * torch.log1p(G / mu) - G


def position_mixture(e, log_pi, radius):
    """(5): log of the pi-weighted mean of exp(e) over the (2r+1)^2 offsets; outside the canvas e = 0 (no information).
    log_pi [(2r+1)^2] in row-major offset order (dy, dx)."""
    V, r = int(e.shape[1]), int(radius)
    k = 2 * r + 1
    kernel = torch.exp(log_pi).to(e.dtype).view(1, 1, k, k).expand(V, 1, k, k)
    padded = F.pad(torch.exp(e.clamp(-E_CLIP, E_CLIP)), (r, r, r, r), value=1.0)
    return torch.log(F.conv2d(padded, kernel, groups=V))


def presence_support(counts, radius=1):
    """Anchor support: an event within one pixel (the pixel grid's rounding tolerance) of each position."""
    present = (counts > 0).to(counts.dtype)
    return F.max_pool2d(present, 2 * int(radius) + 1, stride=1, padding=int(radius))


def chain_step(E, support, frame, y, x, pts, age, run, alive):
    """(6) for n events of one age: E [F,6,H,W] and support [F,1,H,W] (None = no anchor) of the step being read,
    frame [n] = their row, y, x [n] birth pixels, pts [5,n,2] px/step, age int -> run, alive [6,n]."""
    n = int(y.shape[0])
    H, W = int(E.shape[2]), int(E.shape[3])
    a = float(age)
    yf, xf = y.to(E.dtype).view(1, n), x.to(E.dtype).view(1, n)
    py = torch.cat([yf + pts[..., 0].to(E.dtype) * a, yf])
    px = torch.cat([xf + pts[..., 1].to(E.dtype) * a, xf])
    f = frame.view(1, n).expand(N_HYPOTHESES, n)
    c = torch.arange(N_HYPOTHESES, device=y.device).view(N_HYPOTHESES, 1).expand(N_HYPOTHESES, n)
    values = sample_bilinear(E, f, c, py, px)
    if support is None:
        return run + values, alive
    with torch.no_grad():
        ry, rx = torch.floor(py + 0.5).long(), torch.floor(px + 0.5).long()
        inside = (ry >= 0) & (ry < H) & (rx >= 0) & (rx < W)
        held = (support[f, 0, ry.clamp(0, H - 1), rx.clamp(0, W - 1)] > 0) & inside
    run = run + torch.where(alive, values, values.clamp(max=0.0))
    return run, alive & held


def score(run, logw):
    """(7): run, logw [6,n] -> F [n]."""
    return torch.logsumexp(run + logw.to(run.dtype), 0)


class LearnedEvidence(object):
    """Verifier state per step: E (formula 5) and the anchor support; readouts and the loss read events along paths.
    head: the network's EvidenceReadoutHead (position weights); background "head" uses the learned log_mu of the
    previous step, "front" the front end's mu0 (ablation: V2's background model)."""
    takes_outputs = True
    n_hypotheses = N_HYPOTHESES

    def __init__(self, dt_ms, head, spacing=1.0, mu_floor=1e-3, source_gate=1e-3, background="head", anchor=True):
        if background not in BACKGROUNDS:
            raise ValueError("background must be one of %s" % (BACKGROUNDS,))
        if float(spacing) < 0 or float(mu_floor) <= 0 or float(source_gate) < 0:
            raise ValueError("need spacing >= 0, mu_floor > 0 and source_gate >= 0")
        self.dt, self.head = float(dt_ms), head
        self.spacing, self.mu_floor, self.gate = float(spacing), float(mu_floor), float(source_gate)
        self.background, self.anchor = background, bool(anchor)

    @property
    def radius(self):
        return self.head.radius

    def background_rate(self, log_mu, mu0):
        if self.background == "head" and log_mu is not None:
            return self.mu_floor + torch.exp(log_mu)
        return mu0

    def maps(self, counts, mu, log_g, motion):
        """(2)-(5) for one or many steps -> (E [N,6,H,W], G [N,6,H,W], pushed sources)."""
        G, sources = forecast_maps(log_g, motion, self.dt, self.spacing, self.gate)
        e = log_likelihood_ratio(counts, mu, G)
        return position_mixture(e, self.head.mix_log_weights(), self.radius), G, sources

    def support(self, counts):
        return presence_support(counts) if self.anchor else None

    def event_cloud(self, motion, frame, y, x):
        """Each event's cloud from the motion of its own step: motion [F,3,H,W]; frame, y, x [n] -> pts, logw."""
        m = motion[frame, :, y, x]
        return sigma_cloud(m[:, :2], m[:, 2], self.dt, self.spacing)

    def init_state(self, batch, height, width, device, dtype=torch.float32):
        zeros = torch.zeros(batch, N_HYPOTHESES, height, width, device=device, dtype=dtype)
        return {"E": zeros, "G": zeros, "support": None, "k": 0, "sources": 0}

    def step(self, state, counts, mu0, log_g_prev, events=None, motion_prev=None, outputs_prev=None):
        k = int(state["k"])
        support = self.support(counts)
        if log_g_prev is None or motion_prev is None:
            zeros = counts.new_zeros((int(counts.shape[0]), N_HYPOTHESES) + tuple(counts.shape[2:]))
            return {"E": zeros, "G": zeros, "support": support, "k": k + 1, "sources": 0}
        log_mu = None if outputs_prev is None else outputs_prev.get("log_mu")
        E, G, sources = self.maps(counts, self.background_rate(log_mu, mu0), log_g_prev, motion_prev)
        return {"E": E, "G": G, "support": support, "k": k + 1, "sources": sources}

    def active_fraction(self, state):
        """Pixels where some forecast is non-zero, dilated by the position mixture: E = 0 exactly elsewhere."""
        active = (state["G"].sum(1, keepdim=True) > 0).to(state["G"].dtype)
        r = self.radius
        return float(F.max_pool2d(active, 2 * r + 1, stride=1, padding=r).mean())

    def operations(self, height, width, active_fraction=1.0, events_per_step=0.0, queries_per_step=0.0, **stats):
        """Per step: cloud and 4-way splat of the pushed sources (queries = sources); likelihood ratio, exp, mixture
        and log over the active region; background rate over the active region and anchor support over the canvas."""
        p = float(int(height) * int(width))
        a, s, V = p * float(active_fraction), float(queries_per_step), float(N_HYPOTHESES)
        k2 = float((2 * self.radius + 1) ** 2)
        parts = [("sigma cloud of pushed sources", 12.0 * s, 10.0 * s, 2.0 * s),
                 ("forecast splat", 8.0 * N_MOVING * s, 6.0 * N_MOVING * s, 0.0),
                 ("per-pixel log-likelihood ratio", V * a, 2.0 * V * a, V * a),
                 ("position mixture", k2 * V * a, V * a, 2.0 * V * a)]
        if self.background == "head":
            parts.append(("background rate", 0.0, a, a))
        if self.anchor:
            parts.append(("anchor support", 0.0, 10.0 * p, 0.0))
        return [{"part": name, "mac": float(mac), "elementwise": float(el), "transcendental": float(tr)}
                for name, mac, el, tr in parts]
