"""V17 spiking decision units (SDU) on shared evidence maps (11-13; equations: docs/V17_METHOD.md).

A clock-driven streaming backbone commits every event when its 50 ms window closes, so events whose evidence is still
arriving (target onsets, object borders) are decided too early. Sequential analysis treats each uncertain event as a
two-hypothesis test (target / background) that starts from the backbone's prior and stops once the evidence suffices;
its neural form is a decision unit that integrates evidence towards a bound (drift-diffusion). The classical test
assumes observations that belong to the test, are independent and are not cut by a deadline. On an event stream none
of this holds, so a decision unit
    reads evidence from shared per-window maps through a fixed foveated set of boxes around its event   (Eq. 1-2)
    weighs every box by the probability that it holds the event's own source (attribution)            (Eq. 3)
    treats the attributed evidence of one window as one observation (attributed mean: the events of one source in
    one window are strongly correlated) and lets every new window add a new observation                (Eq. 4)
    recomputes its membrane from all windows so far: target channels excite, background ones inhibit   (Eq. 5)
    forms its queries from a context updated by its last audit (what it has seen conditions its reading) (Eq. 6)
When it fires (the stopping rule, Eq. 7) is calibrated on held-out recordings: tools/v17/stopping.py.
No neighbour search: the maps are built once per window and shared by every unit; a unit reads a fixed number of boxes.
Cost (11-13 evening): with events as observations a box without events has weight 0, so only boxes with events are
computed (event-driven); the token form "glm" reads each box with a context-modulated linear model of its count
statistics instead of a d-dimensional embedding (about 1/50 of the multiply-accumulates of the form "mlp").
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEAT = 5                      # log1p(n_same), log1p(n_opp), log1p(q_same), log1p(q_opp), in-frame area fraction
CENTRE = 4                      # index of the level-0 centre box (it holds the event itself)
Q_SCALE = 4096                  # fixed-point scale of q in the integer maps (exact, order-independent sums)
EPS = 1e-6
TOKENS = ("mlp", "linear", "glm")  # how a unit reads a box (see SpikingDecisionUnits)


def box_layout(levels=4):
    """Foveated boxes around an event (Eq. 1): level 0 = the 3 x 3 boxes of 3 px around it, level l > 0 = the ring of
    8 boxes of 3^(l+1) px around level l - 1, so the boxes tile the square of side 3^(levels+1) px centred on the event
    without overlap (fine near the event, coarse far away). Returns rect [K, 4] = (x0, x1, y0, y1) inclusive pixel
    offsets, the centre distance r [K] (px) and the level [K]; K = 1 + 8 * levels."""
    rects, lev = [], []
    for level in range(int(levels)):
        s = 3 ** (level + 1)
        h = s // 2
        for j in (-1, 0, 1):
            for i in (-1, 0, 1):
                if level and not i and not j:
                    continue
                rects.append((i * s - h, i * s + h, j * s - h, j * s + h))
                lev.append(level)
    rect = np.asarray(rects, np.int64)
    r = np.hypot((rect[:, 0] + rect[:, 1]) / 2.0, (rect[:, 2] + rect[:, 3]) / 2.0)
    return rect, r, np.asarray(lev, np.int64)


def quantised_q(z0):
    """sigmoid(z0) in the fixed-point unit of the maps (int64 tensor)."""
    return torch.round(torch.sigmoid(z0.float()) * Q_SCALE).long()


class EvidenceMaps(object):
    """Integral images of one recording (Eq. 2): for every window and polarity, the event count n and the expected
    number of target events q = sum sigmoid(z0) of each pixel (z0 = calibrated prior of the frozen backbone). Integer
    arithmetic, so every box sum is exact and independent of the summation order; a box sum costs four lookups."""

    def __init__(self, x, y, win, pol, z0, n_win, height, width, device):
        dev = torch.device(device)
        self.n_win, self.H, self.W = int(n_win), int(height), int(width)
        P = (self.H + 1) * (self.W + 1)
        x, y, win, pol = (torch.as_tensor(np.asarray(a), dtype=torch.long, device=dev) for a in (x, y, win, pol))
        z0 = torch.as_tensor(np.asarray(z0), dtype=torch.float32, device=dev)
        if x.numel() and (int(x.min()) < 0 or int(x.max()) >= self.W or int(y.min()) < 0 or int(y.max()) >= self.H):
            raise ValueError("events outside the %d x %d sensor" % (self.W, self.H))
        flat = torch.zeros(self.n_win * 4 * P, dtype=torch.int32, device=dev)
        base = win * (4 * P) + (y + 1) * (self.W + 1) + (x + 1)
        flat.index_add_(0, base + pol * P, torch.ones_like(base, dtype=torch.int32))
        flat.index_add_(0, base + (2 + pol) * P, quantised_q(z0).int())
        maps = flat.view(self.n_win, 4, self.H + 1, self.W + 1)
        maps = torch.cumsum(torch.cumsum(maps, 2, dtype=torch.int32), 3, dtype=torch.int32)
        self.maps = maps.permute(0, 2, 3, 1).reshape(self.n_win, P, 4).contiguous()   # [window, pixel, channel]
        self.device = dev

    def window_boxes(self, x, y, rect):
        """Corner indices and in-frame area fraction of every box of every event: idx [4, B, K], area [B, K]."""
        rect = torch.as_tensor(rect, dtype=torch.long, device=self.device)
        x0 = (x[:, None] + rect[None, :, 0]).clamp(0, self.W)
        x1 = (x[:, None] + rect[None, :, 1] + 1).clamp(0, self.W)
        y0 = (y[:, None] + rect[None, :, 2]).clamp(0, self.H)
        y1 = (y[:, None] + rect[None, :, 3] + 1).clamp(0, self.H)
        full = ((rect[:, 1] - rect[:, 0] + 1) * (rect[:, 3] - rect[:, 2] + 1)).float()
        area = ((x1 - x0) * (y1 - y0)).float() / full[None, :]
        w1 = self.W + 1
        idx = torch.stack([y1 * w1 + x1, y0 * w1 + x1, y1 * w1 + x0, y0 * w1 + x0])
        return idx, area

    def features(self, x, y, win, pol, z0, rect, n_steps, first=0, own_window=True):
        """Box features of events for the windows win + first .. win + first + n_steps - 1 (Eq. 2):
        feat [B, n_steps, K, N_FEAT] and window mask [B, n_steps] (False past the recording end). Polarity is relative
        to the event (same / opposite); the event itself is removed from its centre box in its own window (the first
        window read when first == 0 and own_window: a streaming caller that reads a later window passes False)."""
        dev = self.device
        x, y, win, pol = (torch.as_tensor(a, dtype=torch.long, device=dev) for a in (x, y, win, pol))
        z0 = torch.as_tensor(z0, dtype=torch.float32, device=dev)
        B, S = x.shape[0], int(n_steps)
        idx, area = self.window_boxes(x, y, rect)
        K = idx.shape[2]
        steps = torch.arange(first, first + S, device=dev)
        w = win[:, None] + steps[None, :]
        ok = w < self.n_win
        wc = w.clamp(max=self.n_win - 1)
        sign = (1, -1, -1, 1)
        tot = torch.zeros(B, S, K, 4, dtype=torch.long, device=dev)
        for c in range(4):
            tot += sign[c] * self.maps[wc[:, :, None], idx[c][:, None, :]].long()
        same = pol.view(B, 1, 1, 1).expand(B, S, K, 1)
        n_same = tot[..., 0:2].gather(3, same).squeeze(3)
        n_opp = tot[..., 0:2].gather(3, 1 - same).squeeze(3)
        q_same = tot[..., 2:4].gather(3, same).squeeze(3)
        q_opp = tot[..., 2:4].gather(3, 1 - same).squeeze(3)
        if first == 0 and own_window:                        # remove the event itself from its own centre box
            n_same[:, 0, CENTRE] -= 1
            q_same[:, 0, CENTRE] -= quantised_q(z0)
        feat = torch.stack([torch.log1p(n_same.clamp(min=0).float()), torch.log1p(n_opp.float()),
                            torch.log1p(q_same.clamp(min=0).float() / Q_SCALE), torch.log1p(q_opp.float() / Q_SCALE),
                            area[:, None, :].expand(B, S, K)], -1)
        return feat * ok[:, :, None, None].float(), ok


class SpikingDecisionUnits(nn.Module):
    """One decision unit per uncertain event; membrane U = posterior log-odds that the event belongs to a target.

    Step k (k = 0 .. n_steps - 1) = after the event's window and the k following ones have arrived. Tokens are the boxes
    of those windows (Eq. 1-2), embedded once: content (linear) + slot (window offset, box) -> keys and values. At every
    step each head h re-audits all windows so far with the unit's current context c (Eq. 3-5):
        a_hj = sigmoid(q_h(c, k) . key_j / sqrt(d_h) + b_h - r_j^2 / (2 sigma_h^2))           attribution of token j
        v_hj = W_out(GELU(value_j + W_c c))_h                                                    its evidence
        e_hw = (sum_{j in w} a_hj m_j v_hj + a0_h v0_h) / (sum_{j in w} a_hj m_j + a0_h)       window w: one observation
        U(k) = z0 + sum_h s_h sum_{w <= k} e_hw                                                  new windows accumulate
    Within one window the events of one source are strongly correlated, so they form one observation (attributed
    mean); every new window is a new observation, so the windows add up (the sequential test). m_j = 1 for every box
    (evidence="boxes") or the number of events in the box (evidence="events": the event is the observation unit, and
    a learned pseudo-observation a0, v0 = "nothing of mine here" smooths windows without attributed events, a Bayesian
    pseudo-count; boxes use no pseudo-observation). s_h = +1 for target heads, -1 for background heads (value="ei":
    every value is softplus(.) >= 0, half of the heads excite, half inhibit). Context for the next step:
    c = c0 + Delta(attributed mean value, log attributed mass, evidence) (Eq. 6).
    Ablations: pool "mean" (one attributed mean over all windows: waiting cannot accumulate evidence) or "sum" (every
    token independent), value signed, gate distance (receptive field only), context off, audit ledger (each window read
    once, evidence added). Starts exactly at the prior: U(k) = z0 until the output weights move.
    Token form (how a box is read; rounds 1-3 used "mlp"):
        mlp     content + slot -> LayerNorm -> keys / values (d-dim), value head on GELU(value + W_c c)
        linear  the same without LayerNorm: keys / values linear in the features (folded into one 5 x d map + a slot
                table at deployment)
        glm     no embedding: v_hj = (theta_h[age, box] + dtheta_h(c)) . x_j + b_h[age, box] + db_h(c), a linear model
                of the box's normalised count statistics x_j with fixed per-slot coefficients that the context shifts;
                the gate "attribution" reads x_j the same way (gamma_h(c) . x_j); the context reads the attributed mean
                of x per head. A box costs a few multiply-accumulates per head instead of 2 d^2 + d H.
    ctx_ff=False drops the feed-forward block of the context update. In the forms linear / glm the receptive fields
    start at sigma0 * 2^(-1/2) and sigma0 * 2^(1/2) inside every channel (a near and a far reader; target and background
    channels alike, so U = z0 still holds), otherwise the heads of a channel would stay identical under the distance
    gate."""

    def __init__(self, n_steps, r, dim=64, heads=4, value="ei", gate="attribution", pool="window", evidence="boxes",
                 context=True, audit="reaudit", tokens="mlp", ctx_ff=True):
        super(SpikingDecisionUnits, self).__init__()
        if (value not in ("ei", "signed") or gate not in ("attribution", "distance")
                or pool not in ("window", "mean", "sum") or evidence not in ("boxes", "events")
                or audit not in ("reaudit", "ledger") or tokens not in TOKENS):
            raise ValueError("unknown option value=%s gate=%s pool=%s evidence=%s audit=%s tokens=%s"
                             % (value, gate, pool, evidence, audit, tokens))
        if dim % heads or (value == "ei" and heads % 2):
            raise ValueError("dim must be a multiple of heads; value=ei needs an even number of heads")
        r = np.asarray(r, np.float64)
        self.n_steps, self.n_box, self.dim, self.heads, self.hd = int(n_steps), len(r), int(dim), int(heads), \
            int(dim) // int(heads)
        self.value, self.gate, self.pool, self.evidence = value, gate, pool, evidence
        self.context, self.audit, self.tokens, self.ctx_ff = bool(context), audit, tokens, bool(ctx_ff)
        self.register_buffer("r2", torch.as_tensor(r ** 2, dtype=torch.float32))
        sign = [1.0] * heads if value == "signed" else [1.0] * (heads // 2) + [-1.0] * (heads // 2)
        self.register_buffer("head_sign", torch.tensor(sign))
        self.register_buffer("feat_mu", torch.zeros(N_FEAT))
        self.register_buffer("feat_sd", torch.ones(N_FEAT))
        self.register_buffer("z_ref", torch.tensor([0.0, 5.0]))           # prior mean / std of the input scaling
        S, K, attr = self.n_steps, self.n_box, gate == "attribution"
        self.c0 = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        if tokens == "mlp":                                                # module set of the round 1-3 checkpoints
            self.content = nn.Linear(N_FEAT, dim)
            self.slot = nn.Parameter(torch.randn(S, K, dim) * 0.02)
            self.ln_k, self.ln_q, self.ln_f = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.age = nn.Embedding(S, dim)
            self.wq, self.wk, self.wv = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
            self.v_ctx, self.v_out = nn.Linear(dim, dim), nn.Linear(dim, heads)
            self.wo, self.wm = nn.Linear(dim, dim), nn.Linear(2 * heads, dim)
            self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        else:
            if tokens == "linear":
                self.content = nn.Linear(N_FEAT, dim)
                self.slot = nn.Parameter(torch.randn(S, K, dim) * 0.02)
                if attr:
                    self.ln_q, self.age = nn.LayerNorm(dim), nn.Embedding(S, dim)
                    self.wq, self.wk = nn.Linear(dim, dim), nn.Linear(dim, dim)
                self.v_ctx, self.v_out = nn.Linear(dim, dim), nn.Linear(dim, heads)
                self.wo = nn.Linear(dim, dim)
            else:
                self.coef = nn.Parameter(torch.zeros(S, K, heads, N_FEAT))   # per window offset, box and head
                self.bias = nn.Parameter(torch.zeros(S, K, heads))
                self.dcoef = nn.Linear(dim, heads * (N_FEAT + 1))           # the context shifts them
                if attr:
                    self.gq = nn.Linear(dim, heads * N_FEAT)
                self.wo = nn.Linear(heads * N_FEAT, dim)
            self.wm = nn.Linear(2 * heads, dim)
            if self.ctx_ff:
                self.ln_f = nn.LayerNorm(dim)
                self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        sigma0 = float(math.sqrt(max((r ** 2).mean(), 1.0)))              # RMS box distance of the layout
        log_sigma = torch.full((heads,), math.log(sigma0))
        if tokens != "mlp":                                                # near and far reader in every channel
            groups = [heads] if value == "signed" else [heads // 2, heads // 2]
            log_sigma = log_sigma + torch.cat([torch.linspace(-0.5, 0.5, g) * math.log(2.0) if g > 1 else
                                               torch.zeros(1) for g in groups])
        self.log_sigma = nn.Parameter(log_sigma)
        self.rf_bias = nn.Parameter(torch.zeros(heads))
        if evidence == "events":
            self.sink_logit = nn.Parameter(torch.zeros(heads))             # weight of the pseudo-observation
            self.sink_val = nn.Linear(dim, heads)                          # its evidence, given the context
        zero = (self.dcoef,) if tokens == "glm" else (self.v_out,)
        with torch.no_grad():                                              # U = z0 and an unused audit at the start
            for lin in zero + (self.wm,) + ((self.sink_val,) if evidence == "events" else ()):
                lin.weight.zero_()
                lin.bias.zero_()

    # ------------------------------------------------------------------ building blocks
    def set_normalization(self, feat_mu, feat_sd, z_mu, z_sd):
        with torch.no_grad():
            self.feat_mu.copy_(torch.as_tensor(feat_mu, dtype=torch.float32))
            self.feat_sd.copy_(torch.as_tensor(feat_sd, dtype=torch.float32).clamp(min=1e-4))
            self.z_ref.copy_(torch.tensor([float(z_mu), max(float(z_sd), 1e-4)]))

    def sigma(self):
        return torch.exp(self.log_sigma)

    def start(self, z0, pol):
        """Initial context c0 from the event's own prior and polarity."""
        z_in = (z0 - self.z_ref[0]) / self.z_ref[1]
        return self.c0(torch.stack([z_in, pol.float()], 1))

    def embed(self, feat, first=0, slot_idx=None):
        """Tokens of windows first .. first + S - 1 (or, for one window, the per-unit window offset slot_idx [B]):
        feat [B, S, K, N_FEAT] -> dict of [B, S*K, ...]: the event counts "cnt" and, by token form, keys [.., H, hd]
        and values [.., d] (mlp; linear: keys only with the attribution gate), or the normalised features "x" [.., 5]
        and their fixed evidence logit "s" [.., H] (glm)."""
        B, S, K = feat.shape[0], feat.shape[1], feat.shape[2]
        T = S * K
        x = (feat - self.feat_mu) / self.feat_sd
        tok = {"cnt": (torch.expm1(feat[..., 0]) + torch.expm1(feat[..., 1])).reshape(B, T)}
        if self.tokens == "glm":
            coef = self.coef[first:first + S][None] if slot_idx is None else self.coef[slot_idx][:, None]
            bias = self.bias[first:first + S][None] if slot_idx is None else self.bias[slot_idx][:, None]
            tok["x"] = x.reshape(B, T, N_FEAT)
            tok["s"] = ((coef * x[..., None, :]).sum(-1) + bias).reshape(B, T, self.heads)
            return tok
        slot = self.slot[first:first + S][None] if slot_idx is None else self.slot[slot_idx][:, None]
        kt = self.content(x) + slot
        if self.tokens == "mlp":
            kt = self.ln_k(kt).reshape(B, T, self.dim)
            tok["keys"], tok["vals"] = self.wk(kt).view(B, T, self.heads, self.hd), self.wv(kt)
        else:
            tok["vals"] = kt.reshape(B, T, self.dim)
            if self.gate == "attribution":
                tok["keys"] = self.wk(tok["vals"]).view(B, T, self.heads, self.hd)
        return tok

    def audit_tokens(self, c, k, tok, n_win, win_mask=None):
        """One audit over the tokens tok (embed) of n_win windows (Eq. 3-4); k = step (int or per unit), win_mask
        [B, n_win] marks the windows that have arrived (None = all). Returns the evidence per head lambda [B, H] (summed
        over windows), the attributed mass [B, H], the attributed mean value [B, d] (glm: mean features [B, 5 H]) and
        the gates [B, H, T]."""
        cnt = tok["cnt"]
        B, T = cnt.shape
        K, H = self.n_box, self.heads
        prior = self.rf_bias[:, None] - self.r2.repeat(n_win)[None, :] / (2.0 * self.sigma()[:, None] ** 2)
        if self.gate == "attribution" and self.tokens == "glm":
            q = self.gq(c).view(B, H, N_FEAT)
            logit = torch.einsum("bhf,btf->bht", q, tok["x"]) / math.sqrt(N_FEAT) + prior[None]
        elif self.gate == "attribution":
            q = self.wq(self.ln_q(c + self.age.weight[k])).view(B, H, self.hd)
            logit = torch.einsum("bhd,bthd->bht", q, tok["keys"]) / math.sqrt(self.hd) + prior[None]
        else:
            logit = prior[None].expand(B, H, T)
        wm = torch.ones(B, n_win, device=cnt.device) if win_mask is None else win_mask.float()
        a = torch.sigmoid(logit) * wm.repeat_interleave(K, 1)[:, None, :]
        if self.tokens == "glm":
            dc = self.dcoef(c).view(B, H, N_FEAT + 1)
            v = tok["s"].permute(0, 2, 1) + torch.einsum("bhf,btf->bht", dc[..., :N_FEAT], tok["x"]) + dc[..., N_FEAT:]
        else:
            v = self.v_out(F.gelu(tok["vals"] + self.v_ctx(c)[:, None, :])).permute(0, 2, 1)   # [B, H, T]
        if self.value == "ei":
            v = F.softplus(v)
        w = a * cnt[:, None, :] if self.evidence == "events" else a
        num = (w * v).view(B, H, n_win, K).sum(-1)
        den = w.view(B, H, n_win, K).sum(-1)
        if self.evidence == "events":                                      # pseudo-observation of every window
            a0 = torch.sigmoid(self.sink_logit)[None, :, None] * wm[:, None, :]
            v0 = self.sink_val(c)
            if self.value == "ei":
                v0 = F.softplus(v0)
            num = num + a0 * v0[:, :, None]
            den = den + a0
        if self.pool == "window":
            lam = (num / den.clamp(min=EPS) * wm[:, None, :]).sum(-1)
        elif self.pool == "mean":
            lam = num.sum(-1) / den.sum(-1).clamp(min=EPS)
        else:
            lam = num.sum(-1)
        mass = w.sum(-1)
        if self.tokens == "glm":
            ubar = torch.einsum("bht,btf->bhf", w, tok["x"]) / mass[..., None].clamp(min=EPS)
        else:
            ubar = torch.einsum("bht,bthd->bhd", w, tok["vals"].view(B, T, H, self.hd)) / mass[..., None].clamp(min=EPS)
        return lam, mass, ubar.reshape(B, -1), a

    def context_delta(self, ubar, mass, lam):
        """Eq. 6: what the last audit found -> change of the context."""
        dc = self.wo(ubar) + self.wm(torch.cat([torch.log1p(mass), lam], -1))
        return dc + self.ff(self.ln_f(dc)) if self.ctx_ff else dc

    def start_sums(self, B, device):
        """Ledger units: running sums of the windows read so far -- attributed mass [B, H], mass-weighted value (glm:
        features) [B, H, hd or 5] and evidence [B, H]."""
        width = N_FEAT if self.tokens == "glm" else self.hd
        return (torch.zeros(B, self.heads, device=device), torch.zeros(B, self.heads, width, device=device),
                torch.zeros(B, self.heads, device=device))

    def step(self, c, c0, U, z0, k, tok, n_win, m, win_mask=None, sums=None):
        """Step k for units whose tokens so far are tok (n_win windows): new membrane and context (Eq. 3-6); units
        with m = False keep their state. Re-audit: U = z0 + evidence of all windows read now, c = c0 + Delta(this
        audit). Ledger (tok = the new window only, sums = start_sums at the first step): U = U + evidence of the new
        window, and c = c0 + Delta(running sums) -- the same context as re-audit computes from one audit of the same
        windows (attributed mean, total mass, total evidence), only the old windows are not read again. Returns U, c,
        lambda, mass, gates, sums (None for re-audit)."""
        lam, mass, ubar, a = self.audit_tokens(c, k, tok, n_win, win_mask)
        ev = (lam * self.head_sign).sum(-1)
        if self.audit == "reaudit":
            U = torch.where(m, z0 + ev, U)
            if self.context:
                c = torch.where(m[:, None], c0 + self.context_delta(ubar, mass, lam), c)
            return U, c, lam, mass, a, None
        U = torch.where(m, U + ev, U)
        B, H = mass.shape
        new = (sums[0] + mass, sums[1] + ubar.view(B, H, -1) * mass[..., None], sums[2] + lam)
        sums = tuple(torch.where(m.view((B,) + (1,) * (n.dim() - 1)), n, o) for n, o in zip(new, sums))
        if self.context:
            M, X, L = sums
            mean = (X / M[..., None].clamp(min=EPS)).reshape(B, -1)
            c = torch.where(m[:, None], c0 + self.context_delta(mean, M, L), c)
        return U, c, lam, mass, a, sums

    # ------------------------------------------------------------------ whole trajectories
    def forward(self, feat, valid, z0, pol, keep=False):
        """feat [B, S, K, N_FEAT] (windows 0 .. S-1 after the event), valid [B, S] (steps that exist), z0 [B] (calibrated
        prior), pol [B] -> membrane U [B, S]; U(k) is frozen after the last valid step. keep=True also stores the
        signed evidence per head and step, the attributed mass and the gates of every step in self.extras."""
        B, S = feat.shape[0], feat.shape[1]
        K = self.n_box
        tok = self.embed(feat)
        c0 = self.start(z0, pol)
        c, U = c0, z0
        sums = self.start_sums(B, feat.device) if self.audit == "ledger" else None
        outs, lam_all, mass_all, gates = [], [], [], []
        for k in range(S):
            if self.audit == "reaudit":
                sl, n_win = slice(0, (k + 1) * K), k + 1
            else:
                sl, n_win = slice(k * K, (k + 1) * K), 1
            U, c, lam, mass, a, sums = self.step(c, c0, U, z0, k, {n: t[:, sl] for n, t in tok.items()}, n_win,
                                                 valid[:, k], sums=sums)
            outs.append(U)
            if keep:
                lam_all.append(lam * self.head_sign)
                mass_all.append(mass)
                gates.append(F.pad(a, (sl.start, S * K - sl.stop)))
        if keep:
            self.extras = {"lam": torch.stack(lam_all, 1), "mass": torch.stack(mass_all, 1),
                           "gates": torch.stack(gates, 1)}
        return torch.stack(outs, 1)

    # ------------------------------------------------------------------ cost and interpretation
    def step_macs(self, k, nz_new=None, nz_aud=None):
        """Multiply-accumulates of step k of one unit: the boxes of the new window read once (embedding or fixed logit),
        the audit of the boxes of every window so far (re-audit; ledger: the new window), the context. nz_new / nz_aud =
        boxes with events among them (arrays broadcast with k); None = every box. With evidence="events" a box without
        events has weight 0 everywhere (membrane, mass, context), so it is skipped exactly; with evidence="boxes" every
        box counts. Folded linear maps count once (linear: W_k W_c is one 5 x d map)."""
        d, H, K, Fn = self.dim, self.heads, self.n_box, N_FEAT
        attr = self.gate == "attribution"
        k = np.asarray(k)
        dense_aud = (k + 1) * K if self.audit == "reaudit" else np.full(k.shape, K)
        if nz_new is None or self.evidence == "boxes":
            nz_new, nz_aud = np.full(k.shape, K), dense_aud
        if self.tokens == "glm":
            new = Fn + Fn * H                                             # normalise + fixed logit
            aud = Fn * H + H + Fn * H + 3 * H + (Fn * H if attr else 0)   # shift, mean feature, sums (+ gate)
            fixed = d * H * (Fn + 1) + (d * H * Fn if attr else 0)        # coefficient shift (+ gate query)
            ctx = H * Fn * d + 2 * H * d
        else:
            new = Fn * d + (d * d if self.tokens == "mlp" else 0)         # values
            if attr:
                new += d * d if self.tokens == "mlp" else Fn * d          # keys
            aud = (d if attr else 0) + d * H + d + 3 * H                  # q.k, value head, mean value, sums
            fixed = d * d * (2 if attr else 1)                            # query + context term of the values
            ctx = d * d + 2 * H * d
        if self.context:
            fixed += ctx + (4 * d * d if self.ctx_ff else 0)              # context update
            if self.audit == "ledger":
                fixed += 2 * H * (Fn if self.tokens == "glm" else self.hd)   # running sums and their mean
        if self.evidence == "events":
            fixed += d * H                                                # pseudo-observation value
        total = nz_new * new + nz_aud * aud + fixed + (k == 0) * (2 * d + d * d)   # c0 at the first step
        return int(total) if np.ndim(total) == 0 else np.asarray(total, np.float64)

    @staticmethod
    def step_acs(ne):
        """Accumulates of the box sums of one unit-step in an event-driven implementation: every event inside the
        unit's square adds 1 to the count and its score to q of its box (ne = events in the square, the unit's own
        event excluded). The GPU code gets the same integer sums from integral images."""
        return 2.0 * np.asarray(ne, np.float64)

    def learned_constants(self):
        out = {"sigma_px": [round(v, 2) for v in self.sigma().detach().cpu().tolist()],
               "rf_bias": [round(v, 3) for v in self.rf_bias.detach().cpu().tolist()],
               "head_sign": self.head_sign.cpu().tolist()}
        if self.evidence == "events":
            out["pseudo_observation_weight"] = [round(v, 3) for v in
                                                torch.sigmoid(self.sink_logit).detach().cpu().tolist()]
        return out


class StreamingDecisionLayer(object):
    """Deployment form of the decision layer, one window at a time.

    Window t: its events' evidence maps are built once; the new events pass the entry rule (column 0 of Eq. 7: commit
    the prior when it is already safe) and the others join the open units; every open unit reads its boxes of window t
    and all units take ONE batched step; a unit fires when its membrane reaches the bound of its step or at its last
    step (deadline or end of the recording). Re-audit units keep the tokens of all their windows (n_steps windows,
    windows still to come are masked) and re-read them; ledger units keep only their membrane and context and read the
    new window once. Returns the committed events with their column (0 = entry, 1 + k = fired at step k) and value."""

    def __init__(self, net, rect, bounds, tau, height, width, device):
        self.net, self.rect = net.eval(), rect
        self.device = torch.device(device)
        self.bounds = torch.tensor([float(b) for b in bounds], device=self.device)
        self.tau, self.H, self.W = float(tau), int(height), int(width)
        self.reset()

    def reset(self):
        self.st, self.meta = None, None

    def open_units(self):
        return 0 if self.st is None else int(self.st["U"].shape[0])

    def _write(self, st, rows, tok):
        """Tokens of one window (embed with S = 1) into the buffers tok_<name> at the units' window offsets."""
        K = self.net.n_box
        pos = st["age"][rows][:, None] * K + torch.arange(K, device=self.device)[None]
        r = rows[:, None]
        for name, t in tok.items():
            st["tok_" + name][r, pos] = t.reshape((len(rows), K) + tuple(t.shape[2:]))

    @staticmethod
    def _tokens(st):
        return {n[4:]: v for n, v in st.items() if n.startswith("tok_")}

    @torch.no_grad()
    def window(self, t, x, y, pol, z0, ids, last_step):
        net, dev = self.net, self.device
        S, K = net.n_steps, net.n_box
        ledger = net.audit == "ledger"
        x, y, pol, ids = (np.asarray(a, np.int64) for a in (x, y, pol, ids))
        z0 = np.asarray(z0, np.float32)
        last_step = np.asarray(last_step, np.int64)
        maps = EvidenceMaps(x, y, np.zeros(len(x), np.int64), pol, z0, 1, self.H, self.W, dev)
        out = ([], [], [])
        z0_t = torch.as_tensor(z0, device=dev)
        entry = (torch.abs(z0_t - self.tau) >= self.bounds[0]).cpu().numpy()
        if entry.any():                                            # safe priors commit at arrival
            out[0].append(ids[entry])
            out[1].append(np.zeros(int(entry.sum()), np.int64))
            out[2].append(z0[entry])
        tok_now = None                                             # ledger: tokens of window t, open units first
        if self.st is not None:                                    # open units read window t
            M = self.open_units()
            f, _ = maps.features(self.meta["x"], self.meta["y"], np.zeros(M, np.int64), self.meta["pol"],
                                 np.zeros(M, np.float32), self.rect, 1, own_window=False)    # z0 unused: not own window
            tok_now = net.embed(f, slot_idx=self.st["age"])
            if not ledger:
                self._write(self.st, torch.arange(M, device=dev), tok_now)
        new = ~entry
        if new.any():                                              # the others start a unit (step 0 at once)
            n = int(new.sum())
            nt = torch.as_tensor(new, device=dev)
            zt, pt = z0_t[nt], torch.as_tensor(pol[new], device=dev)
            c0 = net.start(zt, pt)
            nb = {"age": torch.zeros(n, dtype=torch.long, device=dev), "z0": zt, "c0": c0, "c": c0, "U": zt,
                  "last": torch.as_tensor(last_step[new], device=dev)}
            f, _ = maps.features(x[new], y[new], np.zeros(n, np.int64), pol[new], z0[new], self.rect, 1)
            tok = net.embed(f, slot_idx=nb["age"])
            if ledger:
                tok_now = tok if tok_now is None else {n_: torch.cat([tok_now[n_], tok[n_]]) for n_ in tok}
                nb["sum_mass"], nb["sum_value"], nb["sum_ev"] = net.start_sums(n, dev)
            else:
                for name, t_ in tok.items():                       # token buffers of all n_steps windows
                    nb["tok_" + name] = torch.zeros((n, S * K) + tuple(t_.shape[2:]), device=dev)
                self._write(nb, torch.arange(n, device=dev), tok)
            meta = {"ids": ids[new], "x": x[new], "y": y[new], "pol": pol[new]}
            if self.st is None:
                self.st, self.meta = nb, meta
            else:
                self.st = {k: torch.cat([self.st[k], nb[k]]) for k in self.st}
                self.meta = {k: np.concatenate([self.meta[k], meta[k]]) for k in self.meta}
        if self.st is not None:                                    # one batched step for every open unit
            st = self.st
            M = self.open_units()
            on = torch.ones(M, dtype=torch.bool, device=dev)
            if ledger:
                U, c, _, _, _, sums = net.step(st["c"], st["c0"], st["U"], st["z0"], st["age"], tok_now, 1, on,
                                               sums=(st["sum_mass"], st["sum_value"], st["sum_ev"]))
                st["sum_mass"], st["sum_value"], st["sum_ev"] = sums
            else:
                win_mask = torch.arange(S, device=dev)[None, :] <= st["age"][:, None]
                U, c, _, _, _, _ = net.step(st["c"], st["c0"], st["U"], st["z0"], st["age"], self._tokens(st), S,
                                            on, win_mask)
            fire = (torch.abs(U - self.tau) >= self.bounds[1 + st["age"]]) | (st["last"] == st["age"])
            f = fire.cpu().numpy()
            if f.any():
                out[0].append(self.meta["ids"][f])
                out[1].append((1 + st["age"][fire]).cpu().numpy())
                out[2].append(U[fire].cpu().numpy())
            if f.all():
                self.reset()
            else:
                keep = ~fire
                st["U"], st["c"] = U, c
                self.st = {k: v[keep] for k, v in st.items()}
                self.st["age"] = self.st["age"] + 1
                self.meta = {k: v[~f] for k, v in self.meta.items()}
        cat = lambda parts, dt: np.concatenate(parts) if parts else np.zeros(0, dt)    # noqa: E731
        return cat(out[0], np.int64), cat(out[1], np.int64), cat(out[2], np.float32)
