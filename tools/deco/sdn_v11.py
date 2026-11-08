"""V11 spiking decision neurons (11-08): every pending event is a sequential test with a deadline.

Membrane = posterior log-odds of "the event belongs to a target":
    U(k) = z0 + sum_h lambda_h(k)
z0 is the prior of the clock-driven BIR network; lambda_h(k) is the evidence of every candidate event that has arrived
by step k (the past at step 0, one more window per step), each event entering the sum exactly once.

Evidence of event j for head h (attribution mixture, first order in the attribution probability):  a_hj * v_hj
    a_hj = sigmoid(q_h(c) . key_j / sqrt(d) + b_h - |dx, dy|^2 / (2 sigma_h^2))   P(j and i share a generating process)
    v_hj = MLP(token_j, c)                                                          signed evidence of one related event
The gates are independent probabilities (no softmax): several events can be related at once, and the attributed mass
M_h = sum_j a_hj says how much evidence there is (no empty key needed).

Redundancy: events of one source are correlated, so m of them are not m independent observations. With m_hw the
attributed mass of window w and M_h = sum_w m_hw, the equicorrelated-Gaussian model gives
    lambda_h = S_h * M_h / (sum_w m_hw (1 + (m_hw - 1) rho_w) + rho_t (M_h^2 - sum_w m_hw^2)),   S_h = sum_j a_hj v_hj
             = (effective number of independent observations) x (attributed mean evidence).
rho_w = correlation inside a window, rho_t = rho_w * sigmoid(.) <= rho_w across windows (keeps the implied covariance
positive definite). rho = 0 gives the plain sum, rho = 1 the attributed mean.

Audit: by default every arrived event is re-audited at every step with the current context c (its entry may be revised,
it is never added twice); the context is c = c0 + summary(audit). audit="ledger" instead adds only the newly arrived
window and freezes earlier entries (pure integrator, context accumulates).
Ablations: rho="sum" / "mean" fix rho; gate="softmax" replaces the gates by a softmax over the arrived events (weighted
mean of the values, the attention-style aggregation without evidence mass).
Training: stepwise BCE on U(k) (tools/deco/train_dnm.py --design v11), so every U(k) estimates the posterior and
U(k) - U(k - 1) the conditional likelihood ratio of what arrived. Commit timing (two thresholds from the neuron's own
predicted future range) is fitted afterwards on held-out trajectories by tools/deco/fit_range.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEAT = 6
EPS = 1e-6


def redundancy_evidence(S, M, mw, rho_w, rho_t):
    """lambda = S * M / den with den = sum_w m_w (1 + (m_w - 1) rho_w) + rho_t (M^2 - sum_w m_w^2).
    S, M: [..., H]; mw: [..., H, W]; rho_w, rho_t: [H] (broadcast)."""
    den = (mw * (1.0 + (mw - 1.0) * rho_w[..., None])).sum(-1) + rho_t * (M * M - (mw * mw).sum(-1))
    return S * M / (den + EPS), M * M / (den + EPS)


class SpikingDecisionNeuronsV11(nn.Module):
    def __init__(self, n_off, K, sigma0, dim=64, heads=4, dmax=40, audit="reaudit", rho="learn", gate="sigmoid",
                 rho_init=(0.5, 0.12), ctx_mass=True, n_extra=0):
        super(SpikingDecisionNeuronsV11, self).__init__()
        if audit not in ("reaudit", "ledger") or rho not in ("learn", "sum", "mean") or gate not in ("sigmoid", "softmax"):
            raise ValueError("unknown V11 option: audit=%s rho=%s gate=%s" % (audit, rho, gate))
        self.dim, self.heads, self.hd, self.dmax = dim, heads, dim // heads, dmax
        self.n_off, self.K = n_off, K
        self.audit, self.rho_mode, self.gate = audit, rho, gate
        self.ctx_mass = bool(ctx_mass)      # False: the context (and so the values) never sees the attributed mass,
                                            # so only rho decides how evidence grows with the number of related events
        self.c0 = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.n_extra = int(n_extra)          # extra token channels (the small network's event reports, nested V11)
        self.tok = nn.Sequential(nn.Linear(N_FEAT + self.n_extra, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ln_k, self.ln_q, self.ln_f = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.age = nn.Embedding(dmax + 1, dim)
        self.wq, self.wk = nn.Linear(dim, dim), nn.Linear(dim, dim)          # attribution
        self.log_sigma = nn.Parameter(torch.full((heads,), math.log(sigma0)))
        self.rf_bias = nn.Parameter(torch.zeros(heads))
        self.v_tok, self.v_ctx = nn.Linear(dim, dim), nn.Linear(dim, dim)    # evidence value
        self.v_out = nn.Linear(dim, heads)
        self.we, self.wo = nn.Linear(dim, dim), nn.Linear(dim, dim)          # context summary
        self.wm = nn.Linear(2 * heads, dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        w0, t0 = float(rho_init[0]), float(rho_init[1])                     # rho_w and rho_t / rho_w at the start
        self.r_w = nn.Parameter(torch.full((heads,), math.log(w0 / (1.0 - w0))))   # rho_w = sigmoid(r_w)
        self.r_t = nn.Parameter(torch.full((heads,), math.log(t0 / (1.0 - t0))))   # rho_t = rho_w * sigmoid(r_t)
        self.register_buffer("off_idx", torch.arange(n_off).repeat_interleave(K))
        with torch.no_grad():                    # U starts exactly at z0; the summary of the mass starts unused
            self.v_out.weight.zero_()
            self.v_out.bias.zero_()
            self.wm.weight.zero_()
            self.wm.bias.zero_()
        self.keep_extras = False
        self.extras = None

    def rho(self):
        if self.rho_mode == "sum":
            z = torch.zeros(self.heads, device=self.r_w.device)
            return z, z
        if self.rho_mode == "mean":
            o = torch.ones(self.heads, device=self.r_w.device)
            return o, o
        rho_w = torch.sigmoid(self.r_w)
        return rho_w, rho_w * torch.sigmoid(self.r_t)

    def sigma(self):
        return torch.exp(self.log_sigma)

    def forward(self, feat, ok, dw, valid, z_in, z0, p, D, dist2):
        if D > self.dmax:
            raise ValueError("D = %d exceeds the age table (%d)" % (D, self.dmax))
        B, H, hd = z0.shape[0], self.heads, self.hd
        T = feat.shape[1]
        c0 = self.c0(torch.stack([z_in, p], 1))
        c = c0
        kt = self.ln_k(self.tok(feat))                                       # [B, T, dim]
        Kt = self.wk(kt).view(B, T, H, hd)
        Et = self.we(kt).view(B, T, H, hd)
        Vt = self.v_tok(kt)
        prior = self.rf_bias[None, :, None] - dist2[:, None, :] / (2.0 * self.sigma()[None, :, None] ** 2)
        rho_w, rho_t = self.rho()
        U = z0.clone()
        outs, mass, neff, lam_heads = [], [], [], []
        for d in range(D + 1):
            if self.audit == "reaudit":
                sel = ok & (dw <= d + 0.5)
            elif d == 0:
                sel = ok & (dw <= 0.5)
            else:
                sel = ok & ((dw - d).abs() < 0.5)
            q = self.wq(self.ln_q(c + self.age.weight[d])).view(B, H, hd)
            logit = torch.einsum("bhd,bthd->bht", q, Kt) / math.sqrt(hd) + prior          # [B, H, T]
            selh = sel[:, None, :]
            if self.gate == "sigmoid":
                a = torch.sigmoid(logit) * selh.float()
            else:
                L = logit.masked_fill(~selh, float("-inf"))
                mx = L.max(-1, keepdim=True)[0]
                mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
                e = torch.exp(L - mx) * selh.float()
                a = e / e.sum(-1, keepdim=True).clamp(min=1e-12)
            v = self.v_out(F.gelu(Vt + self.v_ctx(c)[:, None, :])).permute(0, 2, 1)       # [B, H, T]
            S = (a * v).sum(-1)                                                             # [B, H]
            M = a.sum(-1)
            if self.gate == "softmax":
                lam_h, ne = S, M
            else:
                mw = a.view(B, H, self.n_off, self.K).sum(-1)                               # [B, H, n_off]
                lam_h, ne = redundancy_evidence(S, M, mw, rho_w, rho_t)
            lam = lam_h.sum(-1)
            m = valid[:, d]
            if self.audit == "reaudit":
                U = torch.where(m, z0 + lam, U)
            else:
                U = U + m.float() * lam
            ubar = torch.einsum("bht,bthd->bhd", a, Et) / (M[..., None] + EPS)
            mass_in = torch.log1p(M) if self.ctx_mass else torch.zeros_like(M)
            dc = self.wo(ubar.reshape(B, self.dim)) + self.wm(torch.cat([mass_in, lam_h], -1))
            dc = dc + self.ff(self.ln_f(dc))
            if self.audit == "reaudit":
                c = torch.where(m[:, None], c0 + dc, c)
            else:
                c = c + m.float()[:, None] * dc
            outs.append(U)
            if self.keep_extras:
                last_a = torch.where(m[:, None, None], a, last_a) if d else a      # gates at the last valid step
                mass.append(M.sum(-1))
                neff.append(ne.sum(-1))
                lam_heads.append(lam_h)
        if self.keep_extras:
            self.extras = {"mass": torch.stack(mass, 1), "neff": torch.stack(neff, 1),
                           "lam_heads": torch.stack(lam_heads, 1), "gates": last_a}
        return torch.stack(outs, 1)

    def load_stage1(self, state):
        """Load weights of a staged V11 run (6 token features); extra report channels start with zero weights, so the
        neuron computes exactly what the staged run computed until the reports become non-zero."""
        state = dict(state)
        w = state["tok.0.weight"]
        if w.shape[1] != self.tok[0].weight.shape[1]:
            pad = torch.zeros(w.shape[0], self.tok[0].weight.shape[1] - w.shape[1], dtype=w.dtype, device=w.device)
            state["tok.0.weight"] = torch.cat([w, pad], 1)
        self.load_state_dict(state)

    def learned_constants(self):
        rho_w, rho_t = self.rho()
        return {"sigma_px": self.sigma().detach().cpu().tolist(), "rf_bias": self.rf_bias.detach().cpu().tolist(),
                "rho_w": rho_w.detach().cpu().tolist(), "rho_t": rho_t.detach().cpu().tolist()}
