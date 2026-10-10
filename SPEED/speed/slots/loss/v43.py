"""Slot 9 V4-3 variant: train every new part by what inference does with it.

    base        mark BCE + current-step intensity Poisson (the base loss; skipped when the trunk and base heads are
                frozen, e.g. when they are loaded from a base checkpoint)
    background  sum_px [ mu - n0 log mu ] for step k+1 from log_mu of step k (n0 = background events): H0 is a
                calibrated forecast of the background
    motion      soft cross-entropy of the anchor distribution against the label velocity (shared between the two
                nearest speed rings and directions) + smooth-L1 of the residual of the dominant anchor (MultiPath)
    growth      - log sum_a p_a exp(E_a) for target events: the one-step tube evidence of every anchor, mixed by the
                predicted probabilities. Every anchor is judged by the next step's events, however far it is from the
                truth (a non-local, label-free signal; the GROW objective of the verifier's own test)
    evidence    mean over d = 1..D of BCE(m + w_d F_d, y) with the verifier's own hypotheses and functions (m detached)
    stability   mean over d = 0..D-1 of BCE(q_d, 1[(z_d >= theta) == (z_D >= theta)]) on events with all D steps in the
                chunk (inputs detached)
Intensity g and background mu enter the evidence detached: they keep their likelihood meaning; the evidence and growth
terms train the motion probabilities, the position weights and the fusion weights.
"""
import numpy as np
import torch
import torch.nn.functional as F

from speed.slots.loss.mark_intensity import MarkIntensityLoss
from speed.slots.verify.tube_evidence import presence_support, score, tube_step


class V43Loss(object):
    needs_velocity = True
    needs_aux = True

    def __init__(self, dt_ms, verifier, theta, train_base=True, mark_weight=1.0, intensity_weight=1.0,
                 intensity_smooth=3, background_weight=1.0, motion_weight=1.0, residual_weight=1.0, growth_weight=1.0,
                 evidence_weight=1.0, stability_weight=1.0, max_queries=16384):
        self.dt, self.verifier, self.theta = float(dt_ms), verifier, float(theta)
        self.motion, self.head = verifier.motion, verifier.head
        self.base = MarkIntensityLoss(mark_weight, intensity_weight, intensity_smooth) if train_base else None
        self.weights = {"background": float(background_weight), "motion": float(motion_weight),
                        "residual": float(residual_weight), "growth": float(growth_weight),
                        "evidence": float(evidence_weight), "stability": float(stability_weight)}
        if min(self.weights.values()) < 0:
            raise ValueError("loss weights must be non-negative")
        self.max_queries = int(max_queries)

    @staticmethod
    def background_counts(blk):
        src, ev = blk["source"], blk["events"]
        steps = blk["n_steps"]
        return src.accumulate(ev["t"] * src.plane + ev["pixel"], 1.0 - blk["labels"], steps * src.plane).view(
            steps, 1, 1, src.height, src.width)

    def queries(self, blk, rng):
        """Event indices for the motion and evidence terms: all events up to max_queries (targets first)."""
        n = int(blk["labels"].shape[0])
        if n <= self.max_queries:
            return torch.arange(n, device=blk["labels"].device)
        lab = blk["labels"].cpu().numpy() > 0
        tgt, bg = np.flatnonzero(lab), np.flatnonzero(~lab)
        rng = rng if rng is not None else np.random.RandomState(0)
        keep_t = tgt if tgt.size <= self.max_queries // 2 else rng.choice(tgt, self.max_queries // 2, replace=False)
        keep_b = rng.choice(bg, min(bg.size, self.max_queries - keep_t.size), replace=False)
        return torch.from_numpy(np.sort(np.r_[keep_t, keep_b])).to(blk["labels"].device)

    def prepare(self, outputs, logits, blk):
        """Chunk maps (row m = step m) and the anchor distribution at the queried events."""
        T, B = int(outputs["log_g"].shape[0]), int(outputs["log_g"].shape[1])
        rows = lambda t: t.reshape((-1,) + tuple(t.shape[2:]))  # noqa: E731
        aux, ev = blk["aux"], blk["events"]
        phi = outputs["phi"]
        prev = blk.get("prev_phi")
        prev = torch.zeros_like(phi[0]) if prev is None else prev.to(phi.dtype)
        phi_prev = torch.cat([prev.unsqueeze(0), phi[:-1]], 0)
        q = self.queries(blk, blk.get("rng"))
        t, b = ev["t"][q], ev["b"][q]
        logit_a, residual = self.motion.query(self.motion.pyramid(rows(phi)), self.motion.pyramid(rows(phi_prev)),
                                              t * B + b, ev["y"][q], ev["x"][q])
        counts, mu0 = rows(aux["total"]), rows(aux["mu0"])
        log_mu = outputs.get("log_mu")
        if self.verifier.background == "head" and log_mu is not None:
            rate = torch.cat([mu0[:B], self.verifier.mu_floor + torch.exp(rows(log_mu[:-1]))], 0).detach()
        else:
            rate = mu0.detach()
        return {"T": T, "B": B, "q": q, "t": t, "b": b, "y": ev["y"][q], "x": ev["x"][q], "label": blk["labels"][q],
                "mark": logits.detach()[q], "logits": logit_a, "residual": residual, "counts": counts, "rate": rate,
                "g": torch.exp(rows(outputs["log_g"])).detach(),
                "support": presence_support(counts) if self.verifier.anchor else None}

    def chains(self, p):
        """The verifier's own chain for the queried events with a future step -> (selected rows, logw,
        [(F_d [n], valid_d [n]) for d = 1..])."""
        T, B = p["T"], p["B"]
        sel = (p["t"] <= T - 2).nonzero().view(-1)
        if int(sel.numel()) == 0:
            return sel, None, []
        disp, logw, _ = self.motion.hypotheses(p["logits"][sel], p["residual"][sel], self.verifier.n_hypotheses)
        ts, bs, ys, xs = p["t"][sel], p["b"][sel], p["y"][sel], p["x"][sel]
        run = p["g"].new_zeros(int(disp.shape[0]), int(sel.numel()))
        alive = torch.ones_like(run, dtype=torch.bool)
        log_pi = self.head.position_log_weights()
        out = []
        for d in range(1, self.head.max_delay + 1):
            valid = ts + d <= T - 1
            if not bool(valid.any()):
                break
            f_now = (ts + d).clamp(max=T - 1) * B + bs
            f_prev = (ts + d - 1).clamp(max=T - 2) * B + bs
            run, alive = tube_step(p["counts"], f_now, p["rate"], f_now, p["g"], f_prev, p["support"], ys, xs, disp, d,
                                   run, alive, log_pi)
            out.append((score(run, logw), valid))
        return sel, logw, out

    def __call__(self, outputs, logits, blk):
        terms = []
        parts = {k: 0.0 for k in ("mark", "intensity", "background", "motion", "residual", "growth", "evidence",
                                  "stability")}

        def add(name, value):
            terms.append(self.weights.get(name, 1.0) * value)
            parts[name] = float(value.detach())

        if self.base is not None:
            total, p = self.base(outputs, logits, blk)
            if total is not None:
                terms.append(total)
            parts.update(p)
        T = int(outputs["log_g"].shape[0])
        if T > 1 and self.weights["background"] > 0:
            mu = self.verifier.mu_floor + torch.exp(outputs["log_mu"][:-1])
            n0 = self.background_counts(blk)[1:]
            add("background", (mu - n0 * torch.log(mu)).sum())
        if blk["labels"].numel() == 0:
            return self._total(terms), parts
        p = self.prepare(outputs, logits, blk)
        logp = F.log_softmax(p["logits"], 1)
        lab = p["label"]
        vel = blk["events"]["vel"][p["q"]]
        known = (lab > 0) & torch.isfinite(vel).all(1)
        if bool(known.any()) and (self.weights["motion"] > 0 or self.weights["residual"] > 0):
            soft, dom, res_t = self.motion.table.soft_target((vel[known] * self.dt).to(logp.dtype))
            add("motion", -(soft * logp[known]).sum())
            moving = dom > 0
            if bool(moving.any()):
                kn = known.nonzero().view(-1)[moving]
                add("residual", F.smooth_l1_loss(p["residual"][kn, dom[moving]], res_t[moving], reduction="sum"))
        if T < 2:
            return self._total(terms), parts
        B = p["B"]
        tgt_future = (p["t"] <= T - 2) & (lab > 0)
        if self.weights["growth"] > 0 and bool(tgt_future.any()):
            sel = tgt_future.nonzero().view(-1)
            A = self.motion.size
            disp = self.motion.disp.to(p["g"].dtype).view(A, 1, 2).expand(A, int(sel.numel()), 2)
            f_now = (p["t"][sel] + 1) * B + p["b"][sel]
            run = p["g"].new_zeros(A, int(sel.numel()))
            E, _ = tube_step(p["counts"], f_now, p["rate"], f_now, p["g"], p["t"][sel] * B + p["b"][sel], None,
                             p["y"][sel], p["x"][sel], disp, 1, run, torch.ones_like(run, dtype=torch.bool),
                             self.head.position_log_weights().detach())
            add("growth", -torch.logsumexp(logp[sel].t() + E, 0).sum())
        if self.weights["evidence"] > 0 or self.weights["stability"] > 0:
            sel, _, chains = self.chains(p)
            ms, ls, D = p["mark"][sel], lab[sel], self.head.max_delay
            if self.weights["evidence"] > 0 and chains:
                le = 0.0
                for d, (Fd, valid) in enumerate(chains, 1):
                    z = ms[valid] + self.head.fusion_weight(d).to(ms.dtype) * Fd[valid].to(ms.dtype)
                    le = le + F.binary_cross_entropy_with_logits(z, ls[valid], reduction="sum")
                add("evidence", le / D)
            full = p["t"][sel] + D <= T - 1
            if self.weights["stability"] > 0 and len(chains) == D and bool(full.any()):
                with torch.no_grad():
                    Fs = [torch.zeros_like(ms[full])] + [Fd[full].to(ms.dtype) for Fd, _ in chains]
                    zs = [ms[full]] + [ms[full] + self.head.fusion_weight(d).to(ms.dtype) * Fs[d] for d in range(1, D + 1)]
                    final = zs[D] >= self.theta
                ls_ = 0.0
                for d in range(D):
                    target = ((zs[d] >= self.theta) == final).to(ms.dtype)
                    qd = self.head.stability_logit(zs[d] - self.theta, Fs[d], torch.full_like(zs[d], float(d) / D))
                    ls_ = ls_ + F.binary_cross_entropy_with_logits(qd, target.to(qd.dtype), reduction="sum")
                add("stability", ls_ / D)
        return self._total(terms), parts

    @staticmethod
    def _total(terms):
        if not terms:
            return None
        out = terms[0]
        for term in terms[1:]:
            out = out + term
        return out
