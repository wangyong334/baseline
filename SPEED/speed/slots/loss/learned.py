"""Slot 9 V4-2 variant: train every learned part of the verifier and the readouts by what inference does with it.

    mark, forecast, motion   as MarkForecastMotionLoss (V4-1)
    background  sum_px [ mu - n0 log mu ],  mu = mu_floor + exp(log_mu) of step k, n0 = background (label 0) events
                of step k+1: the learned H0 of the verifier is a calibrated forecast of the background
    evidence    mean over d = 1..D of sum_e BCE(m_e + w_d F_d(e), y_e) on the events with d further steps in the chunk;
                F_d by the verifier's own functions (forecast, likelihood ratio, position mixture, anchored chain,
                hypothesis mixture); m and mu are detached: the zero-wait mark keeps its own objective and H0 its
                meaning, the evidence path (w_d, pi, g, motion) learns what the mark misses
    stability   mean over d = 0..D-1 of sum_e BCE(q_d(e), 1[(z_d >= theta) == (z_D >= theta)]) on the events with all D
                further steps in the chunk; inputs detached (the publish rule only learns when waiting stops mattering)
"""
import torch
import torch.nn.functional as F

from speed.slots.loss.motion import MarkForecastMotionLoss
from speed.slots.verify.learned_evidence import chain_step, score


class LearnedEvidenceLoss(MarkForecastMotionLoss):
    needs_velocity = True
    needs_aux = True

    def __init__(self, dt_ms, verifier, theta, mark_weight=1.0, forecast_weight=1.0, motion_weight=1.0,
                 background_weight=1.0, evidence_weight=1.0, stability_weight=1.0, current_weight=0.0,
                 intensity_smooth=3, forecast_floor=2e-4):
        super(LearnedEvidenceLoss, self).__init__(dt_ms, mark_weight, forecast_weight, motion_weight, current_weight,
                                                  intensity_smooth, forecast_floor)
        self.verifier, self.head, self.theta = verifier, verifier.head, float(theta)
        self.background_weight, self.evidence_weight = float(background_weight), float(evidence_weight)
        self.stability_weight = float(stability_weight)
        if min(self.background_weight, self.evidence_weight, self.stability_weight) < 0:
            raise ValueError("loss weights must be non-negative")

    @staticmethod
    def background_counts(blk):
        src, ev = blk["source"], blk["events"]
        steps = blk["n_steps"]
        return src.accumulate(ev["t"] * src.plane + ev["pixel"], 1.0 - blk["labels"], steps * src.plane).view(
            steps, 1, 1, src.height, src.width)

    def chains(self, outputs, aux, ev, steps):
        """F_d [n] for d = 1..D (only meaningful where valid_d) of every event of the chunk."""
        H, W = (int(n) for n in outputs["log_g"].shape[-2:])
        B = int(outputs["log_g"].shape[1])
        rows = lambda t, c: t.reshape(-1, c, H, W)  # noqa: E731
        counts = rows(aux["total"][1:], 1)
        log_mu = outputs.get("log_mu")
        mu = self.verifier.background_rate(None if log_mu is None else rows(log_mu[:-1], 1), rows(aux["mu0"][1:], 1))
        E, _, _ = self.verifier.maps(counts, mu.detach(), rows(outputs["log_g"][:-1], 1), rows(outputs["motion"][:-1], 3))
        support = self.verifier.support(counts)
        t, b, y, x = ev["t"], ev["b"], ev["y"], ev["x"]
        pts, logw = self.verifier.event_cloud(rows(outputs["motion"], 3), t * B + b, y, x)
        run = E.new_zeros(int(logw.shape[0]), int(y.shape[0]))
        alive = torch.ones_like(run, dtype=torch.bool)
        out = []
        for d in range(1, self.head.max_delay + 1):
            valid = t + d <= steps - 1
            if not bool(valid.any()):
                break
            frame = (t + d - 1).clamp(max=steps - 2) * B + b              # E row of step t + d
            run, alive = chain_step(E, support, frame, y, x, pts, d, run, alive)
            out.append((score(run, logw), valid))
        return out

    def __call__(self, outputs, logits, blk):
        total, parts = super(LearnedEvidenceLoss, self).__call__(outputs, logits, blk)
        terms = [] if total is None else [total]
        parts.update(background=0.0, evidence=0.0, stability=0.0)
        steps = int(blk["n_steps"])
        if steps > 1 and self.background_weight > 0 and "log_mu" in outputs:
            mu = self.verifier.mu_floor + torch.exp(outputs["log_mu"][:-1])
            n0 = self.background_counts(blk)[1:]
            lb = (mu - n0 * torch.log(mu)).sum()
            terms.append(self.background_weight * lb)
            parts["background"] = float(lb.detach())
        if steps > 1 and blk["labels"].numel() and (self.evidence_weight > 0 or self.stability_weight > 0):
            chains = self.chains(outputs, blk["aux"], blk["events"], steps)
            labels, m = blk["labels"], logits.detach()
            D = self.head.max_delay
            if self.evidence_weight > 0 and chains:
                le = 0.0
                for d, (Fd, valid) in enumerate(chains, 1):
                    z = m[valid] + self.head.fusion_weight(d).to(m.dtype) * Fd[valid].to(m.dtype)
                    le = le + F.binary_cross_entropy_with_logits(z, labels[valid], reduction="sum")
                le = le / D
                terms.append(self.evidence_weight * le)
                parts["evidence"] = float(le.detach())
            full = blk["events"]["t"] + D <= steps - 1
            if self.stability_weight > 0 and len(chains) == D and bool(full.any()):
                with torch.no_grad():
                    Fs = [torch.zeros_like(m[full])] + [Fd[full].to(m.dtype) for Fd, _ in chains]
                    zs = [m[full]] + [m[full] + self.head.fusion_weight(d).to(m.dtype) * Fs[d] for d in range(1, D + 1)]
                    final = zs[D] >= self.theta
                ls = 0.0
                for d in range(D):
                    target = ((zs[d] >= self.theta) == final).to(m.dtype)
                    q = self.head.stability_logit(zs[d] - self.theta, Fs[d], torch.full_like(zs[d], float(d) / D))
                    ls = ls + F.binary_cross_entropy_with_logits(q, target.to(q.dtype), reduction="sum")
                ls = ls / D
                terms.append(self.stability_weight * ls)
                parts["stability"] = float(ls.detach())
        if not terms:
            return None, parts
        out = terms[0]
        for term in terms[1:]:
            out = out + term
        return out, parts
