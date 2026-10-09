"""Slot 9 V4 variant: train what inference uses.

    mark      sum_e BCE(mark(e), label(e))                                     (as the base loss)
    forecast  sum_px [ F - m' log F ],  F = push(g_k, v_k dt) + floor           (Poisson; V4-30)
              the intensity of step k, pushed along the predicted velocity, forecasts the target events of step k+1
              (m' = target counts of step k+1, smoothed s x s as in the base loss); the verifier uses g exactly so
    motion    sum_px c [ |v - v*|^2 / (2 sigma^2) + 2 log sigma ]               (isotropic Gaussian NLL; V4-1)
              v* = mean label velocity (px/ms) of the c target events at the pixel in that step (tracks seen in a
              single window have none); sigma is the predicted uncertainty
    current   optional base intensity term on the same step (weight 0 by default)
The last step of a chunk has no next step inside the chunk and gets no forecast term.
"""
import torch
import torch.nn.functional as F

from speed.core.splat import bilinear_splat
from speed.slots.loss.mark_intensity import MarkIntensityLoss, smoothed_target


class MarkForecastMotionLoss(object):
    needs_velocity = True

    def __init__(self, dt_ms, mark_weight=1.0, forecast_weight=1.0, motion_weight=1.0, current_weight=0.0,
                 intensity_smooth=3, forecast_floor=2e-4):
        self.dt = float(dt_ms)
        self.mark_weight, self.forecast_weight = float(mark_weight), float(forecast_weight)
        self.motion_weight, self.current_weight = float(motion_weight), float(current_weight)
        self.intensity_smooth = int(intensity_smooth)
        self.floor = float(forecast_floor)
        if min(self.mark_weight, self.forecast_weight, self.motion_weight, self.current_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.floor <= 0:
            raise ValueError("forecast_floor must be > 0")

    def forecast(self, log_g, motion):
        """log_g [T,1,1,H,W], motion [T,1,3,H,W] -> forecast of the next step [T,1,1,H,W]."""
        T = int(log_g.shape[0])
        g = torch.exp(log_g).reshape((T,) + tuple(log_g.shape[2:]))
        v = motion.reshape((T,) + tuple(motion.shape[2:]))
        pushed = bilinear_splat(g, v[:, 0:1] * self.dt, v[:, 1:2] * self.dt)
        return (pushed + self.floor).view(log_g.shape)

    def __call__(self, outputs, logits, blk):
        """-> (total loss or None, {"mark", "intensity" (forecast + current), "motion": floats})"""
        terms, parts = [], {"mark": 0.0, "intensity": 0.0, "motion": 0.0}
        if blk["labels"].numel() and self.mark_weight > 0:
            lm = F.binary_cross_entropy_with_logits(logits, blk["labels"], reduction="sum")
            terms.append(self.mark_weight * lm)
            parts["mark"] = float(lm.detach())
        log_g, motion = outputs["log_g"], outputs["motion"]
        T = int(log_g.shape[0])
        counts = None
        if (self.forecast_weight > 0 and T > 1) or self.current_weight > 0:
            counts = smoothed_target(MarkIntensityLoss.target_counts(blk), self.intensity_smooth)
        intensity = 0.0
        if self.forecast_weight > 0 and T > 1:
            Fc = self.forecast(log_g[:-1], motion[:-1])
            li = (Fc - counts[1:] * torch.log(Fc)).sum()
            terms.append(self.forecast_weight * li)
            intensity += float(li.detach())
        if self.current_weight > 0:
            lc = (log_g.exp() - counts * log_g).sum()
            terms.append(self.current_weight * lc)
            intensity += float(lc.detach())
        parts["intensity"] = intensity
        if self.motion_weight > 0:
            lv = self.motion_nll(motion, blk)
            if lv is not None:
                terms.append(self.motion_weight * lv)
                parts["motion"] = float(lv.detach())
        if not terms:
            return None, parts
        total = terms[0]
        for t in terms[1:]:
            total = total + t
        return total, parts

    @staticmethod
    def motion_nll(motion, blk):
        ev = blk["events"]
        vel = ev.get("vel")
        if vel is None:
            raise ValueError("the motion loss needs per-event label velocities (EventBlocks extras 'vel')")
        known = (blk["labels"] > 0) & torch.isfinite(vel).all(1)
        if not bool(known.any()):
            return None
        src = blk["source"]
        steps = int(blk["n_steps"])
        size = steps * src.plane
        keys = ev["t"][known] * src.plane + ev["pixel"][known]
        v = vel[known].to(motion.dtype)
        count = src.accumulate(keys, torch.ones_like(v[:, 0]), size).to(motion.dtype)
        sum_y = src.accumulate(keys, v[:, 0], size).to(motion.dtype)
        sum_x = src.accumulate(keys, v[:, 1], size).to(motion.dtype)
        has = count > 0
        c = count[has]
        target = torch.stack([sum_y[has] / c, sum_x[has] / c], 1)
        flat = motion.reshape(steps, 3, src.plane).permute(0, 2, 1).reshape(size, 3)[has]
        log_sigma = flat[:, 2]
        err2 = ((flat[:, :2] - target) ** 2).sum(1)
        return (c * (0.5 * err2 * torch.exp(-2.0 * log_sigma) + 2.0 * log_sigma)).sum()
