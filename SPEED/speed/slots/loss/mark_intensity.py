"""Slot 9 base variant: per-event BCE on the mark logits + Poisson NLL of the intensity head (no pos_weight).

    mark       sum_e BCE(mark(e), label(e))
    intensity  sum_px [ g - m log g ],  m = target-event count of the step averaged over an s x s neighbourhood
Both are sums; the training loop divides by the number of events of the whole stream.
"""
import torch.nn.functional as F


def smoothed_target(target_counts, smooth):
    s = int(smooth)
    if s <= 1:
        return target_counts
    if s % 2 == 0:
        raise ValueError("intensity smoothing window must be odd")
    shape = target_counts.shape
    flat = target_counts.reshape((-1, 1) + tuple(shape[-2:]))
    return F.avg_pool2d(flat, s, stride=1, padding=s // 2, count_include_pad=False).view(shape)


class MarkIntensityLoss(object):
    def __init__(self, mark_weight=1.0, intensity_weight=1.0, intensity_smooth=3):
        self.mark_weight, self.intensity_weight = float(mark_weight), float(intensity_weight)
        self.intensity_smooth = int(intensity_smooth)
        if self.mark_weight < 0 or self.intensity_weight < 0 or (self.mark_weight == 0 and self.intensity_weight == 0):
            raise ValueError("loss weights must be non-negative and not both zero")

    @staticmethod
    def target_counts(blk):
        src, ev = blk["source"], blk["events"]
        steps = blk["n_steps"]
        return src.accumulate(ev["t"] * src.plane + ev["pixel"], blk["labels"], steps * src.plane).view(
            steps, 1, 1, src.height, src.width)

    def __call__(self, outputs, logits, blk):
        """-> (total loss or None, {"mark": float, "intensity": float})"""
        terms, parts = [], {"mark": 0.0, "intensity": 0.0}
        if blk["labels"].numel() and self.mark_weight > 0:
            lm = F.binary_cross_entropy_with_logits(logits, blk["labels"], reduction="sum")
            terms.append(self.mark_weight * lm)
            parts["mark"] = float(lm.detach())
        if self.intensity_weight > 0:
            log_g = outputs["log_g"]
            m = smoothed_target(self.target_counts(blk), self.intensity_smooth)
            li = (log_g.exp() - m * log_g).sum()
            terms.append(self.intensity_weight * li)
            parts["intensity"] = float(li.detach())
        if not terms:
            return None, parts
        return (terms[0] if len(terms) == 1 else terms[0] + terms[1]), parts
