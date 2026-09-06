"""Streaming CPU foreground metrics; seg_acc means foreground recall."""
import numpy as np


class ForegroundMetrics:
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, probabilities, labels):
        # Caller supplies CPU numpy arrays. Never mutate probabilities or labels.
        p = np.asarray(probabilities).reshape(-1)
        y = np.asarray(labels).reshape(-1)
        if p.shape != y.shape or not p.size:
            raise ValueError("Nonempty predictions and labels must have equal length")
        if not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Require finite predictions and binary labels")
        predicted, target = p >= self.threshold, y == 1
        self.tp += int(np.count_nonzero(predicted & target))
        self.fp += int(np.count_nonzero(predicted & ~target))
        self.fn += int(np.count_nonzero(~predicted & target))
        self.tn += int(np.count_nonzero(~predicted & ~target))

    def compute(self):
        if self.tp + self.fn == 0:
            raise ValueError("Foreground IoU/recall undefined for a split with no foreground labels")
        return {"iou": self.tp / (self.tp + self.fp + self.fn),
                "seg_acc": self.tp / (self.tp + self.fn)}
