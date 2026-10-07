import math

import cv2
import numpy as np


def _decide(prob, decision, threshold):
    if decision is not None:
        return np.asarray(decision).astype(bool)
    prob = np.asarray(prob)
    thr = prob.dtype.type(threshold) if np.issubdtype(prob.dtype, np.floating) else threshold
    return prob >= thr  # compared in the probability dtype, as the original float32 torch comparison


def _ms_stats(values_us):
    if len(values_us) == 0:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p90_ms": None}
    v = np.asarray(values_us, dtype=np.float64) / 1000.0
    return {"n": int(v.size), "mean_ms": float(v.mean()), "median_ms": float(np.median(v)),
            "p90_ms": float(np.percentile(v, 90))}


class BenchmarkMetrics(object):
    """EV-UAV benchmark metrics (utils/eval.py semantics) for any sensor size and integer-microsecond timestamps.

    IoU / ACC: over all events. Pd / Fa: frames of frame_us; an event belongs to frame i only if
    i*T < t < (i+1)*T (events exactly on a frame edge are skipped), frames 0 .. (t.max - t.min) // T are scanned,
    and Fa = 8-connected false-alarm blobs / (sum of (t.max - t.min) // T) / (width * height).
    False-alarm pixel counts wrap at 256 as in the original uint8 mask.
    Latency (needs publish times): publish time minus event time for correctly detected target events, and the
    first-detection latency of every target (frame criterion as Pd, inclusive frame edges).
    """

    def __init__(self, width, height, frame_us=50000, threshold=0.9, correct_thresh=1e-4):
        self.width, self.height = int(width), int(height)
        self.frame_us, self.threshold, self.correct_thresh = int(frame_us), float(threshold), float(correct_thresh)
        self.tp = self.fp = self.fn = 0
        self.frames = self.objects = self.detected = self.false_blobs = 0
        self.latency_us = []
        self.first = []
        self.per_class = {}
        self.per_recording = []

    def _class(self, c):
        return self.per_class.setdefault(int(c), {"positives": 0, "tp": 0, "objects": 0, "detected": 0})

    def update(self, stream, prob=None, decision=None, publish_us=None):
        pred = _decide(prob, decision, self.threshold)
        if pred.shape != (stream.n_events,):
            raise ValueError("%s: prediction length mismatch" % stream.name)
        target = stream.label == 1
        tp = int(np.count_nonzero(pred & target))
        fp = int(np.count_nonzero(pred & ~target))
        fn = int(np.count_nonzero(~pred & target))
        self.tp, self.fp, self.fn = self.tp + tp, self.fp + fp, self.fn + fn
        for c in np.unique(stream.cls[target]):
            m = target & (stream.cls == c)
            entry = self._class(c)
            entry["positives"] += int(np.count_nonzero(m))
            entry["tp"] += int(np.count_nonzero(m & pred))
        self._pd_fa(stream, pred)
        if publish_us is not None:
            self._latency(stream, pred, np.asarray(publish_us, dtype=np.int64))
        self.per_recording.append({"name": stream.name, "tp": tp, "fp": fp, "fn": fn,
                                   "iou": tp / float(tp + fp + fn) if tp + fp + fn else float("nan")})

    def _pd_fa(self, stream, pred):
        t = stream.t
        if t.size == 0:
            return
        T = self.frame_us
        t_min, t_max = int(t.min()), int(t.max())
        if t_min >= T:
            raise ValueError("%s: frames start at t = 0; rebase streams that start after the first frame" % stream.name)
        span_frames = (t_max - t_min) // T
        self.frames += span_frames
        f = t // T
        valid = (t % T != 0) & (f <= span_frames)
        tid = stream.target_id
        tm = valid & (tid != 0)
        if tm.any():
            key = f[tm] * (int(tid.max()) + 1) + tid[tm]
            uniq, inv = np.unique(key, return_inverse=True)
            label = stream.label[tm]
            n = np.bincount(inv, weights=label.astype(np.float64), minlength=uniq.size)
            correct = np.bincount(inv, weights=(pred[tm] == (label == 1)).astype(np.float64), minlength=uniq.size)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = correct.astype(np.float32) / n.astype(np.float32)
            hit = ratio >= np.float32(self.correct_thresh)
            self.objects += int(uniq.size)
            self.detected += int(np.count_nonzero(hit))
            cls = np.zeros(uniq.size, dtype=np.int64)
            cls[inv] = stream.cls[tm]
            for c in np.unique(cls):
                entry = self._class(c)
                entry["objects"] += int(np.count_nonzero(cls == c))
                entry["detected"] += int(np.count_nonzero(hit & (cls == c)))
        false = valid & (stream.label == 0) & pred
        if false.any():
            W, H = self.width, self.height
            key = f[false] * (W * H) + stream.y[false] * W + stream.x[false]
            uniq, counts = np.unique(key, return_counts=True)
            uniq = uniq[counts % 256 != 0]
            if uniq.size:
                frame_of, pix = uniq // (W * H), uniq % (W * H)
                starts = np.flatnonzero(np.r_[True, np.diff(frame_of) != 0])
                ends = np.r_[starts[1:], uniq.size]
                mask = np.zeros(H * W, dtype=np.uint8)
                for s, e in zip(starts, ends):
                    mask[pix[s:e]] = 1
                    n_labels, _ = cv2.connectedComponents(mask.reshape(H, W), connectivity=8, ltype=cv2.CV_32S)
                    self.false_blobs += int(n_labels) - 1
                    mask[pix[s:e]] = 0

    def _latency(self, stream, pred, publish_us):
        target = stream.label == 1
        known = publish_us >= 0
        ok = target & pred & known
        self.latency_us.append(publish_us[ok] - stream.t[ok])
        T = self.frame_us
        tid = stream.target_id
        for ident in np.unique(tid[target]):
            m = target & (tid == ident)
            times, hit, pub = stream.t[m], (pred & known)[m], publish_us[m]
            t_first = int(times.min())
            win = times // T
            best = None
            for k in np.unique(win):
                in_k = win == k
                n_k = int(np.count_nonzero(in_k))
                hits = np.sort(pub[in_k & hit])
                need = max(1, int(math.ceil(self.correct_thresh * n_k - 1e-12)))
                if hits.size >= need and hits.size / float(n_k) >= self.correct_thresh:
                    when = int(hits[need - 1])
                    best = when if best is None else min(best, when)
            self.first.append({"name": stream.name, "target_id": int(ident), "t_first_us": t_first,
                               "latency_us": None if best is None else best - t_first})

    def result(self):
        union = self.tp + self.fp + self.fn
        positives = self.tp + self.fn
        out = {
            "iou": self.tp / float(union) if union else float("nan"),
            "acc": self.tp / float(positives) if positives else float("nan"),
            "pd": self.detected / float(self.objects) if self.objects else float("nan"),
            "fa": self.false_blobs / float(self.frames * self.width * self.height) if self.frames else float("nan"),
            "counts": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "frames": self.frames, "objects": self.objects,
                       "detected": self.detected, "false_blobs": self.false_blobs},
            "per_class": {str(c): {"acc": e["tp"] / float(e["positives"]) if e["positives"] else float("nan"),
                                   "pd": e["detected"] / float(e["objects"]) if e["objects"] else float("nan"),
                                   "positives": e["positives"], "objects": e["objects"]}
                          for c, e in sorted(self.per_class.items())},
            "per_recording": self.per_recording,
        }
        if self.first:
            lat = np.concatenate(self.latency_us) if self.latency_us else np.zeros(0, np.int64)
            found = [r["latency_us"] for r in self.first if r["latency_us"] is not None]
            out["publish_latency"] = _ms_stats(lat)
            out["first_detection"] = dict(_ms_stats(found), n_targets=len(self.first),
                                          detection_rate=len(found) / float(len(self.first)))
        return out
