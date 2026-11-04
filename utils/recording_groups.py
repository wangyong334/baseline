"""Recording groups for guarded checkpoint selection and grouped reporting (FINAL_PLAN_20261104.md §6).

Rule (physical, fixed before any final-round result; an evaluation grouping only, no model reads it):
    dense(recording) <=> its background (label-0) events are in the dark shot-noise regime:
        polarity-balanced: 1/3 <= ON fraction of the background events <= 2/3
            (shot noise fires ON and OFF at similar rates; leakage / hot pixels are ON-dominated, Graca & Delbruck 2021)
        and background rate >= 0.4 Hz per pixel (a random-coincidence load of 0.5 in a 5x5 neighbourhood per 50 ms,
            half the boundary at which random coincidences form clusters, Inn 13.37).
On EV-UAV this selects exactly val_023 (ON 0.539, 0.54 Hz/px), test_022 and test_023 (0.535 / 0.537, 0.83 / 0.86),
the three recordings the project has called "dense" since 10-22, plus train_088..098 in the training split.
Margins: every other recording with >= 0.3 Hz/px has ON >= 0.90; every other balanced one has <= 0.094 Hz/px.
(A training-quantile cut on the rate alone was rejected: val_023 sits on it and moves across with the duration
convention.) EV-Flying's background is OFF-dominated, so it has no dense group.
"""
import glob
import os

import numpy as np

ON_BAND = (1.0 / 3.0, 2.0 / 3.0)
MIN_RATE_HZ = 0.4


def recording_stats(root, split, height=260, width=346):
    """{file name: (background rate in Hz per pixel, ON fraction of background events)} for one split."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, split, "*.npz"))):
        with np.load(path) as data:
            t = np.asarray(data["ev_loc"][:, 2])
            norm = np.asarray(data["evs_norm"])
        bg = norm[:, 4] < 0.5
        seconds = (float(t.max()) + 1.0) / 1000.0 if t.size else 1.0
        rate = float(bg.sum()) / max(seconds, 1e-3) / float(int(height) * int(width))
        on = float((norm[bg, 3] > 0.5).mean()) if bg.any() else 0.0
        out[os.path.basename(path)] = (rate, on)
    return out


def is_dense(rate, on):
    return ON_BAND[0] <= on <= ON_BAND[1] and rate >= MIN_RATE_HZ


def dense_names(root, splits=("val", "test"), height=260, width=346):
    """{split: sorted dense file names} under the shot-noise rule."""
    out = {}
    for split in splits:
        stats = recording_stats(root, split, height, width)
        out[split] = sorted(n for n, (r, on) in stats.items() if is_dense(r, on))
    return out


def pooled_iou(counts, names=None):
    """IoU of the pooled [TP, FP, FN] over the given names (all when None); None if the union is empty."""
    keys = list(counts) if names is None else [n for n in names if n in counts]
    tp = sum(counts[n][0] for n in keys)
    fp = sum(counts[n][1] for n in keys)
    fn = sum(counts[n][2] for n in keys)
    return tp / float(tp + fp + fn) if tp + fp + fn else None


def macro_iou(counts):
    """Mean per-recording IoU over recordings with a non-empty union (diagnostic only)."""
    vals = [c[0] / float(c[0] + c[1] + c[2]) for c in counts.values() if c[0] + c[1] + c[2] > 0]
    return float(np.mean(vals)) if vals else None


def group_report(counts, dense):
    """{'all', 'normal', 'dense', 'macro'} IoU for one per-recording count table."""
    dense = set(dense)
    normal = [n for n in counts if n not in dense]
    return {"all": pooled_iou(counts), "normal": pooled_iou(counts, normal),
            "dense": pooled_iou(counts, [n for n in counts if n in dense]), "macro": macro_iou(counts),
            "n_normal": len(normal), "n_dense": len(counts) - len(normal)}
