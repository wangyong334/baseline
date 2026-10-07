"""Run the original benchmark evaluation (utils/eval.py) and the legacy latency on SPEED streams, for comparison."""
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "SPEED"))

import torch  # noqa: E402

from speed.eval.metrics import BenchmarkMetrics  # noqa: E402
from utils.eval import evalute  # noqa: E402
from utils.stream_metrics import first_detection_latencies_by_event  # noqa: E402


def original_metrics(streams, probs, threshold=0.9, pd_det_ms=50, correct_thresh=1e-4):
    """streams must use millisecond-integer timestamps scaled to microseconds (EV-UAV layout)."""
    ev = evalute(SimpleNamespace(roc=True, pd_detT=pd_det_ms, correct_thresh=correct_thresh))
    for i, (s, prob) in enumerate(zip(streams, probs)):
        prob = torch.from_numpy(np.asarray(prob, dtype=np.float32))
        label = torch.from_numpy(s.label.astype(np.float32))
        zeros = np.zeros(s.n_events, dtype=np.float32)
        locs = torch.from_numpy(np.stack([zeros, s.x, s.y, s.t // 1000], 1).astype(np.float32))
        ev.roc_update(locs[:, 3], prob, s.target_id.astype(np.float64), label, locs, thresh=threshold)
        ev.matches[str(i)] = {"seg_pred": prob, "seg_gt": label}
    iou, acc = ev.evaluate_iou_and_accuracy(thresh=threshold)
    pd, fa = ev.cal_roc()
    return {"iou": iou, "acc": acc, "pd": pd, "fa": fa, "frames": ev.frame_num, "objects": ev.obj_num,
            "detected": ev.correct_num, "false_blobs": ev.false_num}


def speed_metrics(streams, probs, publish=None, threshold=0.9, frame_us=50000, correct_thresh=1e-4):
    m = BenchmarkMetrics(streams[0].width, streams[0].height, frame_us, threshold, correct_thresh)
    for i, (s, prob) in enumerate(zip(streams, probs)):
        m.update(s, np.asarray(prob, dtype=np.float32), publish_us=None if publish is None else publish[i])
    return m.result()


def legacy_first_detection(stream, prob, publish_window, threshold=0.9, window_ms=50, correct_thresh=1e-4):
    return first_detection_latencies_by_event(stream.t // 1000, stream.label.astype(np.float32),
                                              stream.target_id.astype(np.float64), np.asarray(prob, np.float32),
                                              window_ms, threshold, correct_thresh, publish_window)


def compare(orig, new):
    """Field names whose values differ (exact comparison)."""
    pairs = (("iou", new["iou"]), ("acc", new["acc"]), ("pd", new["pd"]), ("fa", new["fa"]),
             ("frames", new["counts"]["frames"]), ("objects", new["counts"]["objects"]),
             ("detected", new["counts"]["detected"]), ("false_blobs", new["counts"]["false_blobs"]))
    return [k for k, v in pairs if not (orig[k] == v or (orig[k] != orig[k] and v != v))]
