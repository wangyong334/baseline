"""Stage-1 gate for the evaluation: on real EV-UAV recordings, SPEED metrics must equal the original utils/eval.py
bit for bit for several prediction patterns, and the first-detection latency must equal the legacy implementation.

    python bridge/check_eval.py --evuav-zip val.zip test.zip
"""
import argparse
import io
import os
import sys
import warnings
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from legacy_eval import compare, legacy_first_detection, original_metrics, speed_metrics  # noqa: E402
from speed.data.dataset_card import load_card  # noqa: E402
from speed.data.readers import read_evuav_npz  # noqa: E402
from speed.eval.metrics import BenchmarkMetrics  # noqa: E402


def load_streams(archives):
    card = load_card("evuav")
    streams = []
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith(".npz")):
                streams.append(read_evuav_npz(io.BytesIO(zf.read(name)), card, name))
    return streams


def patterns(streams, rng):
    noisy = [np.clip(s.label + rng.normal(0, 0.35, s.n_events), 0, 1).astype(np.float32) for s in streams]
    uniform = [rng.random(s.n_events).astype(np.float32) for s in streams]
    near = [np.where(rng.random(s.n_events) < 0.5, 0.9, 0.8999999).astype(np.float32) for s in streams]
    return {"noisy": noisy, "uniform": uniform, "threshold_edge": near}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evuav-zip", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    streams = load_streams(args.evuav_zip)
    print("recordings:", len(streams), "events:", sum(s.n_events for s in streams))
    ok = True
    for name, probs in patterns(streams, rng).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orig = original_metrics(streams, probs)
        new = speed_metrics(streams, probs)
        diff = compare(orig, new)
        ok &= not diff
        print("%-15s IoU %.6f ACC %.6f Pd %.6f Fa %.4e | blobs %d objects %d -> %s" % (
            name, new["iou"], new["acc"], new["pd"], new["fa"], new["counts"]["false_blobs"],
            new["counts"]["objects"], "identical" if not diff else "DIFF " + ",".join(diff)))
    mismatches = n_targets = 0
    for s, prob in zip(streams, patterns(streams, rng)["noisy"]):
        birth = (s.t // 1000) // 50
        pw = np.minimum(birth + rng.integers(0, 6, s.n_events), 159)
        legacy = {r["target_id"]: r["latency_ms"] for r in legacy_first_detection(s, prob, pw)}
        m = BenchmarkMetrics(s.width, s.height)
        m.update(s, prob, publish_us=(pw + 1) * 50000)
        for r in m.first:
            old = legacy[float(r["target_id"])]
            n_targets += 1
            same = (old is None and r["latency_us"] is None) or (old is not None and old * 1000 == r["latency_us"])
            mismatches += 0 if same else 1
    ok &= mismatches == 0
    print("first-detection latency: %d targets, %d mismatches" % (n_targets, mismatches))
    print("GATE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
