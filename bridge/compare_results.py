"""Compare two SPEED result directories event by event (probabilities and publish times must be identical).

    python bridge/compare_results.py --root /path/to/EV-UAV-dataset --split test \
        --a log/speed_u4/results/s37_test --b log/speed_2b/s37_test --readouts net fused_d1 fused_d2 pub
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "SPEED"))

from speed.data.dataset_card import iter_split, load_card  # noqa: E402
from speed.eval.results import load_result  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", default="evuav")
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--readouts", nargs="+", required=True)
    parser.add_argument("--recordings", nargs="*", default=None)
    args = parser.parse_args()
    card = load_card(args.card)
    totals = {name: {"events": 0, "prob_diff": 0, "publish_diff": 0, "decision_diff": 0, "max_abs": 0.0}
              for name in args.readouts}
    for stream in iter_split(card, args.root, args.split, args.recordings):
        for name in args.readouts:
            a = load_result(os.path.join(args.a, name), stream)
            b = load_result(os.path.join(args.b, name), stream)
            t = totals[name]
            t["events"] += stream.n_events
            t["prob_diff"] += int(np.count_nonzero(a["prob"] != b["prob"]))
            t["publish_diff"] += int(np.count_nonzero(a["publish_us"] != b["publish_us"]))
            t["decision_diff"] += int(np.count_nonzero((a["prob"] >= np.float32(0.9)) != (b["prob"] >= np.float32(0.9))))
            if stream.n_events:
                t["max_abs"] = max(t["max_abs"], float(np.abs(a["prob"].astype(np.float64) - b["prob"]).max()))
    ok = True
    for name, t in totals.items():
        same = t["prob_diff"] == 0 and t["publish_diff"] == 0
        ok &= same
        print("%-9s events %d | prob differs %d (max |d| %.3g) | publish differs %d | decision@0.9 differs %d -> %s" % (
            name, t["events"], t["prob_diff"], t["max_abs"], t["publish_diff"], t["decision_diff"],
            "identical" if same else "DIFF"))
    print("RESULT COMPARISON", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
