"""Evaluate per-event result files of one split with the benchmark metrics.

    python tools/evaluate.py --card evuav --root /path/to/EV-UAV-dataset --split test --results runs/x/test
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speed.data.dataset_card import iter_split, load_card  # noqa: E402
from speed.eval.metrics import BenchmarkMetrics  # noqa: E402
from speed.eval.results import load_result  # noqa: E402


def evaluate(card, root, split, results_dir, threshold=0.9, frame_ms=50.0, correct_thresh=1e-4):
    metrics = BenchmarkMetrics(card["sensor"]["width"], card["sensor"]["height"], int(round(frame_ms * 1000)),
                               threshold, correct_thresh)
    for stream in iter_split(card, root, split):
        res = load_result(results_dir, stream)
        metrics.update(stream, res["prob"], res.get("decision"), res.get("publish_us"))
    return metrics.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--results", required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--frame-ms", type=float, default=50.0)
    parser.add_argument("--correct-thresh", type=float, default=1e-4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = evaluate(load_card(args.card), args.root, args.split, args.results, args.threshold, args.frame_ms,
                   args.correct_thresh)
    brief = {k: v for k, v in out.items() if k != "per_recording"}
    print(json.dumps(brief, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
