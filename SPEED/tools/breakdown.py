"""Target-event recall by speed / size / class for several result directories of one split.

    python tools/breakdown.py --card evflying --root /path/to/evflying --split test \
        --results net=runs/x/test/net pub=runs/x/test/pub k5=runs/k5/test --reference net --out b1.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speed.data.dataset_card import iter_split, load_card  # noqa: E402
from speed.eval.breakdown import RecallBreakdown, format_table  # noqa: E402
from speed.eval.metrics import decide  # noqa: E402
from speed.eval.results import load_result  # noqa: E402


def breakdown(card, root, split, results, threshold=0.9, reference=None, step_ms=50.0, window_us=50000):
    """results: ordered list of (name, directory)."""
    acc = RecallBreakdown([n for n, _ in results], reference, step_ms, window_us)
    for stream in iter_split(card, root, split):
        decisions = {}
        for name, directory in results:
            res = load_result(directory, stream)
            decisions[name] = decide(res["prob"], res.get("decision"), threshold)
        acc.update(stream, decisions)
        print("done %s" % stream.name, flush=True)
    return acc.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--results", nargs="+", required=True, help="name=directory, in display order")
    parser.add_argument("--reference", default=None, help="readout name for the kept-vs-reference columns")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--step-ms", type=float, default=50.0, help="speed unit: px per this many ms")
    parser.add_argument("--window-ms", type=float, default=50.0, help="centroid window of the kinematics")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    results = [tuple(item.split("=", 1)) for item in args.results]
    if any(len(r) != 2 for r in results):
        parser.error("--results entries must be name=directory")
    if args.reference and args.reference not in [n for n, _ in results]:
        parser.error("--reference must be one of the result names")
    out = breakdown(load_card(args.card), args.root, args.split, results, args.threshold, args.reference,
                    args.step_ms, int(round(args.window_ms * 1000)))
    for table in ("speed", "cheb", "size"):
        for key in sorted(out["tables"][table]):
            print("\n[%s / %s]  speed unit: px per %g ms" % (table, key, args.step_ms))
            print(format_table(out, table, key))
    print("\nfalse target events (background %d): %s" % (out["background_events"], out["false_events"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=1)
    print("BREAKDOWN FINISHED")


if __name__ == "__main__":
    main()
