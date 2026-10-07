"""Stage-1 regression gate (U4): the frozen V2-1 / V3 evaluation dumps, converted to SPEED result files and scored by
the SPEED evaluator, must reproduce the frozen evaluation JSON (IoU / ACC / Pd / Fa and first-detection latency) for
every readout. Also reports the event publish latency (new definition) for each readout.

    python bridge/check_frozen.py --root /path/to/EV-UAV-dataset --split test \
        --run s37=log/v3_dump/f_s37_test=log/v21_floor4_seed37/eval_test_best_val_iou_seed37_v3f.json \
        --results-dir log/speed_u4/results --out log/speed_u4/check_frozen_test.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "SPEED", "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert_dumps import convert  # noqa: E402
from evaluate import evaluate  # noqa: E402
from speed.data.dataset_card import load_card  # noqa: E402

FIRST = (("n_targets", "n_targets"), ("n_detected", "n"), ("detection_rate", "detection_rate"),
         ("latency_mean_ms", "mean_ms"), ("latency_median_ms", "median_ms"), ("latency_p90_ms", "p90_ms"))


def differences(frozen, new):
    out = []
    for key in ("iou", "acc", "pd", "fa"):
        if frozen[key] != new[key]:
            out.append("%s %r != %r" % (key, frozen[key], new[key]))
    lat, first = frozen.get("latency"), new.get("first_detection")
    if lat is not None:
        for old_key, new_key in FIRST:
            a, b = lat.get(old_key), first.get(new_key) if first else None
            if a is None and b is None:
                continue
            if a is None or b is None or abs(float(a) - float(b)) > 1e-9 * max(1.0, abs(float(a))):
                out.append("first_detection.%s %r != %r" % (old_key, a, b))
    return out


def check_run(card, root, split, label, dump_dir, frozen_path, results_dir):
    with open(frozen_path, encoding="utf-8") as handle:
        frozen = json.load(handle)["carry"]
    if frozen["split"] != split:
        raise ValueError("%s is a %s evaluation, not %s" % (frozen_path, frozen["split"], split))
    readouts = list(frozen["readouts"])
    out_dir = os.path.join(results_dir, label + "_" + split)
    n = convert(card, root, split, dump_dir, readouts, out_dir)
    if n != frozen["n_sequences"]:
        raise ValueError("%s: %d recordings converted, frozen run has %d" % (label, n, frozen["n_sequences"]))
    rows = {}
    for name in readouts:
        new = evaluate(card, root, split, os.path.join(out_dir, name))
        diff = differences(frozen["readouts"][name], new)
        rows[name] = {"identical": not diff, "differences": diff, "frozen": frozen["readouts"][name],
                      "speed": {k: v for k, v in new.items() if k != "per_recording"}}
        r = rows[name]["speed"]
        print("%-4s %-9s IoU %.4f ACC %.4f Pd %.4f Fa %.3e | first det. median %6.1f ms | publish mean %5.1f "
              "median %5.1f p90 %5.1f ms -> %s" % (
                  label, name, r["iou"], r["acc"], r["pd"], r["fa"], r["first_detection"]["median_ms"] or -1,
                  r["publish_latency"]["mean_ms"], r["publish_latency"]["median_ms"], r["publish_latency"]["p90_ms"],
                  "identical" if not diff else "DIFF " + "; ".join(diff)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--run", nargs="+", required=True, help="label=dump_dir=frozen_eval_json")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    card = load_card("evuav")
    report = {"split": args.split, "runs": {}}
    for item in args.run:
        label, dump_dir, frozen_path = item.split("=", 2)
        report["runs"][label] = check_run(card, args.root, args.split, label, dump_dir, frozen_path, args.results_dir)
    labels = list(report["runs"])
    summary = {}
    for name in report["runs"][labels[0]]:
        rows = [report["runs"][label][name]["speed"] for label in labels]
        summary[name] = {k: float(np.mean([r[k] for r in rows])) for k in ("iou", "acc", "pd", "fa")}
        summary[name]["publish_mean_ms"] = float(np.mean([r["publish_latency"]["mean_ms"] for r in rows]))
        summary[name]["first_median_ms"] = float(np.mean([r["first_detection"]["median_ms"] for r in rows]))
    report["mean_over_runs"] = summary
    ok = all(row["identical"] for run in report["runs"].values() for row in run.values())
    report["pass"] = ok
    print("\nmean over %s:" % ",".join(labels))
    for name, s in summary.items():
        print("  %-9s IoU %.4f ACC %.4f Pd %.4f Fa %.3e | publish mean %5.1f ms | first det. median %6.1f ms" % (
            name, s["iou"], s["acc"], s["pd"], s["fa"], s["publish_mean_ms"], s["first_median_ms"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print("FROZEN REGRESSION", "PASS" if ok else "FAIL", "->", args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
