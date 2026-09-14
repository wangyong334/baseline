"""Aggregate per-seed training runs into one comparable table.

Reports mean +/- std of the final-window IoU across seeds, which is the figure
to quote when comparing methods. `best` columns are maxima over noisy per-epoch
evaluations and are shown only to expose their optimistic bias.

Usage:
    python -m utils.aggregate log/baseline_v2_paperlr_seed*  \
                              --label paper_lr
    python -m utils.aggregate log/run_a* log/run_b* --tail 5
"""
import argparse
import json
import math
from pathlib import Path


def read_run(directory, tail):
    """Prefer summary.json; fall back to recomputing from metrics.jsonl."""
    directory = Path(directory)
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("final_window") == tail or "final_mean_iou" not in summary:
            return summary

    metrics_path = directory / "metrics.jsonl"
    if not metrics_path.is_file():
        return None
    ious, accs, seed, epochs = [], [], None, 0
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        epochs += 1
        seed = record.get("seed", seed)
        if record.get("val_iou") is not None:
            ious.append(record["val_iou"])
            if record.get("val_acc") is not None:
                accs.append(record["val_acc"])
    if not ious:
        return {"seed": seed, "epochs_completed": epochs, "validated_epochs": 0}
    window = min(tail, len(ious))
    tail_ious = ious[-window:]
    tail_mean = sum(tail_ious) / window
    return {
        "seed": seed,
        "epochs_completed": epochs,
        "validated_epochs": len(ious),
        "best_val_iou": max(ious),
        "final_epoch_iou": ious[-1],
        "final_window": window,
        "final_mean_iou": tail_mean,
        "final_std_iou": (math.sqrt(sum((v - tail_mean) ** 2 for v in tail_ious)
                                    / (window - 1)) if window > 1 else 0.0),
        "final_mean_acc": (sum(accs[-window:]) / min(window, len(accs))
                           if accs else None),
        "selection_bias_gap": max(ious) - sum(tail_ious) / window,
    }


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run directories, one per seed")
    parser.add_argument("--tail", type=int, default=5,
                        help="epochs averaged at the end of training (default 5)")
    parser.add_argument("--label", default="runs", help="name for this group")
    args = parser.parse_args()
    if args.tail < 1:
        parser.error("--tail must be >= 1")

    rows = []
    for run in args.runs:
        summary = read_run(run, args.tail)
        if summary is None:
            print("skip (no metrics):", run)
            continue
        rows.append((Path(run).name, summary))

    if not rows:
        raise SystemExit("No readable runs")

    print("\n=== %s (final-%d-epoch window) ===" % (args.label, args.tail))
    header = "%-38s %6s %10s %10s %10s %10s" % (
        "run", "seed", "final_mean", "final_std", "best", "bias_gap")
    print(header)
    print("-" * len(header))
    for name, summary in rows:
        if "final_mean_iou" not in summary:
            print("%-38s %6s   (no validated epochs)" %
                  (name[:38], summary.get("seed", "?")))
            continue
        print("%-38s %6s %10.4f %10.4f %10.4f %10.4f" % (
            name[:38], summary.get("seed", "?"),
            summary["final_mean_iou"], summary.get("final_std_iou", 0.0),
            summary["best_val_iou"], summary["selection_bias_gap"]))

    scored = [s for _, s in rows if "final_mean_iou" in s]
    if len(scored) >= 1:
        mean, std = mean_std([s["final_mean_iou"] for s in scored])
        best_mean, best_std = mean_std([s["best_val_iou"] for s in scored])
        bias, _ = mean_std([s["selection_bias_gap"] for s in scored])
        print("-" * len(header))
        print("ACROSS %d SEEDS" % len(scored))
        print("  final-window IoU : %.4f +/- %.4f   <- quote this" % (mean, std))
        print("  best-epoch  IoU : %.4f +/- %.4f   (optimistically biased)"
              % (best_mean, best_std))
        print("  mean selection-bias gap: %.4f" % bias)
        if std > 0:
            print("  a method must beat ~%.4f IoU to clear 1 std of seed noise"
                  % (2 * std))
    print()


if __name__ == "__main__":
    main()
