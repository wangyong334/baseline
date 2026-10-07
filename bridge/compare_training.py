"""Compare a SPEED training run with a legacy one epoch by epoch (loss per event and validation IoU / ACC).

    python bridge/compare_training.py --legacy log/v21_floor4_seed37/metrics.jsonl --speed log/speed_2b/train_s37/metrics.jsonl
"""
import argparse
import json


def read(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--speed", required=True)
    args = parser.parse_args()
    legacy, speed = read(args.legacy), read(args.speed)
    identical = 0
    for a, b in zip(legacy, speed):
        pairs = ((a["loss_per_event"], b["loss_per_event"]), (a["val_iou"], b["val"]["iou"]), (a["val_acc"], b["val"]["acc"]))
        same = all(x == y for x, y in pairs)
        identical += same
        print("epoch %2d | loss/event %.6f %.6f | val IoU %.4f %.4f | ACC %.4f %.4f | %s" % (
            a["epoch"], pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1], pairs[2][0], pairs[2][1],
            "identical" if same else "differs"))
    best = lambda rows, key: max(range(len(rows)), key=lambda i: key(rows[i]))  # noqa: E731
    n = min(len(legacy), len(speed))
    print("epochs compared %d, identical %d | best epoch legacy %d, speed %d" % (
        n, identical, best(legacy[:n], lambda r: r["val_iou"]), best(speed[:n], lambda r: r["val"]["iou"])))


if __name__ == "__main__":
    main()
