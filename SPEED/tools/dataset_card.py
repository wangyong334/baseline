"""Compute the scale statistics of a dataset split (training split by default) and store them next to its card.

    python tools/dataset_card.py --card evflying --root /path/to/evflying
    python tools/dataset_card.py --card evuav --root /path/to/EV-UAV-dataset --split train
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speed.data.dataset_card import iter_split, load_card, stats_path  # noqa: E402
from speed.data.stats import split_statistics  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    card = load_card(args.card)
    stats = split_statistics(iter_split(card, args.root, args.split), int(round(args.window_ms * 1000)))
    stats.update({"card": card["name"], "split": args.split})
    out = args.out or stats_path(card)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    brief = {k: v for k, v in stats.items() if k != "recordings"}
    print(json.dumps(brief, indent=1, ensure_ascii=False))
    print("written:", out)


if __name__ == "__main__":
    main()
