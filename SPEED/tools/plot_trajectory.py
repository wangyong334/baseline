"""3-D event-stream comparison figure: one row per recording, one column per method, ground truth last.

    python tools/plot_trajectory.py --card evuav --root /path/to/EV-UAV-dataset --split test \
        --recordings test/test_003.npz test/test_012.npz --results Ours=runs/a/test K5=runs/k5/test \
        --t0 0 --t1 8 --zoom auto --out fig_trajectory.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speed.data.dataset_card import list_recordings, load_card  # noqa: E402
from speed.data.readers import read_recording  # noqa: E402
from speed.eval.metrics import decide  # noqa: E402
from speed.eval.results import load_result  # noqa: E402
from speed.viz.trajectory import panel, render  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--recordings", nargs="+", required=True)
    parser.add_argument("--results", nargs="*", default=[], help="label=result_directory")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--t0", type=float, default=None, help="seconds")
    parser.add_argument("--t1", type=float, default=None, help="seconds")
    parser.add_argument("--zoom", default="auto", choices=("auto", "none"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    card = load_card(args.card)
    paths = dict(list_recordings(card, args.root, args.split))
    methods = [item.split("=", 1) for item in args.results]
    t0 = None if args.t0 is None else int(args.t0 * 1e6)
    t1 = None if args.t1 is None else int(args.t1 * 1e6)
    rows = []
    for name in args.recordings:
        stream = read_recording(card, paths[name], name)
        row = []
        for label, directory in methods:
            res = load_result(directory, stream)
            row.append(panel(stream, decide(res["prob"], res.get("decision"), args.threshold), t0, t1, label))
        row.append(panel(stream, stream.label == 1, t0, t1, "GT"))
        rows.append(row)
    render(rows, args.out, zoom="auto" if args.zoom == "auto" else None)
    print("written:", args.out)


if __name__ == "__main__":
    main()
