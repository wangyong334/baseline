"""Convert legacy per-event dumps (train_stream_v2.py --dump-dir) into SPEED result files, one directory per readout.

Publish times: net -> end of the event's window; fused_dD -> end of window min(k + D, last window) (delayed readout
flushes at the end of the sequence); pub -> end of window k + age_pub.

    python bridge/convert_dumps.py --dump log/verify/x_test --root /path/to/EV-UAV-dataset --split test \
        --readouts net fused_d1 fused_d2 pub --out runs/legacy/v21_s37_test
"""
import argparse
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "SPEED"))

from speed.data.dataset_card import iter_split, load_card  # noqa: E402
from speed.eval.results import save_result  # noqa: E402


def publish_windows(readout, birth, dump, n_windows):
    if readout == "net":
        return birth
    delayed = re.fullmatch(r"(?:fused|attr)_d(\d+)", readout)
    if delayed:
        return np.minimum(birth + int(delayed.group(1)), n_windows - 1)
    if readout == "pub":
        return birth + dump["age_pub"].astype(np.int64)
    raise ValueError("unknown readout %s" % readout)


def convert(card, root, split, dump_dir, readouts, out_dir, window_ms=50, n_windows=160):
    window_us = window_ms * 1000
    count = 0
    for stream in iter_split(card, root, split):
        path = os.path.join(dump_dir, os.path.basename(stream.name))
        with np.load(path) as data:
            dump = {k: np.asarray(data[k]) for k in data.files}
        locs = dump["locs"]
        if not (np.array_equal(locs[:, 1], stream.x) and np.array_equal(locs[:, 2], stream.y)
                and np.array_equal(locs[:, 3] * 1000, stream.t)):
            raise ValueError("%s: dump events do not match the recording" % stream.name)
        birth = (stream.t // window_us).astype(np.int64)
        for readout in readouts:
            prob = dump["probabilities"] if readout == "net" else dump["prob_" + readout]
            pw = publish_windows(readout, birth, dump, n_windows)
            save_result(os.path.join(out_dir, readout), stream, prob, publish_us=(pw + 1) * window_us,
                        meta={"source": "legacy dump", "readout": readout, "dump": os.path.abspath(dump_dir)})
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True)
    parser.add_argument("--card", default="evuav")
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--readouts", nargs="+", default=["net"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    n = convert(load_card(args.card), args.root, args.split, args.dump, args.readouts, args.out)
    print("converted %d recordings -> %s" % (n, args.out))


if __name__ == "__main__":
    main()
