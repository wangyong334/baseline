"""Protocol for offline baselines (e.g. K5 / EV-SpSegNet): cut every recording into fixed-length blocks in the EV-UAV
file layout, let the baseline predict each block, then stitch the block predictions back into per-recording SPEED
result files (an event is published at the end of its block, when an offline method can first use it).

    python tools/offline_blocks.py export --card evflying --root /data/evflying --out /data/evflying_k5 \
        --block-ms 8000 --shape 1280 736 8192
    python tools/offline_blocks.py import --card evflying --root /data/evflying --blocks /data/evflying_k5 \
        --split test --dumps log/k5_evflying/dump_test --out runs/k5_evflying/test

Block files: <out>/<split>/<recording with "/" -> "__">__b<k>.npz with
    ev_loc   int64 [N, 3]  x, y, t (ms since the block start, floored)
    evs_norm float32 [N, 6]  x / shape_x, y / shape_y, t_ms / shape_t, p, label, target id
    index    int64 [N]  positions of the events in the recording (file order)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speed.data.dataset_card import iter_split, load_card  # noqa: E402
from speed.eval.results import save_result  # noqa: E402

SPLITS = ("train", "val", "test")


def block_name(recording, k):
    return "%s__b%04d.npz" % (recording.replace("\\", "/").replace("/", "__").rsplit(".", 1)[0], k)


def export_stream(stream, directory, block_us, shape):
    """Write the blocks of one recording; returns the manifest entries."""
    sx, sy, st = (float(v) for v in shape)
    blocks = []
    if stream.n_events == 0:
        return blocks
    k_of = stream.t // block_us
    order = np.argsort(k_of, kind="stable")                           # file order kept inside every block
    bounds = np.searchsorted(k_of[order], np.arange(int(k_of.max()) + 2))
    for k in range(bounds.size - 1):
        idx = order[bounds[k]:bounds[k + 1]]
        if idx.size == 0:
            continue
        t_ms = (stream.t[idx] - k * block_us) / 1000.0
        loc = np.stack([stream.x[idx], stream.y[idx], np.floor(t_ms).astype(np.int64)], 1).astype(np.int64)
        norm = np.stack([stream.x[idx] / sx, stream.y[idx] / sy, t_ms / st, stream.p[idx], stream.label[idx],
                         stream.target_id[idx]], 1).astype(np.float32)
        name = block_name(stream.name, k)
        np.savez(os.path.join(directory, name), ev_loc=loc, evs_norm=norm, index=idx.astype(np.int64))
        blocks.append({"file": name, "start_us": k * block_us, "end_us": min((k + 1) * block_us, stream.span_us),
                       "n": int(idx.size)})
    return blocks


def stitch_stream(stream, blocks, blocks_dir, dumps_dir):
    """Per-event probability and publish time of one recording from its block dumps."""
    prob = np.full(stream.n_events, np.nan, np.float32)
    publish = np.full(stream.n_events, -1, np.int64)
    for b in blocks:
        with np.load(os.path.join(blocks_dir, b["file"])) as blk, np.load(os.path.join(dumps_dir, b["file"])) as dump:
            loc, index = dump["locs"][:, 1:4], blk["index"]
            if loc.shape[0] != b["n"] or not np.array_equal(loc, blk["ev_loc"]):
                raise ValueError("%s: dump events do not match the block" % b["file"])
            prob[index] = dump["probabilities"]
            publish[index] = b["end_us"]
    if np.isnan(prob).any():
        raise ValueError("%s: %d events without a prediction" % (stream.name, int(np.isnan(prob).sum())))
    return prob, publish


def export(args):
    card = load_card(args.card)
    if card["sensor"]["width"] > args.shape[0] or card["sensor"]["height"] > args.shape[1] or args.block_ms > args.shape[2]:
        raise ValueError("block shape %s smaller than the sensor / block length" % (args.shape,))
    manifest = {"block_ms": args.block_ms, "shape": list(args.shape), "card": card["name"], "splits": {}}
    for split in args.splits:
        directory = os.path.join(args.out, split)
        os.makedirs(directory, exist_ok=True)
        entries = {}
        for stream in iter_split(card, args.root, split):
            entries[stream.name] = export_stream(stream, directory, int(args.block_ms * 1000), args.shape)
            print(split, stream.name, "%d blocks" % len(entries[stream.name]), flush=True)
        manifest["splits"][split] = entries
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    print("EXPORT FINISHED ->", args.out)


def stitch(args):
    card = load_card(args.card)
    with open(os.path.join(args.blocks, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest["splits"][args.split]
    count = 0
    for stream in iter_split(card, args.root, args.split):
        prob, publish = stitch_stream(stream, entries[stream.name], os.path.join(args.blocks, args.split), args.dumps)
        save_result(args.out, stream, prob, publish_us=publish,
                    meta={"source": "offline blocks", "block_ms": manifest["block_ms"], "dumps": os.path.abspath(args.dumps)})
        count += 1
    print("IMPORT FINISHED: %d recordings -> %s" % (count, args.out))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--card", required=True)
    e.add_argument("--root", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--splits", nargs="+", default=list(SPLITS))
    e.add_argument("--block-ms", type=float, default=8000.0)
    e.add_argument("--shape", type=int, nargs=3, required=True, help="normalisation shape x y t(ms) of the baseline")
    i = sub.add_parser("import")
    i.add_argument("--card", required=True)
    i.add_argument("--root", required=True)
    i.add_argument("--blocks", required=True)
    i.add_argument("--split", default="test")
    i.add_argument("--dumps", required=True)
    i.add_argument("--out", required=True)
    args = parser.parse_args()
    export(args) if args.command == "export" else stitch(args)


if __name__ == "__main__":
    main()
