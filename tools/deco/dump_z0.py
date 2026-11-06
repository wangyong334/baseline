"""DeCo step 1: frozen backbone -> arrival evidence z0 of every event of a split (one file per recording).

z0 = backbone logit - log(pos_weight), i.e. the calibrated arrival logit; output format = a one-column trajectory
(traj [n_events, 1], labels, log_w, window_ms), readable by tools/final_analysis.load_split.
Optionally compares with an existing head trajectory (column 0 = the same backbone's arrival value).
Usage (code dir, conda activate evuav):
    python tools/deco/dump_z0.py --checkpoint <backbone .pt> --split train --out <dir>/z0_train [--check-traj <dir>]
"""
import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import train_stream_v1 as T  # noqa: E402
from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from dataset.stream_source import make_window_source  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=None, help="default: the root stored in the checkpoint config")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--check-traj", default=None, help="head trajectory dir of the same backbone (column 0 = z0)")
    ap.add_argument("--limit", type=int, default=0, help="first N recordings only (smoke tests)")
    args = ap.parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = dict(ckpt["config"])
    if args.data_root:
        cfg["root"] = args.data_root
    cfg["execution"], cfg["input_device"] = "step", "cpu"
    q99 = np.asarray(ckpt["stats"]["q99"], dtype=np.float32)
    log_w = math.log(float(ckpt["stats"]["pos_weight"]))
    window_ms = int(cfg["window_ms"])
    net = T.build_net(cfg, device)
    net.load_state_dict(ckpt["model"])
    if not bool(net.gain_calibrated):
        raise RuntimeError("backbone gain not calibrated: %s" % args.checkpoint)
    net.eval()
    os.makedirs(args.out, exist_ok=True)
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(cfg["root"], args.split, "*.npz")))
    if args.limit:
        names = names[:args.limit]
    t0, max_diff = time.time(), 0.0
    for ri, name in enumerate(names):
        seq = EvUAVStream(cfg["root"], args.split, cfg, [name])[0]
        with np.load(os.path.join(cfg["root"], args.split, name)) as d:
            labels = np.asarray(d["evs_norm"])[:, 4].astype(np.float32)
        z = np.full(len(labels), np.nan, dtype=np.float32)
        source = make_window_source(seq, cfg, q99, device, next(net.parameters()).dtype)
        states = None
        with torch.no_grad():
            for k in range(seq.n_windows):
                x, events, _, idx = source.window(k)
                logits, states, _ = net(x, events, states, state_mode="carry")
                z[idx] = logits.float().cpu().numpy() - log_w
        if args.check_traj:
            with np.load(os.path.join(args.check_traj, name)) as tr:
                ref = np.asarray(tr["traj"][:, 0], np.float32)
            ok = ~np.isnan(ref) & ~np.isnan(z)
            if ok.any():
                max_diff = max(max_diff, float(np.abs(ref[ok] - z[ok]).max()))
        np.savez(os.path.join(args.out, name), traj=z[:, None], labels=labels, log_w=np.float32(log_w),
                 window_ms=np.int64(window_ms))
        print("%s %3d/%d %s: %d events (%.0f s)%s" % (args.split, ri + 1, len(names), name, len(z), time.time() - t0,
              "" if not args.check_traj else ", max |z - check| so far %.2e" % max_diff), flush=True)
    print("DUMP Z0 FINISHED %s -> %s (window %d ms)" % (args.split, args.out, window_ms), flush=True)


if __name__ == "__main__":
    main()
