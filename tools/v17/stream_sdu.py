"""V17 step 3: window-by-window replay of the whole model as deployed, with timing and an exactness check.

Per 50 ms window: (optional, --backbone) the frozen BIR backbone computes the priors of the window's events; then the
streaming decision layer (model/sdu.StreamingDecisionLayer) builds the window's evidence maps, fires the open units
that reach their bound, commits the safe priors and starts units for the others. Bounds = the calibrated ones in
<run>/eval/eval.json. Checks every committed event against the offline trajectories of tools/v17/train_sdu.py (same
column, same value) -- the offline training and evaluation path is then exactly what a causal stream computes.
Reports the wall time per window of the decision layer (and of the backbone), open units, committed columns.
Usage: python tools/v17/stream_sdu.py --run <dir> [--split test] [--names test_022.npz,...] [--backbone <ckpt>]
Last line: "SDU STREAM FINISHED ... | layer ms/window mean ... p95 ... max ... | mismatched columns ...".
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import stopping as P  # noqa: E402
from model.sdu import StreamingDecisionLayer  # noqa: E402
from train_sdu import build_net  # noqa: E402


def backbone_stream(ckpt_path, data_root, split, name, device):
    """Generator of (window, priors z0 of the window's events, event indices, ms) from the frozen backbone."""
    import train_stream_v1 as T
    from dataset.ev_uav_stream import EvUAVStream
    from dataset.stream_source import make_window_source
    ck = torch.load(ckpt_path, map_location=device)
    cfg = dict(ck["config"])
    cfg.update(root=data_root, execution="step", input_device="cpu")
    q99 = np.asarray(ck["stats"]["q99"], dtype=np.float32)
    log_w = math.log(float(ck["stats"]["pos_weight"]))
    net = T.build_net(cfg, device)
    net.load_state_dict(ck["model"])
    net.eval()
    seq = EvUAVStream(cfg["root"], split, cfg, [name])[0]
    source = make_window_source(seq, cfg, q99, device, next(net.parameters()).dtype)
    states = None
    with torch.no_grad():
        for k in range(seq.n_windows):
            x, events, _, idx = source.window(k)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits, states, _ = net(x, events, states, state_mode="carry")
            z = logits.float().cpu().numpy() - log_w
            ms = (time.perf_counter() - t0) * 1e3
            yield k, z, np.asarray(idx), ms


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--names", default="", help="comma-separated recordings (default: all of the split)")
    ap.add_argument("--backbone", default="", help="backbone checkpoint: time it and compare its priors")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--z0-root", default=None, help="override the prior dir stored in the run (another machine)")
    ap.add_argument("--data-root", default=None, help="override the data root stored in the run")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(os.path.join(args.run, "sdu.pt"), map_location="cpu")
    cfg = C.local_paths(ck["config"], args.z0_root, args.data_root)
    with open(os.path.join(args.run, "eval", "eval.json")) as fh:
        bounds = [float("inf") if b is None else float(b) for b in json.load(fh)["bounds"]]
    net, rect = build_net(cfg, dev)
    net.load_state_dict(ck["model"])
    recs = C.load_split(cfg["z0_root"], cfg["data_root"], args.split, cfg["n_steps"], int(ck["args"].get("limit", 0)))
    tr = np.load(os.path.join(args.run, "traj_%s.npz" % args.split))
    names = [n for n in args.names.split(",") if n] or [r["name"] for r in recs]
    tau = cfg["tau"]
    layer = StreamingDecisionLayer(net, rect, bounds, tau, C.SENSOR[0], C.SENSOR[1], dev)
    res = {"bounds": bounds, "recordings": {}}
    all_ms, all_bb, all_units, mism, n_ev, max_dv, max_dz = [], [], [], 0, 0, 0.0, 0.0
    for ri, rec in enumerate(recs):
        if rec["name"] not in names:
            continue
        a, b = int(tr["start"][ri]), int(tr["start"][ri + 1])
        Z = np.concatenate([rec["z0"][:, None].astype(np.float64), tr["U"][a:b].astype(np.float64)], 1)
        last = 1 + tr["F"][a:b].astype(np.int64)
        L_off = P.commit(Z, last, tau, bounds)
        col = np.full(len(last), -1, np.int64)
        val = np.zeros(len(last), np.float64)
        order = np.argsort(rec["win"], kind="stable")
        bnd = np.searchsorted(rec["win"][order], np.arange(rec["n_win"] + 1))
        bb = backbone_stream(args.backbone, cfg["data_root"], args.split, rec["name"], dev) if args.backbone else None
        layer.reset()
        ms_rec, units_rec, bb_rec = [], [], []
        for t in range(rec["n_win"]):
            ids = order[bnd[t]:bnd[t + 1]]
            if bb is not None:
                k, z_live, idx, ms = next(bb)
                if k != t or not np.array_equal(np.sort(idx), np.sort(ids)):
                    raise RuntimeError("backbone window %d does not hold the events of window %d" % (k, t))
                max_dz = max(max_dz, float(np.abs(z_live - rec["z0"][idx]).max()) if len(idx) else 0.0)
                bb_rec.append(ms)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            e, c, v = layer.window(t, rec["x"][ids], rec["y"][ids], rec["pol"][ids], rec["z0"][ids], ids,
                                   tr["F"][a:b][ids])
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ms_rec.append((time.perf_counter() - t0) * 1e3)
            units_rec.append(layer.open_units())
            col[e], val[e] = c, v
        if (col < 0).any():
            raise RuntimeError("%s: %d events never committed" % (rec["name"], int((col < 0).sum())))
        m = int((col != L_off).sum())
        same = col == L_off
        dv = float(np.abs(val[same] - Z[np.arange(len(col)), L_off][same]).max()) if same.any() else 0.0
        mism += m
        n_ev += len(col)
        max_dv = max(max_dv, dv)
        all_ms += ms_rec
        all_bb += bb_rec
        all_units += units_rec
        res["recordings"][rec["name"]] = {"events": len(col), "mismatched_columns": m, "max_value_diff": dv,
                                          "layer_ms": {"mean": float(np.mean(ms_rec)), "max": float(np.max(ms_rec))},
                                          "open_units_max": int(np.max(units_rec)),
                                          "backbone_ms_mean": float(np.mean(bb_rec)) if bb_rec else None}
        print("%s: %d events, %d windows | layer %.2f ms/window (max %.1f) | open units max %d | mismatched columns %d, "
              "max |dU| %.2e%s" % (rec["name"], len(col), rec["n_win"], np.mean(ms_rec), np.max(ms_rec),
                                   np.max(units_rec), m, dv, " | backbone %.2f ms/window" % np.mean(bb_rec)
                                   if bb_rec else ""), flush=True)
    ms = np.asarray(all_ms)
    res["summary"] = {"events": n_ev, "mismatched_columns": mism, "max_value_diff": max_dv,
                      "layer_ms": {"mean": float(ms.mean()), "p95": float(np.percentile(ms, 95)), "max": float(ms.max())},
                      "open_units": {"mean": float(np.mean(all_units)), "max": int(np.max(all_units))},
                      "backbone_ms": {"mean": float(np.mean(all_bb)), "p95": float(np.percentile(all_bb, 95)),
                                      "max": float(np.max(all_bb))} if all_bb else None,
                      "backbone_prior_max_abs_diff": max_dz if all_bb else None, "device": str(dev)}
    out = args.out or os.path.join(args.run, "eval", "stream_%s.json" % args.split)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    s = res["summary"]
    print("SDU STREAM FINISHED %d events | layer ms/window mean %.2f p95 %.2f max %.1f | open units mean %.1f max %d | "
          "mismatched columns %d, max |dU| %.2e%s" % (
              n_ev, s["layer_ms"]["mean"], s["layer_ms"]["p95"], s["layer_ms"]["max"], s["open_units"]["mean"],
              s["open_units"]["max"], mism, max_dv, " | backbone ms/window mean %.2f p95 %.2f, max |dz0| %.2e" % (
                  s["backbone_ms"]["mean"], s["backbone_ms"]["p95"], max_dz) if all_bb else ""), flush=True)


if __name__ == "__main__":
    main()
