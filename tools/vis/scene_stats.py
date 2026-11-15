"""Candidate scenes for the qualitative figure (V20, 11-15): per recording of a split, the properties that make a scene
hard (background density, targets at once, target onsets, speed) and, given exported predictions, each method's IoU and
the onset events SDNet recovers over the streaming SNN alone. Rows are then picked by stated criteria, not by eye.
Usage: python tools/vis/scene_stats.py --data-root <EV-UAV-dataset> [--sdnet <export dir>] [--method NAME=DIR ...]
           [--window-ms 2000 --step-ms 500]   (also rank event-stream windows: the clips drawn in the figure)
    --sdnet: tools/vis/export_preds.py output (prior and final columns); --method: dump_competitor.py output (prob).
"""
import argparse
import glob
import json
import os

import numpy as np

SLICE_MS = 50.0
ONSET_MS = 250.0                        # events of a target within this long after its first event


def load_events(path):
    d = np.load(path)
    loc, norm = d["ev_loc"], d["evs_norm"]
    return {"x": loc[:, 0].astype(np.float32), "y": loc[:, 1].astype(np.float32), "t": loc[:, 2].astype(np.float64),
            "lab": norm[:, 4] > 0.5, "tid": norm[:, 5].astype(np.int64)}


def iou(pred, lab):
    inter = float((pred & lab).sum())
    union = float((pred | lab).sum())
    return inter / union if union else float("nan")


def scene_row(ev):
    t, lab, tid = ev["t"], ev["lab"], ev["tid"]
    dur_s = max(t.max() - t.min(), 1.0) / 1000.0
    s = (t // SLICE_MS).astype(np.int64)
    targets = np.unique(tid[lab])
    per_slice = {}
    onset = np.zeros(len(t), bool)
    speeds = []
    for k in targets:
        m = lab & (tid == k)
        ts = t[m]
        onset |= m & (t < ts.min() + ONSET_MS)
        for sl in np.unique(s[m]):
            per_slice[sl] = per_slice.get(sl, 0) + 1
        sl, inv = np.unique(s[m], return_inverse=True)
        if len(sl) > 1:
            cx = np.bincount(inv, ev["x"][m]) / np.bincount(inv)
            cy = np.bincount(inv, ev["y"][m]) / np.bincount(inv)
            step = np.hypot(np.diff(cx), np.diff(cy)) / np.maximum(np.diff(sl), 1)
            speeds.append(np.median(step))
    return {"events": len(t), "bg_rate": float((~lab).sum()) / dur_s / (346 * 260),   # background events / px / s
            "targets": len(targets), "max_at_once": max(per_slice.values()) if per_slice else 0,
            "speed": float(np.median(speeds)) if speeds else 0.0, "onset_mask": onset}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sdnet", default=None)
    ap.add_argument("--method", action="append", default=[], help="NAME=DIR of dump_competitor.py predictions")
    ap.add_argument("--window-ms", type=float, default=0, help="also rank sliding windows of this length (0 = off)")
    ap.add_argument("--step-ms", type=float, default=500)
    ap.add_argument("--min-targets", type=int, default=300, help="target events a window must hold")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    tau = None
    if args.sdnet:
        with open(os.path.join(args.sdnet, "meta.json")) as fh:
            tau = json.load(fh)["tau"]
    methods = [m.split("=", 1) for m in args.method]
    head = "%-13s %8s %8s %4s %4s %6s" % ("recording", "events", "bg/px/s", "tgt", "max", "px/sl")
    if args.sdnet:
        head += " | %7s %7s %7s" % ("SNN", "SDNet", "rescued")
    head += "".join(" | %10s" % n for n, _ in methods)
    print(head)
    rows, windows = [], []
    for path in sorted(glob.glob(os.path.join(args.data_root, args.split, "*.npz"))):
        name = os.path.basename(path)
        ev = load_events(path)
        r = scene_row(ev)
        line = "%-13s %8d %8.3f %4d %4d %6.2f" % (name, r["events"], r["bg_rate"], r["targets"], r["max_at_once"],
                                                  r["speed"])
        if args.sdnet:
            p = np.load(os.path.join(args.sdnet, name))
            prior, final = p["prior_logit"] >= tau, p["final_logit"] >= tau
            r["iou_snn"], r["iou_sdnet"] = iou(prior, ev["lab"]), iou(final, ev["lab"])
            r["rescued"] = int((r["onset_mask"] & ~prior & final).sum())     # onset target events SDNet recovers
            line += " | %7.4f %7.4f %7d" % (r["iou_snn"], r["iou_sdnet"], r["rescued"])
        for n, dpath in methods:
            f = os.path.join(dpath, name)
            v = iou(np.load(f)["prob"] >= 0.9, ev["lab"]) if os.path.exists(f) else float("nan")
            r["iou_" + n] = v
            line += " | %10.4f" % v
        r["name"] = name
        if args.window_ms and args.sdnet:
            preds = {"SNN": prior, "SDNet": final}
            for n, dpath in methods:
                f = os.path.join(dpath, name)
                if os.path.exists(f):
                    preds[n] = np.load(f)["prob"] >= 0.9
            t = ev["t"]
            for w0 in np.arange(t.min(), max(t.max() - args.window_ms, t.min()) + 1, args.step_ms):
                m = (t >= w0) & (t < w0 + args.window_ms)
                if (ev["lab"] & m).sum() < args.min_targets:
                    continue
                v = {k: iou(p[m], ev["lab"][m]) for k, p in preds.items()}
                rivals = [v[k] for k in v if k not in ("SNN", "SDNet")] or [v["SNN"]]
                windows.append((v["SDNet"] - max(rivals), name, w0, v))
        rows.append(r)
        print(line, flush=True)
    print("\ncandidates (top 3 per criterion):")
    crit = [("dense background (bg/px/s)", "bg_rate"), ("most targets at once", "max_at_once"),
            ("fastest targets (px/slice)", "speed")]
    if args.sdnet:
        crit.append(("onset events rescued by SDNet", "rescued"))
    for title, key in crit:
        top = sorted(rows, key=lambda r: -r[key])[:3]
        print("  %-32s %s" % (title, ", ".join("%s (%s)" % (r["name"][:-4], ("%.3g" % r[key])) for r in top)))
    if windows:
        print("\nwindows of %.0f ms ranked by SDNet IoU minus the best rival (%s), at most one per recording:" % (
            args.window_ms, ", ".join(n for n, _ in methods) if methods else "the streaming SNN alone"))
        seen = set()
        for gap, name, w0, v in sorted(windows, key=lambda w: -w[0]):
            if name in seen:
                continue
            seen.add(name)
            print("  --scene %s:%d-%d   gap %+.4f | %s" % (name[:-4], w0, w0 + args.window_ms, gap,
                                                       " ".join("%s %.3f" % kv for kv in v.items())))
            if len(seen) >= args.top:
                break
    print("SCENE STATS FINISHED")


if __name__ == "__main__":
    main()
