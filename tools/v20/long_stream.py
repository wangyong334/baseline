"""V20 long-stream evaluation (11-14): the recordings of a split played back to back as one unbounded stream.

The paper claims a streaming SNN for an unbounded event stream, but every other evaluation restarts the state at the
start of each 8 s recording. Here the state is never cleared: one "fresh" pass (state cleared at every recording, the
usual protocol) and then --repeats passes over the concatenated stream (the second pass starts with the state left by
the whole first pass). Reported per pass and per recording: TP / FP / FN at the report threshold (IoU = TP / (TP + FP +
FN), the official pooled definition), messages per window (BIR: sum of |dq| of the sending layers; LIF: spikes of the
sending layers) and the largest |state| in the network. A neuron whose state is bounded should give the same numbers in
every pass; drift shows up as falling IoU or rising messages / state over the passes.
Usage (code dir): python tools/v20/long_stream.py --checkpoint log/final/<run>_seed37/selected_seed37.pt --split test
                  [--repeats 2] [--out <json>] [--device cuda:0] [--limit N]
Last line: "LONG STREAM FINISHED <json>".
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import train_stream_v1 as T  # noqa: E402
from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from dataset.stream_source import make_window_source  # noqa: E402
from model.evspsegnet_stream import LAYER_NAMES  # noqa: E402


def max_abs(state):
    """Largest |value| in a (nested) network state."""
    if state is None:
        return 0.0
    if torch.is_tensor(state):
        return float(state.detach().abs().max()) if state.numel() else 0.0
    if isinstance(state, (list, tuple)):
        return max([max_abs(s) for s in state] + [0.0])
    if isinstance(state, dict):
        return max([max_abs(s) for s in state.values()] + [0.0])
    return 0.0


def run_pass(net, cfg, q99, device, names, root, split, fresh, states, window0):
    """One pass over the recordings; fresh clears the state at every recording. Returns (records, states, windows)."""
    threshold = float(cfg["threshold"])
    sending = net.message_layers()
    records, k_global = [], window0
    for name in names:
        seq = EvUAVStream(root, split, cfg, [name])[0]
        source = make_window_source(seq, cfg, q99, device, next(net.parameters()).dtype)
        if fresh:
            states = None
        tp = fp = fn = 0
        msgs, smax = [], []
        if net.is_tif:
            net.start_tif_stats(per_window=True)
        with torch.no_grad():
            for k in range(seq.n_windows):
                x, events, _, idx = source.window(k)
                logits, states, info = T.window_forward(net, x, events, states, cfg, k if fresh else k_global,
                                                        cfg.get("state_mode", "carry"), collect=not net.is_tif)
                prob = torch.sigmoid(logits).float().cpu().numpy()
                c = T.window_confusion(prob, seq.label[idx], threshold)
                tp, fp, fn = tp + int(c[0]), fp + int(c[1]), fn + int(c[2])
                if not net.is_tif:
                    msgs.append(float(sum(float(info["spikes"][LAYER_NAMES.index(n)].sum()) for n in sending)))
                smax.append(max_abs(states))
                k_global += 1
        if net.is_tif:
            st = net.stop_tif_stats()
            msgs = list(sum(np.asarray(st[n]["per_window_rate"], dtype=np.float64) * float(st[n]["neurons"])
                            for n in sending))
        m = np.asarray(msgs, dtype=np.float64)
        records.append({"name": name, "windows": int(seq.n_windows), "tp": tp, "fp": fp, "fn": fn,
                        "iou": tp / float(max(tp + fp + fn, 1)),
                        "msg_mean": float(m.mean()) if m.size else 0.0,
                        "msg_p95": float(np.percentile(m, 95)) if m.size else 0.0,
                        "msg_first16": float(m[:16].mean()) if m.size else 0.0,
                        "state_max": float(max(smax)) if smax else 0.0,
                        "state_end": float(smax[-1]) if smax else 0.0})
    return records, states, k_global


def summary(records):
    tp, fp, fn = (sum(r[key] for r in records) for key in ("tp", "fp", "fn"))
    w = np.asarray([r["windows"] for r in records], dtype=np.float64)
    return {"iou": tp / float(max(tp + fp + fn, 1)), "tp": tp, "fp": fp, "fn": fn,
            "msg_mean": float(np.average([r["msg_mean"] for r in records], weights=w)),
            "msg_p95_max": float(max(r["msg_p95"] for r in records)),
            "state_max": float(max(r["state_max"] for r in records))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=None, help="default: <checkpoint dir>/long_stream_<split>.json")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="first N recordings (smoke tests)")
    args = ap.parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = dict(ckpt["config"])
    if args.data_root:
        cfg["root"] = args.data_root
    cfg["execution"], cfg["input_device"] = "step", "cpu"
    T.seed_everything(int(cfg["seed"]), bool(cfg.get("deterministic", True)))
    q99 = np.asarray(ckpt["stats"]["q99"], dtype=np.float32)
    net = T.build_net(cfg, device)
    net.load_state_dict(ckpt["model"])
    if not bool(net.gain_calibrated):
        raise RuntimeError("backbone gain not calibrated: %s" % args.checkpoint)
    net.eval()
    root = cfg["root"]
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, args.split, "*.npz")))
    if args.limit:
        names = names[:args.limit]
    t0 = time.time()
    out = {"checkpoint": args.checkpoint, "split": args.split, "neuron": cfg["neuron"],
           "state_mode": cfg.get("state_mode"), "memory": list(T.memory_options(cfg)), "recordings": names,
           "passes": []}
    records, _, _ = run_pass(net, cfg, q99, device, names, root, args.split, True, None, 0)
    out["passes"].append({"kind": "fresh", "summary": summary(records), "records": records})
    print("fresh   IoU %.4f | msg/window %.0f (p95 max %.0f) | max |state| %.2f (%.0f s)" % (
        out["passes"][-1]["summary"]["iou"], out["passes"][-1]["summary"]["msg_mean"],
        out["passes"][-1]["summary"]["msg_p95_max"], out["passes"][-1]["summary"]["state_max"], time.time() - t0),
        flush=True)
    states, k_global = None, 0
    for rep in range(args.repeats):
        records, states, k_global = run_pass(net, cfg, q99, device, names, root, args.split, False, states, k_global)
        s = summary(records)
        out["passes"].append({"kind": "long_%d" % (rep + 1), "summary": s, "records": records})
        print("long %d  IoU %.4f | msg/window %.0f (p95 max %.0f) | max |state| %.2f | %d windows so far (%.0f s)" % (
            rep + 1, s["iou"], s["msg_mean"], s["msg_p95_max"], s["state_max"], k_global, time.time() - t0), flush=True)
    path = args.out or os.path.join(os.path.dirname(args.checkpoint), "long_stream_%s.json" % args.split)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("LONG STREAM FINISHED %s" % path, flush=True)


if __name__ == "__main__":
    main()
