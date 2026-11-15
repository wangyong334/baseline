"""Export the per-event results of a decision-layer run for the figures (V20, 11-15).

For every recording of a split: the prior z0 (column 0: the streaming SNN alone, decided at arrival), the final SDNet
decision under the run's own calibrated bounds (eval/eval.json, not re-calibrated), the column it was decided at, and
the decision-variable trajectory of the events that waited. The official metrics are recomputed from the exported
labels and must equal the run's eval.json (printed as a check).
Run where the run's relative paths resolve (the V18 dir), with this code dir's tools:
    python <code dir>/tools/vis/export_preds.py --run run/v18/v5s37/coopw_s0 --out <dir> [--split test]
Output <out>/<name>.npz:
    prior_logit, final_logit  float32 [N] (calibrated logit scale; decided target iff >= tau)
    commit                    int16 [N]: 0 = decided at arrival, 1 + k = decided by its unit at step k
    last                      int16 [N]: last column (deadline or recording end)
    unit_idx, unit_Z          int32 [M], float16 [M, columns]: trajectories of the events that got a unit
<out>/meta.json: run, split, tau, bounds, alpha, window_ms, and the recomputed / stored metrics.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(CODE, "tools", "v18"))
import common as C  # noqa: E402
import eval_sdu as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--z0-root", default=None, help="override the prior dir stored in the run")
    ap.add_argument("--data-root", default=None, help="override the data root stored in the run")
    args = ap.parse_args()
    ck = torch.load(os.path.join(args.run, "sdu.pt"), map_location="cpu")
    cfg = C.local_paths(ck["config"], args.z0_root, args.data_root)
    with open(os.path.join(args.run, "eval", "eval.json")) as fh:
        ev = json.load(fh)
    tau = float(cfg["tau"])
    bounds = [float("inf") if b is None else float(b) for b in ev["bounds"]]
    d = E.load(args.run, args.split, cfg, int(ck["args"].get("limit", 0)))
    L = E.columns_of("calibrated", d, tau, bounds, ev["alpha"])
    Z = d["Z"]
    n = len(L)
    zc = Z[np.arange(n), L]
    starts = list(zip(d["start"][:-1], d["start"][1:]))
    got = C.official(d["recs"], [zc[a:b] for a, b in starts])
    ref = ev[args.split]["calibrated"]
    print("recomputed %s IoU %.4f ACC %.4f Pd %.4f Fa %.3e | eval.json IoU %.4f ACC %.4f Pd %.4f Fa %.3e" % (
        args.split, got["iou"], got["acc"], got["pd"], got["fa"], ref["iou"], ref["acc"], ref["pd"], ref["fa"]))
    if abs(got["iou"] - ref["iou"]) > 1e-6:
        raise RuntimeError("exported labels do not reproduce the run's metrics")
    os.makedirs(args.out, exist_ok=True)
    for i, (a, b) in enumerate(starts):
        units = np.flatnonzero(L[a:b] >= 1).astype(np.int32)
        np.savez_compressed(os.path.join(args.out, d["recs"][i]["name"]),
                            prior_logit=Z[a:b, 0].astype(np.float32), final_logit=zc[a:b].astype(np.float32),
                            commit=L[a:b].astype(np.int16), last=d["last"][a:b].astype(np.int16),
                            unit_idx=units, unit_Z=Z[a:b][units].astype(np.float16))
    meta = {"run": os.path.abspath(args.run), "split": args.split, "tau": tau, "bounds": ev["bounds"],
            "alpha": ev["alpha"], "window_ms": int(d["wms"]), "recordings": [r["name"] for r in d["recs"]],
            "recomputed": {k: got[k] for k in ("iou", "acc", "pd", "fa")},
            "eval_json": {k: ref[k] for k in ("iou", "acc", "pd", "fa")}}
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("EXPORT PREDS FINISHED %d recordings, %d events (%.2f%% decided by units) -> %s" % (
        len(starts), n, 100.0 * float((L >= 1).mean()), args.out))


if __name__ == "__main__":
    main()
