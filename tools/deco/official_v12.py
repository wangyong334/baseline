"""Official four metrics of a V12 export without the range head (memory-light; for EV-Flying-sized splits, 11-09).

tools/deco/fit_range.py builds feature arrays of every neuron x step in memory (tens of GB for EV-Flying's 26 M test
events); this script only needs the export of tools/deco/train_v12.py (z0_<split>/, evm_<split>_meta.npz,
run/raw_<split>.npz) and works recording by recording. Policies:
    prior      the small network's priors alone (every event decided at arrival by z0)
    crossing   the decision neurons' fixed-band commit (alpha band, the lags saved by the export); the 'zero' row of the
               same pass is the arrival decision U(0)
    deadline   the decision neurons' membrane at the deadline (test only)
Trajectories are written like tools/deco/calibrate_commit.export (frozen after the commit, NaN beyond the recording)
and scored by tools/final_analysis.py with the sensor size of the config (EV-UAV 260 x 346, EV-Flying x4 180 x 320).
Usage: python tools/deco/official_v12.py --run <train_v12 out dir> --data-root <root> --config <yaml>
Last line: "OFFICIAL V12 <run> | <split> <policy> IoU ACC Pd Fa | ...".
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from utils.stream_common import load_flat_config  # noqa: E402


def write_traj(run, data_root, split, meta, lt, L, out_dir):
    """Per-recording trajectories [events, D + 1]: the priors everywhere, the neurons' membranes frozen after their
    commit lag L (lt None = priors only)."""
    os.makedirs(out_dir, exist_ok=True)
    D = int(meta["D"])
    wms = int(meta["window_ms"])
    for ri, name in enumerate(meta["names"]):
        name = str(name)
        with np.load(os.path.join(run, "z0_" + split, name)) as tr:
            z, labels, log_w = np.asarray(tr["traj"][:, 0], np.float32), tr["labels"], tr["log_w"]
        with np.load(os.path.join(data_root, split, name)) as d:
            t = np.asarray(d["ev_loc"])[:, 2]
        win = (t // wms).astype(np.int64)
        avail = np.minimum(D, int(win.max()) - win)
        traj = np.repeat(z[:, None], D + 1, axis=1)
        if lt is not None:
            sel = np.nonzero(meta["rec"] == ri)[0]
            cols = np.minimum(np.arange(D + 1)[None, :], L[sel][:, None])
            traj[meta["idx"][sel]] = lt[sel[:, None], cols]
        traj[np.arange(D + 1)[None, :] > avail[:, None]] = np.nan
        np.savez(os.path.join(out_dir, name), traj=traj, log_w=log_w, labels=labels)
    return wms, D


def score(out_dir, split, data_root, wms, sensor, title):
    out = out_dir + ".json"
    cmd = [sys.executable, os.path.join(ROOT, "tools", "final_analysis.py"), "--traj-dir", out_dir, "--split", split,
           "--data-root", data_root, "--window-ms", str(wms), "--out", out, "--title", title,
           "--height", str(sensor[0]), "--width", str(sensor[1])]
    with open(out_dir + ".log", "w") as fh:
        subprocess.check_call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    with open(out, encoding="utf-8") as fh:
        return json.load(fh)["policies"]


def pick(p):
    return {k: round(float(p[k]), 6) for k in ("iou", "acc", "pd", "fa") if k in p}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="train_v12.py output dir (or its gated/ subdir)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--config", required=True, help="the training config (sensor size)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()
    cfg = load_flat_config(args.config)
    sensor = (int(cfg["height"]), int(cfg["width_px"]))
    res = {"run": args.run, "sensor": sensor}
    parts = []
    for split in args.splits:
        with np.load(os.path.join(args.run, "evm_%s_meta.npz" % split)) as m:
            meta = {k: m[k] for k in ("names", "rec", "idx", "F", "D", "window_ms")}
        with np.load(os.path.join(args.run, "run", "raw_%s.npz" % split)) as r:
            lt, lags = r["lam_traj"].astype(np.float32), r["lags"].astype(np.int64)
        D = int(meta["D"])
        F_ = np.minimum(meta["F"].astype(np.int64), D)
        base = os.path.join(args.run, "official")
        policies = [("prior", None, None), ("crossing", lt, lags)]
        if split == "test":
            policies.append(("deadline", lt, F_))
        res[split] = {}
        for name, traj, L in policies:
            wms, _ = write_traj(args.run, args.data_root, split, meta, traj, L, os.path.join(base, "%s_%s" % (name, split)))
            p = score(os.path.join(base, "%s_%s" % (name, split)), split, args.data_root, wms, sensor, "%s_%s" % (name, split))
            res[split][name] = pick(p["fixed_%d" % D])
            if name == "crossing":
                res[split]["arrival"] = pick(p["zero"])
            for k in ([name, "arrival"] if name == "crossing" else [name]):
                r_ = res[split][k]
                parts.append("%s %s IoU %.4f ACC %.4f Pd %s Fa %s" % (split, k, r_["iou"], r_["acc"],
                                                                      "%.4f" % r_["pd"] if "pd" in r_ else "-",
                                                                      "%.3e" % r_["fa"] if "fa" in r_ else "-"))
            print(parts[-1], flush=True)
    with open(os.path.join(args.run, "official", "official_v12.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("OFFICIAL V12 %s | %s" % (args.run, " | ".join(parts)), flush=True)


if __name__ == "__main__":
    main()
