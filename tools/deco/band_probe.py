"""Read-only probe (11-08): use the decision layer only on events close to the threshold. Events with |z0 - tau| < h take
the decision quantity (+ offset b); all others keep the backbone decision z0 >= tau. (h, b) chosen on val, applied to
test; lag 0 and last lag (deadline). Also the test-oracle (h, b) as a reference only (never reported).
Usage: python tools/deco/band_probe.py --tag-dir run/deco/v5evfs37 --runs run/deco/v5evfs37/dnm_a0.01_s0_nr_v10 \
           run/deco/v5evfs37/dnm_a0.01_s0 --out run/deco/v5evfs37/band_probe.txt"""
import argparse
import glob
import math
import os

import numpy as np

HALFS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.6)
OFFS = np.round(np.arange(-1.0, 3.01, 0.5), 2)


def load(run, tag_dir, split):
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "traj_" + split, "*.npz"))):
        with np.load(f) as t:
            traj = np.asarray(t["traj"], np.float64)
            y = np.asarray(t["labels"]) > 0.5
            tau = math.log(9.0) - float(t["log_w"])
        with np.load(os.path.join(tag_dir, "z0_" + split, os.path.basename(f))) as t:
            z = np.asarray(t["traj"][:, 0], np.float64)
        last = np.where(np.isnan(traj), -np.inf, np.arange(traj.shape[1])[None, :]).max(1).astype(int)
        rows.append((z - tau, traj[:, 0] - tau, traj[np.arange(len(traj)), last] - tau, y))
    return [np.concatenate([r[i] for r in rows]) for i in range(4)]


def iou(m, y):
    tp, fp, fn = (m & y).sum(), (m & ~y).sum(), (~m & y).sum()
    return tp / max(tp + fp + fn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    L = []
    for run in args.runs:
        data = {s: load(run, args.tag_dir, s) for s in ("val", "test")}
        if not len(data["val"][0]):
            L.append("%s: no trajectories" % run)
            continue
        L.append("== %s" % run)
        raw = {s: iou(data[s][0] >= 0, data[s][3]) for s in ("val", "test")}
        L.append("  raw backbone: val %.4f  test %.4f" % (raw["val"], raw["test"]))
        for col, lab in ((1, "lag 0"), (2, "last lag")):
            res = {}
            for h in HALFS:
                for b in OFFS:
                    r = []
                    for s in ("val", "test"):
                        z, q, y = data[s][0], data[s][col], data[s][3]
                        near = np.abs(z) < h
                        r.append(iou(np.where(near, q + b >= 0, z >= 0), y))
                    res[(h, b)] = r
            kv = max(res, key=lambda k: res[k][0])
            kt = max(res, key=lambda k: res[k][1])
            L.append("  %-8s val-best h %.2f b %+.1f: val %.4f (%+.4f vs backbone)  test %.4f (%+.4f) | test-oracle h %.2f b %+.1f:"
                     " test %.4f" % (lab, kv[0], kv[1], res[kv][0], res[kv][0] - raw["val"], res[kv][1],
                                     res[kv][1] - raw["test"], kt[0], kt[1], res[kt][1]))
            for h in HALFS:
                bb = max(OFFS, key=lambda b: res[(h, b)][0])
                L.append("           h %.2f: best b on val %+.1f -> val %.4f  test %.4f" % (h, bb, res[(h, bb)][0], res[(h, bb)][1]))
    text = "\n".join(L) + "\n"
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
