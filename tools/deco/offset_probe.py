"""Read-only probe (11-08): one global offset b added to the decision quantity of the band events, chosen on val.
For each decision-layer run (traj_<split> of train_dnm.py; non-band events keep z0) and offsets b on a grid:
pooled event IoU / recall / precision on val and test at lag 0 and at the last lag (deadline; close to the fixed /
crossing commit). Prints the val-best b with its test result, and the test-oracle b (reference only, never reported).
Usage: python tools/deco/offset_probe.py --tag-dir run/deco/v5evfs37 --runs run/deco/v5evfs37/dnm_a0.01_s0_nr_v10 \
           run/deco/v5evfs37/dnm_a0.01_s0 --alpha 0.01 --out run/deco/v5evfs37/offset_probe.txt"""
import argparse
import glob
import math
import os

import numpy as np

GRID = np.round(np.arange(-3.0, 3.01, 0.25), 2)


def load(run, tag_dir, split, half):
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "traj_" + split, "*.npz"))):
        with np.load(f) as t:
            traj = np.asarray(t["traj"], np.float64)
            y = np.asarray(t["labels"]) > 0.5
            tau = math.log(9.0) - float(t["log_w"])
        with np.load(os.path.join(tag_dir, "z0_" + split, os.path.basename(f))) as t:
            z = np.asarray(t["traj"][:, 0], np.float64)
        last = np.where(np.isnan(traj), -np.inf, np.arange(traj.shape[1])[None, :]).max(1).astype(int)
        rows.append((z - tau, traj[:, 0] - tau, traj[np.arange(len(traj)), last] - tau, y, np.abs(z - tau) < half))
    return [np.concatenate([r[i] for r in rows]) for i in range(5)]


def score(m, y):
    tp, fp, fn = (m & y).sum(), (m & ~y).sum(), (~m & y).sum()
    return tp / max(tp + fp + fn, 1), tp / max(tp + fn, 1), tp / max(tp + fp, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    half = math.log((1 - args.alpha) / args.alpha)
    L = []
    for run in args.runs:
        data = {s: load(run, args.tag_dir, s, half) for s in ("val", "test")}
        if not len(data["val"][0]):
            L.append("%s: no trajectories" % run)
            continue
        L.append("== %s" % run)
        for s in ("val", "test"):
            z, _, _, y, _ = data[s]
            L.append("  raw backbone %-4s IoU / recall / precision %.4f / %.4f / %.4f" % ((s,) + score(z >= 0, y)))
        for col, lab in ((1, "lag 0"), (2, "last lag")):
            tab = {}
            for s in ("val", "test"):
                q, y, band = data[s][col], data[s][3], data[s][4]
                tab[s] = [score(np.where(band, q + b, q) >= 0, y) for b in GRID]
            iv = [r[0] for r in tab["val"]]
            it = [r[0] for r in tab["test"]]
            bv, bt = int(np.argmax(iv)), int(np.argmax(it))
            k0 = int(np.where(GRID == 0)[0][0])
            L.append("  %-8s b = 0: val %.4f  test %.4f | val-best b = %+.2f: val %.4f  test %.4f (recall %.4f, precision %.4f)"
                     " | test-oracle b = %+.2f: test %.4f" % (lab, iv[k0], it[k0], GRID[bv], iv[bv], it[bv], tab["test"][bv][1],
                                                            tab["test"][bv][2], GRID[bt], it[bt]))
            L.append("           curve b: " + " ".join("%+.2f:%.4f/%.4f" % (GRID[k], iv[k], it[k]) for k in range(0, len(GRID), 2)))
    text = "\n".join(L) + "\n"
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
