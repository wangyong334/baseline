"""Read-only diagnosis (11-08): does the decision layer's training data look like the held-out data? CPU, numpy only.
For one backbone TAG (z0 dumps run/deco/<TAG>/z0_<split>) and one decision-layer run (traj_<split> of train_dnm.py):
  A. per split (train / val / test, from z0): events, target share, raw IoU / recall / precision at tau, band share,
     target share inside the band, arrival sign correct inside the band, targets inside the band, confident misses;
  B. inside the band, the target rate per z0 bin (calibration of the backbone score on each split);
  C. per recording (val / test): raw z0 vs decision layer at lag 0 and at the last lag (IoU, recall, precision);
  D. inside the band (val / test): how often targets / background end above tau at arrival, at lag 0 and at the last lag.
Usage: python tools/deco/diag_split_shift.py --tag-dir run/deco/v5evfs37 --run run/deco/v5evfs37/dnm_a0.01_s0_nr_v10 \
           --alpha 0.01 --out run/deco/v5evfs37/diag_split_shift.txt"""
import argparse
import glob
import math
import os

import numpy as np

BINS = [-4.6, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.6]


def counts(pred, y):
    return np.array([(pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()], np.int64)


def iou_rp(c):
    tp, fp, fn = c
    return tp / max(c.sum(), 1), tp / max(tp + fn, 1), tp / max(tp + fp, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    half = math.log((1 - args.alpha) / args.alpha)
    L = []

    L.append("== A. splits (raw backbone score z0, band |z0 - tau| < %.2f)" % half)
    L.append("split  files     events  target%%  raw IoU  recall  precision | band%%  target%% in band  sign ok in band"
             "  targets in band  confident misses")
    band_bins = {}
    for split in ("train", "val", "test"):
        fs = sorted(glob.glob(os.path.join(args.tag_dir, "z0_" + split, "*.npz")))
        if not fs:
            L.append("%-5s  missing" % split)
            continue
        c = np.zeros(3, np.int64)
        n = nt = nb = nbt = nbok = nconf = 0
        hist_t = np.zeros(len(BINS) - 1)
        hist_n = np.zeros(len(BINS) - 1)
        for f in fs:
            with np.load(f) as t:
                z = np.asarray(t["traj"][:, 0], np.float64)
                y = np.asarray(t["labels"]) > 0.5
                tau = math.log(9.0) - float(t["log_w"])
            p = z >= tau
            c += counts(p, y)
            b = np.abs(z - tau) < half
            n += len(z); nt += y.sum(); nb += b.sum(); nbt += (b & y).sum(); nbok += (b & (p == y)).sum()
            nconf += (y & (z <= tau - half)).sum()
            d = (z - tau)[b]
            hist_n += np.histogram(d, BINS)[0]
            hist_t += np.histogram(d[y[b]], BINS)[0]
        i, r, pr = iou_rp(c)
        L.append("%-5s  %5d  %9d  %6.3f  %7.4f  %6.4f  %9.4f | %5.2f  %15.3f  %15.3f  %15.3f  %16.3f" % (
            split, len(fs), n, 100.0 * nt / max(n, 1), i, r, pr, 100.0 * nb / max(n, 1), nbt / max(nb, 1),
            nbok / max(nb, 1), nbt / max(nt, 1), nconf / max(nt, 1)))
        band_bins[split] = (hist_t, hist_n)

    L.append("\n== B. target rate inside the band per z0 - tau bin (calibration; n in parentheses)")
    L.append("bin            " + "  ".join("%-17s" % s for s in band_bins))
    for k in range(len(BINS) - 1):
        cells = []
        for s, (ht, hn) in band_bins.items():
            cells.append("%.3f (%8d)  " % (ht[k] / max(hn[k], 1), hn[k]))
        L.append("[%5.1f, %5.1f)  %s" % (BINS[k], BINS[k + 1], " ".join(cells)))

    for split in ("val", "test"):
        tdir = os.path.join(args.run, "traj_" + split)
        fs = sorted(glob.glob(os.path.join(tdir, "*.npz")))
        if not fs:
            L.append("\n== C/D %s: no trajectories in %s" % (split, tdir))
            continue
        L.append("\n== C. %s per recording: raw z0 -> decision layer lag 0 -> last lag  (IoU / recall / precision)" % split)
        tot = {k: np.zeros(3, np.int64) for k in ("raw", "lag0", "last")}
        dstat = {k: np.zeros(3) for k in ("t", "b")}
        dn = {"t": 0, "b": 0}
        for f in fs:
            name = os.path.basename(f)
            with np.load(f) as t:
                traj = np.asarray(t["traj"], np.float64)
                y = np.asarray(t["labels"]) > 0.5
                tau = math.log(9.0) - float(t["log_w"])
            zf = os.path.join(args.tag_dir, "z0_" + split, name)
            with np.load(zf) as t:
                z = np.asarray(t["traj"][:, 0], np.float64)
            last = np.where(np.isnan(traj), -np.inf, np.arange(traj.shape[1])[None, :]).max(1).astype(int)
            fin = traj[np.arange(len(traj)), last]
            row = {"raw": counts(z >= tau, y), "lag0": counts(traj[:, 0] >= tau, y), "last": counts(fin >= tau, y)}
            for k in row:
                tot[k] += row[k]
            L.append("%-26s %s" % (name, "  ->  ".join("%.4f / %.4f / %.4f" % iou_rp(row[k]) for k in ("raw", "lag0", "last"))))
            b = np.abs(z - tau) < half
            for key, m in (("t", b & y), ("b", b & ~y)):
                dn[key] += m.sum()
                dstat[key] += [(z[m] >= tau).sum(), (traj[m, 0] >= tau).sum(), (fin[m] >= tau).sum()]
        L.append("%-26s %s" % ("ALL", "  ->  ".join("%.4f / %.4f / %.4f" % iou_rp(tot[k]) for k in ("raw", "lag0", "last"))))
        L.append("\n== D. %s inside the band: share above tau at arrival (raw) -> lag 0 -> last lag" % split)
        for key, lab in (("t", "targets"), ("b", "background")):
            s = dstat[key] / max(dn[key], 1)
            L.append("%-10s n %8d   %.3f -> %.3f -> %.3f" % (lab, dn[key], s[0], s[1], s[2]))
    text = "\n".join(L) + "\n"
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
