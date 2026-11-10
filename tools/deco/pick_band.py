"""V13 diagnostic (11-10): share of the training events inside the martingale (Ville) band at a grid of risks, for one
backbone. The method keeps alpha_train = alpha / 10 = 0.001 for every backbone; a budget-capped band was considered on
11-10 and dropped (no data-chosen constant). The last line names the smallest alpha >= --alpha-min whose band holds at
most --budget of the events (diagnostic only).

Why: the band edges come from the bound P(flip) <= sigma(z0) / q (background side), which holds for a calibrated
prior; but the bulk of the background events of a BIR backbone sits only 1-2 nats below the lower edge at
alpha / 10, so the share inside the band jumps between backbones (seed 37: 7.5%, seed 38: 21%, seed 39: 62% of the
training events). The share is reported per backbone in the paper; training time grows with it.
Usage: python tools/deco/pick_band.py --z0-dir run/deco/<TAG>/z0_train [--alpha-min 0.001] [--budget 0.08]
Last line: "ALPHA_TRAIN <alpha> share <share of the training events> band <lo> <hi>".
"""
import argparse
import glob
import math
import os

import numpy as np

GRID = (0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05)


def logit(p):
    return math.log(p / (1.0 - p))


def ville_band(tau, alpha):
    q = 1.0 / (1.0 + math.exp(-tau))
    return logit(alpha * q), logit(1.0 - alpha * (1.0 - q))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--z0-dir", required=True)
    ap.add_argument("--alpha-min", type=float, default=0.001)
    ap.add_argument("--budget", type=float, default=0.08)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.z0_dir, "*.npz")))
    if not files:
        raise SystemExit("no z0 files in %s" % args.z0_dir)
    z, log_w = [], None
    for f in files:
        with np.load(f) as d:
            z.append(np.asarray(d["traj"][:, 0], np.float64))
            log_w = float(d["log_w"])
    z = np.concatenate(z)
    z = z[np.isfinite(z)]
    tau = logit(0.9) - log_w
    pick = None
    for a in GRID:
        if a < args.alpha_min - 1e-12:
            continue
        lo, hi = ville_band(tau, a)
        share = float(((z > lo) & (z < hi)).mean())
        print("alpha %.4g band (%.2f, %.2f) share %.4f" % (a, lo, hi, share))
        if pick is None and share <= args.budget:
            pick = (a, share, lo, hi)
    if pick is None:
        raise SystemExit("no alpha on the grid keeps the share within the budget %.3f (prior not calibrated?)" % args.budget)
    print("ALPHA_TRAIN %g share %.4f band %.3f %.3f" % pick)


if __name__ == "__main__":
    main()
