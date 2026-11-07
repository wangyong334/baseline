"""DeCo step 4 (11-07): risk-calibrated firing thresholds for a trained decision-neuron run (no retraining).

Per wait step d, among the val events still open at d, estimate P(the decision at d differs from the decision at the
last available step | margin |Lambda(d) - tau|) with isotonic regression (pool-adjacent-violators, non-increasing in
the margin, no bins); the calibrated threshold b_d is the smallest margin from which every isotonic block has a
Clopper-Pearson upper bound (level delta / D, Bonferroni over the steps) below alpha (rule iso_ucb, default; rule iso
uses the point estimate only and over-commits at late steps where few events remain, 11-07 server check). An event
fires (commits) at the first d with margin >= b_d, else at its last available step. Calibrated on val, applied to test.
For the fixed rule (b = logit(1 - alpha) at every step) and the calibrated rule, writes 'frozen after commit'
trajectories (every column from the commit step on equals the committed value) and runs tools/final_analysis.py, so the
fixed_D row of its report gives the official IoU / ACC / Pd / Fa of the policy (its latency columns do not apply; the
policy's own mean wait is reported here).
Usage: python tools/deco/calibrate_commit.py --run <dnm run dir> --evidence <evidence prefix> --z0-root <dir with z0_*>
       --data-root <root> [--alpha 0.01] [--out <run>/calibrated]
Last line: "CALIBRATE FINISHED ..." with the test IoU and mean waits of both rules.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
from scipy.stats import beta as beta_dist


def logit(p):
    return math.log(p / (1.0 - p))


def pav_nonincreasing(y):
    vals, wts, lens = [], [], []
    for yi in -np.asarray(y, float):
        vals.append(yi)
        wts.append(1.0)
        lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            n = lens[-2] + lens[-1]
            vals.pop(), wts.pop(), lens.pop()
            vals[-1], wts[-1], lens[-1] = v, w, n
    return -np.repeat(vals, lens)


def load(args, split):
    ev = np.load("%s_%s.npz" % (args.evidence, split))
    raw = np.load(os.path.join(args.run, "raw_%s.npz" % split))
    lt = raw["lam_traj"]
    D = lt.shape[1] - 1
    return ev, lt, np.minimum(ev["F"], D), D


def commit(lt, F_, tau, margins):
    L = F_.copy()
    done = np.zeros(len(F_), bool)
    for d in range(lt.shape[1]):
        hit = ~done & (d < F_) & (np.abs(lt[:, d] - tau) >= margins[d])
        L[hit] = d
        done |= hit
    return L


def ucb(k, n, level):
    return 1.0 if k >= n else float(beta_dist.ppf(1.0 - level, k + 1, n - k))


def calibrate(lt, F_, tau, alpha, rule="iso_ucb", delta=0.1):
    final = lt[np.arange(len(F_)), F_] >= tau
    open_ = np.ones(len(F_), bool)
    margins = []
    level = delta / max(lt.shape[1] - 1, 1)
    for d in range(lt.shape[1]):
        cand = open_ & (d < F_)
        m = np.abs(lt[cand, d] - tau)
        change = (lt[cand, d] >= tau) != final[cand]
        b = float("inf")
        if m.size:
            order = np.argsort(m, kind="stable")
            ms, cs = m[order], change[order].astype(float)
            fit = pav_nonincreasing(cs)
            if rule == "iso":
                below = np.nonzero(fit < alpha)[0]
                b = float(ms[below[0]]) if below.size else float("inf")
            else:
                starts = np.flatnonzero(np.r_[True, fit[1:] != fit[:-1]])
                ends = np.r_[starts[1:], len(fit)]
                for a, e in zip(starts[::-1], ends[::-1]):     # from the largest margins down
                    if ucb(int(cs[a:e].sum()), int(e - a), level) >= alpha:
                        break
                    b = float(ms[a])
        margins.append(b)
        open_ &= ~(cand & (np.abs(lt[:, d] - tau) >= b))
    return margins


def export(ev, lt, L, D, args, split, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    wms = None
    for ri, name in enumerate(ev["names"]):
        name = str(name)
        with np.load(os.path.join(args.z0_root, "z0_" + split, name)) as tr:
            z, labels, log_w = np.asarray(tr["traj"][:, 0], np.float32), tr["labels"], tr["log_w"]
            wms = int(tr["window_ms"])
        with np.load(os.path.join(args.data_root, split, name)) as d:
            t = np.asarray(d["ev_loc"])[:, 2]
        win = (t // wms).astype(np.int64)
        avail = np.minimum(D, int(win.max()) - win)
        traj = np.repeat(z[:, None], D + 1, axis=1)
        sel = np.nonzero(ev["rec"] == ri)[0]
        cols = np.minimum(np.arange(D + 1)[None, :], L[sel][:, None])
        traj[ev["idx"][sel]] = lt[sel[:, None], cols]
        traj[np.arange(D + 1)[None, :] > avail[:, None]] = np.nan
        np.savez(os.path.join(out_dir, name), traj=traj, log_w=log_w, labels=labels)
    return wms


def official(args, split, traj_dir, wms, title, D):
    out = traj_dir + ".json"
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "final_analysis.py"),
           "--traj-dir", traj_dir, "--split", split, "--data-root", args.data_root, "--window-ms", str(wms),
           "--out", out, "--title", title]
    with open(traj_dir + ".log", "w") as fh:
        subprocess.check_call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    row = [ln for ln in open(os.path.splitext(out)[0] + ".md", encoding="utf-8") if ln.startswith("| fixed_%d " % D)]
    cells = [c.strip() for c in row[0].strip().strip("|").split("|")] if row else []
    return {"iou": float(cells[1]), "acc": float(cells[2]), "pd": float(cells[3]), "fa": float(cells[4])} if cells else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--z0-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--rule", choices=("iso_ucb", "iso"), default="iso_ucb")
    ap.add_argument("--delta", type=float, default=0.1, help="confidence budget of iso_ucb, split over the steps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_root = args.out or os.path.join(args.run, "calibrated")
    os.makedirs(out_root, exist_ok=True)
    ev_v, lt_v, F_v, D = load(args, "val")
    tau = logit(0.9) - float(ev_v["log_w"])
    rules = {"fixed": [logit(1.0 - args.alpha)] * (D + 1),
             "calibrated": calibrate(lt_v, F_v, tau, args.alpha, args.rule, args.delta)}
    res = {"alpha": args.alpha, "rule": args.rule, "delta": args.delta, "D": D, "margins": {k: [None if not np.isfinite(b) else round(b, 4) for b in v]
                                                     for k, v in rules.items()}}
    for split in ("val", "test"):
        ev, lt, F_, _ = (ev_v, lt_v, F_v, D) if split == "val" else load(args, split)
        final = lt[np.arange(len(F_)), F_] >= tau
        for name, mg in rules.items():
            L = commit(lt, F_, tau, mg)
            changed = float(((lt[np.arange(len(L)), L] >= tau) != final).mean()) if len(L) else 0.0
            d = os.path.join(out_root, "%s_%s" % (name, split))
            wms = export(ev, lt, L, D, args, split, d)
            met = official(args, split, d, wms, "%s_%s" % (name, split), D)
            res["%s_%s" % (name, split)] = dict(met, mean_wait_pending=float(L.mean()) if len(L) else 0.0,
                                                changed_vs_final=changed)
            print("%-4s %-10s %s  mean wait %.2f windows  changed-vs-final %.4f" % (
                split, name, " / ".join("%s %.4g" % (k, v) for k, v in met.items()), res["%s_%s" % (name, split)]
                ["mean_wait_pending"], changed), flush=True)
    with open(os.path.join(out_root, "calibrated.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("margins (calibrated): %s" % " ".join("inf" if b is None else "%.2f" % b for b in res["margins"]["calibrated"]))
    print("CALIBRATE FINISHED alpha %.3g | test IoU fixed %.4f calibrated %.4f | test mean wait fixed %.2f calibrated %.2f"
          % (args.alpha, res["fixed_test"].get("iou", float("nan")), res["calibrated_test"].get("iou", float("nan")),
             res["fixed_test"]["mean_wait_pending"], res["calibrated_test"]["mean_wait_pending"]), flush=True)


if __name__ == "__main__":
    main()
