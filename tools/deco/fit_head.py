"""DeCo step 4 (V10, 11-07): change-risk head -- when each decision neuron commits.

For a trained decision-neuron run, the head learns p_d = P(decision at step d != decision at the deadline | state at d)
from HELD-OUT trajectories of that same model (val): training-set trajectories are over-confident (in-sample change
rates are about half the held-out ones, DECO_PROBE_LOG 7.6). A pending event commits at the first step d with
p_d < alpha, else at its last available step. alpha (risk level) is the only constant; the commit boundary is learned
and tightens towards the deadline by itself (DECO_PROBE_LOG 7.7).
State at step d (causal, same meaning for every model): margin |Lambda_d - tau|, arrival margin |z0 - tau|, last step
size, largest step so far, sign flips so far, d / D (steps left = D - d: the deadline, not the recording end).
Head: MLP 6 -> 32 -> 32 -> 1, BCE on the full-horizon events (F = D) of val, steps d < D, standardised inputs.
Honest val numbers: val is split in two by recording; the head fitted on one half is applied to the other. Test: the
head fitted on all of val (saved as head.pt).
With --z0-root and --data-root: official four metrics of the policies (tools/final_analysis.py, fixed_D row of the
"frozen after commit" trajectories, as tools/deco/calibrate_commit.py). Without them (e.g. *_meta.npz summaries of
server runs, which carry the non-pending counts): pooled IoU and event recall only.
Usage: python tools/deco/fit_head.py --run <dnm run dir> --evidence <evidence prefix> [--z0-root <dir>]
       [--data-root <root>] [--alpha 0.01] [--out <run>/head]
Last line: "HEAD FINISHED alpha ... | test IoU fixed ... head ... | test mean wait fixed ... head ...".
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_commit as CC  # noqa: E402

N_STATE = 6


def logit(p):
    return math.log(p / (1.0 - p))


def load(args, split):
    """Evidence arrays (full evidence file or its *_meta.npz summary) and the run's trajectories of one split."""
    for path in ("%s_%s.npz" % (args.evidence, split), "%s_%s_meta.npz" % (args.evidence, split)):
        if os.path.exists(path):
            break
    else:
        raise FileNotFoundError("no evidence for %s: %s_%s[_meta].npz" % (split, args.evidence, split))
    with np.load(path) as m:
        ev = {k: m[k] for k in ("idx", "rec", "F", "z0", "label", "names", "log_w", "alpha", "nonpending",
                                "band_lo", "band_hi") if k in m.files}
    lt = np.load(os.path.join(args.run, "raw_%s.npz" % split))["lam_traj"].astype(np.float64)
    D = lt.shape[1] - 1
    return ev, lt, np.minimum(ev["F"], D), D


def nonpending(args, ev, split, tau):
    """TP / FP / FN of the events that keep their arrival decision (outside the pending band)."""
    if args.z0_root:
        half = logit(1.0 - float(ev["alpha"]))
        lo, hi = (float(ev["band_lo"]), float(ev["band_hi"])) if "band_lo" in ev else (tau - half, tau + half)
        c = np.zeros(3, np.int64)
        for name in ev["names"]:
            with np.load(os.path.join(args.z0_root, "z0_" + split, str(name))) as tr:
                z, y = np.asarray(tr["traj"][:, 0]), np.asarray(tr["labels"]) > 0.5
            keep = ~((z > lo) & (z < hi))
            pred, pos = (z >= tau)[keep], y[keep]
            c += [int((pred & pos).sum()), int((pred & ~pos).sum()), int((~pred & pos).sum())]
        return c
    if "nonpending" in ev:
        return np.asarray(ev["nonpending"], np.int64)
    raise ValueError("need --z0-root or an evidence summary with 'nonpending' counts")


def state_features(z0, lt, tau, D):
    """[N, D + 1, 6] causal state of every pending event at every step (see the module docstring)."""
    prev = np.concatenate([z0[:, None], lt[:, :-1]], 1)
    step = np.abs(lt - prev)                                         # |Lambda_d - Lambda_{d-1}|, Lambda_{-1} = z0
    side = lt >= tau
    first = side[:, :1] != (z0 >= tau)[:, None]
    flips = np.cumsum(np.concatenate([first, side[:, 1:] != side[:, :-1]], 1), 1)
    d = np.broadcast_to(np.arange(D + 1)[None, :], lt.shape)
    X = np.stack([np.log1p(np.abs(lt - tau)), np.broadcast_to(np.log1p(np.abs(z0 - tau))[:, None], lt.shape),
                  np.log1p(step), np.log1p(np.maximum.accumulate(step, 1)), np.log1p(flips), d / float(max(D, 1))], -1)
    return X.astype(np.float32)


class ChangeRiskHead(nn.Module):
    def __init__(self, n_in=N_STATE, width=32):
        super(ChangeRiskHead, self).__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x)[..., 0]


def fit(X, lt, F_, tau, D, rows, steps=3000, batch=4096, seed=0):
    """Fit the head on the full-horizon events in `rows`; returns (predict(X) -> p, state for saving)."""
    torch.manual_seed(seed)
    final = lt[np.arange(len(F_)), F_] >= tau
    sel = rows & (F_ == D)
    x = X[sel, :D].reshape(-1, X.shape[-1])
    y = ((lt[sel, :D] >= tau) != final[sel, None]).reshape(-1).astype(np.float32)
    if len(y) == 0:
        raise ValueError("no full-horizon training rows for the head")
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt, yt = torch.as_tensor((x - mu) / sd), torch.as_tensor(y)
    head = ChangeRiskHead(x.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        b = torch.randint(0, len(yt), (min(batch, len(yt)),))
        loss = F.binary_cross_entropy_with_logits(head(xt[b]), yt[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()

    def predict(Xq):
        with torch.no_grad():                                        # in blocks of rows (all-event exports are large)
            if not len(Xq):
                return np.zeros(Xq.shape[:-1], np.float32)
            return np.concatenate([torch.sigmoid(head(torch.as_tensor((Xq[a:a + 65536] - mu) / sd))).numpy()
                                   for a in range(0, len(Xq), 65536)])

    return predict, {"model": head.state_dict(), "mu": mu.tolist(), "sd": sd.tolist(), "rows": int(len(y)),
                     "change_rate": float(y.mean())}


def commit_by_prob(p, F_, alpha):
    """First step d < F with p_d < alpha, else F."""
    L = F_.copy()
    done = np.zeros(len(F_), bool)
    for d in range(p.shape[1]):
        hit = ~done & (d < F_) & (p[:, d] < alpha)
        L[hit] = d
        done |= hit
    return L


def summarise(lt, L, F_, y, c_np, tau):
    pred = lt[np.arange(len(L)), L] >= tau
    c = c_np + [(pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()]
    final = lt[np.arange(len(F_)), F_] >= tau
    early = L < F_
    changed = pred != final
    return {"iou": float(c[0]) / float(max(c.sum(), 1)), "recall": float(c[0]) / float(max(c[0] + c[2], 1)),
            "mean_wait_pending": float(L.mean()) if len(L) else 0.0, "changed_vs_final": float(changed.mean()),
            "early_commits": int(early.sum()), "early_change_rate": float(changed[early].mean()) if early.any() else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--z0-root", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_root = args.out or os.path.join(args.run, "head")
    os.makedirs(out_root, exist_ok=True)
    ev_v, lt_v, F_v, D = load(args, "val")
    ev_t, lt_t, F_t, _ = load(args, "test")
    tau = logit(0.9) - float(ev_v["log_w"])
    X_v = state_features(ev_v["z0"].astype(np.float64), lt_v, tau, D)
    X_t = state_features(ev_t["z0"].astype(np.float64), lt_t, tau, D)
    y_v, y_t = ev_v["label"].astype(bool), ev_t["label"].astype(bool)
    c_v, c_t = nonpending(args, ev_v, "val", tau), nonpending(args, ev_t, "test", tau)
    fixed = [logit(1.0 - args.alpha)] * (D + 1)
    res = {"alpha": args.alpha, "D": D, "val": {}, "test": {}}
    # val, honest: two halves by recording, each half committed by the head fitted on the other half
    recs = np.unique(ev_v["rec"])
    half = np.isin(ev_v["rec"], np.random.RandomState(0).permutation(recs)[: len(recs) // 2])
    L_v = F_v.copy()
    for a, b in ((half, ~half), (~half, half)):
        predict, _ = fit(X_v, lt_v, F_v, tau, D, a)
        L_v[b] = commit_by_prob(predict(X_v[b]), F_v[b], args.alpha)
    pols_v = {"fixed": CC.commit(lt_v, F_v, tau, fixed), "head": L_v}
    # test: head fitted on all of val
    predict, state = fit(X_v, lt_v, F_v, tau, D, np.ones(len(F_v), bool))
    p_t = predict(X_t)
    pols_t = {"fixed": CC.commit(lt_t, F_t, tau, fixed), "head": commit_by_prob(p_t, F_t, args.alpha)}
    torch.save(state, os.path.join(out_root, "head.pt"))
    official = bool(args.z0_root and args.data_root)
    for split, ev, lt, F_, y, c_np, pols in (("val", ev_v, lt_v, F_v, y_v, c_v, pols_v),
                                            ("test", ev_t, lt_t, F_t, y_t, c_t, pols_t)):
        for name, L in pols.items():
            r = summarise(lt, L, F_, y, c_np, tau)
            if official:
                d_ = os.path.join(out_root, "%s_%s" % (name, split))
                wms = CC.export(ev, lt, L, D, args, split, d_)
                r["official"] = CC.official(args, split, d_, wms, "%s_%s" % (name, split), D)
            res[split][name] = r
            print("%-4s %-5s IoU %.4f  recall %.4f  mean wait %.2f windows  early commits %d, changed later %.4f%s" % (
                split, name, r["iou"], r["recall"], r["mean_wait_pending"], r["early_commits"], r["early_change_rate"],
                "  official %s" % r["official"] if official else ""), flush=True)
    # calibration of p on test (full-horizon steps) and the learned boundary on a margin grid
    final_t = lt_t[np.arange(len(F_t)), F_t] >= tau
    full = F_t == D
    pp = p_t[full, :D].reshape(-1)
    ch = ((lt_t[full, :D] >= tau) != final_t[full, None]).reshape(-1)
    res["test_calibration"] = []
    for lo_, hi_ in ((0, 0.003), (0.003, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 1.01)):
        s = (pp >= lo_) & (pp < hi_)
        if s.sum():
            res["test_calibration"].append({"bin": [lo_, hi_], "mean_p": float(pp[s].mean()),
                                            "change_rate": float(ch[s].mean()), "n": int(s.sum())})
    med = np.median(X_t[full, :D].reshape(-1, N_STATE), 0) if full.any() and D > 0 else np.zeros(N_STATE, np.float32)
    res["boundary"] = {}
    for d in range(D):
        row = []
        for m in (0.5, 1.5, 2.5, 3.5, 5.0):
            x = med.copy()
            x[0], x[5] = np.log1p(m), d / float(max(D, 1))
            row.append(round(float(predict(x[None].astype(np.float32))[0]), 5))
        res["boundary"]["d%d" % d] = row
    res["head_training"] = {"rows": state["rows"], "change_rate": state["change_rate"]}
    with open(os.path.join(out_root, "head.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("HEAD FINISHED alpha %.3g | test IoU fixed %.4f head %.4f | test mean wait fixed %.2f head %.2f | "
          "val (honest halves) IoU fixed %.4f head %.4f" % (
              args.alpha, res["test"]["fixed"]["iou"], res["test"]["head"]["iou"],
              res["test"]["fixed"]["mean_wait_pending"], res["test"]["head"]["mean_wait_pending"],
              res["val"]["fixed"]["iou"], res["val"]["head"]["iou"]), flush=True)


if __name__ == "__main__":
    main()
