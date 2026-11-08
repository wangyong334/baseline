"""DeCo V11 step 4 (11-08): the decision neuron's two thresholds from its own predicted future range.

For a trained decision-neuron run, a small net predicts from the neuron's causal state at step d the alpha and
1 - alpha quantiles (q_lo, q_hi) of how much its membrane will still move before the deadline, U(D) - U(d) (pinball
loss). The neuron fires
    "target"      when U(d) >= tau and U(d) + q_lo(d) >= tau      (threshold theta+ = tau - q_lo)
    "background"  when U(d) <  tau and U(d) + q_hi(d) <  tau      (threshold theta- = tau - q_hi)
else at its last available step (at the deadline q = 0, so the rule reduces to the sign). Labels are not used: the
range is fitted on the neuron's own HELD-OUT trajectories (val), because training-set trajectories are over-confident.
The committed label is always the sign of the trained membrane at the commit step (held-out data only decides WHEN).
Regret bound: if P(overturn) <= alpha, the expected loss exceeds that of waiting to the deadline by at most c * alpha.
State at step d (neuron state variables): signed margin U - tau, signed arrival margin z0 - tau, last step, largest
step so far, sign flips so far, d / D, and for V11 runs the attributed evidence mass and effective evidence count.
Honest val numbers: two halves by recording, each committed by the range fitted on the other half; test: fitted on
all val. For comparison the same file reports the fixed band and the V10 change-risk head (tools/deco/fit_head.py).
With --z0-root and --data-root: official four metrics of every policy (fixed_D row of tools/final_analysis.py on the
'frozen after commit' trajectories).
Robustness checks (from the sequential-testing literature, Res/v11_design): a hand-tapered threshold
b(d) = b0 (1 - d / D) with b0 matched on val to the range policy's change rate (is learning needed?); spread of the
commit time; change rate by arrival margin |z0 - tau| (dependent evidence makes stopping rules fail on low-margin
events while the overall accuracy hides it); a recording-clustered 95% upper bound of the change rate (events of one
recording are not independent).
Usage: python tools/deco/fit_range.py --run <dnm run dir> --evidence <evidence prefix> [--z0-root <dir>]
       [--data-root <root>] [--alpha 0.01] [--out <run>/range]
Last line: "RANGE FINISHED alpha ... | test IoU fixed ... head ... range ... | test mean wait ...".
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_commit as CC  # noqa: E402
import fit_head as FH  # noqa: E402


def sl(x):
    return np.sign(x) * np.log1p(np.abs(x))


def range_features(z0, lt, tau, D, extras):
    """[N, D + 1, n] causal neuron state at every step."""
    prev = np.concatenate([z0[:, None], lt[:, :-1]], 1)
    step = lt - prev
    side = lt >= tau
    first = side[:, :1] != (z0 >= tau)[:, None]
    flips = np.cumsum(np.concatenate([first, side[:, 1:] != side[:, :-1]], 1), 1)
    d = np.broadcast_to(np.arange(D + 1)[None, :], lt.shape)
    cols = [sl(lt - tau), np.broadcast_to(sl(z0 - tau)[:, None], lt.shape), sl(step),
            np.log1p(np.maximum.accumulate(np.abs(step), 1)), np.log1p(flips), d / float(max(D, 1))]
    for k in ("mass", "neff"):
        if k in extras:
            cols.append(np.log1p(np.maximum(extras[k], 0.0)))
    return np.stack(cols, -1).astype(np.float32)


def fit_range(X, lt, F_, D, alpha, rows, steps=4000, seed=0):
    """Quantile net on the full-horizon events in `rows`; returns predict(X) -> (q_lo, q_hi)."""
    torch.manual_seed(seed)
    sel = rows & (F_ == D)
    x = X[sel, :D].reshape(-1, X.shape[-1])
    mv = (lt[sel, D:D + 1] - lt[sel, :D]).reshape(-1).astype(np.float32)
    if len(mv) == 0:
        raise ValueError("no full-horizon rows for the range")
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt, mt = torch.as_tensor((x - mu) / sd), torch.as_tensor(mv)
    net = nn.Sequential(nn.Linear(x.shape[1], 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    qs = torch.tensor([alpha, 1.0 - alpha])
    for _ in range(steps):
        b = torch.randint(0, len(mt), (min(8192, len(mt)),))
        e = mt[b, None] - net(xt[b])
        loss = torch.max(qs * e, (qs - 1.0) * e).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()

    def predict(Xq):
        with torch.no_grad():
            out = net(torch.as_tensor((Xq - mu) / sd)).numpy()
        lo, hi = out[..., 0].copy(), np.maximum(out[..., 1], out[..., 0])
        lo[..., D], hi[..., D] = 0.0, 0.0                           # the deadline: nothing left to come
        return lo, hi
    return predict


def commit_by_range(lt, F_, tau, q_lo, q_hi):
    """First step d < F where the predicted range of the deadline membrane stays on the current side of tau."""
    L = F_.copy()
    done = np.zeros(len(F_), bool)
    for d in range(lt.shape[1]):
        up = (lt[:, d] >= tau) & (lt[:, d] + q_lo[:, d] >= tau)
        dn = (lt[:, d] < tau) & (lt[:, d] + q_hi[:, d] < tau)
        hit = ~done & (d < F_) & (up | dn)
        L[hit] = d
        done |= hit
    return L


def coverage(lt, F_, D, q_lo, q_hi):
    full = F_ == D
    mv = lt[full, D:D + 1] - lt[full, :D]
    return {"below_q_lo": float((mv < q_lo[full, :D]).mean()) if full.any() else None,
            "above_q_hi": float((mv > q_hi[full, :D]).mean()) if full.any() else None}


def conformal_shift(predict, X, lt, F_, D, alpha, rows):
    """Split-conformal (CQR) widening of a fitted range on held-out rows: the (1 - alpha)(1 + 1/n) quantiles of how far
    the realised movement falls below q_lo / above q_hi. Restores the per-step coverage of each side (marginal)."""
    sel = rows & (F_ == D)
    lo, hi = predict(X[sel])
    mv = lt[sel, D:D + 1] - lt[sel, :D]
    e_lo, e_hi = (lo[:, :D] - mv).reshape(-1), (mv - hi[:, :D]).reshape(-1)
    lev = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / max(e_lo.size, 1)))
    return float(np.quantile(e_lo, lev)), float(np.quantile(e_hi, lev))


def widen(lo, hi, D, s_lo, s_hi):
    lo, hi = lo - s_lo, hi + s_hi
    lo[..., D], hi[..., D] = 0.0, 0.0
    return lo, hi


def commit_taper(lt, F_, tau, b0, D):
    return CC.commit(lt, F_, tau, [b0 * (1.0 - d / float(max(D, 1))) for d in range(D + 1)])


def match_taper(lt, F_, tau, D, target):
    """Smallest b0 on a grid whose change rate (vs the deadline decision) is <= target."""
    final = lt[np.arange(len(F_)), F_] >= tau
    for b0 in np.round(np.arange(0.25, 30.01, 0.25), 2):
        L = commit_taper(lt, F_, tau, b0, D)
        if ((lt[np.arange(len(L)), L] >= tau) != final).mean() <= target:
            return float(b0)
    return 30.0


def robustness(lt, L, F_, z0, rec, tau):
    final = lt[np.arange(len(F_)), F_] >= tau
    changed = (lt[np.arange(len(L)), L] >= tau) != final
    m = np.abs(z0 - tau)
    strata = {}
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 1e9)):
        s_ = (m >= lo) & (m < hi)
        if s_.any():
            strata["%g-%g" % (lo, hi if hi < 1e9 else np.inf)] = [int(s_.sum()), round(float(changed[s_].mean()), 5)]
    recs = np.unique(rec)
    n_r = np.array([(rec == r).sum() for r in recs], float)
    k_r = np.array([changed[rec == r].sum() for r in recs], float)
    rate = k_r.sum() / max(n_r.sum(), 1.0)
    G = len(recs)                                # ratio estimator, recordings as clusters
    var = ((k_r - rate * n_r) ** 2).sum() / max(n_r.sum(), 1.0) ** 2 * G / max(G - 1, 1)
    return {"commit_time_std": round(float(L.std()), 3), "commit_time_p90": float(np.quantile(L, 0.9)),
            "changed_by_arrival_margin": strata, "changed_rate": round(float(rate), 5),
            "changed_ucb95_recording_clusters": round(float(rate + 1.645 * np.sqrt(var)), 5), "recordings": int(G)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--z0-root", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_root = args.out or os.path.join(args.run, "range")
    os.makedirs(out_root, exist_ok=True)
    ev_v, lt_v, F_v, D = FH.load(args, "val")
    ev_t, lt_t, F_t, _ = FH.load(args, "test")
    ex_v = dict(np.load(os.path.join(args.run, "raw_val.npz")))
    ex_t = dict(np.load(os.path.join(args.run, "raw_test.npz")))
    tau = FH.logit(0.9) - float(ev_v["log_w"])
    z_v, z_t = ev_v["z0"].astype(np.float64), ev_t["z0"].astype(np.float64)
    X_v, X_t = range_features(z_v, lt_v, tau, D, ex_v), range_features(z_t, lt_t, tau, D, ex_t)
    S_v, S_t = FH.state_features(z_v, lt_v, tau, D), FH.state_features(z_t, lt_t, tau, D)
    y_v, y_t = ev_v["label"].astype(bool), ev_t["label"].astype(bool)
    c_v, c_t = FH.nonpending(args, ev_v, "val", tau), FH.nonpending(args, ev_t, "test", tau)
    fixed = [FH.logit(1.0 - args.alpha)] * (D + 1)
    res = {"alpha": args.alpha, "D": D, "state": X_v.shape[-1], "val": {}, "test": {}}
    recs = np.unique(ev_v["rec"])
    half = np.isin(ev_v["rec"], np.random.RandomState(0).permutation(recs)[: len(recs) // 2])
    L_head, L_range = F_v.copy(), F_v.copy()
    for a, b in ((half, ~half), (~half, half)):
        ph, _ = FH.fit(S_v, lt_v, F_v, tau, D, a)
        L_head[b] = FH.commit_by_prob(ph(S_v[b]), F_v[b], args.alpha)
        pr = fit_range(X_v, lt_v, F_v, D, args.alpha, a)
        lo, hi = pr(X_v[b])
        L_range[b] = commit_by_range(lt_v[b], F_v[b], tau, lo, hi)
    final_v = lt_v[np.arange(len(F_v)), F_v] >= tau
    L_taper = F_v.copy()
    for a, b in ((half, ~half), (~half, half)):
        target = float(((lt_v[np.arange(len(F_v)), L_range] >= tau) != final_v)[a].mean())
        b0 = match_taper(lt_v[a], F_v[a], tau, D, target)
        L_taper[b] = commit_taper(lt_v[b], F_v[b], tau, b0, D)
    pols_v = {"fixed": CC.commit(lt_v, F_v, tau, fixed), "head": L_head, "range": L_range, "taper": L_taper}
    ph, _ = FH.fit(S_v, lt_v, F_v, tau, D, np.ones(len(F_v), bool))
    pr = fit_range(X_v, lt_v, F_v, D, args.alpha, np.ones(len(F_v), bool))
    lo_t, hi_t = pr(X_t)
    pols_t = {"fixed": CC.commit(lt_t, F_t, tau, fixed), "head": FH.commit_by_prob(ph(S_t), F_t, args.alpha),
              "range": commit_by_range(lt_t, F_t, tau, lo_t, hi_t)}
    pr_a = fit_range(X_v, lt_v, F_v, D, args.alpha, half)            # CQR: fit on one val half, calibrate on the other
    s_lo, s_hi = conformal_shift(pr_a, X_v, lt_v, F_v, D, args.alpha, ~half)
    lo_c, hi_c = widen(*pr_a(X_t), D, s_lo, s_hi)
    pols_t["range_cqr"] = commit_by_range(lt_t, F_t, tau, lo_c, hi_c)
    res["cqr_shift"] = [s_lo, s_hi]
    res["test_coverage_cqr"] = coverage(lt_t, F_t, D, lo_c, hi_c)
    target_all = float(((lt_v[np.arange(len(F_v)), L_range] >= tau) != final_v).mean())
    b0_all = match_taper(lt_v, F_v, tau, D, target_all)
    pols_t["taper"] = commit_taper(lt_t, F_t, tau, b0_all, D)
    res["taper_b0"] = b0_all
    official = bool(args.z0_root and args.data_root)
    for split, ev, lt, F_, y, c_np, pols in (("val", ev_v, lt_v, F_v, y_v, c_v, pols_v),
                                            ("test", ev_t, lt_t, F_t, y_t, c_t, pols_t)):
        for name, L in pols.items():
            r = FH.summarise(lt, L, F_, y, c_np, tau)
            r["robustness"] = robustness(lt, L, F_, ev["z0"].astype(np.float64), ev["rec"], tau)
            if official:
                d_ = os.path.join(out_root, "%s_%s" % (name, split))
                wms = CC.export(ev, lt, L, D, args, split, d_)
                r["official"] = CC.official(args, split, d_, wms, "%s_%s" % (name, split), D)
            res[split][name] = r
            print("%-4s %-5s IoU %.4f  recall %.4f  mean wait %.2f windows  early commits %d, changed later %.4f%s" % (
                split, name, r["iou"], r["recall"], r["mean_wait_pending"], r["early_commits"], r["early_change_rate"],
                "  official %s" % r["official"] if official else ""), flush=True)
    res["test_coverage"] = coverage(lt_t, F_t, D, lo_t, hi_t)
    full = F_t == D
    res["thresholds_median"] = {"d%d" % d: [round(float(np.median(tau - lo_t[full, d])), 3),
                                            round(float(np.median(tau - hi_t[full, d])), 3)] for d in range(D + 1)} \
        if full.any() else {}
    res["tau"] = tau
    with open(os.path.join(out_root, "range.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    t = res["test"]
    print("RANGE FINISHED alpha %.3g | test IoU fixed %.4f head %.4f range %.4f taper %.4f | test mean wait fixed %.2f "
          "head %.2f range %.2f taper %.2f | test changed fixed %.4f head %.4f range %.4f taper %.4f | val (honest halves)"
          " IoU head %.4f range %.4f | test range_cqr IoU %.4f wait %.2f changed %.4f coverage %s" % (
              args.alpha, t["fixed"]["iou"], t["head"]["iou"], t["range"]["iou"], t["taper"]["iou"],
              t["fixed"]["mean_wait_pending"], t["head"]["mean_wait_pending"], t["range"]["mean_wait_pending"],
              t["taper"]["mean_wait_pending"], t["fixed"]["changed_vs_final"], t["head"]["changed_vs_final"],
              t["range"]["changed_vs_final"], t["taper"]["changed_vs_final"], res["val"]["head"]["iou"],
              res["val"]["range"]["iou"], t["range_cqr"]["iou"], t["range_cqr"]["mean_wait_pending"],
              t["range_cqr"]["changed_vs_final"], res["test_coverage_cqr"]), flush=True)


if __name__ == "__main__":
    main()
