"""DeCo V13 stages 3-4 (11-10): fit the Value-of-Waiting unit on held-out trajectories, evaluate the whole policy.

Input: a V13 SDU run (tools/deco/train_dnm.py --design v11 on evidence of the wide training band,
tools/deco/evidence.py --band martingale --alpha 0.001 --radius 0), its evidence prefix, z0 root and data root.
Stage 3 (val recordings A, no labels): VoW gate (k = 0 tap, the event's past in the event memory) on U(F) - z0 and
VoW commit (k >= 1 tap, the SDU state) on U(D) - U(k), pinball at alpha / 0.5 / 1 - alpha (tools/deco/vow.py).
Stage 4 (val recordings B): split-conformal (CQR) shifts. Test: heads from A, shifts from B. Val: two folds A -> B and
B -> A without CQR (honest, the same protocol as tools/deco/fit_range.py).
Policies (birth rule + commit rule; the label is always the sign of U - tau at the commit step, or of z0 - tau without
an SDU):
    arrival   no SDU                                                     (lower reference)
    sym       V11 birth rule |z0 - tau| < logit(1 - alpha)               + VoW commit
    ville     martingale band at alpha                                   + VoW commit
    outer     every event in the training band (martingale, alpha_train) + VoW commit
    vow       VoW gate (past traces)                                     + VoW commit      <- V13
    vow_z0    VoW gate without traces (margin, polarity)                 + VoW commit      (ablation)
    margin_m  |z0 - tau| < t, t matching the births of vow               + VoW commit      (ablation)
    wait_all  every event in the training band, commit at the deadline                     (upper reference)
Reported per policy: official IoU / ACC / Pd / Fa (tools/final_analysis.py, fixed_D row), births (share of all events),
mean wait over all events (ms) and over the spawned ones (windows), overturn of the gate (no SDU although the SDU's
deadline decision differs) and of early commits, recording-clustered 95% upper bounds. Interpretability: learned
memory of the temporal BIR, predicted range / drift per context group against the realised movement, coverage, births
by arrival margin, three example events for the figure.
Pareto (test, official four metrics): the gate at risk levels 0.002 ... 0.2 (the same heads, CQR at each level)
against the distance-only gate with the same number of births and the gate without traces. Commit risk per side (test,
official four metrics): stopping as background at a_bg (a missed target costs Pd), as target at a_tg (a false alarm
costs Fa), a_bg / a_tg in {alpha, 0.003, 0.001}. --fast: both sweeps with pooled metrics only. Per-event test arrays for the figures:
<out>/vow_test_events.npz (margin, calibrated range, U(0), U(F), commit step, overturn, label, births, groups).
Usage: python tools/deco/fit_vow.py --run <SDU run> --evidence <prefix> --z0-root <dir> --data-root <root>
       [--alpha 0.01] [--sym-alpha 0.01] [--out <run>/vow] [--steps 4000]
Last line: "VOW FINISHED ...".
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_commit as CC  # noqa: E402
import fit_head as FH  # noqa: E402
import fit_range as FR  # noqa: E402
import vow as VW  # noqa: E402


def load_split(args, split):
    ev, lt, F_, D = FH.load(args, split)
    path = "%s_%s.npz" % (args.evidence, split)
    with np.load(path) as m:
        past, k_near = int(m["past"]), int(m["k_near"])
        n = len(m["idx"])
        tok, val = m["tok"], m["tok_valid"]
        tr = np.concatenate([VW.trace_features(tok[i:i + 50000], val[i:i + 50000], m["nall"][i:i + 50000],
                                               m["nevid"][i:i + 50000], past, k_near) for i in range(0, n, 50000)])
        info = {"p": m["p"].astype(np.float32), "n_events": int(m["n_events"]) if "n_events" in m.files else 0,
                "window_ms": int(m["window_ms"]), "past": past}
    ex = dict(np.load(os.path.join(args.run, "raw_%s.npz" % split)))
    return ev, lt, F_, D, tr, info, ex


def zfit(x, rows):
    flat = x[rows].reshape(-1, x.shape[-1])
    return flat.mean(0), flat.std(0) + 1e-6


def fit_gate(tr, m0, p, y, rows, alpha, steps, dev, traces=True, seed=0):
    """Fit on the rows of one split; returns (predict(trace, m0, p) for any split, net)."""
    mu, sd = zfit(tr, rows)
    net = VW.VoWGate(traces=traces)
    trn = ((tr[rows] - mu) / sd).astype(np.float32)
    VW.fit(net, [trn, m0[rows], p[rows]], y[rows], alpha, steps=steps, seed=seed, device=dev)

    def pred(tr_q, m_q, p_q):
        return VW.predict(net, [((tr_q - mu) / sd).astype(np.float32), m_q, p_q], device=dev)
    return pred, net


def fit_commit(X, lt, F_, D, rows, alpha, steps, dev, seed=0):
    """Fit on the full-horizon rows of one split; returns predict(X) for any split."""
    sel = rows & (F_ == D)
    x = X[sel, :D].reshape(-1, X.shape[-1])
    mv = (lt[sel, D:D + 1] - lt[sel, :D]).reshape(-1)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    net = VW.VoWCommit(X.shape[-1])
    VW.fit(net, [((x - mu) / sd).astype(np.float32)], mv, alpha, steps=steps, seed=seed, device=dev)

    def pred(Xq):
        xi = ((Xq - mu) / sd).astype(np.float32)
        lo, md, hi = VW.predict(net, [xi.reshape(-1, Xq.shape[-1])], device=dev)
        lo, md, hi = (a.reshape(len(Xq), D + 1) for a in (lo, md, hi))
        lo[:, D], md[:, D], hi[:, D] = 0.0, 0.0, 0.0                    # the deadline: nothing left to come
        return lo, md, hi
    return pred


class Calibrated:
    """A VoW head fitted on some recordings plus its raw ranges on held-out calibration recordings: split-conformal
    shifts at any risk level per side (the lower quantile decides stopping as target, the upper one as background)."""

    def __init__(self, pred, lo_c, hi_c, y_c):
        self.pred, self.lo_c, self.hi_c, self.y_c = pred, lo_c, hi_c, y_c

    def shifts(self, a_bg, a_tg):
        return (VW.cqr_shift(self.lo_c, self.hi_c, self.y_c, a_tg)[0],
                VW.cqr_shift(self.lo_c, self.hi_c, self.y_c, a_bg)[1])


def gate_cal(tr, m0, p, y, fit_rows, cal_rows, alpha, steps, dev, traces=True):
    g, net = fit_gate(tr, m0, p, y, fit_rows, alpha, steps, dev, traces=traces)
    ic = np.nonzero(cal_rows)[0]
    lo, _, hi = g(tr[ic], m0[ic], p[ic])
    return Calibrated(g, lo, hi, y[ic]), net


def commit_cal(X, lt, F_, D, fit_rows, cal_rows, alpha, steps, dev):
    pc = fit_commit(X, lt, F_, D, fit_rows, alpha, steps, dev)
    idx = np.nonzero(cal_rows & (F_ == D))[0]
    lo, _, hi = pc(X[idx])
    mv = lt[idx, D:D + 1] - lt[idx, :D]
    return Calibrated(pc, lo[:, :D].reshape(-1), hi[:, :D].reshape(-1), mv.reshape(-1))


def gate_apply(cg, a_bg, a_tg, tr_q, m_q, p_q, z_q, tau):
    s_lo, s_hi = cg.shifts(a_bg, a_tg)
    lo, md, hi = cg.pred(tr_q, m_q, p_q)
    lo, hi = lo - s_lo, hi + s_hi
    return VW.stop_now(z_q, tau, lo, hi), lo, md, hi


def commit_apply(cc, a_bg, a_tg, Xq, ltq, Fq, D, tau):
    s_lo, s_hi = cc.shifts(a_bg, a_tg)
    lo, md, hi = cc.pred(Xq)
    lo, hi = widen(lo, hi, D, s_lo, s_hi)
    return FR.commit_by_range(ltq, Fq, tau, lo, hi), lo, md, hi


def split_recordings(rec, rows, frac, seed):
    """Split the recordings of `rows` into a fit part (share `frac`) and a calibration part."""
    r = np.random.RandomState(seed).permutation(np.unique(rec[rows]))
    k = min(max(1, int(round(len(r) * frac))), len(r) - 1)
    return rows & np.isin(rec, r[:k]), rows & np.isin(rec, r[k:])


def widen(lo, hi, D, s_lo, s_hi):
    lo, hi = lo - s_lo, hi + s_hi
    if lo.ndim == 2:
        lo[:, D], hi[:, D] = 0.0, 0.0
    return lo, hi


def cluster_ucb(flag, rec, rows):
    recs = np.unique(rec[rows])
    if not len(recs) or not rows.any():
        return None
    n_r = np.array([(rows & (rec == r)).sum() for r in recs], float)
    k_r = np.array([(flag & rows & (rec == r)).sum() for r in recs], float)
    rate = k_r.sum() / max(n_r.sum(), 1.0)
    G = len(recs)
    var = ((k_r - rate * n_r) ** 2).sum() / max(n_r.sum(), 1.0) ** 2 * G / max(G - 1, 1)
    return round(float(rate + 1.645 * np.sqrt(var)), 5)


def evaluate(args, split, name, ev, lt, F_, D, z0, tau, S, L, c_np, info, out_root, official=True, quiet=False):
    n = len(z0)
    i = np.arange(n)
    lt_p = lt.copy()
    lt_p[~S] = z0[~S, None]
    L_p = np.where(S, L, 0)
    pred = lt_p[i, L_p] >= tau
    final = lt[i, F_] >= tau
    y = ev["label"].astype(bool)
    c = c_np + [(pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()]
    n_all = info["n_events"] or n
    gate_rows, early = ~S, S & (L_p < F_)
    changed = pred != final
    r = {"iou_pooled": float(c[0]) / float(max(c.sum(), 1)), "births": float(S.sum()) / n_all,
         "spawned": int(S.sum()), "mean_wait_ms_all_events": float(L_p[S].sum()) * info["window_ms"] / n_all,
         "mean_wait_windows_spawned": float(L_p[S].mean()) if S.any() else 0.0,
         "gate_overturn": float(changed[gate_rows].mean()) if gate_rows.any() else 0.0,
         "gate_overturn_ucb95": cluster_ucb(changed, ev["rec"], gate_rows),
         "commit_overturn": float(changed[early].mean()) if early.any() else 0.0,
         "commit_overturn_ucb95": cluster_ucb(changed, ev["rec"], early),
         "changed_vs_deadline": float(changed.mean())}
    if official and args.z0_root and args.data_root:
        d_ = os.path.join(out_root, "%s_%s" % (name, split))
        wms = CC.export(ev, lt_p, L_p, D, args, split, d_)
        r["official"] = CC.official(args, split, d_, wms, "%s_%s" % (name, split), D)
        if not args.keep_traj:                                           # json / md / log stay next to it
            shutil.rmtree(d_, ignore_errors=True)
    if quiet:
        return r
    o = r.get("official", {})
    print("%-4s %-8s IoU %.4f ACC %.4f Pd %.4f Fa %.3g | births %.4f%%, wait %.2f ms/event (%.2f windows/spawned) | "
          "gate overturn %.4f (ucb %s), commit overturn %.4f (ucb %s)" % (
              split, name, o.get("iou", r["iou_pooled"]), o.get("acc", float("nan")), o.get("pd", float("nan")),
              o.get("fa", float("nan")), 100 * r["births"], r["mean_wait_ms_all_events"],
              r["mean_wait_windows_spawned"], r["gate_overturn"], r["gate_overturn_ucb95"], r["commit_overturn"],
              r["commit_overturn_ucb95"]), flush=True)
    return r


def policies(z0, tau, q, alpha, sym_alpha, F_, stop_vow, stop_z0, L):
    m = np.abs(z0 - tau)
    S = {"arrival": np.zeros(len(z0), bool),
         "sym": m < FH.logit(1.0 - sym_alpha),
         "ville": (z0 > FH.logit(alpha * q)) & (z0 < FH.logit(1.0 - alpha * (1.0 - q))),
         "outer": np.ones(len(z0), bool),
         "vow": ~stop_vow,
         "vow_z0": ~stop_z0}
    k = int(S["vow"].sum())
    S["margin_m"] = np.zeros(len(z0), bool)
    if k:
        S["margin_m"][np.argsort(m, kind="stable")[:k]] = True
    Ls = {nm: L for nm in S}
    S["wait_all"], Ls["wait_all"] = np.ones(len(z0), bool), F_.copy()
    return S, Ls


def groups(tr, z0, tau, past):
    """Context groups from the past traces: kd = distance of the farthest picked neighbour per window."""
    kd = np.expm1(tr[..., 1])
    act = kd <= 32.0
    pst = act[:, :past].sum(1)
    now = act[:, past]
    side = z0 >= tau
    kd0 = kd[:, past]
    return {"onset (past <= 2 active, now active)": (pst <= 2) & now, "persistent (>= 8 of 10 active)": pst >= 8,
            "background side": ~side, "target side": side, "dense (kd0 < 4 px)": kd0 < 4,
            "medium (4-16 px)": (kd0 >= 4) & (kd0 < 16), "sparse (16-64 px)": (kd0 >= 16) & (kd0 < 64),
            "isolated (>= 64 px)": kd0 >= 64}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--z0-root", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--sym-alpha", type=float, default=0.01)
    ap.add_argument("--alpha-bg", type=float, default=None,
                    help="risk of stopping as background (a missed target costs Pd); default --alpha")
    ap.add_argument("--alpha-tg", type=float, default=None,
                    help="risk of stopping as target (a false alarm costs Fa); default --alpha")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-traj", action="store_true", help="keep the exported per-policy trajectories")
    ap.add_argument("--fast", action="store_true", help="Pareto / commit-side sweeps with pooled metrics only")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_root = args.out or os.path.join(args.run, "vow")
    os.makedirs(out_root, exist_ok=True)
    dev = args.device
    ev_v, lt_v, F_v, D, tr_v, inf_v, ex_v = load_split(args, "val")
    ev_t, lt_t, F_t, _, tr_t, inf_t, ex_t = load_split(args, "test")
    tau = FH.logit(0.9) - float(ev_v["log_w"])
    q = 1.0 / (1.0 + np.exp(-tau))
    z_v, z_t = ev_v["z0"].astype(np.float64), ev_t["z0"].astype(np.float64)
    m_v, m_t = (z_v - tau).astype(np.float32), (z_t - tau).astype(np.float32)
    X_v, X_t = FR.range_features(z_v, lt_v, tau, D, ex_v), FR.range_features(z_t, lt_t, tau, D, ex_t)
    d_v = lt_v[np.arange(len(F_v)), F_v] - z_v                      # gate target: movement from the prior
    d_t = lt_t[np.arange(len(F_t)), F_t] - z_t
    c_v, c_t = FH.nonpending(args, ev_v, "val", tau), FH.nonpending(args, ev_t, "test", tau)
    a_bg = args.alpha if args.alpha_bg is None else args.alpha_bg
    a_tg = args.alpha if args.alpha_tg is None else args.alpha_tg
    rec_v = ev_v["rec"]
    recs = np.unique(rec_v)
    A = np.isin(rec_v, np.random.RandomState(0).permutation(recs)[: len(recs) // 2])
    print("V13 VoW: tau %.3f, D %d, val %d / test %d events in the training band (val A %d), heads at alpha %.3g, risk "
          "per side: background %.3g, target %.3g, device %s" % (tau, D, len(z_v), len(z_t), int(A.sum()), args.alpha,
                                                                 a_bg, a_tg, dev), flush=True)
    res = {"alpha": args.alpha, "alpha_bg": a_bg, "alpha_tg": a_tg, "sym_alpha": args.sym_alpha, "tau": tau, "D": D,
           "val": {}, "test": {}}
    p_v, p_t = inf_v["p"], inf_t["p"]

    # ---- val: two folds by recording; in each, the heads are fitted on 2/3 of the training half's recordings and
    # calibrated (CQR) on the other 1/3, then applied to the held-out half (the same protocol as test)
    stop_v, stopz_v = np.zeros(len(z_v), bool), np.zeros(len(z_v), bool)
    L_v = F_v.copy()
    for a, b in ((A, ~A), (~A, A)):
        ib = np.nonzero(b)[0]
        fit_r, cal_r = split_recordings(rec_v, a, 2.0 / 3.0, seed=1)
        cgv, _ = gate_cal(tr_v, m_v, p_v, d_v, fit_r, cal_r, args.alpha, args.steps, dev)
        stop_v[ib] = gate_apply(cgv, a_bg, a_tg, tr_v[ib], m_v[ib], p_v[ib], z_v[ib], tau)[0]
        czv, _ = gate_cal(tr_v, m_v, p_v, d_v, fit_r, cal_r, args.alpha, args.steps, dev, traces=False)
        stopz_v[ib] = gate_apply(czv, a_bg, a_tg, tr_v[ib], m_v[ib], p_v[ib], z_v[ib], tau)[0]
        ccv = commit_cal(X_v, lt_v, F_v, D, fit_r, cal_r, args.alpha, args.steps, dev)
        L_v[ib] = commit_apply(ccv, a_bg, a_tg, X_v[ib], lt_v[ib], F_v[ib], D, tau)[0]
    S_v, Ls_v = policies(z_v, tau, q, args.alpha, args.sym_alpha, F_v, stop_v, stopz_v, L_v)
    for nm in S_v:
        res["val"][nm] = evaluate(args, "val", nm, ev_v, lt_v, F_v, D, z_v, tau, S_v[nm], Ls_v[nm], c_v, inf_v,
                                  out_root)

    # ---- test: heads on val A, CQR on val B
    cg, gate_net = gate_cal(tr_v, m_v, p_v, d_v, A, ~A, args.alpha, args.steps, dev)
    stop_t, lo0, md0, hi0 = gate_apply(cg, a_bg, a_tg, tr_t, m_t, p_t, z_t, tau)
    cz, _ = gate_cal(tr_v, m_v, p_v, d_v, A, ~A, args.alpha, args.steps, dev, traces=False)
    stopz_t = gate_apply(cz, a_bg, a_tg, tr_t, m_t, p_t, z_t, tau)[0]
    cc = commit_cal(X_v, lt_v, F_v, D, A, ~A, args.alpha, args.steps, dev)
    L_t, lok, mdk, hik = commit_apply(cc, a_bg, a_tg, X_t, lt_t, F_t, D, tau)
    res["cqr_shift"] = {"gate": list(cg.shifts(a_bg, a_tg)), "gate_z0": list(cz.shifts(a_bg, a_tg)),
                        "commit": list(cc.shifts(a_bg, a_tg))}
    S_t, Ls_t = policies(z_t, tau, q, args.alpha, args.sym_alpha, F_t, stop_t, stopz_t, L_t)
    for nm in S_t:
        res["test"][nm] = evaluate(args, "test", nm, ev_t, lt_t, F_t, D, z_t, tau, S_t[nm], Ls_t[nm], c_t, inf_t,
                                   out_root)

    # ---- Pareto over the gate's risk level (both sides, the same heads, CQR at each level, commit unchanged) against
    # the distance-only gate with the same births and the gate without traces; official four metrics at every point
    par = []
    order_m = np.argsort(np.abs(m_t), kind="stable")
    keys = ("iou_pooled", "births", "mean_wait_ms_all_events", "gate_overturn")

    def four(r):
        o = r.get("official", {})
        out = {k: round(float(r[k]), 6) for k in keys}
        out.update({k: o[k] for k in ("iou", "acc", "pd", "fa") if k in o})
        return out
    for a2 in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        Sv = ~gate_apply(cg, a2, a2, tr_t, m_t, p_t, z_t, tau)[0]
        Sz = ~gate_apply(cz, a2, a2, tr_t, m_t, p_t, z_t, tau)[0]
        Sm = np.zeros(len(z_t), bool)
        Sm[order_m[:int(Sv.sum())]] = True
        row = {"alpha_gate": a2}
        for nm, Sx in (("vow", Sv), ("vow_z0", Sz), ("margin_matched", Sm)):
            r = evaluate(args, "test", "pareto%g_%s" % (a2, nm), ev_t, lt_t, F_t, D, z_t, tau, Sx, L_t, c_t, inf_t,
                         out_root, official=not args.fast, quiet=True)
            row[nm] = four(r)
        par.append(row)
        print("pareto alpha %.3f | births vow %.3f%% / vow_z0 %.3f%% | IoU ACC Pd Fa: vow %s | distance-only, same births "
              "%s | vow_z0 %s" % (a2, 100 * row["vow"]["births"], 100 * row["vow_z0"]["births"],
                                  *[" ".join(str(row[nm].get(k, "-")) for k in ("iou", "acc", "pd", "fa"))
                                    for nm in ("vow", "margin_matched", "vow_z0")]), flush=True)
    res["pareto_test"] = par

    # ---- commit risk per side (VoW gate, official four metrics): stopping as background at s_bg, as target at s_tg
    side = []
    for s_bg, s_tg in ((args.alpha, args.alpha), (0.003, args.alpha), (0.001, args.alpha), (args.alpha, 0.003),
                       (0.003, 0.003), (a_bg, a_tg)):
        L_s = commit_apply(cc, s_bg, s_tg, X_t, lt_t, F_t, D, tau)[0]
        r = evaluate(args, "test", "commit_bg%g_tg%g" % (s_bg, s_tg), ev_t, lt_t, F_t, D, z_t, tau, S_t["vow"], L_s,
                     c_t, inf_t, out_root, official=not args.fast)
        side.append({"alpha_bg": s_bg, "alpha_tg": s_tg, **four(r), "commit_overturn": r["commit_overturn"]})
    res["commit_sides_test"] = side

    # ---- interpretability (test)
    full = F_t == D
    mv_k = lt_t[full, D:D + 1] - lt_t[full, :D]
    res["coverage_test"] = {"gate_below_q_lo": float((d_t < lo0).mean()), "gate_above_q_hi": float((d_t > hi0).mean()),
                            "commit_below_q_lo": float((mv_k < lok[full, :D]).mean()) if full.any() else None,
                            "commit_above_q_hi": float((mv_k > hik[full, :D]).mean()) if full.any() else None}
    bir = gate_net.bir
    res["temporal_bir"] = {"memory_windows": [round(float(x), 2) for x in bir.memory_windows()],
                           "reference_k": [round(float(x), 3) for x in
                                           torch.nn.functional.softplus(bir.k_raw).detach().cpu().tolist()],
                           "step_theta": [round(float(x), 3) for x in torch.exp(bir.log_theta).detach().cpu().tolist()]}
    flip = (z_t >= tau) != (lt_t[np.arange(len(F_t)), F_t] >= tau)
    width = hi0 - lo0
    grp = {}
    gmask = groups(tr_t, z_t, tau, inf_t["past"])
    for gname, gm in gmask.items():
        if gm.sum() < 50:
            continue
        grp[gname] = {"n": int(gm.sum()), "pred_width": round(float(width[gm].mean()), 3),
                      "pred_drift_q50": round(float(md0[gm].mean()), 3),
                      "real_abs_move": round(float(np.abs(d_t[gm]).mean()), 3),
                      "real_flip_rate": round(float(flip[gm].mean()), 5),
                      "spawn_vow": round(float(S_t["vow"][gm].mean()), 4), "spawn_sym": round(float(S_t["sym"][gm].mean()), 4)}
    res["groups_test"] = grp
    try:
        from scipy.stats import spearmanr
        res["spearman_width_vs_abs_move"] = round(float(spearmanr(width, np.abs(d_t)).correlation), 4)
    except Exception:                                                 # noqa: BLE001
        res["spearman_width_vs_abs_move"] = None
    mb = {}
    m_abs = np.abs(m_t)
    for lo_, hi_ in ((0, 2), (2, 4.6), (4.6, 5.6), (5.6, 6.6), (6.6, 1e9)):
        sel = (m_abs >= lo_) & (m_abs < hi_)
        if sel.any():
            mb["%g-%g" % (lo_, hi_)] = {"n": int(sel.sum()), "spawn_vow": round(float(S_t["vow"][sel].mean()), 4),
                                        "spawn_sym": round(float(S_t["sym"][sel].mean()), 4),
                                        "flip": round(float(flip[sel].mean()), 5)}
    res["births_by_margin_test"] = mb
    ex = {}
    pick = {"waits, V11 does not": S_t["vow"] & ~S_t["sym"] & flip, "stops, V11 waits": ~S_t["vow"] & S_t["sym"] & ~flip,
            "both stop": ~S_t["vow"] & ~S_t["sym"]}
    for nm, sel in pick.items():
        idx = np.nonzero(sel)[0]
        if len(idx):
            j = int(idx[np.argsort(np.abs(m_t[idx] - np.median(m_t[idx])))[0]])
            ex[nm] = {"m0": round(float(m_t[j]), 3), "q_lo": round(float(lo0[j]), 3), "q50": round(float(md0[j]), 3),
                      "q_hi": round(float(hi0[j]), 3), "U_deadline_minus_tau": round(float(lt_t[j, F_t[j]] - tau), 3)}
    res["examples_test"] = ex
    torch.save({"gate": gate_net.state_dict(), "alpha": args.alpha, "cqr": res["cqr_shift"]},
               os.path.join(out_root, "vow_heads.pt"))
    i_t = np.arange(len(F_t))                                         # per-event test arrays for the figures
    np.savez_compressed(os.path.join(out_root, "vow_test_events.npz"), m0=m_t.astype(np.float16),
                        q_lo=lo0.astype(np.float16), q50=md0.astype(np.float16), q_hi=hi0.astype(np.float16),
                        U0=(lt_t[:, 0] - tau).astype(np.float16), UF=(lt_t[i_t, F_t] - tau).astype(np.float16),
                        L=L_t.astype(np.int8), F=F_t.astype(np.int8), flip=flip, label=ev_t["label"].astype(bool),
                        spawn_vow=S_t["vow"], spawn_sym=S_t["sym"], rec=ev_t["rec"].astype(np.int16),
                        onset=gmask["onset (past <= 2 active, now active)"],
                        persistent=gmask["persistent (>= 8 of 10 active)"])
    with open(os.path.join(out_root, "vow.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    t = res["test"]

    def iou(nm):
        return t[nm].get("official", {}).get("iou", t[nm]["iou_pooled"])
    print("VOW FINISHED alpha %.3g | test IoU arrival %.4f sym %.4f ville %.4f outer %.4f vow %.4f vow_z0 %.4f margin_m "
          "%.4f wait_all %.4f | births %% sym %.3f vow %.3f outer %.3f | wait ms/event sym %.2f vow %.2f | gate "
          "overturn sym %.4f vow %.4f | coverage %s | spearman %s" % (
              args.alpha, iou("arrival"), iou("sym"), iou("ville"), iou("outer"), iou("vow"), iou("vow_z0"),
              iou("margin_m"), iou("wait_all"), 100 * t["sym"]["births"], 100 * t["vow"]["births"],
              100 * t["outer"]["births"], t["sym"]["mean_wait_ms_all_events"], t["vow"]["mean_wait_ms_all_events"],
              t["sym"]["gate_overturn"], t["vow"]["gate_overturn"], res["coverage_test"],
              res["spearman_width_vs_abs_move"]), flush=True)


if __name__ == "__main__":
    main()
