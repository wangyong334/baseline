"""V17 step 2: stopping rule and evaluation of a trained decision-unit run (tools/v17/train_sdu.py output).

Calibrates the bounds of the stopping rule (tools/v17/stopping.py, Eq. 7) on val, applies them to test, and reports per
policy: official IoU / ACC / Pd / Fa at 0.9, the share of events that get a decision unit, waits and decision latencies
(publish time - event time; compute excluded), how often the committed decision differs from the deadline decision
(overall, recording-clustered 95% upper bound, by prior margin), and the decision-unit load per window (units,
multiply-accumulates, accumulates of the box sums, energy at 4.6 pJ per MAC and 0.9 pJ per AC, per 8 s of stream).
Event-driven cost: a box without events is skipped (exact with events as observations); the box counts per event and
step come from the trajectories (runs after 11-13 evening) or are recomputed from the maps (older runs, --device).
Policies, all from the same trajectories:
    arrival        the frozen backbone alone
    step0_all      every event gets a unit that commits at step 0 (its own window, no wait)
    deadline_all   every event gets a unit and waits to the deadline (accuracy reference)
    band_v10       V10/V11 rule: unit if |z0 - tau| < logit(1 - alpha), commit once |U - tau| >= logit(1 - alpha)
    entry_only     calibrated entry (column 0), then every unit waits to the deadline
    entry_wait<k>  calibrated entry, then every unit waits exactly k windows (k = 0, 1, 2, 3, 5): a uniform delay at the
                   same entry, the reference of the adaptive stopping (does deciding per event beat waiting the same?)
    calibrated     the full stopping rule (calibrated entry + calibrated bound at every step): the model
Honest val: bounds fitted on one half of the val recordings are applied to the other half and vice versa.
Usage: python tools/v17/eval_sdu.py --run <dir> [--alpha 0.01] [--delta 0.1] [--out <dir>/eval]
Last line: "SDU EVAL FINISHED alpha ... | test calibrated IoU/ACC/Pd/Fa ... units ... wait ...".
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import stopping as P  # noqa: E402
from train_sdu import build_net  # noqa: E402

PJ_PER_MAC, PJ_PER_AC = 4.6, 0.9
STREAM_MS = 8000.0                       # energy per 8 s of stream (CLAUDE.md energy convention)
FIXED_WAITS = (0, 1, 2, 3, 5)
POLICIES = ("arrival", "step0_all", "deadline_all", "band_v10", "entry_only") + tuple(
    "entry_wait%d" % k for k in FIXED_WAITS) + ("calibrated",)


def load(run, split, cfg, limit, rect=None, device=None):
    recs = C.load_split(cfg["z0_root"], cfg["data_root"], split, cfg["n_steps"], limit)
    tr = np.load(os.path.join(run, "traj_%s.npz" % split))
    if [str(n) for n in tr["names"]] != [r["name"] for r in recs]:
        raise RuntimeError("trajectories of %s do not match the recordings" % split)
    z0 = np.concatenate([r["z0"] for r in recs]).astype(np.float64)
    Z = np.concatenate([z0[:, None], tr["U"].astype(np.float64)], 1)
    rec = np.concatenate([np.full(len(r["F"]), i) for i, r in enumerate(recs)])
    if "nz" in tr.files:
        nz, ne = tr["nz"], tr["ne"]
        ne1 = tr["ne1"] if "ne1" in tr.files else None
    elif device:
        if cfg.get("track", "off") != "off":
            raise RuntimeError("runs with moving fields store their box counts in the trajectories")
        nz, ne, ne1 = C.split_counts(recs, rect, cfg["n_steps"], device)
    else:
        nz = ne = ne1 = None
    return {"recs": recs, "Z": Z, "last": 1 + tr["F"].astype(np.int64), "rec": rec, "start": tr["start"],
            "win": np.concatenate([r["win"] for r in recs]).astype(np.int64),
            "t": np.concatenate([r["t"] for r in recs]).astype(np.int64),
            "lab": np.concatenate([r["lab"] for r in recs]), "n_win": {i: r["n_win"] for i, r in enumerate(recs)},
            "wms": recs[0]["window_ms"], "nz": nz, "ne": ne, "ne1": ne1}


def step_costs(net, d, S):
    """Cost of step k of event i: multiply-accumulates (boxes without events skipped; tracked fields add their
    tracker) and accumulates of the box sums, [N, S] each; without box counts the dense multiply-accumulates (a
    function of k) and no accumulates."""
    if d["nz"] is None:
        return net.step_macs, None
    k = np.arange(S)[None, :]
    nz = d["nz"].astype(np.int64)
    aud = np.cumsum(nz, 1) if net.audit == "reaudit" else nz
    macs = np.broadcast_to(net.step_macs(k, nz, aud), nz.shape).astype(np.float64)
    if net.track or net.memb:
        if d.get("ne1") is None:
            raise RuntimeError("tracked fields / membership cue without level-0/1 event counts in the trajectories")
        macs = macs + net.track_macs(d["ne1"])
    return macs.astype(np.float32), net.step_acs(d["ne"]).astype(np.float32)


def columns_of(name, d, tau, bounds, alpha):
    Z, last = d["Z"], d["last"]
    C_ = Z.shape[1]
    if name == "arrival":
        return np.zeros(len(last), np.int64)
    if name == "step0_all":
        return np.minimum(1, last)
    if name == "deadline_all":
        return last.copy()
    if name == "band_v10":
        b = C.logit(1.0 - alpha)
        return P.commit(Z, last, tau, [b] * (C_ - 1) + [0.0])
    if name == "entry_only":
        return P.commit(Z, last, tau, [bounds[0]] + [float("inf")] * (C_ - 2) + [0.0])
    if name.startswith("entry_wait"):                     # units wait exactly k windows (or to the recording end)
        k = int(name[len("entry_wait"):])
        L = P.commit(Z, last, tau, [bounds[0]] + [float("inf")] * (C_ - 2) + [0.0])
        born = L >= 1
        L[born] = np.minimum(1 + k, last[born])
        return L
    if name == "calibrated":
        return P.commit(Z, last, tau, bounds)
    raise ValueError(name)


def cluster_ucb(changed, rec):
    recs = np.unique(rec)
    n_r = np.array([(rec == r).sum() for r in recs], float)
    k_r = np.array([changed[rec == r].sum() for r in recs], float)
    rate = k_r.sum() / max(n_r.sum(), 1.0)
    G = len(recs)
    var = ((k_r - rate * n_r) ** 2).sum() / max(n_r.sum(), 1.0) ** 2 * G / max(G - 1, 1)
    return float(rate + 1.645 * np.sqrt(var))


def summarise(d, L, tau, net, costs, n_steps):
    Z, last = d["Z"], d["last"]
    n = len(L)
    zc = Z[np.arange(n), L]
    final = Z[np.arange(n), last] >= tau
    changed = (zc >= tau) != final
    born = L >= 1
    wait = np.maximum(L - 1, 0)
    lat = (d["win"] + 1 + wait) * d["wms"] - d["t"]
    vals = [zc[a:b] for a, b in zip(d["start"][:-1], d["start"][1:])]
    out = C.official(d["recs"], vals)
    m0 = np.abs(Z[:, 0] - tau)
    strata = {}
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, np.inf)):
        s = (m0 >= lo) & (m0 < hi)
        if s.any():
            strata["%g-%g" % (lo, hi)] = [int(s.sum()), round(float(born[s].mean()), 4),
                                          round(float(changed[s].mean()), 5)]
    mac, ac = costs
    units, loads = P.unit_load(d["win"], L, d["n_win"], d["rec"], n_steps,
                               [mac, net.step_macs] + ([ac] if ac is not None else []))
    macs, dense = loads[0], loads[1]
    acs = loads[2] if ac is not None else np.zeros_like(macs)
    per_win_pj = macs.mean() * PJ_PER_MAC + acs.mean() * PJ_PER_AC
    out.update(born_frac=float(born.mean()), units_total=int(born.sum()),
               wait_windows={"all_mean": float(wait.mean()), "born_mean": float(wait[born].mean()) if born.any()
                             else 0.0, "born_p90": float(np.percentile(wait[born], 90)) if born.any() else 0.0},
               latency_ms={"all_mean": float(lat.mean()), "target_mean": float(lat[d["lab"]].mean()),
                           "target_p90": float(np.percentile(lat[d["lab"]], 90))},
               changed={"all": float(changed.mean()), "born": float(changed[born].mean()) if born.any() else 0.0,
                        "ucb95_recordings": cluster_ucb(changed, d["rec"]), "by_prior_margin": strata},
               load={"units_per_window": {"mean": float(units.mean()), "p95": float(np.percentile(units, 95)),
                                          "max": int(units.max())},
                     "event_driven": ac is not None,
                     "mmac_per_window": {"mean": float(macs.mean()) / 1e6,
                                         "p95": float(np.percentile(macs, 95)) / 1e6, "max": float(macs.max()) / 1e6},
                     "mmac_per_window_dense": float(dense.mean()) / 1e6,
                     "kac_per_window": float(acs.mean()) / 1e3,
                     "energy_uJ_per_window": float(per_win_pj) / 1e6,
                     "energy_mJ_per_8s": float(per_win_pj) * STREAM_MS / d["wms"] / 1e9})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--alpha", type=float, default=0.01, help="declared risk: P(decision changes before deadline)")
    ap.add_argument("--delta", type=float, default=0.1, help="confidence budget of the calibration (Bonferroni)")
    ap.add_argument("--no-collapse", action="store_true", help="bounds per step without the collapse envelope")
    ap.add_argument("--stop", choices=("window", "isotonic"), default="window",
                    help="how a bound is certified: by the events just below it (window, V18: robust to a rare late "
                         "flip of a confident unit) or by every isotonic block above it (isotonic, V17)")
    ap.add_argument("--window-factor", type=int, default=4,
                    help="window rule: events per certification window = factor x the fewest that certify alpha "
                         "without any change (sensitivity of the one hand-set constant)")
    ap.add_argument("--z0-root", default=None, help="override the prior dir stored in the run (another machine)")
    ap.add_argument("--data-root", default=None, help="override the data root stored in the run")
    ap.add_argument("--device", default="cpu",
                    help="recompute the box counts on this device when the trajectories have none (older runs); "
                         "'none' = report dense multiply-accumulates only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ck = torch.load(os.path.join(args.run, "sdu.pt"), map_location="cpu")
    cfg = C.local_paths(ck["config"], args.z0_root, args.data_root)
    collapse = not args.no_collapse
    limit = int(ck["args"].get("limit", 0))
    out_root = args.out or os.path.join(args.run, "eval")
    os.makedirs(out_root, exist_ok=True)
    net, rect = build_net(cfg, torch.device("cpu"))
    tau, S = cfg["tau"], cfg["n_steps"]
    dev = None if args.device == "none" else args.device
    val, test = (load(args.run, s, cfg, limit, rect, dev) for s in ("val", "test"))
    costs = {"val": step_costs(net, val, S), "test": step_costs(net, test, S)}
    bounds = P.calibrate(val["Z"], val["last"], tau, args.alpha, args.delta, collapse, args.stop, args.window_factor)
    res = {"alpha": args.alpha, "delta": args.delta, "collapse": collapse, "stop": args.stop,
           "window_factor": args.window_factor, "tau": tau,
           "bounds": [None if not np.isfinite(b) else round(b, 4) for b in bounds],
           "val": {}, "test": {}, "macs_per_step": [net.step_macs(k) for k in range(S)]}
    recs_v = np.unique(val["rec"])
    half = np.isin(val["rec"], np.random.RandomState(0).permutation(recs_v)[: len(recs_v) // 2])
    honest = {}
    for name in ("entry_only", "calibrated") + tuple("entry_wait%d" % k for k in FIXED_WAITS):
        L = np.zeros(len(val["last"]), np.int64)
        for a, b in ((half, ~half), (~half, half)):
            sub = {k: val[k][a] for k in ("Z", "last")}
            bd = P.calibrate(sub["Z"], sub["last"], tau, args.alpha, args.delta, collapse, args.stop,
                             args.window_factor)
            L[b] = columns_of(name, {"Z": val["Z"][b], "last": val["last"][b]}, tau, bd, args.alpha)
        honest[name] = L
    for split, d in (("val", val), ("test", test)):
        for name in POLICIES:
            L = honest[name] if (split == "val" and name in honest) else columns_of(name, d, tau, bounds, args.alpha)
            r = summarise(d, L, tau, net, costs[split], S)
            res[split][name] = r
            ld = r["load"]
            print("%-4s %-12s IoU %.4f ACC %.4f Pd %.4f Fa %.3e | units %.2f%% wait %.2f win (born %.2f) | changed "
                  "%.5f (ucb %.5f) | units/window mean %.1f max %d, %.3f MMAC/window (dense %.2f, max %.1f), %.1f kAC "
                  "| %.3f mJ / 8 s" % (
                      split, name, r["iou"], r["acc"], r["pd"], r["fa"], 100 * r["born_frac"],
                      r["wait_windows"]["all_mean"], r["wait_windows"]["born_mean"], r["changed"]["all"],
                      r["changed"]["ucb95_recordings"], ld["units_per_window"]["mean"], ld["units_per_window"]["max"],
                      ld["mmac_per_window"]["mean"], ld["mmac_per_window_dense"], ld["mmac_per_window"]["max"],
                      ld["kac_per_window"], ld["energy_mJ_per_8s"]), flush=True)
    print("bounds (calibrated on val): %s" % " ".join("inf" if b is None else "%.2f" % b for b in res["bounds"]))
    t = res["test"]["calibrated"]
    e_all, e_first = t["load"]["energy_mJ_per_8s"], 0.0
    if cfg.get("stage1"):                                     # V18: the first population runs as well
        with open(os.path.join(cfg["stage1"], "eval", "eval.json")) as fh:
            e_first = json.load(fh)["test"]["calibrated"]["load"].get("energy_mJ_per_8s", 0.0)
        e_all += e_first
        res["energy_mJ_per_8s_both_populations"] = e_all
    with open(os.path.join(out_root, "eval.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("SDU EVAL FINISHED alpha %.3g | test calibrated IoU %.4f ACC %.4f Pd %.4f Fa %.3e | units %.2f%% of events, "
          "mean wait %.2f windows (born %.2f) | changed %.5f | %.3f MMAC/window, first population %.3f mJ, %.3f mJ / 8 s "
          "| val (honest) IoU %.4f"
          % (args.alpha, t["iou"], t["acc"], t["pd"], t["fa"], 100 * t["born_frac"], t["wait_windows"]["all_mean"],
             t["wait_windows"]["born_mean"], t["changed"]["all"], t["load"]["mmac_per_window"]["mean"], e_first,
             e_all, res["val"]["calibrated"]["iou"]), flush=True)


if __name__ == "__main__":
    main()
