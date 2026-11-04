"""Calibrated commit interface for module three (V9 optimisation 1 / D19; evaluation only, no retraining).

The adaptive commit is a Bayes-risk rule and needs a calibrated posterior: commit 1 at P >= sigma(logit tau* +
logit(1 - alpha)), commit 0 when the provisional label is 1 and P <= sigma(logit tau* - logit(1 - alpha)), else wait
(at most D). On EV-Flying the backbone's posterior never reaches the upper bound, so every event waits D (D19). The
interface between the head and the commit rule is therefore calibrated on val with one temperature anchored at the
decision threshold:
    c(z) = tau + (z - tau) / T          z = calibrated logit at any lag, tau = logit tau*
Standard temperature scaling keeps the arg-max; this one keeps every thresholded decision (c(z) >= tau iff z >= tau),
so only the confidence changes. T is fitted by NLL on val over every (event, lag) value of the trajectories. The commit
test uses c(z); the committed value is still the head's own value at the commit lag (official threshold), so the
calibration changes when an event is committed, never what the head outputs at a given lag.
With one anchored temperature the calibrated commit at alpha is exactly the uncalibrated commit at alpha' with
logit(1 - alpha') = T logit(1 - alpha) (both bounds move by the same factor). So the calibration does not move the
accuracy-latency frontier; it maps alpha to the risk it states, the same on every dataset (EV-UAV val: T = 1.57,
calibrated alpha 0.05 = uncalibrated alpha 0.0098, test 0.8929 at 141 ms for both).
Input: trajectories of `train_track_head.py --mode eval --trajectory-dir` (val for the fit, test reported once).
Usage:
    python tools/commit_calibration.py --val-traj run/final/traj/<run>_seed37_val --test-traj run/final/traj/<run>_seed37_test \
        --data-root <root> --window-ms 50 --out run/final/analysis/commit_cal_<run>_seed37.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_analysis import (ALPHAS, THRESHOLD, PolicyStats, _logit, anchored, lag_adaptive,  # noqa: E402
                            lag_fixed, load_split, references, report_prob, value_at)
from utils.recording_groups import dense_names  # noqa: E402

T_MIN, T_MAX = 0.05, 20.0


def equivalent_alpha(alpha, temperature):
    """The uncalibrated alpha' whose commit equals the calibrated commit at alpha: logit(1 - a') = T logit(1 - a)."""
    return 1.0 / (1.0 + math.exp(float(temperature) * _logit(1.0 - float(alpha))))


def nll_sums(recs, tau, temps):
    """Sum over every non-NaN (event, lag) value of BCE(sigmoid(c_T(z)), y), for each T in temps; and the count."""
    out = np.zeros(len(temps), dtype=np.float64)
    n = 0
    for rec in recs:
        z = rec["traj"].astype(np.float64)
        ok = ~np.isnan(z)
        y = np.broadcast_to(rec["pos"][:, None], z.shape)[ok].astype(np.float64)
        d = z[ok] - tau
        n += d.size
        for i, t in enumerate(temps):
            c = tau + d / t
            out[i] += float(np.sum(np.logaddexp(0.0, c) - y * c))
    return out, n


def fit_temperature(recs, tau):
    """NLL-optimal anchored temperature on the given recordings: log grid, then a finer grid around the minimum."""
    coarse = np.exp(np.linspace(math.log(T_MIN), math.log(T_MAX), 49))
    nll, n = nll_sums(recs, tau, coarse)
    i = int(np.argmin(nll))
    lo, hi = coarse[max(i - 1, 0)], coarse[min(i + 1, len(coarse) - 1)]
    fine = np.exp(np.linspace(math.log(lo), math.log(hi), 41))
    nll_f, _ = nll_sums(recs, tau, fine)
    j = int(np.argmin(nll_f))
    nll_1, _ = nll_sums(recs, tau, [1.0])
    return {"temperature": float(fine[j]), "nll_mean": float(nll_f[j] / max(n, 1)),
            "alpha_equivalent": {"%g" % a: equivalent_alpha(a, float(fine[j])) for a in ALPHAS},
            "nll_mean_uncalibrated": float(nll_1[0] / max(n, 1)), "n_values": int(n),
            "at_grid_edge": bool(i in (0, len(coarse) - 1))}


def policy_names(depth, alphas):
    return ["zero", "fixed_%d" % depth] + ["adaptive_%g" % a for a in alphas] + ["adaptive_cal_%g" % a for a in alphas]


def evaluate(recs, dense, window_ms, temperature, alphas=ALPHAS, full=True, sensor=(260, 346)):
    """Uncalibrated and calibrated adaptive commits (plus zero / fixed D) with final_analysis.PolicyStats."""
    depth = recs[0]["traj"].shape[1] - 1
    tau = _logit(THRESHOLD) - recs[0]["log_w"]
    names = policy_names(depth, alphas)
    stats = {n: PolicyStats(n, window_ms, full=full, sensor=sensor) for n in names}
    for rec in recs:
        ref = references(rec, tau, window_ms)
        traj = rec["traj"]
        cal = anchored(traj, tau, temperature)
        for n in names:
            source = None
            if n == "zero":
                lag = np.zeros(traj.shape[0], dtype=np.int64)
            elif n.startswith("fixed_"):
                lag = lag_fixed(traj, depth)
            elif n.startswith("adaptive_cal_"):
                lag, source = lag_adaptive(cal, tau, float(n.rsplit("_", 1)[1]))
            else:
                lag, source = lag_adaptive(traj, tau, float(n.rsplit("_", 1)[1]))
            prob = report_prob(value_at(traj, lag), rec["log_w"])
            stats[n].add(rec, prob, lag, ref, source, rec["name"] in dense)
    return {"D": depth, "tau_cal": tau, "policies": {n: s.result() for n, s in stats.items()}}


def markdown(result):
    eq = result["fit"].get("alpha_equivalent", {})
    lines = ["温度 T = %.4f（val 上按 NLL 拟合；NLL %.5f -> %.5f）；校准后的 alpha 等价于未校准的：%s" % (
        result["fit"]["temperature"], result["fit"]["nll_mean_uncalibrated"], result["fit"]["nll_mean"],
        "、".join("%s -> %.4f" % (a, v) for a, v in eq.items())), "",
             "| 划分 | 策略 | IoU | Pd | Fa | 目标事件延迟 均值 / 中位 ms | 到达即提交 / 中途 / 等满 | 首检中位 ms |",
             "|---|---|---|---|---|---|---|---|"]
    for split in ("val", "test"):
        res = result.get(split)
        if not res:
            continue
        for n in ["zero", "fixed_%d" % res["D"], "adaptive_0.05", "adaptive_cal_0.05", "adaptive_0.2", "adaptive_cal_0.2"]:
            r = res["policies"].get(n)
            if r is None:
                continue
            src = r.get("target_commit_source") or {}
            lines.append("| %s | %s | %.4f | %s | %s | %.1f / %.1f | %s | %s |" % (
                split, n, r["iou"], "%.4f" % r["pd"] if "pd" in r else "-", "%.3e" % r["fa"] if "fa" in r else "-",
                r["latency_ms"]["target_mean"], r["latency_ms"]["target_median"],
                "%.3f / %.3f / %.3f" % (src.get("arrival", 0), src.get("mid_wait", 0), src.get("deadline", 0)) if src else "-",
                "%.1f" % r["first_detection_ms"]["median"] if r["first_detection_ms"]["median"] is not None else "-"))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--val-traj", required=True)
    parser.add_argument("--test-traj", default=None, help="reported once with the temperature fitted on val")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--window-ms", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-roc", action="store_true", help="skip Pd / Fa (faster)")
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--width", type=int, default=346)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = {"val_traj": args.val_traj, "test_traj": args.test_traj, "window_ms": args.window_ms}
    recs = load_split(args.val_traj, args.data_root, "val")
    tau = _logit(THRESHOLD) - recs[0]["log_w"]
    result["fit"] = fit_temperature(recs, tau)
    T = result["fit"]["temperature"]
    print("val 拟合的温度 T = %.4f（NLL %.5f -> %.5f，%d 个值）%s" % (
        T, result["fit"]["nll_mean_uncalibrated"], result["fit"]["nll_mean"], result["fit"]["n_values"],
        "  [在网格边上]" if result["fit"]["at_grid_edge"] else ""), flush=True)
    sensor = (args.height, args.width)
    dense = dense_names(args.data_root, ("val", "test") if args.test_traj else ("val",), args.height, args.width)
    result["val"] = evaluate(recs, set(dense["val"]), args.window_ms, T, full=not args.no_roc, sensor=sensor)
    del recs
    if args.test_traj:
        recs = load_split(args.test_traj, args.data_root, "test")
        result["test"] = evaluate(recs, set(dense["test"]), args.window_ms, T, full=not args.no_roc, sensor=sensor)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    md = markdown(result)
    with open(os.path.splitext(args.out)[0] + ".md", "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print(md)
    print("COMMIT CALIBRATION FINISHED:", args.out, flush=True)


if __name__ == "__main__":
    main()
