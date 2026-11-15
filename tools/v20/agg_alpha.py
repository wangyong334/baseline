"""V20 (11-15): tables of the risk-level sweep of the V18 runs (outputs of tools/v18/reeval_v18.sh, no retraining).

Table 1 (hyperparameter alpha, like SpTopoNet Tab. 5): per setting the mean over the runs of val (honest: two-half
calibration) and test IoU / ACC / Pd / Fa, units, waits, target latency, the realised change rate on test and the
decision-layer energy. alpha 0.01 = the runs' own evaluation; the others = run/v18/alpha/<setting>/<bb>__<run>/.
For the two-population runs only the second (final) population is re-calibrated; the frozen first one keeps its bounds.
Table 2 (accuracy vs latency): fixed waits and the calibrated rule at every alpha, test IoU against target latency.
Usage (any dir): python tools/v20/agg_alpha.py --root <V18 dir> [--json out.json]
"""
import argparse
import glob
import json
import os

import numpy as np

SETTINGS = (("alpha 0.005", "alpha/a0.005"), ("alpha 0.01", None), ("alpha 0.02", "alpha/a0.02"),
            ("alpha 0.05", "alpha/a0.05"), ("0.01, no collapse", "alpha/nocollapse"))
SETS = (("two populations (coopw)", "run/v18/v5s3?/coopw_s?"), ("one population (stage1w)", "run/v18/stage1w/v5s3?/final_s?"))
FIXED = ("arrival", "step0_all", "entry_wait1", "entry_wait2", "entry_wait3", "entry_wait5", "deadline_all")


def evals(root, pattern, sub):
    out = []
    for run in sorted(glob.glob(os.path.join(root, pattern))):
        name = "%s__%s" % (os.path.basename(os.path.dirname(run)), os.path.basename(run))
        path = os.path.join(run, "eval", "eval.json") if sub is None else os.path.join(root, "run/v18", sub, name,
                                                                                    "eval.json")
        if os.path.exists(path):
            with open(path) as fh:
                out.append(json.load(fh))
    return out


def energy(e):
    return e.get("energy_mJ_per_8s_both_populations", e["test"]["calibrated"]["load"]["energy_mJ_per_8s"])


def row(es, split, policy):
    r = [e[split][policy] for e in es]
    m = lambda f: float(np.mean([f(x) for x in r]))                     # noqa: E731
    return {"iou": m(lambda x: x["iou"]), "acc": m(lambda x: x["acc"]), "pd": m(lambda x: x["pd"]),
            "fa": m(lambda x: x["fa"]), "units": m(lambda x: x["born_frac"]),
            "wait_born": m(lambda x: x["wait_windows"]["born_mean"]),
            "lat_target": m(lambda x: x["latency_ms"]["target_mean"]), "lat_all": m(lambda x: x["latency_ms"]["all_mean"]),
            "changed": m(lambda x: x["changed"]["all"]), "changed_ucb": m(lambda x: x["changed"]["ucb95_recordings"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="V18 code dir (EV-UAV-v18b)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dump = {}
    for title, pattern in SETS:
        print("== %s: hyperparameter alpha (mean over runs; val = honest two-half calibration)" % title)
        print("%-18s %3s | %-30s | %-30s | units  wait  lat_t  changed (ucb)    mJ/8s" % (
            "setting", "n", "val IoU ACC Pd Fa(e-6)", "test IoU ACC Pd Fa(e-6)"))
        dump[title] = {}
        for name, sub in SETTINGS:
            es = evals(args.root, pattern, sub)
            if not es:
                print("%-18s   0 | (missing)" % name)
                continue
            v, t = row(es, "val", "calibrated"), row(es, "test", "calibrated")
            mj = float(np.mean([energy(e) for e in es]))
            dump[title][name] = {"n": len(es), "val": v, "test": t, "energy_mJ": mj}
            print("%-18s %3d | %.2f %.2f %.2f %5.2f | %.2f %.2f %.2f %5.2f | %4.1f%% %4.2f %5.1f  %.2e (%.2e) %5.3f" % (
                name, len(es), 100 * v["iou"], 100 * v["acc"], 100 * v["pd"], 1e6 * v["fa"], 100 * t["iou"],
                100 * t["acc"], 100 * t["pd"], 1e6 * t["fa"], 100 * t["units"], t["wait_born"], t["lat_target"],
                t["changed"], t["changed_ucb"], mj))
        print("-- paper table (validation set, like SpTopoNet Tab. 5): alpha & IoU & ACC & Pd & Fa(1e-6) & target latency")
        for name in ("alpha 0.005", "alpha 0.01", "alpha 0.02", "alpha 0.05"):
            d = dump[title].get(name)
            if d:
                v = d["val"]
                print("   %s & %.2f & %.2f & %.2f & %.2f & %.1f \\\\" % (name.split()[1], 100 * v["iou"], 100 * v["acc"],
                                                                    100 * v["pd"], 1e6 * v["fa"], v["lat_target"]))
        es = evals(args.root, pattern, None)
        if es:
            print("-- accuracy vs latency (alpha 0.01 runs; fixed waits; test target / all-event latency in ms)")
            curve = {}
            for p in FIXED:
                v, t = row(es, "val", p), row(es, "test", p)
                curve[p] = {"val_iou": v["iou"], "test_iou": t["iou"], "lat_target": t["lat_target"], "lat_all": t["lat_all"]}
                print("   %-13s val %.2f test %.2f | latency target %6.1f all %6.1f" % (
                    p, 100 * v["iou"], 100 * t["iou"], t["lat_target"], t["lat_all"]))
            for name, d in dump[title].items():
                print("   %-13s val %.2f test %.2f | latency target %6.1f all %6.1f  (calibrated, %s)" % (
                    "calibrated", 100 * d["val"]["iou"], 100 * d["test"]["iou"], d["test"]["lat_target"],
                    d["test"]["lat_all"], name))
            dump[title]["curve"] = curve
        print()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dump, fh, indent=1)
    print("AGG ALPHA DONE")


if __name__ == "__main__":
    main()
