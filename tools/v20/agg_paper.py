"""Main-text tables of the paper (V20, 11-16) from the result packs, so that no number is copied by hand.

Every table is a mean over the runs listed below (three backbone seeds, or the 9 runs of the paper model), test split,
IoU / ACC / Pd (%) / Fa (1e-6) and the mean latency of target events (ms); a missing run is printed as MISSING and the
mean is not formed. --pop selects the decision layer: base = one population, coop = two populations (the paper model).
  Table 2  components: per-slice / streaming backbone, message budget, SDM own slice (step 0) / future slices
  Table 3A when to decide: arrival, step 0, fixed waits 1 / 2 / 3 / 5 slices, deadline, calibrated (paper-model runs)
  Table 3B slice duration 25 / 50 / 100 / 200 / 400 ms: backbone (arrival) and + SDM (calibrated)
  Table 4  plug-in hosts RVT / SAST / YOLOv10: host (arrival), + SDM step 0, + SDM (calibrated), over decision seeds
Usage: python tools/v20/agg_paper.py --v20 <root with run/v18/p20_* and log/final/v20_*> --v18 <root with
       run/v18/v5s3?/coopw_s? and run/v18/stage1w/...> [--pop coop]
"""
import argparse
import glob
import json
import os

import numpy as np

SEEDS3 = ("", "_bb38", "_bb39")
METRICS = ("iou", "acc", "pd", "fa")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def row(ev, mode):
    r = ev["test"][mode]
    return [100 * r["iou"], 100 * r["acc"], 100 * r["pd"], 1e6 * r["fa"], r["latency_ms"]["target_mean"]]


def runs(root, tags, name):
    """eval.json of run/v18/<tag>/<name> for every tag; None where missing."""
    out = []
    for t in tags:
        f = os.path.join(root, "run", "v18", t, name, "eval", "eval.json")
        out.append(load(f) if os.path.exists(f) else None)
    return out


def messages(root, name, seeds=(37, 38, 39)):
    """Mean messages per slice (k) of the backbone runs log/final/<name>_seed<s>."""
    vals = []
    for s in seeds:
        f = os.path.join(root, "log", "final", "%s_seed%d" % (name, s), "eval_test_selected_seed%d.json" % s)
        if not os.path.exists(f):
            return None
        d = load(f)
        r = d.get("reset_each_window") if "reset1" in name else d.get("carry")
        r = r if isinstance(r, dict) else d["carry"]
        m = r.get("overall", r).get("messages_per_window") or r.get("messages_per_window")
        if not isinstance(m, dict):
            return None
        vals.append(m["mean"] / 1000.0)
    return float(np.mean(vals))


def show(label, evs, mode, extra=""):
    if any(e is None for e in evs):
        print("  %-34s MISSING (%d of %d runs)%s" % (label, sum(e is None for e in evs), len(evs), extra))
        return
    a = np.array([row(e, mode) for e in evs])
    m = a.mean(0)
    print("  %-34s IoU %6.2f  ACC %6.2f  Pd %6.2f  Fa %5.2f  lat %6.1f%s   [IoU per run %s]" % (
        label, m[0], m[1], m[2], m[3], m[4], extra, " ".join("%.2f" % v for v in a[:, 0])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v20", required=True)
    ap.add_argument("--v18", default=None)
    ap.add_argument("--pop", choices=("base", "coop"), default="coop")
    args = ap.parse_args()
    sdm = "%s_s0" % args.pop
    R = args.v20

    print("Table 2 (components, test, 3 backbone seeds, SDM = %s)" % ("two populations" if args.pop == "coop" else
                                                                      "one population"))
    msg = {k: messages(R, n) for k, n in (("reset", "v20_bir_reset1"), ("nobudget", "v20_bir_nobudget"),
                                          ("carry", "v20_bir_carry"))}
    fm = lambda k: "  msgs %s k" % ("%.1f" % msg[k] if msg[k] is not None else "?")
    reset, nob, carry = (["p20_bir_%s%s" % (b, s) for s in SEEDS3] for b in ("reset1", "nobudget", "carry"))
    # backbone-only rows: the arrival column of the base runs (identical to the backbone evaluation)
    show("per-slice SNN (budget)", runs(R, reset, "base_s0"), "arrival", fm("reset"))
    show("streaming SNN, no budget", runs(R, nob, "base_s0"), "arrival", fm("nobudget"))
    show("streaming SNN (budget)", runs(R, carry, "base_s0"), "arrival", fm("carry"))
    show("+ SDM own slice", runs(R, carry, sdm), "step0_all", fm("carry"))
    show("per-slice SNN + SDM", runs(R, reset, sdm), "calibrated", fm("reset"))
    show("streaming, no budget + SDM", runs(R, nob, sdm), "calibrated", fm("nobudget"))
    show("streaming (budget) + SDM = SDNet", runs(R, carry, sdm), "calibrated", fm("carry"))

    if args.v18:
        pat = ("run/v18/v5s3?/coopw_s?" if args.pop == "coop" else "run/v18/stage1w/v5s3?/*")
        evs = [load(os.path.join(d, "eval", "eval.json")) for d in sorted(glob.glob(os.path.join(args.v18, pat)))
               if os.path.exists(os.path.join(d, "eval", "eval.json"))]
        print("\nTable 3A (when to decide, test, %d runs of the paper model, %s)" % (len(evs), pat))
        for label, mode in (("arrival (no SDM)", "arrival"), ("own slice only", "step0_all"),
                            ("fixed wait 1 slice", "entry_wait1"), ("fixed wait 2 slices", "entry_wait2"),
                            ("fixed wait 3 slices", "entry_wait3"), ("fixed wait 5 slices", "entry_wait5"),
                            ("wait to the deadline", "deadline_all"), ("calibrated stopping (ours)", "calibrated")):
            show(label, evs if evs else [None], mode)

    print("\nTable 3B (slice duration, test, 3 backbone seeds, SDM = %s)" % args.pop)
    for ms in (25, 50, 100, 200, 400):
        tags = carry if ms == 50 else ["p20_w%d_bb%d" % (ms, s) for s in (37, 38, 39)]
        show("%3d ms backbone" % ms, runs(R, tags, "base_s0"), "arrival")
        show("%3d ms + SDM" % ms, runs(R, tags, sdm), "calibrated")

    print("\nTable 4 (plug-in, test, SDM = %s, decision seeds 0-2 where present)" % args.pop)
    for host in ("rvt", "sast", "yolov10"):
        tag = "p20_" + host
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(R, "run", "v18", tag, args.pop + "_s*")))
        evs = runs(R, [tag] * max(len(names), 1), names[0] if names else sdm) if len(names) <= 1 else \
            [load(os.path.join(R, "run", "v18", tag, n, "eval", "eval.json")) for n in names
             if os.path.exists(os.path.join(R, "run", "v18", tag, n, "eval", "eval.json"))]
        meta = os.path.join(R, "run", "deco", tag, "host_meta.json")
        thr = " (thr* %.3f)" % load(meta)["thr"] if os.path.exists(meta) else ""
        print(" %s%s, %d decision run(s): %s" % (host, thr, len(names), " ".join(names) if names else "none"))
        show("host", evs if names else [None], "arrival")
        show("+ SDM own slice", evs if names else [None], "step0_all")
        show("+ SDM", evs if names else [None], "calibrated")
    print("AGG PAPER DONE")


if __name__ == "__main__":
    main()
