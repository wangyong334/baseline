"""Ablation table of the decision neuron (V20, 11-16): single population, SDM seed 0, three streaming SNNs.

For every ablation run run/v18/<bb>/abl_<name>_s0 (V18 code) the difference to the full single-population model
run/v18/<bb>/base_s0 of the same backbone, on val (honest two-half calibration) and test: IoU / ACC / Pd / Fa (1e-6),
target latency (ms) and the fraction of events that spawn decision neurons. With --v17-fallback a missing seed-37 run is
taken from run/v18/reeval_abl/v5s37__abl_<name>_s0 (V17 code, re-evaluated with the V18 rule) and compared with the V17
main run (--v17-base); such rows are marked '*'. abl_pool_mean of the V17 code is invalid (the switch had no effect in
the V17 ledger) and is never used.
Usage: python tools/v20/agg_sdm_ablation.py --root <dir with run/v18> [--v17-fallback --v17-base <final_s0 eval.json>]
Last line: "AGG SDM ABLATION DONE".
"""
import argparse
import json
import os

import numpy as np

BACKBONES = ("v5s37", "v5s38", "v5s39")
RUNS = (("abl_noattr", "w/o attribution gate", "--gate none"),
        ("abl_pool_sum", "per-event sum (no slice normalization)", "--pool sum"),
        ("abl_boxes", "boxes as observations", "--evidence boxes"),
        ("abl_nopseudo", "w/o leak (pseudo-observation)", "--pseudo off"),
        ("abl_noctx", "w/o context", "--context off"),
        ("abl_signed", "single signed channel", "--value signed"),
        ("abl_pool_mean", "mean over slices (no accumulation)", "--pool mean"),
        ("abl_reaudit", "re-read all slices at every step", "--audit reaudit"),
        ("abl_levels3", "three box levels", "--levels 3"),
        ("abl_gate_content", "content-based attribution gate", "--gate attribution"))
COLS = ("IoU", "ACC", "Pd", "Fa", "Lat", "Wait%")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def metrics(ev, split):
    r = ev[split]["calibrated"]
    return np.array([100 * r["iou"], 100 * r["acc"], 100 * r["pd"], 1e6 * r["fa"], r["latency_ms"]["target_mean"],
                     100 * r.get("born_frac", np.nan)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--v17-fallback", action="store_true")
    ap.add_argument("--v17-base", default=None, help="eval.json of the V17 main run of v5s37 (stage1w final_s0)")
    args = ap.parse_args()
    r18 = os.path.join(args.root, "run", "v18")
    base = {b: load(os.path.join(r18, b, "base_s0", "eval", "eval.json")) for b in BACKBONES}
    v17 = load(args.v17_base) if args.v17_base else None
    print("full single-population model (base_s0): " + "  ".join(
        "%s val %.2f test %.2f" % (b, metrics(e, "val")[0], metrics(e, "test")[0]) for b, e in base.items() if e))
    print("%-42s %-36s | mean dIoU val / test | mean d(test) %s" % ("ablation (flag)", "per backbone dIoU val/test",
                                                                   " ".join(COLS[1:])))
    for name, label, flag in RUNS:
        rows, txt = [], []
        for b in BACKBONES:
            ev, ref, mark = load(os.path.join(r18, b, name + "_s0", "eval", "eval.json")), base[b], " "
            if ev is None and args.v17_fallback and b == "v5s37" and name != "abl_pool_mean" and v17 is not None:
                ev = load(os.path.join(r18, "reeval_abl", "v5s37__%s_s0" % name, "eval.json"))
                ref, mark = v17, "*"
            if ev is None or ref is None:
                txt.append("%s    -     " % b[-2:])
                continue
            dv, dt = metrics(ev, "val") - metrics(ref, "val"), metrics(ev, "test") - metrics(ref, "test")
            rows.append(np.concatenate([dv, dt]))
            txt.append("%s %+.2f/%+.2f%s" % (b[-2:], dv[0], dt[0], mark))
        if not rows:
            print("%-42s (missing)" % ("%s (%s)" % (label, flag)))
            continue
        a = np.mean(rows, 0)
        print("%-42s %s | %+.2f / %+.2f (n=%d) | %s" % ("%s (%s)" % (label, flag), "  ".join(txt), a[0], a[6],
                                                         len(rows), " ".join("%+.2f" % v for v in a[7:])))
    print("('*' = seed-37 run of the V17 code against the V17 main run)")
    print("AGG SDM ABLATION DONE")


if __name__ == "__main__":
    main()
