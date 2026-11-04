"""RCL vs B2 on V5 (decision A, 11-04): read-only comparison of the decision heads over seeds, from a results package
(tools/final/pack.sh) or the code dir. Pooled official metrics only; per-recording counts are an internal check.

Usage: python -I tools/final/compare_heads.py <root> [--seeds 37 38 39]
Prints, per head and seed: val / test IoU at fixed D (pick by val), adaptive commit (calibrated when the analysis has
it) with mean target latency, the no-future eval of the same weights; the mean over the seeds present; per-epoch val
curves (late-epoch mean, epochs where B2 leads); per-recording wins of B2 over RCL.
"""
import argparse
import json
import os

HEADS = [("RCL-feature", "rcl_feature_50_on_v5"), ("RCL-score", "rcl_score_50_on_v5"), ("B2", "b2_lookahead_50_on_v5"),
         ("B2+occ", "b2_lookahead_occ_50_on_v5")]


def load(root, rel):
    path = os.path.join(root, rel)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def track(root, run, seed, split, tag=""):
    d = load(root, "log/final/%s_seed%d/eval_%s_selected_seed%d%s.json" % (run, seed, split, seed, tag))
    return d["track"] if d else None


def adaptive(root, run, seed, split):
    a = load(root, "run/final/analysis/%s_seed%d_%s.json" % (run, seed, split))
    if not a:
        return None, None
    name = "adaptive_cal_0.05" if "adaptive_cal_0.05" in a["policies"] else "adaptive_0.05"
    p = a["policies"][name]
    return p, ("cal T=%.2f" % a["temperature"] if "_cal_" in name and a.get("temperature") else "uncal")


def curve(root, run, seed):
    path = os.path.join(root, "log/final/%s_seed%d/metrics.jsonl" % (run, seed))
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            v = r.get("val")
            out[int(r["epoch"])] = v["track"]["iou"] if isinstance(v, dict) and "track" in v else r.get("val_iou")
    return out


def per_seq_iou(root, run, seed, split):
    d = load(root, "log/final/%s_seed%d/eval_%s_selected_seed%d.json" % (run, seed, split, seed))
    if not d:
        return None
    out = {}
    for k, v in d["per_sequence"]["track"].items():
        tp, fp, fn = (v["tp"], v["fp"], v["fn"]) if isinstance(v, dict) else (v[0], v[1], v[2])
        out[k] = tp / float(tp + fp + fn) if tp + fp + fn else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--seeds", type=int, nargs="+", default=[37, 38, 39])
    args = ap.parse_args()
    R = args.root
    print("== per head and seed: fixed D val / test | adaptive alpha 0.05 val / test (target mean latency) | no future val / test")
    means = {}
    for name, run in HEADS:
        for s in args.seeds:
            fv, ft = track(R, run, s, "val"), track(R, run, s, "test")
            if not fv or not ft:
                continue
            av, _ = adaptive(R, run, s, "val")
            at, how = adaptive(R, run, s, "test")
            pv, pt = track(R, run, s, "val", "_past"), track(R, run, s, "test", "_past")
            print("%-12s seed %d | %.4f / %.4f | %s / %s (%s ms, %s) | %s / %s" % (
                name, s, fv["iou"], ft["iou"], "%.4f" % av["iou"] if av else "-", "%.4f" % at["iou"] if at else "-",
                "%.0f" % at["latency_ms"]["target_mean"] if at else "-", how, "%.4f" % pv["iou"] if pv else "-",
                "%.4f" % pt["iou"] if pt else "-"))
            means.setdefault(name, []).append((fv["iou"], ft["iou"], ft["acc"], ft.get("pd"), ft.get("fa")))
    print("\n== mean over the seeds present (fixed D): val IoU | test IoU / ACC / Pd / Fa")
    for name, rows in means.items():
        n = len(rows)
        m = [sum(r[i] for r in rows) / n for i in range(5)]
        print("%-12s %d seed(s) | %.4f | %.4f / %.4f / %.4f / %.2e" % (name, n, m[0], m[1], m[2], m[3], m[4]))
    print("\n== per-epoch val (late = last 10 epochs) and B2 vs RCL-feature")
    for s in args.seeds:
        cr, cb = curve(R, "rcl_feature_50_on_v5", s), curve(R, "b2_lookahead_50_on_v5", s)
        for name, c in (("RCL-feature", cr), ("B2", cb)):
            if c:
                last = sorted(c)[-10:]
                print("seed %d %-12s %d epochs | best %.4f @ %d | late mean %.4f" % (
                    s, name, len(c), max(c.values()), max(c, key=c.get), sum(c[e] for e in last) / len(last)))
        common = [e for e in sorted(set(cr) & set(cb)) if e >= 10]
        if common:
            print("seed %d B2 ahead in %d / %d epochs (from epoch 10)" % (s, sum(cb[e] > cr[e] for e in common), len(common)))
    print("\n== per recording (fixed D): B2 better / worse / tie (|d| < 0.0005) than RCL-feature")
    for s in args.seeds:
        for split in ("val", "test"):
            a, b = per_seq_iou(R, "rcl_feature_50_on_v5", s, split), per_seq_iou(R, "b2_lookahead_50_on_v5", s, split)
            if a and b:
                d = [b[k] - a[k] for k in a if k in b]
                print("seed %d %-4s %d / %d / %d of %d" % (s, split, sum(x >= 5e-4 for x in d), sum(x <= -5e-4 for x in d),
                                                          sum(abs(x) < 5e-4 for x in d), len(d)))


if __name__ == "__main__":
    main()
