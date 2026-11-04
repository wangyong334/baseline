"""Pre-registered choice of the final backbone among the module-two 2x2 (FINAL_PLAN_20261104.md §3.5). Val only.

Inputs: guarded runs (tools/final/backbone.sh) of V5 / E1 / E3a / E3b with their val evaluations
(eval_val_selected_seed<S>.json, and eval_val_selected_seed<S>_noise<hz>hz_seed0.json for the noise levels).
Rule, written before any of these runs finished:
    1. guardrail against module one: normal-group val IoU >= normal-group val IoU of V5 - 0.005;
    2. primary: the official val IoU (clean);
    3. if the best two candidates are within 0.003 of each other, the more robust one wins:
       higher mean val IoU over the noise levels; if that is also within 0.01, the smaller false-positive growth
       (val FP at the strongest level / FP at the weakest level).
The module-two claim branch (PAPER_PLAN v2 §3.5) is reported alongside from E3b vs E1 (and E3a), on val:
    A  E3b beats E1 on noise IoU (mean over levels, by >= 0.01) and on FP growth  -> "NISI complements noise training"
    B  E3b ~ E1 on noise IoU (within 0.01) but smaller FP growth / messages      -> "keeps false alarms and messages bounded"
    C  otherwise                                                                  -> NISI is analysis only
Test results are never read here. Usage:
    python tools/select_backbone.py --run v5=log/final/bb_v5_50_seed37 --run e1=... --run e3a=... --run e3b=... \
        --data-root <root> [--seed 37] [--hz 0.8 1.6 2.4 4] --out run/final/backbone_choice.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.recording_groups import dense_names, group_report  # noqa: E402

GUARD, TIE, ROBUST_TIE = 0.005, 0.003, 0.01


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def summarize(run, seed, hz, dense):
    stem = os.path.join(run, "eval_val_selected_seed%d" % seed)
    clean = load(stem + ".json")
    if clean is None:
        return None
    c = clean["carry"]
    g = group_report(c["per_sequence"], dense)
    out = {"val_iou": c["iou"], "normal": g["normal"], "dense": g["dense"], "macro": g["macro"],
           "messages": c.get("messages_mean"), "noise": {}}
    for h in hz:
        d = load(stem + "_noise%ghz_seed0.json" % h)
        if d is None:
            continue
        n = d["carry"]
        fp = sum(v[1] for v in n["per_sequence"].values()) if "per_sequence" in n else None
        out["noise"]["%g" % h] = {"iou": n["iou"], "fp": fp, "fa": n.get("fa"), "messages": n.get("messages_mean")}
    levels = [out["noise"]["%g" % h] for h in hz if "%g" % h in out["noise"]]
    out["noise_mean_iou"] = sum(v["iou"] for v in levels) / len(levels) if len(levels) == len(hz) else None
    lo, hi = out["noise"].get("%g" % hz[0]), out["noise"].get("%g" % hz[-1])
    out["fp_growth"] = (hi["fp"] / float(max(lo["fp"], 1))) if lo and hi and lo["fp"] is not None else None
    out["msg_growth"] = (hi["messages"] / out["messages"]) if hi and hi.get("messages") and out["messages"] else None
    return out


def choose(table, ref="v5"):
    """Apply the rule to {name: summary}. Returns (winner, candidates, log)."""
    log = []
    if ref not in table or table[ref] is None:
        raise RuntimeError("缺参照骨干 %s 的 val 评估" % ref)
    if table[ref]["normal"] is None:                    # no normal recording in val: the guardrail is vacuous
        cands = [n for n, s in table.items() if s is not None]
        log.append("参照 %s 没有普通组录像：护栏不适用，候选 %s" % (ref, cands))
    else:
        floor = table[ref]["normal"] - GUARD
        cands = [n for n, s in table.items() if s is not None and s["normal"] is not None and s["normal"] >= floor]
        log.append("护栏：普通组 val IoU >= %.4f（%s 的普通组 - %.3f）：候选 %s" % (floor, ref, GUARD, cands))
    cands.sort(key=lambda n: -table[n]["val_iou"])
    if len(cands) >= 2 and table[cands[0]]["val_iou"] - table[cands[1]]["val_iou"] < TIE:
        a, b = cands[0], cands[1]
        ra, rb = table[a]["noise_mean_iou"], table[b]["noise_mean_iou"]
        log.append("前两名 %s / %s 的 val IoU 相差 < %.3f，比加噪鲁棒性：%s / %s" % (a, b, TIE, ra, rb))
        if ra is not None and rb is not None and abs(ra - rb) >= ROBUST_TIE:
            return (a if ra > rb else b), cands, log
        ga, gb = table[a]["fp_growth"], table[b]["fp_growth"]
        log.append("加噪 IoU 相差 < %.2f，比误报增长：%s / %s" % (ROBUST_TIE, ga, gb))
        if ga is not None and gb is not None:
            return (a if ga <= gb else b), cands, log
    return cands[0], cands, log


def claim_branch(table):
    e3b, e1 = table.get("e3b"), table.get("e1")
    if not e3b or not e1 or e3b["noise_mean_iou"] is None or e1["noise_mean_iou"] is None:
        return "undecided (E1 / E3b val noise results missing)"
    d = e3b["noise_mean_iou"] - e1["noise_mean_iou"]
    fp_better = (e3b["fp_growth"] is not None and e1["fp_growth"] is not None and e3b["fp_growth"] < e1["fp_growth"])
    if d >= ROBUST_TIE and fp_better:
        return "A: NISI complements noise training (noise IoU +%.4f, smaller FP growth)" % d
    if abs(d) < ROBUST_TIE and (fp_better or (e3b["msg_growth"] or 9e9) < (e1["msg_growth"] or 9e9)):
        return "B: same noise IoU (%+.4f) but bounded false alarms / messages" % d
    return "C: NISI as analysis only (noise IoU %+.4f, FP growth better: %s)" % (d, fp_better)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, help="name=dir (names: v5, e1, e3a, e3b)")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--hz", type=float, nargs="+", default=[0.8, 1.6, 2.4, 4.0])
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--width", type=int, default=346)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    dense = dense_names(args.data_root, ("val",), args.height, args.width)["val"]
    runs = dict(item.split("=", 1) for item in args.run)
    table = {name: summarize(path, args.seed, args.hz, dense) for name, path in runs.items()}
    winner, cands, log = choose(table)
    result = {"rule": __doc__.split("Rule")[1].split("The module-two")[0].strip(), "dense_val": dense, "table": table,
              "candidates": cands, "winner": winner, "winner_run": runs[winner], "log": log,
              "claim_branch": claim_branch(table)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    print("| 骨干 | val IoU | 普通 | 极密 | 加噪平均 IoU | 误报增长 | 消息增长 |")
    print("|---|---|---|---|---|---|---|")
    f = lambda v, p="%.4f": "-" if v is None else p % v                       # noqa: E731
    for name, s in table.items():
        if s is None:
            print("| %s | 缺 val 评估 | | | | | |" % name)
            continue
        print("| %s | %s | %s | %s | %s | %s | %s |" % (name, f(s["val_iou"]), f(s["normal"]), f(s["dense"]),
                                                         f(s["noise_mean_iou"]), f(s["fp_growth"], "x%.2f"),
                                                         f(s["msg_growth"], "x%.2f")))
    for line in log:
        print(line)
    print("选定骨干：%s（%s）" % (winner, runs[winner]))
    print("模块二主张分支：%s" % result["claim_branch"])
    print("BACKBONE CHOICE FINISHED:", args.out, flush=True)


if __name__ == "__main__":
    main()
