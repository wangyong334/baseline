"""逐序列误差分解（只读 --dump-dir 写出的逐事件预测 NPZ，CPU，几秒到几分钟）。

主指标 IoU 是整个划分拼接后的全局 IoU = ΣTP / (ΣTP + ΣFP + ΣFN)，所以每条序列的 FP、FN 可以直接相加。
本工具回答"全局 IoU 卡在哪几条序列上"：
  每条序列：事件数、正事件数、TP / FP / FN（prob >= 阈值）、序列 IoU、AUC（目标分数高于背景的概率）、
           本序列最优单一阈值的 IoU、FP / FN 占全划分的比例、去掉这一条后的全局 IoU。
用法：
    python tools/per_sequence_metrics.py --dump-dir run/dump/e3b_test --out run/perseq_e3b_test.json
    多个模型并排：--dump-dir a=run/dump/v5_test --dump-dir b=run/dump/e3b_test（第一个为 A，打印 B - A 的序列 IoU 差）
"""
import argparse
import glob
import json
import os

import numpy as np

GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)


def auc_score(score, positive):
    """Mann-Whitney AUC，相同分数取平均秩。"""
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    avg_rank = upper - (counts - 1) / 2.0
    rank_sum = avg_rank[inverse][positive].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def analyze(path, threshold):
    d = np.load(path)
    prob = d["probabilities"].astype(np.float64)
    pos = d["labels"] > 0.5
    pred = prob >= threshold
    tp = int((pred & pos).sum()); fp = int((pred & ~pos).sum()); fn = int((~pred & pos).sum())
    best = (0.0, None)
    for th in GRID:
        p = prob >= th
        a = int((p & pos).sum()); b = int((p & ~pos).sum()); c = int((~p & pos).sum())
        v = a / (a + b + c) if a + b + c else 0.0
        if v > best[0]:
            best = (v, float(th))
    return {"name": os.path.splitext(os.path.basename(path))[0], "events": int(pos.size), "positives": int(pos.sum()),
            "tp": tp, "fp": fp, "fn": fn, "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "auc": auc_score(prob, pos), "best_iou": best[0], "best_threshold": best[1]}


def summarize(rows):
    TP = sum(r["tp"] for r in rows); FP = sum(r["fp"] for r in rows); FN = sum(r["fn"] for r in rows)
    for r in rows:
        r["fp_share"] = r["fp"] / FP if FP else 0.0
        r["fn_share"] = r["fn"] / FN if FN else 0.0
        rest = (TP - r["tp"]) + (FP - r["fp"]) + (FN - r["fn"])
        r["global_iou_without"] = (TP - r["tp"]) / rest if rest else None
    return {"global_iou": TP / (TP + FP + FN) if TP + FP + FN else None, "tp": TP, "fp": FP, "fn": FN}


def fmt(v, spec=".4f"):
    return "-" if v is None else format(v, spec)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-dir", action="append", required=True, help="目录，或 名称=目录")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = {"threshold": args.threshold, "runs": {}}
    for item in args.dump_dir:
        name, _, folder = item.rpartition("=")
        name = name or os.path.basename(os.path.normpath(folder))
        files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not files:
            raise SystemExit("没有 NPZ：%s" % folder)
        rows = [analyze(f, args.threshold) for f in files]
        total = summarize(rows)
        report["runs"][name] = {"summary": total, "sequences": rows}
        print("== %s  全局 IoU %s  (TP %d FP %d FN %d)" % (name, fmt(total["global_iou"]), total["tp"], total["fp"], total["fn"]))
        print("%-10s %9s %8s %7s %7s %8s %7s %8s %6s %6s %8s" % (
            "序列", "事件", "正事件", "IoU", "AUC", "最优IoU", "最优阈", "去掉后", "FP占", "FN占", "FP/正"))
        for r in sorted(rows, key=lambda r: -(r["fp"] + r["fn"])):
            print("%-10s %9d %8d %7s %7s %8s %7s %8s %5.0f%% %5.0f%% %8.2f" % (
                r["name"], r["events"], r["positives"], fmt(r["iou"]), fmt(r["auc"]), fmt(r["best_iou"]),
                fmt(r["best_threshold"], ".2f"), fmt(r["global_iou_without"]), 100 * r["fp_share"],
                100 * r["fn_share"], r["fp"] / max(r["positives"], 1)))
    names = list(report["runs"])
    if len(names) >= 2:
        a, b = (report["runs"][n]["sequences"] for n in names[:2])
        ia = {r["name"]: r["iou"] for r in a}
        diff = [(r["name"], (r["iou"] or 0) - (ia.get(r["name"]) or 0)) for r in b if r["name"] in ia]
        diff.sort(key=lambda x: x[1])
        print("== 序列 IoU 差 %s - %s（最差 5 / 最好 5）" % (names[1], names[0]))
        for n, v in (diff if len(diff) <= 10 else diff[:5] + diff[-5:]):
            print("  %-10s %+.4f" % (n, v))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
