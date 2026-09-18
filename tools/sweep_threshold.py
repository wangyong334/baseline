"""扫描判定阈值，在同一虚警率下比较不同模型（而不是固定在 0.9 这一个工作点）。

背景：阈值 0.9 下 LIF 与 ReLU 落在完全不同的工作点上——LIF 检出率 Pd 0.95、虚警 1.0e-5，
ReLU 保守得多（Pd 0.85、虚警 6.5e-6）。这种情况下直接比逐事件 IoU 没有意义，
需要把阈值扫一遍，再在相同虚警率（或相同 Pd）处横向比较。

输入是 --dump-dir 导出的逐事件预测（每条序列一个 NPZ，字段 locs[batch,x,y,t]、labels、probabilities、target_id），
流式模型用 `train_stream_v1.py --mode eval --dump-dir ...` 生成，基线用 `tools/dump_baseline_predictions.py`。
每个阈值下计算：
    IoU / 召回 / 精确率  由全部序列拼接后的逐事件预测算出（与 utils/eval.py 的口径一致）
    Pd / Fa              调用原 utils/eval.py 的 roc_update 与 cal_roc（每个阈值用一个新的 evaluator）
再按目标虚警率线性插值，给出"等虚警率下的 Pd 与 IoU"。

用法（服务器）:
    python tools/sweep_threshold.py --target-fa 1e-5 \
        --dump-dir log/verify/v1_s37_test log/verify/relu_s37_test log/verify/baseline_k5_s37_test
默认阈值网格 0.1~0.99；只想快速看 IoU 曲线可加 --no-pd（跳过较慢的 Pd/Fa 统计）。
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

DEFAULT_THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="阈值扫描与等虚警率比较")
    parser.add_argument("--dump-dir", nargs="+", required=True, help="一个或多个逐事件预测目录，每个目录是一个模型")
    parser.add_argument("--thresholds", nargs="*", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--target-fa", nargs="*", type=float, default=[1e-5, 6.5e-6],
                        help="在这些虚警率处插值比较（默认取 LIF 与 ReLU 在 0.9 阈值下的两个虚警率）")
    parser.add_argument("--pd-detT", type=int, default=50)
    parser.add_argument("--correct-thresh", type=float, default=1e-4)
    parser.add_argument("--no-pd", action="store_true", help="跳过 Pd/Fa（快很多，只看 IoU/召回/精确率）")
    parser.add_argument("--max-sequences", type=int, default=0, help="每个目录只用前几条序列（0 = 全部）")
    parser.add_argument("--out", default="log/energy/threshold_sweep.json")
    return parser.parse_args()


def load_dump(directory, max_sequences=0):
    """读取一个目录下的逐事件预测，返回按文件名排序的序列列表。"""
    names = sorted(n for n in os.listdir(directory) if n.endswith(".npz"))
    if max_sequences:
        names = names[:int(max_sequences)]
    if not names:
        raise RuntimeError("目录里没有 NPZ: %s" % directory)
    out = []
    for name in names:
        with np.load(os.path.join(directory, name)) as data:
            out.append({"name": name, "locs": np.asarray(data["locs"]),
                        "labels": np.asarray(data["labels"]).astype(np.float32),
                        "probs": np.asarray(data["probabilities"]).astype(np.float32),
                        "target_id": np.asarray(data["target_id"])})
    return out


def event_metrics(labels, probs, threshold):
    """逐事件指标：正类 IoU、召回（= utils/eval.py 的 ACC）、精确率。"""
    positive = labels == 1
    predicted = probs >= threshold
    tp = int(np.count_nonzero(positive & predicted))
    fp = int(np.count_nonzero(~positive & predicted))
    fn = int(np.count_nonzero(positive & ~predicted))
    return {"iou": tp / float(tp + fp + fn) if tp + fp + fn else float("nan"),
            "recall": tp / float(tp + fn) if tp + fn else float("nan"),
            "precision": tp / float(tp + fp) if tp + fp else float("nan"),
            "tp": tp, "fp": fp, "fn": fn}


def detection_metrics(sequences, threshold, pd_detT, correct_thresh):
    """Pd / Fa：直接调用原仓库的 evalute.roc_update 与 cal_roc，每个阈值用一个新的 evaluator。"""
    import torch
    from utils.eval import evalute
    evaluator = evalute(SimpleNamespace(roc=True, pd_detT=pd_detT, correct_thresh=correct_thresh))
    for seq in sequences:
        locs = seq["locs"]
        zeros = np.zeros(locs.shape[0], dtype=np.float32)
        ev_locs = torch.from_numpy(np.stack([zeros, locs[:, 1], locs[:, 2], locs[:, 3]], 1).astype(np.float32))
        evaluator.roc_update(ev_locs[:, 3], torch.from_numpy(seq["probs"].copy()), seq["target_id"],
                             torch.from_numpy(seq["labels"]), ev_locs, thresh=float(threshold))
    pd, fa = evaluator.cal_roc()
    return {"pd": float(pd), "fa": float(fa)}


def sweep(sequences, thresholds, with_pd, pd_detT, correct_thresh):
    """对一个模型扫描全部阈值，返回每个阈值一行的结果。"""
    labels = np.concatenate([s["labels"] for s in sequences])
    probs = np.concatenate([s["probs"] for s in sequences])
    rows = []
    for threshold in thresholds:
        row = {"threshold": float(threshold)}
        row.update(event_metrics(labels, probs, threshold))
        if with_pd:
            row.update(detection_metrics(sequences, threshold, pd_detT, correct_thresh))
        rows.append(row)
        print("    阈值 %.3f | IoU %.4f 召回 %.4f 精确率 %.4f%s" % (
            threshold, row["iou"], row["recall"], row["precision"],
            (" | Pd %.4f Fa %.2e" % (row["pd"], row["fa"])) if with_pd else ""), flush=True)
    return rows


def interpolate_at_fa(rows, target_fa):
    """在给定虚警率处线性插值（Fa 随阈值单调下降，用 log(Fa) 作插值变量）。

    目标虚警率超出扫描范围时返回 None，不做外推。
    """
    points = sorted([r for r in rows if r.get("fa", 0) > 0], key=lambda r: r["fa"])
    if len(points) < 2 or not (points[0]["fa"] <= target_fa <= points[-1]["fa"]):
        return None
    for low, high in zip(points, points[1:]):
        if low["fa"] <= target_fa <= high["fa"]:
            span = np.log(high["fa"]) - np.log(low["fa"])
            w = 0.0 if span == 0 else (np.log(target_fa) - np.log(low["fa"])) / span
            return {key: float(low[key] + w * (high[key] - low[key]))
                    for key in ("threshold", "iou", "recall", "precision", "pd")}
    return None


def main():
    """入口：逐个目录扫描阈值 -> 打印曲线 -> 在目标虚警率处对齐比较 -> 写 JSON。"""
    args = parse_args()
    missing = [d for d in args.dump_dir if not os.path.isdir(d)]
    if missing:
        raise SystemExit("找不到目录: %s" % missing)
    results = {}
    for directory in args.dump_dir:
        sequences = load_dump(directory, args.max_sequences)
        events = sum(s["labels"].shape[0] for s in sequences)
        print("%s | %d 条序列 | %d 个事件" % (directory, len(sequences), events), flush=True)
        results[directory] = {"sequences": len(sequences), "events": int(events),
                              "rows": sweep(sequences, args.thresholds, not args.no_pd,
                                            args.pd_detT, args.correct_thresh)}
    if not args.no_pd:
        for target in args.target_fa:
            print("\n=== 等虚警率 %.2e 处的对比 ===" % target)
            print("%-48s %9s %8s %8s %8s" % ("模型", "阈值", "IoU", "召回", "Pd"))
            for directory, data in results.items():
                point = interpolate_at_fa(data["rows"], target)
                data.setdefault("at_fa", {})[str(target)] = point
                if point is None:
                    print("%-48s %s" % (directory[-48:], "该虚警率不在扫描范围内"))
                else:
                    print("%-48s %9.4f %8.4f %8.4f %8.4f" % (
                        directory[-48:], point["threshold"], point["iou"], point["recall"], point["pd"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump({"thresholds": args.thresholds, "target_fa": args.target_fa, "results": results},
                  stream, indent=2, ensure_ascii=False)
    print("\n报告:", args.out)


if __name__ == "__main__":
    main()
