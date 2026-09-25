"""证据决定发布：离线评估"证据够确定就发布、不够才继续等"的逐事件读出，零训练。

读取 train_stream_v2.py --mode eval --dump-dir 的导出（需要 logit_net 与 evidence_d{d} 字段；
想要细的等待档位就在评估时给 --readout-delays 1 2 3 4 5 6 8）。对每个事件 i：
    z_i(0) = logit_net                                   网络本窗给出的 logit（d = 0，不等待）
    z_i(d) = logit_net + w * evidence_d{d}               之后 d 窗沿各速度管道的轨迹证据（与 fused_d 同一个分数）
给定决策阈值 t（概率，t* = logit(t)）与犹豫区半宽 Δ（logit 单位）：
    tau_i = 第一个满足 |z_i(d) - t*| >= Δ 的 d（d 从 0 开始，只用 <= d_max 的延迟），都不满足则取 d_max
    预测  = z_i(tau_i) >= t*，在第 k_i + tau_i 窗发布
Δ = 0 就是网络输出本身（全部 d = 0 发布）；Δ 很大时退化为固定 d_max。犹豫区以决策阈值为中心，
所以扫虚警时只动 t、Δ 固定（Δ 应在 val 上选定后冻结再用于 test）。

每个 (d_max, Δ) 扫一遍阈值，指标与 tools/sweep_threshold.py 同口径（二值预测按 0.5 判）：IoU / 召回 / 精确率 / Pd / Fa，
另给等待窗数（平均等待下降可能只来自大量容易的背景事件，所以分开报告）：
    全部事件均值、目标事件均值 / P90 / P99、困难目标事件均值（网络 logit 离决策阈值不到 1 的目标事件）、预测为正的事件均值；
以及逐目标的首次检出延迟（发布时刻口径：该目标第一个被判为正的事件的发布窗末 - 目标第一个事件的时间，
与 train_stream_v2.py 的 latency 一致，未计网络计算时间）。同一导出里的固定 d 读出作为参照一并扫描。
注意 d = 0 是"不额外等待"，结果在本窗结束时发布（mark 用的是整窗特征），不是事件一到就输出。
这是受序贯检验启发的发布策略，不引用 SPRT 的最优性定理（融合分数不满足其条件）。

用法（服务器）:
    python tools/adaptive_publish.py --dump-dir log/verify/<导出目录> --max-delays 2 5 \
        --deltas 0.5 1 2 3 --target-fa 1e-5 6.5e-6 3e-6 --out log/energy/adaptive_publish_val.json
Pd/Fa 较慢（每个阈值要重算连通域）；只看 IoU 与等待可加 --no-pd；Δ 多时可以按 Δ 分几个进程并行。
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from tools.sweep_threshold import DEFAULT_THRESHOLDS, detection_metrics, event_metrics, interpolate_at_fa  # noqa: E402


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="证据决定发布的离线评估")
    parser.add_argument("--dump-dir", required=True, help="eval 导出目录（需要 logit_net 与 evidence_d* 字段）")
    parser.add_argument("--max-delays", nargs="+", type=int, default=[2, 5], help="最长等待 d_max（窗），各跑一遍")
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0], help="犹豫区半宽 Δ（logit）")
    parser.add_argument("--thresholds", nargs="*", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--target-fa", nargs="*", type=float, default=[1e-5, 6.5e-6, 3e-6])
    parser.add_argument("--fusion-weight", type=float, default=1.0, help="z = logit + w * evidence 的 w（与 eval 一致）")
    parser.add_argument("--no-fixed", action="store_true", help="不扫固定 d 的参照读出")
    parser.add_argument("--pd-detT", type=int, default=50)
    parser.add_argument("--correct-thresh", type=float, default=1e-4)
    parser.add_argument("--no-pd", action="store_true", help="跳过 Pd/Fa（快很多）")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--window-ms", type=int, default=50)
    parser.add_argument("--n-windows", type=int, default=160, help="每条序列的窗数（发布窗在序列末尾截断）")
    parser.add_argument("--out", default="log/energy/adaptive_publish.json")
    return parser.parse_args()


def load_scores(directory, max_sequences=0):
    """读取导出：每条序列的 locs、labels、target_id、logit_net 与各延迟的 evidence。返回 (序列列表, 可用延迟升序)。"""
    names = sorted(n for n in os.listdir(directory) if n.endswith(".npz"))
    if max_sequences:
        names = names[:int(max_sequences)]
    if not names:
        raise RuntimeError("目录里没有 NPZ: %s" % directory)
    sequences, delays = [], None
    for name in names:
        with np.load(os.path.join(directory, name)) as data:
            keys = set(data.files)
            if "logit_net" not in keys:
                raise RuntimeError("%s 里没有 logit_net（需要 train_stream_v2.py 的导出）" % name)
            found = sorted(int(k[len("evidence_d"):]) for k in keys if k.startswith("evidence_d"))
            delays = found if delays is None else sorted(set(delays) & set(found))
            sequences.append({"name": name, "locs": np.asarray(data["locs"]),
                              "labels": np.asarray(data["labels"]).astype(np.float32),
                              "target_id": np.asarray(data["target_id"]),
                              "logit": np.asarray(data["logit_net"]).astype(np.float64),
                              "evidence": {d: np.asarray(data["evidence_d%d" % d]).astype(np.float64) for d in found}})
    if not delays:
        raise RuntimeError("导出里没有 evidence_d* 字段（评估时需要 --readout-delays）")
    return sequences, delays


def logit(p):
    """概率 -> logit。"""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def publish_decisions(logit_net, evidence, delays, weight, t_logit, delta, d_max):
    """一条序列的证据决定发布。

    输入: logit_net [N]；evidence {d: [N]}；delays 可用延迟（升序）；weight 融合权重 w；
          t_logit 决策阈值的 logit；delta 犹豫区半宽 Δ；d_max 最长等待（只用 <= d_max 的延迟）。
    返回: (pred 布尔 [N], wait 等待窗数 [N] int)
    """
    cols = [0] + [d for d in delays if d <= int(d_max)]
    z = np.stack([logit_net] + [logit_net + weight * evidence[d] for d in cols[1:]], 1)      # [N, 列]
    decisive = np.abs(z - t_logit) >= float(delta)
    decisive[:, -1] = True                                   # 等到 d_max 仍不确定也必须发布
    first = np.argmax(decisive, axis=1)
    rows = np.arange(z.shape[0])
    return z[rows, first] >= t_logit, np.asarray(cols, dtype=np.int64)[first]


HARD_MARGIN = 1.0      # 困难目标事件：网络 logit 离决策阈值不到 1（与 Δ 无关，各读出用同一个定义）


def first_detection_latencies(seq, pred, wait, window_ms, n_windows):
    """一条序列里每个目标的首次检出延迟（ms，未检出为 None）：第一个被判为正的目标事件的发布窗末 - 目标首个事件时间。"""
    t = seq["locs"][:, 3].astype(np.int64)
    positive = seq["labels"] == 1
    tid = seq["target_id"]
    publish = np.minimum(t // int(window_ms) + wait, int(n_windows) - 1)
    out = []
    for target in np.unique(tid[positive]):
        if target == 0:
            continue
        mask = positive & (tid == target)
        hit = mask & pred
        out.append(float((publish[hit].min() + 1) * int(window_ms) - t[mask].min()) if hit.any() else None)
    return out


def evaluate(sequences, decide, thresholds, with_pd, pd_detT, correct_thresh, label, window_ms=50, n_windows=160):
    """对一种读出扫阈值。decide(seq, t) -> (pred, wait)。返回每个阈值一行。"""
    rows = []
    for threshold in thresholds:
        preds, waits, binaries, latencies, hard = [], [], [], [], []
        t_logit = logit(threshold)
        for seq in sequences:
            pred, wait = decide(seq, threshold)
            preds.append(pred)
            waits.append(wait)
            hard.append(np.abs(seq["logit"] - t_logit) < HARD_MARGIN)
            binaries.append(dict(seq, probs=pred.astype(np.float32)))
            latencies += first_detection_latencies(seq, pred, wait, window_ms, n_windows)
        pred_all, wait_all, hard_all = np.concatenate(preds), np.concatenate(waits), np.concatenate(hard)
        labels = np.concatenate([s["labels"] for s in sequences])
        row = {"threshold": float(threshold)}
        row.update(event_metrics(labels, pred_all.astype(np.float32), 0.5))
        target = labels == 1
        hard_target = target & hard_all
        lat = np.array([v for v in latencies if v is not None], dtype=np.float64)
        row.update({"wait_mean_all": float(wait_all.mean()),
                    "wait_mean_target": float(wait_all[target].mean()) if target.any() else float("nan"),
                    "wait_p90_target": float(np.percentile(wait_all[target], 90)) if target.any() else float("nan"),
                    "wait_p99_target": float(np.percentile(wait_all[target], 99)) if target.any() else float("nan"),
                    "wait_mean_hard_target": float(wait_all[hard_target].mean()) if hard_target.any() else float("nan"),
                    "hard_target_events": int(hard_target.sum()),
                    "wait_mean_predicted": float(wait_all[pred_all].mean()) if pred_all.any() else float("nan"),
                    "targets": len(latencies), "targets_detected": int(lat.size),
                    "latency_mean_ms": float(lat.mean()) if lat.size else None,
                    "latency_median_ms": float(np.median(lat)) if lat.size else None,
                    "latency_p90_ms": float(np.percentile(lat, 90)) if lat.size else None})
        if with_pd:
            row.update(detection_metrics(binaries, 0.5, pd_detT, correct_thresh))
        rows.append(row)
        print("  %-22s 阈值 %.3f | IoU %.4f 召回 %.4f 精确率 %.4f | 等待 目标 %.2f 困难目标 %.2f 全部 %.3f 窗 | "
              "首检延迟中位数 %s ms%s" % (
                  label, threshold, row["iou"], row["recall"], row["precision"], row["wait_mean_target"],
                  row["wait_mean_hard_target"], row["wait_mean_all"], row["latency_median_ms"],
                  (" | Pd %.4f Fa %.2e" % (row["pd"], row["fa"])) if with_pd else ""), flush=True)
    return rows


def at_fa(rows, targets):
    """在各目标虚警率处插值（IoU、Pd、以及等待）。"""
    out = {}
    for target in targets:
        point = interpolate_at_fa(rows, target)
        if point is not None:
            pts = sorted([r for r in rows if r.get("fa", 0) > 0], key=lambda r: r["fa"])
            for low, high in zip(pts, pts[1:]):
                if low["fa"] <= target <= high["fa"]:
                    span = np.log(high["fa"]) - np.log(low["fa"])
                    w = 0.0 if span == 0 else (np.log(target) - np.log(low["fa"])) / span
                    for key in ("wait_mean_target", "wait_mean_all", "wait_mean_hard_target"):
                        point[key] = float(low[key] + w * (high[key] - low[key]))
                    break
        out[str(target)] = point
    return out


def main():
    """入口：读导出 -> 固定 d 参照 -> 各 (d_max, Δ) 的证据决定发布 -> 等虚警对比 -> 写 JSON。"""
    args = parse_args()
    sequences, delays = load_scores(args.dump_dir, args.max_sequences)
    print("%s | %d 条序列 | 可用延迟 %s | w = %g" % (args.dump_dir, len(sequences), delays, args.fusion_weight),
          flush=True)
    with_pd = not args.no_pd
    w = float(args.fusion_weight)
    out = {"dump_dir": args.dump_dir, "delays": delays, "fusion_weight": w, "thresholds": args.thresholds,
           "target_fa": args.target_fa, "fixed": {}, "adaptive": {}}
    if not args.no_fixed:
        for d in [0] + [d for d in delays if d in args.max_delays]:
            def fixed(seq, t, d=d):
                z = seq["logit"] if d == 0 else seq["logit"] + w * seq["evidence"][d]
                return z >= logit(t), np.full(z.shape[0], d, dtype=np.int64)
            rows = evaluate(sequences, fixed, args.thresholds, with_pd, args.pd_detT, args.correct_thresh,
                            "固定 d=%d" % d, args.window_ms, args.n_windows)
            out["fixed"][str(d)] = {"rows": rows, "at_fa": at_fa(rows, args.target_fa) if with_pd else {}}
    for d_max in args.max_delays:
        if not any(d <= d_max for d in delays):
            print("跳过 d_max=%d：导出里没有 <= %d 的延迟" % (d_max, d_max))
            continue
        for delta in args.deltas:
            def adaptive(seq, t, d_max=d_max, delta=delta):
                return publish_decisions(seq["logit"], seq["evidence"], delays, w, logit(t), delta, d_max)
            rows = evaluate(sequences, adaptive, args.thresholds, with_pd, args.pd_detT, args.correct_thresh,
                            "d_max=%d Δ=%g" % (d_max, delta), args.window_ms, args.n_windows)
            out["adaptive"]["dmax%d_delta%g" % (d_max, delta)] = {
                "d_max": d_max, "delta": delta, "rows": rows, "at_fa": at_fa(rows, args.target_fa) if with_pd else {}}
    if with_pd:
        for target in args.target_fa:
            print("\n=== 等虚警率 %.2e ===" % target)
            print("%-24s %8s %8s %12s %14s" % ("读出", "IoU", "Pd", "目标平均等待", "困难目标等待"))
            for group in ("fixed", "adaptive"):
                for name, data in out[group].items():
                    point = data["at_fa"].get(str(target))
                    tag = ("固定 d=" + name) if group == "fixed" else name
                    if point is None:
                        print("%-24s %s" % (tag, "该虚警率不在扫描范围内"))
                    else:
                        print("%-24s %8.4f %8.4f %12.2f %14.2f" % (
                            tag, point["iou"], point["pd"], point.get("wait_mean_target", float("nan")),
                            point.get("wait_mean_hard_target", float("nan"))))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(out, stream, indent=1, ensure_ascii=False)
    print("\n报告:", args.out)


if __name__ == "__main__":
    main()
