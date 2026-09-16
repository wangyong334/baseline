"""诊断基线与流式 V1 的精度差从哪来（只读预测 dump，不需要 GPU / spconv）。

用法（在服务器上，指向 dump_baseline_predictions.py 与 eval 导出的目录）：
    python tools/diagnose_gap.py \
        --pred baseline=dump/baseline_test \
        --pred stream_v1=dump/stream_v1_test \
        --out-json log/diagnose_gap.json

每个 NPZ 需要 locs [N,4] = (batch, x, y, t_ms)、labels [N]、probabilities [N]，
与 tools/verify_predictions.py 的格式一致。

检查三件事：
  A. 时间下采样漏层假设：基线在 t%4==2 的事件上是否显著更差。
     evspsegnet.py 的三次下采样都是 kernel=3 / stride=4 / padding=1，
     规则 p = o*4 - 1 + k (k=0,1,2) 永远命中不到 p%4==2，
     所以这 25% 的体素拿不到任何解码器上下文。V1 不知道 t%4，天然是对照组。
  B. V1 读出粒度的上限：标签在 (50ms 窗, x, y) 粒度上是否纯净。
     V1 的网络特征按像素读出，同窗同像素的事件共享特征，
     若标签在该粒度上不纯，V1 结构上就够不到满分。
  C. 正样本比例与工作点：各方法的召回 / 虚警构成。
"""
import argparse
import glob
import json
import os

import numpy as np


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="基线 vs 流式 V1 的精度差诊断")
    parser.add_argument("--pred", action="append", required=True,
                        help="名称=预测目录，可重复；例如 baseline=dump/baseline_test")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--window-ms", type=int, default=50)
    parser.add_argument("--pd-detT", type=int, default=50,
                        help="原 roc_update 的等效帧长，用于 D 组口径量化")
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def load_dir(directory):
    """读取目录下全部 NPZ，返回 {文件名: 字典}，字段与 verify_predictions.py 一致。"""
    files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not files:
        raise SystemExit("目录中没有 NPZ: %s" % directory)
    out = {}
    for f in files:
        with np.load(f) as d:
            item = {k: np.asarray(d[k]) for k in d.files}
        for key in ("locs", "labels", "probabilities"):
            if key not in item:
                raise SystemExit("%s 缺少字段 %s" % (f, key))
        n = item["labels"].shape[0]
        if item["locs"].shape != (n, 4):
            raise SystemExit("%s 的 locs 形状应为 (N,4)" % f)
        item["probabilities"] = item["probabilities"].reshape(-1)
        out[os.path.basename(f)] = item
    return out


def concat_methods(methods):
    """把每个方法的全部序列按文件名排序后拼接，并核对各方法之间逐元素对齐。

    返回 {方法名: {"t","x","y","label","prob"}}；若任一方法的坐标或标签与
    第一个方法不一致就直接报错——对不齐的两组预测没有比较的意义。
    """
    names = sorted(methods)
    stacked = {}
    for name in names:
        data = methods[name]
        keys = sorted(data)
        locs = np.concatenate([data[k]["locs"] for k in keys], axis=0)
        stacked[name] = {
            "t": locs[:, 3].astype(np.int64),
            "x": locs[:, 1].astype(np.int64),
            "y": locs[:, 2].astype(np.int64),
            "label": np.concatenate([data[k]["labels"] for k in keys]).astype(np.float64),
            "prob": np.concatenate([data[k]["probabilities"] for k in keys]).astype(np.float64),
        }
    ref = stacked[names[0]]
    for name in names[1:]:
        cur = stacked[name]
        for key in ("t", "x", "y", "label"):
            if cur[key].shape != ref[key].shape or not np.array_equal(cur[key], ref[key]):
                raise SystemExit("方法 %s 与 %s 的 %s 不一致，无法比较" % (name, names[0], key))
    return stacked, names


def group_metrics(label, prob, mask, threshold):
    """在 mask 选中的事件子集上计算 IoU / 召回 / 负样本误报率。"""
    lab, pr = label[mask], prob[mask]
    pos = lab == 1
    binary = pr >= threshold
    inter = int(np.count_nonzero(pos & binary))
    union = int(np.count_nonzero(pos | binary))
    n_pos = int(np.count_nonzero(pos))
    n_neg = int(mask.sum()) - n_pos
    return {
        "events": int(mask.sum()),
        "pos_ratio": (n_pos / int(mask.sum())) if mask.sum() else float("nan"),
        "iou": (inter / union) if union else float("nan"),
        "recall": (inter / n_pos) if n_pos else float("nan"),
        "fp_rate_on_neg": (int(np.count_nonzero((~pos) & binary)) / n_neg) if n_neg else float("nan"),
    }


def check_a_temporal_residue(stacked, names, threshold):
    """A. 按 t%4 分组比较各方法；基线若在残差 2 上明显更差，则漏层假设成立。"""
    ref = stacked[names[0]]
    residue = ref["t"] % 4
    rows = []
    for r in range(4):
        mask = residue == r
        row = {"residue": r, "share": float(mask.mean())}
        for name in names:
            row[name] = group_metrics(ref["label"], stacked[name]["prob"], mask, threshold)
        rows.append(row)
    summary = {}
    for name in names:
        orphan = residue == 2
        keep = ~orphan
        a = group_metrics(ref["label"], stacked[name]["prob"], orphan, threshold)
        b = group_metrics(ref["label"], stacked[name]["prob"], keep, threshold)
        summary[name] = {
            "orphan_iou": a["iou"], "kept_iou": b["iou"],
            "iou_drop": b["iou"] - a["iou"],
            "orphan_recall": a["recall"], "kept_recall": b["recall"],
            "recall_drop": b["recall"] - a["recall"],
        }
    return {"per_residue": rows, "orphan_vs_kept": summary}


def check_b_readout_purity(stacked, names, window_ms):
    """B. 标签在 (窗, x, y) 粒度上的纯净度，决定 V1 逐像素读出的结构上限。"""
    ref = stacked[names[0]]
    window = ref["t"] // int(window_ms)
    # 混合进位，各维留足余量（y<512, x<1024），保证不同 (窗,y,x) 不会碰撞
    key = ((window.astype(np.int64) * 512 + ref["y"]) * 1024 + ref["x"]).astype(np.int64)
    order = np.argsort(key, kind="stable")
    sorted_key, sorted_label = key[order], ref["label"][order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_key[1:] != sorted_key[:-1])))
    sums = np.add.reduceat(sorted_label, starts)
    sizes = np.diff(np.concatenate((starts, [sorted_key.shape[0]])))
    mixed = (sums > 0) & (sums < sizes)
    events_in_mixed = int(sizes[mixed].sum())
    # 多数票上限：每组只能给一个答案时，少数派必然判错
    minority = np.minimum(sums, sizes - sums)
    return {
        "groups": int(starts.shape[0]),
        "mixed_groups": int(mixed.sum()),
        "mixed_group_ratio": float(mixed.mean()),
        "events_in_mixed_groups": events_in_mixed,
        "events_in_mixed_ratio": float(events_in_mixed) / float(ref["label"].shape[0]),
        "unavoidable_errors_if_one_label_per_group": int(minority.sum()),
        "unavoidable_error_ratio": float(minority.sum()) / float(ref["label"].shape[0]),
        "note": "V1 读出为每个 (窗,像素) 共享网络特征，只有 p 与 t_local 能再区分；"
                "unavoidable_* 是完全不用 p/t_local 时的下界",
    }


def check_d_eval_caveats(stacked, names, threshold, det_t):
    """D. 量化原仓库 utils/eval.py 里三处已知口径问题对 Pd/Fa 的实际影响。

    这三处都在原仓库代码里，基线与 V1 同等受影响，所以**不要修改 eval.py**——
    改了就和论文 0.5518、基线 v2 0.6188 都不可比。这里只量化，不修正。

      D1 严格不等号：roc_update 用 (ts > i*T) & (ts < (i+1)*T)，
         t 恰好是 T 的整数倍时不落入任何一帧，被静默排除出 Pd/Fa（IoU/ACC 不受影响）。
      D2 分母差一：frame_num 累加 int(跨度/T)，但循环跑 int(跨度/T)+1 次，Fa 分母偏小。
      D3 uint8 溢出：false_mask 用 uint8 累加，同一像素同一帧累计到 256 会归零，
         该像素从连通域统计中消失。这里统计实际的单帧单像素最大虚警数。
    """
    ref = stacked[names[0]]
    t = ref["t"]
    n = int(t.shape[0])
    det_t = int(det_t)

    on_boundary = (t % det_t) == 0
    n_b = int(np.count_nonzero(on_boundary))
    pos = ref["label"] == 1
    out = {
        "D1_boundary_excluded": {
            "events": n_b,
            "ratio": n_b / n if n else float("nan"),
            "positive_events": int(np.count_nonzero(on_boundary & pos)),
            "pos_ratio_on_boundary": (float(np.count_nonzero(on_boundary & pos)) / n_b) if n_b else float("nan"),
            "pos_ratio_overall": float(np.count_nonzero(pos)) / n if n else float("nan"),
        },
        "D2_frame_denominator": {
            "loops": int((t.max() - t.min()) / det_t + 1),
            "frame_num": int((t.max() - t.min()) / det_t),
            "fa_inflation": 1.0 / int((t.max() - t.min()) / det_t),
        },
    }
    # D3：按 (帧, 像素) 统计各方法的虚警计数峰值
    frame = t // det_t
    key = ((frame.astype(np.int64) * 512 + ref["y"]) * 1024 + ref["x"]).astype(np.int64)
    d3 = {}
    for name in names:
        fp = (~pos) & (stacked[name]["prob"] >= threshold)
        if not np.count_nonzero(fp):
            d3[name] = {"max_count_per_pixel_frame": 0, "overflow_risk": False}
            continue
        k = np.sort(key[fp])
        starts = np.flatnonzero(np.concatenate(([True], k[1:] != k[:-1])))
        sizes = np.diff(np.concatenate((starts, [k.shape[0]])))
        peak = int(sizes.max())
        d3[name] = {"max_count_per_pixel_frame": peak, "overflow_risk": peak >= 256,
                    "pixels_at_risk": int(np.count_nonzero(sizes >= 256))}
    out["D3_uint8_overflow"] = d3
    return out


def check_c_operating_point(stacked, names, threshold):
    """C. 各方法在全集上的工作点：整体 IoU / 召回 / 误报，以及预测概率分布。"""
    ref = stacked[names[0]]
    out = {}
    for name in names:
        prob = stacked[name]["prob"]
        full = group_metrics(ref["label"], prob, np.ones_like(ref["label"], dtype=bool), threshold)
        pos = ref["label"] == 1
        full["prob_quantiles_on_pos"] = [float(np.quantile(prob[pos], q)) for q in (0.1, 0.5, 0.9)]
        full["prob_quantiles_on_neg"] = [float(np.quantile(prob[~pos], q)) for q in (0.5, 0.9, 0.99)]
        full["pred_positive_ratio"] = float(np.mean(prob >= threshold))
        out[name] = full
    return out


def main():
    """入口：加载预测，跑 A/B/C 三项检查，打印并可选写 JSON。"""
    args = parse_args()
    methods = {}
    for spec in args.pred:
        if "=" not in spec:
            raise SystemExit("--pred 需要 名称=目录 的形式，收到 %s" % spec)
        name, directory = spec.split("=", 1)
        methods[name] = load_dir(directory)
    stacked, names = concat_methods(methods)
    print("对齐通过：%d 个事件，%d 个方法 %s" % (stacked[names[0]]["label"].shape[0], len(names), names))

    a = check_a_temporal_residue(stacked, names, args.threshold)
    print("\n【A】按 t%4 分组（基线的时间下采样命中不到 t%4==2）")
    header = "  残差  占比   " + "".join("%-28s" % n for n in names)
    print(header)
    for row in a["per_residue"]:
        cells = "".join("IoU %.4f 召回 %.4f 误报 %.2e  " % (
            row[n]["iou"], row[n]["recall"], row[n]["fp_rate_on_neg"]) for n in names)
        print("  %d     %.3f  %s" % (row["residue"], row["share"], cells))
    print("  --- 残差2 vs 其余 ---")
    for n in names:
        s = a["orphan_vs_kept"][n]
        print("  %-12s IoU %.4f vs %.4f (差 %+.4f) | 召回 %.4f vs %.4f (差 %+.4f)" % (
            n, s["orphan_iou"], s["kept_iou"], s["iou_drop"],
            s["orphan_recall"], s["kept_recall"], s["recall_drop"]))

    b = check_b_readout_purity(stacked, names, args.window_ms)
    print("\n【B】标签在 (%dms 窗, 像素) 粒度上的纯净度 —— V1 逐像素读出的上限" % args.window_ms)
    print("  分组数 %d，混合组 %d (%.3f%%)，落在混合组的事件 %d (%.3f%%)" % (
        b["groups"], b["mixed_groups"], 100 * b["mixed_group_ratio"],
        b["events_in_mixed_groups"], 100 * b["events_in_mixed_ratio"]))
    print("  每组只给一个标签时的必然错误 %d (%.4f%%)" % (
        b["unavoidable_errors_if_one_label_per_group"], 100 * b["unavoidable_error_ratio"]))

    c = check_c_operating_point(stacked, names, args.threshold)
    print("\n【C】工作点（阈值 %.2f）" % args.threshold)
    for n in names:
        m = c[n]
        print("  %-12s IoU %.4f 召回 %.4f 负样本误报 %.3e 预测为正占比 %.4f" % (
            n, m["iou"], m["recall"], m["fp_rate_on_neg"], m["pred_positive_ratio"]))
        print("               正样本概率 p10/p50/p90 = %.3f/%.3f/%.3f | 负样本 p50/p90/p99 = %.3f/%.3f/%.3f" % (
            tuple(m["prob_quantiles_on_pos"]) + tuple(m["prob_quantiles_on_neg"])))

    d = check_d_eval_caveats(stacked, names, args.threshold, args.pd_detT)
    print("\n【D】原仓库 eval.py 的三处口径问题（只量化，不修改 eval.py）")
    b1 = d["D1_boundary_excluded"]
    print("  D1 严格不等号排除的事件: %d (%.3f%%)，其中正样本 %d" % (
        b1["events"], 100 * b1["ratio"], b1["positive_events"]))
    print("     被排除子集的正样本比例 %.4f vs 全集 %.4f  →  %s" % (
        b1["pos_ratio_on_boundary"], b1["pos_ratio_overall"],
        "分布接近，对 Pd 近似无偏" if abs(b1["pos_ratio_on_boundary"] - b1["pos_ratio_overall"])
        < 0.2 * max(b1["pos_ratio_overall"], 1e-9) else "分布明显偏离，需单独说明"))
    b2 = d["D2_frame_denominator"]
    print("  D2 循环 %d 次但 frame_num 记 %d  →  Fa 偏大约 %.2f%%" % (
        b2["loops"], b2["frame_num"], 100 * b2["fa_inflation"]))
    for name in names:
        b3 = d["D3_uint8_overflow"][name]
        print("  D3 %-12s 单帧单像素虚警峰值 %d  →  %s" % (
            name, b3["max_count_per_pixel_frame"],
            "【有溢出风险】%d 个像素 >= 256" % b3.get("pixels_at_risk", 0)
            if b3["overflow_risk"] else "远低于 256，uint8 未溢出"))

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"temporal_residue": a, "readout_purity": b, "operating_point": c,
                       "eval_caveats": d}, fh, indent=2, ensure_ascii=False)
        print("\n已写入 %s" % args.out_json)


if __name__ == "__main__":
    main()
