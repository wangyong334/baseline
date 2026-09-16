"""流式 SNN V1 的附加评估指标（只依赖 numpy，可本地单元测试）。

主指标 IoU/ACC/Pd/Fa 不在这里实现：必须回填后调用原仓库 utils/eval.py 的函数，
保证与基线 0.6188 口径一致。本文件只负责逐窗诊断与首次检出延迟。
"""
import numpy as np


def window_confusion(pred_prob, label, threshold):
    """统计一个窗口的 TP/FP/FN 与正事件数（事件级，阈值 threshold）。

    输入: pred_prob 预测概率、label 标签 0/1，长度相同的一维数组。
    输出: (tp, fp, fn, n_pos) 四个整数；空窗口返回全 0。
    """
    if pred_prob.shape[0] == 0:
        return 0, 0, 0, 0
    pred = pred_prob >= float(threshold)
    pos = label == 1
    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    return tp, fp, fn, int(np.count_nonzero(pos))


def iou_from_counts(tp, fp, fn):
    """由累计的 TP/FP/FN 计算 IoU；分母为 0（该段既无正事件也无误报）时返回 NaN。"""
    union = tp + fp + fn
    return float(tp) / float(union) if union > 0 else float("nan")


def rolling_iou(tp, fp, fn, width):
    """滚动窗口 IoU：第 i 项 = sum(TP[i:i+width]) / sum(TP+FP+FN)[i:i+width]。

    先累计再相除（不是平均单窗 IoU），避免事件很少的窗口主导曲线。
    输入为逐窗数组（可以是多个序列按窗口序号相加后的结果），输出长度 n-width+1。
    """
    tp, fp, fn = (np.asarray(a, dtype=np.float64) for a in (tp, fp, fn))
    n = tp.shape[0]
    if width < 1 or width > n:
        raise ValueError("width 必须在 [1, n] 内")
    kernel = np.ones(int(width))
    s_tp = np.convolve(tp, kernel, mode="valid")
    s_union = np.convolve(tp + fp + fn, kernel, mode="valid")
    out = np.full(s_tp.shape, np.nan)
    ok = s_union > 0
    out[ok] = s_tp[ok] / s_union[ok]
    return out


def segment_iou(tp, fp, fn, segments):
    """分段 IoU：对每个闭区间 [a, b] 累计 TP/FP/FN 后计算。

    V1 默认分段 [0,15] 冷启动、[64,79] 中段稳态、[144,159] 序列末端。
    输出: {"a-b": IoU}。
    """
    out = {}
    for a, b in segments:
        sl = slice(int(a), int(b) + 1)
        out["%d-%d" % (a, b)] = iou_from_counts(int(np.sum(tp[sl])), int(np.sum(fp[sl])),
                                                int(np.sum(fn[sl])))
    return out


def first_detection_latencies(t, label, target_id, pred_prob, window_ms, threshold,
                              correct_thresh, net_ms_per_window):
    """计算一个序列内每个目标的首次检出延迟。

    检出条件（沿用原 Pd 判定）：某窗口内该目标的正标签事件中，
        预测为正的数量 / 该目标本窗正事件数 >= correct_thresh
    额外要求至少 1 个事件被预测为正：correct_thresh=1e-4（默认配置）时两者等价，
    只有 correct_thresh=0 时才会不同（原判定会把"一个都没预测对"也算检出）。
    延迟定义：
        latency = (k_d + 1) * window_ms + net_ms_per_window - t_first
        k_d      首个满足检出条件的窗口
        t_first  该目标第一个正标签事件的时间
    即包含"等窗口结束"的时间与网络计算时间；同窗检出也不会得到 0。
    target_id == 0 视为非目标（与原评估代码一致）。
    输出: 列表，每项 {"target_id", "t_first_ms", "detect_window", "latency_ms"}，未检出时后两项为 None。
    """
    results = []
    positive = label == 1
    ids = np.unique(target_id[positive])
    for tid in ids:
        if tid == 0:
            continue
        mask = positive & (target_id == tid)
        times = t[mask]
        t_first = int(times.min())
        windows = times // int(window_ms)
        hit = pred_prob[mask] >= float(threshold)
        detect = None
        for k in np.unique(windows):
            in_k = windows == k
            n_k = int(np.count_nonzero(in_k))
            n_hit = int(np.count_nonzero(hit & in_k))
            if n_k and float(n_hit) / float(n_k) >= float(correct_thresh) and n_hit > 0:
                detect = int(k)
                break
        latency = None
        if detect is not None:
            latency = float((detect + 1) * int(window_ms) + float(net_ms_per_window) - t_first)
        results.append({"target_id": float(tid), "t_first_ms": t_first,
                        "detect_window": detect, "latency_ms": latency})
    return results


def summarize_latencies(records):
    """汇总所有序列的首次检出延迟：目标数、检出数、检出率、延迟均值/中位数/P90（只统计已检出目标）。"""
    lat = np.array([r["latency_ms"] for r in records if r["latency_ms"] is not None], dtype=np.float64)
    n = len(records)
    out = {"n_targets": n, "n_detected": int(lat.size),
           "detection_rate": float(lat.size) / n if n else float("nan")}
    if lat.size:
        out.update({"latency_mean_ms": float(lat.mean()),
                    "latency_median_ms": float(np.median(lat)),
                    "latency_p90_ms": float(np.percentile(lat, 90))})
    return out
