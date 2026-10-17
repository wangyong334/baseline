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


SWEEP_THRESHOLDS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)


def threshold_counts(pred_prob, label, thresholds=SWEEP_THRESHOLDS):
    """一次统计多个阈值下的事件级 TP/FP/FN（判定规则与 window_confusion 相同：prob >= 阈值）。

    输出: int64 数组 [阈值个数, 3]，可在序列之间直接相加（全局拼接口径）。
    """
    pos = np.asarray(label) == 1
    prob = np.asarray(pred_prob)
    out = np.zeros((len(thresholds), 3), dtype=np.int64)
    for i, thr in enumerate(thresholds):
        pred = prob >= float(thr)
        out[i] = (np.count_nonzero(pred & pos), np.count_nonzero(pred & ~pos), np.count_nonzero(~pred & pos))
    return out


def sweep_summary(counts, thresholds=SWEEP_THRESHOLDS):
    """把累计的阈值扫描计数换算成每个阈值的 IoU / 召回 / 精确率，并给出 IoU 最高的阈值。

    只作诊断（比较不同模型的概率尺度与排序能力）；主表仍以原 utils/eval.py 在 0.9 下的结果为准，
    原评测用严格大于号，与这里的 >= 只在概率恰好等于阈值时不同。
    """
    rows = []
    for thr, (tp, fp, fn) in zip(thresholds, np.asarray(counts)):
        tp, fp, fn = int(tp), int(fp), int(fn)
        rows.append({"threshold": float(thr), "iou": iou_from_counts(tp, fp, fn),
                     "recall": tp / float(tp + fn) if tp + fn else float("nan"),
                     "precision": tp / float(tp + fp) if tp + fp else float("nan"),
                     "tp": tp, "fp": fp, "fn": fn})
    valid = [r for r in rows if not np.isnan(r["iou"])]
    best = max(valid, key=lambda r: r["iou"]) if valid else None
    return {"rows": rows, "best_iou": best["iou"] if best else None,
            "best_threshold": best["threshold"] if best else None}


class FastRoc(object):
    """原 utils/eval.py 中 roc_update / cal_roc（Pd / Fa）的逐位等价向量化实现，传感器尺寸可配置。

    原实现对每一帧都扫一遍整条序列的全部事件（帧数 x 事件数），EV-UAV（160 帧、约 10 万事件）还可以，
    EV-Flying（一段 3600 帧、数百万事件）要几个小时。这里先算出每个事件所属的帧再分组，结果与原实现完全相同，
    连原实现的边界行为也一并保留（单元测试逐项核对）：
        帧 i 只含满足 i*T < t < (i+1)*T 的事件（严格不等号：t 恰为 T 的整数倍的事件不计入任何帧）
        帧号从绝对时间 0 起算，只取 i = 0..int((t_max - t_min)/T)；frame_num 每段加 int((t_max - t_min)/T)
        Pd：每帧每个非零 target_id 计一个目标，(预测==标签 的个数) / 标签和 >= correct_thresh 记为检出
        Fa：每帧把虚警事件（标签 0、预测 1）累加到 uint8 掩码（与原实现一样会在 256 处回绕），8 连通域数之和
        判定在 float32 上做：prob >= float32(thresh)（与 torch 张量和 Python 标量比较的结果一致）
    t、x、y 为整数值（毫秒、像素），prob 为 float32 概率，label 为 0/1，target_id 为浮点目标号（0 为背景）。
    """

    def __init__(self, pd_detT, correct_thresh, height, width):
        self.T = int(pd_detT)
        self.correct_thresh = float(correct_thresh)
        self.height, self.width = int(height), int(width)
        self.obj_num = self.correct_num = self.false_num = self.frame_num = 0

    def update(self, t, prob, target_id, label, x, y, thresh):
        import cv2   # 延迟导入：只在完整评估时需要
        t32 = np.asarray(t, dtype=np.float32)
        if t32.size == 0:
            return
        span = np.float32(t32.max()) - np.float32(t32.min())
        frames = int(span / np.float32(self.T))
        self.frame_num += frames
        ti = t32.astype(np.int64)
        frame = ti // self.T
        inside = (ti % self.T != 0) & (frame >= 0) & (frame <= frames)
        pred = (np.asarray(prob, dtype=np.float32) >= np.float32(thresh)).astype(np.float32)
        lab = np.asarray(label, dtype=np.float32)
        tid = np.asarray(target_id, dtype=np.float64)
        f, p_, l_, d_ = frame[inside], pred[inside], lab[inside], tid[inside]
        # Pd：(帧, 目标号) 分组
        obj = d_ != 0
        if np.any(obj):
            keys = np.stack([f[obj].astype(np.float64), d_[obj]], 1)
            groups, inverse = np.unique(keys, axis=0, return_inverse=True)
            equal = np.bincount(inverse, weights=(p_[obj] == l_[obj]).astype(np.float64), minlength=len(groups))
            label_sum = np.bincount(inverse, weights=l_[obj].astype(np.float64), minlength=len(groups))
            with np.errstate(divide="ignore", invalid="ignore"):
                # 原实现：int64 计数 / float32 标签和 -> float32，再与 float32 阈值比较；0/0 -> nan（不检出），k/0 -> inf（检出）
                ratio = equal.astype(np.float32) / label_sum.astype(np.float32)
            self.obj_num += int(len(groups))
            self.correct_num += int(np.count_nonzero(ratio >= np.float32(self.correct_thresh)))
        # Fa：每帧虚警像素计数（uint8 回绕）后数 8 连通域
        false = (l_ == 0) & (p_ == 1)
        if np.any(false):
            xs = np.asarray(x, dtype=np.int64)[inside][false]
            ys = np.asarray(y, dtype=np.int64)[inside][false]
            pixel_key = (f[false] * self.height + ys) * self.width + xs
            keys, counts = np.unique(pixel_key, return_counts=True)
            keep = (counts % 256) != 0
            keys = keys[keep]
            frame_of = keys // (self.height * self.width)
            pixel = keys % (self.height * self.width)
            starts = np.flatnonzero(np.r_[True, frame_of[1:] != frame_of[:-1]]) if keys.size else []
            ends = list(starts[1:]) + [keys.size] if keys.size else []
            for a, b in zip(starts, ends):
                mask = np.zeros(self.height * self.width, dtype=np.uint8)
                mask[pixel[a:b]] = 1
                n_labels = cv2.connectedComponentsWithStats(mask.reshape(self.height, self.width),
                                                            connectivity=8, ltype=cv2.CV_32S)[0]
                self.false_num += int(n_labels) - 1

    def result(self):
        """返回 (Pd, Fa)；Fa 的分母 = 帧数 x 传感器像素数。"""
        pd = self.correct_num / float(self.obj_num) if self.obj_num else float("nan")
        fa = self.false_num / float(self.frame_num * self.height * self.width) if self.frame_num else float("nan")
        return pd, fa


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

    检出条件（判定式沿用原 Pd）：某窗口内该目标的正标签事件中，
        预测为正的数量 / 该目标本窗正事件数 >= correct_thresh
    额外要求至少 1 个事件被预测为正：correct_thresh=1e-4（默认配置）时两者等价，
    只有 correct_thresh=0 时才会不同（原判定会把"一个都没预测对"也算检出）。

    ⚠ 分帧口径与原 roc_update 不同，两者的数不能互相替代：
        本函数  windows = t // window_ms          —— 含边界，每个事件恰好落入一帧
        原代码  (ts > i*T) & (ts < (i+1)*T)       —— 严格不等号，t 为 T 的整数倍时
                                                     不落入任何一帧，被排除出 Pd/Fa
    这里刻意用 t // window_ms：延迟是本仓库新增的指标，没有与论文对齐的包袱，
    正确分帧比复刻原代码的边界行为更有意义。但因此"检出窗口 k_d"与原 Pd 的
    "检出帧"在边界事件上可能不一致，报告时不要把两者当作同一口径。
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
