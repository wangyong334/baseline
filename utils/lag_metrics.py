"""V6 第一版的判决与指标（只依赖 numpy，可本地单元测试）。

开发阶段只报四个指标（阈值 0.9），口径与 V5 相同：
    IoU / ACC   iou_acc_counts 与原 utils/eval.py 的 evaluate_iou_and_accuracy 逐位相同（float32 概率 >= float32 阈值，
                全部序列拼接后计算）；单元测试逐项核对
    Pd / Fa     utils.stream_metrics.FastRoc（与原 roc_update / cal_roc 逐位等价），按事件自己的时间戳分帧
另报机制诊断（不是指标），以及每条序列的 tp / fp / fn / IoU（定位极密序列等误差来源）：
    age_hist        各发布年龄的事件数：检查判决是否真的在自适应
    fp_near_target  误检中落在同窗目标事件 3 像素内的个数：V3 发现的"存在证据泄漏"（等待把目标的存在证据分给邻近背景）
延迟类指标（发布延迟、首检延迟、网络耗时）与能耗一样，放到最后阶段统一计算（用户 10-23 定）。
判决（adaptive_publish，13.12 第一版规则）：
    年龄 d < 截止 D 时，概率 >= 上界 U_d 发布"目标"，<= 下界 L_d 发布"背景"，否则继续等；d = D 时按阈值强制发布；
    U_d = sigmoid(logit(thr) + m * (1 - d/D))，L_d = sigmoid(logit(thr) - m * (1 - d/D))，在 float32 上比较。
    m = 0 时所有事件在出生窗发布（= 零等待），m = inf 时都等到 D（= 固定等 D 窗）。
    录像末尾读不满 D 个滞后的事件，在最后可读的年龄发布（eos）。发布的概率就是该年龄的概率，所以发布标签 = 概率 >= 阈值。
"""
import math

import numpy as np

REASONS = ("upper", "lower", "deadline", "eos")
LEAK_RADIUS = 3


def iou_acc_counts(prob, label, threshold):
    """全局拼接口径的 IoU 与 ACC（正事件召回）。prob 为 float32 概率，label 为 0/1。

    返回 (iou, acc, tp, fp, fn)；与 utils/eval.py 的 evaluate_iou_and_accuracy 相同：binary = prob >= thr（float32）。
    """
    pred = np.asarray(prob, dtype=np.float32) >= np.float32(threshold)
    pos = np.asarray(label) == 1
    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    union = tp + fp + fn
    iou = tp / float(union) if union else float("nan")
    acc = tp / float(tp + fn) if tp + fn else float("nan")
    return iou, acc, tp, fp, fn


def adaptive_publish(prob, avail, deadline, margin, threshold):
    """13.12 第一版发布规则（见模块说明）。

    输入: prob [D+1, N] float32（各年龄的概率；读不到的年龄已用最后可读年龄填充）；avail [N] 最后可读年龄；
          deadline 截止年龄（<= D）；margin m >= 0（可为 inf）；threshold 最终阈值（0.9）
    输出: (prob_pub float32 [N], age int [N], reason int8 [N])，reason 序号见 REASONS
    """
    prob = np.asarray(prob, dtype=np.float32)
    deadline = int(deadline)
    if not 0 <= deadline < prob.shape[0]:
        raise ValueError("deadline 必须在 0..D 内")
    if not margin >= 0:
        raise ValueError("margin 必须 >= 0")
    n = prob.shape[1]
    theta = math.log(threshold / (1.0 - threshold))
    last = np.minimum(np.asarray(avail, dtype=np.int64), deadline)
    pub = np.zeros(n, dtype=np.float32)
    age = np.full(n, -1, dtype=np.int64)
    reason = np.full(n, -1, dtype=np.int8)
    open_ = np.ones(n, dtype=bool)
    for d in range(deadline + 1):
        p = prob[d]
        if d < deadline:
            s = 1.0 - float(d) / float(deadline)
            if math.isinf(margin):
                up, lo = np.float32(np.inf), np.float32(-np.inf)
            else:
                up = np.float32(1.0 / (1.0 + math.exp(-(theta + margin * s))))
                lo = np.float32(1.0 / (1.0 + math.exp(-(theta - margin * s))))
            pos = open_ & (p >= up)
            neg = open_ & ~pos & (p <= lo)
            for mask, why in ((pos, 0), (neg, 1)):
                pub[mask], age[mask], reason[mask] = p[mask], d, why
            open_ &= ~(pos | neg)
            end, why = open_ & (last == d), 3
        else:
            end, why = open_, 2
        pub[end], age[end], reason[end] = p[end], d, why
        open_ &= ~end
    return pub, age, reason


def near_target_keys(s, height, width, radius=LEAK_RADIUS):
    """一条序列中"同一窗内离某个目标事件不超过 radius 像素（切比雪夫距离）"的 (窗, 像素) 键，排好序（结果缓存在 s 里）。"""
    cache = "_near_keys_%d" % radius
    if cache not in s:
        tgt = s["label"] == 1
        plane = int(height) * int(width)
        if not np.any(tgt):
            s[cache] = np.zeros(0, dtype=np.int64)
        else:
            base = np.unique(s["win"][tgt].astype(np.int64) * plane
                             + s["y"][tgt].astype(np.int64) * int(width) + s["x"][tgt].astype(np.int64))
            win, pix = base // plane, base % plane
            ys, xs = pix // int(width), pix % int(width)
            parts = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    yy, xx = ys + dy, xs + dx
                    ok = (yy >= 0) & (yy < int(height)) & (xx >= 0) & (xx < int(width))
                    parts.append(win[ok] * plane + yy[ok] * int(width) + xx[ok])
            s[cache] = np.unique(np.concatenate(parts))
    return s[cache]


def leakage_counts(s, prob, threshold, height, width, radius=LEAK_RADIUS):
    """误检事件总数，以及其中落在同窗目标事件 radius 像素内的个数。"""
    fp = (np.asarray(prob, dtype=np.float32) >= np.float32(threshold)) & (s["label"] != 1)
    if not np.any(fp):
        return 0, 0
    plane = int(height) * int(width)
    keys = (s["win"][fp].astype(np.int64) * plane + s["y"][fp].astype(np.int64) * int(width)
            + s["x"][fp].astype(np.int64))
    near = near_target_keys(s, height, width, radius)
    if near.size == 0:
        return int(fp.sum()), 0
    pos = np.minimum(np.searchsorted(near, keys), near.size - 1)
    return int(fp.sum()), int(np.count_nonzero(near[pos] == keys))


def split_report(seqs, prob_of, age_of, cfg):
    """一个划分上的四个指标与两项机制诊断。

    seqs    列表，每项为一条序列的字典 t / x / y / label / target_id / win（numpy，长度为该序列事件数）
    prob_of 函数 i -> 第 i 条序列每个事件的发布概率（float32）
    age_of  函数 i -> 第 i 条序列每个事件的发布年龄
    cfg     需要 threshold / pd_detT / correct_thresh / height / width_px
    """
    from utils.stream_metrics import FastRoc
    thr = float(cfg["threshold"])
    roc = FastRoc(cfg["pd_detT"], cfg["correct_thresh"], cfg["height"], cfg["width_px"])
    probs, labels, ages, per_sequence = [], [], [], []
    fp_all = fp_near = 0
    for i, s in enumerate(seqs):
        prob, age = prob_of(i), age_of(i)
        roc.update(s["t"], prob, s["target_id"], s["label"], s["x"], s["y"], thr)
        a, b = leakage_counts(s, prob, thr, cfg["height"], cfg["width_px"])
        fp_all, fp_near = fp_all + a, fp_near + b
        s_iou, _, s_tp, s_fp, s_fn = iou_acc_counts(prob, s["label"], thr)
        per_sequence.append({"name": s.get("name", str(i)), "iou": s_iou, "tp": s_tp, "fp": s_fp, "fn": s_fn,
                             "fp_near_target": b, "events": int(prob.shape[0])})
        probs.append(prob)
        labels.append(s["label"])
        ages.append(np.asarray(age, dtype=np.int64))
    iou, acc, tp, fp, fn = iou_acc_counts(np.concatenate(probs), np.concatenate(labels), thr)
    pd, fa = roc.result()
    ages = np.concatenate(ages)
    return {"iou": iou, "acc": acc, "pd": float(pd), "fa": float(fa), "tp": tp, "fp": fp, "fn": fn,
            "fp_near_target": fp_near, "fp_near_target_fraction": fp_near / float(fp_all) if fp_all else 0.0,
            "mean_age": float(ages.mean()) if ages.size else float("nan"),
            "age_hist": np.bincount(ages, minlength=int(ages.max()) + 1 if ages.size else 1).tolist(),
            "per_sequence": per_sequence}
