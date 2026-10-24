"""v6-2（V2/V3 判决层接 V5）的评估：四个指标（阈值 0.9）、逐序列计数、相对 V5 的纠正 / 破坏数、发布统计（numpy）。

IoU / ACC   全部序列拼接：pred = float32 概率 >= float32 阈值（与 utils/eval.py 的 evaluate_iou_and_accuracy 相同）
Pd / Fa     utils.stream_metrics.FastRoc（与原 roc_update / cal_roc 逐位等价）
纠正 / 破坏 相对 V5 零等待（同一阈值）：纠正 = V5 错、现在对；破坏 = V5 对、现在错。分组与 B1（v6-1）相同：
            target 目标事件；onset 目标在本段录像出现的第一窗里的目标事件；
            near_bg 同一窗内离某个目标事件不超过 3 像素（切比雪夫距离）的背景事件；far_bg 其余背景事件
"""
import numpy as np

from model.publish_readout import REASONS

GROUP_NAMES = ("target", "onset", "near_bg", "far_bg")
NEAR_RADIUS = 3


def iou_counts(prob, label, threshold):
    pred = np.asarray(prob, dtype=np.float32) >= np.float32(threshold)
    pos = np.asarray(label) == 1
    return (int(np.count_nonzero(pred & pos)), int(np.count_nonzero(pred & ~pos)),
            int(np.count_nonzero(~pred & pos)))


def ratio(a, b):
    return float(a) / float(b) if b else float("nan")


def split_report(seqs, probs, cfg):
    """seqs：每条序列的 dict（name, t, x, y, label, target_id）；probs：对应的 float32 概率列表。"""
    from utils.stream_metrics import FastRoc
    thr = float(cfg["threshold"])
    roc = FastRoc(cfg["pd_detT"], cfg["correct_thresh"], cfg["height"], cfg["width_px"])
    tp = fp = fn = 0
    per_sequence = []
    for s, p in zip(seqs, probs):
        roc.update(s["t"], p, s["target_id"], s["label"], s["x"], s["y"], thr)
        a, b, c = iou_counts(p, s["label"], thr)
        tp, fp, fn = tp + a, fp + b, fn + c
        per_sequence.append({"name": s["name"], "tp": a, "fp": b, "fn": c,
                             "iou": ratio(a, a + b + c) if a + b + c else None})
    pd, fa = roc.result()
    return {"iou": ratio(tp, tp + fp + fn), "acc": ratio(tp, tp + fn), "pd": float(pd), "fa": float(fa),
            "tp": tp, "fp": fp, "fn": fn, "per_sequence": per_sequence}


def event_windows(seq, window_ms):
    return (np.asarray(seq.t, dtype=np.int64) // int(window_ms)).astype(np.int64)


def near_target_mask(seq, window_ms, radius=NEAR_RADIUS):
    """背景事件中，同一窗内离某个目标事件不超过 radius 像素（切比雪夫距离）的那些（文件顺序，bool）。"""
    n = int(seq.t.shape[0])
    target = np.asarray(seq.label) == 1
    out = np.zeros(n, dtype=bool)
    if not np.any(target):
        return out
    win = event_windows(seq, window_ms)
    x, y = np.asarray(seq.x, dtype=np.int64), np.asarray(seq.y, dtype=np.int64)
    width = int(x.max()) + 2 * radius + 2
    height = int(y.max()) + 2 * radius + 2
    plane = width * height

    def key(w, yy, xx):
        return (w * height + (yy + radius)) * width + (xx + radius)

    base = np.unique(key(win[target], y[target], x[target]))
    wb, rem = base // plane, base % plane
    yb, xb = rem // width - radius, rem % width - radius
    parts = [key(wb, yb + dy, xb + dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    near = np.unique(np.concatenate(parts))
    bg = ~target
    k = key(win[bg], y[bg], x[bg])
    pos = np.minimum(np.searchsorted(near, k), near.size - 1)
    out[np.flatnonzero(bg)] = near[pos] == k
    return out


def onset_mask(seq, window_ms):
    """目标事件（标签 1 且 target_id 非 0）中，处在该目标出现的第一窗里的那些。"""
    n = int(seq.t.shape[0])
    tid = np.round(np.asarray(seq.target_id, dtype=np.float64)).astype(np.int64)
    target = (np.asarray(seq.label) == 1) & (tid != 0)
    out = np.zeros(n, dtype=bool)
    if not np.any(target):
        return out
    ids = np.flatnonzero(target)
    win = event_windows(seq, window_ms)[ids]
    first = {}
    for t_id, w in zip(tid[ids], win):
        if t_id not in first or w < first[t_id]:
            first[t_id] = w
    out[ids] = win == np.array([first[v] for v in tid[ids]], dtype=np.int64)
    return out


def event_groups(seq, window_ms):
    target = np.asarray(seq.label) == 1
    near = near_target_mask(seq, window_ms)
    return {"target": target, "onset": onset_mask(seq, window_ms) & target, "near_bg": near & ~target,
            "far_bg": ~near & ~target}


def flip_counts(seqs, base_probs, probs, threshold):
    """相对 base（V5）的纠正数与破坏数，按 GROUP_NAMES 分组（每条序列的 groups 已算好）。"""
    thr = np.float32(threshold)
    out = {}
    for name in GROUP_NAMES:
        fixed = broken = total = 0
        for s, p0, p1 in zip(seqs, base_probs, probs):
            mask = s["groups"][name]
            pos = np.asarray(s["label"])[mask] == 1
            ok0 = (np.asarray(p0, dtype=np.float32)[mask] >= thr) == pos
            ok1 = (np.asarray(p1, dtype=np.float32)[mask] >= thr) == pos
            fixed += int(np.count_nonzero(~ok0 & ok1))
            broken += int(np.count_nonzero(ok0 & ~ok1))
            total += int(mask.sum())
        out[name] = {"events": total, "corrected": fixed, "broken": broken}
    return out


def publish_summary(labels, ages, reasons, deadline, window_ms):
    """pub 的发布年龄与原因（目标事件 / 全部事件），口径同 v3 utils/publish_eval.PublishStats。"""
    lab = np.concatenate([np.asarray(v) == 1 for v in labels])
    age = np.concatenate(ages).astype(np.int64)
    reason = np.concatenate(reasons).astype(np.int64)
    D = int(deadline)
    hist_t = np.bincount(age[lab], minlength=D + 1)[:D + 1]
    hist_a = np.bincount(age, minlength=D + 1)[:D + 1]
    nt, na = max(int(hist_t.sum()), 1), max(int(hist_a.sum()), 1)
    ages_axis = np.arange(D + 1)
    return {"deadline": D,
            "age_hist_target": (hist_t / float(nt)).tolist(),
            "age_hist_all": (hist_a / float(na)).tolist(),
            "mean_delay_target_ms": float(window_ms) * float((ages_axis * hist_t).sum()) / nt,
            "mean_delay_all_ms": float(window_ms) * float((ages_axis * hist_a).sum()) / na,
            "reason_target": dict(zip(REASONS, (np.bincount(reason[lab], minlength=len(REASONS)) / float(nt)).tolist())),
            "reason_all": dict(zip(REASONS, (np.bincount(reason, minlength=len(REASONS)) / float(na)).tolist()))}
