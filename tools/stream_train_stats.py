"""阶段 1：数据审计 + 训练集统计（只依赖 numpy，不需要 GPU）。

做两件事：
    1. 审计 train/val/test 全部 NPZ：t/x/y/p/label 取值校验、窗口划分覆盖检查、每窗事件数统计
    2. 只用训练集计算：12 个输入通道在非零像素上的 log1p 计数 q99，以及 pos_weight = min(neg/pos, cap)
结果写入 YAML 中 stats_path 指定的 JSON，训练脚本从这里读取，不再重新统计。

用法: python tools/stream_train_stats.py --config configs/evisseg_stream_v1.yaml
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import stream_windows as sw  # noqa: E402
from utils.stream_common import load_flat_config  # noqa: E402


def audit_and_collect(cfg, split, collect_stats):
    """审计一个划分；collect_stats=True 时同时累计通道直方图与正负样本数。

    返回: (audit, hist, n_pos, n_neg)
        audit  文件数、事件总数、每窗事件数分布、空窗口数、无正事件窗口数、最大时间戳
        hist   每个通道的计数值直方图（collect_stats=False 时为 None）
    任一文件校验失败会直接抛出异常并指出文件名。
    """
    directory = os.path.join(cfg["root"], split)
    names = sorted(n for n in os.listdir(directory) if n.endswith(".npz"))
    n_ch = sw.num_input_channels(cfg["time_bins"])
    hist = sw.new_count_histogram(n_ch) if collect_stats else None
    n_pos = n_neg = 0
    per_window, empty, no_target, t_max, total = [], 0, 0, 0, 0
    for i, name in enumerate(names):
        path = os.path.join(directory, name)
        try:
            seq = sw.load_npz_events(path, cfg["height"], cfg["width_px"], cfg["window_ms"],
                                     cfg["n_windows"], cfg["time_bins"])
            sw.check_window_partition(seq, cfg["window_ms"])
        except Exception as error:
            raise RuntimeError("%s/%s 审计失败: %s" % (split, name, error))
        counts = np.diff(seq.bounds)
        per_window.extend(counts.tolist())
        empty += int(np.sum(counts == 0))
        total += seq.n_events
        t_max = max(t_max, int(seq.t.max()) if seq.n_events else 0)
        for k in range(seq.n_windows):
            idx = seq.window_index(k)
            if idx.size and not np.any(seq.label[idx] == 1):
                no_target += 1
            if collect_stats and idx.size:
                c = sw.count_channels(seq.x[idx], seq.y[idx], seq.p[idx], seq.inner_bin[idx],
                                      cfg["time_bins"], cfg["pad_height"], cfg["pad_width"])
                sw.update_count_histogram(hist, c)
        if collect_stats:
            pos = int(np.sum(seq.label == 1))
            n_pos += pos
            n_neg += seq.n_events - pos
        if (i + 1) % 10 == 0 or i + 1 == len(names):
            print("[%s] %d/%d" % (split, i + 1, len(names)), flush=True)
    per_window = np.asarray(per_window)
    audit = {"files": len(names), "events": total, "t_max": t_max,
             "events_per_window_mean": float(per_window.mean()),
             "events_per_window_p99": float(np.percentile(per_window, 99)),
             "events_per_window_max": int(per_window.max()),
             "empty_windows": empty, "nonempty_windows_without_target": no_target,
             "windows": int(per_window.size)}
    return audit, hist, n_pos, n_neg


def main():
    """入口：审计三个划分、统计训练集、写 JSON。已存在的统计文件不会被覆盖。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evisseg_stream_v1.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_flat_config(args.config)
    out = args.out or cfg["stats_path"]
    if os.path.exists(out):
        raise SystemExit("统计文件已存在，拒绝覆盖: %s" % out)
    t0 = time.time()
    audits = {}
    audits["train"], hist, n_pos, n_neg = audit_and_collect(cfg, "train", True)
    for split in ("val", "test"):
        audits[split] = audit_and_collect(cfg, split, False)[0]
    q99 = [sw.quantile_from_histogram(h, 0.99) for h in hist]
    stats = {
        "q99": q99,
        "channel_names": sw.channel_names(cfg["time_bins"]),
        "nonzero_pixels_per_channel": [int(h[1:].sum()) for h in hist],
        "train_pos": n_pos, "train_neg": n_neg,
        "train_neg_pos_ratio": float(n_neg) / float(n_pos),
        "pos_weight_cap": float(cfg["pos_weight_cap"]),
        "pos_weight": sw.pos_weight_from_counts(n_pos, n_neg, cfg["pos_weight_cap"]),
        "audit": audits,
        "config_used": {k: cfg[k] for k in ("window_ms", "n_windows", "time_bins", "height",
                                            "width_px", "pad_height", "pad_width")},
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as stream:
        json.dump(stats, stream, indent=2, ensure_ascii=False)
    print(json.dumps({k: stats[k] for k in ("q99", "train_neg_pos_ratio", "pos_weight")}, indent=2))
    print(json.dumps(audits, indent=2))
    print("STATS WRITTEN:", out)


if __name__ == "__main__":
    main()
