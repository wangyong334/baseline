"""把若干训练运行的 metrics.jsonl 汇总成一张训练过程对比表（只读日志，不读数据、不需要 GPU）。

每个运行一行：
    训练方式（trainer / B / n / k）、完成轮数、最好 val IoU 及其轮次、末 5 轮 val IoU 均值 ± 标准差、
    val IoU 首次达到 0.80 / 0.85 的轮次（收敛速度）、每轮更新次数、每轮训练秒数（中位数）、累计训练小时、
    裁剪前梯度范数中位数与被裁剪比例（各轮中位数）、显存峰值
评估指标（阈值 0.9 下的 IoU / ACC / Pd / Fa）用 tools/summarize_runs.py 汇总。

用法:
    python tools/summarize_training.py log/stream_v1_streams_b8n1_seed37 log/stream_v1_tbptt16_seed37 ...
    python tools/summarize_training.py --glob "log/stream_v1_streams_*_seed37" log/stream_v1_tbptt16_seed37 --out x.json
"""
import argparse
import glob
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="汇总训练过程")
    parser.add_argument("runs", nargs="*", help="运行目录（含 metrics.jsonl）")
    parser.add_argument("--glob", action="append", default=[], help="按通配符追加运行目录，可重复")
    parser.add_argument("--out", default=None, help="同时写一份 JSON")
    return parser.parse_args()


def first_epoch_at(records, level):
    """val IoU 首次达到 level 的轮次；从未达到返回 None。"""
    for r in records:
        if r.get("val_iou") is not None and r["val_iou"] >= level:
            return int(r["epoch"])
    return None


def summarize(directory):
    path = os.path.join(directory, "metrics.jsonl")
    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not records:
        return None
    val = np.array([r["val_iou"] for r in records], dtype=np.float64)
    best = int(np.argmax(val))
    tail = val[-5:]
    clip = [r.get("grad_clip_stats") or {} for r in records]
    last = records[-1]
    trainer = last.get("trainer", "sequence")
    return {
        "run": os.path.basename(os.path.normpath(directory)),
        "trainer": trainer,
        "streams": last.get("streams") if trainer == "streams" else None,
        "accumulate": last.get("accumulate") if trainer == "streams" else None,
        "tbptt_k": last.get("tbptt_k"),
        "epochs": len(records),
        "best_val_iou": float(val[best]), "best_epoch": int(records[best]["epoch"]),
        "last5_mean": float(tail.mean()), "last5_std": float(tail.std(ddof=1)) if tail.size > 1 else 0.0,
        "epoch_to_0.80": first_epoch_at(records, 0.80), "epoch_to_0.85": first_epoch_at(records, 0.85),
        "updates_per_epoch": int(np.median([r["updates"] for r in records])) if "updates" in last else None,
        "train_seconds_median": float(np.median([r["train_seconds"] for r in records])),
        "train_hours_total": float(sum(r["epoch_seconds"] for r in records) / 3600.0),
        "grad_norm_median": float(np.median([c["total_norm_median"] for c in clip if c])) if any(clip) else None,
        "clip_frac_median": float(np.median([c["clip_frac"] for c in clip if c])) if any(clip) else None,
        "peak_memory_gib": max((r.get("peak_memory_gib") or 0.0) for r in records),
        "finished": os.path.exists(os.path.join(directory, "summary.json")),
    }


def fmt(value, pattern):
    return "-" if value is None else pattern % value


def main():
    args = parse_args()
    runs = list(args.runs)
    for pattern in args.glob:
        runs += sorted(glob.glob(pattern))
    rows = [row for row in (summarize(d) for d in runs if os.path.exists(os.path.join(d, "metrics.jsonl"))) if row]
    header = ("运行", "方式", "B", "n", "k", "轮数", "最好val(轮)", "末5轮val", "到0.80", "到0.85",
              "更新/轮", "训练s/轮", "总h", "|g|中位", "裁剪", "显存GiB", "完成")
    print(" | ".join(header))
    for r in rows:
        print(" | ".join([
            r["run"], r["trainer"], fmt(r["streams"], "%d"), fmt(r["accumulate"], "%d"), fmt(r["tbptt_k"], "%d"),
            "%d" % r["epochs"], "%.4f(%d)" % (r["best_val_iou"], r["best_epoch"]),
            "%.4f±%.4f" % (r["last5_mean"], r["last5_std"]), fmt(r["epoch_to_0.80"], "%d"),
            fmt(r["epoch_to_0.85"], "%d"), fmt(r["updates_per_epoch"], "%d"), "%.0f" % r["train_seconds_median"],
            "%.2f" % r["train_hours_total"], fmt(r["grad_norm_median"], "%.2f"),
            fmt(None if r["clip_frac_median"] is None else 100 * r["clip_frac_median"], "%.0f%%"),
            "%.1f" % r["peak_memory_gib"], "是" if r["finished"] else "否"]))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2, ensure_ascii=False)
        print("写入", args.out)


if __name__ == "__main__":
    main()
