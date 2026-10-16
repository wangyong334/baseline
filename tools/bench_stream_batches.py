"""标准 TBPTT（trainer=streams）不同并行流数 B 的显存峰值与训练耗时（在服务器 GPU 上运行）。

对每个 B：从同一份已校准的初始网络出发，用真实训练集跑 warmup 个片段预热，再计时 chunks 个片段
（含按需读取录像、拼批、前向、反向、裁剪、更新，与正式训练相同），记录：
    峰值显存（PyTorch 已分配 / 已预留，GiB）、每个片段的秒数
    推算一轮训练时间 = ceil(训练集总窗数 / (k*B)) 个片段 x 每片段秒数；另测一次验证集评估耗时（与 B 无关）
    推算 epochs 轮总时间（训练 + 每轮验证；不含每 5 轮一次的训练子集评估与保存）
显存不足的 B 记为 OOM 并继续下一个。最后给出预留显存不超过显卡总显存 90% 的最大 B。

用法（服务器，仓库根目录）:
    CUDA_VISIBLE_DEVICES=0 python tools/bench_stream_batches.py --streams 1 2 4 8 12 16 --out log/bench/stream_batches.json
"""
import argparse
import copy
import gc
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_stream_v1 as T  # noqa: E402  先导入：它在导入 torch 前设置环境变量

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from utils.stream_common import load_flat_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="trainer=streams 的显存与耗时")
    parser.add_argument("--config", default="configs/evisseg_stream_v1.yaml")
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    parser.add_argument("--tbptt-k", type=int, default=None)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3, help="预热片段数（不计时）")
    parser.add_argument("--chunks", type=int, default=20, help="计时片段数")
    parser.add_argument("--epochs", type=int, default=None, help="推算总时间用的轮数，默认取 YAML")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="log/bench/stream_batches.json")
    return parser.parse_args()


OOM_MESSAGES = ("out of memory", "unable to find a valid cudnn algorithm", "cudnn_status_alloc_failed",
                "cublas_status_alloc_failed", "cudnn_status_internal_error")


def is_oom(error):
    """显存不足的各种报错。torch 1.9 + 确定性 cuDNN 在显存接近满时常报"找不到可用的 cuDNN 算法"，实质也是显存不够。"""
    text = str(error).lower()
    return any(message in text for message in OOM_MESSAGES)


def main():
    args = parse_args()
    device = torch.device(args.device)
    cuda = device.type == "cuda"     # CPU 只用于本地检查流程，显存一栏为空
    cfg = load_flat_config(args.config)
    cfg.update(trainer="streams", execution="layer", input_device="gpu", accumulate=int(args.accumulate))
    if args.tbptt_k is not None:
        cfg["tbptt_k"] = int(args.tbptt_k)
    k, epochs = int(cfg["tbptt_k"]), int(args.epochs or cfg["epochs"])
    T.seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    stats, q99, pos_weight = T.load_stats(cfg)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    base = T.build_net(cfg, device)
    T.run_calibration(base, cfg, q99, device, out_dir)
    train_set = EvUAVStream(cfg["root"], "train", cfg)
    lengths = [int(train_set[i].n_windows) for i in range(len(train_set))]
    total_windows = sum(lengths)
    total_gib = torch.cuda.get_device_properties(device).total_memory / 2 ** 30 if cuda else float("nan")

    def sync():
        if cuda:
            torch.cuda.synchronize(device)
    print("训练集 %d 条录像 %d 窗 | k=%d accumulate=%d | 显卡 %.1f GiB" % (
        len(lengths), total_windows, k, int(args.accumulate), total_gib), flush=True)

    sync()
    t0 = time.perf_counter()
    T.evaluate_split(base, cfg, q99, device, "val", cfg["state_mode"], collect_layers=True)
    sync()
    val_seconds = time.perf_counter() - t0
    print("验证集评估 %.1f s（每轮一次，与 B 无关）" % val_seconds, flush=True)

    rows = []

    def write_summary(finished):
        """每测完一个 B 就写一次，中途崩溃也保留已测结果。"""
        fits = [r["streams"] for r in rows if r["status"] == "ok" and r["peak_reserved_gib"] <= 0.9 * total_gib]
        summary = {"config": args.config, "tbptt_k": k, "accumulate": int(args.accumulate), "epochs": epochs,
                   "train_windows": total_windows, "val_seconds": val_seconds, "gpu_total_gib": total_gib,
                   "max_streams_within_90pct": max(fits) if fits else None, "finished": finished, "rows": rows}
        T.write_json(args.out, summary)
        return summary

    for streams in sorted(args.streams):
        cfg_b = dict(cfg, streams=int(streams))
        net = copy.deepcopy(base)
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]))
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        row = {"streams": int(streams), "windows_per_update": int(streams) * k * int(args.accumulate)}
        try:
            T.train_streams(net, train_set, lengths, optimizer, cfg_b, q99, pos_weight, device,
                            np.random.RandomState(1), max_chunks=args.warmup)
            sync()
            t0 = time.perf_counter()
            out = T.train_streams(net, train_set, lengths, optimizer, cfg_b, q99, pos_weight, device,
                                  np.random.RandomState(2), max_chunks=args.chunks)
            sync()
            seconds = (time.perf_counter() - t0) / max(out["chunks"], 1)
            chunks_per_epoch = int(math.ceil(total_windows / float(k * int(streams))))
            train_min = chunks_per_epoch * seconds / 60.0
            row.update(status="ok",
                       peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30 if cuda else float("nan"),
                       peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2 ** 30 if cuda else float("nan"),
                       seconds_per_chunk=seconds, chunks_per_epoch=chunks_per_epoch,
                       updates_per_epoch=int(math.ceil(chunks_per_epoch / float(args.accumulate))),
                       train_minutes_per_epoch=train_min,
                       epoch_minutes=train_min + val_seconds / 60.0,
                       total_hours=(train_min + val_seconds / 60.0) * epochs / 60.0)
        except RuntimeError as error:
            if not is_oom(error):
                raise
            row.update(status="OOM", error=str(error).splitlines()[0])
        del net, optimizer
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        rows.append(row)
        write_summary(False)
        if row["status"] == "ok":
            print("B=%-3d 每次更新 %4d 窗 | 显存峰值 已分配 %5.2f / 已预留 %5.2f GiB | %.3f s/片段 | 每轮 %d 片段 %d 次更新 | "
                  "训练 %.1f min/轮，含验证 %.1f min/轮，%d 轮约 %.2f h" % (
                      streams, row["windows_per_update"], row["peak_allocated_gib"], row["peak_reserved_gib"],
                      row["seconds_per_chunk"], row["chunks_per_epoch"], row["updates_per_epoch"],
                      row["train_minutes_per_epoch"], row["epoch_minutes"], epochs, row["total_hours"]), flush=True)
        else:
            print("B=%-3d 显存不足（OOM）: %s；更大的 B 不再测试" % (streams, row["error"]), flush=True)
            break
    summary = write_summary(True)
    print("预留显存 <= 90%% 显卡显存时最多 B = %s" % summary["max_streams_within_90pct"])
    print("BENCH FINISHED:", args.out, flush=True)


if __name__ == "__main__":
    main()
