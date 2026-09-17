"""流式 SNN 训练与评估的计时分解（在服务器 GPU 上运行），用于判断加速选项的实际收益。

测量项（所有计时前后都做 CUDA 同步）：
    1. 读取 NPZ：每条序列 load_npz_events 的耗时
    2. 输入构造：numpy 逐窗（含拷贝到 GPU）vs 设备端按片段构造，折算为每窗毫秒
    3. 训练：同一条序列、同一初始权重，四种组合（execution × input_device）各跑完整 160 窗的 train_sequence，
       折算每窗毫秒，记录峰值显存
    4. 网络本身：输入预先放在 GPU 上，分别测前向与反向；再用 16x16 的小输入（事件数相同）重复一次。
       两者之差是"随输入尺寸增长的部分"，只用来观察规模变化：小输入本身也有 GPU 计算，
       尺寸不同时卷积算法与访存也不同，这个差值不能当作精确的"调度开销 / GPU 计算"拆分
    5. 评估：验证集序列无梯度 run_sequence，每窗毫秒。验证带发放率监控（与训练中每轮验证一致），训练子集不带
    6. 估算一轮耗时区间：下限假设训练集读取完全被 DataLoader worker 掩盖，上限假设读取串行叠加。
       不含日志、checkpoint、worker 启动等，最终以实际训练日志中的 epoch_seconds 为准
各组合在每一轮中轮换先后顺序，取中位数，以减轻机器上其他任务的干扰；相对加速比比绝对值更可信。

用法（服务器）:
    CUDA_VISIBLE_DEVICES=1 python tools/bench_stream_speed.py \
        --checkpoint log/stream_v1_seed37/best_val_iou_seed37.pt --reference-metrics log/stream_v1_seed37/metrics.jsonl
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_stream_v1 as T  # noqa: E402  先导入：它在导入 torch 前设置环境变量

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from dataset.stream_source import NumpyWindowSource, TorchWindowSource  # noqa: E402
from utils.stream_common import select_subset  # noqa: E402

COMBOS = (("step", "cpu"), ("step", "gpu"), ("layer", "cpu"), ("layer", "gpu"))


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="流式 SNN 训练 / 评估计时分解")
    parser.add_argument("--checkpoint", required=True, help="已训练权重（发放率接近真实训练状态）")
    parser.add_argument("--sequences", type=int, default=2, help="训练与评估各用几条序列")
    parser.add_argument("--rounds", type=int, default=3, help="每个组合重复几轮（轮换顺序，取中位数）")
    parser.add_argument("--net-repeats", type=int, default=5, help="第 4 项每种情况重复次数（另有 2 次预热）")
    parser.add_argument("--reference-metrics", default=None, help="原实现训练的 metrics.jsonl，用于对照实测单轮耗时")
    parser.add_argument("--root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None, help="报告路径，默认 log/bench/stream_speed_<时间>.json")
    return parser.parse_args()


class Clock(object):
    """同步 GPU 后读取时间。"""

    def __init__(self, device):
        self.device = device

    def __call__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()


def median(values):
    """中位数（空列表返回 None）。"""
    return float(np.median(values)) if values else None


def bench_loading(dataset, count, clock):
    """第 1 项：读取 NPZ 并构造 StreamSequence 的耗时（秒/条）。返回 (序列列表, 耗时列表)。"""
    sequences, seconds = [], []
    for i in range(min(int(count), len(dataset))):
        t0 = clock()
        sequences.append(dataset[i])
        seconds.append(clock() - t0)
    return sequences, seconds


def bench_inputs(sequences, cfg, q99, device, clock):
    """第 2 项：两种输入构造的每窗毫秒；设备端构造单独记录每条序列的上传耗时。"""
    numpy_ms, torch_ms, setup_ms = [], [], []
    k = int(cfg["tbptt_k"])
    for seq in sequences:
        source = NumpyWindowSource(seq, cfg, q99, device)
        t0 = clock()
        for w in range(seq.n_windows):
            source.window(w)
        numpy_ms.append(1e3 * (clock() - t0) / seq.n_windows)
        t0 = clock()
        source = TorchWindowSource(seq, cfg, q99, device)
        t1 = clock()
        for start in range(0, seq.n_windows, k):
            source.chunk(start, min(start + k, seq.n_windows))
        torch_ms.append(1e3 * (clock() - t1) / seq.n_windows)
        setup_ms.append(1e3 * (t1 - t0))
    return {"numpy_ms_per_window": median(numpy_ms), "device_ms_per_window": median(torch_ms),
            "device_setup_ms_per_sequence": median(setup_ms)}


def bench_training(base_net, sequences, cfg, q99, pos_weight, device, rounds, clock):
    """第 3 项：四种组合各自训练完整序列，返回每窗毫秒与峰值显存（中位数）。"""
    times = {combo: [] for combo in COMBOS}
    memory = {combo: [] for combo in COMBOS}
    losses = {combo: [] for combo in COMBOS}
    for r in range(int(rounds)):
        order = COMBOS[r % len(COMBOS):] + COMBOS[:r % len(COMBOS)]
        for execution, input_device in order:
            run_cfg = dict(cfg, execution=execution, input_device=input_device)
            for seq in sequences:
                net = copy.deepcopy(base_net).train()
                optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]))
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                t0 = clock()
                out = T.train_sequence(net, seq, optimizer, run_cfg, q99, pos_weight, device,
                                       np.random.RandomState(int(cfg["seed"])))
                times[execution, input_device].append(1e3 * (clock() - t0) / seq.n_windows)
                losses[execution, input_device].append(out["loss_sum"])
                if device.type == "cuda":
                    memory[execution, input_device].append(torch.cuda.max_memory_allocated(device) / 2 ** 30)
                del net, optimizer
    return {"%s+%s" % combo: {"ms_per_window": median(times[combo]), "peak_memory_gib": median(memory[combo]),
                              "loss_sum_first": losses[combo][0] if losses[combo] else None}
            for combo in COMBOS}


def synthetic_chunk(steps, height, width, events, device, seed=0):
    """小尺寸输入的一个片段（第 4 项用）：稀疏非负输入与随机事件。"""
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(steps, 1, 12, height, width, generator=g) < 0.05).float() * 2.0
    n = int(events)
    ev = {"t": torch.randint(0, steps, (n,), generator=g), "b": torch.zeros(n, dtype=torch.long),
          "y": torch.randint(0, height, (n,), generator=g), "x": torch.randint(0, width, (n,), generator=g),
          "p": torch.randint(0, 2, (n,), generator=g).float(), "t_local": torch.rand(n, generator=g)}
    labels = (torch.rand(n, generator=g) < 0.1).float()
    return x.to(device), {key: value.to(device) for key, value in ev.items()}, labels.to(device)


def split_windows_of_chunk(x, events, labels):
    """把片段输入拆成逐窗输入列表（逐窗 forward 用），事件按 t 分组，组内保持原顺序。"""
    parts = []
    for t in range(int(x.shape[0])):
        mask = events["t"] == t
        ev = {key: value[mask] for key, value in events.items() if key != "t"}
        parts.append((x[t], ev, labels[mask]))
    return parts


def bench_network(base_net, x, events, labels, pos_weight, repeats, clock):
    """第 4 项的一次测量：同一片段分别用逐窗 forward 与 forward_chunk，测前向、反向每窗毫秒（中位数）。"""
    steps = int(x.shape[0])
    windows = split_windows_of_chunk(x, events, labels)
    weight = torch.tensor([pos_weight], device=x.device)
    result = {}
    for execution in ("step", "layer"):
        forward_ms, backward_ms = [], []
        for i in range(int(repeats) + 2):
            net = copy.deepcopy(base_net).train()
            t0 = clock()
            if execution == "layer":
                logits, _, _ = net.forward_chunk(x, events, None)
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight, reduction="sum")
            else:
                loss, states = 0.0, None
                for xw, ev, lab in windows:
                    logits, states, _ = net(xw, ev, states)
                    if lab.numel():
                        loss = loss + F.binary_cross_entropy_with_logits(logits, lab, pos_weight=weight,
                                                                         reduction="sum")
            t1 = clock()
            if torch.is_tensor(loss):
                (loss / max(int(labels.numel()), 1)).backward()
            t2 = clock()
            if i >= 2:
                forward_ms.append(1e3 * (t1 - t0) / steps)
                backward_ms.append(1e3 * (t2 - t1) / steps)
            del net, loss
        result[execution] = {"forward_ms_per_window": median(forward_ms), "backward_ms_per_window": median(backward_ms)}
    return result


def bench_evaluation(net, sequences, cfg, q99, device, rounds, clock, with_monitor):
    """第 5 项：无梯度 run_sequence 的每窗毫秒（含每条序列构造输入来源的开销）。

    with_monitor=True 时带 LayerMonitor（训练中每轮验证集评估的实际做法）；训练子集评估不带。
    """
    times = {combo: [] for combo in COMBOS}
    net = net.eval()
    for r in range(int(rounds)):
        order = COMBOS[r % len(COMBOS):] + COMBOS[:r % len(COMBOS)]
        for execution, input_device in order:
            run_cfg = dict(cfg, execution=execution, input_device=input_device)
            for seq in sequences:
                t0 = clock()
                monitor = T.LayerMonitor(net.v_threshold) if with_monitor else None
                with torch.no_grad():
                    T.run_sequence(net, seq, run_cfg, q99, device, cfg["state_mode"], monitor)
                    if monitor is not None:
                        monitor.summary()
                times[execution, input_device].append(1e3 * (clock() - t0) / seq.n_windows)
    return {"%s+%s" % combo: {"ms_per_window": median(times[combo])} for combo in COMBOS}


def project_epoch(train_ms, val_ms, subset_ms, load_s, counts, subset_every):
    """估算一轮耗时区间（秒）：训练序列 + 验证集（带监控）+ 训练子集（不带监控）/ 间隔。

    验证集与训练子集在主进程读取，计入 load_s。训练集由 DataLoader worker 读取：
    下限假设读取完全被 worker 掩盖，上限假设每条序列的读取串行叠加在训练时间上。
    不含日志、checkpoint 保存、worker 启动，只用于判断量级。
    """
    windows = counts["windows"]
    train = counts["train"] * windows * train_ms / 1e3
    train_load = counts["train"] * load_s
    val = counts["val"] * (load_s + windows * val_ms / 1e3)
    subset = counts["subset"] * (load_s + windows * subset_ms / 1e3) / float(subset_every)
    low = train + val + subset
    return {"train_s": train, "train_load_s": train_load, "val_s": val, "subset_s": subset,
            "epoch_s_low": low, "epoch_s_high": low + train_load,
            "hours_50_epochs_low": 50 * low / 3600.0, "hours_50_epochs_high": 50 * (low + train_load) / 3600.0}


def reference_epochs(path, count=3):
    """读取原实现 metrics.jsonl 前几轮的实测训练 / 整轮秒数。"""
    rows = []
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                d = json.loads(line)
                rows.append({"epoch": d["epoch"], "train_seconds": d["train_seconds"],
                             "epoch_seconds": d["epoch_seconds"]})
            if len(rows) >= count:
                break
    return rows


def main():
    """入口：加载权重与数据 -> 六项测量 -> 打印表格并写 JSON。"""
    args = parse_args()
    device = torch.device(args.device)
    clock = Clock(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = dict(ckpt["config"])
    if args.root:
        cfg["root"] = args.root
    cfg.setdefault("threshold", 0.9)
    T.seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    q99 = np.asarray(ckpt["stats"]["q99"], dtype=np.float32)
    pos_weight = min(float(ckpt["stats"]["train_neg"]) / float(ckpt["stats"]["train_pos"]), float(cfg["pos_weight_cap"]))
    base_net = T.build_net(dict(cfg, execution="step", input_device="cpu"), device)
    base_net.load_state_dict(ckpt["model"])

    train_names = T.train_file_names(cfg)
    val_set = EvUAVStream(cfg["root"], "val", cfg)
    counts = {"train": len(train_names), "val": len(val_set), "windows": int(cfg["n_windows"]),
              "subset": len(select_subset(train_names, int(cfg["train_subset_size"]), 0))}
    load_avg = os.getloadavg() if hasattr(os, "getloadavg") else None
    report = {"checkpoint": args.checkpoint, "torch": torch.__version__,
              "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "cpu_count": os.cpu_count(), "load_average_at_start": load_avg,
              "deterministic": bool(cfg["deterministic"]), "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
              "counts": counts, "rounds": args.rounds}
    print("设备 %s | torch %s | CPU %s 核，负载 %s | deterministic=%s" % (
        report["device"], torch.__version__, os.cpu_count(), load_avg, cfg["deterministic"]), flush=True)

    train_set = EvUAVStream(cfg["root"], "train", cfg, train_names[:args.sequences])
    train_seqs, train_load = bench_loading(train_set, args.sequences, clock)
    val_seqs, val_load = bench_loading(val_set, args.sequences, clock)
    load_s = median(train_load + val_load)
    report["1_load_seconds_per_sequence"] = {"train": train_load, "val": val_load, "median": load_s}
    print("[1 读取 NPZ] 每条序列 %.2fs（事件数 %s）" % (load_s, [s.n_events for s in train_seqs + val_seqs]), flush=True)

    report["2_inputs"] = bench_inputs(train_seqs, cfg, q99, device, clock)
    r = report["2_inputs"]
    print("[2 输入构造] numpy 逐窗 %.2f ms/窗 | 设备端 %.3f ms/窗（另有上传 %.0f ms/序列）" % (
        r["numpy_ms_per_window"], r["device_ms_per_window"], r["device_setup_ms_per_sequence"]), flush=True)

    report["3_training"] = bench_training(base_net, train_seqs, cfg, q99, pos_weight, device, args.rounds, clock)
    base_train = report["3_training"]["step+cpu"]["ms_per_window"]
    for name, r in report["3_training"].items():
        print("[3 训练] %-10s %.2f ms/窗（%.2fx）| 峰值显存 %.2f GiB" % (
            name, r["ms_per_window"], base_train / r["ms_per_window"], r["peak_memory_gib"] or 0.0), flush=True)

    seq = train_seqs[0]
    source = TorchWindowSource(seq, cfg, q99, device)
    k = int(cfg["tbptt_k"])
    start = min(64, seq.n_windows - k)
    x, events, labels, _ = source.chunk(start, start + k)
    full = bench_network(base_net, x, events, labels, pos_weight, args.net_repeats, clock)
    tiny = bench_network(base_net, *synthetic_chunk(k, 16, 16, int(labels.numel()), device),
                         pos_weight=pos_weight, repeats=args.net_repeats, clock=clock)
    report["4_network"] = {"full": full, "tiny_16x16": tiny, "chunk_windows": k, "chunk_events": int(labels.numel())}
    for execution in ("step", "layer"):
        f, s = full[execution], tiny[execution]
        print("[4 网络] %-5s 前向 %.2f ms/窗（16x16 小输入 %.2f，差值 %.2f）| 反向 %.2f ms/窗（小输入 %.2f，差值 %.2f）" % (
            execution, f["forward_ms_per_window"], s["forward_ms_per_window"],
            f["forward_ms_per_window"] - s["forward_ms_per_window"], f["backward_ms_per_window"],
            s["backward_ms_per_window"], f["backward_ms_per_window"] - s["backward_ms_per_window"]), flush=True)

    print("[4 说明] 差值 = 随输入尺寸增长的部分（含 GPU 计算、访存、算法差异），只用于观察规模变化，不是精确拆分",
          flush=True)

    report["5_evaluation"] = {
        "val_with_monitor": bench_evaluation(base_net, val_seqs, cfg, q99, device, args.rounds, clock, True),
        "subset_no_monitor": bench_evaluation(base_net, val_seqs, cfg, q99, device, args.rounds, clock, False)}
    for kind, rows in report["5_evaluation"].items():
        base_eval = rows["step+cpu"]["ms_per_window"]
        for name, r in rows.items():
            print("[5 评估] %-17s %-10s %.2f ms/窗（%.2fx）" % (
                kind, name, r["ms_per_window"], base_eval / r["ms_per_window"]), flush=True)

    projections = {}
    for name in report["3_training"]:
        for every in (1, 5):
            projections["%s|subset_every=%d" % (name, every)] = project_epoch(
                report["3_training"][name]["ms_per_window"],
                report["5_evaluation"]["val_with_monitor"][name]["ms_per_window"],
                report["5_evaluation"]["subset_no_monitor"][name]["ms_per_window"], load_s, counts, every)
    report["6_projection"] = projections
    for key in ("step+cpu|subset_every=1", "layer+gpu|subset_every=1", "layer+gpu|subset_every=5"):
        p = projections[key]
        print("[6 估算] %-26s 训练 %.0fs（训练集读取另计 0-%.0fs）+ 验证 %.0fs + 训练子集 %.0fs = %.0f-%.0f s/轮，"
              "50 轮 %.1f-%.1f h" % (key, p["train_s"], p["train_load_s"], p["val_s"], p["subset_s"],
                                   p["epoch_s_low"], p["epoch_s_high"], p["hours_50_epochs_low"],
                                   p["hours_50_epochs_high"]), flush=True)
    print("[6 说明] 估算不含日志、checkpoint、worker 启动；加速效果以重训日志中的 epoch_seconds 为准", flush=True)
    if args.reference_metrics:
        report["reference_epochs"] = reference_epochs(args.reference_metrics)
        for row in report["reference_epochs"]:
            print("[对照] 原实现实测 epoch %d：训练 %.0fs，整轮 %.0fs" % (
                row["epoch"], row["train_seconds"], row["epoch_seconds"]), flush=True)
    report["load_average_at_end"] = os.getloadavg() if hasattr(os, "getloadavg") else None

    out_path = args.out or os.path.join("log", "bench", "stream_speed_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    T.write_json(out_path, report)
    print("报告:", out_path)
    print("BENCH FINISHED", flush=True)


if __name__ == "__main__":
    main()
