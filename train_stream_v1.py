"""流式 SNN Baseline-V1（设计版本 v1-3）的训练与评估入口。

模式（--mode）：
    smoke    冒烟测试：校准增益 + 1 个序列前 32 窗的训练步，检查梯度、发放率、显存、耗时
    overfit  单序列过拟合诊断（可配合 --neuron relu / --state-mode reset_each_window 做三方对照）
    train    正式训练：每个序列一次参数更新，每个 epoch 结束验证并保存
    eval     评估 checkpoint：同一权重分别用 carry 与 reset_each_window 跑完整 160 窗
不修改原仓库任何文件；主指标调用原 utils/eval.py 的函数计算。

训练加速选项（TRAIN 小节或命令行；缺省值 = 原实现，结果与旧版本一致）：
    input_device        cpu（numpy 逐窗构造）| gpu（事件常驻显存，按片段构造，输入逐位相同）
    execution           step（逐窗逐层）| layer（片段内逐层时间并行，数学等价）
    train_subset_every  每隔几轮评估一次训练子集（1 = 每轮；验证集始终每轮评估并据此选模型）
    eval_chunk          layer 模式下每轮验证时一次处理的窗口数
--mode eval 始终使用原实现（step + cpu）：论文指标与逐窗延迟都来自它。
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")   # 必须在导入 torch 之前设置

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dataset.ev_uav_stream import EvUAVStream, make_sequence_loader, window_to_device  # noqa: E402
from dataset.stream_source import INPUT_DEVICES, make_window_source  # noqa: E402
from dataset.stream_windows import refill_by_index  # noqa: E402
from model.evspsegnet_stream import (LAYER_NAMES, EvSpSegNetStream, calibrate_gains,  # noqa: E402
                                     count_parameters, estimate_operations)
from model.lif2d_stream import StreamingLIF2d, detach_states  # noqa: E402
from utils.stream_common import (health_warnings, linear_epoch_lr, load_flat_config,  # noqa: E402
                                 make_chunks, select_subset, subset_due, summarize_history)
from utils.stream_metrics import (first_detection_latencies, iou_from_counts, rolling_iou,  # noqa: E402
                                  segment_iou, summarize_latencies, window_confusion)

SOURCE_FILES = ("train_stream_v1.py", "dataset/stream_windows.py", "dataset/ev_uav_stream.py",
                "dataset/stream_source.py", "model/lif2d_stream.py", "model/evspsegnet_stream.py",
                "utils/stream_common.py", "utils/stream_metrics.py", "utils/eval.py")
EXECUTIONS = ("step", "layer")


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------


def parse_args():
    """解析命令行参数。命令行只放"本次运行"相关的选项，模型与训练超参全部来自 YAML。"""
    parser = argparse.ArgumentParser(description="Streaming SNN Baseline-V1")
    parser.add_argument("--config", default="configs/evisseg_stream_v1.yaml")
    parser.add_argument("--mode", choices=("smoke", "overfit", "train", "eval"), required=True)
    parser.add_argument("--save-root", default=None, help="输出目录，默认取 YAML 的 save_root")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 YAML 中的 seed")
    parser.add_argument("--state-mode", choices=("carry", "reset_each_window"), default=None)
    parser.add_argument("--neuron", choices=("lif", "graded", "relu"), default=None,
                        help="lif 有状态+脉冲 | graded 有状态+实数（分离脉冲与记忆的对照）| relu 无状态+实数")
    parser.add_argument("--checkpoint", default=None, help="eval 模式使用的权重文件")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--sequence", default=None, help="overfit 模式的序列文件名")
    parser.add_argument("--steps", type=int, default=300, help="overfit 模式的参数更新次数")
    parser.add_argument("--resume", action="store_true", help="train 模式从 save_root/last.pt 继续")
    parser.add_argument("--overwrite", action="store_true", help="eval 模式允许覆盖已有结果文件")
    parser.add_argument("--readout-ablation", choices=("none", "zero_extra", "zero_feature"),
                        default="none",
                        help="eval 模式的读出消融：zero_extra 屏蔽 p/t_local；zero_feature 屏蔽网络特征")
    parser.add_argument("--dump-dir", default=None,
                        help="eval 模式：把训练时状态模式下的逐事件预测按序列保存为 NPZ，供 tools/verify_predictions.py 独立核验")
    parser.add_argument("--execution", choices=EXECUTIONS, default=None,
                        help="训练与每轮验证的执行方式：step 逐窗（原实现）| layer 逐层时间并行")
    parser.add_argument("--input-device", choices=INPUT_DEVICES, default=None,
                        help="窗口输入在哪里构造：cpu（原实现）| gpu（网络所在设备）")
    parser.add_argument("--train-subset-every", type=int, default=None, help="每隔几轮评估一次训练子集")
    parser.add_argument("--tbptt-k", type=int, default=None, help="覆盖 YAML 中的 TBPTT 片段长度（窗口数）")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def speed_options(cfg):
    """读取并校验训练加速选项，缺省值即原实现。返回 (execution, input_device)。"""
    execution = cfg.get("execution", "step")
    input_device = cfg.get("input_device", "cpu")
    if execution not in EXECUTIONS:
        raise ValueError("execution 必须是 %s 之一，收到 %r" % (EXECUTIONS, execution))
    if input_device not in INPUT_DEVICES:
        raise ValueError("input_device 必须是 %s 之一，收到 %r" % (INPUT_DEVICES, input_device))
    if int(cfg.get("train_subset_every", 1)) < 1 or int(cfg.get("eval_chunk", 32)) < 1:
        raise ValueError("train_subset_every 与 eval_chunk 必须 >= 1")
    return execution, input_device


def build_config(args):
    """读取 YAML，并用命令行覆盖 seed / state_mode / neuron / save_root / 加速选项。返回展平的配置字典。"""
    cfg = load_flat_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.state_mode is not None:
        cfg["state_mode"] = args.state_mode
    if args.neuron is not None:
        cfg["neuron"] = args.neuron
    if args.save_root is not None:
        cfg["save_root"] = args.save_root
    if args.execution is not None:
        cfg["execution"] = args.execution
    if args.input_device is not None:
        cfg["input_device"] = args.input_device
    if args.train_subset_every is not None:
        cfg["train_subset_every"] = args.train_subset_every
    if args.tbptt_k is not None:
        if args.tbptt_k < 1:
            raise ValueError("--tbptt-k 必须 >= 1")
        cfg["tbptt_k"] = args.tbptt_k
    speed_options(cfg)
    return cfg


def seed_everything(seed, deterministic):
    """固定 Python/numpy/torch 随机种子；deterministic=True 时要求确定性算法。

    若冒烟测试报 "does not have a deterministic implementation"，把 YAML 中 deterministic 改为 false。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def load_stats(cfg):
    """读取训练集统计 JSON（由 tools/stream_train_stats.py 生成），返回 (stats, q99, pos_weight)。

    pos_weight 按规则 min(train_neg/train_pos, pos_weight_cap) 重新计算并与 JSON 核对。
    """
    with open(cfg["stats_path"], "r", encoding="utf-8") as stream:
        stats = json.load(stream)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    if q99.shape[0] != 2 + 2 * int(cfg["time_bins"]):
        raise ValueError("统计文件的 q99 通道数与配置不一致")
    pos_weight = min(float(stats["train_neg"]) / float(stats["train_pos"]), float(cfg["pos_weight_cap"]))
    if abs(pos_weight - float(stats["pos_weight"])) > 1e-9:
        raise ValueError("统计文件中的 pos_weight 与当前 cap 规则不一致，请重新生成统计文件")
    return stats, q99, pos_weight


def build_net(cfg, device):
    """按配置构造网络并移动到设备。"""
    if cfg.get("spiking_decoder", False):
        raise ValueError("spiking_decoder（ConvT 后加 LIF 的 10 层版本）已删除；请改用 merged_decoder")
    net = EvSpSegNetStream(
        in_channels=2 + 2 * int(cfg["time_bins"]), channels=tuple(cfg["channels"]),
        neuron=cfg["neuron"], norm=cfg["norm"], state_mode=cfg["state_mode"],
        readout_hidden=int(cfg["readout_hidden"]), dt_ms=float(cfg["window_ms"]),
        tau_init_ms=float(cfg["tau_init_ms"]), tau_min_ms=float(cfg["tau_min_ms"]),
        tau_max_ms=float(cfg["tau_max_ms"]), v_threshold=float(cfg["v_threshold"]),
        merged_decoder=cfg.get("merged_decoder", False))
    return net.to(device)


def train_file_names(cfg):
    """返回训练集全部 NPZ 文件名（排序）。"""
    directory = os.path.join(cfg["root"], "train")
    return sorted(n for n in os.listdir(directory) if n.endswith(".npz"))


def source_hashes():
    """记录本次运行所用源码的 sha256，写进 run_config.json，方便事后核对代码版本。"""
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES if (here / name).exists()}


def write_json(path, payload):
    """以 UTF-8、缩进格式写 JSON。"""
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def peak_memory_gib(device):
    """返回当前设备的峰值显存（GiB）；CPU 运行时返回 None。"""
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3)


# ---------------------------------------------------------------------------
# 增益校准数据
# ---------------------------------------------------------------------------


def calibration_factory(cfg, q99, device):
    """构造增益校准的数据来源：固定种子选出的若干训练序列，每个序列取前 calib_windows 个连续窗口。

    返回 (factory, names)。factory() 每次调用都重新产出相同的数据，供逐层校准反复使用。
    """
    names = select_subset(train_file_names(cfg), int(cfg["calib_sequences"]), int(cfg["seed"]) + 1000)
    dataset = EvUAVStream(cfg["root"], "train", cfg, names)
    n_windows = int(cfg["calib_windows"])

    def windows_of(seq):
        """按时间顺序产出一个序列前 n_windows 个窗口的 (x, events)。"""
        for k in range(min(n_windows, seq.n_windows)):
            x, events, _, _ = window_to_device(seq, k, q99, cfg, device)
            yield x, events

    def factory():
        """每次调用都从头产出全部校准序列（逐层校准会调用 7 次）。"""
        for i in range(len(dataset)):
            yield windows_of(dataset[i])

    return factory, names


def run_calibration(net, cfg, q99, device, out_dir):
    """执行一次性增益校准，并把报告写到 out_dir/calibration.json。"""
    factory, names = calibration_factory(cfg, q99, device)
    reports = calibrate_gains(net, factory, float(cfg["calib_quantile"]),
                              int(cfg["calib_samples_per_channel"]), int(cfg["calib_min_positive"]),
                              float(cfg["calib_gain_min"]), float(cfg["calib_gain_max"]),
                              int(cfg["seed"]) + 4000)
    write_json(Path(out_dir) / "calibration.json", {"sequences": names, "layers": reports})
    for r in reports:
        g = np.asarray(r["gain"])
        print("[calib] %-5s gain min/mean/max = %.3f / %.3f / %.3f  fallback=%d/%d" % (
            r["layer"], g.min(), g.mean(), g.max(), sum(r["fallback"]), len(r["fallback"])), flush=True)
    return reports


# ---------------------------------------------------------------------------
# 监控
# ---------------------------------------------------------------------------


class LayerMonitor(object):
    """累计每层的发放与膜电位统计（在 GPU 上累加，汇总时才同步）。

    统计量：
        firing_rate          整层所有元素的平均发放率（含空像素，用于理论 SOP 估计）
        silent_channel_frac  评估期间一次都没发放的通道占比
        always_on_frac       发放窗口占比 > 90% 的神经元占比（饱和）
        u_abs_max            |U_pre| 最大值
        big_membrane_frac    |U_pre| > 10*阈值 的元素占比
    ReLU 版本把"激活 > 0"视为发放。
    """

    def __init__(self, v_threshold):
        self.v_threshold = float(v_threshold)
        self.data = {}

    def update(self, info):
        """累加 info：逐窗 forward 的每层 [B,C,H,W]，或 forward_chunk 的每层 [T,B,C,H,W]（T 个窗口）。

        两种形状按窗口计数，结果相同；计数用整数累加，避免片段内元素过多时 float32 求和舍入。
        """
        with torch.no_grad():
            for name, spikes, u_pre in zip(LAYER_NAMES, info["spikes"], info["u_pre"]):
                if spikes.dim() == 4:
                    spikes, u_pre = spikes.unsqueeze(0), u_pre.unsqueeze(0)
                active = spikes > 0
                d = self.data.get(name)
                if d is None:
                    d = {"spike_sum": torch.zeros((), device=spikes.device, dtype=torch.float64),
                         "elements": 0, "windows": 0,
                         "channel_spikes": torch.zeros(spikes.shape[2], device=spikes.device,
                                                       dtype=torch.float64),
                         "neuron_windows": torch.zeros(spikes.shape[2:], device=spikes.device),
                         "u_abs_max": torch.zeros((), device=spikes.device),
                         "big": torch.zeros((), device=spikes.device, dtype=torch.float64)}
                    self.data[name] = d
                d["spike_sum"] += active.sum().double()
                d["elements"] += active.numel()
                d["windows"] += int(spikes.shape[0])
                d["channel_spikes"] += active.sum(dim=(0, 1, 3, 4)).double()
                d["neuron_windows"] += (active.sum(1) > 0).sum(0).float()
                u_abs = u_pre.abs()
                d["u_abs_max"] = torch.maximum(d["u_abs_max"], u_abs.max())
                d["big"] += (u_abs > 10.0 * self.v_threshold).sum().double()

    def summary(self):
        """返回 {层名: 统计字典}。"""
        out = {}
        for name, d in self.data.items():
            out[name] = {
                "firing_rate": float(d["spike_sum"]) / max(d["elements"], 1),
                "silent_channel_frac": float((d["channel_spikes"] == 0).float().mean()),
                "always_on_frac": float(((d["neuron_windows"] / max(d["windows"], 1)) > 0.9)
                                        .float().mean()),
                "u_abs_max": float(d["u_abs_max"]),
                "big_membrane_frac": float(d["big"]) / max(d["elements"], 1),
            }
        return out


def tau_statistics(net):
    """返回每个 LIF 层的 tau/beta 最小/平均/最大值，以及贴近上下界的通道占比（ReLU 版本返回空字典）。"""
    out = {}
    for name, block in zip(LAYER_NAMES, net.blocks()):
        neuron = block.neuron
        if not isinstance(neuron, StreamingLIF2d):
            continue
        with torch.no_grad():
            tau = neuron.tau().double().cpu()
            beta = neuron.beta().double().cpu()
        margin = 0.02 * (neuron.tau_max_ms - neuron.tau_min_ms)
        out[name] = {"tau_min": float(tau.min()), "tau_mean": float(tau.mean()), "tau_max": float(tau.max()),
                     "beta_min": float(beta.min()), "beta_mean": float(beta.mean()),
                     "beta_max": float(beta.max()),
                     "near_min_frac": float((tau < neuron.tau_min_ms + margin).double().mean()),
                     "near_max_frac": float((tau > neuron.tau_max_ms - margin).double().mean())}
    return out


def gradient_group_norms(net):
    """按顶层模块（enc1..dec1、up1..up3、readout）计算梯度 L2 范数；没有梯度的组记为 0。"""
    squares = {}
    for name, param in net.named_parameters():
        group = name.split(".")[0]
        value = 0.0 if param.grad is None else float(param.grad.detach().double().pow(2).sum())
        squares[group] = squares.get(group, 0.0) + value
    return {k: math.sqrt(v) for k, v in squares.items()}


class WindowTimer(object):
    """记录每个窗口的预处理、网络、后处理耗时（毫秒），跳过前 warmup 个窗口；计时前后均做 CUDA 同步。"""

    def __init__(self, warmup, device):
        self.warmup = int(warmup)
        self.device = device
        self.seen = 0
        self.records = []

    def now(self):
        """同步 GPU 后返回当前时间（秒）。"""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def add(self, pre_s, net_s, post_s):
        """记录一个窗口的三段耗时（秒）。"""
        self.seen += 1
        if self.seen > self.warmup:
            self.records.append((1e3 * pre_s, 1e3 * net_s, 1e3 * post_s))

    def summary(self, window_ms):
        """返回各段耗时的均值/中位数/P95，以及单窗总耗时是否小于窗长（能否实时）。"""
        if not self.records:
            return {}
        arr = np.asarray(self.records)
        total = arr.sum(1)
        out = {"windows_timed": int(arr.shape[0])}
        for i, key in enumerate(("pre", "net", "post")):
            out[key + "_ms_mean"] = float(arr[:, i].mean())
            out[key + "_ms_p95"] = float(np.percentile(arr[:, i], 95))
        out.update({"total_ms_mean": float(total.mean()), "total_ms_median": float(np.median(total)),
                    "total_ms_p95": float(np.percentile(total, 95)),
                    "ms_per_160_windows": float(total.mean() * 160),
                    "realtime_ok": bool(total.mean() < float(window_ms))})
        return out


# ---------------------------------------------------------------------------
# 训练与推理
# ---------------------------------------------------------------------------


def train_sequence(net, seq, optimizer, cfg, q99, pos_weight, device, rng, max_windows=None):
    """用一个 8 秒序列训练（stateful TBPTT）。

    流程：随机第一段长度（1..K）切片段；每段前向 K 个窗口后立即反向并释放计算图，
    片段边界 detach 状态（只切梯度、保留数值）。
      optimizer_step=per_sequence：梯度跨片段累加，损失除以整条序列事件数，序列结束时裁剪并更新一次
      optimizer_step=per_chunk   ：回退预案，每段除以本段事件数并更新一次
    空窗口照常前向（使状态衰减），只是不产生损失；没有事件的片段不调用 backward。
    execution=layer 时每个片段调用一次 forward_chunk，片段损失是片段内全部事件的求和，与逐窗求和相同。
    max_windows 仅供冒烟测试截断使用。
    返回: {"loss_sum", "events", "grad_norms", "missing_grads", "steps"}
    """
    k = int(cfg["tbptt_k"])
    execution, _ = speed_options(cfg)
    n_windows = seq.n_windows if max_windows is None else min(int(max_windows), seq.n_windows)
    chunks = make_chunks(n_windows, k, int(rng.randint(1, k + 1)))
    per_sequence = cfg["optimizer_step"] == "per_sequence"
    if cfg["optimizer_step"] not in ("per_sequence", "per_chunk"):
        raise ValueError("optimizer_step 必须是 per_sequence 或 per_chunk")
    total_events = int(seq.bounds[n_windows] - seq.bounds[0])
    dtype = next(net.parameters()).dtype
    source = make_window_source(seq, cfg, q99, device, dtype)
    weight = torch.tensor([pos_weight], dtype=dtype, device=device)
    loss_sum, steps, grad_norms, missing = 0.0, 0, {}, []

    def step():
        """记录裁剪前的分组梯度范数与缺失梯度的参数名，然后裁剪并更新一次参数。"""
        norms = gradient_group_norms(net)
        lost = [name for name, p in net.named_parameters() if p.requires_grad and p.grad is None]
        torch.nn.utils.clip_grad_norm_(net.parameters(), float(cfg["grad_clip"]))
        optimizer.step()
        return norms, lost

    optimizer.zero_grad(set_to_none=True)
    states = None
    for start, end in chunks:
        if not per_sequence:
            optimizer.zero_grad(set_to_none=True)
        chunk_loss, chunk_events = None, 0
        if execution == "layer":
            x, events, labels, _ = source.chunk(start, end)
            logits, states, _ = net.forward_chunk(x, events, states)
            if labels.numel():
                chunk_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight,
                                                                reduction="sum")
                chunk_events = int(labels.numel())
        else:
            for w in range(start, end):
                x, events, labels, _ = source.window(w)
                logits, states, _ = net(x, events, states)
                if labels.numel():
                    term = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight,
                                                              reduction="sum")
                    chunk_loss = term if chunk_loss is None else chunk_loss + term
                    chunk_events += int(labels.numel())
        if chunk_loss is not None:
            denominator = total_events if per_sequence else chunk_events
            (chunk_loss / denominator).backward()
            value = float(chunk_loss.detach())
            if not math.isfinite(value):
                raise RuntimeError("损失出现非有限值: 序列 %s 窗口 %d-%d" % (seq.name, start, end))
            loss_sum += value
            if not per_sequence:
                grad_norms, missing = step()
                steps += 1
        states = detach_states(states)
    if per_sequence and total_events > 0:
        grad_norms, missing = step()
        steps = 1
    return {"loss_sum": loss_sum, "events": total_events, "grad_norms": grad_norms,
            "missing_grads": missing, "steps": steps}


def count_nonzero_input(stats, x):
    """累计输入中的非零元素个数与总数（事件驱动口径的能耗要用；stats 为 None 时什么都不做）。"""
    if stats is not None:
        stats["nonzero"] += int((x != 0).sum())
        stats["elements"] += int(x.numel())


def run_sequence(net, seq, cfg, q99, device, state_mode, monitor=None, timer=None, input_stats=None):
    """推理一个完整序列（全部窗口，按时间顺序，状态按 state_mode 处理）。

    返回: (probs, confusion)
        probs      按文件原始事件顺序回填的预测概率（numpy float32，长度 N）
        confusion  [n_windows, 4] 每窗 TP/FP/FN/正事件数（阈值 cfg["threshold"]）
    调用方负责 torch.no_grad() 与 net.eval()。
    execution=layer 时每 eval_chunk 个窗口调用一次 forward_chunk（吞吐快，但不能用来测单窗延迟，timer 必须为空）。
    input_stats 不为空时，累计输入中非零元素的个数与总数（事件驱动口径的能耗要用，见 estimate_operations）。
    """
    execution, _ = speed_options(cfg)
    if execution == "layer" and timer is not None:
        raise ValueError("逐窗延迟只能在 execution=step 下测量")
    threshold = float(cfg["threshold"])
    source = make_window_source(seq, cfg, q99, device, next(net.parameters()).dtype)
    states = None
    index_parts, prob_parts = [], []
    confusion = np.zeros((seq.n_windows, 4), dtype=np.int64)
    if execution == "layer":
        size = int(cfg.get("eval_chunk", 32))
        for start in range(0, seq.n_windows, size):
            end = min(start + size, seq.n_windows)
            x, events, _, idx = source.chunk(start, end)
            count_nonzero_input(input_stats, x)
            logits, states, info = net.forward_chunk(x, events, states, state_mode=state_mode,
                                                     collect=monitor is not None)
            prob = torch.sigmoid(logits).float().cpu().numpy()
            if monitor is not None:
                monitor.update(info)
            offset = 0
            for k in range(start, end):
                count = int(seq.bounds[k + 1] - seq.bounds[k])
                part_idx, part_prob = idx[offset:offset + count], prob[offset:offset + count]
                index_parts.append(part_idx)
                prob_parts.append(part_prob)
                confusion[k] = window_confusion(part_prob, seq.label[part_idx], threshold)
                offset += count
    else:
        for k in range(seq.n_windows):
            t0 = timer.now() if timer else None
            x, events, _, idx = source.window(k)
            count_nonzero_input(input_stats, x)
            t1 = timer.now() if timer else None
            logits, states, info = net(x, events, states, state_mode=state_mode, collect=monitor is not None)
            t2 = timer.now() if timer else None
            prob = torch.sigmoid(logits).float().cpu().numpy()
            if timer:
                timer.add(t1 - t0, t2 - t1, timer.now() - t2)
            if monitor is not None:
                monitor.update(info)
            index_parts.append(idx)
            prob_parts.append(prob)
            confusion[k] = window_confusion(prob, seq.label[idx], threshold)
    probs = refill_by_index(seq.n_events, index_parts, prob_parts)
    return probs, confusion


def evaluate_split(net, cfg, q99, device, split, state_mode, names=None, full=False,
                   collect_layers=False, dump_dir=None):
    """在一个划分（或其子集）上评估。

    主指标 IoU/ACC 调用原仓库 utils/eval.py 的 evaluate_iou_and_accuracy（全局拼接口径）；
    full=True 时额外调用原 roc_update 计算 Pd/Fa，并统计逐窗 IoU、分段 IoU、首次检出延迟、耗时、理论运算量。
    collect_layers=True 时统计每层发放与膜电位（训练中每轮验证也会用到，用于健康门槛）。
    dump_dir 不为空时，把每个序列的逐事件预测保存为 NPZ（字段与原仓库诊断文件一致：
    locs [batch,x,y,t]、labels、probabilities，另加 target_id），文件顺序与原 NPZ 一致。
    """
    from utils.eval import evalute        # 延迟导入：该模块依赖 cv2/pandas，只在评估时需要

    dataset = EvUAVStream(cfg["root"], split, cfg, names)
    evaluator = evalute(SimpleNamespace(roc=full, pd_detT=cfg["pd_detT"],
                                        correct_thresh=cfg["correct_thresh"]))
    monitor = LayerMonitor(net.v_threshold) if (collect_layers or full) else None
    timer = WindowTimer(cfg["timing_warmup_windows"], device) if full else None
    input_stats = {"nonzero": 0, "elements": 0} if full else None
    confusion_sum = np.zeros((int(cfg["n_windows"]), 4), dtype=np.int64)
    kept, total_events = [], 0
    was_training = net.training
    net.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            seq = dataset[i]
            probs, confusion = run_sequence(net, seq, cfg, q99, device, state_mode, monitor, timer,
                                            input_stats)
            confusion_sum += confusion
            total_events += seq.n_events
            evaluator.matches[str(i)] = {"seg_pred": torch.from_numpy(probs),
                                         "seg_gt": torch.from_numpy(seq.label)}
            if dump_dir is not None:
                zeros_i64 = np.zeros(seq.n_events, dtype=np.int64)
                np.savez(os.path.join(dump_dir, seq.name),
                         locs=np.stack([zeros_i64, seq.x, seq.y, seq.t], 1),
                         labels=seq.label.astype(np.float32),
                         probabilities=probs.astype(np.float32),
                         target_id=seq.target_id.astype(np.float64))
            if full:
                zeros = np.zeros(seq.n_events, dtype=np.float32)
                ev_locs = torch.from_numpy(np.stack([zeros, seq.x, seq.y, seq.t], 1).astype(np.float32))
                evaluator.roc_update(ev_locs[:, 3], torch.from_numpy(probs), seq.target_id,
                                     torch.from_numpy(seq.label), ev_locs, thresh=float(cfg["threshold"]))
                kept.append((seq.t, seq.label, seq.target_id, probs))
    net.train(was_training)
    iou, acc = evaluator.evaluate_iou_and_accuracy(thresh=float(cfg["threshold"]))
    result = {"split": split, "state_mode": state_mode, "n_sequences": len(dataset),
              "iou": float(iou), "acc": float(acc)}
    if monitor is not None:
        result["layers"] = monitor.summary()
    if full:
        pd, fa = evaluator.cal_roc()
        tp, fp, fn, npos = (confusion_sum[:, j] for j in range(4))
        timing = timer.summary(cfg["window_ms"])
        net_ms = timing.get("net_ms_mean", 0.0)
        records = []
        for t, label, target_id, probs in kept:
            records += first_detection_latencies(t, label, target_id, probs, cfg["window_ms"],
                                                 cfg["threshold"], cfg["correct_thresh"], net_ms)
        rates = {name: s["firing_rate"] for name, s in result["layers"].items()}
        result.update({
            "pd": float(pd), "fa": float(fa),
            "segment_iou": segment_iou(tp, fp, fn, cfg["segments"]),
            "rolling_iou": [None if math.isnan(v) else v
                            for v in rolling_iou(tp, fp, fn, int(cfg["rolling_window"])).tolist()],
            "per_window": {"tp": tp.tolist(), "fp": fp.tolist(), "fn": fn.tolist(), "pos": npos.tolist()},
            "latency": summarize_latencies(records),
            "timing": timing,
            "input_nonzero_fraction": input_stats["nonzero"] / float(max(input_stats["elements"], 1)),
            "operations": estimate_operations(net, cfg["pad_height"], cfg["pad_width"], rates,
                                              total_events / float(len(dataset) * int(cfg["n_windows"])),
                                              input_stats["nonzero"] / float(max(input_stats["elements"], 1))),
        })
    return result


# ---------------------------------------------------------------------------
# 各模式
# ---------------------------------------------------------------------------


def prepare_run(args, cfg, allow_existing=False):
    """通用准备：设置种子、读取统计、构造网络、创建输出目录并写 run_config.json。"""
    device = torch.device(args.device)
    seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    stats, q99, pos_weight = load_stats(cfg)
    net = build_net(cfg, device)
    root = Path(cfg["save_root"])
    root.mkdir(parents=True, exist_ok=allow_existing)
    write_json(root / ("run_config_%s.json" % args.mode), {
        "design_version": "v1-3", "args": vars(args), "config": cfg, "stats": stats,
        "pos_weight": pos_weight, "parameters": count_parameters(net), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "source_sha256": source_hashes()})
    print("参数量:", count_parameters(net), flush=True)
    return device, stats, q99, pos_weight, net, root


def mode_smoke(args, cfg):
    """冒烟测试：校准 + 1 个训练序列前 32 窗的一次参数更新，打印梯度/发放率/tau/显存/耗时。"""
    device, stats, q99, pos_weight, net, root = prepare_run(args, cfg)
    run_calibration(net, cfg, q99, device, root)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]))
    seq = EvUAVStream(cfg["root"], "train", cfg, [train_file_names(cfg)[0]])[0]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    out = train_sequence(net, seq, optimizer, cfg, q99, pos_weight, device,
                         np.random.RandomState(int(cfg["seed"])), max_windows=32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    monitor = LayerMonitor(net.v_threshold)
    with torch.no_grad():
        net.eval()
        states = None
        for k in range(32):
            x, events, _, _ = window_to_device(seq, k, q99, cfg, device)
            _, states, info = net(x, events, states, collect=True)
            monitor.update(info)
        net.train()
    layers, taus = monitor.summary(), tau_statistics(net)
    report = {"sequence": seq.name, "events_in_32_windows": out["events"], "loss_per_event":
              out["loss_sum"] / max(out["events"], 1), "grad_norms": out["grad_norms"],
              "missing_grads": out["missing_grads"], "train_ms_per_window": 1e3 * elapsed / 32,
              "peak_memory_gib": peak_memory_gib(device), "layers": layers, "tau": taus,
              "warnings": health_warnings(layers, taus, out["grad_norms"])}
    write_json(root / "smoke.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    missing = out["missing_grads"]
    if cfg["state_mode"] == "reset_each_window":
        # reset 模式下 beta 不参与计算，时间常数参数 a 本来就没有梯度，不算错误
        missing = [name for name in missing if not name.endswith(".neuron.a")]
    if missing:
        raise RuntimeError("以下参数没有梯度: %s" % missing)
    print("SMOKE TEST FINISHED:", root, flush=True)


def mode_overfit(args, cfg):
    """单序列过拟合：同一序列反复训练 --steps 次（学习率固定为 lr），定期在该序列上评估 IoU/ACC。

    期望 loss 大幅下降、IoU 升到 0.9 左右；若停在较低位置，用 --neuron relu 和
    --state-mode reset_each_window 各跑一次对照，判断是数据/架构上限还是 SNN/状态链路问题。
    """
    device, stats, q99, pos_weight, net, root = prepare_run(args, cfg)
    run_calibration(net, cfg, q99, device, root)
    name = args.sequence or train_file_names(cfg)[0]
    seq = EvUAVStream(cfg["root"], "train", cfg, [name])[0]
    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]))
    rng = np.random.RandomState(int(cfg["seed"]))
    log = root / "overfit.jsonl"
    for step in range(1, int(args.steps) + 1):
        out = train_sequence(net, seq, optimizer, cfg, q99, pos_weight, device, rng)
        if step == 1 or step % 20 == 0 or step == args.steps:
            net.eval()
            with torch.no_grad():
                _, confusion = run_sequence(net, seq, cfg, q99, device, cfg["state_mode"])
            net.train()
            tp, fp, fn = (int(confusion[:, j].sum()) for j in range(3))
            record = {"step": step, "loss_per_event": out["loss_sum"] / max(out["events"], 1),
                      "iou": iou_from_counts(tp, fp, fn), "acc": tp / max(tp + fn, 1),
                      "neuron": cfg["neuron"], "state_mode": cfg["state_mode"], "sequence": name}
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    print("OVERFIT FINISHED:", root, flush=True)


def save_checkpoint(path, net, optimizer, epoch, best_val_iou, cfg, stats):
    """保存权重（含已校准增益与 tau 参数）、优化器状态与配置；不保存膜电位。"""
    torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "best_val_iou": best_val_iou, "config": cfg, "stats": stats,
                "design_version": "v1-3"}, str(path))


def mode_train(args, cfg):
    """正式训练：每个 epoch 设定学习率 -> 逐序列训练（每序列一次更新）-> 验证集与训练子集评估 -> 保存。

    验证与 checkpoint 只在 epoch 结束时进行（此时所有训练序列均已结束，不存在未完成的状态）。
    每个 epoch 的序列打乱顺序与 TBPTT 首段长度都由 (seed, epoch) 决定，--resume 时可以精确续跑。
    """
    device, stats, q99, pos_weight, net, root = prepare_run(args, cfg, allow_existing=args.resume)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]))
    seed, epochs = int(cfg["seed"]), int(cfg["epochs"])
    start_epoch, best, history = 0, -float("inf"), []
    metrics_path = root / "metrics.jsonl"
    if args.resume:
        ckpt = torch.load(str(root / "last.pt"), map_location=device)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best = int(ckpt["epoch"]) + 1, float(ckpt["best_val_iou"])
        history = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
        history = [h for h in history if h["epoch"] < start_epoch]
        print("从 epoch %d 继续" % start_epoch, flush=True)
    else:
        if metrics_path.exists():
            raise RuntimeError("输出目录已有 metrics.jsonl，请换目录或使用 --resume")
        run_calibration(net, cfg, q99, device, root)
    subset = select_subset(train_file_names(cfg), int(cfg["train_subset_size"]), seed + 2000)
    train_set = EvUAVStream(cfg["root"], "train", cfg)
    execution, input_device = speed_options(cfg)
    print("执行方式 %s | 输入构造 %s | 训练子集每 %d 轮评估 | 验证片段 %d 窗" % (
        execution, input_device, int(cfg.get("train_subset_every", 1)), int(cfg.get("eval_chunk", 32))), flush=True)
    for epoch in range(start_epoch, epochs):
        lr = linear_epoch_lr(epoch, epochs, float(cfg["lr"]), float(cfg["lr_end"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        loader = make_sequence_loader(train_set, True, seed * 1000 + epoch, cfg["num_workers"])
        rng = np.random.RandomState(seed * 1000 + epoch + 7)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        net.train()
        loss_sum, event_sum, grad_acc, n_seq = 0.0, 0, {}, 0
        for seq in loader:
            out = train_sequence(net, seq, optimizer, cfg, q99, pos_weight, device, rng)
            loss_sum += out["loss_sum"]
            event_sum += out["events"]
            for key, value in out["grad_norms"].items():
                grad_acc[key] = grad_acc.get(key, 0.0) + value
            n_seq += 1
        train_seconds = time.perf_counter() - t0
        grad_mean = {k: v / max(n_seq, 1) for k, v in grad_acc.items()}
        val = evaluate_split(net, cfg, q99, device, "val", cfg["state_mode"], collect_layers=True)
        sub = None
        if subset_due(epoch, epochs, int(cfg.get("train_subset_every", 1))):
            sub = evaluate_split(net, cfg, q99, device, "train", cfg["state_mode"], names=subset)
        taus = tau_statistics(net)
        record = {"epoch": epoch, "seed": seed, "lr": lr, "loss_per_event": loss_sum / max(event_sum, 1),
                  "val_iou": val["iou"], "val_acc": val["acc"],
                  "train_subset_iou": sub["iou"] if sub else None, "train_subset_acc": sub["acc"] if sub else None,
                  "execution": execution, "input_device": input_device,
                  "train_seconds": train_seconds, "epoch_seconds": time.perf_counter() - t0,
                  "peak_memory_gib": peak_memory_gib(device), "grad_norms": grad_mean,
                  "layers": val["layers"], "tau": taus,
                  "warnings": health_warnings(val["layers"], taus, grad_mean)}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        history.append(record)
        if val["iou"] > best:
            best = val["iou"]
            save_checkpoint(root / ("best_val_iou_seed%d.pt" % seed), net, optimizer, epoch, best, cfg, stats)
        save_checkpoint(root / "last.pt", net, optimizer, epoch, best, cfg, stats)
        print("epoch %d lr %.2e loss/event %.5f val IoU %.4f ACC %.4f | train-subset IoU %s | %.0fs（训练 %.0fs）| 警告 %d" % (
            epoch, lr, record["loss_per_event"], val["iou"], val["acc"], "%.4f" % sub["iou"] if sub else "-",
            record["epoch_seconds"], train_seconds, len(record["warnings"])), flush=True)
        for w in record["warnings"]:
            print("  [health]", w, flush=True)
    summary = summarize_history(history, seed)
    write_json(root / "summary.json", summary)
    print("SUMMARY_JSON", json.dumps(summary, sort_keys=True), flush=True)
    print("TRAINING FINISHED:", root, flush=True)


def mode_eval(args, cfg):
    """评估 checkpoint：同一权重分别用 carry 与 reset_each_window 跑完整 160 窗，结果写入 checkpoint 同目录。

    网络结构、q99、pos_weight 一律取自 checkpoint（保证与训练一致），只有数据根目录取自当前 YAML。
    增益必须已经校准，评估阶段禁止重新校准。
    """
    if not args.checkpoint:
        raise ValueError("eval 模式需要 --checkpoint")
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_cfg = dict(ckpt["config"])
    train_cfg["root"] = cfg["root"]
    for key in ("threshold", "pd_detT", "correct_thresh", "rolling_window", "segments",
                "timing_warmup_windows"):
        train_cfg[key] = cfg[key]
    # 正式评估始终走原实现：论文指标与逐窗延迟都以它为准，与训练时用的加速选项无关
    if args.execution == "layer" or args.input_device == "gpu":
        print("注意：--mode eval 固定使用 execution=step、input_device=cpu，忽略命令行加速选项", flush=True)
    train_cfg["execution"], train_cfg["input_device"] = "step", "cpu"
    seed_everything(int(train_cfg["seed"]), bool(train_cfg["deterministic"]))
    q99 = np.asarray(ckpt["stats"]["q99"], dtype=np.float32)
    net = build_net(train_cfg, device)
    net.load_state_dict(ckpt["model"])
    if not bool(net.gain_calibrated):
        raise RuntimeError("checkpoint 中的增益未校准")
    net.readout_ablation = args.readout_ablation
    suffix = "" if args.readout_ablation == "none" else ("_" + args.readout_ablation)
    out_path = Path(args.checkpoint).parent / ("eval_%s_%s%s.json" % (
        args.split, Path(args.checkpoint).stem, suffix))
    if out_path.exists() and not args.overwrite:
        raise RuntimeError("结果已存在: %s（使用 --overwrite 覆盖）" % out_path)
    results = {"checkpoint": args.checkpoint, "epoch": ckpt["epoch"], "parameters": count_parameters(net),
               "trained_state_mode": train_cfg["state_mode"], "neuron": train_cfg["neuron"],
               "readout_ablation": args.readout_ablation}
    for mode in ("carry", "reset_each_window"):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        dump = None
        if args.dump_dir and mode == train_cfg["state_mode"]:
            dump = args.dump_dir
            os.makedirs(dump, exist_ok=True)
        r = evaluate_split(net, train_cfg, q99, device, args.split, mode, full=True, dump_dir=dump)
        r["peak_memory_gib"] = peak_memory_gib(device)
        r["tau"] = tau_statistics(net)
        r["warnings"] = health_warnings(r["layers"], r["tau"], {})
        results[mode] = r
        print("[%s] IoU %.4f ACC %.4f Pd %.4f Fa %.3e | 分段 %s | %.2f ms/窗 | 延迟中位数 %s ms" % (
            mode, r["iou"], r["acc"], r["pd"], r["fa"],
            {k: round(v, 4) for k, v in r["segment_iou"].items()},
            r["timing"].get("total_ms_mean", float("nan")), r["latency"].get("latency_median_ms")), flush=True)
    write_json(out_path, results)
    print("EVAL FINISHED:", out_path, flush=True)


def main():
    """入口：解析参数、读取配置、分发到对应模式。"""
    args = parse_args()
    cfg = build_config(args)
    {"smoke": mode_smoke, "overfit": mode_overfit, "train": mode_train, "eval": mode_eval}[args.mode](args, cfg)


if __name__ == "__main__":
    main()
