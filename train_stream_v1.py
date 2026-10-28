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
from dataset.noise_aug import NoiseAugment, inject_background_noise, noise_aug_settings  # noqa: E402
from dataset.stream_windows import num_input_channels, refill_by_index  # noqa: E402
from model.evspsegnet_stream import (LAYER_NAMES, EvSpSegNetStream, calibrate_gains,  # noqa: E402
                                     count_parameters, estimate_operations, tif_settings, window_sop_summary)
from model.lif2d_stream import (StreamingLIF2d, StreamingDualMemoryLIF2d, StreamingTIF2d,  # noqa: E402
                                detach_states)
from utils.comm_budget import CommBudget, mean_messages, window_message_summary  # noqa: E402
from utils.stream_common import (health_warnings, linear_epoch_lr, load_flat_config,  # noqa: E402
                                 make_chunks, select_subset, subset_due, summarize_history)
from utils.stream_metrics import (SWEEP_THRESHOLDS, FastRoc, first_detection_latencies, iou_from_counts,  # noqa: E402
                                  rolling_iou, segment_iou, summarize_latencies, sweep_summary,
                                  threshold_counts, window_confusion)

SOURCE_FILES = ("train_stream_v1.py", "dataset/stream_windows.py", "dataset/ev_uav_stream.py",
                "dataset/stream_source.py", "model/lif2d_stream.py", "model/evspsegnet_stream.py",
                "utils/stream_common.py", "utils/stream_metrics.py", "utils/eval.py", "utils/comm_budget.py")
EXECUTIONS = ("step", "layer")
TRAINERS = ("sequence", "streams")


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
    parser.add_argument("--neuron", choices=("lif", "graded", "relu", "tif"), default=None,
                        help="lif 有状态+脉冲 | graded 有状态+实数（分离脉冲与记忆的对照）| relu 无状态+实数 | "
                             "tif V5 TIF 神经元（配置见 YAML 的 tif 小节）")
    parser.add_argument("--checkpoint", default=None, help="eval 模式使用的权重文件")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--sequence", default=None, help="overfit 模式的序列文件名")
    parser.add_argument("--steps", type=int, default=300, help="overfit 模式的参数更新次数")
    parser.add_argument("--resume", action="store_true", help="train 模式从 save_root/last.pt 继续")
    parser.add_argument("--overwrite", action="store_true", help="eval 模式允许覆盖已有结果文件")
    parser.add_argument("--readout-ablation", choices=("none", "zero_extra", "zero_feature"),
                        default="none",
                        help="eval 模式的读出消融：zero_extra 屏蔽 p/t_local；zero_feature 屏蔽网络特征")
    parser.add_argument("--memory-ablation", choices=("none", "off"), default="none",
                        help="仅 eval：off 关闭所有选中层的双仓 M 读取，保留普通 LIF 跨窗 U；结果另存")
    parser.add_argument("--memory-off-layers", nargs="+", choices=LAYER_NAMES, default=None,
                        help="仅 eval：只关闭指定双仓层的 M 读取，如 enc3 enc4；不可与 --memory-ablation off 合用")
    parser.add_argument("--subthreshold-ablation",
                        choices=("none", "vertical_off", "skip_off", "all_off"), default="none",
                        help="仅 eval：同权重关闭亚阈值残差的纵向路径、事件跳连或两者")
    parser.add_argument("--eval-noise-load", type=float, default=0.0,
                        help="仅 eval：测试时注入齐次泊松背景噪声，强度用 5x5 邻域每窗的随机巧合负荷 L 表示"
                             "（L = 25 x 每像素每窗的噪声率；L = 1 是巧合的物理分界）。0 = 不注入")
    parser.add_argument("--eval-noise-seed", type=int, default=0, help="仅 eval：注入噪声的随机种子")
    parser.add_argument("--dump-dir", default=None,
                        help="eval 模式：把训练时状态模式下的逐事件预测按序列保存为 NPZ，供 tools/verify_predictions.py 独立核验")
    parser.add_argument("--reset-every", type=int, default=0,
                        help="仅 eval（记忆长度诊断）：carry 推理中，窗号 k>0 且 (k-offset)%%M==0 时把全部状态清零；0 为关闭")
    parser.add_argument("--reset-offset", type=int, default=0, help="仅 eval：--reset-every 的相位偏移，0 <= offset < M")
    parser.add_argument("--execution", choices=EXECUTIONS, default=None,
                        help="训练与每轮验证的执行方式：step 逐窗（原实现）| layer 逐层时间并行")
    parser.add_argument("--input-device", choices=INPUT_DEVICES, default=None,
                        help="窗口输入在哪里构造：cpu（原实现）| gpu（网络所在设备）")
    parser.add_argument("--train-subset-every", type=int, default=None, help="每隔几轮评估一次训练子集")
    parser.add_argument("--tbptt-k", type=int, default=None, help="覆盖 YAML 中的 TBPTT 片段长度（窗口数）")
    parser.add_argument("--trainer", choices=TRAINERS, default=None,
                        help="sequence：原实现（每条序列一次更新）| streams：标准有状态 TBPTT（B 条并行流，每 n 个片段一次更新）")
    parser.add_argument("--streams", type=int, default=None, help="trainer=streams 的并行流数 B")
    parser.add_argument("--accumulate", type=int, default=None, help="trainer=streams 每几个片段更新一次参数（n）")
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
    for key in ("trainer", "streams", "accumulate"):
        if getattr(args, key, None) is not None:
            cfg[key] = getattr(args, key)
    cfg.setdefault("trainer", "sequence")
    cfg.setdefault("streams", 4)
    cfg.setdefault("accumulate", 1)
    cfg.setdefault("stream_layout", "greedy")
    if cfg["trainer"] not in TRAINERS:
        raise ValueError("trainer 必须是 %s 之一" % (TRAINERS,))
    if cfg["stream_layout"] not in STREAM_LAYOUTS:
        raise ValueError("stream_layout 必须是 %s 之一" % (STREAM_LAYOUTS,))
    if int(cfg["streams"]) < 1 or int(cfg["accumulate"]) < 1:
        raise ValueError("streams 与 accumulate 必须 >= 1")
    if cfg["trainer"] == "streams" and cfg.get("execution", "step") != "layer":
        raise ValueError("trainer=streams 需要 execution=layer")
    speed_options(cfg)
    if getattr(args, "memory_ablation", "none") != "none" and args.mode != "eval":
        raise ValueError("--memory-ablation 仅用于 eval；训练结构由 YAML 的 dual_memory 决定")
    if getattr(args, "memory_off_layers", None):
        if args.mode != "eval" or getattr(args, "memory_ablation", "none") != "none":
            raise ValueError("--memory-off-layers 仅用于 eval，且不可与 --memory-ablation off 合用")
    if getattr(args, "subthreshold_ablation", "none") != "none" and args.mode != "eval":
        raise ValueError("--subthreshold-ablation 仅用于 eval")
    check_periodic_reset(getattr(args, "reset_every", 0), getattr(args, "reset_offset", 0))
    if getattr(args, "reset_every", 0) and args.mode != "eval":
        raise ValueError("--reset-every 仅用于 eval")
    return cfg


def check_periodic_reset(every, offset):
    """校验周期清零参数：every=0 关闭；every>0 时要求 0 <= offset < every。"""
    every, offset = int(every), int(offset)
    if every < 0 or (every == 0 and offset != 0) or (every > 0 and not 0 <= offset < every):
        raise ValueError("--reset-every 必须 >= 0，且开启时 0 <= --reset-offset < --reset-every")
    return every, offset


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
        merged_decoder=cfg.get("merged_decoder", False), dual_memory=cfg.get("dual_memory"),
        subthreshold_residual=cfg.get("subthreshold_residual"),
        tif=cfg.get("tif") if cfg["neuron"] == "tif" else None)
    return net.to(device)


def design_version(net):
    """写入 run_config 与 checkpoint 的设计版本标记。"""
    if net.is_tif:
        if net.tif_config["adapt"]:
            return "v5-tif-input-adapt"
        return "v5-tif"
    if net.has_subthreshold_residual:
        return "v1-subthreshold-residual"
    return "v1-dual-memory" if net.has_dual_memory else "v1-3"


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
        neurons              本层神经元数（每个样本），取自实际张量；每窗脉冲数 = firing_rate * neurons
    ReLU 版本把"激活 > 0"视为发放。
    """

    def __init__(self, v_threshold):
        self.v_threshold = float(v_threshold)
        self.data = {}
        self.memory_data = {}
        self.residual_data = {}

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
            for name, memory in info.get("memory", {}).items():
                d = self.memory_data.get(name)
                if d is None:
                    d = {"abs_sum": torch.zeros((), device=memory.device, dtype=torch.float64),
                         "abs_max": torch.zeros((), device=memory.device), "elements": 0,
                         "near_bound": torch.zeros((), device=memory.device, dtype=torch.float64)}
                    self.memory_data[name] = d
                magnitude = memory.abs()
                d["abs_sum"] += magnitude.double().sum()
                d["abs_max"] = torch.maximum(d["abs_max"], magnitude.max())
                d["near_bound"] += (magnitude > 0.95).sum().double()
                d["elements"] += memory.numel()
            residual = info.get("subthreshold_residual")
            if residual is not None:
                magnitude = residual.abs()
                d = self.residual_data
                d["abs_sum"] = d.get("abs_sum", torch.zeros((), device=residual.device,
                                                              dtype=torch.float64)) + magnitude.double().sum()
                d["nonzero"] = d.get("nonzero", torch.zeros((), device=residual.device,
                                                              dtype=torch.float64)) + residual.ne(0).sum().double()
                d["positive"] = d.get("positive", torch.zeros((), device=residual.device,
                                                                dtype=torch.float64)) + residual.gt(0).sum().double()
                d["negative"] = d.get("negative", torch.zeros((), device=residual.device,
                                                                dtype=torch.float64)) + residual.lt(0).sum().double()
                d["abs_max"] = torch.maximum(d.get("abs_max", torch.zeros((), device=residual.device)),
                                               magnitude.max())
                d["elements"] = d.get("elements", 0) + residual.numel()
            vertical = info.get("subthreshold_vertical_signal")
            if vertical is not None:
                magnitude = vertical.abs()
                d = self.residual_data
                d["vertical_abs_sum"] = d.get(
                    "vertical_abs_sum", torch.zeros((), device=vertical.device,
                                                      dtype=torch.float64)) + magnitude.double().sum()
                d["vertical_nonzero"] = d.get(
                    "vertical_nonzero", torch.zeros((), device=vertical.device,
                                                      dtype=torch.float64)) + vertical.ne(0).sum().double()
                d["vertical_positive"] = d.get(
                    "vertical_positive", torch.zeros((), device=vertical.device,
                                                       dtype=torch.float64)) + vertical.gt(0).sum().double()
                d["vertical_negative"] = d.get(
                    "vertical_negative", torch.zeros((), device=vertical.device,
                                                       dtype=torch.float64)) + vertical.lt(0).sum().double()
                d["vertical_abs_max"] = torch.maximum(
                    d.get("vertical_abs_max", torch.zeros((), device=vertical.device)), magnitude.max())
                d["vertical_elements"] = d.get("vertical_elements", 0) + vertical.numel()
            gate = info.get("event_residual_gate")
            correction = info.get("event_residual_correction")
            if gate is not None:
                d = self.residual_data
                d["gate_sum"] = d.get("gate_sum", torch.zeros((), device=gate.device,
                                                                dtype=torch.float64)) + gate.double().sum()
                d["gate_support"] = d.get("gate_support", torch.zeros((), device=gate.device,
                                                                        dtype=torch.float64)) + gate.gt(0).sum().double()
                d["events"] = d.get("events", 0) + gate.numel()
                d["correction_abs_sum"] = d.get(
                    "correction_abs_sum", torch.zeros((), device=correction.device,
                                                       dtype=torch.float64)) + correction.abs().double().sum()
                d["correction_elements"] = d.get("correction_elements", 0) + correction.numel()

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
                "neurons": int(d["neuron_windows"].numel()),
            }
            if name in self.memory_data:
                m = self.memory_data[name]
                out[name].update(memory_abs_mean=float(m["abs_sum"]) / max(m["elements"], 1),
                                 memory_abs_max=float(m["abs_max"]),
                                 memory_near_bound_frac=float(m["near_bound"]) / max(m["elements"], 1))
        return out

    def residual_summary(self):
        """返回亚阈值残差和事件门控的聚合统计；没有该机制时返回空字典。"""
        d = self.residual_data
        if not d:
            return {}
        elements = max(d.get("elements", 0), 1)
        out = {"residual_abs_mean": float(d["abs_sum"]) / elements,
               "residual_abs_max": float(d["abs_max"]),
               "residual_nonzero_frac": float(d["nonzero"]) / elements,
               "residual_positive_frac": float(d["positive"]) / elements,
               "residual_negative_frac": float(d["negative"]) / elements}
        if d.get("vertical_elements", 0):
            vertical_elements = d["vertical_elements"]
            out.update(vertical_signal_abs_mean=float(d["vertical_abs_sum"]) / vertical_elements,
                       vertical_signal_abs_max=float(d["vertical_abs_max"]),
                       vertical_signal_nonzero_frac=float(d["vertical_nonzero"]) / vertical_elements,
                       vertical_signal_positive_frac=float(d["vertical_positive"]) / vertical_elements,
                       vertical_signal_negative_frac=float(d["vertical_negative"]) / vertical_elements)
        if d.get("events", 0):
            out.update(event_gate_mean=float(d["gate_sum"]) / d["events"],
                       event_gate_support_frac=float(d["gate_support"]) / d["events"],
                       event_correction_abs_mean=float(d["correction_abs_sum"])
                       / max(d["correction_elements"], 1))
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
        if isinstance(neuron, StreamingDualMemoryLIF2d):
            with torch.no_grad():
                mtau = neuron.memory_tau().double().cpu()
                alpha = neuron.memory_alpha().double().cpu()
                kappa = neuron.kappa().double().cpu()
                for prefix, values in (("memory_tau", mtau), ("memory_alpha", alpha), ("kappa", kappa)):
                    for suffix, value in (("min", values.min()), ("mean", values.mean()), ("max", values.max())):
                        out[name][prefix + "_" + suffix] = float(value)
                out[name]["kappa_abs_mean"] = float(kappa.abs().mean())
                out[name]["memory_read_enabled"] = neuron.memory_read_enabled
    return out


def tif_parameter_statistics(net):
    """TIF 各层参考值 k、量化步长 theta（adapt 时还有基线时间常数 tau_b，毫秒）的最小/平均/最大值；
    非 TIF 网络返回空字典。"""
    out = {}
    for name, block in zip(LAYER_NAMES, net.blocks()):
        neuron = block.neuron
        if not isinstance(neuron, StreamingTIF2d):
            continue
        with torch.no_grad():
            values = {"k": neuron.k(), "theta": neuron.theta()}
            if neuron.adapt:
                values["tau_b"] = neuron.adapt_tau()
            out[name] = {}
            for key, value in values.items():
                value = value.double().cpu()
                out[name].update({key + "_min": float(value.min()), key + "_mean": float(value.mean()),
                                  key + "_max": float(value.max())})
    return out


def health_layer_stats(layers, tif_stats):
    """健康检查的输入：TIF 网络按 TIF 的语义改写（非 TIF 原样返回）。

    TIF 的输出是持续存在的电平，"几乎每窗 q>0"只说明信念一直在，不是饱和，所以去掉 always_on_frac；
    死通道改用"M 始终为 0"（dead_channel_frac），饱和改用"电平到顶"（level_saturation_frac）。
    """
    if not tif_stats:
        return layers
    out = {}
    for name, stats in layers.items():
        stats = dict(stats)
        stats.pop("always_on_frac", None)
        if name in tif_stats and tif_stats[name]:
            stats["silent_channel_frac"] = tif_stats[name]["dead_channel_frac"]
            stats["level_saturation_frac"] = tif_stats[name]["saturation"]
        out[name] = stats
    return out


def subthreshold_parameter_statistics(net):
    """竞争式纵向与选择性横向的阈值、幅度统计；其他方案返回空字典。"""
    values = net.subthreshold_parameter_values()
    if not values:
        return {}
    out = {}
    with torch.no_grad():
        for name, tensor in values.items():
            tensor = tensor.detach().double().cpu()
            out[name + "_min"] = float(tensor.min())
            out[name + "_mean"] = float(tensor.mean())
            out[name + "_max"] = float(tensor.max())
            if name == "scale":
                out["scale_abs_mean"] = float(tensor.abs().mean())
                out["scale_positive_frac"] = float(tensor.gt(0).double().mean())
                out["scale_negative_frac"] = float(tensor.lt(0).double().mean())
    return out


def gradient_group_norms(net):
    """按顶层模块（enc1..dec1、up1..up3、readout）计算梯度 L2 范数；没有梯度的组记为 0。"""
    squares = {}
    for name, param in net.named_parameters():
        group = name.split(".")[0]
        value = 0.0 if param.grad is None else float(param.grad.detach().double().pow(2).sum())
        squares[group] = squares.get(group, 0.0) + value
    return {k: math.sqrt(v) for k, v in squares.items()}


def clip_statistics(total_norms, clipped_steps, clip):
    """汇总一轮内每次更新裁剪前的总梯度范数。

    effective_scale = min(1, clip/|g|) 是裁剪后实际施加的缩放：它越小，同一学习率下的实际步长越小。
    用于检验"k 越大 -> 时间方向梯度越大 -> 裁剪越多 -> 有效学习率越低"这一解释。
    """
    if not total_norms:
        nan = float("nan")
        return {"steps": 0, "total_norm_mean": nan, "total_norm_median": nan, "total_norm_p90": nan,
                "total_norm_max": nan, "clip_frac": 0.0, "effective_scale_mean": nan}
    arr = np.asarray(total_norms, dtype=np.float64)
    scale = np.minimum(1.0, float(clip) / np.maximum(arr, 1e-12))
    return {"steps": int(arr.size), "total_norm_mean": float(arr.mean()),
            "total_norm_median": float(np.median(arr)), "total_norm_p90": float(np.percentile(arr, 90)),
            "total_norm_max": float(arr.max()), "clip_frac": float(clipped_steps) / arr.size,
            "effective_scale_mean": float(scale.mean())}


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


def train_sequence(net, seq, optimizer, cfg, q99, pos_weight, device, rng, max_windows=None,
                   first_len=None):
    """用一个 8 秒序列训练（stateful TBPTT）。

    流程：随机第一段长度（1..K）切片段；每段前向 K 个窗口后立即反向并释放计算图，
    片段边界 detach 状态（只切梯度、保留数值）。
      optimizer_step=per_sequence：梯度跨片段累加，损失除以整条序列事件数，序列结束时裁剪并更新一次
      optimizer_step=per_chunk   ：回退预案，每段除以本段事件数并更新一次
    空窗口照常前向（使状态衰减），只是不产生损失；没有事件的片段不调用 backward。
    execution=layer 时每个片段调用一次 forward_chunk，片段损失是片段内全部事件的求和，与逐窗求和相同。
    max_windows 仅供冒烟测试截断使用。
    first_len 给定时第一段长度固定为它（不消耗 rng），供梯度探针对齐片段边界。
    optimizer=None 时只累加梯度、不裁剪不更新，梯度留在 p.grad 里（梯度探针用，须配合 per_sequence）。
    返回: {"loss_sum", "events", "grad_norms", "missing_grads", "steps",
           "grad_total_norms"（每次更新裁剪前的总范数）, "clipped_steps"（总范数超过 grad_clip 的次数）}
    """
    k = int(cfg["tbptt_k"])
    execution, _ = speed_options(cfg)
    n_windows = seq.n_windows if max_windows is None else min(int(max_windows), seq.n_windows)
    first = int(rng.randint(1, k + 1)) if first_len is None else int(first_len)
    chunks = make_chunks(n_windows, k, first)
    per_sequence = cfg["optimizer_step"] == "per_sequence"
    if cfg["optimizer_step"] not in ("per_sequence", "per_chunk"):
        raise ValueError("optimizer_step 必须是 per_sequence 或 per_chunk")
    if optimizer is None and not per_sequence:
        raise ValueError("optimizer=None（只算梯度）只支持 per_sequence")
    total_events = int(seq.bounds[n_windows] - seq.bounds[0])
    dtype = next(net.parameters()).dtype
    source = make_window_source(seq, cfg, q99, device, dtype)
    weight = torch.tensor([pos_weight], dtype=dtype, device=device)
    loss_sum, steps, grad_norms, missing = 0.0, 0, {}, []
    total_norms = []

    def step():
        """记录裁剪前的分组梯度范数、总范数与缺失梯度的参数名，然后裁剪并更新一次参数。"""
        norms = gradient_group_norms(net)
        lost = [name for name, p in net.named_parameters() if p.requires_grad and p.grad is None]
        if optimizer is None:
            return norms, lost
        total = float(torch.nn.utils.clip_grad_norm_(net.parameters(), float(cfg["grad_clip"])))
        total_norms.append(total)
        optimizer.step()
        return norms, lost

    if optimizer is not None:
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
    clip = float(cfg["grad_clip"])
    clipped = sum(1 for v in total_norms if v > clip)
    return {"loss_sum": loss_sum, "events": total_events, "grad_norms": grad_norms,
            "missing_grads": missing, "steps": steps,
            "grad_total_norms": total_norms, "clipped_steps": clipped}


# ---------------------------------------------------------------------------
# 标准有状态 TBPTT：B 条并行流（trainer=streams）
# ---------------------------------------------------------------------------


STREAM_LAYOUTS = ("greedy", "batchify")


def plan_streams(lengths, n_streams, k, rng, layout="greedy"):
    """把训练录像分配给 B 条并行流。lengths[i] 为数据集第 i 条录像的窗数；先按 rng 打乱录像顺序。

    layout="greedy"（默认，EV-UAV / EV-Flying 的结果都用它）：整段录像依次放进当前总窗数最少的流。
        最长的录像不超过"总窗数 / B"时各流长度接近；超过时（如 EBSSA 一段占训练集 29%）分不平，
        一轮末尾只剩一条流在跑，连续很多次更新来自同一段录像。
    layout="batchify"（语言模型 TBPTT 的标准排法）：打乱后首尾接成一条长带，等分成 B 段，每条流一样长、同时结束；
        跨等分点的录像被切开，后半截作为新起点（该流状态清零），每轮最多切 B-1 刀。
    每条流开头随机延迟 0..k-1 个空窗（随机片段相位）。录像内部的窗顺序不变。
    返回: [(delay, items), ...]，共 n_streams 条；items 元素为 (index, n_windows)（整段）或 (index, first, n_windows)
    （从录像第 first 窗起的 n_windows 窗）。
    """
    if layout not in STREAM_LAYOUTS:
        raise ValueError("stream_layout 必须是 %s 之一" % (STREAM_LAYOUTS,))
    order = rng.permutation(len(lengths))
    n_streams = int(n_streams)
    streams = [[] for _ in range(n_streams)]
    if layout == "greedy":
        totals = [0] * n_streams
        for i in order:
            s = int(np.argmin(totals))
            streams[s].append((int(i), int(lengths[i])))
            totals[s] += int(lengths[i])
    else:
        total = int(sum(int(n) for n in lengths))
        bounds = [s * total // n_streams for s in range(n_streams + 1)]
        pos, s = 0, 0
        for i in order:
            n, first = int(lengths[i]), 0
            while first < n:
                while pos >= bounds[s + 1]:
                    s += 1
                take = min(n - first, bounds[s + 1] - pos)
                streams[s].append((int(i), first, take))
                first += take
                pos += take
    return [(int(rng.randint(0, int(k))), items) for items in streams]


def plan_item(item):
    """计划里的一项 -> (index, first, n_windows)。"""
    return (item[0], 0, item[1]) if len(item) == 2 else tuple(item)


def stream_total_windows(plan):
    """各流时间轴长度（含开头延迟）的最大值，即一轮需要的窗数。"""
    return max(delay + sum(item[-1] for item in items) for delay, items in plan)


def stream_pieces(plan, chunk_start, chunk_end):
    """流时间轴上窗 [chunk_start, chunk_end) 这一片段内，每条流由哪些录像的哪些窗组成。

    返回: 每条流一个列表 [(index, a, b, offset, new_start)]：录像 index 的第 a..b-1 窗放在片段内第 offset 窗起；
          new_start=True 表示计划中的这一项从这一窗开始（整段录像的开头，或 batchify 切开后的后半截；该流状态在此清零）。
          没有录像的位置是空窗（零输入、无事件）。
    """
    out = []
    for delay, items in plan:
        pieces, start = [], delay
        for item in items:
            index, first, n = plan_item(item)
            if start >= chunk_end:
                break
            end = start + n
            if end > chunk_start:
                lo, hi = max(chunk_start, start), min(chunk_end, end)
                pieces.append((index, first + lo - start, first + hi - start, lo - chunk_start, lo == start))
            start = end
        out.append(pieces)
    return out


class StreamSources(object):
    """多流训练按需读取录像并构造窗口数据来源；录像的最后一窗用完即释放。"""

    def __init__(self, dataset, cfg, q99, device, dtype, augment=None):
        self.dataset, self.cfg, self.q99, self.device, self.dtype = dataset, cfg, q99, device, dtype
        self.augment = augment            # E1 噪声增强（dataset/noise_aug.NoiseAugment）或 None
        self.sources, self.bounds = {}, {}

    def get(self, index):
        if index not in self.sources:
            seq = self.dataset[index]
            if self.augment is not None:
                seq = self.augment(seq)
            self.sources[index] = make_window_source(seq, self.cfg, self.q99, self.device, self.dtype)
            self.bounds[index] = seq.bounds
        return self.sources[index]

    def events(self, index, lo, hi):
        """录像 index 第 lo..hi-1 窗的事件数（不经过网络，只读窗口边界）。"""
        self.get(index)
        return int(self.bounds[index][hi] - self.bounds[index][lo])

    def release(self, index):
        self.sources.pop(index, None)
        self.bounds.pop(index, None)


def assemble_streams(sources, pieces, steps, cfg, device, dtype):
    """把各流的片段组成拼成一个批：x [T,B,C,H,W]、events（b = 流号，t = 片段内窗号）、labels [N]、keep [T,B]。"""
    batch = len(pieces)
    x = torch.zeros((steps, batch, num_input_channels(cfg["time_bins"]), int(cfg["pad_height"]),
                     int(cfg["pad_width"])), dtype=dtype, device=device)
    keep = torch.ones((steps, batch), dtype=dtype, device=device)
    parts, labels = {key: [] for key in ("t", "b", "y", "x", "p", "t_local")}, []
    for b, stream in enumerate(pieces):
        for index, lo, hi, offset, new_start in stream:
            xs, events, lab, _ = sources.get(index).chunk(lo, hi)
            x[offset:offset + (hi - lo), b] = xs[:, 0]
            if new_start:
                keep[offset, b] = 0
            parts["t"].append(events["t"] + int(offset))
            parts["b"].append(torch.full_like(events["t"], b))
            for key in ("y", "x", "p", "t_local"):
                parts[key].append(events[key])
            labels.append(lab)
    if labels:
        events = {key: torch.cat(value) for key, value in parts.items()}
        labels = torch.cat(labels)
    else:
        long_empty = torch.zeros(0, dtype=torch.long, device=device)
        float_empty = torch.zeros(0, dtype=dtype, device=device)
        events = {"t": long_empty, "b": long_empty, "y": long_empty, "x": long_empty,
                  "p": float_empty, "t_local": float_empty}
        labels = float_empty
    return x, events, labels, keep


def stream_valid(pieces, steps, device, dtype):
    """[T,B]：该流在片段内该窗有录像为 1，空窗（流错开起点、流尾补齐）为 0。"""
    valid = torch.zeros((steps, len(pieces)), dtype=dtype, device=device)
    for b, stream in enumerate(pieces):
        for _, lo, hi, offset, _ in stream:
            valid[offset:offset + (hi - lo), b] = 1
    return valid


def train_streams(net, dataset, lengths, optimizer, cfg, q99, pos_weight, device, rng, max_chunks=None, plan=None,
                  comm=None):
    """标准有状态 TBPTT（一轮）：B 条并行流一起前向 k 窗、反向 k 窗，每 n 个片段更新一次参数。

    状态跨片段延续，片段边界 detach（只切梯度、保留数值）；某条流开始新录像时只清零该流状态（keep 掩码）。
    损失按"一次更新"归一：一次更新包含的 n 个片段里全部事件的 BCE 之和 / 这 n 个片段的总事件数
    （与梯度累加的常规做法相同；事件数只读窗口边界、不经过网络，所以可以在反传前算好）。
    因此 n=1 即逐片段更新（原 per_chunk），B=1 且 n = 录像窗数 / k 时即原 per_sequence。没有事件的片段不反传。
    推理不受影响，仍是整段连续。max_chunks 只供冒烟和计时截断；plan 只供测试指定分配。
    comm 为 utils.comm_budget.CommBudget（通信预算训练，乘子 mu 跨轮延续）或 None（关闭，与原训练逐位相同）。
    返回字段与 train_sequence 相同（steps = 参数更新次数，grad_norms 为各次更新的平均），另有 chunks；
    开启通信预算时另有 comm（CommBudget.summary()）。
    """
    k, n_streams, accumulate = int(cfg["tbptt_k"]), int(cfg["streams"]), int(cfg["accumulate"])
    if speed_options(cfg)[0] != "layer":
        raise ValueError("trainer=streams 需要 execution=layer（一次前向一个 [k,B] 片段）")
    if plan is None:
        plan = plan_streams(lengths, n_streams, k, rng, cfg.get("stream_layout", "greedy"))
    # 计划中每一项的最后一窗：用完即释放该录像（batchify 切开的录像若后面还要用，会重新读取）
    item_ends = {(index, first + n) for _, items in plan for index, first, n in map(plan_item, items)}
    n_chunks = int(math.ceil(stream_total_windows(plan) / float(k)))
    if max_chunks is not None:
        n_chunks = min(n_chunks, int(max_chunks))
    dtype = next(net.parameters()).dtype
    augment = None
    if noise_aug_settings(cfg)[0] > 0.0:
        # 独立的随机数流：只在开启时从 rng 取一次种子，关闭时 rng 的消耗与原来相同
        augment = NoiseAugment(cfg, np.random.RandomState(int(rng.randint(2 ** 31 - 1))))
    sources = StreamSources(dataset, cfg, q99, device, dtype, augment)
    weight = torch.tensor([pos_weight], dtype=dtype, device=device)
    clip = float(cfg["grad_clip"])
    loss_sum, events_total, updates, grad_acc, total_norms, missing = 0.0, 0, 0, {}, [], []
    if comm is not None:
        comm.start_epoch()
    states = None
    optimizer.zero_grad(set_to_none=True)
    for first in range(0, n_chunks, accumulate):
        block = [stream_pieces(plan, c * k, (c + 1) * k) for c in range(first, min(first + accumulate, n_chunks))]
        block_events = sum(sources.events(index, lo, hi)
                           for pieces in block for stream in pieces for index, lo, hi, _, _ in stream)
        for offset, pieces in enumerate(block):
            x, events, labels, keep = assemble_streams(sources, pieces, k, cfg, device, dtype)
            prev_states = states
            logits, states, info = net.forward_chunk(x, events, states, keep=keep, collect=comm is not None)
            if labels.numel():
                chunk_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight, reduction="sum")
                total = chunk_loss / float(block_events)
                if comm is not None:
                    penalty = comm.chunk_loss(net, info["u_pre"], prev_states, keep, len(block),
                                              stream_valid(pieces, k, device, dtype))
                    if penalty is not None:
                        total = total + penalty
                total.backward()
                value = float(chunk_loss.detach())
                if not math.isfinite(value):
                    raise RuntimeError("损失出现非有限值: 片段 %d" % (first + offset))
                loss_sum += value
                events_total += int(labels.numel())
            states = detach_states(states)
            for stream in pieces:
                for index, lo, hi, _, _ in stream:
                    if (index, hi) in item_ends:
                        sources.release(index)
        if block_events > 0:
            norms = gradient_group_norms(net)
            missing = [name for name, p in net.named_parameters() if p.requires_grad and p.grad is None]
            total_norms.append(float(torch.nn.utils.clip_grad_norm_(net.parameters(), clip)))
            optimizer.step()
            for key, value in norms.items():
                grad_acc[key] = grad_acc.get(key, 0.0) + value
            updates += 1
            if comm is not None:
                comm.end_update()                                    # 对偶上升
        optimizer.zero_grad(set_to_none=True)
    out = {"loss_sum": loss_sum, "events": events_total,
           "grad_norms": {key: value / max(updates, 1) for key, value in grad_acc.items()},
           "missing_grads": missing, "steps": updates, "chunks": n_chunks,
           "grad_total_norms": total_norms, "clipped_steps": sum(1 for v in total_norms if v > clip)}
    if comm is not None:
        out["comm"] = comm.summary()
    if augment is not None:
        out["noise_aug"] = augment.summary()
    return out


def count_nonzero_input(stats, x):
    """累计输入中的非零元素个数与总数（事件驱动口径的能耗要用；stats 为 None 时什么都不做）。

    计数留在张量上累加、结束时才取值：如果每个窗口都 int(...) 取回主机，就会每窗同步一次 GPU，
    既拖慢评估，也会污染逐窗计时（论文的延迟数字来自那里）。调用位置同样要放在计时区间之外。
    """
    if stats is not None:
        stats["nonzero"] = stats["nonzero"] + (x != 0).sum()
        stats["elements"] += int(x.numel())


def nonzero_fraction(stats):
    """把累计结果换算成非零元素占比（在这里才同步一次）。"""
    return float(stats["nonzero"]) / float(max(stats["elements"], 1))


def run_sequence(net, seq, cfg, q99, device, state_mode, monitor=None, timer=None, input_stats=None,
                 reset_every=0, reset_offset=0):
    """推理一个完整序列（全部窗口，按时间顺序，状态按 state_mode 处理）。

    返回: (probs, confusion)
        probs      按文件原始事件顺序回填的预测概率（numpy float32，长度 N）
        confusion  [n_windows, 4] 每窗 TP/FP/FN/正事件数（阈值 cfg["threshold"]）
    调用方负责 torch.no_grad() 与 net.eval()。
    execution=layer 时每 eval_chunk 个窗口调用一次 forward_chunk（吞吐快，但不能用来测单窗延迟，timer 必须为空）。
    input_stats 不为空时，累计输入中非零元素的个数与总数（事件驱动口径的能耗要用，见 estimate_operations）。
    reset_every=M>0（记忆长度诊断，只用于 step 执行）：窗号 k>0 且 (k-reset_offset)%M==0 时先把全部状态清零，
    于是第 k 窗看到的历史长度 h = k - 最近一次清零的窗号 + 1（k=0 本来就从零状态开始）；M=1 等价于 reset_each_window。
    """
    execution, _ = speed_options(cfg)
    if execution == "layer" and timer is not None:
        raise ValueError("逐窗延迟只能在 execution=step 下测量")
    reset_every, reset_offset = check_periodic_reset(reset_every, reset_offset)
    if reset_every and execution == "layer":
        raise ValueError("周期清零只支持 execution=step")
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
            if reset_every and k > 0 and (k - reset_offset) % reset_every == 0:
                states = None
            t0 = timer.now() if timer else None
            x, events, _, idx = source.window(k)
            t1 = timer.now() if timer else None
            logits, states, info = net(x, events, states, state_mode=state_mode, collect=monitor is not None)
            t2 = timer.now() if timer else None
            prob = torch.sigmoid(logits).float().cpu().numpy()
            if timer:
                timer.add(t1 - t0, t2 - t1, timer.now() - t2)
            count_nonzero_input(input_stats, x)          # 放在计时之后，不计入单窗耗时
            if monitor is not None:
                monitor.update(info)
            index_parts.append(idx)
            prob_parts.append(prob)
            confusion[k] = window_confusion(prob, seq.label[idx], threshold)
    probs = refill_by_index(seq.n_events, index_parts, prob_parts)
    return probs, confusion


def eval_noise(seq, cfg, index):
    """鲁棒性评估（Inn 13.51）：cfg 里有 eval_noise_load > 0 时，给第 index 个序列注入齐次泊松背景噪声（标签 0）。

    噪声率 = L / 25（每像素每窗，ON / OFF 各半）；每个序列用固定种子（eval_noise_seed x 1000 + index），
    carry / reset 两次评估、不同模型之间注入的噪声逐事件相同。L = 0 时原样返回。
    """
    load = float(cfg.get("eval_noise_load", 0.0) or 0.0)
    if load <= 0:
        return seq
    rng = np.random.RandomState(int(cfg.get("eval_noise_seed", 0)) * 1000 + int(index))
    return inject_background_noise(seq, load / 25.0, int(cfg["height"]), int(cfg["width_px"]), int(cfg["window_ms"]),
                                   int(seq.bounds.shape[0]) - 1, int(cfg["time_bins"]), rng)


def evaluate_split(net, cfg, q99, device, split, state_mode, names=None, full=False,
                   collect_layers=False, dump_dir=None, reset_every=0, reset_offset=0):
    """在一个划分（或其子集）上评估。

    主指标 IoU/ACC 调用原仓库 utils/eval.py 的 evaluate_iou_and_accuracy（全局拼接口径）；
    full=True 时额外计算 Pd/Fa（utils/stream_metrics.FastRoc：与原 roc_update / cal_roc 逐位等价的向量化实现，
    传感器尺寸取 cfg 的 height / width_px，EV-UAV 即 260x346），并统计逐窗 IoU、分段 IoU、首次检出延迟、耗时、理论运算量。
    序列长度可以不同（n_windows<=0 时按录像自身长度）：逐窗统计按窗号对齐累加，较短的序列只贡献前面的窗。
    collect_layers=True 时统计每层发放与膜电位（训练中每轮验证也会用到，用于健康门槛）。
    dump_dir 不为空时，把每个序列的逐事件预测保存为 NPZ（字段与原仓库诊断文件一致：
    locs [batch,x,y,t]、labels、probabilities，另加 target_id），文件顺序与原 NPZ 一致。
    reset_every / reset_offset 原样传给 run_sequence（周期清零，记忆长度诊断）。
    """
    from utils.eval import evalute        # 延迟导入：该模块依赖 cv2/pandas，只在评估时需要

    dataset = EvUAVStream(cfg["root"], split, cfg, names)
    evaluator = evalute(SimpleNamespace(roc=False, pd_detT=cfg["pd_detT"],
                                        correct_thresh=cfg["correct_thresh"]))
    roc = FastRoc(cfg["pd_detT"], cfg["correct_thresh"], cfg["height"], cfg["width_px"]) if full else None
    monitor = LayerMonitor(net.v_threshold) if (collect_layers or full) else None
    timer = WindowTimer(cfg["timing_warmup_windows"], device) if full else None
    input_stats = {"nonzero": 0, "elements": 0} if full else None
    confusion_sum = np.zeros((0, 4), dtype=np.int64)
    sweep = np.zeros((len(SWEEP_THRESHOLDS), 3), dtype=np.int64)
    kept, total_events, total_windows = [], 0, 0
    was_training = net.training
    net.eval()
    if monitor is not None:
        net.start_tif_stats(per_window=True)  # TIF 消息率与电平统计（含逐窗，用于单窗 SOP 分布）；非 TIF 网络不做任何事
    with torch.no_grad():
        for i in range(len(dataset)):
            seq = eval_noise(dataset[i], cfg, i)
            probs, confusion = run_sequence(net, seq, cfg, q99, device, state_mode, monitor, timer,
                                            input_stats, reset_every, reset_offset)
            if confusion.shape[0] > confusion_sum.shape[0]:
                confusion_sum = np.concatenate(
                    [confusion_sum, np.zeros((confusion.shape[0] - confusion_sum.shape[0], 4), dtype=np.int64)])
            confusion_sum[:confusion.shape[0]] += confusion
            sweep += threshold_counts(probs, seq.label)
            total_events += seq.n_events
            total_windows += seq.n_windows
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
                roc.update(seq.t, probs, seq.target_id, seq.label, seq.x, seq.y, float(cfg["threshold"]))
                kept.append((seq.t, seq.label, seq.target_id, probs))
    tif_stats = net.stop_tif_stats() if monitor is not None else {}
    net.train(was_training)
    iou, acc = evaluator.evaluate_iou_and_accuracy(thresh=float(cfg["threshold"]))
    result = {"split": split, "state_mode": state_mode, "n_sequences": len(dataset),
              "iou": float(iou), "acc": float(acc), "threshold_sweep": sweep_summary(sweep)}
    if monitor is not None:
        result["layers"] = monitor.summary()
        if monitor.residual_summary():
            result["subthreshold_diagnostics"] = monitor.residual_summary()
        if tif_stats:
            # 逐窗分布；逐窗列表用完即删，不写进结果
            #   messages_per_window  每窗实际发出的消息数（通信约束的检验口径，与能耗无关，12.35）
            #   sop_per_window       单窗 SOP（按网络自身结构的能耗口径，只用于报告，12.21）
            if net.tif_config["quantize"]:
                window_msg = window_message_summary(net, tif_stats)
                if window_msg is not None:
                    result["messages_per_window"] = window_msg
                window_sop = window_sop_summary(net, tif_stats, int(cfg["pad_height"]), int(cfg["pad_width"]))
                if window_sop is not None:
                    result["sop_per_window"] = window_sop
            for layer in tif_stats.values():
                layer.pop("per_window_rate", None)
            result["tif"] = tif_stats
        # 每窗实际发出的消息数的平均（TIF 为发送层 |Δq| 之和，LIF 为发送层脉冲数之和；同结构 LIF 的值即通信预算）
        messages = mean_messages(net, result["layers"], result.get("tif"))
        if messages is not None:
            result["messages_mean"] = messages
    if full:
        pd, fa = roc.result()
        tp, fp, fn, npos = (confusion_sum[:, j] for j in range(4))
        timing = timer.summary(cfg["window_ms"])
        net_ms = timing.get("net_ms_mean", 0.0)
        records = []
        for t, label, target_id, probs in kept:
            records += first_detection_latencies(t, label, target_id, probs, cfg["window_ms"],
                                                 cfg["threshold"], cfg["correct_thresh"], net_ms)
        rates = {name: s["firing_rate"] for name, s in result["layers"].items()}
        if net.is_tif and net.tif_config["quantize"]:
            # TIF 层间传的是整数变化量 Δq：按每神经元每窗平均 |Δq| 计突触累加（单位脉冲口径）
            rates.update({name: tif_stats[name]["message_abs_rate"] for name in LAYER_NAMES})
        rates["subthreshold_vertical"] = result.get(
            "subthreshold_diagnostics", {}).get("vertical_signal_nonzero_frac", 0.0)
        result.update({
            "pd": float(pd), "fa": float(fa),
            "segment_iou": segment_iou(tp, fp, fn, cfg["segments"]),
            "rolling_iou": [None if math.isnan(v) else v
                            for v in rolling_iou(tp, fp, fn, int(cfg["rolling_window"])).tolist()],
            "per_window": {"tp": tp.tolist(), "fp": fp.tolist(), "fn": fn.tolist(), "pos": npos.tolist()},
            "latency": summarize_latencies(records),
            "timing": timing,
            "input_nonzero_fraction": nonzero_fraction(input_stats),
            "operations": estimate_operations(net, cfg["pad_height"], cfg["pad_width"], rates,
                                              total_events / float(max(total_windows, 1)),
                                              nonzero_fraction(input_stats)),
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
        "design_version": design_version(net), "tif": net.tif_config,
        "args": vars(args), "config": cfg, "stats": stats,
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
    names = train_file_names(cfg)
    seq = EvUAVStream(cfg["root"], "train", cfg, [names[0]])[0]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    if cfg.get("trainer", "sequence") == "streams":
        # B 条流各一条录像，跑 2 个片段（32 窗 / k=16），覆盖拼批、按流清零、更新
        smoke_set = EvUAVStream(cfg["root"], "train", cfg, names[:max(int(cfg["streams"]), 1)])
        lengths = [int(smoke_set[i].n_windows) for i in range(len(smoke_set))]
        comm = CommBudget.from_config(cfg, net)
        if comm is not None:
            comm.mu = 1.0                       # 冒烟时让惩罚项真正参与反传
        out = train_streams(net, smoke_set, lengths, optimizer, cfg, q99, pos_weight, device,
                            np.random.RandomState(int(cfg["seed"])), max_chunks=max(1, 32 // int(cfg["tbptt_k"])),
                            comm=comm)
    else:
        out = train_sequence(net, seq, optimizer, cfg, q99, pos_weight, device,
                             np.random.RandomState(int(cfg["seed"])), max_windows=32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    monitor = LayerMonitor(net.v_threshold)
    with torch.no_grad():
        net.eval()
        net.start_tif_stats()
        states = None
        for k in range(32):
            x, events, _, _ = window_to_device(seq, k, q99, cfg, device)
            _, states, info = net(x, events, states, collect=True)
            monitor.update(info)
        tif_stats = net.stop_tif_stats()
        net.train()
    layers, taus = monitor.summary(), tau_statistics(net)
    report = {"sequence": seq.name, "events_in_32_windows": out["events"], "loss_per_event":
              out["loss_sum"] / max(out["events"], 1), "grad_norms": out["grad_norms"],
              "missing_grads": out["missing_grads"], "train_ms_per_window": 1e3 * elapsed / 32,
              "peak_memory_gib": peak_memory_gib(device), "layers": layers, "tau": taus,
              "tif": tif_stats, "tif_parameters": tif_parameter_statistics(net),
              "subthreshold_diagnostics": monitor.residual_summary(),
              "subthreshold_parameters": subthreshold_parameter_statistics(net),
              "warnings": health_warnings(health_layer_stats(layers, tif_stats), taus, out["grad_norms"])}
    write_json(root / "smoke.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    missing = out["missing_grads"]
    if cfg["state_mode"] == "reset_each_window":
        # reset 模式下 U/M/b 都不跨窗读取，时间常数、双仓耦合参数与 TIF 基线时间常数没有梯度是预期行为。
        missing = [name for name in missing if not name.endswith(
            (".neuron.a", ".neuron.memory_a", ".neuron.kappa_raw", ".neuron.adapt_a"))]
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
                "design_version": design_version(net)}, str(path))


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
        if ckpt["config"].get("dual_memory") != cfg.get("dual_memory"):
            raise ValueError("续训 dual_memory 配置与 checkpoint 不一致，不能改变时间常数范围或读取上限")
        if ckpt["config"].get("subthreshold_residual") != cfg.get("subthreshold_residual"):
            raise ValueError("续训 subthreshold_residual 配置与 checkpoint 不一致")
        if ckpt["config"].get("neuron") != cfg.get("neuron") or (
                cfg.get("neuron") == "tif"
                and tif_settings(ckpt["config"].get("tif")) != tif_settings(cfg.get("tif"))):
            raise ValueError("续训 neuron / tif 配置与 checkpoint 不一致")
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best = int(ckpt["epoch"]) + 1, float(ckpt["best_val_iou"])
        previous = (ckpt["config"].get("execution", "step"), ckpt["config"].get("input_device", "cpu"))
        if previous != speed_options(cfg):
            print("注意：续训的执行方式与原运行不同（%s -> %s）。两者数学等价，但浮点舍入不同，"
                  "曲线不会与不中断的运行逐位一致" % (previous, speed_options(cfg)), flush=True)
        history = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
        history = [h for h in history if h["epoch"] < start_epoch]
        print("从 epoch %d 继续" % start_epoch, flush=True)
    else:
        if metrics_path.exists():
            raise RuntimeError("输出目录已有 metrics.jsonl，请换目录或使用 --resume")
        run_calibration(net, cfg, q99, device, root)
    comm = CommBudget.from_config(cfg, net)          # 通信预算训练（utils/comm_budget.py）；未配置时为 None
    if comm is not None:
        if history and history[-1].get("comm"):
            comm.mu = float(history[-1]["comm"]["mu_end"])                  # 续训：乘子接着上一轮
        print("通信预算训练（每窗消息数的峰值约束）：每窗消息预算 %.4g，违约比例上限 %.0f%%，对偶步长 %g，"
              "发送层 %s" % (comm.budget, 100.0 * comm.alpha, comm.dual_lr, " ".join(net.message_layers())), flush=True)
    subset = select_subset(train_file_names(cfg), int(cfg["train_subset_size"]), seed + 2000)
    train_set = EvUAVStream(cfg["root"], "train", cfg)
    execution, input_device = speed_options(cfg)
    trainer = cfg.get("trainer", "sequence")
    lengths = None
    if trainer == "streams":
        # 各录像窗数（分配并行流用）；读一遍训练集，此后每轮按需读取
        lengths = [int(train_set[i].n_windows) for i in range(len(train_set))]
        print("标准 TBPTT：%d 条并行流（排法 %s）、k=%d、每 %d 个片段更新一次（每次更新 %d 窗）| 训练集 %d 条录像 %d 窗" % (
            int(cfg["streams"]), cfg.get("stream_layout", "greedy"), int(cfg["tbptt_k"]), int(cfg["accumulate"]),
            int(cfg["streams"]) * int(cfg["tbptt_k"]) * int(cfg["accumulate"]), len(lengths), sum(lengths)), flush=True)
    print("执行方式 %s | 输入构造 %s | 训练子集每 %d 轮评估 | 验证片段 %d 窗" % (
        execution, input_device, int(cfg.get("train_subset_every", 1)), int(cfg.get("eval_chunk", 32))), flush=True)
    for epoch in range(start_epoch, epochs):
        lr = linear_epoch_lr(epoch, epochs, float(cfg["lr"]), float(cfg["lr_end"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        rng = np.random.RandomState(seed * 1000 + epoch + 7)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        net.train()
        loss_sum, event_sum, grad_acc, n_seq = 0.0, 0, {}, 0
        total_norms, clipped_steps, updates = [], 0, 0
        comm_record = None
        if trainer == "streams":
            out = train_streams(net, train_set, lengths, optimizer, cfg, q99, pos_weight, device, rng, comm=comm)
            loss_sum, event_sum, grad_acc, n_seq = out["loss_sum"], out["events"], out["grad_norms"], 1
            total_norms, clipped_steps, updates = out["grad_total_norms"], out["clipped_steps"], out["steps"]
            comm_record = out.get("comm")
            if out.get("noise_aug") is not None:
                print("  noise_aug: %s" % json.dumps(out["noise_aug"]), flush=True)
        else:
            loader = make_sequence_loader(train_set, True, seed * 1000 + epoch, cfg["num_workers"])
            for seq in loader:
                out = train_sequence(net, seq, optimizer, cfg, q99, pos_weight, device, rng)
                loss_sum += out["loss_sum"]
                event_sum += out["events"]
                for key, value in out["grad_norms"].items():
                    grad_acc[key] = grad_acc.get(key, 0.0) + value
                total_norms += out["grad_total_norms"]
                clipped_steps += out["clipped_steps"]
                updates += out["steps"]
                n_seq += 1
        train_seconds = time.perf_counter() - t0
        grad_mean = {k: v / max(n_seq, 1) for k, v in grad_acc.items()}
        grad_clip_stats = clip_statistics(total_norms, clipped_steps, float(cfg["grad_clip"]))
        val = evaluate_split(net, cfg, q99, device, "val", cfg["state_mode"], collect_layers=True)
        sub = None
        if subset_due(epoch, epochs, int(cfg.get("train_subset_every", 1))):
            sub = evaluate_split(net, cfg, q99, device, "train", cfg["state_mode"], names=subset)
        taus = tau_statistics(net)
        if comm_record is not None and val.get("tif"):
            # 只作对照、不参与优化：验证集上每窗消息数的平均与 p95（与预算同口径）
            comm_record["val_messages"] = val.get("messages_mean")
            comm_record["val_messages_p95"] = (val.get("messages_per_window") or {}).get("p95")
        record = {"epoch": epoch, "seed": seed, "lr": lr, "loss_per_event": loss_sum / max(event_sum, 1),
                  "val_iou": val["iou"], "val_acc": val["acc"],
                  "train_subset_iou": sub["iou"] if sub else None, "train_subset_acc": sub["acc"] if sub else None,
                  "execution": execution, "input_device": input_device,
                  "train_seconds": train_seconds, "epoch_seconds": time.perf_counter() - t0,
                  "peak_memory_gib": peak_memory_gib(device), "grad_norms": grad_mean,
                  "grad_clip_stats": grad_clip_stats, "tbptt_k": int(cfg["tbptt_k"]),
                  "trainer": trainer, "streams": int(cfg.get("streams", 1)), "accumulate": int(cfg.get("accumulate", 1)),
                  "stream_layout": cfg.get("stream_layout", "greedy"),
                  "updates": updates,
                  "val_threshold_sweep": val["threshold_sweep"],
                  "layers": val["layers"], "tau": taus,
                  "tif": val.get("tif", {}), "tif_parameters": tif_parameter_statistics(net),
                  "subthreshold_diagnostics": val.get("subthreshold_diagnostics", {}),
                  "subthreshold_parameters": subthreshold_parameter_statistics(net),
                  "comm": comm_record,
                  "warnings": health_warnings(health_layer_stats(val["layers"], val.get("tif")), taus, grad_mean)}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        history.append(record)
        if val["iou"] > best:
            best = val["iou"]
            save_checkpoint(root / ("best_val_iou_seed%d.pt" % seed), net, optimizer, epoch, best, cfg, stats)
        save_checkpoint(root / "last.pt", net, optimizer, epoch, best, cfg, stats)
        print("epoch %d lr %.2e loss/event %.5f val IoU %.4f ACC %.4f | train-subset IoU %s | "
              "更新 %d 次 |g| 中位 %.3f 裁剪 %.0f%% | %.0fs（训练 %.0fs）| 显存峰值 %.2f GiB | 警告 %d" % (
                  epoch, lr, record["loss_per_event"], val["iou"], val["acc"],
                  "%.4f" % sub["iou"] if sub else "-", updates,
                  grad_clip_stats["total_norm_median"], 100.0 * grad_clip_stats["clip_frac"],
                  record["epoch_seconds"], train_seconds, record["peak_memory_gib"] or 0.0,
                  len(record["warnings"])), flush=True)
        if record["tif"]:
            print("  [tif] 每窗 |Δq| " + " ".join("%s %.4f" % (n, record["tif"][n]["message_abs_rate"])
                                                  for n in LAYER_NAMES)
                  + ("" if "baseline_level_mean" not in record["tif"]["enc1"] else
                     " | 基线/theta " + " ".join("%s %.3f" % (n, record["tif"][n]["baseline_level_mean"])
                                                for n in LAYER_NAMES
                                                if "baseline_level_mean" in record["tif"][n])), flush=True)
        if comm_record is not None:
            print("  [comm 每窗消息数] 训练中均值 %.4g、单窗 p95 %.4g / 预算 %.4g | 违约窗 %.1f%%（目标 <= %.0f%%）"
                  " | 验证集均值 %.4g、单窗 p95 %.4g | mu %.4f -> %.4f" % (
                      comm_record["messages_est_mean"] or 0.0, comm_record["messages_window_p95"] or 0.0,
                      comm_record["budget"], 100.0 * (comm_record["violation_mean"] or 0.0), 100.0 * comm_record["alpha"],
                      comm_record.get("val_messages") or 0.0, comm_record.get("val_messages_p95") or 0.0,
                      comm_record["mu_start"], comm_record["mu_end"]), flush=True)
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
    subthreshold_ablation = getattr(args, "subthreshold_ablation", "none")
    if subthreshold_ablation != "none" and not net.has_subthreshold_residual:
        raise ValueError("当前 checkpoint 没有亚阈值残差机制，不能执行该消融")
    net.set_subthreshold_ablation(subthreshold_ablation)
    memory_ablation = getattr(args, "memory_ablation", "none")
    off_layers = getattr(args, "memory_off_layers", None)
    if off_layers and memory_ablation != "none":
        raise ValueError("不可同时指定全部关闭与逐层关闭记忆")
    if memory_ablation == "off":
        net.set_memory_read_enabled(False)
    elif off_layers:
        net.set_memory_read_enabled(False, off_layers)
        off_layers = [name for name in LAYER_NAMES if name in off_layers]
        memory_ablation = "selected"
    suffix = "" if args.readout_ablation == "none" else ("_" + args.readout_ablation)
    if memory_ablation == "selected":
        suffix += "_memory_off_" + "_".join(off_layers)
    elif memory_ablation != "none":
        suffix += "_memory_" + memory_ablation
    if subthreshold_ablation != "none":
        suffix += "_subthreshold_" + subthreshold_ablation
    reset_every, reset_offset = check_periodic_reset(getattr(args, "reset_every", 0),
                                                     getattr(args, "reset_offset", 0))
    if reset_every:
        if train_cfg["state_mode"] != "carry":
            raise ValueError("周期清零只对 carry 训练的模型有意义")
        suffix += "_reset_every%d" % reset_every + ("_offset%d" % reset_offset if reset_offset else "")
    noise_load = float(getattr(args, "eval_noise_load", 0.0) or 0.0)
    if noise_load < 0:
        raise ValueError("--eval-noise-load 不能为负")
    if noise_load > 0:
        train_cfg["eval_noise_load"] = noise_load
        train_cfg["eval_noise_seed"] = int(getattr(args, "eval_noise_seed", 0))
        suffix += "_noiseL%g_seed%d" % (noise_load, train_cfg["eval_noise_seed"])
    out_path = Path(args.checkpoint).parent / ("eval_%s_%s%s.json" % (
        args.split, Path(args.checkpoint).stem, suffix))
    if out_path.exists() and not args.overwrite:
        raise RuntimeError("结果已存在: %s（使用 --overwrite 覆盖）" % out_path)
    results = {"checkpoint": args.checkpoint, "epoch": ckpt["epoch"], "parameters": count_parameters(net),
               "threshold": float(train_cfg["threshold"]), "tbptt_k": int(train_cfg["tbptt_k"]),
               "trained_state_mode": train_cfg["state_mode"], "neuron": train_cfg["neuron"],
               "readout_ablation": args.readout_ablation, "dual_memory": train_cfg.get("dual_memory"),
               "eval_noise_load": noise_load, "eval_noise_seed": int(train_cfg.get("eval_noise_seed", 0)),
               "subthreshold_residual": train_cfg.get("subthreshold_residual"),
               "tif": net.tif_config, "tif_parameters": tif_parameter_statistics(net),
               "subthreshold_ablation": subthreshold_ablation,
               "subthreshold_parameters": subthreshold_parameter_statistics(net),
               "memory_ablation": memory_ablation,
               "memory_off_layers": list(net.dual_memory_layers) if memory_ablation == "off" else (off_layers or []),
               "reset_every": reset_every, "reset_offset": reset_offset}
    # 周期清零时只跑 carry：M=1 即 reset_each_window，常规评估里已有
    for mode in (("carry",) if reset_every else ("carry", "reset_each_window")):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        dump = None
        if args.dump_dir and mode == train_cfg["state_mode"]:
            dump = args.dump_dir
            os.makedirs(dump, exist_ok=True)
        r = evaluate_split(net, train_cfg, q99, device, args.split, mode, full=True, dump_dir=dump,
                           reset_every=reset_every, reset_offset=reset_offset)
        r["peak_memory_gib"] = peak_memory_gib(device)
        r["tau"] = tau_statistics(net)
        r["warnings"] = health_warnings(health_layer_stats(r["layers"], r.get("tif")), r["tau"], {})
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
