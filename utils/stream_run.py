"""流式 SNN 各版本训练入口共用的运行工具（从 train_stream_v1.py 原样移出，V1 行为不变）。

seed_everything   固定随机种子与确定性算法
train_file_names  训练集 NPZ 文件名
file_hashes       源码文件的 sha256（写进 run_config.json）
write_json        UTF-8、缩进格式写 JSON
peak_memory_gib   峰值显存
LayerMonitor      每层发放与膜电位统计
tau_statistics    每个 LIF 层的时间常数统计
V2 起的训练脚本从这里导入，不再依赖 V1 的训练脚本；train_stream_v1.py 仍以原名导出这些函数。
"""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from model.evspsegnet_stream import LAYER_NAMES
from model.lif2d_stream import StreamingLIF2d

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def train_file_names(cfg):
    """返回训练集全部 NPZ 文件名（排序）。"""
    directory = os.path.join(cfg["root"], "train")
    return sorted(n for n in os.listdir(directory) if n.endswith(".npz"))


def file_hashes(files):
    """记录本次运行所用源码（相对仓库根目录的路径）的 sha256，写进 run_config.json，方便事后核对代码版本。"""
    return {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
            for name in files if (REPO_ROOT / name).exists()}


def write_json(path, payload):
    """以 UTF-8、缩进格式写 JSON。"""
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def peak_memory_gib(device):
    """返回当前设备的峰值显存（GiB）；CPU 运行时返回 None。"""
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3)


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
