"""通信预算训练：每窗消息数的峰值约束（损失函数层；Stream-SNN-Innovations.md 12.13~12.21、12.35、12.36）。

与网络、神经元、能耗口径都解耦：
    只读取 forward_chunk(collect=True) 的各层记忆 M 与片段开始前的状态，前向推理完全不变；
    只通过网络的两个接口得知结构：blocks()（各层神经元，取 theta）与 message_layers()（哪些层的输出被其他层接收）；
    各层神经元数取自实际张量的形状，不写死分辨率、通道数或解码器写法；
    不含任何能耗（SOP/MAC）计算——能耗只在评估环节（model/evspsegnet_stream.py 的 estimate_operations 等）。
YAML 的 TRAIN 段没有 comm_budget_messages（或为 null）时 CommBudget.from_config 返回 None，训练与原来逐位相同。

问题：消息"浪费"只相对通信代价才有定义；训练只看精度时，任何发送端规则都会被学习抵消（12.12~12.13）。
所以把通信写进目标，用约束形式，拉格朗日乘子 mu 按对偶上升更新（不手调惩罚系数）。
约束量：每个时间窗、每条流实际发出的消息数 m_{t,b} = sum_{l in message_layers} sum_{神经元} |Δq|。
    q = round(M/theta)，一个单位电平变化记一条消息，与 LIF 的一个脉冲同口径（多级消息按单位计，比按包计更保守）。
约束：m_{t,b} <= B 的违约比例不超过 alpha（机会约束）。
    依据：流式系统要在每个 50 ms 窗的截止时间内处理完，片上网络每个时间步能传的消息数有上限，决定实时性的是
    单窗最坏情况，不是平均值。用分位数而不是硬上限：目标出现等本该多发消息的窗不会让 mu 只涨不降。
    片段损失 += mu * mean_{有效窗} relu(m/B - 1)（只罚超出预算的窗）；
    更新后 mu <- max(0, mu + dual_lr * (违约比例 - alpha))。
    平均预算（E[...] <= B）已删除（12.32）：它的工作点由对偶步长和批次构成决定而不是由 B 决定，且不约束单窗尾部。
预算 B：同结构 LIF 在同一数据集 val 上每窗实际发出的平均消息数（评估结果的 messages_mean）。网络或数据集改变时，
    用同结构 LIF 重新评估得到新的 B；本模块的计算本身不随网络改变而失效。
计数与评估一致：片段第一窗与上一片段末窗比较；某条流开始新录像（keep = 0）的那一窗与 0 比较（评估中每条录像
    从零状态开始）；多流训练里没有录像的空窗（流错开起点、流尾补齐，零输入）不计入——评估中不存在这些窗。
梯度：前向用精确的 |Δq|，反向用 |Δx|（x = M/theta）。依据：电平网格位置随机（均匀抖动）时，x 从 x_prev 变到 x
    平均越过的电平数恰好是 |x - x_prev|，所以 |Δx| 是消息数的无偏光滑代理；它在没越界的抖动处也有梯度。
"""
import math

import numpy as np
import torch

from model.evspsegnet_stream import LAYER_NAMES

OBSOLETE_KEYS = {"comm_budget_sop": "已改为 comm_budget_messages（约束每窗消息数，能耗只在评估环节计算，12.35）",
                 "comm_accounting": "已删除（约束量与能耗口径无关，12.35）"}


def _layer_delta(block, memory, prev, reset):
    """一层在每个 (窗, 流) 上的 |Δq| 总数 [T,B]（前向精确，反向按 |Δx|）。"""
    dtype = memory.dtype
    theta = block.neuron.theta().to(dtype).view(1, 1, -1, 1, 1)
    x = memory / theta
    if isinstance(prev, tuple):
        prev = prev[0]
    first = torch.zeros_like(x[0]) if prev is None else (prev.to(dtype) / theta[0]).detach()
    x_prev = torch.cat([first.unsqueeze(0), x[:-1]], 0) * reset
    dq = (torch.round(x.detach()) - torch.round(x_prev.detach())).abs()
    dx = (x - x_prev).abs()
    return (dq + (dx - dx.detach())).flatten(2).sum(2)


def _reset_mask(memories, keep):
    steps, streams = int(memories[0].shape[0]), int(memories[0].shape[1])
    dtype, device = memories[0].dtype, memories[0].device
    reset = torch.ones(steps, streams, dtype=dtype, device=device) if keep is None else keep.to(dtype)
    return reset.view(steps, streams, 1, 1, 1)


def window_rates(net, memories, prev_states, keep):
    """各层每神经元在每个 (窗, 流) 上的 |Δq|（前向精确，反向按 |Δx|），返回 7 个 [T,B] 张量（诊断与测试用）。

    memories     forward_chunk(collect=True) 的 u_pre（TIF 即各层记忆 M），每层 [T,B,C,H,W]
    prev_states  片段开始前的 7 个状态（TIF 为 M，自校准层为 (M, b, w)），None 表示从零开始
    keep         [T,B] 或 None；keep[t] = 0 表示该流在第 t 窗开始新录像（与 0 比较）
    """
    reset = _reset_mask(memories, keep)
    return [_layer_delta(block, memory, None if prev_states is None else prev_states[i], reset)
            / float(memory[0, 0].numel())
            for i, (block, memory) in enumerate(zip(net.blocks(), memories))]


def window_messages(net, memories, prev_states, keep):
    """每个 (窗, 流) 实际发出的消息数 [T,B]：message_layers() 中各层 |Δq| 之和（只计算这些层）。"""
    sending = set(net.message_layers())
    reset = _reset_mask(memories, keep)
    total = None
    for i, (name, block, memory) in enumerate(zip(LAYER_NAMES, net.blocks(), memories)):
        if name in sending:
            term = _layer_delta(block, memory, None if prev_states is None else prev_states[i], reset)
            total = term if total is None else total + term
    return total


class CommBudget(object):
    """每窗消息数峰值约束的状态与计算：乘子 mu 与本轮统计。"""

    def __init__(self, budget, dual_lr, alpha=0.05):
        if not (math.isfinite(budget) and budget > 0 and math.isfinite(dual_lr) and dual_lr > 0):
            raise ValueError("comm_budget_messages 与 comm_dual_lr 必须是有限正数")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("comm_peak_alpha 必须在 (0, 1) 内")
        self.budget, self.dual_lr, self.alpha = float(budget), float(dual_lr), float(alpha)
        self.mu = 0.0
        self.start_epoch()

    @classmethod
    def from_config(cls, cfg, net):
        """YAML 没有 comm_budget_messages（或为 null）时返回 None（关闭）。"""
        for key, why in OBSOLETE_KEYS.items():
            if key in cfg:
                raise ValueError("%s %s" % (key, why))
        budget = cfg.get("comm_budget_messages")
        if budget is None:
            return None
        if not (net.is_tif and net.tif_config["quantize"]):
            raise ValueError("通信预算训练只用于 neuron: tif 且 tif.quantize: true（消息是 Δq）")
        if cfg.get("trainer", "sequence") != "streams":
            raise ValueError("通信预算训练只实现了 trainer: streams")
        if cfg.get("comm_budget_mode", "peak") != "peak":
            raise ValueError("comm_budget_mode 只能是 peak；平均预算（mean）已删除（Stream-SNN-Innovations.md 12.32）")
        if not net.message_layers():
            raise ValueError("网络没有发出消息的层（message_layers 为空）")
        return cls(float(budget), float(cfg.get("comm_dual_lr", 0.01)), alpha=float(cfg.get("comm_peak_alpha", 0.05)))

    def start_epoch(self):
        self._block, self._msg, self._mu, self._over, self._mu_start = [], [], [], 0, self.mu
        self._viol, self._windows = [], []

    def chunk_loss(self, net, memories, prev_states, keep, block_size, valid=None):
        """一个片段的惩罚项（mu = 0 或没有有效窗时返回 None，不参与反传）；同时记下本片段的统计。"""
        per_window = window_messages(net, memories, prev_states, keep)                   # [T,B]
        weight = torch.ones_like(per_window) if valid is None else valid.to(per_window.dtype)
        count = float(weight.sum())
        if count <= 0:
            return None
        values = per_window.detach()[weight > 0]
        self._windows.extend(values.cpu().tolist())
        self._block.append((float(values.sum()) / count, count, float((values > self.budget).sum())))
        if self.mu <= 0:
            return None
        excess = torch.relu(per_window / self.budget - 1.0) * weight
        return self.mu * excess.sum() / count / float(block_size)

    def end_update(self):
        """一次参数更新之后做对偶上升：按违约窗比例减 alpha。"""
        if not self._block:
            return
        count = sum(c for _, c, _ in self._block)
        mean_msg = sum(v * c for v, c, _ in self._block) / count
        violation = sum(n for _, _, n in self._block) / count
        self.mu = max(0.0, self.mu + self.dual_lr * (violation - self.alpha))
        self._viol.append(violation)
        self._msg.append(mean_msg)
        self._mu.append(self.mu)
        self._over += int(violation > self.alpha)
        self._block = []

    def summary(self):
        """本轮统计（写入 metrics.jsonl 的 comm 项）。over_budget_frac 为违约比例超过 alpha 的更新占比。"""
        n = len(self._msg)
        return {"mode": "peak", "unit": "messages", "budget": self.budget, "dual_lr": self.dual_lr,
                "alpha": self.alpha,
                "messages_est_mean": sum(self._msg) / n if n else None,
                "mu_start": self._mu_start, "mu_end": self.mu,
                "mu_mean": sum(self._mu) / n if n else None,
                "over_budget_frac": self._over / float(max(n, 1)),
                "violation_mean": float(np.mean(self._viol)) if self._viol else None,
                "messages_window_p95": float(np.percentile(self._windows, 95)) if self._windows else None}


def window_message_summary(net, tif_stats):
    """评估：每窗实际发出的消息数的分布（与训练约束同口径）；没有逐窗统计时返回 None。

    每层逐窗消息数 = per_window_rate（每神经元 |Δq|）* neurons（本层神经元数，取自实际张量）。
    """
    sending = net.message_layers()
    if not tif_stats or any("per_window_rate" not in tif_stats.get(n, {}) for n in sending):
        return None
    values = sum(np.asarray(tif_stats[n]["per_window_rate"], dtype=np.float64) * float(tif_stats[n]["neurons"])
                 for n in sending)
    values = np.atleast_1d(values)
    if values.size == 0:
        return None
    return {"windows": int(values.size), "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "max": float(values.max())}


def mean_messages(net, layer_stats, tif_stats=None):
    """评估：每窗实际发出的消息数的平均。

    TIF（量化）：发送层 message_abs_rate * neurons 之和；LIF：发送层 firing_rate * neurons 之和（一个脉冲一条消息）。
    同结构 LIF 的这个值就是通信预算 B。其他神经元（graded / relu / 不量化 TIF，层间是实数）返回 None。
    """
    sending = net.message_layers()
    if net.is_tif:
        if not net.tif_config["quantize"] or not tif_stats:
            return None
        return float(sum(tif_stats[n]["message_abs_rate"] * tif_stats[n]["neurons"] for n in sending))
    if not net.spiking or not layer_stats:
        return None
    return float(sum(layer_stats[n]["firing_rate"] * layer_stats[n]["neurons"] for n in sending))
