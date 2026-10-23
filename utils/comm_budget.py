"""通信预算训练：每窗峰值带宽的机会约束（损失函数层；Stream-SNN-Innovations.md 12.13~12.21、12.32）。

与网络、神经元解耦：只读取 forward_chunk(collect=True) 的各层记忆 M 与片段开始前的状态，前向推理完全不变。
YAML 的 TRAIN 段没有 comm_budget_sop（或为 null）时 CommBudget.from_config 返回 None，训练与原来逐位相同。

问题：消息"浪费"只相对能耗目标才有定义；训练只看精度时，任何发送端规则都会被学习抵消（12.12~12.13）。
所以把通信写进目标，用约束形式，拉格朗日乘子 mu 按对偶上升更新（不手调惩罚系数）。
约束：每个时间窗、每条流的 SOP s_{t,b} <= B 的违约比例不超过 alpha。
    依据：流式系统要在每个 50 ms 窗的截止时间内处理完，决定实时性的是单窗最坏情况的通信量（神经形态硬件
    每个时间步的路由带宽也有上限），不是平均值。用分位数而不是硬上限：目标出现等本该多发消息的窗不会让 mu 只涨不降。
    片段损失 += mu * mean_{有效窗} relu(s/B - 1)（只罚超出预算的窗）；
    更新后 mu <- max(0, mu + dual_lr * (违约比例 - alpha))。
    平均预算（E[SOP] <= B）已删除（12.32）：它的工作点由对偶步长和批次构成决定而不是由 B 决定，且不约束单窗尾部。
SOP 口径（comm_accounting: merged，12.32）：部署时解码器一律换算成电流合并形式（merged_decoder，与原形式函数严格相同），
    所以训练中的 SOP 也按合并形式计：系数由同通道数的合并解码器网络求得，与网络本身是否以合并形式训练无关。
    SOP = base + sum_l c_l * r_l。c_l 由 estimate_operations 对各层逐一把发放率置 1 求得（SOP 对发放率是线性的），
    r_l 为第 l 层每神经元的 |Δq|，q = round(M/theta)。
计数与评估一致：片段第一窗与上一片段末窗比较；某条流开始新录像（keep = 0）的那一窗与 0 比较（评估中每条录像
    从零状态开始）；多流训练里没有录像的空窗（流错开起点、流尾补齐，零输入）不计入——评估中不存在这些窗。
梯度：前向用精确的 |Δq|，反向用 |Δx|（x = M/theta）。依据：电平网格位置随机（均匀抖动）时，x 从 x_prev 变到 x
    平均越过的电平数恰好是 |x - x_prev|，所以 |Δx| 是消息数的无偏光滑代理；它在没越界的抖动处也有梯度。
"""
import math

import numpy as np
import torch

from model.evspsegnet_stream import LAYER_NAMES, EvSpSegNetStream, estimate_operations

ACCOUNTING = ("merged",)


def sop_coefficients(net, height, width):
    """(base, [c_l])：按网络自身结构，SOP = base + sum_l c_l * 第 l 层发放率（与 estimate_operations 同口径）。"""
    zero = {name: 0.0 for name in LAYER_NAMES}
    base = estimate_operations(net, height, width, zero, 0.0)["sop"]
    coef = []
    for name in LAYER_NAMES:
        rates = dict(zero)
        rates[name] = 1.0
        coef.append(estimate_operations(net, height, width, rates, 0.0)["sop"] - base)
    return base, coef


def merged_twin(net):
    """同通道数、电流合并解码器的 TIF 网络（只用来求 SOP 系数；SOP 只取决于卷积形状与发放率）。

    已是合并形式时返回 net 本身。新建网络会消耗随机数，这里用 fork_rng 保持 CPU 随机状态不变，
    保证开启通信预算不改变后续训练的随机序列。
    """
    if net.merged_decoder:
        return net
    channels = tuple(block.out_ch for block in (net.enc1, net.enc2, net.enc3, net.enc4))
    with torch.random.fork_rng(devices=[]):
        return EvSpSegNetStream(in_channels=net.enc1.conv.in_channels, channels=channels, neuron="tif",
                                readout_hidden=net.readout[0].out_features, merged_decoder=True)


def merged_sop_coefficients(net, height, width):
    """(base, [c_l])：按电流合并解码器口径（部署口径）的 SOP 系数。"""
    return sop_coefficients(merged_twin(net), height, width)


def window_rates(net, memories, prev_states, keep):
    """各层每神经元在每个 (窗, 流) 上的 |Δq|（前向精确，反向按 |Δx|），返回 7 个 [T,B] 张量。

    memories     forward_chunk(collect=True) 的 u_pre（TIF 即各层记忆 M），每层 [T,B,C,H,W]
    prev_states  片段开始前的 7 个状态（TIF 为 M，自校准层为 (M, b, w)），None 表示从零开始
    keep         [T,B] 或 None；keep[t] = 0 表示该流在第 t 窗开始新录像（与 0 比较）
    """
    steps, streams = int(memories[0].shape[0]), int(memories[0].shape[1])
    dtype, device = memories[0].dtype, memories[0].device
    reset = torch.ones(steps, streams, dtype=dtype, device=device) if keep is None else keep.to(dtype)
    reset = reset.view(steps, streams, 1, 1, 1)
    out = []
    for index, (block, memory) in enumerate(zip(net.blocks(), memories)):
        theta = block.neuron.theta().to(dtype).view(1, 1, -1, 1, 1)
        x = memory / theta
        prev = None if prev_states is None else prev_states[index]
        if isinstance(prev, tuple):
            prev = prev[0]
        first = torch.zeros_like(x[0]) if prev is None else (prev.to(dtype) / theta[0]).detach()
        x_prev = torch.cat([first.unsqueeze(0), x[:-1]], 0) * reset
        dq = (torch.round(x.detach()) - torch.round(x_prev.detach())).abs()
        dx = (x - x_prev).abs()
        out.append((dq + (dx - dx.detach())).flatten(2).sum(2) / float(x[0, 0].numel()))
    return out


class CommBudget(object):
    """峰值带宽约束的状态与计算：乘子 mu、各层 SOP 系数（合并口径）、本轮统计。"""

    def __init__(self, budget, dual_lr, coef, base=0.0, alpha=0.05):
        if not (math.isfinite(budget) and budget > 0 and math.isfinite(dual_lr) and dual_lr > 0):
            raise ValueError("comm_budget_sop 与 comm_dual_lr 必须是有限正数")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("comm_peak_alpha 必须在 (0, 1) 内")
        self.budget, self.dual_lr = float(budget), float(dual_lr)
        self.coef, self.base = [float(c) for c in coef], float(base)
        self.alpha = float(alpha)
        self.mu = 0.0
        self.start_epoch()

    @classmethod
    def from_config(cls, cfg, net):
        """YAML 没有 comm_budget_sop（或为 null）时返回 None（关闭）。"""
        budget = cfg.get("comm_budget_sop")
        if budget is None:
            return None
        if not (net.is_tif and net.tif_config["quantize"]):
            raise ValueError("通信预算训练只用于 neuron: tif 且 tif.quantize: true（消息是 Δq）")
        if cfg.get("trainer", "sequence") != "streams":
            raise ValueError("通信预算训练只实现了 trainer: streams")
        if cfg.get("comm_budget_mode", "peak") != "peak":
            raise ValueError("comm_budget_mode 只能是 peak；平均预算（mean）已删除（Stream-SNN-Innovations.md 12.32）")
        if cfg.get("comm_accounting") not in ACCOUNTING:
            raise ValueError("必须写 comm_accounting: merged（SOP 按部署用的合并解码器计，12.32）；"
                             "按网络自身口径的旧配置已停用")
        base, coef = merged_sop_coefficients(net, int(cfg["pad_height"]), int(cfg["pad_width"]))
        return cls(float(budget), float(cfg.get("comm_dual_lr", 0.01)), coef, base,
                   alpha=float(cfg.get("comm_peak_alpha", 0.05)))

    def sop(self, rates):
        """各层发放率（张量列表、[T,B] 张量列表或 {层名: 数}）-> SOP 估计（合并口径）。"""
        if isinstance(rates, dict):
            rates = [rates[name] for name in LAYER_NAMES]
        return self.base + sum(c * r for c, r in zip(self.coef, rates))

    def start_epoch(self):
        self._block, self._sop, self._mu, self._over, self._mu_start = [], [], [], 0, self.mu
        self._viol, self._windows = [], []

    def chunk_loss(self, net, memories, prev_states, keep, block_size, valid=None):
        """一个片段的惩罚项（mu = 0 或没有有效窗时返回 None，不参与反传）；同时记下本片段的统计。"""
        per_window = self.sop(window_rates(net, memories, prev_states, keep))           # [T,B]
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
        mean_sop = sum(v * c for v, c, _ in self._block) / count
        violation = sum(n for _, _, n in self._block) / count
        self.mu = max(0.0, self.mu + self.dual_lr * (violation - self.alpha))
        self._viol.append(violation)
        self._sop.append(mean_sop)
        self._mu.append(self.mu)
        self._over += int(violation > self.alpha)
        self._block = []

    def summary(self):
        """本轮统计（写入 metrics.jsonl 的 comm 项）。over_budget_frac 为违约比例超过 alpha 的更新占比。"""
        n = len(self._sop)
        return {"mode": "peak", "accounting": "merged", "budget": self.budget, "dual_lr": self.dual_lr,
                "alpha": self.alpha,
                "sop_est_mean": sum(self._sop) / n if n else None,
                "mu_start": self._mu_start, "mu_end": self.mu,
                "mu_mean": sum(self._mu) / n if n else None,
                "over_budget_frac": self._over / float(max(n, 1)),
                "violation_mean": float(np.mean(self._viol)) if self._viol else None,
                "sop_window_p95": float(np.percentile(self._windows, 95)) if self._windows else None}


def window_sop_summary(net, tif_stats, height, width, coefficients=None):
    """评估：由各层逐窗的 |Δq| 率（stop_stats 的 per_window_rate）算单窗 SOP 的分布；没有逐窗统计时返回 None。

    coefficients 为 (base, [c_l])；None 时按网络自身结构求（评估合并形式的 checkpoint 即合并口径）。
    """
    if not tif_stats or any("per_window_rate" not in tif_stats.get(n, {}) for n in LAYER_NAMES):
        return None
    base, coef = sop_coefficients(net, height, width) if coefficients is None else coefficients
    rates = np.array([tif_stats[n]["per_window_rate"] for n in LAYER_NAMES], dtype=np.float64)   # [7, 窗数]
    if rates.shape[1] == 0:
        return None
    values = base + np.asarray(coef, dtype=np.float64).dot(rates)
    return {"windows": int(values.size), "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "max": float(values.max())}
