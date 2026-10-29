"""模块二：不变输入接口（Noise-Invariant Interface，Stream-SNN-Innovations.md 13.53～13.54）。

问题：固定的流式网络在环境（背景噪声强度）变化时性能崩塌；数据增强只覆盖见过的范围（13.53 鲁棒性曲线）。
原理：不变检验——噪声强度是干扰参数，对它不变的检验统计量（最大不变统计量）天然恒虚警（CFAR），且对任意强度成立。
做法：在零假设形式已知的接口（enc1：输入是泊松计数的 log1p 编码）上，把背景对每个神经元输入电流的影响标准化：
    b_eff = max(b_hat, b_ref)                          b_hat 为本地背景率（每像素、每极性、每窗），b_ref 为参照率
    mu(b), var(b)                                      背景下 enc1 电流（增益后）的均值与方差，按泊松模型精确计算
                                                       （12 通道编码、整窗与时间箱的协方差、带符号权重都计入）
    I' = mu(b_ref) + sqrt(var(b_ref) / var(b_eff)) * (I - mu(b_eff))
    背景高于参照处：背景电流的均值、方差都被拉回参照工作点，与 b_hat 无关（不变性）；
    背景不高于参照处：I' = I（逐位相同）。
    归一化系数不参与反传（作为常数）；网络带着这个接口从头训练（13.46：冻结网络上外挂会失败）。
背景估计：沿用 model/comod.BackgroundEstimator（空心周边环 + 孤立点源门控 + 突变遗忘；只用过去的计数、不用标签、
          不读模型输出），在数据读入时逐窗因果计算，随事件一起送入网络（events["bg"]）。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.comod import REF_RATE, BackgroundEstimator, input_moments

TABLE_RATE_MIN, TABLE_RATE_MAX, TABLE_SIZE = 1e-6, 10.0, 1024


class MomentTable(object):
    """背景率 -> 12 个输入通道的 (均值, 方差) 与整窗 x 时间箱协方差的查表（对数网格、线性插值）。

    input_moments 的精确计算每窗要上百万次运算；矩只依赖每像素每极性的速率，所以预先在网格上算好。
    """

    def __init__(self, q99, time_bins, clip):
        self.time_bins = int(time_bins)
        self.log_min, self.log_max = math.log(TABLE_RATE_MIN), math.log(TABLE_RATE_MAX)
        rates = torch.exp(torch.linspace(self.log_min, self.log_max, TABLE_SIZE, dtype=torch.float64))
        grid = rates.view(1, 1, TABLE_SIZE, 1).expand(1, 2, TABLE_SIZE, 1).contiguous()
        mean, var, cov = input_moments(grid, q99, time_bins, clip)            # [1,12,G,1], [1,12,G,1], [1,2T,G,1]
        self.mean = mean[0, :, :, 0].float()                                    # [12, G]
        self.var = var[0, :, :, 0].float()
        self.cov = cov[0, :, :, 0].float()                                      # [2T, G]
        T = self.time_bins
        # 每个输出通道取哪个极性的速率：整窗 0/1 -> p；时间箱 2+2t+p -> p；协方差 2t+p -> p
        self.mean_pol = torch.tensor([0, 1] + [p for _ in range(T) for p in (0, 1)])
        self.cov_pol = torch.tensor([p for _ in range(T) for p in (0, 1)])
        self._device = None

    def _to(self, device):
        if self._device != device:
            self.mean, self.var, self.cov = self.mean.to(device), self.var.to(device), self.cov.to(device)
            self.mean_pol, self.cov_pol = self.mean_pol.to(device), self.cov_pol.to(device)
            self._device = device

    def lookup(self, b):
        """b [N,2,H,W]（速率） -> (mean [N,12,H,W], var [N,12,H,W], cov [N,2T,H,W])，float32。"""
        self._to(b.device)
        pos = (torch.log(b.float().clamp(TABLE_RATE_MIN, TABLE_RATE_MAX)) - self.log_min) / (
            self.log_max - self.log_min) * (TABLE_SIZE - 1)
        lo = pos.floor().clamp(0, TABLE_SIZE - 2)
        frac = pos - lo
        lo = lo.long()

        def gather(table, pol):
            idx = lo[:, pol]                                                   # [N,K,H,W]
            fr = frac[:, pol]
            k = torch.arange(table.shape[0], device=b.device).view(1, -1, 1, 1)
            a = table[k, idx]
            c = table[k, idx + 1]
            return a + (c - a) * fr

        return gather(self.mean, self.mean_pol), gather(self.var, self.mean_pol).clamp(min=0.0), \
            gather(self.cov, self.cov_pol)


class InvariantInterface(nn.Module):
    """enc1 的不变输入接口：normalize(current, bg, block)。q99 存为缓冲区（随 checkpoint 保存）。"""

    def __init__(self, time_bins, clip, ref_rate=REF_RATE, channels=12):
        super(InvariantInterface, self).__init__()
        self.time_bins, self.clip = int(time_bins), float(clip)
        self.ref_rate = tuple(float(v) for v in ref_rate)
        self.register_buffer("q99", torch.zeros(channels, dtype=torch.float32))
        self._table = None
        self._table_key = None
        self._ref = None

    def set_q99(self, q99):
        self.q99.copy_(torch.as_tensor(np.asarray(q99, dtype=np.float32)))
        self._table, self._table_key, self._ref = None, None, None

    def table(self):
        key = tuple(float(v) for v in self.q99.cpu())
        if not any(key):
            raise RuntimeError("不变输入接口的 q99 尚未设置（训练时调用 set_q99；评估时由 checkpoint 恢复）")
        if self._table is None or self._table_key != key:
            self._table, self._table_key = MomentTable(np.asarray(key), self.time_bins, self.clip), key
        return self._table

    @staticmethod
    def kernels(block):
        W = block.conv.weight.detach()
        g = block.gain.gain().detach().view(1, -1, 1, 1)
        return W, g

    def current_moments(self, b, block):
        """背景率图 b [N,2,H,W] -> enc1 电流（增益后）的背景均值与方差 [N,C,H',W']（不带梯度）。"""
        W, g = self.kernels(block)
        W = W.to(b.dtype if b.is_floating_point() else torch.float32)
        m, v, c = self.table().lookup(b)
        T = self.time_bins
        cov_k = torch.stack([2.0 * W[:, p] * W[:, 2 + 2 * t + p] for t in range(T) for p in (0, 1)], 1)
        stride, pad = block.conv.stride, block.conv.padding
        mu = F.conv2d(m, W, stride=stride, padding=pad) * g
        var = (F.conv2d(v, W * W, stride=stride, padding=pad) + F.conv2d(c, cov_k, stride=stride, padding=pad)) * g * g
        return mu, var.clamp(min=0.0)

    def normalize(self, current, bg, block):
        """current [N,C,H,W]（增益后的 enc1 电流），bg [N,2,H,W]（背景率；None 时原样返回）。"""
        if bg is None:
            return current
        with torch.no_grad():
            bg = bg.to(torch.float32)
            ref = torch.tensor(self.ref_rate, dtype=torch.float32, device=bg.device).view(1, 2, 1, 1)
            b_ref = ref.expand(1, 2, bg.shape[-2], bg.shape[-1])
            b_eff = torch.maximum(bg, b_ref.expand_as(bg))
            mu_ref, var_ref = self.current_moments(b_ref.contiguous(), block)
            mu_eff, var_eff = self.current_moments(b_eff, block)
            # 背景高于参照的输入像素，膨胀一个卷积半径（3x3 核）：只有这些位置的电流被校正，其余位置逐位不变
            above = F.max_pool2d((bg > b_ref).any(1, keepdim=True).float(), 3, stride=1, padding=1) > 0.5
            above = above_map(above, mu_eff)
            scale = torch.where(var_ref > 1e-12, torch.sqrt(var_ref / var_eff.clamp(min=1e-12)),
                                torch.ones_like(var_eff)).clamp(max=1.0)
            scale = torch.where(above, scale, torch.ones_like(scale))
            shift_eff = torch.where(above, mu_eff, torch.zeros_like(mu_eff))
            shift_ref = torch.where(above, mu_ref.expand_as(mu_eff), torch.zeros_like(mu_eff))
        return shift_ref.to(current.dtype) + scale.to(current.dtype) * (current - shift_eff.to(current.dtype))


def above_map(above, like):
    """输入分辨率上的"背景高于参照"掩码 -> 电流分辨率（enc1 stride 1，形状相同；保险起见按需插值）。"""
    if above.shape[-2:] != like.shape[-2:]:
        above = F.interpolate(above.float(), size=like.shape[-2:], mode="nearest") > 0.5
    return above.expand_as(like)


class BackgroundTracker(object):
    """一段录像的逐窗因果背景估计：background(start, end) -> [T,2,H,W]（第 k 窗只用第 k 窗之前的计数）。

    窗口必须按顺序请求（流式训练与评估都是如此）；请求的起点早于已处理的位置时从头重算。
    计数按画布尺寸（pad_height x pad_width）统计：通道 0 = ON（p=1），1 = OFF（p=0），与 comod.counts_from_input 一致。
    画布比传感器大（EV-UAV 264x352 对 260x346，只补下侧和右侧）。默认把补零区当作"没有事件"（E3b 定型的行为，
    下边缘背景被低估约 20%～27%）；cfg["invariant_sensor_mask"] = True 时按传感器掩码只在有效像素上求平均。
    10-29 用户定：默认不开掩码，模块二以 E3b 为准；掩码修正记为细节问题（CLAUDE.md 第 1.5 节 D9），后期再完善。
    """

    def __init__(self, seq, cfg, device):
        self.seq, self.device = seq, device
        self.height, self.width = int(cfg["pad_height"]), int(cfg["pad_width"])
        eps_b = 0.5 * float(cfg.get("invariant_ref_rate", REF_RATE)[0])
        mask = None
        if cfg.get("invariant_sensor_mask", False):    # true = 只在传感器像素上估计背景（细节问题 D9，默认关）
            mask = torch.zeros(1, 1, self.height, self.width, dtype=torch.float64, device=device)
            mask[..., :int(cfg["height"]), :int(cfg["width_px"])] = 1.0
        self.estimator = BackgroundEstimator(eps_b=eps_b, mask=mask)
        self.next_k = 0

    def _counts(self, k):
        idx = self.seq.window_index(k)
        plane = self.height * self.width
        key = np.where(self.seq.p[idx] == 1, 0, 1).astype(np.int64) * plane + \
            self.seq.y[idx].astype(np.int64) * self.width + self.seq.x[idx].astype(np.int64)
        counts = np.bincount(key, minlength=2 * plane).astype(np.float32).reshape(1, 2, self.height, self.width)
        return torch.from_numpy(counts).to(self.device)

    def background(self, start, end):
        if start < self.next_k:
            self.estimator.reset()
            self.next_k = 0
        while self.next_k < start:
            self.estimator.update(self._counts(self.next_k).double())
            self.next_k += 1
        out = []
        for k in range(start, end):
            b = self.estimator.estimate()
            out.append(torch.zeros(1, 2, self.height, self.width, device=self.device) if b is None else b.float())
            self.estimator.update(self._counts(k).double())
            self.next_k = k + 1
        return torch.cat(out)                                                  # [T,2,H,W]
