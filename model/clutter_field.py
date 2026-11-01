"""模块二 v2：学出来的杂波强度场（Learned Clutter Field；Stream-SNN-Innovations.md 13.69）。

问题：模块二的背景估计器（model/comod.BackgroundEstimator）是手写规则：空心环半径 10 / 4、时间常数 20 / 100 窗、截断 2、
      置信界 z = 2、热像素"高于邻域 3 倍"、11x11 子块、5x5 中位数、突变阈值 0.5、告警 2 窗，约 12 个手设常数；
      环境突变时按固定时间常数滞后（合成数据：阶跃后约 0.75 s 虚警偏高）。
原理：背景（杂波）事件是非均匀泊松过程，强度 lambda(x, 极性, t) 用最大似然估计。
做法：只读之前各窗的事件计数（因果；不读标签、不读网络输出，保持恒虚警接口的前提）：
    每个像素、每个极性 K 个漏积分器 m_k <- a_k m_k + (1 - a_k) c_t（a_k 学出来，初值对应 2 / 16 / 128 窗），
    偏差校正 m_k / (1 - a_k^n)（录像开头历史不足时不低估），多尺度汇聚（1 / 4 / 16 倍下采样后上采样，按传感器掩码
    归一化平均：补零区是"没有观测"，不是"没有事件"），得到 K x 尺度数个候选估计（都是观测率的无偏平均）和参照率；
    lambda = 候选估计的凸组合，权重 = softmax(1x1 卷积(log1p(候选 / 参照率), 历史量))：网络只学"每个位置该信哪个时间尺度、
    哪个空间尺度"（热像素信逐像素的长期平均，密集噪声信大尺度，突变时信短时间常数，目标经过时不信被它抬高的局部短时平均），
    输出总在观测率的量程内；末层零初始化（起步时各候选等权）。
    训练：泊松负对数似然 sum(lambda - n log lambda)，n = 下一窗的非目标事件计数（标签只用于训练的监督）。
    使用：不变接口（enc1 的 CFAR 归一化）与目标神经元层（PDA 的杂波密度）都读同一个 lambda，只读、不回传梯度。
手设量：漏积分器个数与时间常数初值、汇聚尺度、隐藏宽度（结构量）；空心环、截断、热像素判别、突变遗忘等规则都交给网络学。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.comod import REF_RATE

EPS = 1e-6


def event_count_maps(events, steps, batch, height, width, weight=None):
    """事件 -> 每窗每流每极性的计数图 [T, B, 2, H, W]（通道 0 = ON（p = 1），1 = OFF（p = 0）；与模块二同一约定）。
    weight [Ne]（可选）：每个事件的权重（例如 1 - label，只数非目标事件）。"""
    t = events["t"].long()
    b = events["b"].long() if "b" in events else torch.zeros_like(t)
    ch = (events["p"] < 0.5).long()
    plane = int(height) * int(width)
    flat = (((t * int(batch) + b) * 2 + ch) * plane + events["y"].long() * int(width) + events["x"].long())
    vals = torch.ones(flat.shape[0], dtype=torch.float32, device=t.device) if weight is None else weight.float()
    buf = torch.zeros(int(steps) * int(batch) * 2 * plane, dtype=torch.float32, device=t.device)
    if flat.numel():
        buf = buf.index_put((flat,), vals, accumulate=True)
    return buf.view(int(steps), int(batch), 2, int(height), int(width))


class ClutterField(nn.Module):
    """forward_chunk(counts [T,B,2,H,W], state, keep [T,B]) -> (log lambda [T,B,2,H,W], state)。

    第 t 窗的 lambda 只用第 t 窗之前的计数（先给出估计，再用本窗计数更新状态）。单位：每像素每窗的事件数。"""

    def __init__(self, height, width, sensor_hw=None, ref_rate=REF_RATE, taus=(2.0, 16.0, 128.0), scales=(1, 4, 16),
                 hidden=16):
        super(ClutterField, self).__init__()
        self.height, self.width = int(height), int(width)
        self.scales = tuple(int(s) for s in scales)
        K = len(taus)
        self.K = K
        a = torch.tensor([math.exp(-1.0 / float(t)) for t in taus])
        self.decay_logit = nn.Parameter(torch.log(a) - torch.log1p(-a))            # a_k = sigmoid(.)，学出来
        self.J = K * len(self.scales) + 1                                            # candidates per polarity (+ reference)
        in_ch = 2 * K * len(self.scales) + K
        self.mix = nn.Sequential(nn.Conv2d(in_ch, int(hidden), 1), nn.ReLU(), nn.Conv2d(int(hidden), 2 * self.J, 1))
        nn.init.zeros_(self.mix[2].weight)
        nn.init.zeros_(self.mix[2].bias)                                             # equal weights at the start
        self.register_buffer("ref", torch.tensor([float(r) for r in ref_rate]).view(1, 2, 1, 1, 1), persistent=False)
        mask = torch.zeros(1, 1, self.height, self.width)
        sh, sw = sensor_hw if sensor_hw is not None else (self.height, self.width)
        mask[..., :int(sh), :int(sw)] = 1.0
        self.register_buffer("mask", mask, persistent=False)

    def init_state(self, batch, device):
        B, K, H, W = int(batch), self.K, self.height, self.width
        return {"m": torch.zeros(B, 2, K, H, W, device=device), "w": torch.zeros(B, K, device=device)}

    @staticmethod
    def detach_state(state):
        return None if state is None else {k: v.detach() for k, v in state.items()}

    def decays(self):
        return torch.sigmoid(self.decay_logit)

    def _pool(self, x, s):
        """按传感器掩码归一化的 s 倍平均（下采样后最近邻上采样回原分辨率）。x [N, C, H, W]。"""
        if s == 1:
            return x
        m = self.mask.to(x.dtype)
        num = F.avg_pool2d(x * m, s, stride=s, ceil_mode=True)
        den = F.avg_pool2d(m.expand(x.shape[0], 1, -1, -1), s, stride=s, ceil_mode=True).clamp(min=EPS)
        r = num / den                                                                    # [N, C, h, w]
        N, C, h, w = r.shape
        up = r[:, :, :, None, :, None].expand(N, C, h, s, w, s).reshape(N, C, h * s, w * s)   # deterministic backward
        return up[..., :self.height, :self.width]

    def predict(self, state):
        """当前状态（只含之前各窗）-> log lambda [B, 2, H, W]。lambda = 候选估计（各时间、空间尺度的平均与参照率）的凸组合。"""
        m, w = state["m"], state["w"]
        B, K, Hh, Ww = m.shape[0], self.K, self.height, self.width
        avg = m / w.clamp(min=EPS)[:, None, :, None, None]                              # bias-corrected averages
        avg = avg.reshape(B, 2 * K, Hh, Ww)
        cands = torch.stack([self._pool(avg, s).view(B, 2, K, Hh, Ww) for s in self.scales], 3)  # [B, 2, K, S, H, W]
        cands = cands.reshape(B, 2, K * len(self.scales), Hh, Ww)
        ref = self.ref.to(cands.dtype)
        feats = torch.log1p(cands / ref).reshape(B, 2 * K * len(self.scales), Hh, Ww)
        hist = w[:, :, None, None].expand(B, K, Hh, Ww)                                 # how much history exists
        weights = torch.softmax(self.mix(torch.cat([feats, hist], 1)).view(B, 2, self.J, Hh, Ww), 2)
        allc = torch.cat([cands, ref.expand(B, 2, 1, Hh, Ww)], 2)                        # [B, 2, J, H, W]
        rate = (weights * allc).sum(2)
        return torch.log(rate.clamp(min=EPS))

    def forward_chunk(self, counts, state=None, keep=None):
        T, B = int(counts.shape[0]), int(counts.shape[1])
        if state is None:
            state = self.init_state(B, counts.device)
        a = self.decays()                                                                # [K]
        out = []
        for t in range(T):
            if keep is not None:
                reset = (keep[t] == 0)
                if bool(reset.any()):
                    state = {"m": torch.where(reset[:, None, None, None, None], torch.zeros_like(state["m"]), state["m"]),
                             "w": torch.where(reset[:, None], torch.zeros_like(state["w"]), state["w"])}
            out.append(self.predict(state))
            c = counts[t][:, :, None]                                                    # [B, 2, 1, H, W]
            state = {"m": a[None, None, :, None, None] * state["m"] + (1.0 - a)[None, None, :, None, None] * c,
                     "w": a[None, :] * state["w"] + (1.0 - a)[None, :]}
        return torch.stack(out), state

    def nll(self, log_rate, target_counts, valid=None):
        """泊松负对数似然（不含常数项），按传感器像素 x 有效窗平均。log_rate / target_counts [T,B,2,H,W]，valid [T,B]。"""
        m = self.mask.to(log_rate.dtype).view(1, 1, 1, self.height, self.width)
        if valid is not None:
            m = m * valid.to(log_rate.dtype)[:, :, None, None, None]
        per = (torch.exp(log_rate) - target_counts * log_rate) * m
        return per.sum() / (m.sum() * 2.0).clamp(min=1.0)
