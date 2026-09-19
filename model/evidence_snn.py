"""流式 SNN V2（方案一）的网络：V1 电流合并解码器的脉冲 U-Net 骨干 + 两个逐像素输出头。

    特征 [B,C,H,W]（dataset/stream_features.EvidenceFrontEnd）
      -> 脉冲 U-Net 骨干（EvSpSegNetStream，use_readout=False）-> u7 [B,C1,H,W] 最后一层复位前膜电位
      -> 逐像素头（1x1 卷积，可带一层隐藏层）-> 两个输出
         mark   本窗该像素上的事件是目标的 logit（零延迟的逐事件判断，与 V1 的读出对应）
         log_g  本窗的目标强度场：该像素附近每窗目标事件数的期望（log）。CUSUM 把它按各速度假设平移一步，
                作为下一窗"可预测"的目标强度（见 model/evidence_neuron.py）
log_g 用平滑上界 m - softplus(m - raw) 限制在 log_g_max 以下（exp 不会溢出，梯度处处非零）。
两个头的初值：最后一层权重缩小 10 倍，偏置取先验（目标事件占比、每像素每窗目标事件数），
这样训练开始时输出接近先验，Poisson 损失里的 sum(exp(log_g)) 不会一开始就很大。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.evspsegnet_stream import EvSpSegNetStream


class EvidenceSNN(nn.Module):
    """forward(x, states) -> (mark [B,1,H,W], log_g [B,1,H,W], new_states, info)
    forward_chunk(x [T,B,C,H,W], states) -> (mark [T,B,1,H,W], log_g [T,B,1,H,W], new_states, info)
    states / state_mode / collect 的含义与 V1 骨干相同。
    """

    def __init__(self, in_channels, channels=(12, 24, 48, 48), neuron="lif", norm="none", state_mode="carry",
                 dt_ms=50.0, tau_init_ms=200.0, tau_min_ms=50.0, tau_max_ms=2000.0, v_threshold=1.0,
                 merged_decoder=True, head_hidden=16, mark_prior=0.03, intensity_prior=2e-4, log_g_max=8.0):
        super(EvidenceSNN, self).__init__()
        self.backbone = EvSpSegNetStream(
            in_channels=in_channels, channels=tuple(channels), neuron=neuron, norm=norm, state_mode=state_mode,
            dt_ms=dt_ms, tau_init_ms=tau_init_ms, tau_min_ms=tau_min_ms, tau_max_ms=tau_max_ms,
            v_threshold=v_threshold, merged_decoder=merged_decoder, use_readout=False)
        c1 = int(channels[0])
        if int(head_hidden) > 0:
            last = nn.Conv2d(int(head_hidden), 2, 1, bias=True)
            self.head = nn.Sequential(nn.Conv2d(c1, int(head_hidden), 1, bias=True), nn.ReLU(), last)
        else:
            last = nn.Conv2d(c1, 2, 1, bias=True)
            self.head = last
        if not (0.0 < float(mark_prior) < 1.0 and float(intensity_prior) > 0.0):
            raise ValueError("mark_prior 必须在 (0,1) 内，intensity_prior 必须为正")
        with torch.no_grad():
            last.weight.mul_(0.1)
            last.bias.copy_(torch.tensor([math.log(mark_prior / (1.0 - mark_prior)), math.log(intensity_prior)]))
        self.log_g_max = float(log_g_max)

    @property
    def v_threshold(self):
        """骨干的发放阈值（监控用）。"""
        return self.backbone.v_threshold

    def blocks(self):
        """骨干的 7 个脉冲卷积单元（增益校准、监控用）。"""
        return self.backbone.blocks()

    def _heads(self, u7):
        """u7 [N,C1,H,W] -> (mark [N,1,H,W], log_g [N,1,H,W])。"""
        out = self.head(u7)
        mark = out[:, 0:1]
        log_g = self.log_g_max - F.softplus(self.log_g_max - out[:, 1:2])
        return mark, log_g

    def forward(self, x, states, state_mode=None, collect=False):
        u7, new_states, info = self.backbone.forward_dense(x, states, state_mode, collect)
        mark, log_g = self._heads(u7)
        return mark, log_g, new_states, info

    def forward_chunk(self, x, states, state_mode=None, collect=False):
        u7, new_states, info = self.backbone.forward_dense_chunk(x, states, state_mode, collect)
        steps, batch = int(u7.shape[0]), int(u7.shape[1])
        mark, log_g = self._heads(u7.reshape((steps * batch,) + tuple(u7.shape[2:])))
        shape = (steps, batch) + tuple(mark.shape[1:])
        return mark.view(shape), log_g.view(shape), new_states, info
