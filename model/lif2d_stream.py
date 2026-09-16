"""流式 SNN V1 的神经元与增益模块（纯 PyTorch，兼容 torch 1.9）。

核心约定（v1-3 锁定）：
    U_pre = beta * U_prev + I           （无上一状态时 U_pre = I）
    S     = H(U_pre - v_th)             （前向阶跃，反向代理梯度 1/(1+|u|)^2）
    U     = U_pre - v_th * S.detach()   （减阈值软复位，复位分支不传梯度）
    tau   = tau_min + (tau_max - tau_min) * sigmoid(a)，beta = exp(-dt / tau)
状态由调用者显式传入和返回，模块内部不保存膜电位（不会进入 checkpoint）。
每个 50 ms 窗口只调用一次神经元：5 个时间 bin 是输入通道，不是 SNN 时间步。
"""
import math

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    """脉冲发放函数：前向为阶跃 H(v)（v>=0 发放），反向用代理梯度 1/(1+|v|)^2。

    输入 v = U_pre - v_th。输出与 v 同形状、同 dtype 的 0/1 张量。
    代理梯度形式与仓库之前 lif_rate.py 中的实现一致，便于和早期实验对照。
    """

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v >= 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        return grad_output / (1.0 + v.abs()).pow(2)


class StreamingLIF2d(nn.Module):
    """逐通道、逐像素独立的 LIF 神经元（膜电位更新不交换邻域状态）。

    参数:
        channels     通道数，每个通道一个可学习时间常数
        dt_ms        时间步长（窗长，V1 为 50 ms）
        tau_init_ms  初始时间常数（V1 为 200 ms -> beta0 约 0.779）
        tau_min_ms / tau_max_ms  时间常数范围，通过 sigmoid 平滑约束，没有硬裁剪
        v_threshold  发放阈值
    forward(current, state) -> (spikes, new_state, u_pre)
        current   [B,C,H,W] 输入电流
        state     上一窗口的膜电位 [B,C,H,W]，None 表示没有历史（序列开头或 reset 模式）
        spikes    [B,C,H,W] 0/1 脉冲
        new_state [B,C,H,W] 软复位后的膜电位，交给下一个窗口
        u_pre     [B,C,H,W] 复位前膜电位，最后一层用它做逐事件读出
    """

    def __init__(self, channels, dt_ms, tau_init_ms, tau_min_ms, tau_max_ms, v_threshold):
        super(StreamingLIF2d, self).__init__()
        if not tau_min_ms < tau_init_ms < tau_max_ms:
            raise ValueError("需要 tau_min < tau_init < tau_max")
        frac = (tau_init_ms - tau_min_ms) / (tau_max_ms - tau_min_ms)
        a0 = math.log(frac / (1.0 - frac))                 # logit，V1 默认约 -2.485
        self.a = nn.Parameter(torch.full((channels,), a0, dtype=torch.float32))
        self.dt_ms = float(dt_ms)
        self.tau_min_ms = float(tau_min_ms)
        self.tau_max_ms = float(tau_max_ms)
        self.v_threshold = float(v_threshold)

    def tau(self):
        """返回每个通道的时间常数（毫秒），形状 [C]。"""
        return self.tau_min_ms + (self.tau_max_ms - self.tau_min_ms) * torch.sigmoid(self.a)

    def beta(self):
        """返回每个通道的衰减系数 beta = exp(-dt/tau)，形状 [C]。"""
        return torch.exp(-self.dt_ms / self.tau())

    def forward(self, current, state):
        if state is None:
            u_pre = current
        else:
            beta = self.beta().to(current.dtype).view(1, -1, 1, 1)
            u_pre = beta * state + current
        spikes = SurrogateSpike.apply(u_pre - self.v_threshold)
        new_state = u_pre - self.v_threshold * spikes.detach()
        return spikes, new_state, u_pre


class ReLUNeuron(nn.Module):
    """ReLU 对照神经元（无状态、非脉冲），用于"脉冲网络 vs 非循环 ANN"对照和过拟合诊断。

    接口与 StreamingLIF2d 相同：forward(current, state) -> (relu(current), None, current)。
    state 参数被忽略；读出头读取的是 ReLU 之前的电流（对应 LIF 的 u_pre）。
    """

    def __init__(self, *args, **kwargs):
        super(ReLUNeuron, self).__init__()

    def forward(self, current, state):
        return torch.relu(current), None, current


class ChannelGain(nn.Module):
    """逐通道正值增益 g = exp(log_gain)，只缩放不平移（零输入仍然输出零）。

    初始 log_gain = 0（增益 1）；训练开始前由 calibrate_gains 一次性写入校准值，之后继续可学习。
    """

    def __init__(self, channels):
        super(ChannelGain, self).__init__()
        self.log_gain = nn.Parameter(torch.zeros(channels, dtype=torch.float32))

    def gain(self):
        """返回当前增益 [C]。"""
        return torch.exp(self.log_gain)

    def set_gain(self, gains):
        """写入校准得到的增益（正数），内部存为 log 值。"""
        gains = torch.as_tensor(gains, dtype=self.log_gain.dtype, device=self.log_gain.device)
        if gains.shape != self.log_gain.shape or bool((gains <= 0).any()):
            raise ValueError("增益形状不匹配或含非正值")
        with torch.no_grad():
            self.log_gain.copy_(torch.log(gains))

    def forward(self, x):
        return x * self.gain().to(x.dtype).view(1, -1, 1, 1)


def detach_states(states):
    """截断梯度但保留膜电位数值（TBPTT 片段边界使用）。None 元素保持 None。"""
    if states is None:
        return None
    return [None if s is None else s.detach() for s in states]
