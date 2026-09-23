"""流式 SNN V1 的神经元与增益模块（纯 PyTorch，兼容 torch 1.9）。

核心约定（v1-3 锁定）：
    U_pre = beta * U_prev + I           （无上一状态时 U_pre = I）
    S     = H(U_pre - v_th)             （前向阶跃，反向代理梯度 1/(1+|u|)^2）
    U     = U_pre - v_th * S.detach()   （减阈值软复位，复位分支不传梯度）
    tau   = tau_min + (tau_max - tau_min) * sigmoid(a)，beta = exp(-dt / tau)
状态由调用者显式传入和返回，模块内部不保存膜电位（不会进入 checkpoint）。
每个 50 ms 窗口只调用一次神经元：5 个时间 bin 是输入通道，不是 SNN 时间步。
三种神经元用于分离"脉冲"与"跨窗记忆"两个因素：StreamingLIF2d（有状态+二值）、
StreamingGraded2d（有状态+实数）、ReLUNeuron（无状态+实数）。
forward 逐窗调用（流式推理、增益校准）；forward_steps 一次处理一个片段的多个窗口（逐层时间并行训练），两者等价。
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
        u_floor / u_ceil  跨窗携带的膜电位下界 / 上界（以 v_th 为单位，None = 不设界；要求 u_floor <= 0 < u_ceil），
                     只夹携带出去的状态，见 bound_state
    forward(current, state) -> (spikes, new_state, u_pre)
        current   [B,C,H,W] 输入电流
        state     上一窗口的膜电位 [B,C,H,W]，None 表示没有历史（序列开头或 reset 模式）
        spikes    [B,C,H,W] 0/1 脉冲
        new_state [B,C,H,W] 软复位后的膜电位，交给下一个窗口
        u_pre     [B,C,H,W] 复位前膜电位，最后一层用它做逐事件读出
    """

    def __init__(self, channels, dt_ms, tau_init_ms, tau_min_ms, tau_max_ms, v_threshold,
                 u_floor=None, u_ceil=None):
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
        # 符号检查同时保证了 u_floor < u_ceil；最常见的误用是命令行漏了负号（-1 写成 1）
        if u_floor is not None and float(u_floor) > 0:
            raise ValueError("u_floor 必须 <= 0（以 v_th 为单位，收到 %s）：正的下界让携带的膜电位永远高于静息，"
                             "多半是漏了负号" % u_floor)
        if u_ceil is not None and float(u_ceil) <= 0:
            raise ValueError("u_ceil 必须 > 0（以 v_th 为单位，收到 %s）" % u_ceil)
        self.u_floor = None if u_floor is None else float(u_floor) * self.v_threshold
        self.u_ceil = None if u_ceil is None else float(u_ceil) * self.v_threshold

    def bound_state(self, state):
        """把跨窗携带的膜电位夹在 [u_floor, u_ceil]（以 v_th 为单位给定，构造时已折算）。

        为什么只夹"携带出去的"状态、不夹本窗的 U_pre：本窗是否发放由 U_pre 决定，夹它会改变发放判据；
        夹住携带值只改变"下一窗从多深处起步"，这正是要修的东西。

        动机（09-23 诊断）：LIF 对持续负电流的直流增益是 1/(1-beta)，而正向有"发放即减阈值"的复位、
        负向没有任何复位，于是 enc4/dec3 的膜电位中位数掉到 -32/-63 个阈值，发放率只剩 7%，等于死了。
        判决层的 CUSUM 神经元由推导自带静息地板 0（C = max(0, C_pre)），骨干的 LIF 缺的就是这一条。
        两个界都为 None 时本方法是恒等映射，数值与旧实现逐位相同。
        """
        if self.u_floor is None and self.u_ceil is None:
            return state
        return torch.clamp(state, min=self.u_floor, max=self.u_ceil)

    def tau(self):
        """返回每个通道的时间常数（毫秒），形状 [C]。"""
        return self.tau_min_ms + (self.tau_max_ms - self.tau_min_ms) * torch.sigmoid(self.a)

    def beta(self):
        """返回每个通道的衰减系数 beta = exp(-dt/tau)，形状 [C]。"""
        return torch.exp(-self.dt_ms / self.tau())

    def activate(self, u_pre):
        """膜电位 -> (输出, 是否发放)。LIF 输出 0/1 脉冲，复位量由"是否发放"决定。"""
        spikes = SurrogateSpike.apply(u_pre - self.v_threshold)
        return spikes, spikes.detach()

    def forward(self, current, state):
        if state is None:
            u_pre = current
        else:
            beta = self.beta().to(current.dtype).view(1, -1, 1, 1)
            u_pre = beta * state + current
        output, fired = self.activate(u_pre)
        new_state = self.bound_state(u_pre - self.v_threshold * fired)
        return output, new_state, u_pre

    def forward_steps(self, current, state, carry=True, keep_u_pre=True):
        """按时间顺序处理一个片段的 T 个窗口，与连续调用 forward T 次数学等价（逐层时间并行训练用）。

        输入: current [T,B,C,H,W] 各窗口电流；state 片段开始前的膜电位 [B,C,H,W] 或 None；
              carry=False 表示 reset_each_window，每一步都不使用上一状态。
        输出: (spikes [T,B,C,H,W], 最后一步软复位后的膜电位 [B,C,H,W], u_pre [T,B,C,H,W] 或 None)
        膜电位递推只在神经元内部逐像素进行，卷积已由调用者在 T*B 维上批量算完。
        """
        steps = int(current.shape[0])
        if steps == 0:
            raise ValueError("片段至少需要 1 个窗口")
        beta = self.beta().to(current.dtype).view(1, -1, 1, 1)
        spikes, u_pres = [], []
        for t in range(steps):
            prev = state if carry else None
            u_pre = current[t] if prev is None else beta * prev + current[t]
            output, fired = self.activate(u_pre)
            state = self.bound_state(u_pre - self.v_threshold * fired)
            spikes.append(output)
            if keep_u_pre:
                u_pres.append(u_pre)
        return torch.stack(spikes), state, (torch.stack(u_pres) if keep_u_pre else None)


class StreamingGraded2d(StreamingLIF2d):
    """有状态但不发放的对照：阈值、复位、时间常数、代理梯度都与 LIF 相同，只有发放幅值不同。

        LIF     发放时传出 1
        graded  发放时传出膜电位 U_pre 本身（本类）
    因此相同输入下膜电位轨迹与 LIF 完全一致，唯一的差别是"下游收到的是二值还是实数"。
    幅值刚好在阈值处等于 v_th = 1，与脉冲同量级，不会因为幅值变小而让深层沉默；
    高于阈值的部分正是二值化丢掉的信息。梯度同样经过代理梯度，阈值以下也有梯度，避免死区把对照拖垮。

    三个对照的关系：lif 有状态+二值；graded 有状态+实数；relu 无状态+实数。
    输出是实数，下游卷积输入不再是脉冲，理论能耗按 MAC 计（见 estimate_operations）。
    """

    def activate(self, u_pre):
        fired = SurrogateSpike.apply(u_pre - self.v_threshold)
        return fired * u_pre, fired.detach()


class ReLUNeuron(nn.Module):
    """ReLU 对照神经元（无状态、非脉冲），用于"脉冲网络 vs 非循环 ANN"对照和过拟合诊断。

    接口与 StreamingLIF2d 相同：forward(current, state) -> (relu(current), None, current)。
    state 参数被忽略；读出头读取的是 ReLU 之前的电流（对应 LIF 的 u_pre）。
    """

    def __init__(self, *args, **kwargs):
        super(ReLUNeuron, self).__init__()

    def forward(self, current, state):
        return torch.relu(current), None, current

    def forward_steps(self, current, state, carry=True, keep_u_pre=True):
        """多窗口版本 [T,B,C,H,W]；无状态，各窗口互不影响。"""
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
