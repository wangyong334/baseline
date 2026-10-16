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
        new_state = u_pre - self.v_threshold * fired
        return output, new_state, u_pre

    def forward_steps(self, current, state, carry=True, keep_u_pre=True, keep=None):
        """按时间顺序处理一个片段的 T 个窗口，与连续调用 forward T 次数学等价（逐层时间并行训练用）。

        输入: current [T,B,C,H,W] 各窗口电流；state 片段开始前的膜电位 [B,C,H,W] 或 None；
              carry=False 表示 reset_each_window，每一步都不使用上一状态；
              keep [T,B]（可选）为 0 的位置表示该流在该窗开始一段新录像：先把旧状态乘 0 再递推。
              零状态与"没有历史"（None）数值完全相同，梯度也在该处截断（多流训练用）。
        输出: (spikes [T,B,C,H,W], 最后一步软复位后的膜电位 [B,C,H,W], u_pre [T,B,C,H,W] 或 None)
        膜电位递推只在神经元内部逐像素进行，卷积已由调用者在 T*B 维上批量算完。
        """
        steps = int(current.shape[0])
        if steps == 0:
            raise ValueError("片段至少需要 1 个窗口")
        beta = self.beta().to(current.dtype).view(1, -1, 1, 1)
        spikes, u_pres = [], []
        for t in range(steps):
            prev = mask_state(state if carry else None, keep, t)
            u_pre = current[t] if prev is None else beta * prev + current[t]
            output, fired = self.activate(u_pre)
            state = u_pre - self.v_threshold * fired
            spikes.append(output)
            if keep_u_pre:
                u_pres.append(u_pre)
        return torch.stack(spikes), state, (torch.stack(u_pres) if keep_u_pre else None)


class StreamingDualMemoryLIF2d(StreamingLIF2d):
    """编码层双仓实验：输入写入有界 M，旧 M 以受限电流参与原 V1 软复位。

    Q=tanh(I/theta); M=alpha*M_prev+(1-alpha)*Q
    U_pre=beta*U_prev+I+(1-beta)*kappa*theta*M_prev
    |kappa|<kappa_max<1。状态为 (U, M)，不存储在模块内。
    memory_read_enabled=False 仅用于同权重读取消融；不清除 U 的正常历史。
    零初始化 kappa 不消耗随机数，因此同种子下其他网络权重仍与 V1 一致。
    """

    def __init__(self, channels, dt_ms, tau_init_ms, tau_min_ms, tau_max_ms, v_threshold,
                 memory_tau_init_ms=300.0, memory_tau_min_ms=50.0,
                 memory_tau_max_ms=1000.0, kappa_max=0.5):
        super(StreamingDualMemoryLIF2d, self).__init__(
            channels, dt_ms, tau_init_ms, tau_min_ms, tau_max_ms, v_threshold)
        values = (dt_ms, v_threshold, memory_tau_init_ms, memory_tau_min_ms,
                  memory_tau_max_ms, kappa_max)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("双仓参数必须有限")
        if dt_ms <= 0 or v_threshold <= 0:
            raise ValueError("双仓 dt_ms 与 v_threshold 必须为正")
        if not 0 < memory_tau_min_ms < memory_tau_init_ms < memory_tau_max_ms:
            raise ValueError("需要 0 < memory_tau_min_ms < memory_tau_init_ms < memory_tau_max_ms")
        if not 0 < kappa_max < 1:
            raise ValueError("双仓需要 0 < kappa_max < 1")
        self.memory_tau_min_ms = float(memory_tau_min_ms)
        self.memory_tau_max_ms = float(memory_tau_max_ms)
        self.kappa_max = float(kappa_max)
        frac = (memory_tau_init_ms - memory_tau_min_ms) / (memory_tau_max_ms - memory_tau_min_ms)
        self.memory_a = nn.Parameter(torch.full((channels,), math.log(frac / (1.0 - frac))))
        self.kappa_raw = nn.Parameter(torch.zeros(channels))
        self.memory_read_enabled = True

    def memory_tau(self):
        return self.memory_tau_min_ms + (self.memory_tau_max_ms - self.memory_tau_min_ms) * torch.sigmoid(self.memory_a)

    def memory_alpha(self):
        return torch.exp(-self.dt_ms / self.memory_tau())

    def kappa(self):
        return self.kappa_max * torch.tanh(self.kappa_raw)

    def _coefficients(self, current):
        beta = self.beta().to(current.dtype).view(1, -1, 1, 1)
        alpha = self.memory_alpha().to(current.dtype).view(1, -1, 1, 1)
        gain = (1.0 - beta) * self.v_threshold * self.kappa().to(current.dtype).view(1, -1, 1, 1)
        if not self.memory_read_enabled:
            gain = gain * 0.0
        return beta, alpha, gain

    def _step(self, current, state, beta, alpha, gain):
        if state is None:
            u_pre = current
            memory = (1.0 - alpha) * torch.tanh(current / self.v_threshold)
        else:
            if not isinstance(state, tuple) or len(state) != 2:
                raise ValueError("双仓状态必须为 (U, M)，新序列使用 None")
            soma, old_memory = state
            if soma.shape != current.shape or old_memory.shape != current.shape:
                raise ValueError("双仓状态与输入形状不匹配；独立序列应使用 None 初始化")
            u_pre = beta * soma + current + gain * old_memory
            memory = alpha * old_memory + (1.0 - alpha) * torch.tanh(current / self.v_threshold)
        output, fired = self.activate(u_pre)
        soma = u_pre - self.v_threshold * fired
        return output, (soma, memory), u_pre

    def forward(self, current, state):
        return self._step(current, state, *self._coefficients(current))

    def forward_steps(self, current, state, carry=True, keep_u_pre=True, collect_memory=False, keep=None):
        if current.shape[0] == 0:
            raise ValueError("片段至少需要 1 个窗口")
        coefficients = self._coefficients(current)
        spikes, u_pres, memories = [], [], []
        for t in range(int(current.shape[0])):
            # (U, M) 同时乘 0 等价于 None：U_pre = I，M = (1-alpha) tanh(I/theta)
            prev = mask_state(state if carry else None, keep, t)
            output, state, u_pre = self._step(current[t], prev, *coefficients)
            spikes.append(output)
            if keep_u_pre:
                u_pres.append(u_pre)
            if collect_memory:
                memories.append(state[1].detach())
        result = (torch.stack(spikes), state, torch.stack(u_pres) if keep_u_pre else None)
        return result + (torch.stack(memories),) if collect_memory else result


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

    def forward_steps(self, current, state, carry=True, keep_u_pre=True, keep=None):
        """多窗口版本 [T,B,C,H,W]；无状态，各窗口互不影响（keep 无作用）。"""
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


def mask_state(state, keep, t):
    """多流训练：第 t 窗按 keep[t]（形状 [B]）逐流保留或清零上一状态；keep 为 None 或状态为 None 时原样返回。

    支持单个张量 [B,...] 与双仓的 (U, M)。乘 0 后的零状态与 None 数值相同，并在此截断梯度。
    """
    if state is None or keep is None:
        return state
    if isinstance(state, tuple):
        return tuple(mask_state(s, keep, t) for s in state)
    factor = keep[t].to(state.dtype).view((-1,) + (1,) * (state.dim() - 1))
    return state * factor


def detach_states(states):
    """递归截断梯度、保留数值及容器结构，支持双仓的 (U, M)。"""
    if states is None:
        return None
    if isinstance(states, tuple):
        return tuple(detach_states(s) for s in states)
    if isinstance(states, list):
        return [detach_states(s) for s in states]
    return states.detach()


def state_tensors(states):
    """按顺序遍历状态树中的张量，供等价性检查及内存统计使用。"""
    if isinstance(states, (list, tuple)):
        for state in states:
            yield from state_tensors(state)
    elif states is not None:
        yield states
