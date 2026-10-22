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
StreamingTIF2d 是 V5 的 TIF 神经元（信念 M 与"已告知"分离 + 有界证据累积 + 多级变化汇报；可选输入层自校准 adapt），见类说明。
forward 逐窗调用（流式推理、增益校准）；forward_steps 一次处理一个片段的多个窗口（逐层时间并行训练），两者等价。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils


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


class BoundedSurrogate(torch.autograd.Function):
    """截断到 [0, cap]：前向 min(max(x, 0), cap) 不变；反向在区间内为 1，两端之外用代理梯度。

        x <= 0      1/(1+|x|)^2          不激活的通道仍能学（否则 relu 梯度为 0，通道死亡后不可恢复）
        x >= cap    1/(1+(x-cap))^2      到顶的元素仍能学着降下来（否则只剩 cap 一条梯度通路）
        cap         在触顶处得到梯度 1（与 torch.minimum 相同，量程 D*theta 继续可学）
    代理函数与 SurrogateSpike 同形：LIF 在阈值两侧都有梯度，TIF 的两个边界也一样（tif.surrogate: true）。
    """

    @staticmethod
    def forward(ctx, x, cap):
        ctx.save_for_backward(x, cap)
        return torch.minimum(torch.clamp(x, min=0.0), cap)

    @staticmethod
    def backward(ctx, grad_output):
        x, cap = ctx.saved_tensors
        below = 1.0 / (1.0 + x.abs()).pow(2)
        above = 1.0 / (1.0 + (x - cap)).pow(2)
        slope = torch.where(x <= 0, below, torch.where(x >= cap, above, torch.ones_like(x)))
        grad_cap = (grad_output * (x >= cap).to(grad_output.dtype)).sum_to_size(cap.shape)
        return grad_output * slope, grad_cap


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


def inverse_softplus(value):
    """softplus 的反函数：返回 x 使 softplus(x) = value（value > 0）。"""
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError("inverse_softplus 需要有限正数，收到 %r" % value)
    return value + math.log(-math.expm1(-value))


class StreamingTIF2d(StreamingLIF2d):
    """TIF 神经元（V5 基础版，见 Stream-SNN-Innovations.md 第 11、12 节；10-19 删除搬运、10-22 删除本地参照 TIF-R）。

    每个窗口、每个通道、每个像素：
        z   = I - k - relu(b_hat)                  k = softplus(.) >= 0（CUSUM 式漂移项）；b_hat 只在 adapt=True 时存在
        M   = min(relu(lam * M_prev + (1 - lam) * z), D * theta)    有界的泄漏证据累积（EMA 形式，稳态 M = z）
        out = theta * round(M / theta)             多级汇报（round 用直通估计）；quantize=False 时 out = M
    lam = exp(-dt/tau) 沿用 V1 的 tau 参数化；theta = exp(.) 逐通道可学（LSQ 式步长），初值 v_th / D。
    adapt（输入层工作点自校准，12.10；网络只在 enc1 打开，属事件表示层的补充项）：偏差校正的慢基线
        （self-starting CUSUM）。b = alpha*b + (1-alpha)*I，w = alpha*w + (1-alpha)（形状 [B,C,1,1]），
        b_hat = b_prev / w_prev 为到上一窗为止输入的（指数遗忘）均值；alpha = exp(-dt/tau_b)，tau_b 逐通道可学，
        限制在 [adapt_tau_min_ms, adapt_tau_max_ms]。平稳输入下 M -> 0，与背景水平无关；只减正的基线。
        w = 0（None 或清零的状态）时不减基线，所以第 1 窗与 reset 模式下与 adapt=False 完全相同。
        10-22 在全部 7 层上做（基线从 0 开始 / 偏差校正两版）已否定并删除：深层失去尺度约束而漂移（12.5）。
    10-22 的滞回汇报（hysteresis）已删除：只看精度的训练下会被学习抵消（12.12~12.13）。
    状态：默认为 M；adapt=True 时为 (M, b, w)。
    None 与零状态数值相同。
    forward(current, state) -> (out, new_state, M)：第三项对应 LIF 的 u_pre。
    训练（forward_steps 且需要梯度）时每一步的神经元运算在反向时重算（梯度检查点），数学不变、只省显存。
    surrogate=True 时 0 处反射与 D*theta 处截断用代理梯度（BoundedSurrogate，10-19 实验失败，默认关）。
    """

    def __init__(self, channels, dt_ms, tau_init_ms, tau_min_ms, tau_max_ms, v_threshold,
                 levels=7, quantize=True, k_init=0.05, surrogate=False,
                 adapt=False, adapt_tau_init_ms=2000.0, adapt_tau_min_ms=1000.0, adapt_tau_max_ms=10000.0):
        super(StreamingTIF2d, self).__init__(channels, dt_ms, tau_init_ms, tau_min_ms, tau_max_ms,
                                             v_threshold)
        if isinstance(levels, bool) or int(levels) != levels or int(levels) < 1:
            raise ValueError("TIF levels 必须是 >= 1 的整数")
        if not all(isinstance(flag, bool) for flag in (quantize, surrogate, adapt)):
            raise ValueError("TIF quantize / surrogate / adapt 必须是布尔值")
        self.levels = int(levels)
        self.quantize = quantize
        self.surrogate = surrogate
        self.adapt = adapt
        self.k_raw = nn.Parameter(torch.full((channels,), inverse_softplus(k_init), dtype=torch.float32))
        self.log_theta = nn.Parameter(torch.full((channels,), math.log(float(v_threshold) / self.levels),
                                                 dtype=torch.float32))
        if adapt:
            taus = (float(adapt_tau_min_ms), float(adapt_tau_init_ms), float(adapt_tau_max_ms))
            if not (all(math.isfinite(v) for v in taus) and 0 < taus[0] < taus[1] < taus[2]):
                raise ValueError("TIF adapt 需要 0 < adapt_tau_min_ms < adapt_tau_init_ms < adapt_tau_max_ms")
            self.adapt_tau_min_ms, self.adapt_tau_max_ms = taus[0], taus[2]
            frac = (taus[1] - taus[0]) / (taus[2] - taus[0])
            self.adapt_a = nn.Parameter(torch.full((channels,), math.log(frac / (1.0 - frac)),
                                                   dtype=torch.float32))
        self.collect_stats = False
        self._stats = {}

    def k(self):
        """参考值 k >= 0，形状 [C]。"""
        return F.softplus(self.k_raw)

    def theta(self):
        """量化步长 theta > 0，形状 [C]。"""
        return torch.exp(self.log_theta)

    def adapt_tau(self):
        """基线时间常数 tau_b（毫秒），形状 [C]；只在 adapt=True 时可用。"""
        return self.adapt_tau_min_ms + (self.adapt_tau_max_ms - self.adapt_tau_min_ms) * torch.sigmoid(self.adapt_a)

    def _coefficients(self, current):
        view = (1, -1, 1, 1)
        lam = self.beta().to(current.dtype).view(view)
        theta = self.theta().to(current.dtype).view(view)
        coefficients = (lam, self.k().to(current.dtype).view(view), theta, theta * self.levels)
        if self.adapt:
            alpha = torch.exp(-self.dt_ms / self.adapt_tau()).to(current.dtype).view(view)
            coefficients = coefficients + (alpha,)
        return coefficients

    def split_state(self, state):
        """状态 -> (M, b, w)；没有的项为 None。"""
        if state is None:
            return None, None, None
        if not self.adapt:
            if isinstance(state, tuple):
                raise ValueError("adapt=False 的 TIF 状态应为张量 M")
            return state, None, None
        if not isinstance(state, tuple) or len(state) != 3:
            raise ValueError("adapt=True 的 TIF 状态应为 (M, b, w)，新序列使用 None")
        return tuple(state)

    def level(self, memory):
        """量化电平 q = round(M/theta)（整数值张量，不带梯度）；memory 为 None 时返回 None。"""
        if memory is None:
            return None
        theta = self.theta().to(memory.dtype).view(1, -1, 1, 1)
        return torch.round(memory.detach() / theta)

    def _report(self, pre, theta, cap):
        """截断到 [0, D*theta] 并量化，返回 (输出, 新记忆 M)。"""
        if self.surrogate:
            memory = BoundedSurrogate.apply(pre, cap)
        else:
            memory = torch.minimum(torch.relu(pre), cap)
        if self.quantize:
            scaled = memory / theta
            output = theta * (scaled + (torch.round(scaled) - scaled).detach())
        else:
            output = memory
        return output, memory

    def _compute(self, current, prev, lam, k, theta, cap):
        """一个窗口的 TIF 运算（adapt=False），返回 (输出, 新记忆 M)。"""
        z = current - k
        if prev is None:
            pre = (1.0 - lam) * z
        else:
            pre = lam * prev + (1.0 - lam) * z
        return self._report(pre, theta, cap)

    def _compute_adapt(self, current, prev, baseline, weight, lam, k, theta, cap, alpha):
        """adapt=True 的一个窗口，返回 (输出, 新 M, 新 b, 新 w)。参照 b/w；w = 0 时 b = 0，参照为 0。"""
        if prev is None:
            pre = (1.0 - lam) * (current - k)
            new_baseline = (1.0 - alpha) * current
            new_weight = (1.0 - alpha).expand(current.shape[0], -1, -1, -1)
        else:
            reference = torch.relu(baseline) / weight.clamp(min=1e-12)
            pre = lam * prev + (1.0 - lam) * (current - k - reference)
            new_baseline = alpha * baseline + (1.0 - alpha) * current
            new_weight = alpha * weight + (1.0 - alpha)
        output, memory = self._report(pre, theta, cap)
        return output, memory, new_baseline, new_weight

    def _step(self, current, state, coefficients, recompute=False):
        prev, baseline, weight = self.split_state(state)
        checkpointed = recompute and prev is not None and torch.is_grad_enabled() and not self.collect_stats
        reference = baseline
        if self.adapt:
            args = (current, prev, baseline, weight) + tuple(coefficients)
            if checkpointed:
                output, memory, baseline, weight = checkpoint_utils.checkpoint(self._compute_adapt, *args,
                                                                               preserve_rng_state=False)
            else:
                output, memory, baseline, weight = self._compute_adapt(*args)
            new_state = (memory, baseline, weight)
            reference = baseline / weight.clamp(min=1e-12)
        else:
            args = (current, prev) + tuple(coefficients)
            if checkpointed:
                output, memory = checkpoint_utils.checkpoint(self._compute, *args, preserve_rng_state=False)
            else:
                output, memory = self._compute(*args)
            new_state = memory
        if self.collect_stats:
            self._accumulate(memory, prev, reference)
        return output, new_state, memory

    def forward(self, current, state):
        return self._step(current, state, self._coefficients(current))

    def forward_steps(self, current, state, carry=True, keep_u_pre=True, keep=None):
        """按时间顺序处理一个片段的 T 个窗口，与连续调用 forward T 次数学等价（接口同 StreamingLIF2d）。"""
        steps = int(current.shape[0])
        if steps == 0:
            raise ValueError("片段至少需要 1 个窗口")
        coefficients = self._coefficients(current)
        outputs, memories = [], []
        for t in range(steps):
            prev = mask_state(state if carry else None, keep, t)
            output, state, memory = self._step(current[t], prev, coefficients, recompute=True)
            outputs.append(output)
            if keep_u_pre:
                memories.append(memory)
        return torch.stack(outputs), state, (torch.stack(memories) if keep_u_pre else None)

    # ----- 诊断统计（只用于验证，不参与前向；训练时关闭） -----

    def start_stats(self, per_window=False):
        """开始累计诊断统计（清空旧值）。per_window=True 时另记每窗（每个样本）的 |Δq| 率，供单窗 SOP 分布用。"""
        self._stats = {}
        self._per_window = [] if per_window else None
        self.collect_stats = True

    def _accumulate(self, memory, prev, baseline):
        with torch.no_grad():
            s = self._stats
            if not s:
                zero = torch.zeros((), device=memory.device, dtype=torch.float64)
                s.update(elements=0, windows=0, occupied=zero.clone(), saturated=zero.clone(),
                         level_sum=zero.clone(), message_abs=zero.clone(), message_nonzero=zero.clone(),
                         baseline_sum=zero.clone(),
                         channel_active=torch.zeros(memory.shape[1], dtype=torch.bool, device=memory.device))
            q = self.level(memory)
            q_prev = torch.zeros_like(q) if prev is None else self.level(prev)
            delta = q - q_prev
            s["elements"] += memory.numel()
            s["windows"] += 1
            s["occupied"] += (memory > 0).sum().double()
            s["channel_active"] |= (memory > 0).flatten(2).any(2).any(0)
            s["saturated"] += (q >= self.levels).sum().double()
            s["level_sum"] += q.double().sum()
            s["message_abs"] += delta.abs().double().sum()
            s["message_nonzero"] += delta.ne(0).sum().double()
            if getattr(self, "_per_window", None) is not None:
                self._per_window.extend((delta.abs().flatten(1).sum(1).double() / float(memory[0].numel())).tolist())
            if baseline is not None:
                theta = self.theta().to(baseline.dtype).view(1, -1, 1, 1)
                s["baseline_sum"] += (torch.relu(baseline) / theta).double().sum()

    def stop_stats(self):
        """停止累计并返回汇总（每个神经元每窗的平均值）；没有数据时返回空字典。

        message_abs_rate      平均 |Δq|（单位脉冲口径：每个单位电平变化 = 一次突触累加）
        message_nonzero_rate  Δq 非零的比例（graded 口径：一条消息携带一个小整数）
        level_mean / occupancy / saturation  电平均值、M>0 的比例、电平到顶（q = D）的比例
        dead_channel_frac     评估期间 M 始终为 0 的通道占比（死通道；不量化时与"从未发放"相同）
        baseline_level_mean   adapt=True：基线 relu(b/w) 的平均值，以 theta 为单位
        """
        self.collect_stats = False
        s = self._stats
        if not s:
            return {}
        n = float(max(s["elements"], 1))
        out = {"message_abs_rate": float(s["message_abs"]) / n,
               "message_nonzero_rate": float(s["message_nonzero"]) / n,
               "level_mean": float(s["level_sum"]) / n,
               "occupancy": float(s["occupied"]) / n,
               "saturation": float(s["saturated"]) / n,
               "dead_channel_frac": 1.0 - float(s["channel_active"].double().mean()),
               "windows": int(s["windows"])}
        if self.adapt:
            out["baseline_level_mean"] = float(s["baseline_sum"]) / n
        if getattr(self, "_per_window", None) is not None:
            out["per_window_rate"] = list(self._per_window)
            self._per_window = None
        return out


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

    支持单个张量 [B,...]、双仓的 (U, M) 与 TIF 的 (M, b, w)、(M, q)。乘 0 后的零状态与 None 数值相同，并在此截断梯度。
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
