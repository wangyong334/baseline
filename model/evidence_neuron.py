"""流式 SNN V2（方案一）的判决神经元：沿速度假设漂移的 CUSUM（纯 PyTorch，兼容 torch 1.9）。

背景模型（每个像素、每个 50 ms 窗；只对每个像素的条件边缘分布做假设，不要求像素之间独立）：
    H0 只有背景        N_k(x) | 过去 ~ Poisson( mu0(x,k) )（或负二项）
    H1(v) 有目标沿 v   N_k(x) | 过去 ~ Poisson( mu0(x,k) * r_v(x,k) ),  r_v = 1 + G_v(x,k) / mu0(x,k)
管道的目标强度 G_v（每条管道自己的"预测痕迹"，只依赖过去，因此是"可预测"的）：
    G_v(x,k) = max( g(x - d_v(k), k-1),  rho * G_v(x - d_v(k), k-1) )
        g(.,k-1) 为网络在第 k-1 窗估计的目标强度场（每像素每窗目标事件数），d_v(k) 为假设 v 在本窗的整数位移；
        rho = exp(-dt / tau_track)。第一项："上一窗目标在哪、多强，按 v 平移后这一窗就该在哪"；第二项：管道记住自己
        携带过的目标强度并缓慢衰减——管道离开目标后仍然预测"这里该有目标事件"，事件不来就持续产生负证据，
        积累的分数随之下降，不会停在高位漂移（只用网络当前判断时会出现这种"冻结"）。
每窗每像素的对数似然比（证据）：
    e_v = N * log r_v - psi_v,  psi 为背景计数的累积量生成函数 log E0[r^N]：泊松 psi = G_v；负二项 psi = -kappa*log(1-G_v/kappa)
    预测这里该有目标而事件没来时 e = -G（负证据，"预测落空"）。
足迹 S（默认 3x3）内的聚合（决定需要什么样的空间假设）：
    sum   sum_y e_y              要求足迹内像素条件独立（相关背景下会失效）
    mean  mean_y e_y             对任意空间相关都成立（Jensen：exp(mean) <= mean exp）
    lme   log mean_y exp(e_y)    对任意空间相关都成立（E0[mean exp] <= 1），且不弱于 mean；默认
沿假设轨迹的 Page CUSUM：
    C_pre_v(x,k) = C_v(x - d_v(k), k-1) + l_v(x,k),   C_v = max(0, C_pre_v),   M(x,k) = logsumexp_v C_v - log V
它是一个 LIF：膜电位 C，输入电流 N*log r，减法泄漏 psi（随预测变化），静息下限 0，阈值 theta，复位 C<-0；
G_v 是它的第二个（预测）隔室。

保证（条件：g 只依赖过去；每个像素的 psi 不小于其背景条件分布的累积量生成函数——泊松时即 mu0 不低于真实背景均值；
      聚合方式为 mean / lme 时对像素间的相关性不作任何假设，sum 时要求条件独立）：
    E0[exp(l_v) | 过去] <= 1。对每条管道，带复位的 CUSUM 在 L 步内的告警次数期望 <= L/(e^theta - 1)
    （Shiryaev-Roberts 上鞅论证：e^C <= 1 + R，R_k = (1+R_{k-1}) e^{l_k}，R_k - k 为上鞅；对任意网络权重、任意时长成立）；
    按位置告警（M >= theta，复位该位置的全部假设）时，每位置每窗的告警率 <= (1 + 1/L)/(e^theta - 1)，与 V 无关
    （紧界：Q_k = sum_x mean_v R_v(x,k) 满足 E0[Q_k | 过去] <= HW + Q_{k-1}，告警处 mean_v R_v >= e^theta - 1 且被清零；
     早先写的 V/(e^theta - 1) 是对 V 个假设取并集的松界）。
注意：这是位置级告警的界，不是原评估代码中按连通域统计的逐事件 Fa。

评估时开关（09-24 阶段 0 实测有效；本类的缺省值 = 原实现，逐位相同；任意取值都不破坏上面的保证。
V2 冻结配置在 YAML 里把 gate_eps 设为 0.03，即默认稀疏执行）：
    memory_gain λ  非对称记忆（D0）。加分用 Gp = max(平移后的网络强度, λ*rho*旧 G)，扣分仍用带记忆的 G：
                   e = N*log(1 + Gp/mu0) - psi(Gp) - (G - Gp)；Gp <= G 时 E0[exp(e)] = exp(-(G - Gp)) <= 1。
                   λ = 1 即原实现；λ < 1 让漂离目标的错误速度管道不再因记忆而对背景事件"过度敏感"
                   （原实现下一个溢出事件能给 log(1 + 1/0.005) ≈ 5.3 nat，θ 抬高 4 nat 只多要 1~2 个事件）。
    gate_eps ε     预测门控（D2）。低于 ε 的网络强度与记忆一律置 0（只用过去信息，仍可预测）；ε = 0 即原实现。
                   门控后的 G 恰好等于"不门控的 G 按 ε 截断"，所以活跃比例可以从 ε = 0 的一次运行里读出。
                   实测 ε = 0.03 时读出与告警都不变，活跃 (假设, 像素) 不到 1%。
    reset_radius r 告警时复位切比雪夫距离 r 以内的全部假设（r = 0 即原来的位置复位）；复位只会减小 R，界不变。
                   实测是抑制"目标附近错位告警"最有效的单个开关。
注意：上界以背景模型保守为前提。真实背景在时间上有持续活动（稠密序列、闪烁、局部杂波），
用"追着上一窗事件跑"的对手预测做压力测试时，高 θ 下会越界；用实际网络预测时纯背景窗没有告警。
论文里保证要写成有条件的形式。
逐事件的延迟读出见 TubeReadout。本模块不含可学习参数，训练时不参与反向传播。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

COMPENSATORS = ("poisson", "negbin")
AGGREGATES = ("sum", "mean", "lme")
LOG_ZERO = -30.0          # 没有预测的位置取 log g = -30（g 约 1e-13，相当于"那里没有目标"）
EXP_CLIP = 80.0           # lme 里 exp 前把证据截断在 80 以下（截断只会让证据变小，保证仍成立）


def shift2d(x, sy, sx, fill=0.0):
    """把最后两维整体平移 (sy, sx) 个像素，移出的部分丢弃，移入的部分填 fill。

    约定: out[..., y, x] = x[..., y - sy, x - sx]（sy>0 向下、sx>0 向右平移）。
    """
    sy, sx = int(sy), int(sx)
    if sy == 0 and sx == 0:
        return x
    h, w = int(x.shape[-2]), int(x.shape[-1])
    out = torch.full_like(x, fill)
    if abs(sy) >= h or abs(sx) >= w:
        return out
    out[..., max(sy, 0):h + min(sy, 0), max(sx, 0):w + min(sx, 0)] = \
        x[..., max(-sy, 0):h - max(sy, 0), max(-sx, 0):w - max(sx, 0)]
    return out


def velocity_grid(axis_values):
    """由单轴速度取值（像素/窗）生成二维速度假设列表 [(vy, vx), ...]，按 (vy, vx) 字典序排列。"""
    values = [float(v) for v in axis_values]
    if len(set(values)) != len(values):
        raise ValueError("速度取值有重复: %s" % values)
    return [(vy, vx) for vy in values for vx in values]


def pixel_evidence(counts, mu0, log_g, compensator="poisson", nb_kappa=None):
    """每窗每像素的对数似然比 e = N*log r - psi（见模块说明），支持广播。

    返回: (e, log_r)。负二项时把 g 截断在 0.9*kappa 以内（截断后的 g 同时用于 log r 与 psi，保证仍成立）。
    """
    if compensator not in COMPENSATORS:
        raise ValueError("compensator 必须是 %s 之一" % (COMPENSATORS,))
    log_mu0 = torch.log(mu0)
    if compensator == "negbin":
        if nb_kappa is None or float(nb_kappa) <= 0:
            raise ValueError("负二项补偿需要正的 nb_kappa")
        kappa = float(nb_kappa)
        log_g = torch.clamp(log_g, max=math.log(0.9 * kappa))
        psi = -kappa * torch.log1p(-torch.exp(log_g) / kappa)
    else:
        psi = torch.exp(log_g)
    log_r = F.softplus(log_g - log_mu0)
    return counts * log_r - psi, log_r


class DriftCUSUM(nn.Module):
    """沿速度假设漂移的 CUSUM 判决层（无可学习参数）。

    参数:
        velocities    速度假设列表 [(vy, vx), ...]，单位 像素/窗
        footprint     证据聚合的足迹边长（奇数），近似目标大小
        compensator   poisson | negbin
        nb_kappa      负二项离散参数（compensator=negbin 时必填）
        aggregate     sum | mean | lme（见模块说明；mean/lme 不要求像素间独立）
        track_decay   管道强度记忆的每窗衰减 rho（0 表示不记忆，只用网络上一窗的强度场）
        memory_gain   非对称记忆的 λ ∈ [0, 1]（1 = 原实现）
        gate_eps      预测门控阈值 ε >= 0（0 = 不门控）
        reset_radius  告警复位的邻域半径 r >= 0（0 = 只复位告警位置）
    状态（字典）：G 各管道带记忆的目标强度 [B,V,H,W]（扣分用）；Gp 加分用的强度（λ = 1 时与 G 是同一个张量）；
                  C 膜电位；C_pre 截断到 0 之前的膜电位；ell 本窗的管道证据；shifts 本窗各假设的整数位移；k 已处理的窗数。
    第 k 窗时第 v 个假设相对第 0 窗的累计整数位移为 round(v*k)，每窗平移量取相邻两窗之差（相位累加，长期无漂移）。
    """

    def __init__(self, velocities, footprint=3, compensator="poisson", nb_kappa=None, aggregate="lme",
                 track_decay=0.0, memory_gain=1.0, gate_eps=0.0, reset_radius=0):
        super(DriftCUSUM, self).__init__()
        if int(footprint) < 1 or int(footprint) % 2 == 0:
            raise ValueError("footprint 必须是正奇数")
        if not velocities:
            raise ValueError("至少需要一个速度假设")
        if compensator not in COMPENSATORS:
            raise ValueError("compensator 必须是 %s 之一" % (COMPENSATORS,))
        if aggregate not in AGGREGATES:
            raise ValueError("aggregate 必须是 %s 之一" % (AGGREGATES,))
        if not 0.0 <= float(track_decay) < 1.0:
            raise ValueError("track_decay 必须在 [0, 1) 内")
        if not 0.0 <= float(memory_gain) <= 1.0:
            raise ValueError("memory_gain 必须在 [0, 1] 内（> 1 时加分用的强度会超过扣分用的强度，保证不成立）")
        if float(gate_eps) < 0:
            raise ValueError("gate_eps 必须 >= 0")
        if int(reset_radius) < 0 or int(reset_radius) != float(reset_radius):
            raise ValueError("reset_radius 必须是非负整数")
        self.velocities = [(float(vy), float(vx)) for vy, vx in velocities]
        self.footprint = int(footprint)
        self.compensator = compensator
        self.nb_kappa = nb_kappa
        self.aggregate = aggregate
        self.track_decay = float(track_decay)
        self.memory_gain = float(memory_gain)
        self.gate_eps = float(gate_eps)
        self.reset_radius = int(reset_radius)

    @property
    def n_hypotheses(self):
        """速度假设个数 V。"""
        return len(self.velocities)

    def offsets(self, k):
        """第 k 窗时每个假设相对第 0 窗的累计整数位移，返回 [(dy, dx), ...]。"""
        return [(int(math.floor(vy * k + 0.5)), int(math.floor(vx * k + 0.5))) for vy, vx in self.velocities]

    def step_shifts(self, k):
        """第 k 窗每个假设的整数位移 d_v(k) = offset(k) - offset(k-1)（k=0 时为 0）。"""
        if k <= 0:
            return [(0, 0)] * self.n_hypotheses
        before, after = self.offsets(k - 1), self.offsets(k)
        return [(a[0] - b[0], a[1] - b[1]) for a, b in zip(after, before)]

    def init_state(self, batch, height, width, device, dtype=torch.float32):
        """序列开头的状态：强度记忆与膜电位全 0。"""
        zeros = torch.zeros(batch, self.n_hypotheses, height, width, device=device, dtype=dtype)
        return {"G": zeros, "Gp": zeros, "C": zeros, "C_pre": zeros, "ell": zeros, "shifts": self.step_shifts(0),
                "k": 0}

    def _shift_each(self, tensor, shifts, fill=0.0):
        """tensor [B,V,H,W] 的第 v 个通道平移 shifts[v]；tensor 为 [B,1,H,W] 时对每个假设各平移一份。"""
        if tensor.shape[1] == 1:
            return torch.stack([shift2d(tensor[:, 0], sy, sx, fill) for sy, sx in shifts], 1)
        return torch.stack([shift2d(tensor[:, v], sy, sx, fill) for v, (sy, sx) in enumerate(shifts)], 1)

    def _gate(self, G):
        """预测门控：低于 gate_eps 的强度置 0（gate_eps = 0 时原样返回）。"""
        if self.gate_eps <= 0:
            return G
        return G * (G >= self.gate_eps).to(G.dtype)

    def predicted_pair(self, G_old, g_prev, shifts):
        """两个强度 (G, Gp)，[B,V,H,W]：
            G  = max( 平移后的网络强度场, rho * 平移后的旧 G )          带记忆，用于扣分（预测落空）
            Gp = max( 平移后的网络强度场, λ * rho * 平移后的旧 G )      用于加分（λ = 1 时就是 G，返回同一个张量）
        两者都经过门控；门控在取大之前作用于新鲜强度、之后作用于结果，所以 Gp <= G 始终成立。
        """
        carried = None
        if self.track_decay > 0:
            carried = self.track_decay * self._shift_each(G_old, shifts)
        fresh = None if g_prev is None else self._gate(self._shift_each(g_prev, shifts))
        if fresh is None:
            G = carried if carried is not None else torch.zeros_like(G_old)
        else:
            G = fresh if carried is None else torch.maximum(fresh, carried)
        G = self._gate(G)
        if self.memory_gain >= 1.0 or carried is None:
            return G, G
        scaled = self.memory_gain * carried
        Gp = scaled if fresh is None else torch.maximum(fresh, scaled)
        return G, self._gate(Gp)

    def predicted_intensity(self, G_old, g_prev, shifts):
        """管道带记忆的目标强度 G_v = max( 平移后的网络强度场, rho * 平移后的旧 G_v )，[B,V,H,W]。"""
        return self.predicted_pair(G_old, g_prev, shifts)[0]

    def aggregate_evidence(self, e):
        """足迹内聚合逐像素证据 e [B,V,H,W]（画面外按证据 0 计入，对 mean/lme 相当于"该处 E0[exp(e)] = 1"）。"""
        f = self.footprint
        if f == 1:
            return e
        pool = lambda z: F.avg_pool2d(z, f, stride=1, padding=f // 2, count_include_pad=True)  # noqa: E731
        if self.aggregate == "sum":
            return pool(e) * float(f * f)
        if self.aggregate == "mean":
            return pool(e)
        r = f // 2
        padded = F.pad(torch.exp(torch.clamp(e, max=EXP_CLIP)), (r, r, r, r), value=1.0)   # 画面外 exp(0) = 1
        out = torch.log(F.avg_pool2d(padded, f, stride=1))
        return torch.clamp(out, min=-1e4)             # 全部下溢时 log(0) = -inf，换成等价的 -1e4

    def tube_evidence(self, counts, mu0, G, Gp=None):
        """各管道的证据 l [B,V,H,W]。counts、mu0 [B,1,H,W]；G [B,V,H,W] 带记忆的强度；
        Gp 加分用的强度（None 或与 G 是同一个张量时即原实现）：e = N*log(1+Gp/mu0) - psi(Gp) - (G - Gp)。"""
        if Gp is None or Gp is G:
            log_g = torch.log(torch.clamp(G, min=math.exp(LOG_ZERO)))
            e, _ = pixel_evidence(counts, mu0, log_g, self.compensator, self.nb_kappa)
            return self.aggregate_evidence(e)
        log_gp = torch.log(torch.clamp(Gp, min=math.exp(LOG_ZERO)))
        e, _ = pixel_evidence(counts, mu0, log_gp, self.compensator, self.nb_kappa)
        return self.aggregate_evidence(e - (G - Gp))        # 预测落空的扣分沿用记忆

    def accumulate(self, C, shifts, ell, theta=None, reset=False):
        """膜电位递推（可对同一份证据维护多份膜电位，例如不同告警阈值各一份）。

        返回: (C_new, C_pre, M [B,1,H,W], alarm [B,1,H,W] 布尔或 None)
        """
        if any(s != (0, 0) for s in shifts):
            C = self._shift_each(C, shifts)
        C_pre = C + ell
        C = torch.relu(C_pre)
        M = torch.logsumexp(C, dim=1, keepdim=True) - math.log(self.n_hypotheses)
        alarm = None
        if theta is not None:
            alarm = M >= float(theta)
            if reset:
                zone = alarm
                if self.reset_radius > 0:            # 邻域复位：告警位置切比雪夫距离 r 以内的全部假设
                    size = 2 * self.reset_radius + 1
                    zone = F.max_pool2d(alarm.to(C.dtype), size, stride=1, padding=self.reset_radius) > 0
                C = C.masked_fill(zone.expand_as(C), 0.0)
        return C, C_pre, M, alarm

    def step(self, state, counts, mu0, log_g_prev, theta=None, reset_on_alarm=False):
        """处理一个窗口。

        输入: counts [B,1,H,W] 本窗事件数；mu0 [B,1,H,W] 只用过去估计的背景；
              log_g_prev [B,1,H,W] 网络在上一窗估计的目标强度场（log），None 表示还没有；
              theta 告警阈值；reset_on_alarm=True 时告警位置的全部假设复位为 0。
        输出: (新状态, M [B,1,H,W] 证据分数, alarm [B,1,H,W] 布尔或 None)
        """
        k = int(state["k"])
        shifts = self.step_shifts(k)
        g_prev = None if log_g_prev is None else torch.exp(log_g_prev)
        G, Gp = self.predicted_pair(state["G"], g_prev, shifts)
        ell = self.tube_evidence(counts, mu0, G, Gp)
        C, C_pre, M, alarm = self.accumulate(state["C"], shifts, ell, theta, reset_on_alarm)
        return {"G": G, "Gp": Gp, "C": C, "C_pre": C_pre, "ell": ell, "shifts": shifts, "k": k + 1}, M, alarm

    def estimate_operations(self, height, width, events_per_window=0.0, readout_delays=(), alarm_thetas=0,
                            active_fraction=1.0):
        """估计判决层每窗的理论运算量（无可学习参数，但每窗都要在 V 个假设 × 全画面上算）。

        分类与前端相同（mac / elementwise / transcendental / reducible_ops），口径见
        EvidenceFrontEnd.estimate_operations。告警通道每个阈值各维护一份膜电位，所以按 1 + len(alarm_thetas) 份计。
        延迟读出按事件计：第 k 窗的事件要在之后 d 窗里沿 V 条管道各取一次值并累加。

        输入: height/width 画布大小；events_per_window 平均每窗事件数；readout_delays 延迟读出的 d 列表；
              alarm_thetas 告警阈值个数（0 表示不跑告警通道）；
              active_fraction 稀疏同步执行时需要计算的 (假设, 像素) 占比（1 = 稠密，即现实现）。门控 ε > 0 时，
              足迹内没有任何 G >= ε 的神经元证据恒为 0、状态不变，可以跳过；调用方应传入"足迹膨胀后的活跃比例"
              （train_stream_v2.py 的 decision_activity）。只缩放逐 (假设, 像素) 的项，逐像素的项与逐事件读出不变，
              告警的混合分数也按活跃比例计（实际只需在 c >= θ 的候选处算，偏保守）。
        """
        p = float(int(height) * int(width))
        e = float(events_per_window)
        v = float(self.n_hypotheses)
        f = int(self.footprint)
        a = float(active_fraction)
        if not 0.0 <= a <= 1.0:
            raise ValueError("active_fraction 必须在 [0, 1] 内")
        vp = v * p * a
        parts = []

        def add(name, mac=0.0, elementwise=0.0, transcendental=0.0, reducible=0.0):
            parts.append({"part": name, "mac": float(mac), "elementwise": float(elementwise),
                          "transcendental": float(transcendental), "reducible_ops": float(reducible)})

        # 强度预测：exp(log_g) 一次；每个假设平移一次网络强度；有记忆时再平移旧 G、乘 rho、取大
        if self.track_decay > 0:
            add("管道强度预测（含记忆）", mac=vp, elementwise=3 * vp, transcendental=p)
        else:
            add("管道强度预测（无记忆）", elementwise=vp, transcendental=p)
        if self.memory_gain < 1.0 and self.track_decay > 0:      # Gp：乘 λ、取大；证据里再减 (G - Gp)
            add("非对称记忆", mac=vp, elementwise=3 * vp)
        if self.gate_eps > 0 or a < 1.0:        # 逐像素比较网络强度 g >= ε，活跃神经元上比较 + 置零（新鲜强度与结果各一次）
            add("预测门控", elementwise=2 * p + 2 * vp)
        if a < 1.0:
            # 稀疏执行的簿记：每个活跃神经元的寻址 / 查表 / 分配回收约 6 次，
            # 活跃集按足迹扩展时每个神经元查 |S| 个邻居；逐像素扫描一次门控掩码
            add("稀疏簿记（寻址、查表、足迹扩展）", elementwise=(float(f * f) + 6.0) * vp + p)
        # 逐像素证据 e = N*log r - psi：log(clamp(G))、log(mu0)、psi、softplus(log g - log mu0)
        tr = 2 * vp + p                                        # log(G) 与 softplus 里的 log1p
        tr += 2 * vp if self.compensator == "negbin" else vp   # psi：negbin 多一次 log1p
        add("逐像素对数似然比", mac=vp, elementwise=2 * vp, transcendental=tr + vp)
        if f > 1:
            box = float(f * f)
            separable = 2.0 * f
            if self.aggregate == "lme":
                add("足迹聚合 lme", elementwise=(box + 2) * vp, transcendental=2 * vp,
                    reducible=max(box - separable, 0.0) * vp)
            else:
                add("足迹聚合 %s" % self.aggregate, mac=(vp if self.aggregate == "sum" else 0.0),
                    elementwise=box * vp, reducible=max(box - separable, 0.0) * vp)
        copies = 1 + int(alarm_thetas)
        # 膜电位：平移、加证据、relu、logsumexp（减最大值、exp、求和、log）；告警另加比较与复位
        add("膜电位累加（%d 份：读出 1 + 告警 %d）" % (copies, int(alarm_thetas)),
            elementwise=copies * 5 * vp + int(alarm_thetas) * (v * a + 1) * p,
            transcendental=copies * (vp + p * a))
        if self.reset_radius > 0 and int(alarm_thetas) > 0:     # 邻域复位：告警图的最大池化
            side = float(2 * self.reset_radius + 1)
            add("邻域复位（r=%d）" % self.reset_radius, elementwise=int(alarm_thetas) * 2 * side * p)
        for d in readout_delays:
            add("延迟读出 d=%d（逐事件）" % int(d),
                elementwise=e * (2 * float(d) * v + v), transcendental=e * (v + 1))
        total = {key: float(sum(part[key] for part in parts))
                 for key in ("mac", "elementwise", "transcendental", "reducible_ops")}
        total["state_elements"] = int(2 * vp * copies)        # 每份膜电位 V 通道，另加管道强度 G
        total["hypotheses"] = int(self.n_hypotheses)
        total["active_fraction"] = a
        total["per_part"] = parts
        if a < 1.0:
            total["note"] = ("稀疏同步估计：只计足迹内有 G >= ε 的 (假设, 像素)，占比 %.4g；"
                             "逐像素与逐事件的项不缩放，告警混合分数按活跃比例计（偏保守）。" % a)
        else:
            total["note"] = ("判决层是稠密的：无论事件多稀疏，每窗都要在 V 个假设 × 全画面上更新。"
                             "要让它随事件稀疏度下降，需要块稀疏执行（方案三），当前实现没有。")
        return total

    def gather_along(self, tensor, k_from, k_to, b, y, x):
        """取 tensor [B,V,H,W] 在"第 k_from 窗经过 (y,x) 的各假设管道"于第 k_to 窗所在位置的值。

        返回: (values [V,N], inside [V,N] 布尔)，管道已移出画面处 values 为 0。
        """
        V, H, W = self.n_hypotheses, int(tensor.shape[2]), int(tensor.shape[3])
        n = int(y.shape[0])
        start, now = self.offsets(int(k_from)), self.offsets(int(k_to))
        dy = torch.tensor([now[v][0] - start[v][0] for v in range(V)], device=y.device, dtype=torch.long)
        dx = torch.tensor([now[v][1] - start[v][1] for v in range(V)], device=y.device, dtype=torch.long)
        yy = y.view(1, n) + dy.view(V, 1)
        xx = x.view(1, n) + dx.view(V, 1)
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        vv = torch.arange(V, device=y.device).view(V, 1).expand(V, n)
        values = tensor[b.view(1, n).expand(V, n), vv, yy.clamp(0, H - 1), xx.clamp(0, W - 1)]
        return torch.where(inside, values, torch.zeros_like(values)), inside


class TubeReadout(object):
    """逐事件的延迟读出（轨迹证据融合分数）：第 k 窗的事件在第 k+d 窗得到"之后 d 窗沿各速度管道的证据"。

    F(d) = logsumexp_v [ sum_{m=k+1..k+d} l_v(x + 位移_v(k->m), m) ] - log V
    调用方把它与网络在第 k 窗给出的逐事件 logit 融合：score = mark + w * F(d)（w 在验证集上校准）。
    注意：mark 判断"这个事件是不是目标"，管道证据判断"这个邻域之后有没有符合预测的目标活动"，两者的假设并不相同，
    所以融合分数不是严格的后验概率（例如紧挨真目标的背景事件也会分到正证据），称为融合分数、在验证集上校准。
    目标沿真实速度继续出现时 F 为正；网络不预测目标处 F 约为 0；预测落空时 F 为负。管道移出画面后不再累加证据。
    用法：每处理完一个窗口 k 调用 step(state, k, b, y, x, key)，返回本步到期的 [(key, d, F [N], 实际发布窗号)]；
    序列结束调用 flush()，尚未到期的延迟在最后一窗读出（实际发布窗号 < k+d，延迟被截断）。本类不修改 CUSUM 状态。
    """

    def __init__(self, cusum, delays):
        self.cusum = cusum
        self.delays = sorted(set(int(d) for d in delays))
        if any(d < 1 for d in self.delays):
            raise ValueError("延迟必须 >= 1（延迟 0 就是网络输出本身）")
        self.max_delay = max(self.delays) if self.delays else 0
        self.pending = []
        self.last_k = -1

    def _score(self, run):
        return torch.logsumexp(run, dim=0) - math.log(self.cusum.n_hypotheses)

    def step(self, state, k, b, y, x, key):
        out = []
        k = int(k)
        for entry in self.pending:
            values, _ = self.cusum.gather_along(state["ell"], entry["k"], k, entry["b"], entry["y"], entry["x"])
            entry["run"] = entry["run"] + values
            if (k - entry["k"]) in self.delays:
                out.append((entry["key"], k - entry["k"], self._score(entry["run"]), k))
        self.pending = [e for e in self.pending if k - e["k"] < self.max_delay]
        if self.max_delay > 0:
            run = state["ell"].new_zeros(self.cusum.n_hypotheses, int(y.shape[0]))
            self.pending.append({"k": k, "b": b, "y": y, "x": x, "key": key, "run": run})
        self.last_k = k
        return out

    def flush(self):
        out = []
        for entry in self.pending:
            for d in self.delays:
                if d > self.last_k - entry["k"]:
                    out.append((entry["key"], d, self._score(entry["run"]), self.last_k))
        self.pending = []
        return out
