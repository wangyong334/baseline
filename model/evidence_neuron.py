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
    按位置告警（M >= theta，复位该位置的全部假设）时，每位置每窗的告警率 <= V/(e^theta - 1)。
注意：这是位置级告警的界，不是原评估代码中按连通域统计的逐事件 Fa。
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
    状态（字典）：G 各管道的目标强度 [B,V,H,W]；C 膜电位；C_pre 截断到 0 之前的膜电位；ell 本窗的管道证据；
                  shifts 本窗各假设的整数位移；k 已处理的窗数。
    第 k 窗时第 v 个假设相对第 0 窗的累计整数位移为 round(v*k)，每窗平移量取相邻两窗之差（相位累加，长期无漂移）。
    """

    def __init__(self, velocities, footprint=3, compensator="poisson", nb_kappa=None, aggregate="lme",
                 track_decay=0.0):
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
        self.velocities = [(float(vy), float(vx)) for vy, vx in velocities]
        self.footprint = int(footprint)
        self.compensator = compensator
        self.nb_kappa = nb_kappa
        self.aggregate = aggregate
        self.track_decay = float(track_decay)

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
        return {"G": zeros, "C": zeros, "C_pre": zeros, "ell": zeros, "shifts": self.step_shifts(0), "k": 0}

    def _shift_each(self, tensor, shifts, fill=0.0):
        """tensor [B,V,H,W] 的第 v 个通道平移 shifts[v]；tensor 为 [B,1,H,W] 时对每个假设各平移一份。"""
        if tensor.shape[1] == 1:
            return torch.stack([shift2d(tensor[:, 0], sy, sx, fill) for sy, sx in shifts], 1)
        return torch.stack([shift2d(tensor[:, v], sy, sx, fill) for v, (sy, sx) in enumerate(shifts)], 1)

    def predicted_intensity(self, G_old, g_prev, shifts):
        """管道的目标强度 G_v = max( 平移后的网络强度场, rho * 平移后的旧 G_v )，[B,V,H,W]。"""
        carried = None
        if self.track_decay > 0:
            carried = self.track_decay * self._shift_each(G_old, shifts)
        if g_prev is None:
            return carried if carried is not None else torch.zeros_like(G_old)
        fresh = self._shift_each(g_prev, shifts)
        return fresh if carried is None else torch.maximum(fresh, carried)

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

    def tube_evidence(self, counts, mu0, G):
        """各管道的证据 l [B,V,H,W]。counts、mu0 [B,1,H,W]；G [B,V,H,W]。"""
        log_g = torch.log(torch.clamp(G, min=math.exp(LOG_ZERO)))
        e, _ = pixel_evidence(counts, mu0, log_g, self.compensator, self.nb_kappa)
        return self.aggregate_evidence(e)

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
                C = C.masked_fill(alarm.expand_as(C), 0.0)
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
        G = self.predicted_intensity(state["G"], g_prev, shifts)
        ell = self.tube_evidence(counts, mu0, G)
        C, C_pre, M, alarm = self.accumulate(state["C"], shifts, ell, theta, reset_on_alarm)
        return {"G": G, "C": C, "C_pre": C_pre, "ell": ell, "shifts": shifts, "k": k + 1}, M, alarm

    def estimate_operations(self, height, width, events_per_window=0.0, readout_delays=(), alarm_thetas=0):
        """估计判决层每窗的理论运算量（无可学习参数，但每窗都要在 V 个假设 × 全画面上算）。

        分类与前端相同（mac / elementwise / transcendental / reducible_ops），口径见
        EvidenceFrontEnd.estimate_operations。告警通道每个阈值各维护一份膜电位，所以按 1 + len(alarm_thetas) 份计。
        延迟读出按事件计：第 k 窗的事件要在之后 d 窗里沿 V 条管道各取一次值并累加。

        输入: height/width 画布大小；events_per_window 平均每窗事件数；readout_delays 延迟读出的 d 列表；
              alarm_thetas 告警阈值个数（0 表示不跑告警通道）。
        """
        p = float(int(height) * int(width))
        e = float(events_per_window)
        v = float(self.n_hypotheses)
        f = int(self.footprint)
        parts = []

        def add(name, mac=0.0, elementwise=0.0, transcendental=0.0, reducible=0.0):
            parts.append({"part": name, "mac": float(mac), "elementwise": float(elementwise),
                          "transcendental": float(transcendental), "reducible_ops": float(reducible)})

        # 强度预测：exp(log_g) 一次；每个假设平移一次网络强度；有记忆时再平移旧 G、乘 rho、取大
        if self.track_decay > 0:
            add("管道强度预测（含记忆）", mac=v * p, elementwise=3 * v * p, transcendental=p)
        else:
            add("管道强度预测（无记忆）", elementwise=v * p, transcendental=p)
        # 逐像素证据 e = N*log r - psi：log(clamp(G))、log(mu0)、psi、softplus(log g - log mu0)
        tr = 2 * v * p + p                                     # log(G) 与 softplus 里的 log1p
        tr += 2 * v * p if self.compensator == "negbin" else v * p   # psi：negbin 多一次 log1p
        add("逐像素对数似然比", mac=v * p, elementwise=2 * v * p, transcendental=tr + v * p)
        if f > 1:
            box = float(f * f)
            separable = 2.0 * f
            if self.aggregate == "lme":
                add("足迹聚合 lme", elementwise=(box + 2) * v * p, transcendental=2 * v * p,
                    reducible=max(box - separable, 0.0) * v * p)
            else:
                add("足迹聚合 %s" % self.aggregate, mac=(v * p if self.aggregate == "sum" else 0.0),
                    elementwise=box * v * p, reducible=max(box - separable, 0.0) * v * p)
        copies = 1 + int(alarm_thetas)
        # 膜电位：平移、加证据、relu、logsumexp（减最大值、exp、求和、log）；告警另加比较与复位
        add("膜电位累加（%d 份：读出 1 + 告警 %d）" % (copies, int(alarm_thetas)),
            elementwise=copies * 5 * v * p + int(alarm_thetas) * (v + 1) * p,
            transcendental=copies * (v * p + p))
        for d in readout_delays:
            add("延迟读出 d=%d（逐事件）" % int(d),
                elementwise=e * (2 * float(d) * v + v), transcendental=e * (v + 1))
        total = {key: float(sum(part[key] for part in parts))
                 for key in ("mac", "elementwise", "transcendental", "reducible_ops")}
        total["state_elements"] = int(2 * v * p * copies)     # 每份膜电位 V 通道，另加管道强度 G
        total["hypotheses"] = int(self.n_hypotheses)
        total["per_part"] = parts
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
