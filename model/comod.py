"""增益–时间协同调制的推理期插件（Stream-SNN-Innovations.md 13.40～13.43；不训练、不改模型文件与权重）。

原理（13.41）：对 BIR（EMA 形式）M <- lam*M + (1-lam)*(I - k)，若已知背景对该神经元的期望驱动 D，
    令 g = k / (k + D)，把泄漏改为 lam**g（时间常数 / g），同时目标值改为 I - k - D。则
        目标稳态电平 = 目标驱动 - k（与 D 无关）；背景平均漂移 = -k；背景涨落方差 ∝ g*D < k（有界）；
        代价是积累时间变长 1/g 倍。时间拉伸以目标在该感受野上的停留时间为上限（13.42：超过停留时间，
        运动目标还没积累起来就离开了），上限时 g_min = tau / tau_dwell。
    mode = "sub"      只减 D、不拉伸时间（对照：均值校正，涨落不受控）
           "comod"    减 D，且时间拉伸 1/max(g, g_min)
           "nocap"    减 D，时间拉伸 1/g 不封顶（对照）
只作用在 enc1：输入端的背景统计（泊松计数）形式已知；D 由 enc1 自己的卷积作用在"背景期望输入"上得到。

背景估计（BackgroundEstimator，只用过去的计数、不读模型输出）：
    环形参照：每像素、每极性，(2*R_OUT+1)^2 盒减 (2*R_IN+1)^2 盒（挖空中心，避开目标本体与短尾迹），
              偏差校正的指数平均（tau_s 窗），首窗即可用。
    中心分量：逐像素计数截到 trunc（不丢弃），指数平均（tau_c 窗）；只有其单侧下置信界高于环形参照时才接管（热像素）。
    突变遗忘：在 SUB x SUB 子块网格上，取 5x5 子块邻域的中位速率（少数被目标占据的子块不影响中位数），
              快速平均与环形参照的子块均值偏离超过 max(50%, eps_b) 连续 2 窗时，重启该处的环形参照。
              eps_b 由网络自身导出：背景率变化使 enc1 的 g 改变约 10% 的量（0.1 * k / 单位背景率的驱动）。
    b_hat = max(环形参照, 中心下置信界)。

oracle 背景：整条序列中标签为背景的事件按像素、极性计数 / 窗数（上限对照，用到标签，只作诊断）。
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

MODES = ("sub", "comod", "nocap")
BACKGROUNDS = ("online", "oracle")
R_OUT, R_IN = 10, 4
SUB = 11
N_TERMS = 24          # 期望输入的泊松级数项数（背景率 <= 约 5 /窗 时截断误差可忽略）


def box_sum(x, radius):
    """(2r+1)^2 盒式求和，零填充（与卷积的零填充一致）。x: [B,C,H,W]。"""
    k = 2 * int(radius) + 1
    return F.avg_pool2d(x, k, stride=1, padding=int(radius), count_include_pad=True) * float(k * k)


def counts_from_input(x, q99):
    """由网络输入的整窗通道反推每像素、每极性计数（0 = ON / p=1，1 = OFF / p=0）。x: [B,12,H,W]。
    输入为 clip(log1p(C)/q99, 0, clip)，截断处的计数被低估，估计器本来就会把计数截到 trunc，影响可以忽略。"""
    q = torch.as_tensor(np.asarray(q99[:2], dtype=np.float64), dtype=x.dtype, device=x.device).view(1, 2, 1, 1)
    return torch.round(torch.expm1(x[:, :2] * q)).clamp(min=0.0)


def expected_input(b_hat, q99, time_bins, clip):
    """背景期望输入 E[clip(log1p(C)/q99, 0, clip)]，C ~ Poisson(该通道的背景率)。b_hat: [B,2,H,W]（ON, OFF）。
    通道布局（stream_windows）：0 ON 整窗、1 OFF 整窗、2+2b ON 第 b 箱、3+2b OFF 第 b 箱（箱速率 = 整窗 / time_bins）。"""
    dtype, device = b_hat.dtype, b_hat.device
    n = torch.arange(N_TERMS, dtype=dtype, device=device).view(1, 1, -1, 1, 1)
    log_fact = torch.lgamma(n + 1.0)
    q = np.asarray(q99, dtype=np.float64)
    chans = []
    for c in range(2 + 2 * int(time_bins)):
        pol = c % 2                                  # 0 -> ON（偶数通道），1 -> OFF
        rate = b_hat[:, pol:pol + 1].unsqueeze(2)    # [B,1,1,H,W]
        if c >= 2:
            rate = rate / float(time_bins)
        rate = rate.clamp(min=1e-12)
        logp = n * torch.log(rate) - rate - log_fact
        value = torch.clamp(torch.log1p(n) / float(q[c]), 0.0, float(clip))
        chans.append((torch.exp(logp) * value).sum(2))
    return torch.cat(chans, 1)


class BackgroundEstimator(object):
    """在线背景估计（见模块说明）。step(counts) 先返回本窗可用的 b_hat（只依赖之前的窗），再用本窗计数更新。"""

    def __init__(self, tau_s=20.0, tau_c=100.0, trunc=2.0, z=2.0, forget=True, eps_b=0.0):
        self.a_s, self.g_c = math.exp(-1.0 / float(tau_s)), math.exp(-1.0 / float(tau_c))
        self.trunc, self.z, self.forget, self.eps_b = float(trunc), float(z), bool(forget), float(eps_b)
        self.reset()

    def reset(self):
        self.As = self.Ws = self.Ac = self.Bc = self.fast = self.alarm = None
        self.resets = 0

    def estimate(self):
        if self.As is None:
            return None
        ring = self.As / self.Ws
        lcb = (self.Ac - self.z * torch.sqrt(self.Ac.clamp(min=0.0))) / self.Bc
        return torch.maximum(ring, lcb)

    def update(self, counts):
        valid = box_sum(torch.ones_like(counts[:, :1]), R_OUT) - box_sum(torch.ones_like(counts[:, :1]), R_IN)
        ring = (box_sum(counts, R_OUT) - box_sum(counts, R_IN)) / valid.clamp(min=1.0)
        truncated = counts.clamp(max=self.trunc)
        if self.As is None:
            self.As, self.Ws = (1 - self.a_s) * ring, torch.full_like(ring, 1 - self.a_s)
            self.Ac, self.Bc = truncated, 1.0
        else:
            self.As = self.a_s * self.As + (1 - self.a_s) * ring
            self.Ws = self.a_s * self.Ws + (1 - self.a_s)
            self.Ac = self.g_c * self.Ac + truncated
            self.Bc = self.g_c * self.Bc + 1.0
        if self.forget:
            self._forget(counts)

    def _forget(self, counts):
        H, W = counts.shape[-2:]
        sub = F.avg_pool2d(counts, SUB, stride=SUB, ceil_mode=True)                 # 子块速率 [B,2,h,w]
        h, w = sub.shape[-2:]
        patches = F.unfold(F.pad(sub, (2, 2, 2, 2), mode="replicate"), 5)           # [B,2*25,h*w]
        med = patches.view(sub.shape[0], 2, 25, h, w).median(dim=2).values
        a_f = math.exp(-1.0 / 2.0)
        self.fast = med if self.fast is None else a_f * self.fast + (1 - a_f) * med
        slow = F.avg_pool2d(self.As / self.Ws, SUB, stride=SUB, ceil_mode=True)
        off = (self.fast - slow).abs() > torch.maximum(0.5 * slow, torch.full_like(slow, self.eps_b))
        self.alarm = torch.where(off, (self.alarm if self.alarm is not None else torch.zeros_like(slow)) + 1,
                                 torch.zeros_like(slow))
        trig = self.alarm >= 2
        if bool(trig.any()):
            up = lambda t: t.repeat_interleave(SUB, -2).repeat_interleave(SUB, -1)[..., :H, :W]  # noqa: E731
            full, restart = up(trig), up(self.fast)
            self.As = torch.where(full, (1 - self.a_s) * restart, self.As)
            self.Ws = torch.where(full, torch.full_like(self.Ws, 1 - self.a_s), self.Ws)
            self.alarm = torch.where(trig, torch.zeros_like(self.alarm), self.alarm)
            self.resets += int(trig.sum())


def oracle_background(seq, height, width, n_windows):
    """整条序列中背景事件（label=0）的逐像素、逐极性平均速率 [1,2,H,W]（只作诊断上限，用到标签）。"""
    out = np.zeros((2, height, width), dtype=np.float64)
    neg = seq.label == 0
    pol = np.where(seq.p[neg] == 1, 0, 1)
    np.add.at(out, (pol, seq.y[neg], seq.x[neg]), 1.0)
    return torch.from_numpy(out / float(max(n_windows, 1))).unsqueeze(0)


class CoModPlugin(object):
    """把协同调制挂到 net.enc1 的 TIF 神经元上（只替换该实例的 _compute；detach() 恢复原样）。

    每窗调用 set_window(x) 一次（在 net(x, ...) 之前），它计算本窗的背景期望驱动 D 并交给神经元。
    dwell_ms：目标在 enc1 感受野上的停留时间（窗长 * (目标尺寸 + 感受野) / 速度，由标签统计按规则给出）。"""

    def __init__(self, net, mode, background, q99, time_bins, clip, dwell_ms, dt_ms=50.0, estimator=None):
        if mode not in MODES or background not in BACKGROUNDS:
            raise ValueError("mode 取 %s，background 取 %s" % (MODES, BACKGROUNDS))
        self.net, self.mode, self.background = net, mode, background
        self.q99, self.time_bins, self.clip = q99, int(time_bins), float(clip)
        self.block = net.enc1
        self.neuron = net.enc1.neuron
        if getattr(self.neuron, "adapt", False):
            raise ValueError("协同调制插件只用于 adapt=False 的 TIF enc1")
        self.dwell_ms, self.dt_ms = float(dwell_ms), float(dt_ms)
        self.reference = None
        self.oracle = None
        self.stats = {"g_sum": 0.0, "g_n": 0, "D_sum": 0.0}
        self.estimator = estimator if estimator is not None else BackgroundEstimator(eps_b=self._eps_b())
        self._orig = self.neuron._compute
        self.neuron._compute = self._compute

    def detach(self):
        self.neuron._compute = self._orig

    def _drive(self, b_hat):
        """背景期望输入经过 enc1 的卷积与增益 = 背景对每个 enc1 神经元的期望驱动 [B,C,H,W]。"""
        ex = expected_input(b_hat.to(next(self.net.parameters()).dtype), self.q99, self.time_bins, self.clip)
        with torch.no_grad():
            return self.block._currents(ex)[0]

    def _eps_b(self):
        """突变遗忘的绝对门槛：均匀背景率变化到使 enc1 的 g 改变约 10% 所需的量（由网络自身参数导出）。"""
        p = next(self.net.parameters())
        probe = 1e-3
        b = torch.full((1, 2, 8, 8), probe, dtype=p.dtype, device=p.device)
        slope = self._drive(b)[0, :, 4, 4] / probe                   # 每单位背景率的驱动（各通道）
        k = self.neuron.k().to(p.dtype)
        ok = slope > 0
        return float((0.1 * k[ok] / slope[ok]).min()) if bool(ok.any()) else 0.0

    def start_sequence(self, seq=None, height=None, width=None, n_windows=None):
        self.estimator.reset()
        self.reference = None
        self.oracle = None
        if self.background == "oracle":
            p = next(self.net.parameters())
            self.oracle = self._drive(oracle_background(seq, height, width, n_windows).to(p.device))

    def set_window(self, x):
        if self.background == "oracle":
            self.reference = self.oracle
            return
        b_hat = self.estimator.estimate()
        self.reference = None if b_hat is None else self._drive(b_hat)
        self.estimator.update(counts_from_input(x, self.q99).to(torch.float64))

    def _compute(self, current, prev, lam, k, theta, cap):
        if self.reference is None:
            return self._orig(current, prev, lam, k, theta, cap)
        D = torch.relu(self.reference.to(current.dtype))
        if self.mode == "sub":
            lam_eff = lam.expand_as(D)
        else:
            g = k / (k + D)
            if self.mode == "comod":
                tau_ms = -self.dt_ms / torch.log(lam.clamp(1e-12, 1 - 1e-12))
                g = torch.maximum(g, (tau_ms / self.dwell_ms).clamp(max=1.0).expand_as(g))
            lam_eff = lam ** g
            with torch.no_grad():
                self.stats["g_sum"] += float(g.mean())
                self.stats["g_n"] += 1
        with torch.no_grad():
            self.stats["D_sum"] += float(D.mean())
        z = current - k - D
        pre = (1.0 - lam_eff) * z if prev is None else lam_eff * prev + (1.0 - lam_eff) * z
        return self.neuron._report(pre, theta, cap)

    def sequence_summary(self):
        s = self.stats
        out = {"g_mean": s["g_sum"] / s["g_n"] if s["g_n"] else None, "resets": int(self.estimator.resets)}
        self.stats = {"g_sum": 0.0, "g_n": 0, "D_sum": 0.0}
        return out
