"""噪声校准插件：推理期把 enc1 每个神经元的背景统计"矩匹配"回训练时的参照工作点（Stream-SNN-Innovations.md 13.40～13.45；
不训练、不改模型文件与权重）。

原理（13.44 推导，13.45 修正）：
    enc1 的输入电流 I = 目标驱动 + 背景驱动；背景驱动在位置 x、通道 c 上的均值 mu(b)、方差 R(b) 由该处的背景率 b
    （每像素、每极性、每窗）按泊松模型精确算出（12 通道的 log1p/q99 编码、整窗与时间箱通道之间的相关、带符号的卷积权重
    与逐通道增益都计入；像素之间、极性之间按独立处理）。
    网络是在参照背景率 b_ref（训练集逐序列背景率的中位数，按极性）下学到的；在更密的背景下，只校正"超出参照"的部分：
        均值：I' = I - (mu(b_eff) - mu(b_ref))，b_eff = max(b_hat, b_ref)，背景不高于参照处完全不变
        方差：rho = R(b_eff) / R(b_ref) >= 1；先用时间偿还——时间常数乘 s，最多拉到目标在 enc1 感受野上的停留时间
              （s = min(rho, max(1, tau_dwell / tau))）；剩余的 rho / s 倍方差，ntt 模式下用增益偿还
              （I' = mu(b_ref) + sqrt(s / rho) * (I - mu(b_eff))，即 CFAR：该处目标幅度按 sqrt(s/rho) 下降，这是真实的
              可检测性损失）；其他模式不偿还（交给网络）。
    mode = "sub"      只校正均值（s = 1）
           "nocap"    均值 + 时间，s = rho 不封顶（对照：运动目标会在积累起来之前离开）
           "comod"    均值 + 时间，按停留时间封顶，剩余方差不处理
           "ntt"      均值 + 时间（封顶）+ 增益偿还剩余方差（先时间、后阈值）
只作用在 enc1：输入端背景的统计形式已知（泊松计数）；深层是学到的特征，零假设未知。

背景估计（BackgroundEstimator，只用过去的计数、不读模型输出）：
    环形参照：每像素、每极性，(2*R_OUT+1)^2 盒减 (2*R_IN+1)^2 盒（挖空中心，避开目标本体与短尾迹），
              偏差校正的指数平均（tau_s 窗），首窗即可用。
    中心分量：逐像素计数截到 trunc（不丢弃），指数平均（tau_c 窗）；只有其单侧下置信界高于环形参照、
              并且该像素是孤立点源（速率高于 8 邻域平均的 3 倍）时才接管。热像素是单像素缺陷，目标有空间范围，
              邻居同样偏高，所以静止的目标不会被当成热像素吸收（13.44）。
    突变遗忘：在 SUB x SUB 子块网格上，取 5x5 子块邻域的中位速率（少数被目标占据的子块不影响中位数），
              快速平均与环形参照的子块均值偏离超过 max(50%, eps_b) 连续 2 窗时，重启该处的环形参照
              （eps_b = 参照背景率的一半：低于它的变化不会触发任何校正）。
    b_hat = max(环形参照, 中心下置信界)。

oracle 背景：整条序列中标签为背景的事件按像素、极性计数 / 窗数（上限对照，用到标签，只作诊断）。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("sub", "nocap", "comod", "ntt")
BACKGROUNDS = ("online", "oracle")
R_OUT, R_IN = 10, 4
SUB = 11
N_WHOLE, N_BIN = 24, 12       # 泊松级数的截断项数（整窗 / 时间箱）；背景率 <= 约 5 /窗 时截断误差可忽略
# 参照背景率（每像素每窗，按极性 ON / OFF）：训练集 99 条序列的逐序列平均背景率的中位数（13.45，本地由训练集 zip 统计）
REF_RATE = (0.0021, 0.00001)


def box_sum(x, radius):
    """(2r+1)^2 盒式求和，零填充（与卷积的零填充一致）。x: [B,C,H,W]。"""
    k = 2 * int(radius) + 1
    return F.avg_pool2d(x, k, stride=1, padding=int(radius), count_include_pad=True) * float(k * k)


def counts_from_input(x, q99):
    """由网络输入的整窗通道反推每像素、每极性计数（0 = ON / p=1，1 = OFF / p=0）。x: [B,12,H,W]。
    输入为 clip(log1p(C)/q99, 0, clip)，截断处的计数被低估，估计器本来就会把计数截到 trunc，影响可以忽略。"""
    q = torch.as_tensor(np.asarray(q99[:2], dtype=np.float64), dtype=x.dtype, device=x.device).view(1, 2, 1, 1)
    return torch.round(torch.expm1(x[:, :2] * q)).clamp(min=0.0)


def _pmf(rate, n_max):
    """泊松概率 P(n; rate)，n = 0..n_max。rate: [B,2,H,W] -> [B,2,n_max+1,H,W]。"""
    n = torch.arange(n_max + 1, dtype=rate.dtype, device=rate.device).view(1, 1, -1, 1, 1)
    r = rate.clamp(min=1e-12).unsqueeze(2)
    return torch.exp(n * torch.log(r) - r - torch.lgamma(n + 1.0))


def input_moments(b_hat, q99, time_bins, clip):
    """背景下 12 个输入通道的均值、方差，以及同一像素同一极性"整窗通道 × 第 b 个时间箱通道"的协方差。

    b_hat: [B,2,H,W]（ON, OFF 每窗速率）。通道布局（stream_windows）：0 ON 整窗、1 OFF 整窗、2+2b ON 第 b 箱、3+2b OFF 第 b 箱。
    每个时间箱计数 ~ Poisson(b/T)，彼此独立；整窗计数 = 各箱之和。x_c = clip(log1p(n)/q_c, 0, clip)。
    返回 (mean [B,12,H,W], var [B,12,H,W], cov [B,2T,H,W])，cov 的第 2b+p 个通道 = Cov(x_整窗(p), x_箱b(p))。"""
    T = int(time_bins)
    q = np.asarray(q99, dtype=np.float64)
    dt, dev = b_hat.dtype, b_hat.device
    f = lambda n, qc: torch.clamp(torch.log1p(n) / float(qc), 0.0, float(clip))  # noqa: E731
    nw = torch.arange(N_WHOLE + 1, dtype=dt, device=dev)
    nb = torch.arange(N_BIN + 1, dtype=dt, device=dev)
    mean, var = [None] * (2 + 2 * T), [None] * (2 + 2 * T)
    cov = [None] * (2 * T)
    for p in (0, 1):
        rate = b_hat[:, p]
        PN = _pmf(rate.unsqueeze(1), N_WHOLE)[:, 0]            # [B,NW+1,H,W] 整窗计数
        Pi = _pmf((rate / T).unsqueeze(1), N_BIN)[:, 0]         # 本箱计数
        Pj = _pmf((rate * (T - 1) / T).unsqueeze(1), N_WHOLE)[:, 0]   # 其余各箱之和
        fw = f(nw, q[p]).view(1, -1, 1, 1)
        mean[p] = (PN * fw).sum(1)
        var[p] = (PN * fw * fw).sum(1) - mean[p] ** 2
        F_ij = f(nb.view(-1, 1) + nw.view(1, -1), q[p])            # [NB+1, NW+1]：f_整窗(i + j)
        A = torch.einsum("bjhw,ij->bihw", Pj, F_ij)                 # E[f_整窗(i + J)]，按本箱计数 i
        for t in range(T):
            c = 2 + 2 * t + p
            fb = f(nb, q[c]).view(1, -1, 1, 1)
            mean[c] = (Pi * fb).sum(1)
            var[c] = (Pi * fb * fb).sum(1) - mean[c] ** 2
            cov[2 * t + p] = (Pi * fb * A).sum(1) - mean[p] * mean[c]
    return torch.stack(mean, 1), torch.stack(var, 1).clamp(min=0.0), torch.stack(cov, 1)


class BackgroundEstimator(object):
    """在线背景估计（见模块说明）。step(counts) 先返回本窗可用的 b_hat（只依赖之前的窗），再用本窗计数更新。"""

    def __init__(self, tau_s=20.0, tau_c=100.0, trunc=2.0, z=2.0, forget=True, eps_b=0.0, mask=None):
        """mask: 可选的传感器有效区掩码 [1,1,H,W]（1 = 传感器像素，0 = 画布补零区）。
        给定时按归一化卷积（Knutsson & Westin 1993）只在有效像素上求平均：补零区是"没有观测"，不是"没有事件"，
        否则靠近补零边的环形参照、邻域与子块速率都会被低估（10-29：下边缘误检）。None 时与原实现逐位相同。"""
        self.a_s, self.g_c = math.exp(-1.0 / float(tau_s)), math.exp(-1.0 / float(tau_c))
        self.trunc, self.z, self.forget, self.eps_b = float(trunc), float(z), bool(forget), float(eps_b)
        self.mask = mask
        self.reset()

    def _valid(self, like):
        return torch.ones_like(like[:, :1]) if self.mask is None else self.mask.to(like)

    def reset(self):
        self.As = self.Ws = self.Ac = self.Bc = self.fast = self.alarm = None
        self.resets = 0

    def estimate(self):
        if self.As is None:
            return None
        ring = self.As / self.Ws
        lcb = (self.Ac - self.z * torch.sqrt(self.Ac.clamp(min=0.0))) / self.Bc
        rate = self.Ac / self.Bc
        ones = self._valid(rate)
        neigh = (box_sum(rate, 1) - rate) / (box_sum(ones, 1) - ones).clamp(min=1.0)
        lcb = torch.where(rate > 3.0 * neigh, lcb, torch.zeros_like(lcb))
        return torch.maximum(ring, lcb)

    def update(self, counts):
        ones = self._valid(counts)
        valid = box_sum(ones, R_OUT) - box_sum(ones, R_IN)
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
        pool = lambda v: F.avg_pool2d(v, SUB, stride=SUB, ceil_mode=True)             # noqa: E731
        if self.mask is None:
            sub, slow = pool(counts), pool(self.As / self.Ws)                        # 子块速率 [B,2,h,w]
        else:                                                                        # 只在传感器像素上平均
            cover = pool(self._valid(counts)).clamp(min=1e-6)
            sub, slow = pool(counts) / cover, pool(self.As / self.Ws * self._valid(counts)) / cover
        h, w = sub.shape[-2:]
        patches = F.unfold(F.pad(sub, (2, 2, 2, 2), mode="replicate"), 5)           # [B,2*25,h*w]
        # 用排序取中位数（25 个取第 13 个）：确定性模式下 CUDA 的 median/kthvalue 会报错，sort 不会
        if self.mask is None:
            med = patches.view(sub.shape[0], 2, 25, h, w).sort(dim=2).values[:, :, 12]
        else:                                                                        # 完全在补零区的子块不参与中位数
            seen = F.unfold(F.pad((cover > 1e-3).to(sub).expand_as(sub), (2, 2, 2, 2), mode="replicate"), 5)
            seen = seen.view(sub.shape[0], 2, 25, h, w) > 0.5
            raw = patches.view(sub.shape[0], 2, 25, h, w)
            vals = torch.where(seen, raw, torch.full_like(raw, float("inf")))
            k = ((seen.sum(2, keepdim=True) - 1).clamp(min=0) // 2)
            med = vals.sort(dim=2).values.gather(2, k)[:, :, 0]
            med = torch.where(torch.isinf(med), torch.zeros_like(med), med)
        a_f = math.exp(-1.0 / 2.0)
        self.fast = med if self.fast is None else a_f * self.fast + (1 - a_f) * med
        off = (self.fast - slow).abs() > torch.maximum(0.5 * slow, torch.full_like(slow, self.eps_b))
        self.alarm = torch.where(off, (self.alarm if self.alarm is not None else torch.zeros_like(slow)) + 1,
                                 torch.zeros_like(slow))
        trig = self.alarm >= 2
        if self.mask is not None:                                                    # 补零区子块没有观测，不重启
            trig = trig & (cover > 1e-3)
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
    """把噪声校准挂到 net.enc1 的 TIF 神经元上（只替换该实例的 _compute；detach() 恢复原样）。

    每窗调用 set_window(x) 一次（在 net(x, ...) 之前）。
    dwell_ms：目标在 enc1 感受野上的停留时间（窗长 * (目标尺寸 + 感受野) / 速度，由标签统计按规则给出）。
    ref_rate：参照背景率 (ON, OFF)，见 REF_RATE。"""

    def __init__(self, net, mode, background, q99, time_bins, clip, dwell_ms, dt_ms=50.0, ref_rate=REF_RATE,
                 estimator=None):
        if mode not in MODES or background not in BACKGROUNDS:
            raise ValueError("mode 取 %s，background 取 %s" % (MODES, BACKGROUNDS))
        self.net, self.mode, self.background = net, mode, background
        self.q99, self.time_bins, self.clip = q99, int(time_bins), float(clip)
        self.block = net.enc1
        if not isinstance(self.block.norm, nn.Identity):
            raise ValueError("噪声校准插件要求 enc1 没有归一化层（电流对输入是线性的）")
        self.neuron = net.enc1.neuron
        if getattr(self.neuron, "adapt", False):
            raise ValueError("噪声校准插件只用于 adapt=False 的 TIF enc1")
        self.dwell_ms, self.dt_ms = float(dwell_ms), float(dt_ms)
        self.ref_rate = tuple(float(v) for v in ref_rate)
        self.params = None
        self.ref = None
        self.state = None                    # 本窗的 (dmu, rho, mu_ref, mu_eff)；None = 不校正
        self.oracle_b = None
        self.stats = {"inv_s_sum": 0.0, "n": 0, "rho_sum": 0.0}
        eps_b = 0.5 * self.ref_rate[0]
        self.estimator = estimator if estimator is not None else BackgroundEstimator(eps_b=eps_b)
        self._orig = self.neuron._compute
        self.neuron._compute = self._compute

    def detach(self):
        self.neuron._compute = self._orig

    def _kernels(self, dtype, device):
        """卷积核：均值用 W，方差用 W^2，整窗-时间箱协方差用 2 * W_整窗 * W_箱（逐通道增益另乘）。"""
        if self.params is None or self.params[0].dtype != dtype or self.params[0].device != device:
            with torch.no_grad():
                Wt = self.block.conv.weight.detach().to(dtype=dtype, device=device)
                g = self.block.gain.gain().detach().to(dtype=dtype, device=device).view(1, -1, 1, 1)
                T = self.time_bins
                cov_k = torch.stack([2.0 * Wt[:, p] * Wt[:, 2 + 2 * t + p] for t in range(T) for p in (0, 1)], 1)
            self.params = (Wt, Wt * Wt, cov_k, g)
        return self.params

    def moments(self, b):
        """背景率图 b [B,2,H,W] -> enc1 电流（增益之后）的背景均值与方差 [B,C,H,W]。"""
        Wt, W2, cov_k, g = self._kernels(b.dtype, b.device)
        m, v, c = input_moments(b, self.q99, self.time_bins, self.clip)
        stride = self.block.conv.stride
        mu = F.conv2d(m, Wt, stride=stride, padding=1) * g
        var = (F.conv2d(v, W2, stride=stride, padding=1) + F.conv2d(c, cov_k, stride=stride, padding=1)) * g * g
        return mu, var.clamp(min=0.0)

    def start_sequence(self, seq=None, height=None, width=None, n_windows=None):
        self.estimator.reset()
        self.state = None
        self.ref = None
        self.oracle_b = None
        if self.background == "oracle":
            p = next(self.net.parameters())
            self.oracle_b = oracle_background(seq, height, width, n_windows).to(device=p.device, dtype=torch.float64)

    def _ref_map(self, like):
        if self.ref is None or self.ref[0].shape[-2:] != like.shape[-2:]:
            b = torch.ones_like(like) * torch.tensor(self.ref_rate, dtype=like.dtype, device=like.device).view(1, 2, 1, 1)
            mu, var = self.moments(b)
            self.ref = (b, mu, var)
        return self.ref

    def set_window(self, x):
        if self.background == "oracle":
            b_hat = self.oracle_b
        else:
            b_hat = self.estimator.estimate()
            self.estimator.update(counts_from_input(x, self.q99).to(torch.float64))
        if b_hat is None:
            self.state = None
            return
        b_ref, mu_ref, var_ref = self._ref_map(b_hat)
        b_eff = torch.maximum(b_hat, b_ref)
        mu_eff, var_eff = self.moments(b_eff)
        rho = torch.maximum(var_eff / var_ref.clamp(min=1e-12), torch.ones_like(var_eff))
        rho = torch.where(var_ref > 1e-12, rho, torch.ones_like(rho))
        self.state = (mu_eff - mu_ref, rho, mu_ref, mu_eff)

    def _compute(self, current, prev, lam, k, theta, cap):
        if self.state is None:
            return self._orig(current, prev, lam, k, theta, cap)
        dmu, rho, mu_ref, mu_eff = (t.to(current.dtype) for t in self.state)
        if self.mode == "sub":
            s = torch.ones_like(rho)
        elif self.mode == "nocap":
            s = rho
        else:
            tau_ms = -self.dt_ms / torch.log(lam.clamp(1e-12, 1 - 1e-12))
            s = torch.minimum(rho, (self.dwell_ms / tau_ms).clamp(min=1.0).expand_as(rho))
        lam_eff = lam ** (1.0 / s)
        if self.mode == "ntt":
            I = mu_ref + torch.sqrt(s / rho) * (current - mu_eff)
        else:
            I = current - dmu
        with torch.no_grad():
            self.stats["inv_s_sum"] += float((1.0 / s).mean())
            self.stats["rho_sum"] += float(rho.mean())
            self.stats["n"] += 1
        z = I - k
        pre = (1.0 - lam_eff) * z if prev is None else lam_eff * prev + (1.0 - lam_eff) * z
        return self.neuron._report(pre, theta, cap)

    def sequence_summary(self):
        s = self.stats
        out = {"inv_s_mean": s["inv_s_sum"] / s["n"] if s["n"] else None,
               "rho_mean": s["rho_sum"] / s["n"] if s["n"] else None,
               "resets": int(self.estimator.resets)}
        self.stats = {"inv_s_sum": 0.0, "n": 0, "rho_sum": 0.0}
        return out
