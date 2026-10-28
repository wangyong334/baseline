"""S0：增益–时间协同调制的合成验证（Stream-SNN-Innovations.md 13.40～13.42；只用 numpy，不加载模型、不读数据集）。

在合成事件流上，用一个与 BIR 同形的积分神经元，检验"把噪声变成时间"的理论性质：
    固定参数时背景推满状态；只调增益时目标被压没；只调时间时背景积累更多；
    协同调制时目标电平不变、背景停在下限、涨落有界、只是延迟变长。

神经元（EMA 形式，与 BIR 相同，量化省略）：
    M <- clip(lam_eff * M + (1 - lam_eff) * target, 0, cap)
    各模式的 lam_eff 与 target：
        fixed       lam0,       I - k
        sub_blind   lam0,       I - k - D_blind      （adapt3 式：逐位置、盲更新的输入均值）
        sub         lam0,       I - k - D_hat        （好的背景估计，但只减均值、不拉长时间）
        gain        lam0,       g*I - k
        time        lam0**g,    I - k
        comod       lam0**g,    I - k - D_hat        （协同调制；等价于 输入×g、时间常数÷g）
        comod_oracle 同上，D 取真值
    g = k / (k + D)，D 为该神经元感受野内背景事件的期望驱动（每窗）。
输入：每像素每窗的事件计数 N（泊松），驱动 I = 3x3 盒式求和（w0 = 1）。

背景估计器（不读模型输出、只用过去的计数）：
    环形参照：21x21 盒减 9x9 盒（挖空中心，避开目标本体），偏差校正的指数平均，tau_s 窗；首窗即可用。
    中心分量：逐像素、每窗计数截到 c，指数平均（tau_c 窗）；只有当它显著高于环形参照时才接管
             （单侧检验的下置信界），用于热像素。
    b_hat = max(b_ring, LCB_center)；D_hat = 3x3 盒式求和(b_hat)。

场景：
    sweep     均匀背景密度扫描 + 3 个匀速运动目标（四种做法与估计器对比）
    hover     目标出现后静止不动（吸收时间）
    step      背景在中途突增、再突降（恢复）
    hot       少量热像素
    gradient  背景沿 x 线性变化
    misest    背景估计乘上偏差系数（估计误差的传递）
用法：python tools/comod_synthetic.py [--scenario all] [--trials 4] [--out log/comod_s0/s0.json]
"""
import argparse
import json
import math
import os

import numpy as np

H = W = 96
T = 160
K_COMP = 0.3        # 补偿项 k（每窗驱动单位）
TAU0 = 4.0          # 安静背景下的时间常数（窗）
CAP = 2.0           # 状态上界
THR = 0.39          # 判决阈值（约为安静背景下目标稳态电平的一半）
S_TGT = 0.12        # 目标强度：每像素每窗事件数（3x3 覆盖时驱动约 1.08）
BLOB = 5            # 目标边长（像素）
SPEED = 1.5         # 目标速度（像素/窗）
TAU_S, TAU_C, TAU_BLIND = 20.0, 100.0, 36.0
TRUNC_C, Z_LCB = 2, 2.0
ISOLATION = True    # 13.44：中心分量只在孤立点源处接管
LEVELS = (0.002, 0.01, 0.04, 0.11, 0.33)    # 每像素每窗背景率；感受野驱动 D = 9b
MODES = ("fixed", "sub_blind", "sub", "gain", "time", "comod", "comod_oracle")
# 第二轮（13.43）：按停留时间封顶的拉伸、两尺度（粗尺度门控细尺度）、沿运动方向积累的上限对照
MODES2 = ("fixed", "sub", "comod", "comod_cap", "ms_fixed", "ms", "comod_lagr")
# 第三轮（13.44）："先换时间、再换阈值"——时间拉伸到停留时间为止，超出部分用 CFAR 式增益保住虚警
MODES3 = ("fixed", "sub", "comod", "comod_cap", "ntt", "ms", "ms_ntt", "comod_lagr")
K_C, THR_C, CAP_C = 1.0, 1.0, 8.0     # 粗尺度（9x9 感受野）的补偿项、阈值、上界
BLOCK, SUB = 16, 4                    # 突变检测：16x16 区块、4x4 子块中位数


def box(a, r):
    """(2r+1)^2 盒式求和，周期边界。a: [..., H, W]。"""
    p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(r, r), (r, r)], mode="wrap")
    c = p.cumsum(-1).cumsum(-2)
    c = np.pad(c, [(0, 0)] * (a.ndim - 2) + [(1, 0), (1, 0)])
    n = 2 * r + 1
    return c[..., n:, n:] - c[..., :-n, n:] - c[..., n:, :-n] + c[..., :-n, :-n]


def blob_mask(cx, cy, half):
    """以 (cx, cy) 为中心、边长 2*half+1 的方块掩码（周期边界）。"""
    m = np.zeros((H, W), bool)
    xs = (np.arange(-half, half + 1) + int(round(cx))) % W
    ys = (np.arange(-half, half + 1) + int(round(cy))) % H
    m[np.ix_(ys, xs)] = True
    return m


class Scene(object):
    """一条合成流：背景率场 b[t]（H, W）与目标轨迹。"""

    def __init__(self, rate_fn, targets, rng):
        self.rate_fn, self.targets, self.rng = rate_fn, targets, rng

    def velocity_rows(self):
        """每一行的水平速度（像素/窗）：目标所在行带内取该目标的速度，其余为 0（沿运动方向积累的上限对照用）。"""
        v = np.zeros(H)
        for (x0, y0, vx, t_on, t_off) in self.targets:
            rows = (np.arange(-(BLOB + 4), BLOB + 5) + int(round(y0))) % H
            v[rows] = vx
        return v

    def positions(self, t):
        out = []
        for (x0, y0, vx, t_on, t_off) in self.targets:
            if t_on <= t < t_off:
                out.append(((x0 + vx * (t - t_on)) % W, y0))
        return out

    def window(self, t):
        b = self.rate_fn(t)
        lam = b.copy()
        tmask = np.zeros((H, W), bool)
        for (x, y) in self.positions(t):
            m = blob_mask(x, y, BLOB // 2)
            lam[m] += S_TGT
            tmask |= m
        return self.rng.poisson(lam).astype(np.float64), b, tmask


class Estimator(object):
    """背景估计器：环形参照 + 中心分量（见模块说明）。"""

    def __init__(self, bias=1.0, forget=False):
        self.a_s = math.exp(-1.0 / TAU_S)
        self.g_c = math.exp(-1.0 / TAU_C)
        self.As = self.Ws = None
        self.Ac = self.Bc = None
        self.bias = float(bias)
        self.forget = bool(forget)
        self.fast = None          # 每个区块子块中位速率的快速平均（2 窗）
        self.alarm = None         # 每个区块连续偏离的窗数
        self.resets = 0

    def estimate(self):
        """返回第 t 窗使用的 b_hat（只依赖第 t 窗之前的计数）；尚无数据时返回 None。"""
        if self.As is None:
            return None
        b_ring = self.As / self.Ws
        lcb = (self.Ac - Z_LCB * np.sqrt(np.maximum(self.Ac, 0.0))) / self.Bc
        if ISOLATION:
            # 中心接管只给"孤立的点源"（热像素是单像素缺陷；目标有空间范围，邻居同样偏高）
            rate = self.Ac / self.Bc
            neigh = (box(rate, 1) - rate) / 8.0
            lcb = np.where(rate > 3.0 * neigh, lcb, 0.0)
        return self.bias * np.maximum(b_ring, lcb)

    def update(self, n):
        ring = (box(n, 10) - box(n, 4)) / float(21 * 21 - 9 * 9)
        if self.As is None:
            self.As, self.Ws = (1 - self.a_s) * ring, np.full((H, W), 1 - self.a_s)
            self.Ac, self.Bc = np.minimum(n, TRUNC_C), 1.0
        else:
            self.As = self.a_s * self.As + (1 - self.a_s) * ring
            self.Ws = self.a_s * self.Ws + (1 - self.a_s)
            self.Ac = self.g_c * self.Ac + np.minimum(n, TRUNC_C)
            self.Bc = self.g_c * self.Bc + 1.0
        if self.forget:
            self._forget(n)

    def _forget(self, n):
        """区域突变时重启环形参照：子块计数的中位数对少数被目标占据的子块不敏感；
        只关心会改变 g 的变化（绝对门槛 eps = 0.1*k/9，即感受野驱动变化达到 k 的 10%）。"""
        nb, ns = H // BLOCK, BLOCK // SUB
        sub = n.reshape(nb, ns, SUB, nb, ns, SUB).sum(axis=(2, 5)) / float(SUB * SUB)   # [nb, ns, nb, ns]
        med = np.median(sub.transpose(0, 2, 1, 3).reshape(nb, nb, ns * ns), axis=-1)
        a_f = math.exp(-1.0 / 2.0)
        self.fast = med if self.fast is None else a_f * self.fast + (1 - a_f) * med
        slow = (self.As / self.Ws).reshape(nb, BLOCK, nb, BLOCK).mean(axis=(1, 3))
        eps = 0.1 * K_COMP / 9.0
        off = np.abs(self.fast - slow) > np.maximum(0.5 * slow, eps)
        self.alarm = np.where(off, (self.alarm if self.alarm is not None else 0) + 1, 0)
        trig = self.alarm >= 2
        if trig.any():
            full = np.kron(trig, np.ones((BLOCK, BLOCK), bool))
            restart = np.kron(self.fast, np.ones((BLOCK, BLOCK)))
            self.As = np.where(full, (1 - self.a_s) * restart, self.As)
            self.Ws = np.where(full, 1 - self.a_s, self.Ws)
            self.alarm = np.where(trig, 0, self.alarm)
            self.resets += int(trig.sum())


def dwell(footprint):
    """目标在一个感受野上的停留时间上限（窗）：(目标尺寸 + 感受野) / 速度；静止目标为无穷。"""
    return math.inf if SPEED == 0 else (BLOB + footprint) / abs(SPEED)


def shift_rows(a, vrow, t):
    """把每一行按该行速度平移到第 t+1 窗的位置（整数像素，累计取整）。"""
    out = a.copy()
    for v in np.unique(vrow):
        if v == 0:
            continue
        d = int(round(v * (t + 1)) - round(v * t))
        if d:
            rows = vrow == v
            out[rows] = np.roll(a[rows], d, axis=1)
    return out


def simulate(scene, modes=MODES, bias=1.0, forget=False):
    """对一条流运行全部模式，返回每窗的状态 M[mode] 列表、目标掩码、背景率。"""
    lam0 = math.exp(-1.0 / TAU0)
    est = Estimator(bias, forget)
    M = {m: np.zeros((H, W)) for m in modes}
    Mc = {m: np.zeros((H, W)) for m in ("ms_fixed", "ms", "ms_ntt") if m in modes}     # 粗尺度状态
    Mf = {m: np.zeros((H, W)) for m in ("ms_fixed", "ms", "ms_ntt") if m in modes}     # 细尺度状态
    vrow = scene.velocity_rows()
    gmin_f, gmin_c = TAU0 / dwell(3), TAU0 / dwell(9)    # 时间拉伸上限：1/g 不超过 停留时间/tau0
    blind_A, blind_W = np.zeros((H, W)), 0.0
    a_b = math.exp(-1.0 / TAU_BLIND)
    rec = {"M": {m: [] for m in modes}, "tmask": [], "b": [], "D_hat": []}
    for t in range(T):
        n, b, tmask = scene.window(t)
        I = box(n, 1)
        D_true = box(b, 1)
        b_hat = est.estimate()
        D_hat = box(b_hat, 1) if b_hat is not None else np.zeros((H, W))
        D_blind = blind_A / blind_W if blind_W > 0 else np.zeros((H, W))
        for m in modes:
            D = D_true if m == "comod_oracle" else D_hat
            g = K_COMP / (K_COMP + np.maximum(D, 0.0))
            if m == "ntt":
                gn = g
                gg = np.maximum(gn, gmin_f)
                gam = np.sqrt(gn / gg)                 # 被停留时间封顶时，按 CFAR 把剩余涨落压回上界
                lam = lam0 ** gg
                M[m] = np.clip(lam * M[m] + (1 - lam) * (gam * (I - D) - K_COMP), 0.0, CAP)
                rec["M"][m].append(M[m].copy())
                continue
            if m == "ms_ntt":
                Ic, Dc = box(n, 4), box(b_hat, 4) if b_hat is not None else np.zeros((H, W))
                gn_f, gn_c = g, K_C / (K_C + np.maximum(Dc, 0.0))
                gf, gc = np.maximum(gn_f, gmin_f), np.maximum(gn_c, gmin_c)
                lf, lc = lam0 ** gf, lam0 ** gc
                Mf[m] = np.clip(lf * Mf[m] + (1 - lf) * (np.sqrt(gn_f / gf) * (I - D) - K_COMP), 0.0, CAP)
                Mc[m] = np.clip(lc * Mc[m] + (1 - lc) * (np.sqrt(gn_c / gc) * (Ic - Dc) - K_C), 0.0, CAP_C)
                M[m] = np.where(Mc[m] > THR_C, Mf[m], 0.0)
                rec["M"][m].append(M[m].copy())
                continue
            if m in ("ms_fixed", "ms"):
                Ic, Dc = box(n, 4), box(b_hat, 4) if b_hat is not None else np.zeros((H, W))
                if m == "ms_fixed":
                    lf, tf, lc, tc = lam0, I - K_COMP, lam0, Ic - K_C
                else:
                    gc = K_C / (K_C + np.maximum(Dc, 0.0))
                    lf, tf = lam0 ** np.maximum(g, gmin_f), I - K_COMP - D
                    lc, tc = lam0 ** np.maximum(gc, gmin_c), Ic - K_C - Dc
                Mf[m] = np.clip(lf * Mf[m] + (1 - lf) * tf, 0.0, CAP)
                Mc[m] = np.clip(lc * Mc[m] + (1 - lc) * tc, 0.0, CAP_C)
                M[m] = np.where(Mc[m] > THR_C, Mf[m], 0.0)    # 粗尺度检出门控细尺度
                rec["M"][m].append(M[m].copy())
                continue
            if m == "comod_lagr":
                M[m] = shift_rows(M[m], vrow, t - 1) if t > 0 else M[m]
            if m == "fixed":
                lam, tgt = lam0, I - K_COMP
            elif m == "sub_blind":
                lam, tgt = lam0, I - K_COMP - np.maximum(D_blind, 0.0)
            elif m == "sub":
                lam, tgt = lam0, I - K_COMP - D
            elif m == "gain":
                lam, tgt = lam0, g * I - K_COMP
            elif m == "time":
                lam, tgt = lam0 ** g, I - K_COMP
            elif m == "comod_cap":
                lam, tgt = lam0 ** np.maximum(g, gmin_f), I - K_COMP - D
            else:
                lam, tgt = lam0 ** g, I - K_COMP - D
            M[m] = np.clip(lam * M[m] + (1 - lam) * tgt, 0.0, CAP)
            rec["M"][m].append(M[m].copy())
        est.update(n)
        blind_A = a_b * blind_A + (1 - a_b) * I
        blind_W = a_b * blind_W + (1 - a_b)
        rec["tmask"].append(tmask)
        rec["b"].append(b)
        rec["D_hat"].append(D_hat)
    return rec


def near_mask(scene, t, past=15, grow=4):
    """第 t 窗时目标当前及过去 past 窗位置的膨胀并集（用来把目标及尾迹从背景统计中排除）。"""
    m = np.zeros((H, W), bool)
    for j in range(past + 1):
        for (x, y) in scene.positions(t - j) if t - j >= 0 else []:
            m |= blob_mask(x, y, BLOB // 2 + grow)
    return m


def core_mask(scene, t):
    """目标核心像素（3x3，感受野被目标完全覆盖）。"""
    m = np.zeros((H, W), bool)
    for (x, y) in scene.positions(t):
        m |= blob_mask(x, y, 1)
    return m


def metrics(scene, rec, t0=20, hot=None):
    """虚警（远离目标的背景像素）、检出（目标核心像素）、目标平均电平、背景电平标准差。"""
    out = {}
    for m, Ms in rec["M"].items():
        fa_n = fa_d = pd_n = pd_d = 0
        lev, bg = [], []
        for t in range(t0, T):
            Mt = Ms[t] > THR
            bgm = ~near_mask(scene, t)
            if hot is not None:
                bgm &= ~hot
            core = core_mask(scene, t)
            fa_n += Mt[bgm].sum()
            fa_d += bgm.sum()
            pd_n += Mt[core].sum()
            pd_d += core.sum()
            if core.any():
                lev.append(Ms[t][core].mean())
            bg.append(Ms[t][bgm])
        bgv = np.concatenate(bg)
        out[m] = {"fa": fa_n / max(fa_d, 1), "pd": pd_n / max(pd_d, 1) if pd_d else None,
                  "target_level": float(np.mean(lev)) if lev else None,
                  "bg_mean": float(bgv.mean()), "bg_std": float(bgv.std())}
        if hot is not None:
            out[m]["fa_hot"] = float(np.mean([(Ms[t] > THR)[hot].mean() for t in range(t0, T)]))
    return out


def moving_targets():
    return [(10.0, 20.0, SPEED, 0, T), (50.0, 50.0, SPEED, 0, T), (80.0, 80.0, -SPEED, 0, T)]


def uniform(b):
    field = np.full((H, W), float(b))
    return lambda t: field


def run_sweep(trials, rng):
    rows = []
    for b in LEVELS:
        acc = []
        for _ in range(trials):
            sc = Scene(uniform(b), moving_targets(), rng)
            rec = simulate(sc)
            res = metrics(sc, rec)
            err = []
            for t in range(20, T):
                far = ~near_mask(sc, t)
                Dh, Dt = rec["D_hat"][t][far], 9 * b
                err.append((Dh.mean() / Dt - 1.0, Dh.std() / Dt))
            res["_est"] = {"bias": float(np.mean([e[0] for e in err])), "rel_std": float(np.mean([e[1] for e in err]))}
            # 运动目标尾迹对估计的影响：目标前方 / 当前位置的 D_hat 相对真值
            trail = []
            for t in range(20, T):
                for (x, y) in sc.positions(t):
                    trail.append(rec["D_hat"][t][int(round(y)) % H, int(round(x)) % W] / (9 * b) - 1.0)
            res["_est"]["at_target_bias"] = float(np.mean(trail))
            acc.append(res)
        rows.append({"b": b, "D": 9 * b, "res": average(acc)})
    return rows


def average(results):
    out = {}
    for key in results[0]:
        out[key] = {}
        for f in results[0][key]:
            vals = [r[key][f] for r in results if r[key][f] is not None]
            out[key][f] = float(np.mean(vals)) if vals else None
    return out


def run_hover(rng, b=0.11):
    """目标在第 20 窗出现后静止到结尾：检出率随出现后时长的变化（吸收）。"""
    sc = Scene(uniform(b), [(48.0, 48.0, 0.0, 20, T)], rng)
    rec = simulate(sc, modes=("fixed", "sub_blind", "comod", "comod_oracle"))
    curve = {}
    for m, Ms in rec["M"].items():
        core = blob_mask(48, 48, 1)
        curve[m] = [float((Ms[t] > THR)[core].mean()) for t in range(20, T)]
    bins = [(0, 10), (10, 40), (40, 80), (80, 140)]
    return {m: {"%d-%d" % bb: float(np.mean(c[bb[0]:bb[1]])) for bb in bins} for m, c in curve.items()}


def run_step(rng):
    """背景 0.002 -> 0.11（第 60 窗）-> 0.002（第 110 窗）：远离目标处的虚警随时间变化（恢复）。"""
    lo, hi = np.full((H, W), 0.002), np.full((H, W), 0.11)
    sc = Scene(lambda t: hi if 60 <= t < 110 else lo, moving_targets(), rng)
    rec = simulate(sc, modes=("fixed", "sub", "comod"))
    out = {}
    for m, Ms in rec["M"].items():
        fa = []
        for t in range(T):
            bgm = ~near_mask(sc, t)
            fa.append(float((Ms[t] > THR)[bgm].mean()))
        segs = [(55, 60), (60, 65), (65, 75), (75, 110), (110, 115), (115, 130)]
        out[m] = {"%d-%d" % s: float(np.mean(fa[s[0]:s[1]])) for s in segs}
    return out


def run_hot(rng, b=0.002, n_hot=40, rate=1.0):
    field = np.full((H, W), b)
    hot = np.zeros((H, W), bool)
    idx = rng.choice(H * W, n_hot, replace=False)
    hot.flat[idx] = True
    field[hot] = rate
    sc = Scene(lambda t: field, moving_targets(), rng)
    rec = simulate(sc, modes=("fixed", "sub_blind", "comod"))
    return metrics(sc, rec, hot=hot)


def run_gradient(rng):
    xs = np.linspace(0.002, 0.11, W)
    field = np.tile(xs, (H, 1))
    sc = Scene(lambda t: field, moving_targets(), rng)
    rec = simulate(sc, modes=("fixed", "comod", "comod_oracle"))
    out = {}
    for m, Ms in rec["M"].items():
        cols = {}
        for name, sl in (("sparse_side", slice(0, 32)), ("middle", slice(32, 64)), ("dense_side", slice(64, 96))):
            fa = np.mean([((Ms[t] > THR) & ~near_mask(sc, t))[:, sl].sum() / max((~near_mask(sc, t))[:, sl].sum(), 1)
                          for t in range(20, T)])
            cols[name] = float(fa)
        out[m] = cols
    return out


def run_misest(rng, trials):
    rows = []
    for bias in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5):
        row = {}
        for b in (0.11, 0.33):
            acc = []
            for _ in range(trials):
                sc = Scene(uniform(b), moving_targets(), rng)
                acc.append(metrics(sc, simulate(sc, modes=("comod",), bias=bias)))
            row["b=%g" % b] = average(acc)["comod"]
        rows.append({"bias": bias, "res": row})
    return rows


def run_round2(trials, rng, forget=True, modes=MODES2):
    """第二 / 三轮：在均匀背景扫描下的 虚警 / 检出（目标强度与速度由命令行给定）。"""
    rows = []
    for b in LEVELS:
        acc = []
        for _ in range(trials):
            sc = Scene(uniform(b), moving_targets(), rng)
            acc.append(metrics(sc, simulate(sc, modes=modes, forget=forget)))
        rows.append({"b": b, "D": 9 * b, "res": average(acc)})
    return rows


def run_step2(rng):
    """阶跃场景：估计器有 / 无突变遗忘时 comod 的远处虚警随时间变化。"""
    lo, hi = np.full((H, W), 0.002), np.full((H, W), 0.11)
    out = {}
    for forget in (False, True):
        sc = Scene(lambda t: hi if 60 <= t < 110 else lo, moving_targets(), rng)
        rec = simulate(sc, modes=("comod",), forget=forget)
        fa = [float((rec["M"]["comod"][t] > THR)[~near_mask(sc, t)].mean()) for t in range(T)]
        segs = [(55, 60), (60, 62), (62, 65), (65, 75), (75, 110), (110, 115), (115, 130)]
        out["forget" if forget else "no_forget"] = {"%d-%d" % g: float(np.mean(fa[g[0]:g[1]])) for g in segs}
    return out


def fmt(v, f="%.3f"):
    return "-" if v is None else f % v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default="all", choices=("all", "sweep", "hover", "step", "hot", "gradient", "misest", "round2", "step2", "round3"))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--out", default="log/comod_s0/s0.json")
    ap.add_argument("--target", type=float, default=S_TGT, help="目标强度（每像素每窗事件数）")
    ap.add_argument("--speed", type=float, default=SPEED, help="目标速度（像素/窗）；0 = 静止")
    ap.add_argument("--no-isolation", action="store_true", help="关闭中心分量的孤立点源门控（对照）")
    args = ap.parse_args()
    globals()["S_TGT"], globals()["SPEED"] = float(args.target), float(args.speed)
    globals()["ISOLATION"] = not args.no_isolation
    rng = np.random.default_rng(args.seed)
    report = {"config": {"k": K_COMP, "tau0": TAU0, "cap": CAP, "thr": THR, "s": S_TGT, "speed": SPEED,
                         "tau_s": TAU_S, "tau_c": TAU_C, "levels": LEVELS, "trials": args.trials}}
    want = lambda s: args.scenario in ("all", s)  # noqa: E731
    if want("sweep"):
        rows = run_sweep(args.trials, rng)
        report["sweep"] = rows
        print("== sweep：每格为 虚警 / 检出（目标核心）；D = 感受野背景驱动，g = k/(k+D)")
        print("%-6s %-6s " % ("b", "D") + " ".join("%-14s" % m for m in MODES))
        for r in rows:
            print("%-6g %-6.3g " % (r["b"], r["D"]) + " ".join(
                "%-14s" % ("%s/%s" % (fmt(r["res"][m]["fa"]), fmt(r["res"][m]["pd"], "%.2f"))) for m in MODES))
        print("   目标平均电平（comod / fixed）与背景电平标准差（comod / sub / fixed）；估计器：远处偏差、相对标准差、目标处偏差")
        for r in rows:
            e = r["res"]["_est"]
            print("   b=%-6g level %s / %s   bg_std %s / %s / %s   est bias %+.1f%% std %.1f%% at-target %+.1f%%" % (
                r["b"], fmt(r["res"]["comod"]["target_level"]), fmt(r["res"]["fixed"]["target_level"]),
                fmt(r["res"]["comod"]["bg_std"]), fmt(r["res"]["sub"]["bg_std"]), fmt(r["res"]["fixed"]["bg_std"]),
                100 * e["bias"], 100 * e["rel_std"], 100 * e["at_target_bias"]))
    if want("hover"):
        report["hover"] = run_hover(rng)
        print("== hover（b=0.11，目标静止）：出现后各时段的检出率")
        for m, v in report["hover"].items():
            print("   %-13s " % m + "  ".join("%s:%s" % (k, fmt(x, "%.2f")) for k, x in v.items()))
    if want("step"):
        report["step"] = run_step(rng)
        print("== step（0.002 -> 0.11 @60 -> 0.002 @110）：远处背景虚警")
        for m, v in report["step"].items():
            print("   %-6s " % m + "  ".join("%s:%s" % (k, fmt(x)) for k, x in v.items()))
    if want("hot"):
        report["hot"] = run_hot(rng)
        print("== hot（40 个热像素，速率 1/窗）：虚警（非热像素）/ 热像素处虚警 / 检出")
        for m, v in report["hot"].items():
            print("   %-10s fa %s  fa_hot %s  pd %s" % (m, fmt(v["fa"]), fmt(v["fa_hot"]), fmt(v["pd"], "%.2f")))
    if want("gradient"):
        report["gradient"] = run_gradient(rng)
        print("== gradient（b 沿 x 从 0.002 到 0.11）：远处背景虚警")
        for m, v in report["gradient"].items():
            print("   %-13s " % m + "  ".join("%s:%s" % (k, fmt(x)) for k, x in v.items()))
    if want("misest"):
        report["misest"] = run_misest(rng, max(1, args.trials - 1))
        print("== misest：背景估计乘偏差系数后的 虚警 / 检出 / 目标电平")
        for r in report["misest"]:
            print("   x%-5g " % r["bias"] + "  ".join("%s: %s/%s/%s" % (k, fmt(v["fa"]), fmt(v["pd"], "%.2f"), fmt(v["target_level"]))
                                                   for k, v in r["res"].items()))
    if args.scenario == "round2":
        rows = run_round2(args.trials, rng)
        report["round2"] = rows
        print("== round2（target=%g, speed=%g；停留上限 细 %.1f 窗 / 粗 %.1f 窗）：虚警 / 检出" % (S_TGT, SPEED, dwell(3), dwell(9)))
        print("%-6s " % "b" + " ".join("%-13s" % m for m in MODES2))
        for r in rows:
            print("%-6g " % r["b"] + " ".join("%-13s" % ("%s/%s" % (fmt(r["res"][m]["fa"]), fmt(r["res"][m]["pd"], "%.2f")))
                                             for m in MODES2))
    if args.scenario == "round3":
        rows = run_round2(args.trials, rng, modes=MODES3)
        report["round3"] = rows
        print("== round3（target=%g, speed=%g；停留上限 细 %.1f 窗 / 粗 %.1f 窗）：虚警 / 检出" % (S_TGT, SPEED, dwell(3), dwell(9)))
        print("%-6s " % "b" + " ".join("%-13s" % m for m in MODES3))
        for r in rows:
            print("%-6g " % r["b"] + " ".join("%-13s" % ("%s/%s" % (fmt(r["res"][m]["fa"]), fmt(r["res"][m]["pd"], "%.2f")))
                                             for m in MODES3))
    if args.scenario == "step2":
        report["step2"] = run_step2(rng)
        print("== step2（0.002 -> 0.11 @60 -> 0.002 @110）：comod 远处虚警，有 / 无突变遗忘")
        for m, v in report["step2"].items():
            print("   %-9s " % m + "  ".join("%s:%s" % (k, fmt(x)) for k, x in v.items()))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("S0 FINISHED:", args.out)


if __name__ == "__main__":
    main()
