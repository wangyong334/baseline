"""模块三 RCL v1.3（model/changepoint_layer.py）的测试（CPU、合成数据；方案 RCL_V1_PLAN_20261101.md §10）。

    T1 不动极限：前向（按来源归一化）与 RTS 平滑 = 单格两状态 HMM 的精确滤波 / 平滑
    T2 Codex 三格例子：B 的过去在场与精确值 0.0111 相差 <= 0.01（旧的两消息后向是 0.338）
    T3 随机单目标条带：卷积实现 = 逐来源循环的定义；平滑误差均值 <= 0.02，移入格的过度归因均值 <= 0.05
    T4 融合：L = 0 恒等、对 L 单调、不超过 max(m, p0)
    T5 提交协议：fixed 在第 D 窗提交、录像末 flush；adaptive 不提前把暂定 0 的事件判成背景；每个事件只提交一次
    T6 零输入零证据；编码 = 数据集 count_channels + normalize_counts；片段切分不变；换录像只清该流
    T7 梯度：三个损失有限、可反传，梯度到达证据卷积、mu、rho、p、转移核、归属头；与有限差分一致
    T8 融合的数值例子（Res/rcl_backward_check_20261101/fusion_output.txt 的 v1.3 final）
    另：目标轨迹占据（插值、编号 0 的目标）；配置接上 build_head（端到端训练 / 评估在 test_track_head）；
        第一阶段关口工具 tools/rcl_stage1_check 的计数
    v2：证据两种写法（linear = g - mu，log1p = v1）；product 提交 = 平滑在场 x 归属（逐事件按定义重算）；
        转移损失 L_dyn 能从标签占据学出运动（位移核集中到真实位移、存活升高）；流内 dyn_sum 与逐窗计算一致、换录像不算；
        v2 配置只在三处诊断点、轮数、输出目录上与 v1 不同
"""
import math
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.stream_windows import StreamSequence, count_channels, normalize_counts  # noqa: E402
from model.changepoint_layer import (STATUS_CONFIRM, STATUS_DEADLINE, STATUS_FLUSH, STATUS_REJECT,  # noqa: E402
                                     ChangepointLayer, fuse, occupancy_boxes, occupancy_maps)

H, W, FEAT, BINS = 16, 24, 4, 5
Q99 = np.linspace(0.8, 1.6, 12).astype(np.float32)
POS_W = 30.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_layer(seed=0, double=False, **kwargs):
    torch.manual_seed(seed)
    params = dict(feat_dim=FEAT, height=H, width=W, q99=Q99, input_clip=3.0, time_bins=BINS, pos_weight=POS_W,
                  threshold=0.9, defer_max=4, rings=(1, 2), cell=4)
    params.update(kwargs)
    layer = ChangepointLayer(**params)
    return layer.double() if double else layer


def set_kernel(layer, weights):
    """转移核只放在给定位移上：{(dy, dx): 概率}。"""
    with torch.no_grad():
        logits = torch.full_like(layer.pi_logit, -1e4)
        for off, w in weights.items():
            logits[layer.offsets.index(off)] = math.log(w)
        layer.pi_logit.copy_(logits)


def set_rates(layer, rho, p):
    with torch.no_grad():
        layer.rho_raw.fill_(math.log(rho / (1.0 - rho)))
        layer.p_raw.fill_(math.log(p / (1.0 - p)))


def density_evidence(layer, w1=0.5, w2=0.3):
    """证据 = 局部事件密度（正权重），让在场场随事件多少变化。"""
    with torch.no_grad():
        layer.ev1.weight.fill_(w1)
        layer.ev2.weight.fill_(w2)


def make_events(n_windows, batch=1, seed=0, per_window=12, target=True, start=(4.0, 6.0), velocity=(1.5, 0.5)):
    """随机背景事件 + 每条流一个匀速运动的小目标（每窗 6 个事件）。
    返回 numpy 字典（t / b / x / y / p / t_local / bin / label / tid）与 z0、feat（目标事件 z0 偏高）。"""
    rng = np.random.RandomState(seed)
    cols = {key: [] for key in ("t", "b", "x", "y", "p", "t_local", "bin", "label", "tid")}
    z0 = []
    for t in range(n_windows):
        for b in range(batch):
            n_bg = rng.randint(0, per_window + 1)
            xs, ys = rng.randint(0, W, n_bg), rng.randint(0, H, n_bg)
            items = [(x, y, 0.0, rng.normal(-1.0, 3.0)) for x, y in zip(xs, ys)]
            if target:
                cx = (start[0] + velocity[0] * t + 5 * b) % W
                cy = (start[1] + velocity[1] * t + 3 * b) % H
                for _ in range(6):
                    x = int(np.clip(round(cx + rng.randint(-1, 2)), 0, W - 1))
                    y = int(np.clip(round(cy + rng.randint(-1, 2)), 0, H - 1))
                    items.append((x, y, 1.0, rng.normal(5.0, 3.0)))
            for x, y, lab, z in items:
                tl = rng.uniform(0.0, 1.0)
                cols["t"].append(t)
                cols["b"].append(b)
                cols["x"].append(x)
                cols["y"].append(y)
                cols["p"].append(float(rng.randint(0, 2)))
                cols["t_local"].append(tl)
                cols["bin"].append(int(tl * BINS))
                cols["label"].append(lab)
                cols["tid"].append(1.0 + b if lab else 0.0)
                z0.append(z)
    ev = {key: np.asarray(v, dtype=np.int64 if key in ("t", "b", "x", "y", "bin") else np.float32)
          for key, v in cols.items()}
    feat = rng.normal(0.0, 1.0, size=(len(z0), FEAT)).astype(np.float32)
    return ev, np.asarray(z0, dtype=np.float32), feat


def occ_from_events(ev, n_windows, batch, cell=4):
    """测试用的在场标签：目标事件所在的格（不插值，够用）。"""
    occ = np.zeros((n_windows, batch, H // cell, W // cell), dtype=np.float32)
    sel = ev["label"] > 0.5
    occ[ev["t"][sel], ev["b"][sel], ev["y"][sel] // cell, ev["x"][sel] // cell] = 1.0
    return occ


def chunk_slice(ev, z0, feat, lo, hi, dtype=torch.float32):
    sel = (ev["t"] >= lo) & (ev["t"] < hi)
    idx = np.nonzero(sel)[0]
    out = {key: torch.from_numpy(ev[key][sel].copy()) for key in ("t", "b", "x", "y", "p", "t_local", "bin")}
    out["t"] = out["t"] - lo
    for key in ("p", "t_local"):
        out[key] = out[key].to(dtype)
    labels = torch.from_numpy(ev["label"][sel].copy()).to(dtype)
    return out, torch.from_numpy(z0[sel].copy()).to(dtype), torch.from_numpy(feat[sel].copy()).to(dtype), labels, \
        torch.from_numpy(idx.astype(np.int64))


def run_eval(layer, ev, z0, feat, n_windows, cuts, batch=1, keep=None):
    """按 cuts 切片段连续推理（评估：没有标签），返回 {key: (z, lag, status)}（含录像末 flush）。"""
    state = layer.init_state(batch, torch.device("cpu"))
    emitted = {}

    def collect(d):
        if d["delayed_z"] is None:
            return
        for k, z, lag, st in zip(d["delayed_key"].tolist(), d["delayed_z"].tolist(), d["delayed_lag"].tolist(),
                                 d["delayed_status"].tolist()):
            if k in emitted:
                raise AssertionError("事件 %d 提交了两次" % k)
            emitted[k] = (z, lag, st)

    bounds = [0] + list(cuts) + [n_windows]
    with torch.no_grad():
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            events, z, f, _, idx = chunk_slice(ev, z0, feat, lo, hi, dtype=layer.p_raw.dtype)
            kp = None if keep is None else torch.from_numpy(keep[lo:hi]).to(layer.p_raw.dtype)
            _, state, aux = layer.forward_chunk(events, z, f, state, hi - lo, keep=kp, keys=idx)
            collect(aux)
        collect(layer.flush(state))
    return emitted


# ---------------------------------------------------------------------------
# numpy 参照：单目标条带的精确 HMM、逐来源循环的前向（Res/rcl_backward_check_20261101）
# ---------------------------------------------------------------------------

OFFS = (-1, 0, 1)


def exact_smoother(Lam, rho, p, pi, prior):
    T, N = Lam.shape
    S = N + 1
    A = np.zeros((S, S))
    A[0, 0] = 1 - rho * N
    A[0, 1:] = rho
    for i in range(N):
        A[i + 1, 0] += 1 - p
        for o, w in zip(OFFS, pi):
            j = i + o
            if 0 <= j < N:
                A[i + 1, j + 1] += p * w
            else:
                A[i + 1, 0] += p * w
    lik = np.concatenate([np.ones((T, 1)), Lam], 1)
    alpha = np.zeros((T, S))
    a = prior * lik[0]
    alpha[0] = a / a.sum()
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * lik[t]
        alpha[t] = a / a.sum()
    beta = np.ones((T, S))
    for t in range(T - 2, -1, -1):
        b = A @ (lik[t + 1] * beta[t + 1])
        beta[t] = b / b.sum()
    g = alpha * beta
    g /= g.sum(1, keepdims=True)
    return g[:, 1:]


def source_forward_loop(Lam, rho, p, pi, P0, eps=1e-6):
    """逐来源的定义：每个来源格一次归一化（不在 / 出生 / 消亡 / 各位移，出画布 = 似然比 1）。"""
    T, N = Lam.shape
    P, pred, prev = np.zeros((T, N)), np.zeros((T, N)), P0.copy()
    for t in range(T):
        L = Lam[t]
        post, pr = np.zeros(N), np.zeros(N)
        for d in range(N):
            a, q = 1 - prev[d], prev[d]
            terms = [(None, a * (1 - rho), 1.0), (d, a * rho, L[d]), (None, q * (1 - p), 1.0)]
            for o, wo in zip(OFFS, pi):
                x = d + o
                terms.append((x if 0 <= x < N else None, q * p * wo, L[x] if 0 <= x < N else 1.0))
            Z = sum(w * lv for _, w, lv in terms)
            for x, w, lv in terms:
                if x is not None:
                    pr[x] += w
                    post[x] += w * lv / Z
        pred[t], P[t] = np.clip(pr, eps, 1 - eps), np.clip(post, eps, 1 - eps)
        prev = P[t]
    return P, pred


def layer_strip(layer, Lam):
    """在 1 x N 的格上跑本层的前向 + RTS（双精度），返回 (过滤, 平滑) [T, N]。"""
    T, N = Lam.shape
    with torch.no_grad():
        k = layer.kernel()
        P = torch.zeros(1, 1, N, dtype=torch.float64)
        filt, pred = [], []
        for t in range(T):
            P, pr = layer.forward_step(P, torch.from_numpy(np.log(Lam[t]))[None, None], k)
            filt.append(P)
            pred.append(pr)
        S = [None] * T
        S[-1] = filt[-1]
        for t in range(T - 2, -1, -1):
            S[t] = layer.rts_step(filt[t], pred[t + 1], S[t + 1], k)
    return (np.stack([f[0, 0].numpy() for f in filt]), np.stack([s[0, 0].numpy() for s in S]))


class RecursionTests(unittest.TestCase):
    def test_no_motion_matches_exact_hmm(self):
        """T1：转移核只有"不动"时，每格是独立的两状态 HMM，前向与 RTS 应当逐位等于精确滤波与平滑。"""
        layer = make_layer(double=True, rings=(1,))
        set_kernel(layer, {(0, 0): 1.0})
        rho, p = 0.01, 0.9
        set_rates(layer, rho, p)
        rng = np.random.RandomState(1)
        T, hc, wc = 30, 3, 4
        ell = rng.normal(0.0, 1.5, size=(T, 1, hc, wc))
        k = layer.kernel()
        P = torch.zeros(1, hc, wc, dtype=torch.float64)
        filt, pred = [], []
        for t in range(T):
            P, pr = layer.forward_step(P, torch.from_numpy(ell[t]), k)
            filt.append(P)
            pred.append(pr)
        S = [None] * T
        S[-1] = filt[-1]
        for t in range(T - 2, -1, -1):
            S[t] = layer.rts_step(filt[t], pred[t + 1], S[t + 1], k)
        A = np.array([[1 - rho, rho], [1 - p, p]])
        lam = np.exp(ell[:, 0])
        for i in range(hc):
            for j in range(wc):
                f, prev = np.zeros((T, 2)), np.array([1.0, 0.0])
                for t in range(T):
                    a = (prev @ A) * np.array([1.0, lam[t, i, j]])
                    f[t] = a / a.sum()
                    prev = f[t]
                beta, sm = np.ones(2), np.zeros(T)
                sm[-1] = f[-1, 1]
                for t in range(T - 2, -1, -1):
                    beta = A @ (np.array([1.0, lam[t + 1, i, j]]) * beta)
                    beta /= beta.sum()
                    g = f[t] * beta
                    sm[t] = g[1] / g.sum()
                np.testing.assert_allclose([float(x[0, i, j]) for x in filt], f[:, 1], atol=1e-9)
                np.testing.assert_allclose([float(x[0, i, j]) for x in S], sm, atol=1e-9)

    def test_codex_toy(self):
        """T2：A .89、B .01、没有目标 .10；不生不灭，不动 .5、右移 .5；下一窗 B 的似然比 100。
        精确：P(B 过去在场) = .01 x 50.5 / (.01 x 50.5 + .89 x 50.5 + .10) = 0.0111。"""
        layer = make_layer(double=True, rings=(1,))
        set_kernel(layer, {(0, 0): 0.5, (0, 1): 0.5})
        set_rates(layer, 1e-12, 1.0 - 1e-9)
        k = layer.kernel()
        P = torch.zeros(1, 1, 3, dtype=torch.float64)
        P[0, 0, 0], P[0, 0, 1] = 0.89, 0.01
        ell = torch.tensor([[[0.0, math.log(100.0), 0.0]]], dtype=torch.float64)
        P1, pred1 = layer.forward_step(P, ell, k)
        S0 = layer.rts_step(P, pred1, P1, k)
        self.assertLess(abs(float(S0[0, 0, 1]) - 0.0111), 0.01)
        self.assertLess(abs(float(S0[0, 0, 0]) - 0.9867), 0.02)

    def test_random_strips(self):
        """T3：单目标条带（12 格、14 窗、200 次）：本层（卷积写法）= 逐来源循环的定义；
        平滑在场与精确单目标 HMM 的平均误差 <= 0.02，"上一窗不在、目标从邻格移入"的格的过度归因均值 <= 0.05。"""
        rng = np.random.default_rng(0)
        N, T, rho, p, pi = 12, 14, 0.005, 0.95, (0.25, 0.5, 0.25)
        layer = make_layer(double=True, rings=(1,))
        set_kernel(layer, {(0, -1): pi[0], (0, 0): pi[1], (0, 1): pi[2]})
        set_rates(layer, rho, p)
        errs, inflow = [], []
        for trial in range(200):
            x, state = np.zeros(T, dtype=int), 0
            for t in range(T):
                if state == 0:
                    if rng.random() < rho * N * 8:
                        state = int(rng.integers(1, N + 1))
                elif rng.random() > p:
                    state = 0
                else:
                    o = rng.choice(OFFS, p=pi)
                    state = state + o if 1 <= state + o <= N else 0
                x[t] = state
            Lam = np.exp(rng.normal(np.log(0.6), 0.6, size=(T, N)))
            for t in range(T):
                if x[t] > 0:
                    Lam[t, x[t] - 1] = np.exp(rng.normal(np.log(8.0), 1.0))
            filt, sm = layer_strip(layer, Lam)
            if trial < 5:
                ref, _ = source_forward_loop(Lam, rho, p, pi, np.zeros(N))
                np.testing.assert_allclose(filt, ref, atol=1e-9)
            prior = np.zeros(N + 1)
            prior[0], prior[1:] = 1 - rho * N, rho
            ex = exact_smoother(Lam, rho, p, pi, prior)
            errs.append(np.abs(sm - ex).ravel())
            for u in range(T - 1):
                c = x[u + 1]
                if c > 0 and x[u] > 0 and x[u] != c:
                    inflow.append(sm[u, c - 1] - ex[u, c - 1])
        self.assertLessEqual(float(np.concatenate(errs).mean()), 0.02)
        self.assertGreater(len(inflow), 100)
        self.assertLessEqual(float(np.mean(inflow)), 0.05)


class FusionTests(unittest.TestCase):
    def test_identity_monotone_cap(self):
        """T4：L = 0 时输出 = p0；对 L 单调不减；不超过 max(m, p0)。"""
        g = torch.Generator().manual_seed(0)
        n = 500
        lp0 = torch.randn(n, generator=g, dtype=torch.float64) * 4.0
        m = torch.rand(n, generator=g, dtype=torch.float64)
        q = torch.rand(n, generator=g, dtype=torch.float64)
        torch.testing.assert_allclose(fuse(lp0, m, q, torch.zeros(n, dtype=torch.float64)), lp0, atol=1e-9, rtol=0)
        prev = None
        for L in torch.linspace(-20.0, 20.0, 81, dtype=torch.float64):
            z = fuse(lp0, m, q, L.expand(n))
            if prev is not None:
                self.assertTrue(bool((z >= prev - 1e-9).all()))
            top = torch.maximum(m, torch.sigmoid(lp0))
            self.assertTrue(bool((torch.sigmoid(z) <= top + 1e-6).all()))
            prev = z

    def test_reference_numbers(self):
        """T8：方案 §4.4 的数值表（v1.3 final = min(max(m, p0), Bayes)），Lam = 10 / 100 / 无穷。"""
        table = {(0.10, 0.50): {0.10: (0.100, 0.100, 0.100), 0.50: (0.217, 0.246, 0.250),
                                0.85: (0.338, 0.444, 0.460), 0.98: (0.486, 0.791, 0.850)},
                 (0.20, 0.40): {0.10: (0.200, 0.200, 0.200), 0.50: (0.435, 0.493, 0.500),
                                0.85: (0.579, 0.714, 0.733), 0.98: (0.691, 0.916, 0.950)}}
        for (p0, q), rows in table.items():
            for m, expect in rows.items():
                for lam, want in zip((10.0, 100.0, math.exp(30.0)), expect):
                    z = fuse(torch.tensor([math.log(p0 / (1 - p0))], dtype=torch.float64),
                             torch.tensor([m], dtype=torch.float64), torch.tensor([q], dtype=torch.float64),
                             torch.tensor([math.log(lam)], dtype=torch.float64))
                    self.assertAlmostEqual(float(torch.sigmoid(z)), want, delta=6e-4, msg=(p0, q, m, lam))


class ProtocolTests(unittest.TestCase):
    def test_fixed_commit(self):
        """T5（固定滞后）：每个事件恰好提交一次；第 D 窗提交（状态 deadline、lag = D），录像末不足 D 窗的 flush。"""
        layer = make_layer(defer_max=4)
        density_evidence(layer)
        ev, z0, feat = make_events(12, seed=1)
        out = run_eval(layer, ev, z0, feat, 12, cuts=(5, 9))
        self.assertEqual(set(out), set(range(len(z0))))
        for key, (z, lag, st) in out.items():
            t = int(ev["t"][key])
            if t + 4 <= 11:
                self.assertEqual((lag, st), (4, STATUS_DEADLINE))
            else:
                self.assertEqual((lag, st), (11 - t, STATUS_FLUSH))
            self.assertTrue(math.isfinite(z))

    def test_adaptive_commit(self):
        """T5（自适应）：P >= 1 - alpha 提交 1；只有暂定为 1 的事件可以因 P <= alpha 提前提交 0；其余等到第 D 窗。"""
        layer = make_layer(defer_max=4, commit="adaptive", defer_alpha=0.05)
        density_evidence(layer, w1=0.8, w2=0.5)
        with torch.no_grad():
            layer.member[2].bias.fill_(2.0)
        ev, z0, feat = make_events(16, seed=2)
        z0 = np.where(np.arange(z0.size) % 7 == 0, 8.0, z0).astype(np.float32)     # some commit at arrival
        out = run_eval(layer, ev, z0, feat, 16, cuts=(6, 11))
        self.assertEqual(set(out), set(range(len(z0))))
        hi = math.log(0.95 / 0.05) + math.log(POS_W)
        lo = math.log(0.05 / 0.95) + math.log(POS_W)
        prov = z0 - math.log(POS_W) >= layer.tau_logit
        seen = set()
        for key, (z, lag, st) in out.items():
            seen.add(st)
            if st == STATUS_CONFIRM:
                self.assertGreaterEqual(z, hi - 1e-4)
                if lag == 0:
                    self.assertAlmostEqual(z, float(z0[key]), places=5)
            elif st == STATUS_REJECT:
                self.assertTrue(bool(prov[key]), "暂定为 0 的事件不能提前判成背景")
                self.assertLessEqual(z, lo + 1e-4)
                self.assertGreaterEqual(lag, 1)
            elif st == STATUS_DEADLINE:
                self.assertEqual(lag, 4)
            else:
                self.assertEqual(st, STATUS_FLUSH)
        self.assertIn(STATUS_CONFIRM, seen)
        self.assertIn(STATUS_DEADLINE, seen)


class InputAndStateTests(unittest.TestCase):
    def test_zero_input_zero_evidence(self):
        """T6：零输入 -> g = 0 -> 证据 = -mu（沉默代价），与证据卷积的权重无关。"""
        layer = make_layer()
        ell = layer.evidence(torch.zeros(2, 12, H, W))
        torch.testing.assert_allclose(ell, torch.full_like(ell, -float(layer.mu())))

    def test_encode_matches_dataset(self):
        """T6：本层的编码与骨干输入（dataset/stream_windows.count_channels + normalize_counts）逐位相同。"""
        layer = make_layer()
        rng = np.random.RandomState(3)
        n = 400
        b = rng.randint(0, 2, n)
        x, y = rng.randint(0, W, n), rng.randint(0, H, n)
        p = rng.randint(0, 2, n)
        tb = rng.randint(0, BINS, n)
        x[:50], y[:50] = 3, 5                                   # many events on one pixel (saturation)
        ev = {"b": torch.from_numpy(b), "x": torch.from_numpy(x), "y": torch.from_numpy(y),
              "p": torch.from_numpy(p.astype(np.float32)), "bin": torch.from_numpy(tb)}
        got = layer.encode(ev, 2).numpy()
        for s in range(2):
            sel = b == s
            ref = normalize_counts(count_channels(x[sel], y[sel], p[sel], tb[sel], BINS, H, W), Q99, 3.0)
            np.testing.assert_allclose(got[s], ref, atol=1e-6)

    def test_chunk_split_invariance(self):
        """T6：一次处理 12 窗与切成 3 段（状态跨段延续）的提交结果相同；训练时三个损失之和也相同。"""
        layer = make_layer(defer_max=4)
        density_evidence(layer)
        ev, z0, feat = make_events(12, batch=2, seed=4)
        one = run_eval(layer, ev, z0, feat, 12, cuts=(), batch=2)
        three = run_eval(layer, ev, z0, feat, 12, cuts=(4, 7), batch=2)
        self.assertEqual(set(one), set(three))
        for key in one:
            self.assertAlmostEqual(one[key][0], three[key][0], places=4)
            self.assertEqual(one[key][1:], three[key][1:])
        occ = torch.from_numpy(occ_from_events(ev, 12, 2))
        totals = []
        for cuts in ((), (4, 7)):
            state, tot = None, {"exist": 0.0, "mem": 0.0, "evt": 0.0, "evt_n": 0}
            bounds = [0] + list(cuts) + [12]
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                events, z, f, labels, _ = chunk_slice(ev, z0, feat, lo, hi)
                events["occ"] = occ[lo:hi]
                keep = torch.ones(hi - lo, 2)
                if lo == 0:
                    keep[0] = 0
                _, state, aux = layer.forward_chunk(events, z, f, state, hi - lo, keep=keep, labels=labels,
                                                    keys=labels)
                tot["exist"] += float(aux["exist_sum"])
                tot["mem"] += float(aux["mem_sum"])
                tot["evt"] += float(aux["evt_sum"])
                tot["evt_n"] += aux["evt_n"]
                state = layer.detach_state(state)
            totals.append(tot)
        for key in ("exist", "mem", "evt"):
            self.assertAlmostEqual(totals[0][key], totals[1][key], delta=1e-3 * max(1.0, abs(totals[0][key])))
        self.assertEqual(totals[0]["evt_n"], totals[1]["evt_n"])

    def test_reset_clears_only_that_stream(self):
        """T6：第 6 窗流 1 换录像（keep = 0）：流 1 的挂起事件以 flush 提交（lag < D），之后的输出与从第 6 窗
        重新开始的单独一条流相同；流 0 的输出不受影响。"""
        D = 4
        layer = make_layer(defer_max=D)
        density_evidence(layer)
        ev, z0, feat = make_events(12, batch=2, seed=5)
        keep = np.ones((12, 2), dtype=np.float32)
        keep[6, 1] = 0.0
        reset = run_eval(layer, ev, z0, feat, 12, cuts=(5,), batch=2, keep=keep)
        plain = run_eval(layer, ev, z0, feat, 12, cuts=(5,), batch=2)
        for key in np.nonzero(ev["b"] == 0)[0]:                      # stream 0 untouched
            self.assertAlmostEqual(reset[key][0], plain[key][0], places=5)
            self.assertEqual(reset[key][1:], plain[key][1:])
        for key in np.nonzero((ev["b"] == 1) & (ev["t"] < 6) & (ev["t"] + D > 6))[0]:
            self.assertEqual(reset[key][2], STATUS_FLUSH)               # flushed at the reset
            self.assertLess(reset[key][1], D)
        # stream 1 after the reset = a fresh single stream started at window 6
        sel = np.nonzero((ev["b"] == 1) & (ev["t"] >= 6))[0]
        sub = {key: ev[key][sel].copy() for key in ev}
        sub["t"] = sub["t"] - 6
        sub["b"] = np.zeros_like(sub["b"])
        fresh = run_eval(layer, sub, z0[sel], feat[sel], 6, cuts=(), batch=1)
        for j, key in enumerate(sel):
            self.assertAlmostEqual(reset[key][0], fresh[j][0], places=4)
            self.assertEqual(reset[key][1:], fresh[j][1:])


class GradientTests(unittest.TestCase):
    def _losses(self, layer, ev, z0, feat, occ, n_windows):
        events, z, f, labels, _ = chunk_slice(ev, z0, feat, 0, n_windows, dtype=layer.p_raw.dtype)
        events["occ"] = occ.to(layer.p_raw.dtype)
        keep = torch.ones(n_windows, 1, dtype=layer.p_raw.dtype)
        _, _, aux = layer.forward_chunk(events, z, f, None, n_windows, keep=keep, labels=labels, keys=labels)
        return aux

    def test_losses_backward(self):
        """T7：三个损失有限、可反传；梯度到达证据卷积、mu、rho、p、转移核、归属头。"""
        layer = make_layer(defer_max=3)
        density_evidence(layer)
        with torch.no_grad():
            layer.ev1.weight.add_(0.05 * torch.randn_like(layer.ev1.weight))
        ev, z0, feat = make_events(10, seed=6)
        occ = torch.from_numpy(occ_from_events(ev, 10, 1))
        aux = self._losses(layer, ev, z0, feat, occ, 10)
        self.assertGreater(aux["exist_n"], 0)
        self.assertGreater(aux["mem_n"], 0)
        self.assertGreater(aux["evt_n"], 0)
        loss = aux["exist_sum"] / aux["exist_n"] + aux["mem_sum"] / aux["mem_n"] + aux["evt_sum"] / aux["evt_n"]
        self.assertTrue(math.isfinite(float(loss)))
        loss.backward()
        for name, prm in layer.named_parameters():
            self.assertIsNotNone(prm.grad, name)
            self.assertTrue(bool(torch.isfinite(prm.grad).all()), name)
            self.assertGreater(float(prm.grad.abs().sum()), 0.0, name)

    def test_finite_difference(self):
        """T7：双精度下与中心差分一致。在场参数只看 L_pres（L_evt 里在场是 detach 的）；归属头看 L_mem + L_evt。"""
        layer = make_layer(double=True, defer_max=3)
        density_evidence(layer)
        with torch.no_grad():
            layer.pi_logit.copy_(0.3 * torch.randn_like(layer.pi_logit))
            layer.ev1.weight.add_(0.05 * torch.randn_like(layer.ev1.weight))
        ev, z0, feat = make_events(8, seed=7)
        occ = torch.from_numpy(occ_from_events(ev, 8, 1))

        def value(kind):
            aux = self._losses(layer, ev, z0, feat, occ, 8)
            return aux["exist_sum"] if kind == "pres" else aux["mem_sum"] + aux["evt_sum"]

        checks = [("pres", layer.rho_raw, ()), ("pres", layer.p_raw, ()), ("pres", layer.mu_raw, ()),
                  ("pres", layer.pi_logit, (0,)), ("pres", layer.pi_logit, (3,)), ("pres", layer.ev2.weight, (0, 2, 1, 1)),
                  ("mem", layer.member[0].weight, (1, 2)), ("mem", layer.member[2].bias, (0,))]
        for kind, prm, index in checks:
            layer.zero_grad()
            value(kind).backward()
            grad = float(prm.grad[index]) if index else float(prm.grad)
            h = 1e-6
            with torch.no_grad():
                base = float(prm[index]) if index else float(prm)
                prm[index] = base + h
                up = float(value(kind))
                prm[index] = base - h
                down = float(value(kind))
                prm[index] = base
            fd = (up - down) / (2 * h)
            self.assertAlmostEqual(grad, fd, delta=1e-5 * max(1.0, abs(fd)), msg=(kind, index))


class LabelAndConfigTests(unittest.TestCase):
    def test_occupancy_boxes(self):
        """目标轨迹占据：有事件的窗取外接框；中间没有事件的窗线性插值，不往外推；编号 0 的目标（EV-Flying）也算；
        背景事件不算。"""
        rows = []                       # (t_ms, x, y, label, tid)
        rows += [(10, 4, 4, 1, 0), (12, 7, 5, 1, 0)]          # target 0, window 0: x 4..7, y 4..5
        rows += [(160, 16, 12, 1, 0), (170, 19, 13, 1, 0)]     # target 0, window 3: x 16..19, y 12..13
        rows += [(60, 20, 2, 1, 2)]                           # target 2, window 1 only
        rows += [(30, 0, 0, 0, 0), (110, 23, 15, 0, 0)]       # background
        t = np.array([r[0] for r in rows], dtype=np.int64)
        order = np.argsort(t, kind="stable").astype(np.int64)
        bounds = np.searchsorted(t[order] // 50, np.arange(7), side="left").astype(np.int64)
        seq = StreamSequence("toy", np.array([r[1] for r in rows]), np.array([r[2] for r in rows]), t,
                             np.zeros(len(rows), dtype=np.int8), np.array([r[3] for r in rows], dtype=np.float32),
                             np.array([r[4] for r in rows], dtype=np.float64), np.zeros(len(rows), dtype=np.int64),
                             np.zeros(len(rows), dtype=np.float32), order, bounds)
        cells = occupancy_boxes(seq, 4)
        self.assertEqual(len(cells), 6)
        # rows are [cy0, cy1, cx0, cx1]
        self.assertEqual(sorted(map(tuple, cells[0].tolist())), [(1, 1, 1, 1)])
        # window 1: target 2 + target 0 interpolated (w = 1/3): x 8..11, y 6.67..7.67
        self.assertEqual(sorted(map(tuple, cells[1].tolist())), [(0, 0, 5, 5), (1, 1, 2, 2)])
        # window 2: target 0 interpolated (w = 2/3): x 12..15, y 9.33..10.33
        self.assertEqual(sorted(map(tuple, cells[2].tolist())), [(2, 2, 3, 3)])
        self.assertEqual(sorted(map(tuple, cells[3].tolist())), [(3, 3, 4, 4)])
        self.assertEqual(cells[4].shape[0], 0)                    # no extrapolation
        maps = occupancy_maps(cells, 0, 6, H // 4, W // 4)
        self.assertEqual(maps.shape, (6, H // 4, W // 4))
        self.assertEqual(float(maps[0].sum()), 1.0)
        self.assertEqual(float(maps[3, 3, 4]), 1.0)
        self.assertEqual(float(maps[4:].sum()), 0.0)

    def test_config_builds_head(self):
        """configs/changepoint_v1.yaml 接上 train_track_head.build_head：D = 10、固定滞后、33 个位移、4 像素格。"""
        try:
            import train_track_head as TH
        except ImportError as error:
            self.skipTest("训练入口需要 cv2/pandas/tqdm: %s" % error)
        from utils.stream_common import load_flat_config
        cfg = load_flat_config(os.path.join(ROOT, "configs", "changepoint_v1.yaml"))
        net = SimpleNamespace(readout=[SimpleNamespace(in_features=FEAT + 2)])
        base = dict(pad_height=264, pad_width=352, input_clip=3.0, time_bins=5, threshold=0.9)
        head = TH.build_head(cfg, net, base, torch.device("cpu"), q99=np.ones(12, dtype=np.float32), pos_weight=30.0)
        self.assertIsInstance(head, ChangepointLayer)
        self.assertEqual((head.defer_max, head.commit, head.cell, len(head.offsets)), (10, "fixed", 4, 33))
        self.assertEqual(head.occupancy_spec, (4, 66, 88))
        self.assertAlmostEqual(head.tau_logit, math.log(9.0) - math.log(30.0))
        self.assertEqual((head.evidence_form, head.fusion, cfg["lambda_dynamics"]), ("log1p", "bayes", 0.0))
        with self.assertRaises(ValueError):
            TH.build_head(cfg, net, base, torch.device("cpu"))
        # v2 differs from v1 only in the three diagnosed points, the epoch count and the output directory
        cfg2 = load_flat_config(os.path.join(ROOT, "configs", "changepoint_v2.yaml"))
        self.assertEqual({k for k in set(cfg) | set(cfg2) if cfg.get(k) != cfg2.get(k)},
                         {"evidence_form", "fusion", "lambda_dynamics", "epochs", "save_root"})
        head2 = TH.build_head(cfg2, net, base, torch.device("cpu"), q99=np.ones(12, dtype=np.float32), pos_weight=30.0)
        self.assertEqual((head2.evidence_form, head2.fusion, cfg2["lambda_dynamics"], cfg2["epochs"]),
                         ("linear", "product", 1.0, 50))


class V2Tests(unittest.TestCase):
    def test_evidence_forms(self):
        """linear：l = g - mu；log1p（v1）：l = log(1 + g) - mu；两者共用同一个 g，零输入都等于 -mu。"""
        lin = make_layer(evidence_form="linear")
        old = make_layer(evidence_form="log1p")
        old.load_state_dict(lin.state_dict())
        density_evidence(lin)
        density_evidence(old)
        x = torch.rand(2, 12, H, W) * 3.0
        mu = float(lin.mu())
        g = lin.evidence(x) + mu
        torch.testing.assert_allclose(old.evidence(x) + mu, torch.log1p(g))
        self.assertGreater(float(g.max()), 5.0)            # linear form reaches strong evidence directly
        with self.assertRaises(ValueError):
            make_layer(fusion="mean")

    def test_product_commit(self):
        """fusion = product：在第 D 窗提交的校准概率 = 平滑在场 P(s|s+D)(c_e) x 归属 m（逐个事件按定义重算）。"""
        D = 3
        layer = make_layer(defer_max=D)
        density_evidence(layer)
        ev, z0, feat = make_events(9, seed=8)
        out = run_eval(layer, ev, z0, feat, 9, cuts=(4,))
        with torch.no_grad():
            k = layer.kernel()
            P = layer.init_state(1, torch.device("cpu"))["P"]
            filt, pred = [], []
            for t in range(9):
                sel = ev["t"] == t
                e = {key: torch.from_numpy(ev[key][sel]) for key in ("b", "x", "y", "p", "bin")}
                P, pr = layer.forward_step(P, layer.evidence(layer.encode(e, 1)), k)
                filt.append(P)
                pred.append(pr)
            for key, (z, lag, st) in out.items():
                s = int(ev["t"][key])
                if st != STATUS_DEADLINE:
                    continue
                sm = filt[s + D]
                for j in range(s + D - 1, s - 1, -1):
                    sm = layer.rts_step(filt[j], pred[j + 1], sm, k)
                cy, cx = int(ev["y"][key]) // 4, int(ev["x"][key]) // 4
                lp0 = torch.tensor([float(z0[key]) - layer.log_w])
                m = layer.membership(torch.from_numpy(feat[key:key + 1]), torch.tensor([float(ev["p"][key])]),
                                     torch.tensor([float(ev["t_local"][key])]), lp0)
                want = float(sm[0, cy, cx] * m[0])
                self.assertAlmostEqual(1.0 / (1.0 + math.exp(-(z - layer.log_w))), want, delta=1e-4)

    def test_transition_loss_learns_motion(self):
        """L_dyn：一个占 1 格、每窗右移 1 格的目标（标签占据），只用 L_dyn 训练出生 / 存活 / 位移核：
        位移核把概率集中到 (0, +1)，存活升高，出生降低。"""
        layer = make_layer(rings=(1, 2))
        k_before = float(torch.softmax(layer.pi_logit, 0)[layer.offsets.index((0, 1))])
        occ = torch.zeros(12, 1, H // 4, W // 4)
        for t in range(12):
            occ[t, 0, 1, t % (W // 4)] = 1.0
        opt = torch.optim.Adam([layer.rho_raw, layer.p_raw, layer.pi_logit], lr=0.05)
        for _ in range(300):
            k = layer.kernel()
            loss = sum(layer.transition_loss(occ[t - 1], occ[t], k).sum() for t in range(1, 12) if t % (W // 4))
            opt.zero_grad()
            loss.backward()
            opt.step()
        w = torch.softmax(layer.pi_logit, 0).detach()
        self.assertGreater(float(w[layer.offsets.index((0, 1))]), 0.8)
        self.assertGreater(float(w[layer.offsets.index((0, 1))]), 10 * k_before)
        self.assertGreater(float(layer.p()), 0.95)
        self.assertLess(float(layer.rho()), 1e-3)

    def test_transition_loss_in_stream(self):
        """训练时 forward_chunk 给出 dyn_sum：换录像的那一窗不算（上一窗属于别的录像）；与逐窗直接计算一致。"""
        layer = make_layer(defer_max=3)
        ev, z0, feat = make_events(8, batch=2, seed=9)
        occ = torch.from_numpy(occ_from_events(ev, 8, 2))
        events, z, f, labels, _ = chunk_slice(ev, z0, feat, 0, 8)
        events["occ"] = occ
        keep = torch.ones(8, 2)
        keep[0] = 0
        keep[5, 1] = 0
        _, _, aux = layer.forward_chunk(events, z, f, None, 8, keep=keep, labels=labels, keys=labels)
        k = layer.kernel()
        want, n = 0.0, 0
        for t in range(1, 8):
            for b in range(2):
                if t == 5 and b == 1:
                    continue
                want += float(layer.transition_loss(occ[t - 1, b:b + 1], occ[t, b:b + 1], k).sum())
                n += (H // 4) * (W // 4)
        self.assertAlmostEqual(float(aux["dyn_sum"]), want, delta=1e-3 * max(1.0, want))
        self.assertEqual(aux["dyn_n"], n)


class StageCheckTests(unittest.TestCase):
    def test_stage1_numbers(self):
        """tools/rcl_stage1_check：IoU、普通 / 极密分组、极密录像目标附近误报、第一拍召回、翻转与救回。"""
        import tempfile
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import rcl_stage1_check as S

        def save(directory, name, t, x, y, lab, tid, base, fin):
            n = len(t)
            np.savez(os.path.join(directory, name + ".npz"),
                     locs=np.stack([np.zeros(n, np.int64), np.asarray(x), np.asarray(y), np.asarray(t)], 1),
                     labels=np.asarray(lab, np.float32), target_id=np.asarray(tid, np.float64),
                     base_probabilities=np.asarray(base, np.float32), now_probabilities=np.asarray(base, np.float32),
                     probabilities=np.asarray(fin, np.float32), decision_wait=np.full(n, 10, np.int8),
                     decision_status=np.full(n, 4, np.int8))

        with tempfile.TemporaryDirectory() as d:
            split = os.path.join(d, "x_val")
            os.makedirs(split)
            # normal: one target over windows 0..3, backbone first detects it at window 2; the final output also
            # catches window 1 (a rescue) and drops one background false positive
            save(split, "val_000", [10, 60, 110, 160, 70], [5, 6, 7, 8, 7], [5, 5, 5, 5, 6], [1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 0], [0.1, 0.5, 0.95, 0.95, 0.95], [0.1, 0.95, 0.95, 0.95, 0.2])
            # dense: near-target false positives 2 -> 3
            save(split, "val_001", [10, 20, 20, 20], [10, 12, 13, 12], [10, 10, 11, 12], [1, 0, 0, 0], [1, 0, 0, 0],
                 [0.95, 0.95, 0.95, 0.2], [0.95, 0.95, 0.95, 0.95])
            r = S.check_split(split, {"val_001"})
            self.assertAlmostEqual(r["iou"]["base"], 3 / 8.0)
            self.assertAlmostEqual(r["iou"]["final"], 4 / 8.0)
            self.assertAlmostEqual(r["iou_normal"]["base"], 2 / 5.0)
            self.assertAlmostEqual(r["iou_normal"]["final"], 3 / 4.0)
            self.assertEqual(r["dense_near_fp"], {"base": 2, "final": 3})
            self.assertAlmostEqual(r["dense_near_fp_rise"], 0.5)
            self.assertEqual(r["first_beat"]["n"], 4)
            self.assertAlmostEqual(r["first_beat"]["gain"], 0.25)
            self.assertEqual(r["flips"], {"up_tp": 1, "up_fp": 1, "down_tn": 1, "down_fn": 0})
            self.assertEqual((r["rescued"]["n"], r["rescued"]["mean_wait_windows"]), (1, 10.0))
            g = S.gates({"val": r}, 0.4)
            self.assertTrue(g["val_iou >= 0.4000"] and g["normal_not_lower"] and g["first_beat_gain >= 0.08"])
            self.assertFalse(g["dense_near_fp_rise <= 0.10"])
            self.assertFalse(g["pass"])


if __name__ == "__main__":
    unittest.main()
