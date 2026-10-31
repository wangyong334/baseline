"""模块三：目标级航迹头 v2（model/track_head.py、train_track_head.py）的测试（CPU、合成数据）。

    1. 初始化时最终 logit 与骨干 logit 逐位相同（修正项末层零初始化）
    2. 片段切分不变：一次处理 12 窗与分 3 段（段间 detach）处理，输出逐位相同（零等待与一窗滞后两种）
    3. 匀速运动的目标：只出生一条航迹，位置、速度跟得上；短暂消失 2 窗时航迹外推并重新接上（同一个槽）
    4. 关联：门内一个近处的峰与一个身后更强的峰，选近处的（v1 的"门内最强"会被身后的团拉走）
    5. 槽位回收：槽满时新目标替换存在概率最低的航迹
    6. 新录像开始（keep = 0）时只清空该流的航迹，也不再重判该流上一窗的事件
    7. 近期轨迹距离、沿迹坐标；对照组（context = none）看不到航迹
    8. 一窗滞后：除最后一窗外每个事件都恰好重判一次，初始化时滞后输出与 z0 相同；RTS 平滑与矩阵公式一致
    9. 监督：标注目标与航迹的匹配、存在 / 位置损失可反传，梯度只到航迹头；确定性模式下前向、反向不报错
   10. 配置：三个 v2 配置只差 decision_lag / context 与输出目录；v3 配置只比 v2 多 existence = ipda（及 decision_lag）
   11. 端到端：训练一个小骨干 -> 航迹头训练 -> 评估（零等待、一窗滞后、v3）
   12. v3（existence = ipda）：初始化恒等、片段切分不变；存在概率的伯努利递推与公式一致；检测概率低时航迹撑过缺口；
       卡尔曼更新按关联概率加权（像杂波的量测不拉动航迹）；梯度到达 P_D、似然比、存活概率
"""
import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.track_head import TrackHead, greedy_match, rts_position, window_targets  # noqa: E402

H, W, FEAT = 32, 48, 4
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_head(seed=0, **kwargs):
    torch.manual_seed(seed)
    params = dict(cell=4, slots=6, hidden=16, history=21)
    params.update(kwargs)
    return TrackHead(FEAT, H, W, **params)


def randomize(head, std=0.3):
    with torch.no_grad():
        for name in ("corr", "corr_lag"):
            if hasattr(head, name):
                for p in getattr(head, name).parameters():
                    p.normal_(0.0, std)


def blob_windows(n_windows, start=(8.0, 12.0), velocity=(2.0, 0.0), gaps=(), batch=1, stream=0, noise=10, seed=0,
                 keep_out=()):
    """每窗一个 7 个事件的目标（z0 = +6），外加随机背景事件（z0 = -6）。返回片段事件与 z0、feat、标签、target_id。"""
    rng = np.random.RandomState(seed)
    offs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
    cols = {key: [] for key in ("t", "b", "x", "y", "p", "t_local")}
    z0, lab, tid = [], [], []
    for t in range(n_windows):
        for b in range(batch):
            if b == stream and t not in gaps and t not in keep_out:
                cx, cy = start[0] + velocity[0] * t, start[1] + velocity[1] * t
                for dx, dy in offs:
                    cols["t"].append(t); cols["b"].append(b)
                    cols["x"].append(int(round(cx + dx))); cols["y"].append(int(round(cy + dy)))
                    cols["p"].append(float(rng.randint(2))); cols["t_local"].append(float(rng.rand()))
                    z0.append(6.0); lab.append(1.0); tid.append(1.0)
            for _ in range(noise):
                cols["t"].append(t); cols["b"].append(b)
                cols["x"].append(int(rng.randint(W))); cols["y"].append(int(rng.randint(H)))
                cols["p"].append(float(rng.randint(2))); cols["t_local"].append(float(rng.rand()))
                z0.append(-6.0); lab.append(0.0); tid.append(0.0)
    events = {key: torch.tensor(v, dtype=torch.long if key in ("t", "b", "x", "y") else torch.float32)
              for key, v in cols.items()}
    z0 = torch.tensor(z0, dtype=torch.float32)
    feat = torch.from_numpy(rng.randn(z0.shape[0], FEAT).astype(np.float32)) * 0.1
    return events, z0, feat, torch.tensor(lab), torch.tensor(tid)


def window_events(points, z=6.0, b=0):
    """一窗的事件字典（forward_window 用）：points 为 [(x, y), ...]。"""
    n = len(points)
    return {"b": torch.full((n,), b, dtype=torch.long), "x": torch.tensor([p[0] for p in points]),
            "y": torch.tensor([p[1] for p in points]), "p": torch.zeros(n), "t_local": torch.full((n,), 0.5),
            "z0": torch.full((n,), float(z)), "feat": torch.zeros(n, FEAT)}


def run_split_chunks(head, events, z0, feat, batch, bounds, keys=None):
    """分段处理，返回 (零等待输出, 片段结束状态, {key: 滞后输出})。"""
    state, z_out, delayed = head.init_state(batch, z0.device), torch.zeros_like(z0), {}
    for a, b in bounds:
        sel = (events["t"] >= a) & (events["t"] < b)
        ev = {key: v[sel] for key, v in events.items()}
        ev["t"] = ev["t"] - a
        z_part, state, aux = head.forward_chunk(ev, z0[sel], feat[sel], state, b - a,
                                                keys=None if keys is None else keys[sel])
        state = head.detach_state(state)
        z_out[sel] = z_part
        if aux["delayed_z"] is not None:
            delayed.update(zip(aux["delayed_key"].long().tolist(), aux["delayed_z"].tolist()))
    return z_out, state, delayed


class IdentityAndChunkTests(unittest.TestCase):
    def test_zero_init_identity(self):
        head = make_head()
        events, z0, feat, _, _ = blob_windows(8, noise=30)
        with torch.no_grad():
            z, _, _ = head.forward_chunk(events, z0, feat, None, 8)
        self.assertTrue(torch.equal(z, z0))

    def test_chunk_split_invariance(self):
        for lag in (0, 1):
            head = make_head(decision_lag=lag)
            randomize(head)
            events, z0, feat, _, _ = blob_windows(12, batch=2, noise=20)
            keys = torch.arange(z0.shape[0]).float()
            with torch.no_grad():
                z_full, s_full, d_full = run_split_chunks(head, events, z0, feat, 2, [(0, 12)], keys)
                z_split, s_split, d_split = run_split_chunks(head, events, z0, feat, 2, [(0, 3), (3, 8), (8, 12)], keys)
            self.assertTrue(torch.equal(z_full, z_split), lag)
            for key in s_full:
                if key != "pending":
                    self.assertTrue(torch.equal(s_full[key], s_split[key]), (lag, key))
            self.assertEqual(d_full, d_split)


class TrackingTests(unittest.TestCase):
    def run_blob(self, n, **kwargs):
        head = make_head()
        events, z0, feat, _, _ = blob_windows(n, **kwargs)
        log = []
        with torch.no_grad():
            _, state, _ = head.forward_chunk(events, z0, feat, head.init_state(1, z0.device), n, log=log)
        return head, state, log

    def test_follows_constant_velocity_target(self):
        _, state, log = self.run_blob(12, noise=15)
        active = state["active"][0]
        self.assertEqual(int(active.sum()), 1)                     # 噪声事件证据太弱，不出生
        slot = int(torch.nonzero(active)[0])
        pos, vel = state["pos"][0, slot], state["vel"][0, slot]
        self.assertLess(float((pos - torch.tensor([8.0 + 2.0 * 11, 12.0])).abs().max()), 1.5)
        self.assertLess(abs(float(vel[0]) - 2.0), 0.7)
        self.assertLess(abs(float(vel[1])), 0.7)

    def test_coasts_through_short_gap(self):
        _, state, log = self.run_blob(12, noise=15, gaps=(6, 7))
        slots = [set(np.flatnonzero(w["active"][0]).tolist()) for w in log]
        self.assertEqual(slots[5], slots[6])                       # 消失的两窗里航迹还在（外推）
        self.assertEqual(slots[5], slots[11])                      # 目标回来后接上同一个槽
        self.assertEqual(int(state["miss"][0, sorted(slots[11])[0]]), 0)

    def test_reset_clears_only_that_stream(self):
        head = make_head()
        events, z0, feat, _, _ = blob_windows(10, batch=2, stream=0, noise=10)
        ev1, z1, f1, _, _ = blob_windows(10, batch=2, stream=1, noise=0, keep_out=range(6, 10), seed=1)
        events = {key: torch.cat([events[key], ev1[key]]) for key in events}
        z0, feat = torch.cat([z0, z1]), torch.cat([feat, f1])
        keep = torch.ones(10, 2)
        keep[6, 1] = 0
        with torch.no_grad():
            _, state, _ = head.forward_chunk(events, z0, feat, head.init_state(2, z0.device), 10, keep=keep)
        self.assertEqual(int(state["active"][0].sum()), 1)
        self.assertEqual(int(state["active"][1].sum()), 0)         # 清空后流 1 没有新证据


class AssociationTests(unittest.TestCase):
    def track_state(self, head, pos, vel, ex=0.9, slots=(0,)):
        st = head.init_state(1, torch.device("cpu"))
        for n, p, v in zip(slots, pos, vel):
            st["active"][0, n] = True
            st["ex"][0, n] = ex
            st["pos"][0, n] = torch.tensor(p)
            st["vel"][0, n] = torch.tensor(v)
            st["P"][0, n] = torch.eye(2)
        return st

    def test_prefers_near_peak_over_stronger_peak_behind(self):
        head = make_head()
        st = self.track_state(head, [(20.0, 16.0)], [(2.0, 0.0)])
        # 预测位置 (22, 16)：那里 5 个事件；身后 12 像素处 12 个事件（证据更强，仍在验证门内）
        ev = window_events([(22, 16)] * 5 + [(10, 16)] * 12)
        with torch.no_grad():
            _, new, info = head.forward_window(st, ev)
        self.assertTrue(bool(info["has"][0, 0]))
        self.assertLess(abs(float(new["pos"][0, 0, 0]) - 22.0), 1.0)

    def test_one_birth_per_target(self):
        head = make_head()
        events, z0, feat, _, _ = blob_windows(1, noise=0)           # 7 个事件落在 2x2 格内：四个格的 3x3 和相等
        with torch.no_grad():
            _, state, aux = head.forward_chunk(events, z0, feat, None, 1)
        self.assertEqual(aux["births"], 1)
        self.assertEqual(int(state["active"].sum()), 1)

    def test_recycles_weakest_track_when_full(self):
        head = make_head(slots=2)
        st = self.track_state(head, [(4.0, 4.0), (44.0, 4.0)], [(0.0, 0.0), (0.0, 0.0)], ex=0.1, slots=(0, 1))
        st["ex"][0, 1] = 0.05                                       # 槽 1 最弱
        ev = window_events([(24, 26), (25, 26), (24, 27), (23, 26), (24, 25), (25, 27), (23, 25)])
        with torch.no_grad():
            _, new, info = head.forward_window(st, ev)
        self.assertEqual(info["recycled"], 1)
        self.assertTrue(bool(info["born"][0, 1]))
        self.assertLess(float((new["pos"][0, 1] - torch.tensor([24.0, 26.0])).abs().max()), 1.5)


class ContextTests(unittest.TestCase):
    def state_with_track(self, head):
        st = head.init_state(1, torch.device("cpu"))
        st["active"][0, 0] = True
        st["ex"][0, 0] = 0.9
        st["pos"][0, 0] = torch.tensor([40.0, 20.0])
        st["vel"][0, 0] = torch.tensor([2.0, 0.0])                 # 向 +x 运动
        st["P"][0, 0] = torch.eye(2)
        st["hist"][0, 0, 5] = torch.tensor([30.0, 20.0])          # 5 窗之前在 (30, 20)
        st["hist_ok"][0, 0, 5] = True
        return st

    def test_recent_path_and_along_track(self):
        head = make_head()
        st = self.state_with_track(head)
        ev = window_events([(30, 20), (2, 2), (44, 20)], z=0.0)
        ctx = head._context(ev, st)
        self.assertLess(float(ctx[0, 5]), 0.05)                    # 落在近期轨迹上
        self.assertEqual(float(ctx[1, 5]), 1.0)                    # 远离近期轨迹
        self.assertLess(float(ctx[0, 7]), -0.2)                    # 在航迹身后
        self.assertGreater(float(ctx[2, 7]), 0.05)                 # 在航迹前方
        self.assertLess(float(ctx[2, 8]), 0.05)                    # 横迹距离接近 0

    def test_notrack_control_sees_no_track(self):
        head = make_head(context="none")
        ev = window_events([(30, 20), (46, 0)], z=0.0)
        self.assertEqual(float(head._context(ev, self.state_with_track(head)).abs().sum()), 0.0)


class LagTests(unittest.TestCase):
    def test_identity_and_coverage(self):
        head = make_head(decision_lag=1)
        events, z0, feat, _, _ = blob_windows(8, noise=10)
        keys = torch.arange(z0.shape[0]).float()
        with torch.no_grad():
            z, st, aux = head.forward_chunk(events, z0, feat, None, 8, keys=keys)
        self.assertTrue(torch.equal(z, z0))
        dk = aux["delayed_key"].long()
        expected = torch.nonzero(events["t"] < 7, as_tuple=False).flatten()
        self.assertEqual(sorted(dk.tolist()), sorted(expected.tolist()))   # 除最后一窗外恰好各一次
        self.assertTrue(torch.equal(aux["delayed_z"], z0[dk]))
        self.assertEqual(int(st["pending"]["z0"].shape[0]), int((events["t"] == 7).sum()))

    def test_reset_drops_pending_of_that_stream(self):
        head = make_head(decision_lag=1)
        events, z0, feat, _, _ = blob_windows(10, batch=2, stream=0, noise=5)
        keys = torch.arange(z0.shape[0]).float()
        keep = torch.ones(10, 2)
        keep[5, 1] = 0
        with torch.no_grad():
            _, _, aux = head.forward_chunk(events, z0, feat, None, 10, keep=keep, keys=keys)
        resolved = set(aux["delayed_key"].long().tolist())
        w4 = events["t"] == 4
        self.assertFalse(any(int(i) in resolved for i in torch.nonzero(w4 & (events["b"] == 1)).flatten()))
        self.assertTrue(all(int(i) in resolved for i in torch.nonzero(w4 & (events["b"] == 0)).flatten()))

    def test_rts_matches_matrix_formula(self):
        rng = np.random.RandomState(0)
        F_ = np.array([[1.0, 1.0], [0.0, 1.0]])
        for _ in range(5):
            A = rng.randn(2, 2)
            P_prev = A @ A.T + 0.5 * np.eye(2)
            Q = np.array([[0.25, 0.5], [0.5, 1.0]]) * 0.3
            Ppred = F_ @ P_prev @ F_.T + Q
            x_prev = rng.randn(2, 2)                                   # [轴, (位置, 速度)]
            x_pred = x_prev @ F_.T
            x_post = x_pred + rng.randn(2, 2)
            C = P_prev @ F_.T @ np.linalg.inv(Ppred)
            expected = np.array([x_prev[a] + C @ (x_post[a] - x_pred[a]) for a in range(2)])[:, 0]
            t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
            got = rts_position(t(x_prev[:, 0]), t(P_prev), t(x_pred[:, 0]), t(x_pred[:, 1]), t(Ppred),
                               t(x_post[:, 0]), t(x_post[:, 1]))
            np.testing.assert_allclose(got.numpy(), expected, rtol=1e-9, atol=1e-9)


class SupervisionTests(unittest.TestCase):
    def test_window_targets_and_greedy(self):
        eb = torch.tensor([0, 0, 0, 1, 1])
        ex = torch.tensor([10, 12, 30, 5, 7])
        ey = torch.tensor([10, 10, 30, 5, 5])
        label = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
        tid = torch.tensor([1.0, 1.0, 0.0, 2.0, 2.0])
        gb, gc, gr = window_targets(eb, ex, ey, label, tid, 8.0)
        self.assertEqual(gb.tolist(), [0, 1])
        self.assertTrue(torch.allclose(gc, torch.tensor([[11.0, 10.0], [6.0, 5.0]])))
        self.assertTrue(torch.all(gr >= 8.0))
        dist = np.array([[1.0, 5.0], [2.0, np.inf]])
        self.assertEqual(greedy_match(dist, np.array([0, 0])), [(0, 0)])   # 同一流内一个槽只配一次
        self.assertEqual(sorted(greedy_match(dist, np.array([0, 1]))), [(0, 0), (1, 0)])

    def test_losses_backward_to_head_only(self):
        head = make_head(decision_lag=1)
        events, z0, feat, lab, tid = blob_windows(10, noise=20)
        z, _, aux = head.forward_chunk(events, z0, feat, None, 10, labels=lab, tid=tid, keys=lab)
        self.assertGreater(aux["exist_n"], 0)
        self.assertGreater(aux["pos_n"], 0)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        loss = bce(z, lab) + bce(aux["delayed_z"], aux["delayed_key"]) + aux["exist_sum"] / aux["exist_n"] \
            + aux["pos_sum"] / aux["pos_n"]
        loss.backward()
        grads = {name: p.grad for name, p in head.named_parameters()}
        for name in ("exist.weight", "gru.weight_ih", "corr.2.weight", "corr_lag.2.weight", "r_net.2.bias", "log_q"):
            self.assertIsNotNone(grads[name], name)
            self.assertTrue(torch.isfinite(grads[name]).all(), name)

    def test_deterministic_mode(self):
        torch.use_deterministic_algorithms(True)
        try:
            for existence in ("gru", "ipda"):
                head = make_head(decision_lag=1, existence=existence)
                events, z0, feat, lab, tid = blob_windows(6, batch=2, noise=10)
                z, _, aux = head.forward_chunk(events, z0, feat, None, 6, labels=lab, tid=tid, keys=lab)
                (z.sum() + aux["delayed_z"].sum() + aux["exist_sum"]).backward()
        finally:
            torch.use_deterministic_algorithms(False)


def single_track_state(head, pos, vel, ex=0.9, slots=(0,)):
    st = head.init_state(1, torch.device("cpu"))
    for n, p, v in zip(slots, pos, vel):
        st["active"][0, n] = True
        st["ex"][0, n] = ex
        st["pos"][0, n] = torch.tensor(p)
        st["vel"][0, n] = torch.tensor(v)
        st["P"][0, n] = torch.eye(2)
    return st


class IpdaTests(unittest.TestCase):
    def test_identity_and_chunk_invariance(self):
        for lag in (0, 1):
            head = make_head(existence="ipda", decision_lag=lag)
            events, z0, feat, _, _ = blob_windows(12, batch=2, noise=20, gaps=(5, 6))
            keys = torch.arange(z0.shape[0]).float()
            with torch.no_grad():
                z, _, _ = head.forward_chunk(events, z0, feat, None, 12, keys=keys)
            self.assertTrue(torch.equal(z, z0), lag)
            randomize(head)
            with torch.no_grad():
                for p in list(head.pd_head.parameters()) + list(head.lr_net.parameters()):
                    p.add_(0.2 * torch.randn_like(p))
                z_full, s_full, d_full = run_split_chunks(head, events, z0, feat, 2, [(0, 12)], keys)
                z_split, s_split, d_split = run_split_chunks(head, events, z0, feat, 2, [(0, 3), (3, 8), (8, 12)], keys)
            self.assertTrue(torch.equal(z_full, z_split), lag)
            for key in s_full:
                if key != "pending":
                    self.assertTrue(torch.equal(s_full[key], s_split[key]), (lag, key))
            self.assertEqual(d_full, d_split)

    def test_existence_recursion_matches_formula(self):
        head = make_head(existence="ipda")
        st = single_track_state(head, [(20.0, 16.0), (40.0, 8.0)], [(0.0, 0.0), (0.0, 0.0)], ex=0.9, slots=(0, 1))
        ev = window_events([(20, 16)] * 7)                             # 只有槽 0 的预测位置有证据
        with torch.no_grad():
            _, new, info = head.forward_window(st, ev)
        self.assertTrue(bool(info["has"][0, 0]))
        self.assertFalse(bool(info["has"][0, 1]))
        rp = 0.9 / (1.0 + math.exp(-math.log(0.995 / 0.005)))
        L_pred = math.log(rp / (1.0 - rp))
        pd, lr = 0.8, math.exp(2.0)                                    # 初始化：P_D = 0.8，似然比 e^2
        self.assertAlmostEqual(float(info["ex_logit"][0, 0]), L_pred + math.log(1.0 - pd + pd * lr), places=4)
        self.assertAlmostEqual(float(info["ex_logit"][0, 1]), L_pred + math.log(1.0 - pd), places=4)
        self.assertAlmostEqual(float(new["pd"][0, 1]), pd, places=5)

    def test_low_detection_probability_keeps_track_through_gap(self):
        ex_end = {}
        for pd in (0.3, 0.9):
            head = make_head(existence="ipda")
            with torch.no_grad():
                head.pd_head.bias.fill_(math.log(pd / (1.0 - pd)))
            events, z0, feat, _, _ = blob_windows(12, noise=10, gaps=(6, 7, 8, 9))
            log = []
            with torch.no_grad():
                head.forward_chunk(events, z0, feat, None, 12, log=log)
            slot = int(np.flatnonzero(log[5]["active"][0])[0])
            ex_end[pd] = float(log[9]["ex"][0, slot])
        self.assertGreater(ex_end[0.3], 0.9)                           # 检测不可靠：漏 4 窗仍相信目标在
        self.assertLess(ex_end[0.9], 0.5)                              # 检测可靠：漏 4 窗就不再相信

    def test_update_weighted_by_association_probability(self):
        moved = {}
        for bias in (-12.0, 12.0):
            head = make_head(existence="ipda")
            with torch.no_grad():
                head.lr_net[2].bias.fill_(bias)
            st = single_track_state(head, [(20.0, 16.0)], [(0.0, 0.0)])
            ev = window_events([(24, 16)] * 5)                         # 预测位置 4 像素外的一个量测（在验证门内）
            with torch.no_grad():
                _, new, info = head.forward_window(st, ev)
            self.assertTrue(bool(info["has"][0, 0]))
            moved[bias] = float(new["pos"][0, 0, 0]) - 20.0
        self.assertLess(abs(moved[-12.0]), 0.01)                       # 像杂波：航迹保持预测
        self.assertGreater(moved[12.0], 1.0)                           # 像目标：普通卡尔曼更新

    def test_survival_bound_keeps_log_odds_finite(self):
        head = make_head(existence="ipda")
        with torch.no_grad():
            head.surv_logit.fill_(20.0)                                # 想让 p_S -> 1：被限制在 0.999
        st = single_track_state(head, [(40.0, 8.0)], [(0.0, 0.0)], ex=1.0 - 1e-6)
        with torch.no_grad():
            _, _, info = head.forward_window(st, window_events([(2, 30)] * 3, z=-6.0))
        self.assertLess(float(info["ex_logit"][0, 0]), math.log(0.999 / 0.001) + math.log(0.2) + 1e-3)

    def test_pda_association_uses_local_clutter(self):
        head = make_head(existence="ipda")
        st = single_track_state(head, [(20.0, 16.0)], [(2.0, 0.0)])
        ev = window_events([(21, 16)], z=0.0)
        weights = []
        for c in (0.0, 30.0):
            ev["clutter"] = torch.tensor([c])
            weights.append(float(head._context(ev, st)[0, 0]))       # 第 0 维：事件归为杂波（无航迹）的权重
        self.assertGreater(weights[1], weights[0] + 0.1)               # 杂波越密，同一位置的事件越不归航迹

    def test_calibration_targets_and_gradients(self):
        eb = torch.tensor([0, 0, 0, 1, 1])
        label = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
        tid = torch.tensor([2.0, 2.0, 0.0, 1.0, 1.0])
        prob = torch.tensor([0.4, 0.3, 0.9, 0.1, 0.2])
        from model.track_head import target_mass
        self.assertTrue(torch.allclose(target_mass(eb, label, tid, prob), torch.tensor([0.7, 0.3])))   # 与 window_targets 同序
        head = make_head(existence="ipda")
        events, z0, feat, lab, tid = blob_windows(10, noise=20)
        weak = (lab > 0.5) & ((events["t"] == 5) | (events["t"] == 6))
        z0 = torch.where(weak, torch.full_like(z0, -4.0), z0)          # 第 5、6 窗目标在、但骨干几乎没看见（未检出）
        sums = {}
        for name, labels in (("all", lab), ("detected", torch.where(weak, torch.zeros_like(lab), lab))):
            _, _, aux = head.forward_chunk(events, z0, feat, None, 10, labels=labels, tid=tid)
            sums[name] = (float(aux["pd_sum"]), aux["pd_n"])
        self.assertGreater(sums["all"][1], sums["detected"][1])        # 未检出的两窗也给了 P_D 标签（标签为 0）
        _, _, aux = head.forward_chunk(events, z0, feat, None, 10, labels=lab, tid=tid)
        self.assertGreater(aux["org_n"], 0)
        (aux["pd_sum"] / aux["pd_n"]).backward(retain_graph=True)
        self.assertGreater(float(head.pd_head.bias.grad.abs()), 0.0)
        head.zero_grad()
        (aux["org_sum"] / aux["org_n"]).backward()
        self.assertGreater(float(head.lr_net[2].bias.grad.abs()), 0.0)

    def test_gradients_reach_ipda_parameters(self):
        head = make_head(existence="ipda", decision_lag=1)
        events, z0, feat, lab, tid = blob_windows(10, noise=20, gaps=(5,))
        z, _, aux = head.forward_chunk(events, z0, feat, None, 10, labels=lab, tid=tid, keys=lab)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        loss = bce(z, lab) + bce(aux["delayed_z"], aux["delayed_key"]) + aux["exist_sum"] / aux["exist_n"] \
            + aux["pos_sum"] / aux["pos_n"]
        loss.backward()
        grads = {name: p.grad for name, p in head.named_parameters()}
        for name in ("pd_head.weight", "pd_head.bias", "lr_net.2.weight", "lr_net.2.bias", "surv_logit",
                     "exist.weight", "corr.2.weight"):
            self.assertIsNotNone(grads[name], name)
            self.assertTrue(torch.isfinite(grads[name]).all(), name)
            self.assertGreater(float(grads[name].abs().sum()), 0.0, name)


class ConfigTests(unittest.TestCase):
    def test_v2_variants(self):
        from utils.stream_common import load_flat_config
        base = load_flat_config(os.path.join(ROOT, "configs", "track_head_v2.yaml"))
        for name, key, value in (("track_head_v2_lag1", "decision_lag", 1), ("track_head_v2_notrack", "context", "none")):
            cfg = load_flat_config(os.path.join(ROOT, "configs", name + ".yaml"))
            self.assertEqual(set(cfg), set(base))
            diff = {k for k in base if cfg[k] != base[k]}
            self.assertEqual(diff, {key, "save_root"}, name)
            self.assertEqual(cfg[key], value)
            self.assertEqual(cfg["save_root"], "log/%s_seed37" % name)
        self.assertEqual((base["decision_lag"], base["context"]), (0, "track"))

    def test_v3_variants(self):
        from utils.stream_common import load_flat_config
        base = load_flat_config(os.path.join(ROOT, "configs", "track_head_v2.yaml"))
        for name, extra in (("track_head_v3", {}), ("track_head_v3_lag1", {"decision_lag": 1})):
            cfg = load_flat_config(os.path.join(ROOT, "configs", name + ".yaml"))
            self.assertEqual(set(cfg), set(base) | {"existence", "lambda_pd", "lambda_origin"})
            self.assertEqual((cfg["lambda_pd"], cfg["lambda_origin"]), (1.0, 1.0))
            self.assertEqual(cfg["existence"], "ipda")
            self.assertEqual({k for k in base if cfg[k] != base[k]}, set(extra) | {"save_root"}, name)
            for k, v in extra.items():
                self.assertEqual(cfg[k], v)
            self.assertEqual(cfg["save_root"], "log/%s_seed37" % name)


class EndToEndTests(unittest.TestCase):
    def test_train_and_eval(self):
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
        except ImportError as error:
            self.skipTest("完整训练入口需要 cv2/pandas/tqdm: %s" % error)
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        import yaml
        import train_stream_v1 as T
        import train_track_head as TH
        from test_stream_layerwise import H as SH, Q99, SMALL, W as SW, synthetic_sequence
        from utils.stream_common import load_flat_config
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pos = neg = 0
            for split, seeds in (("train", (3, 4, 5)), ("val", (6,)), ("test", (7,))):
                (root / split).mkdir()
                for seed in seeds:
                    seq = synthetic_sequence(seed=seed, n=3000)
                    loc = np.stack((seq.x, seq.y, seq.t), axis=1)
                    norm = np.column_stack((loc, seq.p, seq.label, seq.target_id))
                    np.savez(root / split / ("seq_%d.npz" % seed), ev_loc=loc, evs_norm=norm)
                    if split == "train":
                        pos += int(seq.label.sum())
                        neg += seq.n_events - int(seq.label.sum())
            stats = dict(q99=Q99.tolist(), train_pos=pos, train_neg=neg, pos_weight=min(neg / pos, 30.0))
            (root / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
            base_dir = root / "base"
            cfg = load_flat_config(os.path.join(ROOT, "configs", "evisseg_stream_tif_comm_peak_invariant.yaml"))
            cfg.update(root=str(root), stats_path=str(root / "stats.json"), save_root=str(base_dir),
                       channels=list(SMALL), height=SH, width_px=SW, pad_height=SH, pad_width=SW,
                       n_windows=12, epochs=2, num_workers=0, tbptt_k=4, train_subset_size=1,
                       calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
                       segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=4,
                       execution="layer", input_device="gpu", streams=2)
            T.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
            for name in ("track_head_v2", "track_head_v2_lag1", "track_head_v3"):
                head_cfg = load_flat_config(os.path.join(ROOT, "configs", name + ".yaml"))
                head_cfg.update(root=str(root), base_checkpoint=str(base_dir / "best_val_iou_seed37.pt"),
                                save_root=str(root / name), epochs=2, streams=2, tbptt_k=4, eval_chunk=4,
                                slots=4, segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1)
                cfg_path = root / (name + ".yaml")
                cfg_path.write_text(yaml.safe_dump({"ALL": head_cfg}), encoding="utf-8")
                TH.main(["--config", str(cfg_path), "--mode", "train", "--device", "cpu"])
                ckpt = root / name / "best_val_iou_seed37.pt"
                self.assertTrue(ckpt.exists())
                TH.main(["--config", str(cfg_path), "--mode", "eval", "--device", "cpu", "--split", "test",
                         "--checkpoint", str(ckpt), "--dump-dir", str(root / ("dump_" + name)),
                         "--tracks-dir", str(root / ("tracks_" + name))])
                result = json.loads((root / name / "eval_test_best_val_iou_seed37.json").read_text(encoding="utf-8"))
                kinds = ("base", "track", "track_now") if name.endswith("lag1") else ("base", "track")
                for kind in kinds:
                    for key in ("iou", "acc", "pd", "fa"):
                        self.assertTrue(np.isfinite(result[kind][key]), (name, kind, key))
                self.assertIn("pass", result["gate"])
                with np.load(str(root / ("dump_" + name) / "seq_7.npz")) as dump:
                    self.assertEqual(dump["probabilities"].shape, dump["base_probabilities"].shape)
                with np.load(str(root / ("tracks_" + name) / "seq_7.npz")) as tracks:
                    self.assertEqual(tracks["pos"].shape[1:], (4, 2))
                    self.assertEqual(tracks["pd"].shape[1:], (4,))
                    self.assertEqual(tracks["beta"].shape[1:], (4,))
            # base 一列是骨干的原输出：与正式评估（逐窗执行）的 IoU 一致
            args = SimpleNamespace(checkpoint=str(base_dir / "best_val_iou_seed37.pt"), device="cpu", mode="eval",
                                   split="test", overwrite=True, execution=None, input_device=None,
                                   readout_ablation="none", subthreshold_ablation="none", memory_ablation="none",
                                   memory_off_layers=None, reset_every=0, reset_offset=0, dump_dir=None,
                                   eval_noise_load=0.0, eval_noise_seed=0)
            T.mode_eval(args, cfg)
            official = json.loads((base_dir / "eval_test_best_val_iou_seed37.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(result["base"]["iou"], official["carry"]["iou"], places=3)


if __name__ == "__main__":
    unittest.main()
