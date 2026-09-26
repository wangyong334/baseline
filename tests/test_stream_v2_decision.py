"""判决层评估时开关（非对称记忆 λ、预测门控 ε、邻域复位 r）与告警诊断的单元测试。全部在 CPU 上运行，不需要数据集。

守住四件事：
    1. 默认值 = 原实现，逐位相同（与按原公式重写的参考实现逐元素相等）
    2. 每个开关都不破坏保证：非对称记忆的单像素证据 E0[exp(e)] = exp(-(G - Gp)) <= 1（精确求期望）；
       门控后的 G 恰好是未门控 G 的截断；邻域复位只清零、不增加；三个开关同时打开时纯背景告警率不超过紧界
    3. 机制：漂离目标的错误速度管道在原实现下每个背景事件 +4.3 nat，λ = 0 时变成负证据
    4. 诊断与工具：活跃比例、稀疏运算量、告警诊断（距离分档、重复位置、去重、纯背景窗）、证据决定发布、命令行
"""
import math
import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

from dataset import stream_windows as sw
from model.evidence_neuron import LOG_ZERO, DriftCUSUM, pixel_evidence, shift2d, velocity_grid

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs",
                      "evisseg_stream_v2.yaml")


def reference_step(cusum, state, counts, mu0, log_g_prev):
    """按 09-19 原实现的公式重写的一步（不经过新代码路径），用来核对默认值逐位相同。"""
    k = int(state["k"])
    shifts = cusum.step_shifts(k)
    carried = None
    if cusum.track_decay > 0:
        carried = cusum.track_decay * torch.stack(
            [shift2d(state["G"][:, v], sy, sx) for v, (sy, sx) in enumerate(shifts)], 1)
    if log_g_prev is None:
        G = carried if carried is not None else torch.zeros_like(state["G"])
    else:
        g = torch.exp(log_g_prev)
        fresh = torch.stack([shift2d(g[:, 0], sy, sx) for sy, sx in shifts], 1)
        G = fresh if carried is None else torch.maximum(fresh, carried)
    e, _ = pixel_evidence(counts, mu0, torch.log(torch.clamp(G, min=math.exp(LOG_ZERO))))
    padded = F.pad(torch.exp(torch.clamp(e, max=80.0)), (1, 1, 1, 1), value=1.0)
    ell = torch.clamp(torch.log(F.avg_pool2d(padded, 3, stride=1)), min=-1e4)
    C = torch.stack([shift2d(state["C"][:, v], sy, sx) for v, (sy, sx) in enumerate(shifts)], 1)
    C = torch.relu(C + ell)
    return {"G": G, "C": C, "ell": ell, "k": k + 1}


def random_inputs(seed, steps=8, h=10, w=12):
    gen = torch.Generator().manual_seed(seed)
    counts = torch.poisson(torch.full((steps, 1, 1, h, w), 0.2, dtype=torch.float64), generator=gen)
    mu0 = torch.full_like(counts, 0.2)
    log_g = torch.randn(steps, 1, 1, h, w, dtype=torch.float64, generator=gen) - 1.5
    return counts, mu0, log_g


def poisson_expectation(fn, mu, n_max=80):
    """E[fn(N)]，N ~ Poisson(mu)，对 n = 0..n_max 精确求和（mu 小时截断误差可忽略）。"""
    total, log_p = 0.0, -mu
    for n in range(n_max + 1):
        if n > 0:
            log_p += math.log(mu) - math.log(n)
        total += math.exp(log_p) * fn(n)
    return total


class DefaultsAreBitIdenticalTests(unittest.TestCase):
    def test_defaults_match_reference_implementation(self):
        counts, mu0, log_g = random_inputs(0)
        velocities = velocity_grid([-1.0, 0.0, 2.0])
        for explicit in (False, True):
            kwargs = dict(memory_gain=1.0, gate_eps=0.0, reset_radius=0) if explicit else {}
            cusum = DriftCUSUM(velocities, footprint=3, track_decay=0.8, **kwargs)
            state = cusum.init_state(1, 10, 12, "cpu", torch.float64)
            ref = dict(state)
            for k in range(counts.shape[0]):
                prev = log_g[k - 1] if k > 0 else None
                state, _, _ = cusum.step(state, counts[k], mu0[k], prev)
                ref = reference_step(cusum, ref, counts[k], mu0[k], prev)
                for key in ("G", "C", "ell"):
                    self.assertTrue(torch.equal(state[key], ref[key]), "%s 第 %d 窗不同" % (key, k))
                self.assertIs(state["Gp"], state["G"])          # λ = 1：加分与扣分用同一个张量

    def test_invalid_switch_values_are_rejected(self):
        v = velocity_grid([0.0])
        for kwargs in (dict(memory_gain=1.2), dict(memory_gain=-0.1), dict(gate_eps=-1e-3), dict(reset_radius=-1),
                       dict(reset_radius=1.5)):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                DriftCUSUM(v, **kwargs)


class AsymmetricMemoryTests(unittest.TestCase):
    def test_single_pixel_expectation_is_at_most_one(self):
        """D0：Gp <= G 时 E0[exp(e)] = exp(-(G - Gp))，相等时恰为 1。"""
        mu0 = 0.0048
        cusum = DriftCUSUM([(0.0, 0.0)], footprint=1)
        for G, Gp, want in ((1.0, 0.5, math.exp(-0.5)), (1.0, 1.0, 1.0), (2.0, 0.0, math.exp(-2.0))):
            Gt = torch.full((1, 1, 1, 1), G, dtype=torch.float64)
            Gpt = torch.full((1, 1, 1, 1), Gp, dtype=torch.float64)
            m = torch.full((1, 1, 1, 1), mu0, dtype=torch.float64)

            def expo(n):
                c = torch.full((1, 1, 1, 1), float(n), dtype=torch.float64)
                return math.exp(float(cusum.tube_evidence(c, m, Gt, Gpt if Gp != G else None)))
            self.assertAlmostEqual(poisson_expectation(expo, mu0), want, places=6)

    def test_drifted_wrong_velocity_tube(self):
        """机制：记忆 G ≈ 1 漂到背景上（mu0 = 0.005），新鲜预测为 0。
        原实现一个事件 +log(1 + 1/0.005) - 1 ≈ 4.30 nat；λ = 0 时加分用 Gp = 0，同一个事件只剩 -1。"""
        mu0 = torch.full((1, 1, 1, 1), 0.005, dtype=torch.float64)
        one = torch.ones(1, 1, 1, 1, dtype=torch.float64)
        G_old = torch.full((1, 1, 1, 1), 1.0 / 0.8, dtype=torch.float64)          # rho * G_old = 1
        fresh = torch.zeros(1, 1, 1, 1, dtype=torch.float64)                       # 网络认为这里没有目标
        for gain, want in ((1.0, math.log(1 + 1 / 0.005) - 1.0), (0.0, -1.0)):
            cusum = DriftCUSUM([(0.0, 0.0)], footprint=1, track_decay=0.8, memory_gain=gain)
            G, Gp = cusum.predicted_pair(G_old, fresh, [(0, 0)])
            self.assertAlmostEqual(float(G), 1.0, places=12)
            self.assertAlmostEqual(float(cusum.tube_evidence(one, mu0, G, Gp)), want, places=6)

    def test_fresh_prediction_is_never_discounted(self):
        """网络当前就预测有目标时，λ 不影响加分：Gp = max(新鲜强度, λ·rho·旧 G) >= 新鲜强度。"""
        counts, mu0, log_g = random_inputs(1)
        cusum = DriftCUSUM(velocity_grid([0.0, 1.0]), footprint=1, track_decay=0.8, memory_gain=0.3)
        state = cusum.init_state(1, 10, 12, "cpu", torch.float64)
        for k in range(counts.shape[0]):
            state, _, _ = cusum.step(state, counts[k], mu0[k], log_g[k - 1] if k > 0 else None)
            self.assertTrue(bool((state["Gp"] <= state["G"] + 1e-15).all()))
            if k > 0:
                fresh = cusum._shift_each(torch.exp(log_g[k - 1]), state["shifts"])
                self.assertTrue(bool((state["Gp"] >= fresh - 1e-15).all()))


class GateTests(unittest.TestCase):
    def test_gated_intensity_is_truncation_of_ungated(self):
        """门控后的 G 恰好等于不门控的 G 按 ε 截断（所以一次 ε = 0 的运行就能读出各档活跃比例）。"""
        counts, mu0, log_g = random_inputs(2)
        velocities = velocity_grid([-1.0, 0.0, 1.0])
        eps = 0.15
        plain = DriftCUSUM(velocities, footprint=3, track_decay=0.8)
        gated = DriftCUSUM(velocities, footprint=3, track_decay=0.8, gate_eps=eps)
        s1 = plain.init_state(1, 10, 12, "cpu", torch.float64)
        s2 = gated.init_state(1, 10, 12, "cpu", torch.float64)
        for k in range(counts.shape[0]):
            prev = log_g[k - 1] if k > 0 else None
            s1, _, _ = plain.step(s1, counts[k], mu0[k], prev)
            s2, _, _ = gated.step(s2, counts[k], mu0[k], prev)
            want = s1["G"] * (s1["G"] >= eps).to(s1["G"].dtype)
            self.assertTrue(torch.equal(s2["G"], want), "第 %d 窗" % k)

    def test_inactive_neurons_get_zero_evidence(self):
        """足迹内没有 G >= ε 的神经元，证据为 0（在 LOG_ZERO 的数值精度内），膜电位不动。"""
        cusum = DriftCUSUM([(0.0, 0.0)], footprint=3, gate_eps=0.5)
        counts = torch.poisson(torch.full((1, 1, 9, 9), 0.3, dtype=torch.float64))
        g = torch.full((1, 1, 9, 9), 0.1, dtype=torch.float64)
        g[0, 0, 1, 1] = 1.0
        state = cusum.init_state(1, 9, 9, "cpu", torch.float64)
        state, _, _ = cusum.step(state, counts, torch.full_like(counts, 0.3), torch.log(g))
        far = state["ell"][0, 0, 4:, 4:]
        self.assertLess(float(far.abs().max()), 1e-9)
class ResetRadiusTests(unittest.TestCase):
    def test_neighbourhood_reset(self):
        C = torch.full((1, 2, 9, 9), 3.0, dtype=torch.float64)
        C[0, :, 4, 4] = 20.0
        ell = torch.zeros_like(C)
        for radius in (0, 1, 2):
            cusum = DriftCUSUM([(0.0, 0.0), (0.0, 1.0)], reset_radius=radius)
            out, _, _, alarm = cusum.accumulate(C.clone(), [(0, 0), (0, 0)], ell, theta=10.0, reset=True)
            self.assertEqual(int(alarm.sum()), 1)
            zone = (out == 0).all(1)[0]
            self.assertEqual(int(zone.sum()), (2 * radius + 1) ** 2)
            self.assertTrue(bool(zone[4 - radius:5 + radius, 4 - radius:5 + radius].all()))
            self.assertTrue(bool((out <= C).all()))                   # 复位只会减小


class CombinedValidityTests(unittest.TestCase):
    def test_all_switches_respect_the_tight_bound(self):
        """小规模合成核对：纯泊松背景、"追着上一窗事件跑"的对手预测，三个开关同时打开，
        V = 9 个假设的位置级告警率不超过紧界 (1 + 1/L)/(e^θ - 1)（并集界会是它的 9 倍）。"""
        gen = torch.Generator().manual_seed(11)
        size, steps, mu, theta = 24, 300, 0.05, 4.0
        tight = (1.0 + 1.0 / steps) / (math.exp(theta) - 1.0)
        cusum = DriftCUSUM(velocity_grid([-1.0, 0.0, 1.0]), footprint=3, track_decay=0.8, memory_gain=0.5,
                           gate_eps=0.01, reset_radius=1)
        state = cusum.init_state(1, size, size, "cpu", torch.float64)
        prev = torch.zeros(1, 1, size, size, dtype=torch.float64)
        alarms = 0
        for k in range(steps):
            counts = torch.poisson(torch.full((1, 1, size, size), mu, dtype=torch.float64), generator=gen)
            state, _, alarm = cusum.step(state, counts, torch.full_like(counts, mu),
                                         torch.log(0.5 * prev + 1e-3) if k > 0 else None,
                                         theta=theta, reset_on_alarm=True)
            alarms += int(alarm.sum())
            prev = counts
        self.assertLessEqual(alarms / float(steps * size * size), tight)


class ActivityAndOperationsTests(unittest.TestCase):
    def test_decision_activity_fractions(self):
        import train_stream_v2 as tv2
        cusum = DriftCUSUM([(0.0, 0.0), (0.0, 1.0)], footprint=3)
        G = torch.zeros(1, 2, 5, 5, dtype=torch.float64)
        G[0, 0, 2, 2] = 0.05                                     # 一个 (假设, 像素) 活跃，足迹膨胀后 9 个
        activity = tv2.DecisionActivity([1e-2, 1e-1], cusum)
        activity.update({"G": G, "C": torch.zeros_like(G)})
        out = activity.summary()
        self.assertEqual(out["eps"], [1e-2, 1e-1])
        self.assertAlmostEqual(out["active_fraction"][0], 1 / 50.0)
        self.assertAlmostEqual(out["footprint_active_fraction"][0], 9 / 50.0)
        self.assertAlmostEqual(out["active_fraction"][1], 0.0)
        gated = tv2.DecisionActivity([1e-3, 1e-2, 1e-1], DriftCUSUM([(0.0, 0.0)], gate_eps=0.02))
        self.assertEqual(gated.eps, [0.02, 0.1])                  # 低于当前门控的档位没有意义

    def test_sparse_operations_scale_down(self):
        cusum = DriftCUSUM(velocity_grid([-1.0, 0.0, 1.0]), footprint=3, track_decay=0.8)
        dense = cusum.estimate_operations(64, 64, 500.0, (1, 2), 3)
        same = cusum.estimate_operations(64, 64, 500.0, (1, 2), 3, active_fraction=1.0)
        for key in ("mac", "elementwise", "transcendental"):
            self.assertEqual(dense[key], same[key])
        sparse = cusum.estimate_operations(64, 64, 500.0, (1, 2), 3, active_fraction=0.01)
        self.assertLess(sparse["elementwise"], 0.1 * dense["elementwise"])
        self.assertLess(sparse["transcendental"], 0.1 * dense["transcendental"])
        with self.assertRaises(ValueError):
            cusum.estimate_operations(64, 64, active_fraction=1.5)

    def test_new_switches_add_their_own_cost(self):
        base = DriftCUSUM(velocity_grid([0.0, 1.0]), footprint=3, track_decay=0.8).estimate_operations(32, 32)
        for kwargs in (dict(memory_gain=0.5), dict(gate_eps=0.01)):
            ops = DriftCUSUM(velocity_grid([0.0, 1.0]), footprint=3, track_decay=0.8, **kwargs).estimate_operations(32, 32)
            self.assertGreater(ops["elementwise"], base["elementwise"], str(kwargs))


def toy_sequence(t, x, y, label, tid):
    order, bounds = sw.split_windows(t, 50, 60)
    bins, local = sw.time_bin_and_local(t, 50, 5)
    return sw.StreamSequence("toy.npz", x, y, t, np.ones(len(t), np.int8), label, tid, bins, local, order, bounds)


class AlarmDiagnosticsTests(unittest.TestCase):
    def test_distance_repeat_dedup_and_pure_windows(self):
        from utils.alarm_metrics import AlarmEvaluator
        # 目标只在第 30~31 窗出现在 (10, 10)；背景事件若干
        t = np.array([1520, 1530, 1560, 5], dtype=np.int64)
        x = np.array([10, 11, 10, 60], dtype=np.int64)
        y = np.array([10, 10, 11, 40], dtype=np.int64)
        label = np.array([1, 1, 1, 0], dtype=np.float32)
        tid = np.array([1, 1, 1, 0], dtype=np.float64)
        seq = toy_sequence(t, x, y, label, tid)
        ev = AlarmEvaluator([5.0], 48, 64, 50.0, 2, hypotheses=9, history_windows=20, pure_windows=20, dedup_radius=8)
        ev.begin_sequence(seq, 60)
        a = np.zeros((48, 64), dtype=bool)
        a[40, 60] = True                          # 第 5 窗：纯背景窗（目标还没出现），远离任何目标
        ev.update(5.0, 5, a.copy())
        ev.update(5.0, 6, a.copy())               # 第 6 窗同一位置：重复位置、且不是新的虚警事件
        a[:] = False
        a[10, 22] = True                          # 第 31 窗：离目标 12 像素（近处漂移告警）
        ev.update(5.0, 31, a.copy())
        a[:] = False
        a[10, 11] = True                          # 第 31 窗：真告警
        ev.update(5.0, 31, a.copy())
        a[:] = False
        a[30, 40] = True                          # 第 45 窗：目标 14 窗前出现过，距离 30 像素
        ev.update(5.0, 45, a.copy())
        ev.end_sequence()
        row = ev.summary()["5.0"]
        self.assertEqual(row["false_components"], 4)
        dist = row["false_distance_to_recent_target"]
        self.assertEqual(dist["no_target_recently"], 2)
        self.assertEqual(dist["<=16"], 1)
        self.assertEqual(dist["<=32"], 1)
        self.assertEqual(row["false_repeat_components"], 1)
        self.assertEqual(row["false_events"], 3)
        # 纯背景窗：第 0~29 窗（30 个），以及第 52 窗起（之前 20 窗无目标）的 8 个
        self.assertEqual(row["pure_windows"], 30 + 8)
        self.assertEqual((row["pure_start_windows"], row["pure_after_windows"]), (30, 8))
        self.assertEqual((row["pure_start_alarm_pixels"], row["pure_after_alarm_pixels"]), (2, 0))
        self.assertGreater(row["pure_start_violation_p"], 0.5)          # 2 个连通域远少于紧界下的期望数
        self.assertEqual(row["pure_alarm_pixels"], 2)
        self.assertAlmostEqual(row["pure_alarm_location_rate"], 2.0 / (38 * 48 * 64))
        self.assertAlmostEqual(row["tight_bound_per_location_window"], (1 + 1 / 60.0) / (math.exp(5.0) - 1.0))
        self.assertAlmostEqual(row["bound_per_location_window"], 9 / (math.exp(5.0) - 1.0))
        self.assertEqual(row["n_detected"], 1)


class AdaptivePublishTests(unittest.TestCase):
    def setUp(self):
        from tools.adaptive_publish import publish_decisions
        self.decide = publish_decisions
        self.logit = np.array([3.0, -3.0, 0.2, 0.2, -0.3])
        self.evidence = {1: np.array([0.0, 0.0, 2.0, -0.1, 0.1]), 2: np.array([0.0, 0.0, 2.5, -2.0, 0.2]),
                         5: np.array([0.0, 0.0, 4.0, -3.0, 0.1])}

    def test_zero_delta_is_the_network_output(self):
        pred, wait = self.decide(self.logit, self.evidence, [1, 2, 5], 1.0, 0.0, 0.0, 5)
        self.assertEqual(pred.tolist(), (self.logit >= 0).tolist())
        self.assertEqual(wait.tolist(), [0] * 5)

    def test_huge_delta_is_fixed_max_delay(self):
        pred, wait = self.decide(self.logit, self.evidence, [1, 2, 5], 1.0, 0.0, 1e9, 2)
        self.assertEqual(wait.tolist(), [2] * 5)
        self.assertEqual(pred.tolist(), ((self.logit + self.evidence[2]) >= 0).tolist())

    def test_stops_at_first_decisive_delay(self):
        pred, wait = self.decide(self.logit, self.evidence, [1, 2, 5], 1.0, 0.0, 1.0, 5)
        # 事件 0/1 一眼能判；事件 2 在 d=1 时 z=2.2；事件 3 在 d=2 时 z=-1.8；事件 4 始终犹豫，等到 d_max=5
        self.assertEqual(wait.tolist(), [0, 0, 1, 2, 5])
        self.assertEqual(pred.tolist(), [True, False, True, False, False])
class PublishLatencyTests(unittest.TestCase):
    def test_first_detection_uses_publish_time(self):
        from tools.adaptive_publish import first_detection_latencies
        seq = {"locs": np.array([[0, 0, 0, 60], [0, 0, 0, 70], [0, 0, 0, 130], [0, 0, 0, 10]]),
               "labels": np.array([1, 1, 1, 0], np.float32), "target_id": np.array([1.0, 1.0, 1.0, 0.0])}
        pred = np.array([False, True, True, True])
        # 第 2 个事件（第 1 窗）等 2 窗 -> 第 3 窗末 200 ms 发布；第 3 个事件（第 2 窗）不等 -> 第 2 窗末 150 ms：取更早的
        wait = np.array([0, 2, 0, 0])
        self.assertEqual(first_detection_latencies(seq, pred, wait, 50, 160), [150.0 - 60.0])
        self.assertEqual(first_detection_latencies(seq, np.zeros(4, bool), wait, 50, 160), [None])
        # 发布窗在序列末尾截断
        self.assertEqual(first_detection_latencies(seq, pred, np.array([0, 9, 9, 0]), 50, 3), [150.0 - 60.0])


class CommandLineTests(unittest.TestCase):
    def test_switch_flags_reach_the_decision_layer(self):
        import train_stream_v2 as tv2
        argv = ["train_stream_v2.py", "--config", CONFIG, "--mode", "eval", "--cusum-memory-gain", "0.5",
                "--cusum-gate-eps", "0.01", "--cusum-reset-radius", "2",
                "--cusum-footprint", "5", "--alarm-thetas", "6", "8", "--eval-state-modes", "carry",
                "--eval-sequences", "2"]
        with mock.patch.object(sys, "argv", argv):
            args = tv2.parse_args()
        cfg = tv2.build_config(args)
        cusum = tv2.build_cusum(cfg)
        self.assertEqual((cusum.memory_gain, cusum.gate_eps, cusum.reset_radius, cusum.footprint), (0.5, 0.01, 2, 5))
        self.assertEqual(cfg["alarm_thetas"], [6.0, 8.0])
        self.assertEqual(args.eval_state_modes, ["carry"])
        self.assertEqual(args.eval_sequences, 2)

    def test_yaml_defaults_are_the_final_v2_decision_layer(self):
        """V2 定稿（09-26）：判决层默认稀疏执行 ε = 0.03、不算告警、只跑 carry；--cusum-gate-eps 0 回到原实现。"""
        import train_stream_v2 as tv2
        argv = ["train_stream_v2.py", "--config", CONFIG, "--mode", "eval"]
        with mock.patch.object(sys, "argv", argv):
            args = tv2.parse_args()
        cfg = tv2.build_config(args)
        cusum = tv2.build_cusum(cfg)
        self.assertEqual((cusum.memory_gain, cusum.gate_eps, cusum.reset_radius, cusum.aggregate), (1.0, 0.03, 0, "lme"))
        self.assertEqual(cfg["alarm_thetas"], [])
        self.assertEqual(cfg["eval_state_modes"], ["carry"])
        self.assertIsNone(args.eval_state_modes)
        self.assertEqual(args.eval_sequences, 0)
        with mock.patch.object(sys, "argv", argv + ["--cusum-gate-eps", "0"]):
            self.assertEqual(tv2.build_cusum(tv2.build_config(tv2.parse_args())).gate_eps, 0.0)


if __name__ == "__main__":
    unittest.main()
