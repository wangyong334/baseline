"""膜电位上下界（件 1）的单元测试。

要守住三条：
    1. 两个界都为 None 时与旧实现**逐位相同**——现有三种子的结果必须原样可复现
    2. forward 与 forward_steps 仍然等价（跨窗状态被夹住，最容易在这里出现不一致）
    3. 下界确实解决它要解决的问题：持续负电流下，无界的膜电位按 1/(1-beta) 发散，
       有界时停在 u_floor，且一旦来正电流能在一窗内恢复发放
"""
import unittest

import torch

from model.evidence_snn import EvidenceSNN
from model.lif2d_stream import StreamingGraded2d, StreamingLIF2d

KW = dict(dt_ms=50.0, tau_init_ms=200.0, tau_min_ms=50.0, tau_max_ms=2000.0, v_threshold=1.0)
B, C, H, W = 2, 3, 5, 7


def neuron(**extra):
    torch.manual_seed(0)
    return StreamingLIF2d(C, **dict(KW, **extra)).double()


def max_error(a, b):
    return float((a.detach() - b.detach()).abs().max())


class DefaultIsIdenticalTests(unittest.TestCase):
    def test_no_bounds_matches_old_behaviour(self):
        """不设界时 bound_state 是恒等映射，逐窗输出与手写的旧递推逐位相同。"""
        lif = neuron()
        torch.manual_seed(1)
        current = torch.randn(6, B, C, H, W, dtype=torch.float64)
        state, manual = None, None
        for t in range(6):
            spikes, state, u_pre = lif(current[t], state)
            beta = lif.beta().to(current.dtype).view(1, -1, 1, 1)
            u_manual = current[t] if manual is None else beta * manual + current[t]
            fired = (u_manual >= lif.v_threshold).to(u_manual.dtype)
            manual = u_manual - lif.v_threshold * fired
            self.assertEqual(max_error(u_pre, u_manual), 0.0)
            self.assertEqual(max_error(state, manual), 0.0)
            self.assertEqual(max_error(spikes, fired), 0.0)

    def test_bound_state_is_identity_when_unset(self):
        lif = neuron()
        x = torch.randn(4, 5, dtype=torch.float64) * 100.0
        self.assertIs(lif.bound_state(x), x)


class EquivalenceTests(unittest.TestCase):
    def test_forward_steps_matches_forward_with_bounds(self):
        """有界时逐窗 forward 与片段 forward_steps 仍然等价（两条路径都要夹）。"""
        for extra in (dict(u_floor=-1.0), dict(u_ceil=3.0), dict(u_floor=-1.0, u_ceil=3.0)):
            lif = neuron(**extra)
            torch.manual_seed(2)
            current = torch.randn(7, B, C, H, W, dtype=torch.float64) * 3.0
            spikes_c, state_c, u_pre_c = lif.forward_steps(current, None)
            state = None
            for t in range(7):
                spikes, state, u_pre = lif(current[t], state)
                self.assertLess(max_error(spikes, spikes_c[t]), 1e-12, str(extra))
                self.assertLess(max_error(u_pre, u_pre_c[t]), 1e-12, str(extra))
            self.assertLess(max_error(state, state_c), 1e-12, str(extra))

    def test_reset_mode_ignores_state(self):
        """carry=False 时不使用上一状态，加不加界都一样。"""
        lif = neuron(u_floor=-1.0)
        torch.manual_seed(3)
        current = torch.randn(4, B, C, H, W, dtype=torch.float64)
        spikes, _, u_pre = lif.forward_steps(current, None, carry=False)
        self.assertLess(max_error(u_pre, current), 1e-12)

    def test_graded_neuron_inherits_bounds(self):
        """graded 对照继承同一套界（它只改发放幅值）。"""
        torch.manual_seed(0)
        g = StreamingGraded2d(C, **dict(KW, u_floor=-1.0)).double()
        self.assertAlmostEqual(g.u_floor, -1.0, places=12)


class FloorFixesDeepSuppressionTests(unittest.TestCase):
    def test_unbounded_membrane_diverges_to_dc_gain(self):
        """无界时持续负电流把膜电位压到 I/(1-beta)，这正是诊断里 -32/-63 个阈值的来源。"""
        lif = neuron()
        beta = float(lif.beta()[0])
        current = torch.full((1, 1, 1, 1), -2.0, dtype=torch.float64)
        state = None
        for _ in range(200):
            _, state, _ = lif(current.expand(1, C, 1, 1), state)
        self.assertLess(float(state.detach().min()), -2.0 / (1.0 - beta) * 0.95)
        self.assertGreater(1.0 / (1.0 - beta), 4.0)          # 直流增益确实是几倍

    def test_floor_stops_the_divergence(self):
        """有下界时膜电位停在 u_floor，不再随负电流继续下沉。"""
        lif = neuron(u_floor=-1.0)
        current = torch.full((1, C, 1, 1), -2.0, dtype=torch.float64)
        state = None
        for _ in range(200):
            _, state, _ = lif(current, state)
        self.assertAlmostEqual(float(state.detach().min()), -1.0, places=12)
        self.assertAlmostEqual(float(state.detach().max()), -1.0, places=12)

    def test_floor_allows_recovery_within_one_window(self):
        """被压死之后给一次正电流：有下界的能立刻发放，无界的还差得很远。"""
        deep = torch.full((1, C, 1, 1), -2.0, dtype=torch.float64)
        wake = torch.full((1, C, 1, 1), 2.0, dtype=torch.float64)
        results = {}
        for name, extra in (("无界", {}), ("有下界", dict(u_floor=-1.0))):
            lif = neuron(**extra)
            state = None
            for _ in range(200):
                _, state, _ = lif(deep, state)
            spikes, _, u_pre = lif(wake, state)
            results[name] = (float(spikes.detach().sum()), float(u_pre.detach().max()))
        self.assertEqual(results["无界"][0], 0.0)
        self.assertEqual(results["有下界"][0], float(C))
        self.assertLess(results["无界"][1], -3.0)
        self.assertGreaterEqual(results["有下界"][1], 1.0)

    def test_ceiling_bounds_the_positive_side(self):
        """上界夹住持续正电流下的膜电位（dec2/dec1 的双向发散用得上）。"""
        lif = neuron(u_ceil=3.0)
        current = torch.full((1, C, 1, 1), 5.0, dtype=torch.float64)
        state = None
        for _ in range(50):
            _, state, _ = lif(current, state)
        self.assertLessEqual(float(state.detach().max()), 3.0 + 1e-12)


class ModelWiringTests(unittest.TestCase):
    def test_evidence_snn_passes_bounds_to_every_block(self):
        """配置项要真的传到七个脉冲单元上，而不是只改了构造签名。"""
        model = EvidenceSNN(6, channels=(4, 4, 4, 4), u_floor=-1.5, u_ceil=4.0)
        blocks = model.blocks()
        self.assertEqual(len(blocks), 7)
        for block in blocks:
            self.assertAlmostEqual(block.neuron.u_floor, -1.5, places=12)
            self.assertAlmostEqual(block.neuron.u_ceil, 4.0, places=12)
        default = EvidenceSNN(6, channels=(4, 4, 4, 4))
        for block in default.blocks():
            self.assertIsNone(block.neuron.u_floor)
            self.assertIsNone(block.neuron.u_ceil)

    def test_bounds_scale_with_v_threshold(self):
        """界以 v_th 为单位给定，构造时折算成绝对值。"""
        lif = StreamingLIF2d(C, **dict(KW, v_threshold=0.5, u_floor=-2.0, u_ceil=3.0))
        self.assertAlmostEqual(lif.u_floor, -1.0, places=12)
        self.assertAlmostEqual(lif.u_ceil, 1.5, places=12)

    def test_rejects_inverted_bounds(self):
        with self.assertRaises(ValueError):
            StreamingLIF2d(C, **dict(KW, u_floor=2.0, u_ceil=1.0))

    def test_rejects_positive_floor_and_nonpositive_ceiling(self):
        """符号检查：漏了负号（--neuron-u-floor 1）、上界为 0 或为负都直接报错。"""
        for extra in (dict(u_floor=1.0), dict(u_ceil=0.0), dict(u_ceil=-4.0)):
            with self.assertRaises(ValueError, msg=str(extra)):
                StreamingLIF2d(C, **dict(KW, **extra))

    def test_zero_floor_is_allowed(self):
        """下界 0 就是判决层 CUSUM 的静息地板 max(0, C)，是合法设置。"""
        lif = StreamingLIF2d(C, **dict(KW, u_floor=0.0))
        self.assertEqual(lif.u_floor, 0.0)

    def test_relu_control_ignores_bounds(self):
        """ReLU 对照没有膜电位，传了界也不应报错（它吃 **kwargs）。"""
        model = EvidenceSNN(6, channels=(4, 4, 4, 4), neuron="relu", u_floor=-1.0)
        x = torch.randn(1, 6, 8, 8)
        mark, log_g, states, _ = model(x, None)
        self.assertEqual(mark.shape[-2:], x.shape[-2:])
        self.assertIsNone(states[0])


if __name__ == "__main__":
    unittest.main()
