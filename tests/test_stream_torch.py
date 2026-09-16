"""LIF / 网络 / 校准 / 训练步骤的单元测试（需要 torch，CPU 即可运行，不需要数据集）。

服务器上运行: python -m pytest tests/test_stream_torch.py -q
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from dataset import stream_windows as sw  # noqa: E402
from model.evspsegnet_stream import (LAYER_NAMES, EvSpSegNetStream, calibrate_gains,  # noqa: E402
                                     count_parameters, estimate_operations)
from model.lif2d_stream import ChannelGain, StreamingLIF2d, detach_states  # noqa: E402

H, W = 264, 352


def lif(**kw):
    """构造一个默认参数的 LIF（dt=50, tau0=200, 范围 [50,2000], 阈值 1）。"""
    args = dict(dt_ms=50.0, tau_init_ms=200.0, tau_min_ms=50.0, tau_max_ms=2000.0, v_threshold=1.0)
    args.update(kw)
    return StreamingLIF2d(3, **args)


def random_events(n, seed, device="cpu"):
    """生成 n 个随机事件的读出输入字典。"""
    g = torch.Generator().manual_seed(seed)
    return {"b": torch.zeros(n, dtype=torch.long),
            "y": torch.randint(0, 260, (n,), generator=g),
            "x": torch.randint(0, 346, (n,), generator=g),
            "p": torch.randint(0, 2, (n,), generator=g).float(),
            "t_local": torch.rand(n, generator=g)}


def small_net(seed=0, dtype=torch.float32, **kw):
    """固定种子构造网络。"""
    torch.manual_seed(seed)
    return EvSpSegNetStream(**kw).to(dtype)


def random_input(seed, scale=3.0, dtype=torch.float32, density=0.02):
    """生成稀疏的非负输入 [1,12,H,W]（模拟事件计数图）。"""
    g = torch.Generator().manual_seed(seed)
    mask = (torch.rand(1, 12, H, W, generator=g) < density).to(dtype)
    return mask * torch.rand(1, 12, H, W, generator=g).to(dtype) * scale


class LIFTests(unittest.TestCase):
    def test_tau_init_and_beta(self):
        n = lif()
        self.assertTrue(torch.allclose(n.tau(), torch.full((3,), 200.0), atol=1e-3))
        self.assertTrue(torch.allclose(n.beta(), torch.full((3,), math.exp(-0.25)), atol=1e-5))

    def test_zero_input_pure_decay(self):
        n = lif().double()
        state = torch.rand(1, 3, 4, 4, dtype=torch.float64) * 0.9
        spikes, new_state, u_pre = n(torch.zeros_like(state), state)
        beta = n.beta().double().view(1, -1, 1, 1)
        self.assertTrue(torch.allclose(u_pre, beta * state))
        self.assertEqual(float(spikes.sum()), 0.0)
        self.assertTrue(torch.allclose(new_state, u_pre))

    def test_spike_and_soft_reset(self):
        n = lif()
        current = torch.tensor([1.5, 0.99, 1.0]).view(1, 3, 1, 1)
        spikes, new_state, u_pre = n(current, None)
        self.assertEqual(spikes.view(-1).tolist(), [1.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(new_state.view(-1), torch.tensor([0.5, 0.99, 0.0])))
        self.assertTrue(torch.allclose(u_pre, current))

    def test_surrogate_gradient_value(self):
        n = lif()
        current = torch.tensor([1.0, 2.0, 0.0]).view(1, 3, 1, 1).requires_grad_()
        spikes, _, _ = n(current, None)
        spikes.sum().backward()
        self.assertTrue(torch.allclose(current.grad.view(-1), torch.tensor([1.0, 0.25, 0.25])))

    def test_tau_parameter_receives_gradient(self):
        n = lif()
        _, state, _ = n(torch.full((1, 3, 2, 2), 0.6), None)
        _, _, u_pre = n(torch.zeros(1, 3, 2, 2), state)
        u_pre.sum().backward()
        self.assertTrue(bool((n.a.grad.abs() > 0).all()))

    def test_detach_keeps_values(self):
        x = torch.rand(2, 2, requires_grad=True) * 3
        out = detach_states([x, None])
        self.assertIsNone(out[1])
        self.assertFalse(out[0].requires_grad)
        self.assertTrue(torch.equal(out[0], x.detach()))

    def test_gain_scale_only(self):
        g = ChannelGain(3)
        g.set_gain(torch.tensor([0.5, 2.0, 20.0]))
        self.assertEqual(float(g(torch.zeros(1, 3, 2, 2)).abs().sum()), 0.0)
        self.assertTrue(torch.allclose(g.gain(), torch.tensor([0.5, 2.0, 20.0])))
        with self.assertRaises(ValueError):
            g.set_gain(torch.tensor([1.0, 0.0, 1.0]))


class NetworkTests(unittest.TestCase):
    def test_parameter_count_matches_design(self):
        # 卷积 104400 + 读出 513 + 增益 216 + tau 参数 216
        self.assertEqual(count_parameters(small_net())["total"], 105345)

    def test_shapes_and_empty_window(self):
        net = small_net()
        logits, states, _ = net(random_input(1), random_events(50, 1), None)
        self.assertEqual(tuple(logits.shape), (50,))
        self.assertEqual(len(states), 7)
        empty, _, _ = net(torch.zeros(1, 12, H, W), random_events(0, 2), states)
        self.assertEqual(tuple(empty.shape), (0,))

    def test_one_neuron_call_per_layer_per_window(self):
        net = small_net()
        calls = {name: 0 for name in LAYER_NAMES}
        hooks = [getattr(net, name).neuron.register_forward_hook(
            lambda m, i, o, name=name: calls.__setitem__(name, calls[name] + 1)) for name in LAYER_NAMES]
        net(random_input(3), random_events(10, 3), None)
        for h in hooks:
            h.remove()
        self.assertEqual(set(calls.values()), {1})

    def test_no_membrane_in_state_dict(self):
        net = small_net()
        net(random_input(4), random_events(10, 4), None)
        buffers = [name for name, _ in net.named_buffers()]
        self.assertEqual(buffers, ["gain_calibrated"])
        self.assertFalse(any("state" in k or k.endswith(".U") for k in net.state_dict()))

    def test_zero_input_produces_no_spikes(self):
        net = small_net()
        _, states, info = net(torch.zeros(1, 12, H, W), random_events(5, 5), None, collect=True)
        self.assertEqual(sum(float(s.sum()) for s in info["spikes"]), 0.0)
        self.assertEqual(sum(float(s.abs().sum()) for s in states), 0.0)

    def active_net(self):
        """把所有层增益调大，保证状态非零，用于检验 carry / reset 行为。"""
        net = small_net(seed=7)
        for block in net.blocks():
            block.gain.set_gain(torch.full((block.out_ch,), 8.0))
        return net

    def test_reset_each_window_ignores_previous_state(self):
        net = self.active_net()
        ev = random_events(64, 6)
        _, states, _ = net(random_input(6), ev, None)
        a, _, _ = net(random_input(7), ev, states, state_mode="reset_each_window")
        b, _, _ = net(random_input(7), ev, None, state_mode="reset_each_window")
        self.assertTrue(torch.equal(a, b))

    def test_carry_uses_previous_state(self):
        net = self.active_net()
        ev = random_events(256, 8)
        _, states, _ = net(random_input(8), ev, None)
        self.assertGreater(sum(float(s.abs().sum()) for s in states), 0.0)
        a, _, _ = net(random_input(9), ev, states, state_mode="carry")
        b, _, _ = net(random_input(9), ev, None, state_mode="carry")
        self.assertFalse(torch.equal(a, b))

    def test_relu_variant_runs(self):
        net = small_net(neuron="relu")
        logits, states, _ = net(random_input(10), random_events(20, 10), None)
        self.assertEqual(tuple(logits.shape), (20,))
        self.assertTrue(all(s is None for s in states))


class GradientAccumulationTests(unittest.TestCase):
    def test_chunk_backward_equals_single_graph(self):
        """逐片段 backward 累加梯度，必须等于"同样 detach 边界"下一次性 backward 的梯度。"""
        inputs = [random_input(20 + k, dtype=torch.float64) for k in range(6)]
        events = [random_events(40, 30 + k) for k in range(6)]
        labels = [(torch.rand(40, generator=torch.Generator().manual_seed(40 + k)) < 0.2).double()
                  for k in range(6)]
        chunks = [(0, 2), (2, 6)]
        total = 240.0

        def loss_of(net, k, states):
            logits, states, _ = net(inputs[k], events[k], states)
            return torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels[k], pos_weight=torch.tensor([5.0], dtype=torch.float64),
                reduction="sum"), states

        net_a = small_net(seed=3, dtype=torch.float64)
        states = None
        for a, b in chunks:
            chunk = 0.0
            for k in range(a, b):
                term, states = loss_of(net_a, k, states)
                chunk = chunk + term
            (chunk / total).backward()
            states = detach_states(states)

        net_b = small_net(seed=3, dtype=torch.float64)
        states, whole = None, 0.0
        for a, b in chunks:
            for k in range(a, b):
                term, states = loss_of(net_b, k, states)
                whole = whole + term
            states = detach_states(states)
        (whole / total).backward()

        for (name, pa), (_, pb) in zip(net_a.named_parameters(), net_b.named_parameters()):
            if pa.grad is None:
                self.assertIsNone(pb.grad, name)
                continue
            self.assertTrue(torch.allclose(pa.grad, pb.grad, rtol=1e-9, atol=1e-12), name)


class CalibrationTests(unittest.TestCase):
    def factory(self):
        """两段序列，每段 3 个连续窗口的合成校准数据。"""
        def sequence(offset):
            for k in range(3):
                yield random_input(100 + offset + k), random_events(30, 200 + offset + k)
        for offset in (0, 10):
            yield sequence(offset)

    def test_deterministic_bounded_and_once(self):
        gains = []
        for _ in range(2):
            net = small_net(seed=11)
            calibrate_gains(net, self.factory, 0.99, 256, 64, 0.05, 20.0, seed=5)
            self.assertTrue(bool(net.gain_calibrated))
            g = torch.cat([b.gain.gain().detach() for b in net.blocks()])
            self.assertTrue(bool(((g >= 0.05 - 1e-6) & (g <= 20.0 + 1e-5)).all()))
            gains.append(g)
            with self.assertRaises(RuntimeError):
                calibrate_gains(net, self.factory, 0.99, 256, 64, 0.05, 20.0, seed=5)
        self.assertTrue(torch.equal(gains[0], gains[1]))

    def test_operation_estimate_consistency(self):
        net = small_net()
        ops = estimate_operations(net, H, W, {name: 1.0 for name in LAYER_NAMES}, 100)
        self.assertAlmostEqual(ops["mac"] + ops["sop"], ops["dense_ops"], delta=1e-3)
        zero = estimate_operations(net, H, W, {name: 0.0 for name in LAYER_NAMES}, 100)
        self.assertEqual(zero["sop"], 0.0)


class TrainSequenceTests(unittest.TestCase):
    def synthetic_sequence(self, n=3000):
        """构造一个合成的 StreamSequence（不读磁盘）。"""
        rng = np.random.RandomState(0)
        x = rng.randint(0, 346, n).astype(np.int64)
        y = rng.randint(0, 260, n).astype(np.int64)
        t = rng.randint(0, 8000, n).astype(np.int64)
        p = rng.randint(0, 2, n).astype(np.int8)
        label = (rng.rand(n) < 0.1).astype(np.float32)
        order, bounds = sw.split_windows(t, 50, 160)
        inner_bin, t_local = sw.time_bin_and_local(t, 50, 5)
        return sw.StreamSequence("synthetic.npz", x, y, t, p, label, label.astype(np.float64),
                                 inner_bin, t_local, order, bounds)

    def test_step_counts(self):
        import train_stream_v1 as T
        cfg = {"tbptt_k": 16, "grad_clip": 1.0, "time_bins": 5, "pad_height": H, "pad_width": W,
               "input_clip": 3.0, "optimizer_step": "per_sequence"}
        seq, q99 = self.synthetic_sequence(), np.full(12, math.log(2.0), dtype=np.float32)
        for mode, expect_one in (("per_sequence", True), ("per_chunk", False)):
            cfg["optimizer_step"] = mode
            net = small_net(seed=1)
            before = [p.detach().clone() for p in net.parameters()]
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            out = T.train_sequence(net, seq, opt, cfg, q99, 5.0, torch.device("cpu"),
                                   np.random.RandomState(0), max_windows=40)
            self.assertTrue(math.isfinite(out["loss_sum"]))
            self.assertEqual(out["steps"] == 1, expect_one)
            self.assertGreater(out["steps"], 0)
            changed = any(not torch.equal(a, b.detach()) for a, b in zip(before, net.parameters()))
            self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
