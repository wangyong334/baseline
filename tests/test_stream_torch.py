"""V5 BIR readout/state/gradient/calibration numerical regressions for V15."""

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


H, W = 16, 24


def random_events(n, seed, device="cpu"):
    """生成 n 个随机事件的读出输入字典。"""
    g = torch.Generator().manual_seed(seed)
    return {"b": torch.zeros(n, dtype=torch.long),
            "y": torch.randint(0, H, (n,), generator=g),
            "x": torch.randint(0, W, (n,), generator=g),
            "p": torch.randint(0, 2, (n,), generator=g).float(),
            "t_local": torch.rand(n, generator=g)}


def small_net(seed=0, dtype=torch.float32, **kw):
    """固定种子构造网络。"""
    torch.manual_seed(seed)
    options = dict(channels=(2, 4, 4, 4), neuron="tif", v_threshold=0.05, tif={"k_init": 0.001})
    options.update(kw)
    net = EvSpSegNetStream(**options).to(dtype)
    for block in net.blocks():
        block.gain.set_gain(torch.full((block.out_ch,), 20.0, dtype=dtype))
    return net


def random_input(seed, scale=3.0, dtype=torch.float32, density=0.02):
    """生成稀疏的非负输入 [1,12,H,W]（模拟事件计数图）。"""
    g = torch.Generator().manual_seed(seed)
    mask = (torch.rand(1, 12, H, W, generator=g) < density).to(dtype)
    return mask * torch.rand(1, 12, H, W, generator=g).to(dtype) * scale


class StateAndGainTests(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
