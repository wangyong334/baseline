"""V5 numerical regression retained for the V15 release; legacy variants removed."""
import copy


import os


import sys


import unittest


import numpy as np


import torch


import torch.nn.functional as F


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from dataset.stream_source import TorchWindowSource  # noqa: E402


from model.evspsegnet_stream import (LAYER_NAMES, TIF_DEFAULTS, EvSpSegNetStream,  # noqa: E402
                                     calibrate_gains, estimate_operations, tif_settings)


from model.lif2d_stream import BoundedSurrogate, StreamingTIF2d, inverse_softplus  # noqa: E402


from test_stream_layerwise import (CFG, CHUNKS, H, Q99, SMALL, W, max_error, rel_error,  # noqa: E402
                                   synthetic_sequence)


LIF_ARGS = dict(dt_ms=50.0, tau_init_ms=200.0, tau_min_ms=50.0, tau_max_ms=2000.0)


ACTIVE_TIF = {"k_init": 1e-3}


def tif_neuron(channels=3, v_threshold=1.0, **kw):
    """默认参数的 float64 TIF 神经元（dt=50, tau0=200, 范围 [50,2000]）。"""
    return StreamingTIF2d(channels, v_threshold=v_threshold, **dict(LIF_ARGS, **kw)).double()


def random_map(generator, shape, scale=1.0, density=0.5):
    """非负稀疏随机图。"""
    mask = (torch.rand(shape, generator=generator, dtype=torch.float64) < density).double()
    return mask * torch.rand(shape, generator=generator, dtype=torch.float64) * scale


def coefficients(n):
    view = (1, -1, 1, 1)
    return (n.beta().double().view(view), n.k().double().view(view), n.theta().double().view(view))


class TIFDefinitionTests(unittest.TestCase):
    def test_initial_parameters(self):
        n = tif_neuron(3)
        lam, k, theta = coefficients(n)
        self.assertLess(max_error(lam, torch.full_like(lam, float(np.exp(-0.25)))), 1e-6)
        self.assertLess(max_error(k, torch.full_like(k, 0.05)), 1e-6)
        self.assertLess(max_error(theta, torch.full_like(theta, 1.0 / 7)), 1e-6)
        self.assertEqual(sorted(name for name, _ in n.named_parameters()), ["a", "k_raw", "log_theta"])
        for value in (1e-3, 0.05, 2.0):
            self.assertAlmostEqual(float(F.softplus(torch.tensor(inverse_softplus(value), dtype=torch.float64))),
                                   value, places=12)
        with self.assertRaises(ValueError):
            tif_neuron(3, levels=0)
        with self.assertRaises(ValueError):
            tif_neuron(3, quantize="yes")
        for removed in ({"transport": False}, {"reference": False}):           # 搬运、本地参照已删除
            with self.assertRaises(TypeError):
                tif_neuron(3, **removed)

    def test_rest_and_none_equals_zero_state(self):
        n = tif_neuron(3)
        zero = torch.zeros(2, 3, 4, 5, dtype=torch.float64)
        for state in (None, zero):
            out, memory, _ = n(zero, state)
            self.assertEqual(float(out.abs().max()), 0.0)
            self.assertEqual(float(memory.abs().max()), 0.0)
        current = torch.randn(2, 3, 4, 5, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        for x, y in zip(n(current, None), n(current, zero)):
            self.assertEqual(max_error(x, y), 0.0)

    def test_reflected_accumulator(self):
        n = tif_neuron(3)
        g = torch.Generator().manual_seed(2)
        prev = random_map(g, (2, 3, 5, 6), scale=0.8)
        current = torch.randn(2, 3, 5, 6, generator=g, dtype=torch.float64)
        lam, k, theta = coefficients(n)
        expected = torch.minimum(torch.relu(lam * prev + (1 - lam) * (current - k)), 7 * theta)
        _, state, memory = n(current, prev)
        self.assertLess(max_error(memory, expected), 1e-14)
        self.assertIs(state, memory)                                           # 不启用参照时状态就是 M

    def test_quantized_levels_float_variant_and_bound(self):
        g = torch.Generator().manual_seed(3)
        for quantize in (True, False):
            n = tif_neuron(3, quantize=quantize)
            _, _, theta = coefficients(n)
            state = None
            for _ in range(4):
                current = random_map(g, (2, 3, 5, 6), scale=4.0)       # 大于量程的输入也有
                out, state, memory = n(current, state)
                self.assertTrue(bool((memory <= 7 * theta + 1e-12).all()))
                self.assertGreaterEqual(float(memory.min()), 0.0)
                if quantize:
                    levels = out / theta
                    self.assertLess(max_error(levels, torch.round(levels)), 1e-9)
                    self.assertLess(max_error(out, theta * torch.round(memory / theta)), 1e-12)
                    self.assertLessEqual(float(levels.max()), 7 + 1e-9)
                else:
                    self.assertEqual(max_error(out, memory), 0.0)
            self.assertGreater(float(memory.max()), 0.0)


class TIFStepsTests(unittest.TestCase):
    def test_forward_steps_matches_forward_with_keep(self):
        from model.lif2d_stream import mask_state
        for quantize in (True, False):
            with self.subTest(quantize=quantize):
                n_a = tif_neuron(3, quantize=quantize)
                n_b = copy.deepcopy(n_a)
                g = torch.Generator().manual_seed(11)
                current = torch.randn(5, 2, 3, 6, 7, generator=g, dtype=torch.float64).clamp(min=-0.2) * 0.8
                keep = torch.ones(5, 2, dtype=torch.float64)
                keep[2, 1] = 0.0                                    # 第 2 条流在第 2 窗开始新录像
                w_out = torch.rand(current.shape, generator=g, dtype=torch.float64)
                w_mem = torch.rand(current.shape, generator=g, dtype=torch.float64)
                state, outs, mems = None, [], []
                for t in range(5):
                    out, state, mem = n_a(current[t], mask_state(state, keep, t))
                    outs.append(out)
                    mems.append(mem)
                outs, mems = torch.stack(outs), torch.stack(mems)
                outs_b, state_b, mems_b = n_b.forward_steps(current.clone(), None, True, True, keep=keep)
                self.assertLess(max_error(outs, outs_b), 1e-12)
                self.assertLess(max_error(mems, mems_b), 1e-12)
                self.assertLess(max_error(state, state_b), 1e-12)
                ((outs * w_out).sum() + (mems * w_mem).sum()).backward()
                ((outs_b * w_out).sum() + (mems_b * w_mem).sum()).backward()
                for (name, pa), (_, pb) in zip(n_a.named_parameters(), n_b.named_parameters()):
                    self.assertIsNotNone(pa.grad, name)
                    self.assertTrue(bool(torch.isfinite(pa.grad).all()), name)
                    self.assertLess(max_error(pa.grad, pb.grad) / max(float(pa.grad.abs().max()), 1.0), 1e-10, name)
                    if quantize or name != "log_theta":     # 不量化时 theta 只经上界起作用，没触顶时梯度为 0
                        self.assertGreater(float(pa.grad.abs().max()), 0.0, name)
                # 不跨窗（reset）：每一步都从零开始
                _, _, mems_r = n_b.forward_steps(current, None, False, True)
                for t in range(5):
                    self.assertLess(max_error(mems_r[t], n_b(current[t], None)[2]), 1e-14)

    def test_statistics_match_manual_count(self):
        n = tif_neuron(2)
        g = torch.Generator().manual_seed(5)
        currents = [random_map(g, (1, 2, 6, 6), scale=1.5) for _ in range(4)]
        n.start_stats()
        state, states = None, []
        for current in currents:
            _, state, memory = n(current, state)
            states.append(memory)
        stats = n.stop_stats()
        self.assertFalse(n.collect_stats)
        _, _, theta = coefficients(n)
        levels = [torch.round(s / theta) for s in states]
        deltas = [levels[0]] + [levels[t] - levels[t - 1] for t in range(1, 4)]
        numel = 4.0 * states[0].numel()
        self.assertAlmostEqual(stats["message_abs_rate"], float(sum(d.abs().sum() for d in deltas)) / numel, places=12)
        self.assertAlmostEqual(stats["message_nonzero_rate"], float(sum(d.ne(0).sum() for d in deltas)) / numel,
                               places=12)
        self.assertEqual(stats["windows"], 4)


def make_tif_net(merged=False, quantize=True, seed=5, threshold=0.05, **extra):
    """float64 小 TIF 网络；阈值（即 theta 初值）与参考值调低，让未校准的小网络各层都有活动。"""
    torch.manual_seed(seed)
    tif = dict(ACTIVE_TIF, quantize=quantize, **extra)
    net = EvSpSegNetStream(channels=SMALL, neuron="tif", v_threshold=threshold,
                           tif=tif).double()
    # 输出量程 D*theta = threshold；与 graded 对照相同，把增益放大 1/threshold，让信号逐层保持量级
    for block in net.blocks():
        block.gain.set_gain(torch.full((block.out_ch,), 1.0 / threshold, dtype=torch.float64))
    return net


class TIFNetworkTests(unittest.TestCase):

    def test_forward_chunk_matches_step_forward(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        cases = [(False, True, {}), (False, False, {}), (False, True, {"surrogate": True})]
        for merged, quantize, extra in cases:
            for mode in ("carry", "reset_each_window"):
                with self.subTest(merged=merged, quantize=quantize, extra=extra, mode=mode):
                    self.check_pair(source, merged, quantize, mode, extra)

    def check_pair(self, source, merged, quantize, mode, extra=None):
        from model.lif2d_stream import state_tensors
        net_a = make_tif_net(merged, quantize, **(extra or {}))
        net_b = copy.deepcopy(net_a)
        g = torch.Generator().manual_seed(3)
        states, logits_a, outs_a, mems_a, chunk_states = None, [], [], [], []
        for start, end in CHUNKS:
            for k in range(start, end):
                x, events, _, _ = source.window(k)
                logits, states, info = net_a(x, events, states, state_mode=mode, collect=True)
                logits_a.append(logits)
                outs_a.append(info["spikes"])
                mems_a.append(info["u_pre"])
            chunk_states.append(states)
        weights = torch.rand(sum(int(l.numel()) for l in logits_a), generator=g, dtype=torch.float64)
        logits_a = torch.cat(logits_a)
        (logits_a * weights).sum().backward()

        # 片段路径需要梯度时走"反向重算"分支：这里同时核对重算后的梯度与逐窗前向逐位一致
        states, logits_b, active = None, [], [0] * 7
        for i, (start, end) in enumerate(CHUNKS):
            x, events, _, _ = source.chunk(start, end)
            logits, states, info = net_b.forward_chunk(x, events, states, state_mode=mode, collect=True)
            logits_b.append(logits)
            for layer in range(7):
                out_a = torch.stack([w[layer] for w in outs_a[start:end]])
                mem_a = torch.stack([w[layer] for w in mems_a[start:end]])
                self.assertLess(rel_error(out_a, info["spikes"][layer]), 1e-10, LAYER_NAMES[layer])
                self.assertLess(rel_error(mem_a, info["u_pre"][layer]), 1e-10, LAYER_NAMES[layer])
                active[layer] += int((info["spikes"][layer] > 0).sum())
            a_tensors, b_tensors = list(state_tensors(chunk_states[i])), list(state_tensors(states))
            self.assertEqual(len(a_tensors), len(b_tensors))
            for a, b in zip(a_tensors, b_tensors):
                self.assertLess(rel_error(a, b), 1e-10)
        logits_b = torch.cat(logits_b)
        self.assertLess(rel_error(logits_a, logits_b), 1e-10)
        self.assertTrue(all(n > 0 for n in active), active)            # 每层都有活动
        (logits_b * weights).sum().backward()
        for (name, pa), (_, pb) in zip(net_a.named_parameters(), net_b.named_parameters()):
            if pa.grad is None:
                self.assertIsNone(pb.grad, name)
                continue
            scale = max(float(pa.grad.abs().max()), 1.0)
            self.assertLess(max_error(pa.grad, pb.grad) / scale, 1e-9, name)

    def test_statistics_step_equals_chunk(self):
        # 诊断统计只在评估（无梯度）时开启：逐窗与逐层两条路径统计相同
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        for _ in range(1):
            net = make_tif_net()
            stats = []
            with torch.no_grad():
                for chunked in (False, True):
                    net.start_tif_stats()
                    states = None
                    for start, end in CHUNKS:
                        if chunked:
                            x, events, _, _ = source.chunk(start, end)
                            _, states, _ = net.forward_chunk(x, events, states)
                        else:
                            for k in range(start, end):
                                x, events, _, _ = source.window(k)
                                _, states, _ = net(x, events, states)
                    stats.append(net.stop_tif_stats())
            keys = ["message_abs_rate", "message_nonzero_rate", "occupancy", "saturation"]
            for name in LAYER_NAMES:
                for key in keys:
                    self.assertAlmostEqual(stats[0][name][key], stats[1][name][key], places=9, msg=(name, key))
                self.assertEqual(stats[0][name]["windows"], CHUNKS[-1][1])
            self.assertGreater(min(stats[0][name]["message_abs_rate"] for name in LAYER_NAMES), 0.0)

    def test_calibration_runs_with_tif(self):
        seq = synthetic_sequence(seed=2)
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)

        def factory():
            def windows():
                for k in range(6):
                    x, events, _, _ = source.window(k)
                    yield x, events
            for _ in range(2):
                yield windows()

        for _ in range(1):
            gains = []
            for _ in range(2):
                torch.manual_seed(4)
                net = EvSpSegNetStream(channels=SMALL, neuron="tif", ).double()
                calibrate_gains(net, factory, 0.99, 64, 8, 0.05, 20.0, seed=5)
                g = torch.cat([b.gain.gain().detach() for b in net.blocks()])
                self.assertTrue(bool(((g >= 0.05 - 1e-6) & (g <= 20.0 + 1e-5)).all()))
                gains.append(g)
            self.assertEqual(max_error(gains[0], gains[1]), 0.0)

    def test_operation_estimate(self):
        rates = {name: 0.5 for name in LAYER_NAMES}
        for quantize in (True, False):
            net = EvSpSegNetStream(channels=SMALL, neuron="tif", tif={"quantize": quantize})
            ops = estimate_operations(net, H, W, rates, 100)
            self.assertIn("tif", ops)
            if quantize:
                # 输入为 Δq 的部分按 |Δq| 率计 SOP：MAC + SOP / 率 = 稠密运算数
                self.assertAlmostEqual(ops["mac"] + ops["sop"] / 0.5, ops["dense_ops"], delta=1e-3)
                self.assertGreater(ops["sop"], 0.0)
                self.assertEqual(ops["tif"]["message_payload_bits"], 4)
            else:
                self.assertEqual(ops["sop"], 0.0)
                self.assertAlmostEqual(ops["mac"], ops["dense_ops"], delta=1e-3)
            self.assertGreater(ops["state_elements"], 0)


class TIFFixTests(unittest.TestCase):

    def test_bounded_surrogate_forward_and_gradient(self):
        x = torch.tensor([[-2.0, -0.5, 0.0, 0.3, 1.0, 1.5, 3.0]], dtype=torch.float64, requires_grad=True)
        cap = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True)
        y = BoundedSurrogate.apply(x, cap)
        self.assertEqual(max_error(y, torch.clamp(x, 0.0, 1.0)), 0.0)
        y.sum().backward()
        # 下方 1/(1+|x|)^2；区间内 1；上方 1/(1+(x-cap))^2（x = cap 处取上方公式，等于 1）
        expected = torch.tensor([[1 / 9.0, 1 / 2.25, 1.0, 1.0, 1.0, 1 / 2.25, 1 / 9.0]], dtype=torch.float64)
        self.assertLess(max_error(x.grad, expected), 1e-12)
        self.assertEqual(float(cap.grad), 3.0)                         # 触顶的 3 个元素（1.0、1.5、3.0）

    def test_dead_channel_gets_gradient_only_with_surrogate(self):
        # 全程负驱动 = 通道不激活：普通反射下它的输入与参考值 k 收不到梯度；代理梯度下仍能学
        grads = {}
        for surrogate in (False, True):
            n = tif_neuron(2, surrogate=surrogate)
            current = torch.full((1, 2, 4, 4), -1.0, dtype=torch.float64, requires_grad=True)
            state, outs = None, []
            for _ in range(3):
                out, state, _ = n(current, state)
                outs.append(out)
            outs = torch.stack(outs)
            self.assertEqual(float(outs.abs().max()), 0.0)                 # 前向相同：都不激活
            outs.sum().backward()
            grads[surrogate] = (float(current.grad.abs().max()), float(n.k_raw.grad.abs().max()))
        self.assertEqual(grads[False], (0.0, 0.0))
        self.assertGreater(min(grads[True]), 0.0)
        # 有正有负的一般输入：两种反射前向逐位相同
        g = torch.Generator().manual_seed(9)
        current = torch.randn(2, 3, 5, 6, generator=g, dtype=torch.float64)
        a, b = tif_neuron(3, surrogate=False), tif_neuron(3, surrogate=True)
        state_a = state_b = None
        for _ in range(3):
            out_a, state_a, _ = a(current, state_a)
            out_b, state_b, _ = b(current, state_b)
            self.assertEqual(max_error(out_a, out_b), 0.0)
            self.assertEqual(max_error(state_a, state_b), 0.0)

    def test_dead_channel_statistic(self):
        n = tif_neuron(3)
        g = torch.Generator().manual_seed(4)
        current = random_map(g, (1, 3, 6, 6), scale=1.5, density=0.8)
        current[:, 2] = -1.0                                            # 第 3 个通道全程不激活
        n.start_stats()
        state = None
        for _ in range(3):
            _, state, _ = n(current, state)
        self.assertAlmostEqual(n.stop_stats()["dead_channel_frac"], 1.0 / 3, places=12)


if __name__ == "__main__":
    unittest.main()
