"""V5 TIF 神经元（StreamingTIF2d）与 TIF 网络的单元测试（CPU、float64 小网络，不需要数据集）。

    1. 定义：初值、零输入静息、None 与零状态相同、关闭搬运 = 反射累积器、量化电平、状态上界
    2. 新息配对搬运（两侧限量）：守恒、非负、搬走的 <= 失去支持的记忆、搬来的 <= 新证据、同一位置不会同时
       流出和流入、有支持的记忆不动、静默时原地保持、逐例数值（右移、亚像素、静默 + 噪声）、运动的点被跟随
    3. forward_steps（含 keep 掩码）与逐窗 forward 等价（前向与梯度），2x2 配置
    4. 诊断统计：|Δq| 与搬走比例与手算一致
    5. 网络：配置校验、与 LIF 同种子初始化逐位相同、forward_chunk 与逐窗 forward 等价、增益校准、
       运算量估计、完整训练 + 评估入口（2 轮合成数据）
误差一律用 (a - b).abs().max() 比较。
"""
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
from model.lif2d_stream import StreamingTIF2d, inverse_softplus  # noqa: E402
from test_stream_layerwise import (CFG, CHUNKS, H, Q99, SMALL, W, max_error, rel_error,  # noqa: E402
                                   synthetic_sequence)

LIF_ARGS = dict(dt_ms=50.0, tau_init_ms=200.0, tau_min_ms=50.0, tau_max_ms=2000.0)
# 小网络没有经过增益校准，驱动量级很小：调低参考值、阈值（即 theta），让每层都有活动
ACTIVE_TIF = {"k_init": 1e-3}
COMBOS = ((True, True), (False, True), (True, False), (False, False))     # (transport, quantize)


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
        # 搬运没有可学参数：两种配置的参数都只有 a（tau）、k_raw、log_theta
        for transport in (True, False):
            names = sorted(name for name, _ in tif_neuron(3, transport=transport).named_parameters())
            self.assertEqual(names, ["a", "k_raw", "log_theta"])
        self.assertEqual([k for k in n.state_dict() if "neighbor" in k], [])     # 固定核与掩码不进 checkpoint
        for value in (1e-3, 0.05, 2.0):
            self.assertAlmostEqual(float(F.softplus(torch.tensor(inverse_softplus(value), dtype=torch.float64))),
                                   value, places=12)
        with self.assertRaises(ValueError):
            tif_neuron(3, levels=0)
        with self.assertRaises(ValueError):
            tif_neuron(3, transport="yes")

    def test_rest_and_none_equals_zero_state(self):
        n = tif_neuron(3)
        zero = torch.zeros(2, 3, 4, 5, dtype=torch.float64)
        for state in (None, zero):
            out, memory, u = n(zero, state)
            self.assertEqual(float(out.abs().max()), 0.0)
            self.assertEqual(float(memory.abs().max()), 0.0)
        current = torch.randn(2, 3, 4, 5, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        a, b = n(current, None), n(current, zero)
        for x, y in zip(a, b):
            self.assertEqual(max_error(x, y), 0.0)

    def test_transport_off_is_reflected_accumulator(self):
        n = tif_neuron(3, transport=False)
        g = torch.Generator().manual_seed(2)
        prev = random_map(g, (2, 3, 5, 6), scale=0.8)
        current = torch.randn(2, 3, 5, 6, generator=g, dtype=torch.float64)
        lam, k, theta = coefficients(n)
        expected = torch.minimum(torch.relu(lam * prev + (1 - lam) * (current - k)), 7 * theta)
        _, memory, _ = n(current, prev)
        self.assertLess(max_error(memory, expected), 1e-14)

    def test_quantized_levels_float_variant_and_bound(self):
        g = torch.Generator().manual_seed(3)
        for quantize in (True, False):
            n = tif_neuron(3, quantize=quantize)
            _, _, theta = coefficients(n)
            state = None
            for _ in range(4):
                current = random_map(g, (2, 3, 5, 6), scale=4.0)       # 大于量程的输入也有
                out, state, _ = n(current, state)
                self.assertTrue(bool((state <= 7 * theta + 1e-12).all()))
                self.assertGreaterEqual(float(state.min()), 0.0)
                if quantize:
                    levels = out / theta
                    self.assertLess(max_error(levels, torch.round(levels)), 1e-9)
                    self.assertLess(max_error(out, theta * torch.round(state / theta)), 1e-12)
                    self.assertGreaterEqual(float(levels.min()), 0.0)
                    self.assertLessEqual(float(levels.max()), 7 + 1e-9)
                else:
                    self.assertEqual(max_error(out, state), 0.0)
            self.assertGreater(float(state.max()), 0.0)


class TransportTests(unittest.TestCase):
    def test_conservation_nonnegative_one_way_and_caps(self):
        n = tif_neuron(4)
        for seed in range(20):
            g = torch.Generator().manual_seed(seed)
            memory = random_map(g, (2, 4, 9, 11))
            support = random_map(g, (2, 4, 9, 11))
            moved, outflow = n.transport_memory(memory, support)
            # 每张图、每个通道各自守恒（补零边界，质量不会流出图外）
            self.assertLess(max_error(moved.sum((2, 3)), memory.sum((2, 3))), 1e-12)
            self.assertGreaterEqual(float(moved.min()), -1e-15)
            surplus = torch.relu(memory - support)
            self.assertTrue(bool((outflow <= surplus + 1e-15).all()))
            inflow = moved - memory + outflow
            self.assertGreaterEqual(float(inflow.min()), -1e-15)
            self.assertTrue(bool((inflow <= torch.relu(support - memory) + 1e-15).all()))    # 搬来的 <= 新证据
            self.assertEqual(int(((outflow > 1e-15) & (inflow > 1e-15)).sum()), 0)
            self.assertGreater(float(outflow.sum()), 0.0)

    def test_worked_cases(self):
        # 1 维逐例（记忆 0.8）：记忆搬多少，正好等于这一拍的变化能解释多少
        n = tif_neuron(1)
        cases = [([0, .8, 0, 0], [0, 0, .8, 0], [0, 0, .8, 0]),               # 右移一格：全部过去
                 ([0, .8, 0, 0], [0, .4, .4, 0], [0, .4, .4, 0]),             # 亚像素：移走一半
                 ([0, .8, 0, 0], [0, 0, .2, 0], [0, .6, .2, 0]),              # 静默一拍 + 噪声 0.2：只搬 0.2
                 ([0, .8, 0, 0], [0, 0, .05, 0], [0, .75, .05, 0]),           # 弱噪声：只搬 0.05
                 ([0, 0, .8, 0, 0], [0, .2, 0, .8, 0], [0, .16, 0, .64, 0]),  # 两处新证据：按比例
                 ([0, .8, .3, 0], [0, 0, .3, 0], [0, .8, .3, 0]),             # 旁边是已被解释的静止背景：不吸
                 ([0, .8, 0, 0, 0], [0, 0, 0, .8, 0], [0, .8, 0, 0, 0])]      # 跳两格：超出单步支撑，不动
        for memory, support, expected in cases:
            with self.subTest(memory=memory, support=support):
                m = torch.tensor(memory, dtype=torch.float64).view(1, 1, 1, -1)
                s = torch.tensor(support, dtype=torch.float64).view(1, 1, 1, -1)
                moved, _ = n.transport_memory(m, s)
                self.assertLess(max_error(moved.flatten(), torch.tensor(expected, dtype=torch.float64)), 1e-9)
        # 两个源挤向同一个目的地：目的地收到的不超过它的新证据，其余留在源处
        m = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
        m[0, 0, 0, 0] = m[0, 0, 0, 2] = 0.8
        s = torch.zeros_like(m)
        s[0, 0, 1, 1] = 0.8
        moved, _ = n.transport_memory(m, s)
        self.assertAlmostEqual(float(moved[0, 0, 1, 1]), 0.8, places=9)
        self.assertAlmostEqual(float(moved[0, 0, 0, 0]), 0.4, places=9)

    def test_supported_memory_stays_and_silence_holds(self):
        n = tif_neuron(3)
        g = torch.Generator().manual_seed(7)
        memory = random_map(g, (1, 3, 8, 8))
        for support in (memory.clone(), torch.zeros_like(memory)):        # 有支持 / 完全静默
            moved, outflow = n.transport_memory(memory, support)
            self.assertEqual(max_error(moved, memory), 0.0)
            self.assertEqual(float(outflow.abs().max()), 0.0)

    def test_moving_point_is_followed(self):
        # 一个点每窗右移 1 格：搬运时记忆沿轨迹累积、身后几乎不留尾迹；不搬运时当前位置只有新证据，身后留下衰减的尾迹
        final = {}
        for transport in (True, False):
            n = tif_neuron(1, levels=50, quantize=False, transport=transport)
            state = None
            for t in range(6):
                current = torch.zeros(1, 1, 1, 12, dtype=torch.float64)
                current[0, 0, 0, 2 + t] = 1.0
                _, state, _ = n(current, state)
            final[transport] = state[0, 0, 0]
        on, off = final[True], final[False]
        self.assertGreater(float(on[7]), 2.0 * float(off[7]))
        self.assertLess(float(on[:7].sum()), 0.5 * float(off[:7].sum()))


class TIFStepsTests(unittest.TestCase):
    def test_forward_steps_matches_forward_with_keep(self):
        for transport, quantize in COMBOS:
            with self.subTest(transport=transport, quantize=quantize):
                n_a = tif_neuron(3, transport=transport, quantize=quantize)
                n_b = copy.deepcopy(n_a)
                g = torch.Generator().manual_seed(11)
                current = torch.randn(5, 2, 3, 6, 7, generator=g, dtype=torch.float64).clamp(min=-0.2) * 0.8
                keep = torch.ones(5, 2, dtype=torch.float64)
                keep[2, 1] = 0.0                                    # 第 2 条流在第 2 窗开始新录像
                w_out = torch.rand(current.shape, generator=g, dtype=torch.float64)
                w_mem = torch.rand(current.shape, generator=g, dtype=torch.float64)
                state, outs, mems = None, [], []
                for t in range(5):
                    prev = None if state is None else state * keep[t].view(-1, 1, 1, 1)
                    out, state, mem = n_a(current[t], prev)
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
                outs_r, _, mems_r = n_b.forward_steps(current, None, False, True)
                for t in range(5):
                    out_t, _, mem_t = n_b(current[t], None)
                    self.assertLess(max_error(mems_r[t], mem_t), 1e-14)

    def test_statistics_match_manual_count(self):
        n = tif_neuron(2)
        g = torch.Generator().manual_seed(5)
        currents = [random_map(g, (1, 2, 6, 6), scale=1.5) for _ in range(4)]
        n.start_stats()
        state, states = None, []
        for current in currents:
            _, state, _ = n(current, state)
            states.append(state)
        stats = n.stop_stats()
        self.assertFalse(n.collect_stats)
        _, k, theta = coefficients(n)
        levels = [torch.round(s / theta) for s in states]
        deltas = [levels[0]] + [levels[t] - levels[t - 1] for t in range(1, 4)]
        numel = 4.0 * states[0].numel()
        self.assertAlmostEqual(stats["message_abs_rate"], float(sum(d.abs().sum() for d in deltas)) / numel, places=12)
        self.assertAlmostEqual(stats["message_nonzero_rate"], float(sum(d.ne(0).sum() for d in deltas)) / numel,
                               places=12)
        moved = mass = 0.0
        for t in range(1, 4):
            _, outflow = n.transport_memory(states[t - 1], torch.relu(currents[t] - k))
            moved += float(outflow.sum())
            mass += float(states[t - 1].sum())
        self.assertAlmostEqual(stats["moved_mass_frac"], moved / mass, places=10)
        self.assertEqual(stats["windows"], 4)
        self.assertGreater(stats["message_abs_rate"], 0.0)
        self.assertGreater(stats["moved_mass_frac"], 0.0)


def make_tif_net(merged=False, transport=True, quantize=True, seed=5, threshold=0.05):
    """float64 小 TIF 网络；阈值（即 theta 初值）与参考值调低，让未校准的小网络各层都有活动。"""
    torch.manual_seed(seed)
    tif = dict(ACTIVE_TIF, transport=transport, quantize=quantize)
    net = EvSpSegNetStream(channels=SMALL, neuron="tif", merged_decoder=merged, v_threshold=threshold,
                           tif=tif).double()
    # 输出量程 D*theta = threshold；与 graded 对照相同，把增益放大 1/threshold，让信号逐层保持量级
    for block in net.blocks():
        block.gain.set_gain(torch.full((block.out_ch,), 1.0 / threshold, dtype=torch.float64))
    return net


class TIFNetworkTests(unittest.TestCase):
    def test_config_validation(self):
        self.assertEqual(tif_settings(None), TIF_DEFAULTS)
        for bad in ({"unknown": 1}, {"levels": 0}, {"levels": 2.5}, {"quantize": "yes"}, {"transport": 1},
                    {"k_init": 0.0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tif_settings(bad)
        # 第 1 轮旧 checkpoint 的配置带 eps_init：可以加载，键被丢弃
        self.assertEqual(tif_settings({"eps_init": 0.1, "transport": False}), dict(TIF_DEFAULTS, transport=False))
        with self.assertRaises(ValueError):
            EvSpSegNetStream(channels=SMALL, neuron="lif", tif={"levels": 7})
        with self.assertRaises(ValueError):
            EvSpSegNetStream(channels=SMALL, neuron="tif", dual_memory={"layers": ["enc2"]})
        net = EvSpSegNetStream(channels=SMALL, neuron="tif")
        self.assertTrue(net.is_tif and net.stateful and not net.spiking)
        self.assertEqual(net.tif_config, TIF_DEFAULTS)

    def test_same_initialization_as_lif(self):
        torch.manual_seed(3)
        lif = EvSpSegNetStream(channels=SMALL, neuron="lif")
        torch.manual_seed(3)
        tif = EvSpSegNetStream(channels=SMALL, neuron="tif")
        lif_params = dict(lif.named_parameters())
        for name, param in tif.named_parameters():
            if name in lif_params:
                self.assertEqual(max_error(param, lif_params[name]), 0.0, name)
            else:
                self.assertTrue(name.split(".")[-1] in ("k_raw", "log_theta"), name)

    def test_forward_chunk_matches_step_forward(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        cases = [(False, t, q) for t, q in COMBOS] + [(True, True, True)]
        for merged, transport, quantize in cases:
            for mode in ("carry", "reset_each_window"):
                with self.subTest(merged=merged, transport=transport, quantize=quantize, mode=mode):
                    self.check_pair(source, merged, transport, quantize, mode)

    def check_pair(self, source, merged, transport, quantize, mode):
        net_a = make_tif_net(merged, transport, quantize)
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
            for a, b in zip(chunk_states[i], states):
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
        # 诊断统计只在评估（无梯度）时开启：逐窗与逐层两条路径统计相同，搬运在工作
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
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
        for name in LAYER_NAMES:
            for key in ("message_abs_rate", "message_nonzero_rate", "occupancy", "saturation", "moved_mass_frac"):
                self.assertAlmostEqual(stats[0][name][key], stats[1][name][key], places=9, msg=(name, key))
            self.assertEqual(stats[0][name]["windows"], CHUNKS[-1][1])
            self.assertEqual(stats[1][name]["windows"], CHUNKS[-1][1])
        self.assertGreater(max(stats[0][name]["moved_mass_frac"] for name in LAYER_NAMES), 0.0)
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

        gains = []
        for _ in range(2):
            torch.manual_seed(4)
            net = EvSpSegNetStream(channels=SMALL, neuron="tif").double()
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


class TIFEndToEndTests(unittest.TestCase):
    def test_mode_train_and_eval(self):
        # 文件读取、TIF 增益校准、一轮多流训练、验证（含 TIF 统计）、checkpoint、正式评估（逐窗执行、运算量）
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
        except ImportError as error:
            self.skipTest("完整训练入口需要原 V1 的 cv2/pandas/tqdm 依赖: %s" % error)
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        import train_stream_v1 as T
        from utils.stream_common import load_flat_config
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pos = neg = 0
            for split, seeds in (("train", (3, 4, 5)), ("val", (6,))):
                (root / split).mkdir()
                for seed in seeds:
                    seq = synthetic_sequence(seed=seed)
                    loc = np.stack((seq.x, seq.y, seq.t), axis=1)
                    norm = np.column_stack((loc, seq.p, seq.label, seq.target_id))
                    np.savez(root / split / ("seq_%d.npz" % seed), ev_loc=loc, evs_norm=norm)
                    if split == "train":
                        pos += int(seq.label.sum())
                        neg += seq.n_events - int(seq.label.sum())
            stats = dict(q99=Q99.tolist(), train_pos=pos, train_neg=neg, pos_weight=min(neg / pos, 30.0))
            (root / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
            cfg = load_flat_config("configs/evisseg_stream_tif.yaml")
            self.assertEqual((cfg["neuron"], cfg["trainer"], cfg["streams"], cfg["accumulate"]),
                             ("tif", "streams", 4, 1))
            cfg.update(root=str(root), stats_path=str(root / "stats.json"), save_root=str(root / "run"),
                       channels=list(SMALL), height=H, width_px=W, pad_height=H, pad_width=W,
                       n_windows=12, epochs=2, num_workers=0, tbptt_k=4, train_subset_size=1,
                       calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
                       segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=4,
                       execution="layer", input_device="gpu", streams=2)
            T.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
            record = json.loads((root / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(set(record["tif"]), set(LAYER_NAMES))
            self.assertEqual(set(record["tif_parameters"]), set(LAYER_NAMES))
            self.assertIn("theta_mean", record["tif_parameters"]["enc2"])
            checkpoint = root / "run" / "best_val_iou_seed37.pt"
            self.assertTrue(checkpoint.exists())
            args = SimpleNamespace(checkpoint=str(checkpoint), device="cpu", mode="eval", split="val",
                                   overwrite=True, execution=None, input_device=None, readout_ablation="none",
                                   subthreshold_ablation="none", memory_ablation="none", memory_off_layers=None,
                                   reset_every=0, reset_offset=0, dump_dir=None)
            T.mode_eval(args, cfg)
            result = json.loads((root / "run" / "eval_val_best_val_iou_seed37.json").read_text(encoding="utf-8"))
            self.assertEqual(result["tif"]["levels"], 7)
            for mode in ("carry", "reset_each_window"):
                self.assertEqual(set(result[mode]["tif"]), set(LAYER_NAMES))
                self.assertIn("tif", result[mode]["operations"])


if __name__ == "__main__":
    unittest.main()
