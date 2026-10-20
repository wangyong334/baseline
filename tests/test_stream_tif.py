"""V5 TIF 神经元（StreamingTIF2d，10-19 起不再搬运）与 TIF 网络的单元测试（CPU、float64 小网络，不需要数据集）。

    1. 定义：初值、零输入静息、None 与零状态相同、反射累积器、量化电平、状态上界
    2. 本地参照（TIF-R）：参照环安静时与 TIF 逐位相同（float32 / float64）、位置-尺度等变、
       不大于保护区的持续目标不被吸收而成片背景被归一、参照环与偏差校正的手算核对、环内无参照格时不归一
    3. forward_steps（含 keep 掩码）与逐窗 forward 等价（前向与梯度），参照开 / 关 × 量化开 / 关
    4. 诊断统计：|Δq| 与手算一致；参照的局部增益统计
    5. 网络：配置校验（已删除的搬运键）、与 LIF 同种子初始化逐位相同、forward_chunk 与逐窗 forward 等价、
       增益校准、运算量估计、完整训练 + 评估入口（2 轮合成数据）
    6. 两端代理梯度开关（BoundedSurrogate）；死通道统计与 TIF 语义的健康检查；配置文件只差该差的键
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
from model.evspsegnet_stream import (LAYER_NAMES, REFERENCE_ELEMENTWISE_OPS, TIF_DEFAULTS,  # noqa: E402
                                     EvSpSegNetStream, calibrate_gains, estimate_operations, tif_settings)
from model.lif2d_stream import BoundedSurrogate, StreamingTIF2d, inverse_softplus, ring_kernel  # noqa: E402
from test_stream_layerwise import (CFG, CHUNKS, H, Q99, SMALL, W, max_error, rel_error,  # noqa: E402
                                   synthetic_sequence)

LIF_ARGS = dict(dt_ms=50.0, tau_init_ms=200.0, tau_min_ms=50.0, tau_max_ms=2000.0)
# 小网络没有经过增益校准，驱动量级很小：调低参考值、阈值（即 theta），让每层都有活动
ACTIVE_TIF = {"k_init": 1e-3}
COMBOS = ((True, True), (False, True), (True, False), (False, False))     # (reference, quantize)


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


def zero_state(n, current):
    """与 None 等价的显式零状态：TIF 为 M，TIF-R 为 (M, m1, m2, c)。"""
    zero = torch.zeros_like(current)
    if not n.reference:
        return zero
    return (zero, zero.clone(), zero.clone(),
            torch.zeros(current.shape[0], current.shape[1], 1, 1, dtype=current.dtype))


def sparse_points(generator, shape, spacing=4, scale=1.0):
    """只在相距 spacing 的格点上有正值、其余为 0 的图：每个点的参照环（距离 2..3）里没有别的点。"""
    out = torch.zeros(shape, dtype=torch.float64)
    values = torch.rand(shape, generator=generator, dtype=torch.float64) * scale + 0.05
    out[..., ::spacing, ::spacing] = values[..., ::spacing, ::spacing]
    return out


def manual_ring_mean(x):
    """参照环平均的逐点手算（numpy，画面外不计格数；没有参照格时为 0）。"""
    x = x.detach().numpy()
    _, _, h, w = x.shape
    out = np.zeros_like(x)
    for i in range(h):
        for j in range(w):
            total, count = 0.0, 0
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    if max(abs(di), abs(dj)) <= 1 or not (0 <= i + di < h and 0 <= j + dj < w):
                        continue
                    total = total + x[:, :, i + di, j + dj]
                    count += 1
            out[:, :, i, j] = total / count if count else 0.0
    return torch.from_numpy(out)


class TIFDefinitionTests(unittest.TestCase):
    def test_initial_parameters(self):
        n = tif_neuron(3)
        lam, k, theta = coefficients(n)
        self.assertLess(max_error(lam, torch.full_like(lam, float(np.exp(-0.25)))), 1e-6)
        self.assertLess(max_error(k, torch.full_like(k, 0.05)), 1e-6)
        self.assertLess(max_error(theta, torch.full_like(theta, 1.0 / 7)), 1e-6)
        self.assertEqual(sorted(name for name, _ in n.named_parameters()), ["a", "k_raw", "log_theta"])
        self.assertIsNone(n.tau_reference())
        r = tif_neuron(3, reference=True)
        self.assertEqual(sorted(name for name, _ in r.named_parameters()), ["a", "k_raw", "log_theta", "ref_a"])
        self.assertLess(max_error(r.tau_reference(), r.tau()), 1e-12)          # 与 tau 同一参数化、同一初值
        self.assertEqual([key for key in r.state_dict() if "ring" in key], [])  # 固定核不进 checkpoint
        self.assertEqual(int(ring_kernel().sum()), 40)
        for value in (1e-3, 0.05, 2.0):
            self.assertAlmostEqual(float(F.softplus(torch.tensor(inverse_softplus(value), dtype=torch.float64))),
                                   value, places=12)
        with self.assertRaises(ValueError):
            tif_neuron(3, levels=0)
        with self.assertRaises(ValueError):
            tif_neuron(3, reference="yes")
        with self.assertRaises(TypeError):
            tif_neuron(3, transport=False)                                      # 搬运已删除

    def test_rest_and_none_equals_zero_state(self):
        for reference in (False, True):
            n = tif_neuron(3, reference=reference)
            zero = torch.zeros(2, 3, 4, 5, dtype=torch.float64)
            for state in (None, zero_state(n, zero)):
                out, _, memory = n(zero, state)
                self.assertEqual(float(out.abs().max()), 0.0)
                self.assertEqual(float(memory.abs().max()), 0.0)
            current = torch.randn(2, 3, 4, 5, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
            a, b = n(current, None), n(current, zero_state(n, current))
            self.assertEqual(max_error(a[0], b[0]), 0.0, reference)
            self.assertEqual(max_error(a[2], b[2]), 0.0, reference)
            if reference:
                for x, y in zip(a[1], b[1]):
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
        for reference in (False, True):
            for quantize in (True, False):
                n = tif_neuron(3, quantize=quantize, reference=reference)
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


class ReferenceTests(unittest.TestCase):
    def test_quiet_ring_equals_tif(self):
        # 只有相距 4 格的稀疏点有输入：每个点的参照环里没有别的活动，TIF-R 与 TIF 逐位相同（多窗、跨窗累积；
        # float32 与 float64 都要逐位相同）
        g = torch.Generator().manual_seed(4)
        for dtype in (torch.float64, torch.float32):
            a, b = tif_neuron(2).to(dtype), tif_neuron(2, reference=True).to(dtype)
            state_a = state_b = None
            for _ in range(6):
                current = sparse_points(g, (2, 2, 13, 17), scale=2.0).to(dtype)
                out_a, state_a, mem_a = a(current, state_a)
                out_b, state_b, mem_b = b(current, state_b)
                self.assertEqual(max_error(out_a, out_b), 0.0, dtype)
                self.assertEqual(max_error(mem_a, mem_b), 0.0, dtype)
            self.assertGreater(float(mem_a.max()), 0.0)
        # 参照环里一个格子也没有（1x2 的图）：同样不归一
        tiny = (torch.rand(1, 2, 1, 2, generator=g, dtype=torch.float64) * 3).to(dtype)
        self.assertEqual(max_error(a(tiny, None)[2], b(tiny, None)[2]), 0.0)

    def test_local_reference_by_hand(self):
        n = tif_neuron(2, reference=True)
        g = torch.Generator().manual_seed(5)
        _, _, theta, _, alpha = n._coefficients(torch.zeros(1, dtype=torch.float64))
        i0 = torch.randn(1, 2, 6, 8, generator=g, dtype=torch.float64)
        i1 = torch.randn(1, 2, 6, 8, generator=g, dtype=torch.float64) + 0.5
        r0, stats0, gain0 = n.local_reference(i0, None, theta, alpha)
        r1, stats1, gain1 = n.local_reference(i1, stats0, theta, alpha)
        # 偏差校正：第一窗 m^ = I；第二窗 m^ = 按 (1 - alpha, 1) 加权的平均
        w0 = 1.0 - alpha
        for current, mean1, mean2, r, gain in (
                (i0, i0, i0 * i0, r0, gain0),
                (i1, (w0 * i0 + i1) / (w0 + 1), (w0 * i0 * i0 + i1 * i1) / (w0 + 1), r1, gain1)):
            b = manual_ring_mean(mean1)
            variance = torch.relu(manual_ring_mean(mean2) - b * b)
            expected_gain = theta / torch.sqrt(theta * theta + variance)
            self.assertLess(max_error(gain, expected_gain), 1e-12)
            self.assertLess(max_error(r, (current - b) * expected_gain), 1e-12)
        self.assertLess(max_error(stats1[2], (w0 * alpha + alpha).expand(1, -1, -1, -1)), 1e-15)

    def test_location_scale_equivariance(self):
        # I' = A I + B（空间均匀）且 theta、k 同乘 A：M' = A M，电平相同
        g = torch.Generator().manual_seed(6)
        n1 = tif_neuron(3, reference=True)
        n2 = copy.deepcopy(n1)
        scale, shift = 2.5, 0.7
        with torch.no_grad():
            n2.log_theta.add_(float(np.log(scale)))
            k2 = scale * n1.k()
            n2.k_raw.copy_(k2 + torch.log(-torch.expm1(-k2)))
        state1 = state2 = None
        for _ in range(8):
            current = torch.randn(2, 3, 11, 12, generator=g, dtype=torch.float64) * 0.6 + 0.3
            out1, state1, mem1 = n1(current, state1)
            out2, state2, mem2 = n2(scale * current + shift, state2)
            self.assertLess(max_error(mem2 / scale, mem1), 1e-9)
            self.assertLess(max_error(out2 / scale, out1), 1e-9)
        self.assertGreater(float(mem1.max()), 0.0)

    def test_guard_protects_small_persistent_target(self):
        # 背景 1 + 0.3 N(0,1)；第 0 条流中心 1 个像素持续 +3（不大于保护区），第 1 条流中心 7x7 方块持续 +3
        g = torch.Generator().manual_seed(7)
        n = tif_neuron(1, v_threshold=10.0, levels=100, quantize=False, reference=True)   # theta = 0.1，量程 10
        target = torch.zeros(2, 1, 21, 21, dtype=torch.float64)
        target[0, 0, 10, 10] = 3.0
        target[1, 0, 7:14, 7:14] = 3.0
        state, history = None, []
        for _ in range(40):
            current = 1.0 + 0.3 * torch.randn(2, 1, 21, 21, generator=g, dtype=torch.float64) + target
            _, state, memory = n(current, state)
            history.append(memory[:, 0, 10, 10].clone())
        small_early, small_late = float(history[5][0]), float(history[-1][0])
        self.assertGreater(small_early, 0.5)
        self.assertGreater(small_late, 0.8 * small_early)                     # 持续 40 窗（2 s）也不被吸收
        self.assertLess(float(history[-1][1]), 0.2 * small_late)              # 成片的方块内部被归一
        background = memory[0, 0, :4, :4].mean()
        self.assertLess(float(background), 0.2 * small_late)                  # 背景本身也被归一


class TIFStepsTests(unittest.TestCase):
    def test_forward_steps_matches_forward_with_keep(self):
        from model.lif2d_stream import mask_state
        for reference, quantize in COMBOS:
            with self.subTest(reference=reference, quantize=quantize):
                n_a = tif_neuron(3, reference=reference, quantize=quantize)
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
                states_a = state if isinstance(state, tuple) else (state,)
                states_b = state_b if isinstance(state_b, tuple) else (state_b,)
                self.assertEqual(len(states_a), 4 if reference else 1)
                for x, y in zip(states_a, states_b):
                    self.assertLess(max_error(x, y), 1e-12)
                ((outs * w_out).sum() + (mems * w_mem).sum()).backward()
                ((outs_b * w_out).sum() + (mems_b * w_mem).sum()).backward()
                for (name, pa), (_, pb) in zip(n_a.named_parameters(), n_b.named_parameters()):
                    self.assertIsNotNone(pa.grad, name)
                    self.assertTrue(bool(torch.isfinite(pa.grad).all()), name)
                    self.assertLess(max_error(pa.grad, pb.grad) / max(float(pa.grad.abs().max()), 1.0), 1e-10, name)
                    if quantize or reference or name != "log_theta":   # 不量化且不参照时 theta 只经上界起作用
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
        self.assertNotIn("reference_gain_mean", stats)
        # 参照的局部增益统计
        r = tif_neuron(2, reference=True)
        _, _, theta, _, alpha = r._coefficients(currents[0])
        stats_prev, gains = None, []
        for current in currents:
            _, stats_prev, gain = r.local_reference(current, stats_prev, theta, alpha)
            gains.append(gain)
        r.start_stats()
        state = None
        for current in currents:
            _, state, _ = r(current, state)
        stats = r.stop_stats()
        gains = torch.stack(gains)
        self.assertAlmostEqual(stats["reference_gain_mean"], float(gains.mean()), places=12)
        self.assertAlmostEqual(stats["reference_normalized_frac"], float((gains < 0.5 ** 0.5).double().mean()),
                               places=12)
        self.assertTrue(0.0 < stats["reference_gain_mean"] < 1.0)


def make_tif_net(merged=False, reference=False, quantize=True, seed=5, threshold=0.05, **extra):
    """float64 小 TIF 网络；阈值（即 theta 初值）与参考值调低，让未校准的小网络各层都有活动。"""
    torch.manual_seed(seed)
    tif = dict(ACTIVE_TIF, reference=reference, quantize=quantize, **extra)
    net = EvSpSegNetStream(channels=SMALL, neuron="tif", merged_decoder=merged, v_threshold=threshold,
                           tif=tif).double()
    # 输出量程 D*theta = threshold；与 graded 对照相同，把增益放大 1/threshold，让信号逐层保持量级
    for block in net.blocks():
        block.gain.set_gain(torch.full((block.out_ch,), 1.0 / threshold, dtype=torch.float64))
    return net


class TIFNetworkTests(unittest.TestCase):
    def test_config_validation(self):
        self.assertEqual(tif_settings(None), TIF_DEFAULTS)
        for bad in ({"unknown": 1}, {"levels": 0}, {"levels": 2.5}, {"quantize": "yes"}, {"reference": 1},
                    {"k_init": 0.0}, {"surrogate": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tif_settings(bad)
        # 旧 checkpoint 的配置：不搬运时可以加载，已删除机制的键被丢弃
        legacy = {"levels": 7, "quantize": True, "transport": False, "k_init": 0.05, "surrogate": False,
                  "transport_ref_init": 0.0, "transport_layers": list(LAYER_NAMES), "eps_init": 0.1}
        self.assertEqual(tif_settings(legacy), TIF_DEFAULTS)
        for bad in ({"transport": True}, {"transport_ref_init": 0.05}, dict(legacy, transport=True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tif_settings(bad)
        with self.assertRaises(ValueError):
            EvSpSegNetStream(channels=SMALL, neuron="lif", tif={"levels": 7})
        with self.assertRaises(ValueError):
            EvSpSegNetStream(channels=SMALL, neuron="tif", dual_memory={"layers": ["enc2"]})
        net = EvSpSegNetStream(channels=SMALL, neuron="tif")
        self.assertTrue(net.is_tif and net.stateful and not net.spiking)
        self.assertEqual(net.tif_config, TIF_DEFAULTS)
        net = EvSpSegNetStream(channels=SMALL, neuron="tif", tif={"reference": True})
        self.assertTrue(all(block.neuron.reference for block in net.blocks()))

    def test_same_initialization_as_lif(self):
        torch.manual_seed(3)
        lif = EvSpSegNetStream(channels=SMALL, neuron="lif")
        lif_params = dict(lif.named_parameters())
        for reference in (False, True):
            torch.manual_seed(3)
            tif = EvSpSegNetStream(channels=SMALL, neuron="tif", tif={"reference": reference})
            for name, param in tif.named_parameters():
                if name in lif_params:
                    self.assertEqual(max_error(param, lif_params[name]), 0.0, name)
                else:
                    self.assertTrue(name.split(".")[-1] in ("k_raw", "log_theta", "ref_a"), name)

    def test_forward_chunk_matches_step_forward(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        cases = ([(False, r, q, {}) for r, q in COMBOS] + [(True, True, True, {}), (True, False, True, {})]
                 + [(False, True, True, {"surrogate": True}), (False, False, True, {"surrogate": True})])
        for merged, reference, quantize, extra in cases:
            for mode in ("carry", "reset_each_window"):
                with self.subTest(merged=merged, reference=reference, quantize=quantize, extra=extra, mode=mode):
                    self.check_pair(source, merged, reference, quantize, mode, extra)

    def check_pair(self, source, merged, reference, quantize, mode, extra=None):
        from model.lif2d_stream import state_tensors
        net_a = make_tif_net(merged, reference, quantize, **(extra or {}))
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
        for reference in (False, True):
            net = make_tif_net(reference=reference)
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
            if reference:
                keys += ["reference_gain_mean", "reference_normalized_frac"]
            for name in LAYER_NAMES:
                for key in keys:
                    self.assertAlmostEqual(stats[0][name][key], stats[1][name][key], places=9, msg=(name, key))
                self.assertEqual(stats[0][name]["windows"], CHUNKS[-1][1])
            self.assertGreater(min(stats[0][name]["message_abs_rate"] for name in LAYER_NAMES), 0.0)
            if reference:
                self.assertLess(min(stats[0][name]["reference_gain_mean"] for name in LAYER_NAMES), 1.0)

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

        for reference in (False, True):
            gains = []
            for _ in range(2):
                torch.manual_seed(4)
                net = EvSpSegNetStream(channels=SMALL, neuron="tif", tif={"reference": reference}).double()
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
            self.assertEqual(ops["tif"]["reference_elementwise_ops_per_window"], 0)
        plain = estimate_operations(EvSpSegNetStream(channels=SMALL, neuron="tif"), H, W, rates, 100)
        ref = estimate_operations(EvSpSegNetStream(channels=SMALL, neuron="tif", tif={"reference": True}),
                                  H, W, rates, 100)
        elements = plain["state_elements"]
        self.assertEqual(ref["state_elements"], 3 * elements)
        self.assertEqual(ref["tif"]["reference_elementwise_ops_per_window"], REFERENCE_ELEMENTWISE_OPS * elements)
        self.assertEqual(ref["tif"]["elementwise_ops_per_window"] - plain["tif"]["elementwise_ops_per_window"],
                         REFERENCE_ELEMENTWISE_OPS * elements)
        self.assertEqual((ref["mac"], ref["sop"]), (plain["mac"], plain["sop"]))


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

    def test_health_uses_tif_semantics(self):
        import train_stream_v1 as T
        from utils.stream_common import health_warnings
        layers = {"enc1": {"firing_rate": 0.5, "silent_channel_frac": 0.0, "always_on_frac": 0.9,
                           "big_membrane_frac": 0.0}}
        self.assertIs(T.health_layer_stats(layers, {}), layers)
        warnings = health_warnings(T.health_layer_stats(layers, {"enc1": {"dead_channel_frac": 0.7,
                                                                           "saturation": 0.2}}), {}, {})
        text = " ".join(warnings)
        self.assertIn("70%", text)
        self.assertIn("电平到顶", text)
        self.assertNotIn("每窗发放", text)


class TIFConfigTests(unittest.TestCase):
    def test_evflying_config_differs_from_v1_only_in_neuron(self):
        # EV-Flying 上的 TIF：除 neuron / tif / save_root 外与 V1 的 EV-Flying 配置逐项相同
        from utils.stream_common import load_flat_config
        v1 = load_flat_config("configs/evisseg_stream_v1_evflying.yaml")
        cfg = load_flat_config("configs/evisseg_stream_tif_notransport_evflying.yaml")
        self.assertEqual(set(cfg) - {"tif"}, set(v1))
        for key in v1:
            if key not in ("neuron", "save_root"):
                self.assertEqual(cfg[key], v1[key], key)
        self.assertEqual(cfg["neuron"], "tif")
        self.assertEqual(tif_settings(cfg["tif"]), TIF_DEFAULTS)
        self.assertNotEqual(cfg["save_root"], v1["save_root"])

    def test_tifr_configs_differ_only_in_reference(self):
        import glob
        from utils.stream_common import load_flat_config
        roots = set()
        for base, tifr in (("evisseg_stream_tif_notransport", "evisseg_stream_tifr"),
                           ("evisseg_stream_tif_notransport_evflying", "evisseg_stream_tifr_evflying")):
            a = load_flat_config("configs/%s.yaml" % base)
            b = load_flat_config("configs/%s.yaml" % tifr)
            self.assertEqual(set(a), set(b))
            for key in a:
                if key not in ("tif", "save_root"):
                    self.assertEqual(a[key], b[key], (tifr, key))
            self.assertEqual(tif_settings(b["tif"]), dict(tif_settings(a["tif"]), reference=True))
            roots.update((a["save_root"], b["save_root"]))
        self.assertEqual(len(roots), 4)
        for path in glob.glob("configs/evisseg_stream_tif*.yaml"):                # 现存配置不再带搬运的键
            tif = load_flat_config(path).get("tif") or {}
            self.assertFalse(set(tif) & {"transport", "transport_ref_init", "transport_layers", "eps_init"}, path)


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
            for config in ("evisseg_stream_tif_notransport", "evisseg_stream_tifr"):
                run = root / config
                cfg = load_flat_config("configs/%s.yaml" % config)
                self.assertEqual((cfg["neuron"], cfg["trainer"], cfg["streams"], cfg["accumulate"]),
                                 ("tif", "streams", 4, 1))
                cfg.update(root=str(root), stats_path=str(root / "stats.json"), save_root=str(run),
                           channels=list(SMALL), height=H, width_px=W, pad_height=H, pad_width=W,
                           n_windows=12, epochs=2, num_workers=0, tbptt_k=4, train_subset_size=1,
                           calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
                           segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=4,
                           execution="layer", input_device="gpu", streams=2)
                T.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
                record = json.loads((run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
                self.assertEqual(set(record["tif"]), set(LAYER_NAMES), config)
                self.assertEqual(set(record["tif_parameters"]), set(LAYER_NAMES), config)
                self.assertIn("theta_mean", record["tif_parameters"]["enc2"])
                self.assertIn("dead_channel_frac", record["tif"]["dec3"])
                reference = config.endswith("tifr")
                self.assertEqual("tau_ref_mean" in record["tif_parameters"]["enc2"], reference)
                self.assertEqual("reference_gain_mean" in record["tif"]["enc2"], reference)
                checkpoint = run / "best_val_iou_seed37.pt"
                self.assertTrue(checkpoint.exists())
                args = SimpleNamespace(checkpoint=str(checkpoint), device="cpu", mode="eval", split="val",
                                       overwrite=True, execution=None, input_device=None, readout_ablation="none",
                                       subthreshold_ablation="none", memory_ablation="none", memory_off_layers=None,
                                       reset_every=0, reset_offset=0, dump_dir=None)
                T.mode_eval(args, cfg)
                result = json.loads((run / "eval_val_best_val_iou_seed37.json").read_text(encoding="utf-8"))
                self.assertEqual(result["tif"]["levels"], 7)
                self.assertEqual(result["tif"]["reference"], reference)
                for mode in ("carry", "reset_each_window"):
                    self.assertEqual(set(result[mode]["tif"]), set(LAYER_NAMES))
                    self.assertIn("tif", result[mode]["operations"])


if __name__ == "__main__":
    unittest.main()
