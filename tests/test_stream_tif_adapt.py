"""TIF 工作点自校准（tif.adapt，Stream-SNN-Innovations.md 12.1）的单元测试（CPU、float64，不需要数据集）。

    1. 定义：与手算递推一致；首窗、None 与零状态与 adapt=False 相同；状态为 (M, b)；参数与校验
    2. P5 平移不变：平稳输入下 M 回到 0，与背景水平无关；均值为负的通道与 adapt=False 相同；有界性不变
    3. forward_steps（含 keep 掩码、反向重算）与逐窗 forward 等价（前向与梯度），tau_b 有梯度
    4. 网络：配置校验、同种子下其余参数与基础版逐位相同、forward_chunk 与逐窗等价（carry / reset）、
       reset 模式下与 adapt=False 输出逐位相同、统计与运算量、配置文件只差 adapt 四项与 save_root
    5. adapt_debias（12.4）：参照等于历史加权均值、无热身过程、None/零状态/reset/片段等价、关闭时与第一版相同
    6. adapt_layers（12.5）：只有所选层自校准、其余层与基础版相同、运算量只计所选层、配置只差该差的键
"""
import copy
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.stream_source import TorchWindowSource  # noqa: E402
from model.evspsegnet_stream import (LAYER_NAMES, TIF_DEFAULTS, EvSpSegNetStream,  # noqa: E402
                                     estimate_operations, tif_settings)
from model.lif2d_stream import mask_state, state_tensors  # noqa: E402
from test_stream_layerwise import CFG, CHUNKS, H, Q99, SMALL, W, max_error, synthetic_sequence  # noqa: E402
import test_stream_tif as base  # noqa: E402  （只导入模块：直接导入 TestCase 类会让 unittest 重复收集它的测试）
from test_stream_tif import coefficients, make_tif_net, random_map, tif_neuron  # noqa: E402

# 测试用短基线时间常数（合成序列只有几十窗）
FAST = {"adapt_tau_init_ms": 200.0, "adapt_tau_min_ms": 100.0, "adapt_tau_max_ms": 1000.0}


def adapt_neuron(channels=3, **kw):
    return tif_neuron(channels, adapt=True, **dict(FAST, **kw))


def alpha_of(n):
    return torch.exp(-50.0 / n.adapt_tau()).double().view(1, -1, 1, 1)


class AdaptDefinitionTests(unittest.TestCase):
    def test_parameters_and_validation(self):
        n = adapt_neuron(3)
        self.assertEqual(sorted(name for name, _ in n.named_parameters()), ["a", "adapt_a", "k_raw", "log_theta"])
        self.assertLess(max_error(n.adapt_tau().double(), torch.full((3,), 200.0, dtype=torch.float64)), 1e-4)
        for bad in ({"adapt_tau_min_ms": 300.0}, {"adapt_tau_max_ms": 150.0}, {"adapt_tau_min_ms": 0.0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                adapt_neuron(3, **bad)
        with self.assertRaises(ValueError):
            tif_neuron(3, adapt=1)
        self.assertNotIn("adapt_a", dict(tif_neuron(3).named_parameters()))   # 关闭时不新增参数

    def test_manual_recurrence(self):
        n = adapt_neuron(3)
        g = torch.Generator().manual_seed(2)
        lam, k, theta = coefficients(n)
        alpha = alpha_of(n)
        currents = [torch.randn(2, 3, 5, 6, generator=g, dtype=torch.float64) * 0.5 + 0.3 for _ in range(6)]
        state, m, b = None, None, None
        for t, current in enumerate(currents):
            out, state, memory = n(current, state)
            if t == 0:
                m = torch.minimum(torch.relu((1 - lam) * (current - k)), 7 * theta)
                b = (1 - alpha) * current
            else:
                m = torch.minimum(torch.relu(lam * m + (1 - lam) * (current - k - torch.relu(b))), 7 * theta)
                b = alpha * b + (1 - alpha) * current
            self.assertIsInstance(state, tuple)
            self.assertLess(max_error(memory, m), 1e-14)
            self.assertLess(max_error(state[0], m), 1e-14)
            self.assertLess(max_error(state[1], b), 1e-14)
            self.assertLess(max_error(out, theta * torch.round(m / theta)), 1e-12)

    def test_first_window_none_and_zero_state_match_plain(self):
        plain = tif_neuron(3)
        adapt = adapt_neuron(3)
        current = torch.randn(2, 3, 4, 5, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        zero = (torch.zeros_like(current), torch.zeros_like(current))
        ref = plain(current, None)
        for state in (None, zero):
            out, new_state, memory = adapt(current, state)
            self.assertEqual(max_error(out, ref[0]), 0.0)
            self.assertEqual(max_error(memory, ref[2]), 0.0)
        with self.assertRaises(ValueError):
            adapt(current, current)                     # adapt=True 的状态必须是 (M, b)
        with self.assertRaises(ValueError):
            plain(current, zero)                        # adapt=False 的状态必须是张量

    def test_offset_invariance_p5(self):
        # 平稳输入：adapt 下 M 回到 0，与背景水平无关；基础版稳态 M = min(mu - k, D*theta)
        plain, adapt = tif_neuron(1), adapt_neuron(1)
        _, k, theta = coefficients(plain)
        for mu in (0.1, 0.5, 3.0):
            current = torch.full((1, 1, 2, 2), mu, dtype=torch.float64)
            s_p, s_a = None, None
            for _ in range(400):
                _, s_p, m_p = plain(current, s_p)
                _, s_a, m_a = adapt(current, s_a)
            expected = torch.minimum(torch.relu(current - k), 7 * theta)
            self.assertLess(max_error(m_p, expected), 1e-9, mu)
            self.assertEqual(float(m_a.abs().max()), 0.0, mu)
            self.assertLess(max_error(s_a[1], current), 1e-9, mu)          # 基线收敛到输入均值
        # 均值为负的通道：relu(b) = 0，与基础版逐位相同
        g = torch.Generator().manual_seed(4)
        s_p, s_a = None, None
        for _ in range(30):
            current = -1.0 + 0.3 * torch.rand(2, 1, 3, 3, generator=g, dtype=torch.float64)
            out_p, s_p, m_p = plain(current, s_p)
            out_a, s_a, m_a = adapt(current, s_a)
            self.assertEqual(max_error(m_p, m_a), 0.0)
            self.assertEqual(max_error(out_p, out_a), 0.0)

    def test_step_rise_then_target_pulse(self):
        # 背景升高后先被积累、再被基线吸收；之后短时目标脉冲仍能把 M 推高
        n = adapt_neuron(1)
        state = None
        for _ in range(200):
            _, state, memory = n(torch.full((1, 1, 1, 1), 2.0, dtype=torch.float64), state)
        self.assertEqual(float(memory.max()), 0.0)
        peak = 0.0
        for _ in range(3):
            _, state, memory = n(torch.full((1, 1, 1, 1), 4.0, dtype=torch.float64), state)
            peak = max(peak, float(memory.max()))
        self.assertGreater(peak, 0.5)

    def test_bounds_kept(self):
        n = adapt_neuron(3)
        _, _, theta = coefficients(n)
        g = torch.Generator().manual_seed(3)
        state = None
        for _ in range(8):
            out, state, memory = n(random_map(g, (2, 3, 5, 6), scale=4.0), state)
            self.assertTrue(bool((memory <= 7 * theta + 1e-12).all()))
            self.assertGreaterEqual(float(memory.min()), 0.0)


class AdaptStepsTests(unittest.TestCase):
    def test_forward_steps_matches_forward_with_keep(self):
        n_a = adapt_neuron(3)
        n_b = copy.deepcopy(n_a)
        g = torch.Generator().manual_seed(11)
        current = torch.randn(6, 2, 3, 6, 7, generator=g, dtype=torch.float64).clamp(min=-0.2) * 0.8 + 0.2
        keep = torch.ones(6, 2, dtype=torch.float64)
        keep[3, 1] = 0.0
        w_out = torch.rand(current.shape, generator=g, dtype=torch.float64)
        w_mem = torch.rand(current.shape, generator=g, dtype=torch.float64)
        state, outs, mems = None, [], []
        for t in range(6):
            out, state, mem = n_a(current[t], mask_state(state, keep, t))
            outs.append(out)
            mems.append(mem)
        outs, mems = torch.stack(outs), torch.stack(mems)
        outs_b, state_b, mems_b = n_b.forward_steps(current.clone(), None, True, True, keep=keep)
        self.assertLess(max_error(outs, outs_b), 1e-12)
        self.assertLess(max_error(mems, mems_b), 1e-12)
        for x, y in zip(state, state_b):
            self.assertLess(max_error(x, y), 1e-12)
        ((outs * w_out).sum() + (mems * w_mem).sum()).backward()
        ((outs_b * w_out).sum() + (mems_b * w_mem).sum()).backward()
        for (name, pa), (_, pb) in zip(n_a.named_parameters(), n_b.named_parameters()):
            self.assertIsNotNone(pa.grad, name)
            self.assertLess(max_error(pa.grad, pb.grad) / max(float(pa.grad.abs().max()), 1.0), 1e-10, name)
            self.assertGreater(float(pa.grad.abs().max()), 0.0, name)
        # reset：每一步都从零开始，与基础版相同
        plain = tif_neuron(3)
        _, _, mems_r = n_b.forward_steps(current, None, False, True)
        _, _, mems_p = plain.forward_steps(current, None, False, True)
        self.assertEqual(max_error(mems_r, mems_p), 0.0)

    def test_statistics_baseline_level(self):
        n = adapt_neuron(2)
        g = torch.Generator().manual_seed(5)
        currents = [random_map(g, (1, 2, 6, 6), scale=1.5) for _ in range(4)]
        n.start_stats()
        state, baselines = None, []
        for current in currents:
            _, state, _ = n(current, state)
            baselines.append(state[1])
        stats = n.stop_stats()
        _, _, theta = coefficients(n)
        expected = float(sum((torch.relu(b) / theta).sum() for b in baselines)) / (4.0 * baselines[0].numel())
        self.assertAlmostEqual(stats["baseline_level_mean"], expected, places=12)
        self.assertNotIn("baseline_level_mean", tif_neuron(2).stop_stats() or {"x": 0})


class AdaptNetworkTests(unittest.TestCase):
    def test_config_validation(self):
        self.assertFalse(TIF_DEFAULTS["adapt"])
        cfg = tif_settings({"adapt": True})
        self.assertTrue(cfg["adapt"])
        for bad in ({"adapt": "yes"}, {"adapt_tau_min_ms": 3000.0}, {"adapt_tau_max_ms": 1500.0},
                    {"adapt_tau_init_ms": float("nan")}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tif_settings(bad)

    def test_same_initialization_as_plain(self):
        torch.manual_seed(3)
        plain = dict(EvSpSegNetStream(channels=SMALL, neuron="tif").named_parameters())
        torch.manual_seed(3)
        net = EvSpSegNetStream(channels=SMALL, neuron="tif", tif={"adapt": True})
        for name, param in net.named_parameters():
            if name in plain:
                self.assertEqual(max_error(param, plain[name]), 0.0, name)
            else:
                self.assertTrue(name.endswith(".neuron.adapt_a"), name)

    def test_forward_chunk_matches_step_forward(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        helper = base.TIFNetworkTests("test_forward_chunk_matches_step_forward")
        for mode in ("carry", "reset_each_window"):
            with self.subTest(mode=mode):
                helper.check_pair(source, False, True, mode, dict(FAST, adapt=True))

    def test_reset_mode_equals_plain(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        net_a = make_tif_net(adapt=True, **FAST)
        net_p = make_tif_net()
        state = {k: v for k, v in net_a.state_dict().items() if not k.endswith("adapt_a")}
        net_p.load_state_dict(state)
        with torch.no_grad():
            for k in range(CHUNKS[-1][1]):
                x, events, _, _ = source.window(k)
                la, _, _ = net_a(x, events, None, state_mode="reset_each_window")
                lp, _, _ = net_p(x, events, None, state_mode="reset_each_window")
                self.assertEqual(max_error(la, lp), 0.0)

    def test_carry_state_structure_and_statistics(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        net = make_tif_net(adapt=True, **FAST)
        with torch.no_grad():
            net.start_tif_stats()
            states = None
            for k in range(CHUNKS[-1][1]):
                x, events, _, _ = source.window(k)
                _, states, _ = net(x, events, states)
            stats = net.stop_tif_stats()
        self.assertTrue(all(isinstance(s, tuple) and len(s) == 2 for s in states))
        self.assertEqual(len(list(state_tensors(states))), 14)
        for name in LAYER_NAMES:
            self.assertGreaterEqual(stats[name]["baseline_level_mean"], 0.0)
        self.assertGreater(stats["enc1"]["baseline_level_mean"], 0.0)

    def test_operation_estimate(self):
        rates = {name: 0.5 for name in LAYER_NAMES}
        plain = estimate_operations(EvSpSegNetStream(channels=SMALL, neuron="tif"), H, W, rates, 100)
        adapt = estimate_operations(EvSpSegNetStream(channels=SMALL, neuron="tif", tif={"adapt": True}),
                                    H, W, rates, 100)
        self.assertEqual(adapt["state_elements"], 2 * plain["state_elements"])
        self.assertEqual(adapt["tif"]["elementwise_ops_per_window"],
                         plain["tif"]["elementwise_ops_per_window"] * 15 // 11)
        self.assertEqual(adapt["sop"], plain["sop"])
        self.assertTrue(adapt["tif"]["adapt"] and not plain["tif"]["adapt"])


def debias_neuron(channels=3, **kw):
    return adapt_neuron(channels, adapt_debias=True, **kw)


class DebiasTests(unittest.TestCase):
    """adapt_debias（12.4）：偏差校正的基线 b/w。"""

    def test_validation_and_state(self):
        with self.assertRaises(ValueError):
            tif_neuron(3, adapt_debias=True)                     # 需要 adapt
        with self.assertRaises(ValueError):
            tif_settings({"adapt_debias": True})
        self.assertFalse(TIF_DEFAULTS["adapt_debias"])
        n = debias_neuron(3)
        self.assertEqual(sorted(name for name, _ in n.named_parameters()), ["a", "adapt_a", "k_raw", "log_theta"])
        current = torch.rand(2, 3, 4, 5, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        _, state, _ = n(current, None)
        self.assertEqual(len(state), 3)
        self.assertEqual(tuple(state[2].shape), (2, 3, 1, 1))
        with self.assertRaises(ValueError):
            n(current, state[:2])

    def test_manual_running_mean(self):
        # b/w 等于按 alpha 加权的历史输入均值；首窗不减参照
        n = debias_neuron(3)
        lam, k, theta = coefficients(n)
        alpha = alpha_of(n)
        g = torch.Generator().manual_seed(2)
        currents = [torch.randn(2, 3, 5, 6, generator=g, dtype=torch.float64) * 0.5 + 0.3 for _ in range(6)]
        state, m = None, None
        for t, current in enumerate(currents):
            if t == 0:
                m = torch.minimum(torch.relu((1 - lam) * (current - k)), 7 * theta)
            else:
                weights = [alpha ** (t - 1 - i) for i in range(t)]
                mean = sum(w * c for w, c in zip(weights, currents[:t])) / sum(weights)
                m = torch.minimum(torch.relu(lam * m + (1 - lam) * (current - k - torch.relu(mean))), 7 * theta)
            _, state, memory = n(current, state)
            self.assertLess(max_error(memory, m), 1e-12, t)
            weights = [alpha ** (t - i) for i in range(t + 1)]
            mean_now = sum(w * c for w, c in zip(weights, currents[:t + 1])) / sum(weights)
            self.assertLess(max_error(state[1] / state[2], mean_now), 1e-12, t)

    def test_no_warmup_transient(self):
        # 大直流：未校正的 b 要追很久（期间 M 到顶），校正后第 2 窗起 M 就开始回落、不再到顶
        plain_adapt, debias = adapt_neuron(1), debias_neuron(1)
        _, k, theta = coefficients(debias)
        current = torch.full((1, 1, 2, 2), 50.0, dtype=torch.float64)
        s_a, s_d, top_a, top_d = None, None, 0, 0
        for t in range(40):
            _, s_a, m_a = plain_adapt(current, s_a)
            _, s_d, m_d = debias(current, s_d)
            top_a += int(bool((m_a >= 7 * theta - 1e-12).all()))
            top_d += int(bool((m_d >= 7 * theta - 1e-12).all()))
        self.assertLessEqual(top_d, 1)                            # 只有首窗可能到顶
        self.assertGreater(top_a, 10)                             # 未校正：长时间到顶
        self.assertEqual(float(m_d.abs().max()), 0.0)

    def test_none_zero_state_reset_and_steps(self):
        n_a = debias_neuron(3)
        n_b = copy.deepcopy(n_a)
        plain = tif_neuron(3)
        g = torch.Generator().manual_seed(11)
        current = torch.randn(6, 2, 3, 6, 7, generator=g, dtype=torch.float64).clamp(min=-0.2) * 0.8 + 0.2
        with torch.no_grad():
            zero = tuple(torch.zeros_like(x) for x in n_a(current[0], None)[1])
            for x, y in zip(n_a(current[1], None), n_a(current[1], zero)):
                pairs = zip(x, y) if isinstance(x, tuple) else [(x, y)]
                for xa, ya in pairs:
                    self.assertEqual(max_error(xa, ya), 0.0)
        keep = torch.ones(6, 2, dtype=torch.float64)
        keep[3, 1] = 0.0
        w_out = torch.rand(current.shape, generator=g, dtype=torch.float64)
        state, outs = None, []
        for t in range(6):
            out, state, _ = n_a(current[t], mask_state(state, keep, t))
            outs.append(out)
        outs = torch.stack(outs)
        outs_b, state_b, _ = n_b.forward_steps(current.clone(), None, True, True, keep=keep)
        self.assertLess(max_error(outs, outs_b), 1e-12)
        for x, y in zip(state, state_b):
            self.assertLess(max_error(x, y), 1e-12)
        (outs * w_out).sum().backward()
        (outs_b * w_out).sum().backward()
        for (name, pa), (_, pb) in zip(n_a.named_parameters(), n_b.named_parameters()):
            self.assertIsNotNone(pa.grad, name)
            self.assertLess(max_error(pa.grad, pb.grad) / max(float(pa.grad.abs().max()), 1.0), 1e-10, name)
            self.assertGreater(float(pa.grad.abs().max()), 0.0, name)
        with torch.no_grad():
            _, _, mems_r = n_b.forward_steps(current, None, False, True)
            _, _, mems_p = plain.forward_steps(current, None, False, True)
        self.assertEqual(max_error(mems_r, mems_p), 0.0)

    def test_network_chunk_matches_step_and_statistics(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        helper = base.TIFNetworkTests("test_forward_chunk_matches_step_forward")
        for mode in ("carry", "reset_each_window"):
            with self.subTest(mode=mode):
                helper.check_pair(source, False, True, mode, dict(FAST, adapt=True, adapt_debias=True))
        net = make_tif_net(adapt=True, adapt_debias=True, **FAST)
        with torch.no_grad():
            net.start_tif_stats()
            states = None
            for k in range(CHUNKS[-1][1]):
                x, events, _, _ = source.window(k)
                _, states, _ = net(x, events, states)
            stats = net.stop_tif_stats()
        self.assertTrue(all(isinstance(s, tuple) and len(s) == 3 for s in states))
        self.assertGreater(stats["enc1"]["baseline_level_mean"], 0.0)
        ops = estimate_operations(EvSpSegNetStream(channels=SMALL, neuron="tif",
                                                   tif={"adapt": True, "adapt_debias": True}), H, W,
                                  {name: 0.5 for name in LAYER_NAMES}, 100)
        self.assertTrue(ops["tif"]["adapt_debias"])

    def test_debias_off_matches_first_adapt(self):
        # adapt_debias 关闭时与第一版 adapt 完全相同（卡 0 上那一轮的评估依赖这一点）
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        net_a = make_tif_net(adapt=True, **FAST)
        net_b = make_tif_net(adapt=True, adapt_debias=False, **FAST)
        with torch.no_grad():
            sa = sb = None
            for k in range(CHUNKS[-1][1]):
                x, events, _, _ = source.window(k)
                la, sa, _ = net_a(x, events, sa)
                lb, sb, _ = net_b(x, events, sb)
                self.assertEqual(max_error(la, lb), 0.0)
        self.assertTrue(all(len(s) == 2 for s in sa))


class AdaptLayersTests(unittest.TestCase):
    """adapt_layers（12.5）：只在所选层做自校准，其余层为基础版 TIF。"""

    def test_validation(self):
        self.assertIsNone(TIF_DEFAULTS["adapt_layers"])
        self.assertEqual(tif_settings({"adapt": True, "adapt_layers": ["dec1", "enc1"]})["adapt_layers"],
                         ["enc1", "dec1"])
        for bad in ({"adapt_layers": ["enc1"]}, {"adapt": True, "adapt_layers": []},
                    {"adapt": True, "adapt_layers": ["enc5"]}, {"adapt": True, "adapt_layers": ["enc1", "enc1"]},
                    {"adapt": True, "adapt_layers": "enc1"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tif_settings(bad)

    def test_only_selected_layers_adapt(self):
        torch.manual_seed(3)
        plain = dict(EvSpSegNetStream(channels=SMALL, neuron="tif").named_parameters())
        torch.manual_seed(3)
        net = EvSpSegNetStream(channels=SMALL, neuron="tif",
                               tif={"adapt": True, "adapt_debias": True, "adapt_layers": ["enc1"]})
        self.assertEqual(net.tif_adapt_layers, ("enc1",))
        for name, block in zip(LAYER_NAMES, net.blocks()):
            self.assertEqual(block.neuron.adapt, name == "enc1", name)
            self.assertEqual(block.neuron.adapt_debias, name == "enc1", name)
        for name, param in net.named_parameters():
            if name in plain:
                self.assertEqual(max_error(param, plain[name]), 0.0, name)
            else:
                self.assertEqual(name, "enc1.neuron.adapt_a")
        self.assertEqual(EvSpSegNetStream(channels=SMALL, neuron="tif", tif={"adapt": True}).tif_adapt_layers,
                         LAYER_NAMES)                                  # None = 全部层（前两版）
        self.assertEqual(EvSpSegNetStream(channels=SMALL, neuron="tif").tif_adapt_layers, ())

    def test_deep_layers_equal_plain_and_chunk_matches_step(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        helper = base.TIFNetworkTests("test_forward_chunk_matches_step_forward")
        extra = dict(FAST, adapt=True, adapt_debias=True, adapt_layers=["enc1"])
        for mode in ("carry", "reset_each_window"):
            with self.subTest(mode=mode):
                helper.check_pair(source, False, True, mode, extra)
        net = make_tif_net(**extra)
        with torch.no_grad():
            net.start_tif_stats()
            states = None
            for k in range(CHUNKS[-1][1]):
                x, events, _, _ = source.window(k)
                _, states, _ = net(x, events, states)
            stats = net.stop_tif_stats()
        self.assertEqual(len(states[0]), 3)
        self.assertTrue(all(torch.is_tensor(s) for s in states[1:]))     # 深层状态仍是张量 M
        self.assertIn("baseline_level_mean", stats["enc1"])
        self.assertTrue(all("baseline_level_mean" not in stats[n] for n in LAYER_NAMES[1:]))

    def test_operation_estimate_counts_selected_layers(self):
        rates = {name: 0.5 for name in LAYER_NAMES}
        plain = estimate_operations(EvSpSegNetStream(channels=SMALL, neuron="tif"), H, W, rates, 100)
        net = EvSpSegNetStream(channels=SMALL, neuron="tif",
                               tif={"adapt": True, "adapt_debias": True, "adapt_layers": ["enc1"]})
        ops = estimate_operations(net, H, W, rates, 100)
        enc1 = ops["tif"]["adapt_state_elements"]
        self.assertEqual(enc1, net.enc1.out_ch * H * W)
        self.assertEqual(ops["state_elements"], plain["state_elements"] + enc1)
        self.assertEqual(ops["tif"]["elementwise_ops_per_window"],
                         plain["tif"]["elementwise_ops_per_window"] + 5 * enc1)
        self.assertEqual(ops["tif"]["adapt_layers"], ["enc1"])


class AdaptConfigTests(unittest.TestCase):
    def test_configs_differ_only_in_adapt(self):
        from utils.stream_common import load_flat_config
        for base, new in (("configs/evisseg_stream_tif_notransport.yaml", "configs/evisseg_stream_tif_adapt.yaml"),
                          ("configs/evisseg_stream_tif_notransport_evflying.yaml",
                           "configs/evisseg_stream_tif_adapt_evflying.yaml")):
            with self.subTest(new=new):
                a, b = load_flat_config(base), load_flat_config(new)
                self.assertEqual(set(a), set(b))
                for key in a:
                    if key not in ("tif", "save_root"):
                        self.assertEqual(a[key], b[key], key)
                self.assertNotEqual(a["save_root"], b["save_root"])
                ta, tb = tif_settings(a["tif"]), tif_settings(b["tif"])
                self.assertEqual({k: v for k, v in tb.items() if not k.startswith("adapt")},
                                 {k: v for k, v in ta.items() if not k.startswith("adapt")})
                self.assertFalse(ta["adapt"])
                self.assertEqual((tb["adapt"], tb["adapt_tau_init_ms"], tb["adapt_tau_min_ms"], tb["adapt_tau_max_ms"]),
                                 (True, 2000.0, 1000.0, 10000.0))
                self.assertFalse(tb["adapt_debias"])

    def test_debias_configs_differ_only_in_debias(self):
        from utils.stream_common import load_flat_config
        for base_path, new in (("configs/evisseg_stream_tif_adapt.yaml", "configs/evisseg_stream_tif_adapt2.yaml"),
                               ("configs/evisseg_stream_tif_adapt_evflying.yaml",
                                "configs/evisseg_stream_tif_adapt2_evflying.yaml")):
            with self.subTest(new=new):
                a, b = load_flat_config(base_path), load_flat_config(new)
                self.assertEqual(set(a), set(b))
                for key in a:
                    if key not in ("tif", "save_root"):
                        self.assertEqual(a[key], b[key], key)
                self.assertNotEqual(a["save_root"], b["save_root"])
                ta, tb = tif_settings(a["tif"]), tif_settings(b["tif"])
                self.assertEqual(dict(ta, adapt_debias=True), tb)

    def test_input_layer_configs_differ_only_in_layers(self):
        from utils.stream_common import load_flat_config
        for base_path, new in (("configs/evisseg_stream_tif_adapt2.yaml", "configs/evisseg_stream_tif_adapt3.yaml"),
                               ("configs/evisseg_stream_tif_adapt2_evflying.yaml",
                                "configs/evisseg_stream_tif_adapt3_evflying.yaml")):
            with self.subTest(new=new):
                a, b = load_flat_config(base_path), load_flat_config(new)
                self.assertEqual(set(a), set(b))
                for key in a:
                    if key not in ("tif", "save_root"):
                        self.assertEqual(a[key], b[key], key)
                self.assertNotEqual(a["save_root"], b["save_root"])
                self.assertEqual(dict(tif_settings(a["tif"]), adapt_layers=["enc1"]), tif_settings(b["tif"]))


if __name__ == "__main__":
    unittest.main()
