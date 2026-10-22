"""TIF 滞回汇报（tif.hysteresis，Stream-SNN-Innovations.md 12.8~12.9）的单元测试（CPU、float64，不需要数据集）。

    1. 定义：规则 |x - q_prev| < 1 时保持、否则 q = round(x)；首窗 / None / 零状态一致；误差 |x - q| < 1；
       M 回到 0 时汇报必回到 0（开区间，不卡在 1 级）；校验
    2. 消息：电平分界处的噪声不再产生成串消息；消息事件数 <= 2 * TV(x)；统计用实际汇报的电平
    3. forward_steps（含 keep 掩码、反向重算）与逐窗 forward 等价（前向与梯度）
    4. 网络：关闭时与基础版逐位相同、片段与逐窗等价（carry / reset）、运算量、配置文件只差该差的键
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
from model.lif2d_stream import mask_state  # noqa: E402
from test_stream_layerwise import CFG, CHUNKS, H, Q99, SMALL, W, max_error, synthetic_sequence  # noqa: E402
import test_stream_tif as base  # noqa: E402  （只导入模块，避免 unittest 重复收集它的测试）
from test_stream_tif import coefficients, make_tif_net, random_map, tif_neuron  # noqa: E402


def hyst_neuron(channels=3, **kw):
    return tif_neuron(channels, hysteresis=True, **kw)


def drive_for(n, targets):
    """构造一串输入，使 M/theta 依次精确等于 targets（逐窗反解 EMA；只用单通道、未截断区间）。"""
    lam, k, theta = coefficients(n)
    lam, k, theta = float(lam.flatten()[0]), float(k.flatten()[0]), float(theta.flatten()[0])
    currents, m_prev = [], 0.0
    for x in targets:
        m = x * theta
        currents.append(torch.full((1, 1, 1, 1), (m - lam * m_prev) / (1 - lam) + k, dtype=torch.float64))
        m_prev = m
    return currents


class HysteresisDefinitionTests(unittest.TestCase):
    def test_validation(self):
        self.assertFalse(TIF_DEFAULTS["hysteresis"])
        for bad in ({"quantize": False}, {"adapt": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                hyst_neuron(3, **bad)
            with self.subTest(cfg=bad), self.assertRaises(ValueError):
                tif_settings(dict(bad, hysteresis=True))
        with self.assertRaises(ValueError):
            tif_settings({"hysteresis": 1})
        n = hyst_neuron(3)
        self.assertEqual(sorted(name for name, _ in n.named_parameters()), ["a", "k_raw", "log_theta"])

    def test_rule_on_a_trajectory(self):
        # x 依次为下列值时，汇报电平应为 expected（q_prev 从 0 开始）
        xs = [0.6, 0.95, 1.2, 1.4, 1.6, 2.4, 2.6, 1.5, 0.3, 0.0, 0.4, 3.3]
        expected = [0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 3]
        n = hyst_neuron(1)
        _, _, theta = coefficients(n)
        state = None
        for x, q in zip(xs, expected):
            current = drive_for(n, [x])[0] if state is None else None
            if state is not None:
                lam, k, _ = coefficients(n)
                m_prev = state[0]
                current = (x * theta - lam * m_prev) / (1 - lam) + k
            out, state, memory = n(current, state)
            self.assertAlmostEqual(float(memory / theta), x, places=9)
            self.assertEqual(float(state[1]), q, x)
            self.assertLess(max_error(out, q * theta), 1e-12)

    def test_zero_release_and_error_bound(self):
        # 任意输入下 |x - q| < 1；M 为 0 处汇报必为 0
        n = hyst_neuron(3)
        _, _, theta = coefficients(n)
        g = torch.Generator().manual_seed(3)
        state = None
        for t in range(60):
            scale = 3.0 if t < 30 else 0.0                   # 后半段无输入：M 衰减到 0
            out, state, memory = n(random_map(g, (2, 3, 5, 6), scale=scale), state)
            x, q = memory / theta, state[1]
            self.assertLess(float((x - q).abs().max()), 1.0)
            self.assertTrue(bool((q[memory == 0] == 0).all()))
            self.assertLess(max_error(out, theta * q), 1e-12)
        self.assertEqual(float(state[1].abs().max()), 0.0)

    def test_none_zero_state_and_reset(self):
        n = hyst_neuron(3)
        current = torch.rand(2, 3, 4, 5, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 2
        zero = (torch.zeros_like(current), torch.zeros_like(current))
        a, b = n(current, None), n(current, zero)
        self.assertEqual(max_error(a[0], b[0]), 0.0)
        self.assertEqual(max_error(a[1][1], b[1][1]), 0.0)
        with self.assertRaises(ValueError):
            n(current, current)


class HysteresisMessageTests(unittest.TestCase):
    def test_boundary_noise_no_chatter(self):
        # x 停在 2.5 附近、带 ±0.2 的噪声：基础版每窗都可能在 2 和 3 之间跳，滞回版至多一条消息
        g = torch.Generator().manual_seed(7)
        xs = [2.5 + 0.2 * (2 * float(torch.rand(1, generator=g)) - 1) for _ in range(80)]
        counts = {}
        for name, n in (("plain", tif_neuron(1)), ("hyst", hyst_neuron(1))):
            currents = drive_for(n, xs)
            _, _, theta = coefficients(n)
            n.start_stats()
            state = None
            for current in currents:
                _, state, _ = n(current, state)
            counts[name] = n.stop_stats()["message_nonzero_rate"] * len(xs)
        self.assertGreater(counts["plain"], 20)
        self.assertLessEqual(counts["hyst"], 2)                 # 首次升到 2（或 3）

    def test_message_events_bounded_by_total_variation(self):
        n = hyst_neuron(2)
        _, _, theta = coefficients(n)
        g = torch.Generator().manual_seed(9)
        state, x_prev, tv, events = None, torch.zeros(1, 2, 4, 4, dtype=torch.float64), None, None
        q_prev = torch.zeros(1, 2, 4, 4, dtype=torch.float64)
        for t in range(50):
            _, state, memory = n(random_map(g, (1, 2, 4, 4), scale=2.5, density=0.7), state)
            x = memory / theta
            step_tv = (x - x_prev).abs()
            step_ev = (state[1] != q_prev).double()
            tv = step_tv if tv is None else tv + step_tv
            events = step_ev if events is None else events + step_ev
            x_prev, q_prev = x, state[1]
        self.assertTrue(bool((events <= 2 * tv + 1e-9).all()))
        self.assertGreater(float(events.sum()), 0.0)

    def test_statistics_use_reported_levels(self):
        n = hyst_neuron(2)
        g = torch.Generator().manual_seed(5)
        currents = [random_map(g, (1, 2, 6, 6), scale=1.5) for _ in range(6)]
        n.start_stats()
        state, levels = None, []
        for current in currents:
            _, state, _ = n(current, state)
            levels.append(state[1])
        stats = n.stop_stats()
        deltas = [levels[0]] + [levels[t] - levels[t - 1] for t in range(1, 6)]
        numel = 6.0 * levels[0].numel()
        self.assertAlmostEqual(stats["message_abs_rate"], float(sum(d.abs().sum() for d in deltas)) / numel, places=12)
        self.assertAlmostEqual(stats["level_mean"], float(sum(l.sum() for l in levels)) / numel, places=12)


class HysteresisStepsTests(unittest.TestCase):
    def test_forward_steps_matches_forward_with_keep(self):
        n_a = hyst_neuron(3)
        n_b = copy.deepcopy(n_a)
        g = torch.Generator().manual_seed(11)
        current = torch.randn(6, 2, 3, 6, 7, generator=g, dtype=torch.float64).clamp(min=-0.2) * 1.5 + 0.3
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


class HysteresisNetworkTests(unittest.TestCase):
    def test_off_is_plain(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        net_a, net_b = make_tif_net(), make_tif_net(hysteresis=False)
        with torch.no_grad():
            sa = sb = None
            for k in range(CHUNKS[-1][1]):
                x, events, _, _ = source.window(k)
                la, sa, _ = net_a(x, events, sa)
                lb, sb, _ = net_b(x, events, sb)
                self.assertEqual(max_error(la, lb), 0.0)

    def test_forward_chunk_matches_step_forward(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        helper = base.TIFNetworkTests("test_forward_chunk_matches_step_forward")
        for mode in ("carry", "reset_each_window"):
            with self.subTest(mode=mode):
                helper.check_pair(source, False, True, mode, {"hysteresis": True})

    def test_operation_estimate_and_state(self):
        rates = {name: 0.5 for name in LAYER_NAMES}
        plain = estimate_operations(EvSpSegNetStream(channels=SMALL, neuron="tif"), H, W, rates, 100)
        hyst = estimate_operations(EvSpSegNetStream(channels=SMALL, neuron="tif", tif={"hysteresis": True}),
                                   H, W, rates, 100)
        self.assertEqual(hyst["state_elements"], 2 * plain["state_elements"])
        self.assertEqual(hyst["tif"]["elementwise_ops_per_window"],
                         plain["tif"]["elementwise_ops_per_window"] * 13 // 11)
        self.assertEqual(hyst["sop"], plain["sop"])
        self.assertTrue(hyst["tif"]["hysteresis"])

    def test_configs_differ_only_in_hysteresis(self):
        from utils.stream_common import load_flat_config
        for base_path, new in (("configs/evisseg_stream_tif_notransport.yaml", "configs/evisseg_stream_tif_hyst.yaml"),
                               ("configs/evisseg_stream_tif_notransport_evflying.yaml",
                                "configs/evisseg_stream_tif_hyst_evflying.yaml")):
            with self.subTest(new=new):
                a, b = load_flat_config(base_path), load_flat_config(new)
                self.assertEqual(set(a), set(b))
                for key in a:
                    if key not in ("tif", "save_root"):
                        self.assertEqual(a[key], b[key], key)
                self.assertNotEqual(a["save_root"], b["save_root"])
                self.assertEqual(dict(tif_settings(a["tif"]), hysteresis=True), tif_settings(b["tif"]))


if __name__ == "__main__":
    unittest.main()
