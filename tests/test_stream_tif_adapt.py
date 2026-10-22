"""TIF 输入层工作点自校准（tif.adapt，只在 enc1，偏差校正的慢基线；Stream-SNN-Innovations.md 12.5、12.10）的单元测试。

CPU、float64，不需要数据集。10-22 在全部 7 层上做的两版已否定并删除，这里同时核对它们不能再被构造。
    1. 神经元：参照 = 历史加权均值、首窗不减参照；平稳输入 M -> 0（与背景水平无关）；均值为负的通道与基础版相同；
       大直流下没有热身过程；None / 零状态 / reset 与基础版一致；forward_steps（含 keep、反向重算）与逐窗等价；统计
    2. 网络：配置校验（只接受 enc1 + 偏差校正；旧两版报错；旧配置里关闭时的键可读入）、只有 enc1 自校准、
       其余参数与基础版同初始化、片段与逐窗等价（carry / reset）、reset 模式与基础版逐位相同、运算量
    3. 配置文件：adapt3 与基础版只差 adapt 各项与 save_root
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
# 网络配置里打开自校准的唯一合法写法
ADAPT = dict(FAST, adapt=True, adapt_debias=True, adapt_layers=["enc1"])


def adapt_neuron(channels=3, **kw):
    return tif_neuron(channels, adapt=True, **dict(FAST, **kw))


def alpha_of(n):
    return torch.exp(-50.0 / n.adapt_tau()).double().view(1, -1, 1, 1)


class AdaptNeuronTests(unittest.TestCase):
    def test_parameters_and_validation(self):
        n = adapt_neuron(3)
        self.assertEqual(sorted(name for name, _ in n.named_parameters()), ["a", "adapt_a", "k_raw", "log_theta"])
        self.assertLess(max_error(n.adapt_tau().double(), torch.full((3,), 200.0, dtype=torch.float64)), 1e-4)
        for bad in ({"adapt_tau_min_ms": 300.0}, {"adapt_tau_max_ms": 150.0}, {"adapt_tau_min_ms": 0.0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                adapt_neuron(3, **bad)
        with self.assertRaises(ValueError):
            tif_neuron(3, adapt=1)
        with self.assertRaises(TypeError):
            tif_neuron(3, adapt=True, adapt_debias=True)                     # 开关已删除
        self.assertNotIn("adapt_a", dict(tif_neuron(3).named_parameters()))
        current = torch.rand(2, 3, 4, 5, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        _, state, _ = n(current, None)
        self.assertEqual(len(state), 3)
        self.assertEqual(tuple(state[2].shape), (2, 3, 1, 1))
        with self.assertRaises(ValueError):
            n(current, state[:2])

    def test_manual_running_mean(self):
        n = adapt_neuron(3)
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
            out, state, memory = n(current, state)
            self.assertLess(max_error(memory, m), 1e-12, t)
            self.assertLess(max_error(out, theta * torch.round(m / theta)), 1e-12, t)
            weights = [alpha ** (t - i) for i in range(t + 1)]
            mean_now = sum(w * c for w, c in zip(weights, currents[:t + 1])) / sum(weights)
            self.assertLess(max_error(state[1] / state[2], mean_now), 1e-12, t)

    def test_offset_invariance_and_negative_channels(self):
        plain, adapt = tif_neuron(1), adapt_neuron(1)
        _, k, theta = coefficients(plain)
        for mu in (0.1, 0.5, 3.0, 50.0):
            current = torch.full((1, 1, 2, 2), mu, dtype=torch.float64)
            s_p, s_a, top = None, None, 0
            for t in range(200):
                _, s_p, m_p = plain(current, s_p)
                _, s_a, m_a = adapt(current, s_a)
                top += int(bool((m_a >= 7 * theta - 1e-12).all()))
            self.assertLess(max_error(m_p, torch.minimum(torch.relu(current - k), 7 * theta)), 1e-9, mu)
            self.assertEqual(float(m_a.abs().max()), 0.0, mu)                   # 与背景水平无关
            self.assertLessEqual(top, 1, mu)                                    # 大直流也没有热身过程
        g = torch.Generator().manual_seed(4)
        s_p, s_a = None, None
        for _ in range(30):
            current = -1.0 + 0.3 * torch.rand(2, 1, 3, 3, generator=g, dtype=torch.float64)
            out_p, s_p, m_p = plain(current, s_p)
            out_a, s_a, m_a = adapt(current, s_a)
            self.assertEqual(max_error(m_p, m_a), 0.0)
            self.assertEqual(max_error(out_p, out_a), 0.0)

    def test_target_pulse_after_background_rise(self):
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

    def test_none_zero_state_reset_and_steps(self):
        n_a = adapt_neuron(3)
        n_b = copy.deepcopy(n_a)
        plain = tif_neuron(3)
        g = torch.Generator().manual_seed(11)
        current = torch.randn(6, 2, 3, 6, 7, generator=g, dtype=torch.float64).clamp(min=-0.2) * 0.8 + 0.2
        with torch.no_grad():
            ref = plain(current[0], None)
            for state in (None, tuple(torch.zeros_like(x) for x in n_a(current[0], None)[1])):
                out, _, memory = n_a(current[0], state)
                self.assertEqual(max_error(out, ref[0]), 0.0)
                self.assertEqual(max_error(memory, ref[2]), 0.0)
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

    def test_statistics_baseline_level(self):
        n = adapt_neuron(2)
        g = torch.Generator().manual_seed(5)
        n.start_stats()
        state, refs = None, []
        for _ in range(4):
            _, state, _ = n(random_map(g, (1, 2, 6, 6), scale=1.5), state)
            refs.append(state[1] / state[2])
        stats = n.stop_stats()
        _, _, theta = coefficients(n)
        expected = float(sum((torch.relu(r) / theta).sum() for r in refs)) / (4.0 * refs[0].numel())
        self.assertAlmostEqual(stats["baseline_level_mean"], expected, places=12)


class AdaptNetworkTests(unittest.TestCase):
    def test_config_validation(self):
        self.assertFalse(TIF_DEFAULTS["adapt"])
        self.assertTrue(tif_settings({"adapt": True, "adapt_debias": True, "adapt_layers": ["enc1"]})["adapt"])
        for bad in ({"adapt": True},                                                       # 第一版
                    {"adapt": True, "adapt_debias": True},                                 # 第二版（全部层）
                    {"adapt": True, "adapt_debias": False, "adapt_layers": ["enc1"]},
                    {"adapt": True, "adapt_debias": True, "adapt_layers": ["enc1", "enc2"]},
                    {"adapt_debias": True}, {"adapt_layers": ["enc1"]}, {"adapt": "yes"},
                    dict(ADAPT, adapt_tau_min_ms=3000.0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tif_settings(bad)
        # 改动前保存的配置里，关闭时的键（adapt_debias: false、adapt_layers: null）可以读入并丢弃
        self.assertEqual(tif_settings({"adapt": False, "adapt_debias": False, "adapt_layers": None}), TIF_DEFAULTS)
        self.assertNotIn("adapt_debias", tif_settings(ADAPT))

    def test_only_enc1_and_same_initialization(self):
        torch.manual_seed(3)
        plain = dict(EvSpSegNetStream(channels=SMALL, neuron="tif").named_parameters())
        torch.manual_seed(3)
        net = EvSpSegNetStream(channels=SMALL, neuron="tif", tif=ADAPT)
        self.assertEqual(net.tif_adapt_layers, ("enc1",))
        for name, block in zip(LAYER_NAMES, net.blocks()):
            self.assertEqual(block.neuron.adapt, name == "enc1", name)
        for name, param in net.named_parameters():
            if name in plain:
                self.assertEqual(max_error(param, plain[name]), 0.0, name)
            else:
                self.assertEqual(name, "enc1.neuron.adapt_a")
        self.assertEqual(EvSpSegNetStream(channels=SMALL, neuron="tif").tif_adapt_layers, ())

    def test_forward_chunk_matches_step_forward(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        helper = base.TIFNetworkTests("test_forward_chunk_matches_step_forward")
        for mode in ("carry", "reset_each_window"):
            with self.subTest(mode=mode):
                helper.check_pair(source, False, True, mode, ADAPT)

    def test_reset_mode_equals_plain_and_state_structure(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        net_a = make_tif_net(**ADAPT)
        net_p = make_tif_net()
        net_p.load_state_dict({k: v for k, v in net_a.state_dict().items() if not k.endswith("adapt_a")})
        with torch.no_grad():
            for k in range(CHUNKS[-1][1]):
                x, events, _, _ = source.window(k)
                la, _, _ = net_a(x, events, None, state_mode="reset_each_window")
                lp, _, _ = net_p(x, events, None, state_mode="reset_each_window")
                self.assertEqual(max_error(la, lp), 0.0)
            net_a.start_tif_stats()
            states = None
            for k in range(CHUNKS[-1][1]):
                x, events, _, _ = source.window(k)
                _, states, _ = net_a(x, events, states)
            stats = net_a.stop_tif_stats()
        self.assertEqual(len(states[0]), 3)
        self.assertTrue(all(torch.is_tensor(s) for s in states[1:]))           # 深层状态仍是张量 M
        self.assertEqual(len(list(state_tensors(states))), 9)
        self.assertGreater(stats["enc1"]["baseline_level_mean"], 0.0)
        self.assertTrue(all("baseline_level_mean" not in stats[n] for n in LAYER_NAMES[1:]))

    def test_operation_estimate(self):
        rates = {name: 0.5 for name in LAYER_NAMES}
        plain = estimate_operations(EvSpSegNetStream(channels=SMALL, neuron="tif"), H, W, rates, 100)
        net = EvSpSegNetStream(channels=SMALL, neuron="tif", tif=ADAPT)
        ops = estimate_operations(net, H, W, rates, 100)
        enc1 = ops["tif"]["adapt_state_elements"]
        self.assertEqual(enc1, net.enc1.out_ch * H * W)
        self.assertEqual(ops["state_elements"], plain["state_elements"] + enc1)
        self.assertEqual(ops["tif"]["elementwise_ops_per_window"],
                         plain["tif"]["elementwise_ops_per_window"] + 5 * enc1)
        self.assertEqual(ops["sop"], plain["sop"])
        self.assertEqual(ops["tif"]["adapt_layers"], ["enc1"])


class AdaptConfigTests(unittest.TestCase):
    def test_configs_differ_only_in_adapt(self):
        from utils.stream_common import load_flat_config
        for base_path, new in (("configs/evisseg_stream_tif_notransport.yaml", "configs/evisseg_stream_tif_adapt3.yaml"),
                               ("configs/evisseg_stream_tif_notransport_evflying.yaml",
                                "configs/evisseg_stream_tif_adapt3_evflying.yaml")):
            with self.subTest(new=new):
                a, b = load_flat_config(base_path), load_flat_config(new)
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


if __name__ == "__main__":
    unittest.main()
