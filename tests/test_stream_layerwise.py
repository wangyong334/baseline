"""训练加速选项的等价性测试：逐层时间并行（execution=layer）与设备端输入构造（input_device=gpu）。

全部在 CPU 上用 float64 小网络运行，不需要数据集：
    1. 查找表归一化与 normalize_counts 逐位相同；torch 输入构造与 numpy 输入构造逐位相同
    2. LIF.forward_steps 与逐窗 forward 等价（前向与梯度）
    3. forward_chunk 与逐窗 forward 等价：两种解码器 × LIF/ReLU × carry/reset，跨片段传递状态，含梯度
    4. train_sequence / run_sequence 在四种选项组合下结果一致（更新后参数、概率、每窗混淆矩阵、发放统计）
误差一律用 (a - b).abs().max() 比较，不用 torch.testing.assert_close（torch 1.9 默认比较步长）。
"""
import copy
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from dataset import stream_windows as sw
from dataset.stream_source import NumpyWindowSource, TorchWindowSource, make_window_source
from model.evspsegnet_stream import LAYER_NAMES, EvSpSegNetStream
from model.lif2d_stream import StreamingLIF2d
from utils.stream_common import subset_due

H, W, WINDOWS = 16, 24, 12
SMALL = (2, 4, 4, 4)
TOL = 1e-10
Q99 = np.linspace(0.5, 1.5, 12).astype(np.float32)
CFG = dict(time_bins=5, pad_height=H, pad_width=W, input_clip=3.0, tbptt_k=4, optimizer_step="per_sequence",
           grad_clip=1.0, threshold=0.5, eval_chunk=5, state_mode="carry")
CHUNKS = ((0, 3), (3, 8), (8, 12))


def max_error(a, b):
    """两个张量的最大绝对误差（空张量记为 0）。"""
    return float((a.detach() - b.detach()).abs().max()) if a.numel() else 0.0


def synthetic_sequence(seed=0, n=900):
    """合成序列：含两个空窗、一个计数超过查表饱和值的热点像素，文件顺序打乱（检验下标回填）。"""
    rng = np.random.RandomState(seed)
    t = rng.randint(0, 50 * WINDOWS, n)
    t = t[(t // 50 != 3) & (t // 50 != 7)]
    x, y, p = rng.randint(0, W, t.size), rng.randint(0, H, t.size), rng.randint(0, 2, t.size)
    hot = 150
    t = np.concatenate([t, np.full(hot, 5 * 50 + 12)])
    x, y, p = np.concatenate([x, np.full(hot, 7)]), np.concatenate([y, np.full(hot, 5)]), np.concatenate([p, np.ones(hot)])
    perm = rng.permutation(t.size)
    t, x, y, p = t[perm].astype(np.int64), x[perm].astype(np.int64), y[perm].astype(np.int64), p[perm].astype(np.int8)
    label = (rng.rand(t.size) < 0.2).astype(np.float32)
    order, bounds = sw.split_windows(t, 50, WINDOWS)
    bins, local = sw.time_bin_and_local(t, 50, 5)
    return sw.StreamSequence("synthetic.npz", x, y, t, p, label, label.astype(np.float64), bins, local, order, bounds)


def make_net(merged=False, neuron="lif", seed=5, threshold=0.05):
    """float64 小网络；调低阈值让各层都有发放，核对才不是在沉默网络上做的。"""
    torch.manual_seed(seed)
    net = EvSpSegNetStream(channels=SMALL, neuron=neuron, merged_decoder=merged).double()
    for block in net.blocks():
        if hasattr(block.neuron, "v_threshold"):
            block.neuron.v_threshold = threshold
    return net


class InputSourceTests(unittest.TestCase):
    def test_normalization_table_matches_formula(self):
        table, n_sat = sw.normalization_table(Q99, 3.0)
        rng = np.random.RandomState(1)
        counts = rng.randint(0, 3 * n_sat, size=(12, 9, 13)).astype(np.int64)
        expected = sw.normalize_counts(counts, Q99, 3.0)
        looked_up = table[np.arange(12)[:, None, None], np.minimum(counts, n_sat)]
        self.assertTrue(np.array_equal(expected, looked_up))
        self.assertEqual(expected.dtype, looked_up.dtype)

    def test_torch_source_matches_numpy_source(self):
        seq = synthetic_sequence()
        reference = NumpyWindowSource(seq, CFG, Q99, torch.device("cpu"))
        candidate = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"))
        saturated = False
        for k in range(WINDOWS):
            a, b = reference.window(k), candidate.window(k)
            self.assertTrue(torch.equal(a[0], b[0]), k)                  # 输入逐位相同
            self.assertEqual(set(a[1]), set(b[1]))
            for key in a[1]:
                self.assertTrue(torch.equal(a[1][key], b[1][key]), (k, key))
                self.assertEqual(a[1][key].dtype, b[1][key].dtype, key)
            self.assertTrue(torch.equal(a[2], b[2]))
            self.assertTrue(np.array_equal(a[3], b[3]))
            saturated = saturated or bool((a[0] == 3.0).any())
        self.assertTrue(saturated)                                       # 热点像素确实超过饱和计数
        for start, end in CHUNKS + ((3, 4), (0, WINDOWS)):
            a, b = reference.chunk(start, end), candidate.chunk(start, end)
            self.assertEqual(tuple(b[0].shape), (end - start, 1, 12, H, W))
            self.assertTrue(torch.equal(a[0], b[0]))
            for key in ("t", "b", "y", "x", "p", "t_local"):
                self.assertTrue(torch.equal(a[1][key], b[1][key]), key)
            self.assertTrue(torch.equal(a[2], b[2]))
            self.assertTrue(np.array_equal(a[3], b[3]))
        empty = candidate.chunk(3, 4)
        self.assertEqual(float(empty[0].abs().sum()), 0.0)
        self.assertEqual(int(empty[2].numel()), 0)

    def test_factory_and_dtype(self):
        seq = synthetic_sequence()
        self.assertIsInstance(make_window_source(seq, CFG, Q99, torch.device("cpu")), NumpyWindowSource)
        source = make_window_source(seq, dict(CFG, input_device="gpu"), Q99, torch.device("cpu"), torch.float64)
        self.assertIsInstance(source, TorchWindowSource)
        x, events, labels, _ = source.chunk(0, 2)
        self.assertEqual((x.dtype, events["p"].dtype, labels.dtype, events["y"].dtype),
                         (torch.float64, torch.float64, torch.float64, torch.long))
        with self.assertRaises(ValueError):
            make_window_source(seq, dict(CFG, input_device="npu"), Q99, torch.device("cpu"))


class LIFStepsTests(unittest.TestCase):
    def test_forward_steps_matches_repeated_forward(self):
        for carry in (True, False):
            with self.subTest(carry=carry):
                torch.manual_seed(0)
                neuron = StreamingLIF2d(3, 50.0, 200.0, 50.0, 2000.0, 1.0).double()
                neuron.a.data.uniform_(-3, 3)
                current = (torch.rand(6, 2, 3, 4, 5, dtype=torch.float64) * 2).requires_grad_()
                start = torch.rand(2, 3, 4, 5, dtype=torch.float64)
                weights = torch.rand(6, 2, 3, 4, 5, dtype=torch.float64)

                state, spikes, u_pres = start, [], []
                for t in range(6):
                    s, state, u = neuron(current[t], state if carry else None)
                    spikes.append(s)
                    u_pres.append(u)
                loss_a = (torch.stack(u_pres) * weights).sum() + (torch.stack(spikes) * weights).sum() + state.sum()
                grad_a = torch.autograd.grad(loss_a, [current, neuron.a], allow_unused=True)

                s, final, u = neuron.forward_steps(current, start, carry=carry)
                self.assertEqual(max_error(s, torch.stack(spikes)), 0.0)
                self.assertLess(max_error(u, torch.stack(u_pres)), TOL)
                self.assertLess(max_error(final, state), TOL)
                loss_b = (u * weights).sum() + (s * weights).sum() + final.sum()
                grad_b = torch.autograd.grad(loss_b, [current, neuron.a], allow_unused=True)
                self.assertLess(max_error(grad_a[0], grad_b[0]), TOL)
                if carry:
                    self.assertLess(max_error(grad_a[1], grad_b[1]), TOL)
                    self.assertGreater(float(grad_b[1].abs().sum()), 0.0)
                else:
                    self.assertIsNone(grad_b[1])                        # reset 模式不使用 beta
                self.assertIsNone(neuron.forward_steps(current, start, carry, keep_u_pre=False)[2])


class ForwardChunkTests(unittest.TestCase):
    def test_forward_chunk_matches_step_forward(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        for merged in (False, True):
            for neuron in ("lif", "relu"):
                for mode in ("carry", "reset_each_window"):
                    with self.subTest(merged=merged, neuron=neuron, mode=mode):
                        self.check_pair(source, merged, neuron, mode)

    def check_pair(self, source, merged, neuron, mode):
        net_a = make_net(merged, neuron)
        net_b = copy.deepcopy(net_a)
        g = torch.Generator().manual_seed(3)

        states, logits_a, loss_a, info_a = None, [], 0.0, {"spikes": [], "u_pre": []}
        chunk_states_a = []
        for start, end in CHUNKS:
            for k in range(start, end):
                x, events, _, _ = source.window(k)
                logits, states, info = net_a(x, events, states, state_mode=mode, collect=True)
                logits_a.append(logits)
                info_a["spikes"].append(info["spikes"])
                info_a["u_pre"].append(info["u_pre"])
            chunk_states_a.append(states)
        weights = torch.rand(sum(int(l.numel()) for l in logits_a), generator=g, dtype=torch.float64)
        logits_a = torch.cat(logits_a)
        (logits_a * weights).sum().backward()

        states, logits_b, active = None, [], [0] * 7
        for i, (start, end) in enumerate(CHUNKS):
            x, events, _, _ = source.chunk(start, end)
            logits, states, info = net_b.forward_chunk(x, events, states, state_mode=mode, collect=True)
            logits_b.append(logits)
            for layer in range(7):
                spikes_a = torch.stack([w[layer] for w in info_a["spikes"][start:end]])
                u_pre_a = torch.stack([w[layer] for w in info_a["u_pre"][start:end]])
                self.assertEqual(int(((spikes_a > 0) != (info["spikes"][layer] > 0)).sum()), 0, LAYER_NAMES[layer])
                self.assertLess(max_error(u_pre_a, info["u_pre"][layer]), TOL, LAYER_NAMES[layer])
                active[layer] += int((info["spikes"][layer] > 0).sum())
            for a, b in zip(chunk_states_a[i], states):
                if a is None:
                    self.assertIsNone(b)
                else:
                    self.assertLess(max_error(a, b), TOL)
        logits_b = torch.cat(logits_b)
        self.assertLess(max_error(logits_a, logits_b), TOL)
        (logits_b * weights).sum().backward()
        self.assertTrue(all(n > 0 for n in active), active)             # 每层都有发放
        for (name, pa), (_, pb) in zip(net_a.named_parameters(), net_b.named_parameters()):
            if pa.grad is None:
                self.assertIsNone(pb.grad, name)
                continue
            scale = max(float(pa.grad.abs().max()), 1.0)
            self.assertLess(max_error(pa.grad, pb.grad) / scale, 1e-9, name)

    def test_rejects_wrong_input_rank(self):
        net = make_net()
        x, events, _, _ = TorchWindowSource(synthetic_sequence(), CFG, Q99, torch.device("cpu"),
                                            torch.float64).window(0)
        with self.assertRaises(ValueError):
            net.forward_chunk(x, events, None)


class TrainingLoopTests(unittest.TestCase):
    def test_train_sequence_equivalent_across_options(self):
        import train_stream_v1 as T
        seq = synthetic_sequence(seed=2)
        for optimizer_step in ("per_sequence", "per_chunk"):
            results = {}
            for execution in ("step", "layer"):
                for input_device in ("cpu", "gpu"):
                    cfg = dict(CFG, optimizer_step=optimizer_step, execution=execution, input_device=input_device)
                    net = make_net(merged=True, seed=9)
                    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
                    out = T.train_sequence(net, seq, optimizer, cfg, Q99, 3.0, torch.device("cpu"),
                                           np.random.RandomState(4))
                    results[(execution, input_device)] = (net, out)
            ref_net, ref = results[("step", "cpu")]
            self.assertGreater(ref["steps"], 0)
            self.assertEqual(ref["missing_grads"], [])
            for key, (net, out) in results.items():
                with self.subTest(optimizer_step=optimizer_step, option=key):
                    self.assertEqual(out["steps"], ref["steps"])
                    self.assertEqual(out["events"], ref["events"])
                    self.assertLess(abs(out["loss_sum"] - ref["loss_sum"]), 1e-9 * max(ref["loss_sum"], 1.0))
                    for (name, pa), (_, pb) in zip(ref_net.named_parameters(), net.named_parameters()):
                        self.assertLess(max_error(pa.detach(), pb.detach()), 1e-9, name)

    def test_run_sequence_layer_matches_step(self):
        import train_stream_v1 as T
        seq = synthetic_sequence(seed=3)
        net = make_net(seed=11).eval()
        for mode in ("carry", "reset_each_window"):
            outputs = {}
            for execution, input_device in (("step", "cpu"), ("layer", "gpu"), ("layer", "cpu")):
                cfg = dict(CFG, execution=execution, input_device=input_device)
                monitor = T.LayerMonitor(net.v_threshold)
                with torch.no_grad():
                    probs, confusion = T.run_sequence(net, seq, cfg, Q99, torch.device("cpu"), mode, monitor)
                outputs[execution, input_device] = (probs, confusion, monitor.summary())
            ref_probs, ref_confusion, ref_layers = outputs["step", "cpu"]
            self.assertGreater(int(ref_confusion[:, 3].sum()), 0)
            for key, (probs, confusion, layers) in outputs.items():
                with self.subTest(mode=mode, option=key):
                    self.assertLess(float(np.abs(probs - ref_probs).max()), 1e-6)
                    self.assertTrue(np.array_equal(confusion, ref_confusion))
                    for name in LAYER_NAMES:
                        for stat, value in ref_layers[name].items():
                            self.assertAlmostEqual(layers[name][stat], value, delta=1e-9, msg=(name, stat))
        with self.assertRaises(ValueError):
            T.run_sequence(net, seq, dict(CFG, execution="layer"), Q99, torch.device("cpu"), "carry",
                           timer=T.WindowTimer(0, torch.device("cpu")))

    def test_options_validation_and_subset_schedule(self):
        import train_stream_v1 as T
        self.assertEqual(T.speed_options({}), ("step", "cpu"))
        for bad in ({"execution": "fast"}, {"input_device": "npu"}, {"train_subset_every": 0}, {"eval_chunk": 0}):
            with self.assertRaises(ValueError):
                T.speed_options(bad)
        self.assertEqual([e for e in range(50) if subset_due(e, 50, 1)], list(range(50)))
        self.assertEqual([e for e in range(12) if subset_due(e, 12, 5)], [4, 9, 11])
        args = SimpleNamespace(config="configs/evisseg_stream_v1.yaml", seed=None, state_mode=None, neuron=None,
                               save_root=None, execution="layer", input_device="gpu", train_subset_every=5)
        cfg = T.build_config(args)
        self.assertEqual((cfg["execution"], cfg["input_device"], cfg["train_subset_every"]), ("layer", "gpu", 5))
        self.assertEqual(T.speed_options(T.build_config(SimpleNamespace(**dict(
            vars(args), execution=None, input_device=None, train_subset_every=None)))), ("step", "cpu"))


if __name__ == "__main__":
    unittest.main()
