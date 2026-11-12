"""V5 numerical regression retained for the V15 release; legacy variants removed."""
import copy


import math


import unittest


from types import SimpleNamespace


import numpy as np


import torch


from dataset import stream_windows as sw


from dataset.stream_source import NumpyWindowSource, TorchWindowSource, make_window_source


from model.evspsegnet_stream import LAYER_NAMES, EvSpSegNetStream


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


def rel_error(a, b):
    """相对误差：最大绝对误差 / max(量级, 1)。graded 对照放大了增益，膜电位量级很大，只能比相对误差。"""
    return max_error(a, b) / max(float(a.detach().abs().max()), 1.0) if a.numel() else 0.0


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


if __name__ == "__main__":
    unittest.main()
