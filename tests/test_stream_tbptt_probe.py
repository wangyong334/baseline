"""TBPTT 排查工具的测试（CPU、float64 小网络、合成序列，不需要数据集）。

    1. train_sequence 只算梯度模式（optimizer=None）：不改参数、可复现、与训练时裁剪前的梯度逐位相同
    2. 裁剪统计：裁剪前总范数、被裁剪次数、有效缩放
    3. 梯度探针：k=1 没有时间路径（时间占比 0）、k=全长有；余弦 / 范数比的数学
    4. 阈值扫描计数与 window_confusion 一致
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_stream_v1 as T  # noqa: E402
from test_stream_layerwise import CFG, Q99, WINDOWS, make_net, synthetic_sequence  # noqa: E402
from tools import tbptt_grad_probe as P  # noqa: E402
from utils import stream_metrics as sm  # noqa: E402

CPU = torch.device("cpu")
POS_WEIGHT = 4.0


class RecordingOptimizer(torch.optim.SGD):
    """step() 时记录当前梯度（裁剪之后）再做一次 SGD 更新。"""

    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.recorded = []

    def step(self, closure=None):
        self.recorded.append([None if p.grad is None else p.grad.detach().clone()
                              for g in self.param_groups for p in g["params"]])
        return super().step(closure)


def raw_gradient(net, seq, cfg, k):
    """只算梯度模式下的分组梯度。"""
    return P.sequence_gradient(net, seq, cfg, Q99, POS_WEIGHT, CPU, k)[0]


class GradientOnlyTests(unittest.TestCase):
    def test_parameters_unchanged_and_reproducible(self):
        for execution in ("step", "layer"):
            cfg = dict(CFG, execution=execution, input_device="cpu")
            net, seq = make_net(), synthetic_sequence()
            before = [p.detach().clone() for p in net.parameters()]
            out = T.train_sequence(net, seq, None, dict(cfg, tbptt_k=4), Q99, POS_WEIGHT, CPU, None, first_len=4)
            self.assertEqual(out["grad_total_norms"], [])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, net.parameters())), execution)
            self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.parameters()))
            g1, g2 = raw_gradient(net, seq, cfg, 4), raw_gradient(net, seq, cfg, 4)
            for name in g1:
                self.assertTrue(np.array_equal(g1[name], g2[name]), name)

    def test_matches_unclipped_training_gradient(self):
        cfg = dict(CFG, grad_clip=1e12, tbptt_k=4)
        seq = synthetic_sequence()
        net_a = make_net()
        T.train_sequence(net_a, seq, None, cfg, Q99, POS_WEIGHT, CPU, None, first_len=4)
        expect = [None if p.grad is None else p.grad.detach().clone() for p in net_a.parameters()]
        net_b = make_net()
        opt = RecordingOptimizer(net_b.parameters(), lr=1e-3)
        out = T.train_sequence(net_b, seq, opt, cfg, Q99, POS_WEIGHT, CPU, None, first_len=4)
        self.assertEqual(out["steps"], 1)
        for got, want in zip(opt.recorded[0], expect):
            self.assertEqual(got is None, want is None)
            if got is not None:
                self.assertTrue(torch.equal(got, want))
        norm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in expect if g is not None)))
        self.assertAlmostEqual(out["grad_total_norms"][0], norm, delta=1e-9)

    def test_rejects_per_chunk_without_optimizer(self):
        with self.assertRaises(ValueError):
            T.train_sequence(make_net(), synthetic_sequence(), None, dict(CFG, optimizer_step="per_chunk"),
                             Q99, POS_WEIGHT, CPU, None, first_len=4)


class ClipStatisticsTests(unittest.TestCase):
    def test_counts_and_scale(self):
        cfg = dict(CFG, grad_clip=1e-9)       # 任何非零梯度都会被裁剪
        net = make_net()
        out = T.train_sequence(net, synthetic_sequence(), torch.optim.SGD(net.parameters(), lr=0.0),
                               cfg, Q99, POS_WEIGHT, CPU, np.random.RandomState(0))
        self.assertEqual(out["clipped_steps"], 1)
        self.assertEqual(len(out["grad_total_norms"]), 1)
        cfg_chunk = dict(cfg, optimizer_step="per_chunk")
        out = T.train_sequence(net, synthetic_sequence(), torch.optim.SGD(net.parameters(), lr=0.0),
                               cfg_chunk, Q99, POS_WEIGHT, CPU, np.random.RandomState(0))
        self.assertEqual(out["clipped_steps"], out["steps"])
        self.assertEqual(len(out["grad_total_norms"]), out["steps"])
        stats = T.clip_statistics([0.5, 2.0, 4.0], 2, 1.0)
        self.assertAlmostEqual(stats["clip_frac"], 2.0 / 3.0)
        self.assertAlmostEqual(stats["effective_scale_mean"], (1.0 + 0.5 + 0.25) / 3.0)
        self.assertAlmostEqual(stats["total_norm_median"], 2.0)
        self.assertEqual(T.clip_statistics([], 0, 1.0)["steps"], 0)


class ProbeTests(unittest.TestCase):
    def test_k1_has_no_temporal_path(self):
        net, seq = make_net(), synthetic_sequence()
        g1 = raw_gradient(net, seq, CFG, 1)
        gfull = raw_gradient(net, seq, CFG, WINDOWS)
        self.assertEqual(P.temporal_share(g1, g1), 0.0)
        self.assertGreater(P.temporal_share(gfull, g1), 1e-6)     # 状态跨窗传递，时间路径确实存在
        self.assertAlmostEqual(P.compare(gfull, gfull)["cos"], 1.0, delta=1e-12)

    def test_compare_math(self):
        a = {"enc1": np.array([1.0, 0.0]), "readout": np.array([0.0])}
        b = {"enc1": np.array([1.0, 1.0]), "readout": np.array([0.0])}
        r = P.compare(a, b)
        self.assertAlmostEqual(r["cos"], 1.0 / np.sqrt(2.0))
        self.assertAlmostEqual(r["norm_ratio"], 1.0 / np.sqrt(2.0))
        self.assertTrue(np.isnan(r["group_cos"]["readout"]))
        self.assertAlmostEqual(P.temporal_share(b, a), 1.0 / np.sqrt(2.0))


class ThresholdSweepTests(unittest.TestCase):
    def test_counts_match_window_confusion_and_best(self):
        rng = np.random.RandomState(3)
        prob = rng.rand(500).astype(np.float32)
        label = (rng.rand(500) < 0.3).astype(np.float32)
        counts = sm.threshold_counts(prob, label)
        for i, thr in enumerate(sm.SWEEP_THRESHOLDS):
            tp, fp, fn, _ = sm.window_confusion(prob, label, thr)
            self.assertEqual(tuple(counts[i]), (tp, fp, fn))
        summary = sm.sweep_summary(counts + counts)          # 序列之间直接相加
        ious = [r["iou"] for r in summary["rows"]]
        self.assertAlmostEqual(summary["best_iou"], max(ious))
        self.assertEqual(summary["rows"][0]["tp"], 2 * int(counts[0][0]))
        empty = sm.sweep_summary(np.zeros((len(sm.SWEEP_THRESHOLDS), 3), dtype=np.int64))
        self.assertIsNone(empty["best_iou"])


if __name__ == "__main__":
    unittest.main()
