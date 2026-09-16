"""工具函数与附加指标的单元测试（只依赖 numpy）: python -m pytest tests/test_stream_common_metrics.py"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import stream_common as sc  # noqa: E402
from utils import stream_metrics as sm  # noqa: E402


class ChunkTests(unittest.TestCase):
    def test_every_window_once_for_all_offsets(self):
        for first in range(1, 17):
            chunks = sc.make_chunks(160, 16, first)
            covered = [k for a, b in chunks for k in range(a, b)]
            self.assertEqual(covered, list(range(160)))
            self.assertEqual(chunks[0], (0, first))
            self.assertTrue(all(0 < b - a <= 16 for a, b in chunks))

    def test_invalid_first_len(self):
        with self.assertRaises(ValueError):
            sc.make_chunks(160, 16, 0)
        with self.assertRaises(ValueError):
            sc.make_chunks(160, 16, 17)


class ScheduleAndSubsetTests(unittest.TestCase):
    def test_linear_lr_endpoints(self):
        self.assertEqual(sc.linear_epoch_lr(0, 50, 1e-3, 1e-4), 1e-3)
        self.assertEqual(sc.linear_epoch_lr(49, 50, 1e-3, 1e-4), 1e-4)
        self.assertAlmostEqual(sc.linear_epoch_lr(1, 50, 1e-3, 1e-4) - sc.linear_epoch_lr(2, 50, 1e-3, 1e-4),
                               0.0009 / 49, places=15)

    def test_subset_deterministic_sorted_unique(self):
        names = ["train_%03d.npz" % i for i in range(99)]
        a = sc.select_subset(names, 24, 2037)
        b = sc.select_subset(list(reversed(names)), 24, 2037)
        self.assertEqual(a, b)
        self.assertEqual(a, sorted(set(a)))
        self.assertEqual(len(a), 24)
        self.assertEqual(sc.select_subset(names[:5], 24, 1), names[:5])

    def test_summary_fields(self):
        hist = [{"epoch": e, "val_iou": v, "val_acc": v + 0.1} for e, v in
                enumerate([0.1, 0.5, 0.4, 0.45, 0.42, 0.44, 0.43])]
        s = sc.summarize_history(hist, 37, tail=5)
        self.assertEqual(s["best_val_iou"], 0.5)
        self.assertEqual(s["best_val_iou_epoch"], 1)
        self.assertAlmostEqual(s["final_mean_iou"], np.mean([0.4, 0.45, 0.42, 0.44, 0.43]))
        self.assertAlmostEqual(s["selection_bias_gap"], 0.5 - s["final_mean_iou"])


class GainMathTests(unittest.TestCase):
    def test_quantile_target_and_clamp_and_fallback(self):
        rng = np.random.RandomState(0)
        samples = [rng.uniform(0, 2.0, 5000),      # 正常通道
                   rng.uniform(0, 1e-3, 5000),     # 极小尺度 -> 增益被截断到 20
                   rng.uniform(0, 5.0, 10),        # 样本太少 -> 回退整层分位数
                   np.zeros(0)]                    # 无正电流 -> 回退整层分位数
        counts = [5000, 5000, 10, 0]
        gains, info = sc.gains_from_positive_samples(samples, counts, 1.0, 0.99, 1024, 0.05, 20.0)
        self.assertAlmostEqual(gains[0], 1.0 / np.quantile(samples[0], 0.99), places=9)
        self.assertEqual(gains[1], 20.0)
        self.assertEqual(info["fallback"], [False, False, True, True])
        self.assertAlmostEqual(gains[2], min(max(1.0 / info["q_layer"], 0.05), 20.0), places=9)

    def test_zero_scale_gives_unit_gain(self):
        gains, _ = sc.gains_from_positive_samples([np.zeros(0)], [0], 1.0, 0.99, 1024, 0.05, 20.0)
        self.assertEqual(gains.tolist(), [1.0])


class HealthTests(unittest.TestCase):
    def test_flags(self):
        w = sc.health_warnings(
            {"enc1": {"firing_rate": 1e-7, "silent_channel_frac": 0.8, "always_on_frac": 0.0,
                      "big_membrane_frac": 0.0}},
            {"enc1": {"near_min_frac": 0.95, "near_max_frac": 0.0}},
            {"readout": 0.0, "dec1": float("nan")})
        self.assertEqual(len(w), 5)
        self.assertEqual(sc.health_warnings(
            {"enc1": {"firing_rate": 0.01, "silent_channel_frac": 0.0, "always_on_frac": 0.0,
                      "big_membrane_frac": 0.0}}, {}, {"enc1": 0.3}), [])


class MetricTests(unittest.TestCase):
    def test_window_confusion(self):
        tp, fp, fn, npos = sm.window_confusion(np.array([0.95, 0.2, 0.91, 0.5]),
                                               np.array([1, 1, 0, 0]), 0.9)
        self.assertEqual((tp, fp, fn, npos), (1, 1, 1, 2))
        self.assertEqual(sm.window_confusion(np.zeros(0), np.zeros(0), 0.9), (0, 0, 0, 0))

    def test_rolling_iou_uses_summed_counts(self):
        tp = np.array([1, 0, 9, 0])
        fp = np.array([0, 0, 1, 0])
        fn = np.array([1, 0, 0, 0])
        r = sm.rolling_iou(tp, fp, fn, 2)
        self.assertAlmostEqual(r[0], 1 / 2)            # (1+0)/(2+0)
        self.assertAlmostEqual(r[1], 9 / 10)           # (0+9)/(0+10)，而不是单窗 IoU 的平均
        r2 = sm.rolling_iou(np.zeros(3), np.zeros(3), np.zeros(3), 1)
        self.assertTrue(np.all(np.isnan(r2)))

    def test_segment_iou(self):
        tp = np.arange(160) % 2
        fp = np.zeros(160, dtype=int)
        fn = 1 - tp
        seg = sm.segment_iou(tp, fp, fn, [[0, 15], [144, 159]])
        self.assertAlmostEqual(seg["0-15"], 0.5)
        self.assertAlmostEqual(seg["144-159"], 0.5)

    def test_first_detection_latency(self):
        # 目标 7: 第一个事件 t=12（第 0 窗），第 0/1 窗未检出，第 2 窗检出
        t = np.array([12, 30, 60, 110, 120, 5])
        label = np.array([1, 1, 1, 1, 1, 0])
        tid = np.array([7, 7, 7, 7, 7, 0], dtype=float)
        prob = np.array([0.1, 0.2, 0.3, 0.95, 0.1, 0.99])
        rec = sm.first_detection_latencies(t, label, tid, prob, 50, 0.9, 1e-4, 4.0)
        self.assertEqual(len(rec), 1)
        self.assertEqual(rec[0]["detect_window"], 2)
        self.assertAlmostEqual(rec[0]["latency_ms"], 3 * 50 + 4.0 - 12)
        missed = sm.first_detection_latencies(t, label, tid, np.zeros(6), 50, 0.9, 1e-4, 4.0)
        self.assertIsNone(missed[0]["latency_ms"])
        s = sm.summarize_latencies(rec + missed)
        self.assertEqual((s["n_targets"], s["n_detected"]), (2, 1))
        self.assertTrue(math.isclose(s["detection_rate"], 0.5))


if __name__ == "__main__":
    unittest.main()
