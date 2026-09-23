"""膜电位诊断工具（tools/diagnose_membrane.py）的单元测试。

全部在 CPU 上跑，不需要数据集，合成序列复用 test_stream_v2_evidence 里的构造。
核心要验证的是：**新的正负分开统计是旧统计量的细分**，即
    frac_big_positive + frac_big_negative == LayerMonitor 的 big_membrane_frac
    firing_rate 与 LayerMonitor 一致
否则诊断结论无法和已有评估日志对上号。
"""
import unittest

import numpy as np
import torch

from model.evidence_neuron import DriftCUSUM, velocity_grid
from model.evspsegnet_stream import LAYER_NAMES
from tests.test_stream_v2_evidence import CFG, H, W, WINDOWS, make_frontend, make_model, synthetic_sequence
from tools.diagnose_membrane import (SAMPLE_PER_CHUNK, MembraneProbe, Reservoir, backbone_threshold,
                                     merge_polarity, polarity_stats, window_occupancy)
from train_stream_v1 import LayerMonitor
from train_stream_v2 import run_sequence


class ReservoirTests(unittest.TestCase):
    def test_quantiles_close_to_truth(self):
        """水库抽样得到的分位数应接近真实分位数（正态分布，容差放宽到 0.05）。"""
        res = Reservoir(0)
        torch.manual_seed(1)
        for _ in range(20):
            res.add(torch.randn(50000))
        summary = res.summary()
        self.assertGreater(summary["n_samples"], 1000)
        self.assertLess(abs(summary["quantiles"]["0.5"]), 0.05)
        self.assertLess(abs(summary["quantiles"]["0.95"] - 1.645), 0.08)
        self.assertLess(abs(summary["quantiles"]["0.05"] + 1.645), 0.08)

    def test_scale_divides_values(self):
        """summary(scale) 把样本折算成以 scale 为单位。"""
        res = Reservoir(0)
        res.add(torch.full((1000,), 4.0))
        self.assertAlmostEqual(res.summary(2.0)["quantiles"]["0.5"], 2.0, places=6)

    def test_empty_reservoir_returns_none(self):
        self.assertIsNone(Reservoir(0).summary())

    def test_generator_matches_tensor_device(self):
        """随机数生成器必须与被采样张量同设备，且按设备缓存、只播种一次。

        回归 2026-09-23 服务器上的 RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'——
        原来所有水库共用一个 CPU 生成器，张量在 GPU 上时 torch.randint 直接报错。
        本地没有 GPU，只能验证"生成器是按张量的设备建立的"这个不变量。
        """
        res = Reservoir(0)
        x = torch.randn(SAMPLE_PER_CHUNK * 3)          # 必须超过每片段上限才会走到抽样分支
        res.add(x)
        self.assertEqual(list(res.generators), [str(x.device)])
        self.assertIs(res.generator(x.device), res.generators[str(x.device)])
        self.assertEqual(res.generator(x.device).device.type, x.device.type)

    def test_same_seed_gives_same_samples(self):
        """同一个种子两次抽样结果相同（按设备缓存不能破坏可复现性）。"""
        x = torch.randn(SAMPLE_PER_CHUNK * 3)
        a, b = Reservoir(7), Reservoir(7)
        a.add(x)
        b.add(x)
        self.assertEqual(a.summary()["quantiles"], b.summary()["quantiles"])


class WindowOccupancyTests(unittest.TestCase):
    def test_matches_brute_force(self):
        """有目标事件的窗数与直接按事件时间统计一致。"""
        seq = synthetic_sequence(seed=1)
        occ = window_occupancy(seq, CFG["window_ms"])
        pos = (seq.label == 1) & (seq.target_id != 0)
        expected = np.unique(seq.t[pos] // CFG["window_ms"])
        expected = expected[(expected >= 0) & (expected < seq.n_windows)]
        self.assertEqual(occ["n_windows"], WINDOWS)
        self.assertEqual(occ["target_windows"], int(expected.size))
        self.assertEqual(occ["background_windows"], WINDOWS - int(expected.size))
        self.assertEqual(occ["first_target_window"], int(expected.min()))
        self.assertEqual(occ["last_target_window"], int(expected.max()))

    def test_sequence_without_targets(self):
        """全是背景的序列：纯背景窗 = 总窗数，首末目标窗为 None。"""
        seq = synthetic_sequence(seed=2)
        seq.label[:] = 0.0
        occ = window_occupancy(seq, CFG["window_ms"])
        self.assertEqual(occ["background_windows"], WINDOWS)
        self.assertIsNone(occ["first_target_window"])


class PolarityStatsTests(unittest.TestCase):
    """极性统计要和暴力计数一致——09-23 曾把像素占用率误当成事件占比，这组测试把两者分开钉住。"""

    def setUp(self):
        self.seq = synthetic_sequence(seed=1)
        self.stats = polarity_stats(self.seq, CFG["window_ms"], CFG["pad_height"], CFG["pad_width"])

    def test_event_counts_match_brute_force(self):
        on = int(np.count_nonzero(self.seq.p != 0))
        off = int(np.count_nonzero(self.seq.p == 0))
        self.assertEqual(self.stats["on"]["events"], on)
        self.assertEqual(self.stats["off"]["events"], off)
        self.assertEqual(self.stats["n_events"], on + off)
        self.assertAlmostEqual(self.stats["on"]["event_frac"] + self.stats["off"]["event_frac"], 1.0, places=12)

    def test_lit_pixels_never_exceed_events(self):
        """点亮的(像素,窗)数不可能多于事件数；每点亮像素事件数 >= 1。"""
        for name in ("on", "off"):
            d = self.stats[name]
            self.assertLessEqual(d["lit_pixel_windows"], d["events"])
            self.assertGreaterEqual(d["events_per_lit_pixel"], 1.0)

    def test_occupancy_is_not_event_fraction(self):
        """像素占用率与事件占比是两个量，不能互相代替（这条就是那次误读的成因）。"""
        for name in ("on", "off"):
            d = self.stats[name]
            self.assertLessEqual(d["pixel_occupancy"], 1.0)
            self.assertNotAlmostEqual(d["pixel_occupancy"], d["event_frac"], places=3)

    def test_merge_adds_up(self):
        merged = merge_polarity([self.stats, self.stats])
        self.assertEqual(merged["n_events"], 2 * self.stats["n_events"])
        self.assertEqual(merged["on"]["events"], 2 * self.stats["on"]["events"])
        self.assertAlmostEqual(merged["on"]["event_frac"], self.stats["on"]["event_frac"], places=12)
        self.assertAlmostEqual(merged["on"]["events_per_lit_pixel"],
                               self.stats["on"]["events_per_lit_pixel"], places=12)

    def test_single_polarity_sequence(self):
        """全是同一极性时，另一侧全为 0 且不除零。"""
        seq = synthetic_sequence(seed=3)
        seq.p[:] = 1
        stats = polarity_stats(seq, CFG["window_ms"], CFG["pad_height"], CFG["pad_width"])
        self.assertEqual(stats["off"]["events"], 0)
        self.assertEqual(stats["off"]["events_per_lit_pixel"], 0.0)
        self.assertAlmostEqual(stats["on"]["event_frac"], 1.0, places=12)


class MembraneProbeTests(unittest.TestCase):
    def setUp(self):
        self.seq = synthetic_sequence(seed=1)
        self.frontend = make_frontend()
        self.model = make_model(self.frontend)
        self.model.eval()
        self.cusum = DriftCUSUM(velocity_grid([0.0]), footprint=3, track_decay=0.8)
        self.v_th = backbone_threshold(self.model)

    def _run(self, probe, monitor=None):
        with torch.no_grad():
            run_sequence(self.model, self.frontend, self.cusum, self.seq, CFG, torch.device("cpu"), "carry",
                         monitor=probe if monitor is None else monitor, with_cusum=False,
                         feature_probe=probe.features)

    def test_split_statistics_refine_layer_monitor(self):
        """正负两侧之和 == LayerMonitor 的 big_membrane_frac；发放率也一致。"""
        probe = MembraneProbe(self.v_th, WINDOWS, self.frontend.feature_names())
        probe.begin_sequence()
        self._run(probe)
        monitor = LayerMonitor(self.v_th)
        probe2 = MembraneProbe(self.v_th, WINDOWS, self.frontend.feature_names())
        probe2.begin_sequence()
        self._run(probe2, monitor=monitor)
        old, new = monitor.summary(), probe.summary()["layers"]
        self.assertEqual(sorted(new), sorted(old))
        for name in old:
            total = new[name]["frac_big_positive"] + new[name]["frac_big_negative"]
            self.assertAlmostEqual(total, old[name]["big_membrane_frac"], places=9, msg=name)
            self.assertAlmostEqual(new[name]["firing_rate"], old[name]["firing_rate"], places=9, msg=name)

    def test_backbone_threshold_reads_the_neurons(self):
        """阈值必须取自神经元本身，而不是构造时记下的配置值（两者可能不同）。"""
        self.assertAlmostEqual(self.v_th, float(self.model.blocks()[0].neuron.v_threshold), places=12)
        self.model.blocks()[0].neuron.v_threshold = 0.11
        with self.assertRaises(SystemExit):
            backbone_threshold(self.model)

    def test_above_threshold_matches_firing_rate(self):
        """LIF 里本窗发放等价于 U_pre >= v_th，两个统计量应当相等（阈值取错时这条会失败）。"""
        probe = MembraneProbe(self.v_th, WINDOWS, self.frontend.feature_names())
        probe.begin_sequence()
        self._run(probe)
        for name, d in probe.summary()["layers"].items():
            self.assertAlmostEqual(d["frac_above_threshold"], d["firing_rate"], places=9, msg=name)

    def test_report_shape_and_monotone_quantiles(self):
        """七层齐全、特征通道名对齐、分位数单调、极值包住分位数、逐窗曲线长度正确。"""
        probe = MembraneProbe(self.v_th, WINDOWS, self.frontend.feature_names())
        probe.begin_sequence()
        self._run(probe)
        report = probe.summary()
        self.assertEqual(sorted(report["layers"]), sorted(LAYER_NAMES))
        self.assertEqual(sorted(report["features"]), sorted(self.frontend.feature_names()))
        self.assertIsNotNone(report["background_mu0"])
        for name, d in report["layers"].items():
            q = [d["u_pre_in_vth"]["quantiles"][str(x)] for x in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)]
            self.assertEqual(q, sorted(q), name)
            self.assertLessEqual(d["u_min_in_vth"], q[0] + 1e-9, name)
            self.assertGreaterEqual(d["u_max_in_vth"], q[-1] - 1e-9, name)
            self.assertEqual(len(d["firing_rate_by_window"]), WINDOWS, name)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in d["firing_rate_by_window"]), name)

    def test_window_curve_covers_every_window(self):
        """逐窗曲线应覆盖全部窗口（片段划分不应漏窗），且与整体发放率量级一致。"""
        probe = MembraneProbe(self.v_th, WINDOWS, self.frontend.feature_names())
        probe.begin_sequence()
        self._run(probe)
        self.assertTrue(np.all(probe.windows_seen > 0))
        for name, d in probe.summary()["layers"].items():
            mean_curve = float(np.mean(d["firing_rate_by_window"]))
            self.assertAlmostEqual(mean_curve, d["firing_rate"], places=6, msg=name)

    def test_two_sequences_restart_window_index(self):
        """begin_sequence 之后窗号从 0 重新开始，两条序列的曲线应当叠加而不是前后拼接。"""
        probe = MembraneProbe(self.v_th, WINDOWS, self.frontend.feature_names())
        for _ in range(2):
            probe.begin_sequence()
            self._run(probe)
        self.assertTrue(np.all(probe.windows_seen == 2))
        for name, d in probe.summary()["layers"].items():
            self.assertEqual(len(d["firing_rate_by_window"]), WINDOWS, name)

    def test_feature_channel_ranges_are_finite(self):
        """前端各通道的统计必须有限；count 组非负，dipole 组允许为负。"""
        probe = MembraneProbe(self.v_th, WINDOWS, self.frontend.feature_names())
        probe.begin_sequence()
        self._run(probe)
        features = probe.summary()["features"]
        for name, d in features.items():
            self.assertTrue(np.isfinite(d["min"]) and np.isfinite(d["max"]), name)
            if name.startswith("count") or name.startswith("ratio") or name.startswith("age"):
                self.assertGreaterEqual(d["min"], 0.0, name)


if __name__ == "__main__":
    unittest.main()
