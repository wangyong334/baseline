"""数据层单元测试（只依赖 numpy，本地即可运行）: python -m pytest tests/test_stream_windows.py"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import stream_windows as sw  # noqa: E402

H, W, PH, PW = 260, 346, 264, 352
WIN, NWIN, BINS = 50, 160, 5


def make_npz(path, x, y, t, p, label, tid, float_loc=False):
    """写一个与 EV-UAV 格式相同的临时 NPZ（ev_loc 与 evs_norm）。"""
    n = len(t)
    ev_loc = np.stack([x, y, t], 1).astype(np.float64 if float_loc else np.int64)
    evs_norm = np.zeros((n, 6), dtype=np.float64)
    evs_norm[:, 0] = x / W
    evs_norm[:, 1] = y / H
    evs_norm[:, 2] = t / 8000.0
    evs_norm[:, 3] = p
    evs_norm[:, 4] = label
    evs_norm[:, 5] = tid
    np.savez(path, ev_loc=ev_loc, evs_norm=evs_norm, ev=np.zeros((n, 6)))


def random_events(n, seed=0):
    """生成随机事件，并强制包含时间边界 0/49/50/7999 与坐标边界。"""
    rng = np.random.RandomState(seed)
    x = rng.randint(0, W, n)
    y = rng.randint(0, H, n)
    t = rng.randint(0, 8000, n)
    t[:4] = [0, 49, 50, 7999]
    x[:2] = [0, W - 1]
    y[:2] = [0, H - 1]
    p = rng.randint(0, 2, n)
    label = (rng.rand(n) < 0.05).astype(np.float64)
    tid = np.where(label == 1, rng.randint(1, 4, n), 0)
    return x, y, t, p, label, tid


class WindowSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "train_000.npz")
        self.raw = random_events(5000)
        make_npz(self.path, *self.raw)
        self.seq = sw.load_npz_events(self.path, H, W, WIN, NWIN, BINS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_partition_covers_every_event_once(self):
        sw.check_window_partition(self.seq, WIN)
        joined = np.concatenate([self.seq.window_index(k) for k in range(NWIN)])
        self.assertTrue(np.array_equal(np.sort(joined), np.arange(self.seq.n_events)))

    def test_restored_fields_equal_originals(self):
        x, y, t, p, label, tid = self.raw
        parts = [self.seq.window_index(k) for k in range(NWIN)]
        for field, original in (("x", x), ("y", y), ("t", t), ("p", p), ("label", label),
                                ("target_id", tid)):
            values = getattr(self.seq, field)
            restored = sw.refill_by_index(self.seq.n_events, parts,
                                          [values[i] for i in parts], dtype=np.float64)
            self.assertTrue(np.array_equal(restored, original.astype(np.float64)), field)

    def test_time_bin_and_local_boundaries(self):
        t = np.array([0, 49, 50, 7999], dtype=np.int64)
        bins, local = sw.time_bin_and_local(t, WIN, BINS)
        self.assertEqual(bins.tolist(), [0, 4, 0, 4])
        np.testing.assert_allclose(local, [0.0, 0.98, 0.0, 0.98], atol=1e-6)
        order, bounds = sw.split_windows(t, WIN, NWIN)
        self.assertEqual(bounds[1] - bounds[0], 2)          # t=0, 49 在第 0 窗
        self.assertEqual(bounds[2] - bounds[1], 1)          # t=50 在第 1 窗
        self.assertEqual(bounds[160] - bounds[159], 1)      # t=7999 在第 159 窗

    def test_empty_window_exists(self):
        t = np.array([10, 7990], dtype=np.int64)
        _, bounds = sw.split_windows(t, WIN, NWIN)
        self.assertEqual(len(bounds), NWIN + 1)
        self.assertEqual(int(np.sum(np.diff(bounds) == 0)), NWIN - 2)

    def test_float_integer_ev_loc_is_accepted(self):
        path = os.path.join(self.tmp.name, "float.npz")
        make_npz(path, *self.raw, float_loc=True)
        seq = sw.load_npz_events(path, H, W, WIN, NWIN, BINS)
        self.assertTrue(np.array_equal(seq.t, self.raw[2]))


class ValidationTests(unittest.TestCase):
    def check_raises(self, **overrides):
        x, y, t, p, label, tid = random_events(100)
        values = dict(x=x, y=y, t=t, p=p, label=label)
        for key, (index, bad) in overrides.items():
            values[key] = values[key].astype(np.float64).copy()
            values[key][index] = bad
        with self.assertRaises(ValueError):
            sw.validate_events(values["x"], values["y"], values["t"], values["p"],
                               values["label"], H, W, 8000)

    def test_time_out_of_range(self):
        self.check_raises(t=(5, 8000))

    def test_negative_polarity(self):
        self.check_raises(p=(5, -1))

    def test_x_out_of_range(self):
        self.check_raises(x=(5, W))

    def test_non_integer_ev_loc(self):
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, "bad.npz")
        x, y, t, p, label, tid = random_events(10)
        ev_loc = np.stack([x, y, t], 1).astype(np.float64)
        ev_loc[0, 2] += 0.5
        np.savez(path, ev_loc=ev_loc, evs_norm=np.zeros((10, 6)))
        with self.assertRaises(ValueError):
            sw.load_npz_events(path, H, W, WIN, NWIN, BINS)
        tmp.cleanup()


class InputTensorTests(unittest.TestCase):
    def test_counts_and_padding_keep_coordinates(self):
        # 两个正事件落在右下角原始像素 (345,259)，一个负事件在 (0,0) 的 bin 4
        x = np.array([W - 1, W - 1, 0])
        y = np.array([H - 1, H - 1, 0])
        p = np.array([1, 1, 0], dtype=np.int8)
        bins = np.array([0, 0, 4])
        c = sw.count_channels(x, y, p, bins, BINS, PH, PW)
        self.assertEqual(c.shape, (12, PH, PW))
        self.assertEqual(c[0, H - 1, W - 1], 2)     # 正极性整窗计数
        self.assertEqual(c[2, H - 1, W - 1], 2)     # bin0 正极性
        self.assertEqual(c[1, 0, 0], 1)             # 负极性整窗计数
        self.assertEqual(c[2 + 2 * 4 + 1, 0, 0], 1)  # bin4 负极性
        self.assertEqual(int(c.sum()), 6)           # 每个事件写入整窗通道 + bin 通道各一次
        self.assertEqual(int(c[:, H:, :].sum() + c[:, :, W:].sum()), 0)  # 补边区域全零

    def test_empty_window_is_all_zero(self):
        e = np.zeros(0, dtype=np.int64)
        c = sw.count_channels(e, e, e.astype(np.int8), e, BINS, PH, PW)
        x = sw.normalize_counts(c, np.ones(12), 3.0)
        self.assertEqual(float(np.abs(x).sum()), 0.0)

    def test_normalization_scale_and_clip(self):
        c = np.zeros((12, 2, 2), dtype=np.int64)
        c[0, 0, 0] = 1
        c[0, 1, 1] = 1000
        q = np.full(12, np.log1p(1.0))
        x = sw.normalize_counts(c, q, 3.0)
        self.assertAlmostEqual(float(x[0, 0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(x[0, 1, 1]), 3.0, places=6)
        self.assertEqual(float(x[0, 0, 1]), 0.0)


class StatisticsTests(unittest.TestCase):
    def test_histogram_quantile_matches_brute_force(self):
        rng = np.random.RandomState(1)
        hist = sw.new_count_histogram(12)
        all_values = [[] for _ in range(12)]
        for _ in range(20):
            counts = rng.poisson(0.05, size=(12, 30, 40)) * rng.randint(1, 6, size=(12, 30, 40))
            sw.update_count_histogram(hist, counts)
            for ch in range(12):
                nz = counts[ch][counts[ch] > 0]
                all_values[ch].extend(nz.tolist())
        for ch in range(12):
            values = np.sort(np.log1p(np.array(all_values[ch], dtype=np.float64)))
            q = 0.99
            expected = values[int(np.ceil(q * values.size)) - 1]   # 逆 CDF 定义
            self.assertAlmostEqual(sw.quantile_from_histogram(hist[ch], q), expected, places=9)

    def test_pos_weight_rule(self):
        self.assertEqual(sw.pos_weight_from_counts(10, 1000, 30.0), 30.0)
        self.assertEqual(sw.pos_weight_from_counts(10, 150, 30.0), 15.0)
        with self.assertRaises(ValueError):
            sw.pos_weight_from_counts(0, 10, 30.0)


if __name__ == "__main__":
    unittest.main()
