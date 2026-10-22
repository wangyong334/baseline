"""tools/noise_profile.py（负事件类型画像）的单元测试：用已知类型的合成事件检验各项统计（CPU，不需要数据集）。

    1. 均匀泊松噪声：空间离散约 1、跨窗离散约 1、极性改变比约 1、时空支持度约 1、相邻窗没有整体位移
    2. 闪烁像素（每 10 ms 一次、极性交替）：间隔众数 10 ms、8~11 ms 占比高、极性改变比 > 1、少数像素占多数事件
    3. 匀速移动的竖线（每窗右移 2 像素）：显著的非零最佳位移、支持度 >> 1、跨窗离散 > 1
    4. 只统计负事件：正事件不影响负事件的统计，只出现在正事件支持度里
    5. 分块计算与一次算完相同；按录像自身长度取窗数；长录像只抽取部分相邻窗对
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import stream_windows as sw  # noqa: E402
from tools import noise_profile as NP  # noqa: E402

H, W, WIN, N = 20, 30, 50, 40


def make_seq(x, y, t, p, label):
    order = np.argsort(t, kind="stable")
    x, y, t, p, label = (np.asarray(a)[order] for a in (x, y, t, p, label))
    t = t.astype(np.int64)
    o, bounds = sw.split_windows(t, WIN, N)
    bins, local = sw.time_bin_and_local(t, WIN, 5)
    return sw.StreamSequence("synthetic.npz", x.astype(np.int64), y.astype(np.int64), t, p.astype(np.int8),
                             label.astype(np.float32), label.astype(np.float64), bins, local, o, bounds)


def poisson_noise(rng, n):
    return (rng.randint(0, W, n), rng.randint(0, H, n), rng.randint(0, WIN * N, n), rng.randint(0, 2, n),
            np.zeros(n))


def profile(seq):
    return NP.profile_sequence(seq, H, W, WIN, N)


class NoiseProfileTests(unittest.TestCase):
    def test_uniform_poisson(self):
        r = profile(make_seq(*poisson_noise(np.random.RandomState(0), 60000)))
        self.assertAlmostEqual(r["spatial"]["dispersion"], 1.0, delta=0.25)
        self.assertAlmostEqual(r["temporal"]["dispersion_median"], 1.0, delta=0.25)
        self.assertAlmostEqual(r["polarity_interval"]["alternation_ratio"], 1.0, delta=0.05)
        self.assertAlmostEqual(r["support_neg"], 1.0, delta=0.08)
        self.assertLess(r["motion"]["moved_frac"], 0.1)                  # 没有显著的整体位移
        self.assertLess(abs(r["motion"]["zero_corr"]), 0.1)
        self.assertLess(r["motion"]["best_corr"], 0.15)
        self.assertAlmostEqual(r["rate"]["per_window"], 60000 / N, delta=1)
        self.assertIsNone(r["support_pos"])

    def test_flicker(self):
        rng = np.random.RandomState(1)
        xs, ys, ts, ps = [], [], [], []
        for px, py in [(3, 4), (10, 12), (25, 7)]:
            times = np.arange(0, WIN * N, 10)
            xs.append(np.full(times.size, px))
            ys.append(np.full(times.size, py))
            ts.append(times)
            ps.append(np.arange(times.size) % 2)
        nx, ny, nt, npol, _ = poisson_noise(rng, 2000)
        x = np.concatenate(xs + [nx])
        y = np.concatenate(ys + [ny])
        t = np.concatenate(ts + [nt])
        p = np.concatenate(ps + [npol])
        r = profile(make_seq(x, y, t, p, np.zeros(t.size)))
        pi = r["polarity_interval"]
        self.assertEqual(pi["interval_mode_ms"], 10)
        self.assertGreater(pi["interval_8_11"], 0.25)                    # 597 个 10 ms 间隔 / 约 2000 个间隔
        self.assertGreater(pi["alternation_ratio"], 1.2)                 # 期望 (597 + 0.5*1400) / 1997 / 0.5 = 1.30
        self.assertGreater(r["spatial"]["top1"], 0.15)                   # 3 个像素约 600/2600 = 0.23；均匀噪声约 0.02

    def test_moving_line(self):
        rng = np.random.RandomState(2)
        xs, ys, ts = [], [], []
        for k in range(N):
            col = (2 + 2 * k) % W
            for _ in range(200):
                xs.append(col)
                ys.append(rng.randint(0, H))
                ts.append(k * WIN + rng.randint(0, WIN))
        x, y, t = np.array(xs), np.array(ys), np.array(ts)
        r = profile(make_seq(x, y, t, rng.randint(0, 2, t.size), np.zeros(t.size)))
        self.assertGreater(r["motion"]["moved_frac"], 0.8)
        self.assertGreater(r["motion"]["shift_gain"], 0.3)
        self.assertGreater(r["support_neg"], 3.0)
        self.assertGreater(r["temporal"]["dispersion_median"] or 0.0, 1.5)

    def test_only_negatives(self):
        rng = np.random.RandomState(3)
        x, y, t, p, label = poisson_noise(rng, 20000)
        base = profile(make_seq(x, y, t, p, label))
        # 加一个移动的目标（正事件，3x3 的团、每窗右移 1 像素）：负事件统计不变，正事件支持度 >> 1
        tt = rng.randint(0, WIN * N, 3000)
        tx = (tt // WIN) % (W - 3) + rng.randint(0, 3, 3000)
        ty = 8 + rng.randint(0, 3, 3000)
        r = profile(make_seq(np.concatenate([x, tx]), np.concatenate([y, ty]), np.concatenate([t, tt]),
                             np.concatenate([p, np.ones(3000)]), np.concatenate([label, np.ones(3000)])))
        self.assertEqual(r["neg_events"], base["neg_events"])
        self.assertAlmostEqual(r["spatial"]["dispersion"], base["spatial"]["dispersion"], places=12)
        self.assertAlmostEqual(r["support_neg"], base["support_neg"], places=12)
        self.assertGreater(r["support_pos"], 5.0)
        self.assertAlmostEqual(r["pos_per_window"], 3000 / N, places=9)

    def test_blocks_and_recording_length(self):
        # 分块计算与一次算完逐项相同；n_windows=None 时用录像自身的窗数（EV-Flying 长录像）
        rng = np.random.RandomState(4)
        x, y, t, p, label = poisson_noise(rng, 15000)
        seq = make_seq(x, y, t, p, label)
        whole = NP.profile_sequence(seq, H, W, WIN, N, max_pairs=1000, block=1000)
        blocks = NP.profile_sequence(seq, H, W, WIN, None, max_pairs=1000, block=7)
        self.assertEqual(blocks["windows"], N)
        for key in ("rate", "spatial", "temporal", "motion"):
            for k, v in whole[key].items():
                if isinstance(v, float):
                    self.assertAlmostEqual(v, blocks[key][k], places=9, msg=(key, k))
                else:
                    self.assertEqual(v, blocks[key][k], (key, k))
        thin = NP.profile_sequence(seq, H, W, WIN, N, max_pairs=5)
        self.assertLessEqual(thin["motion"]["pairs"], 8)                        # 长录像只抽取部分相邻窗对


if __name__ == "__main__":
    unittest.main()
