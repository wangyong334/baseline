"""E1 噪声增强（dataset/noise_aug.py）的测试（CPU、合成序列，不需要数据集）。

    1. 注入：原事件原样保留在前面；新事件标签 0、target_id 0，坐标 / 时间在范围内；数量与极性符合泊松设定；切窗正确
    2. prob = 0：不加噪、不消耗随机数；训练与没有该键时逐位相同
    3. prob = 1：每条录像都加噪，速率在 [rate_min, rate_max] 内；训练能跑通、事件数变多
    4. 配置：E1 配置与 V5 定型配置只差 noise_aug_* 三项与 save_root
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_stream_v1 as T  # noqa: E402
from dataset.noise_aug import NoiseAugment, inject_background_noise, noise_aug_settings  # noqa: E402
from test_stream_layerwise import CFG, H, Q99, W, WINDOWS, make_net, synthetic_sequence  # noqa: E402

CPU = torch.device("cpu")
AUG_CFG = dict(height=H, width_px=W, window_ms=50, time_bins=5)


class InjectTests(unittest.TestCase):
    def test_injected_events(self):
        seq = synthetic_sequence(0)
        rng = np.random.RandomState(3)
        rate = 0.5                                          # 大速率，便于检验数量
        out = inject_background_noise(seq, rate, H, W, 50, WINDOWS, 5, rng)
        n0, n = seq.n_events, out.n_events
        for key in ("x", "y", "t", "p", "label", "target_id"):
            np.testing.assert_array_equal(getattr(out, key)[:n0], getattr(seq, key))
        added = n - n0
        expect = rate * H * W * WINDOWS
        self.assertLess(abs(added - expect), 5 * np.sqrt(expect))
        self.assertTrue(np.all(out.label[n0:] == 0) and np.all(out.target_id[n0:] == 0))
        self.assertTrue(np.all((out.x[n0:] >= 0) & (out.x[n0:] < W) & (out.y[n0:] >= 0) & (out.y[n0:] < H)))
        self.assertTrue(np.all((out.t[n0:] >= 0) & (out.t[n0:] < 50 * WINDOWS)))
        self.assertLess(abs(out.p[n0:].mean() - 0.5), 0.05)           # ON / OFF 各一半
        for k in range(WINDOWS):                                       # 切窗与时间戳一致
            idx = out.window_index(k)
            self.assertTrue(np.all(out.t[idx] // 50 == k))
        self.assertEqual(int(out.bounds[-1]), n)


class AugmentTests(unittest.TestCase):
    def test_prob_zero_is_identity(self):
        rng = np.random.RandomState(5)
        state = rng.get_state()[1].copy()
        aug = NoiseAugment(dict(AUG_CFG), rng)
        seq = synthetic_sequence(1)
        self.assertIs(aug(seq), seq)
        self.assertFalse(aug.enabled)
        np.testing.assert_array_equal(rng.get_state()[1], state)        # 没有消耗随机数

    def test_prob_one_rates_in_range(self):
        cfg = dict(AUG_CFG, noise_aug_prob=1.0, noise_aug_rate_min=0.01, noise_aug_rate_max=0.1)
        aug = NoiseAugment(cfg, np.random.RandomState(7))
        for seed in range(6):
            seq = synthetic_sequence(seed)
            self.assertGreaterEqual(aug(seq).n_events, seq.n_events)
        s = aug.summary()
        self.assertEqual((s["loaded"], s["noised"]), (6, 6))
        self.assertTrue(all(0.01 <= r <= 0.1 for r in aug.stats["rates"]))

    def test_settings_validation(self):
        self.assertEqual(noise_aug_settings({}), (0.0, 0.002, 0.04))
        with self.assertRaises(ValueError):
            noise_aug_settings({"noise_aug_prob": 1.5})
        with self.assertRaises(ValueError):
            noise_aug_settings({"noise_aug_rate_min": 0.1, "noise_aug_rate_max": 0.01})


class TrainTests(unittest.TestCase):
    def cfg(self, **kw):
        base = dict(CFG, execution="layer", tbptt_k=4, streams=1, accumulate=1, **AUG_CFG)
        base.update(kw)
        return base

    def run_train(self, cfg, rng):
        dataset = [synthetic_sequence(seed) for seed in (0, 1)]
        lengths = [seq.n_windows for seq in dataset]
        net = make_net()
        opt = torch.optim.SGD(net.parameters(), lr=0.05)
        out = T.train_streams(net, dataset, lengths, opt, cfg, Q99, 4.0, CPU, rng,
                              plan=[(0, [(0, lengths[0]), (1, lengths[1])])])
        return net, out

    def test_prob_zero_bit_identical(self):
        net_a, out_a = self.run_train(self.cfg(), np.random.RandomState(0))
        net_b, out_b = self.run_train(self.cfg(noise_aug_prob=0.0), np.random.RandomState(0))
        self.assertNotIn("noise_aug", out_b)
        self.assertEqual(out_a["events"], out_b["events"])
        for pa, pb in zip(net_a.parameters(), net_b.parameters()):
            self.assertTrue(torch.equal(pa, pb))

    def test_prob_one_trains_with_more_events(self):
        _, clean = self.run_train(self.cfg(), np.random.RandomState(0))
        _, noisy = self.run_train(self.cfg(noise_aug_prob=1.0, noise_aug_rate_min=0.05, noise_aug_rate_max=0.1),
                                  np.random.RandomState(0))
        self.assertEqual(noisy["noise_aug"]["noised"], 2)
        self.assertGreater(noisy["events"], clean["events"])
        self.assertTrue(np.isfinite(noisy["loss_sum"]))


class EvalNoiseTests(unittest.TestCase):
    def test_eval_noise_is_deterministic_and_off_by_default(self):
        cfg = dict(AUG_CFG)
        seq = synthetic_sequence(2)
        self.assertIs(T.eval_noise(seq, cfg, 0), seq)                      # 未配置：原样返回
        cfg.update(eval_noise_load=12.5, eval_noise_seed=3)                  # 噪声率 0.5（每像素每窗）
        a, b = T.eval_noise(seq, cfg, 4), T.eval_noise(seq, cfg, 4)
        np.testing.assert_array_equal(a.t, b.t)                              # 同一序列、同一种子：逐事件相同
        np.testing.assert_array_equal(a.x, b.x)
        c = T.eval_noise(seq, cfg, 5)                                        # 不同序列：不同噪声
        self.assertFalse(a.n_events == c.n_events and np.array_equal(a.x, c.x))
        added = a.n_events - seq.n_events
        expect = 0.5 * H * W * WINDOWS
        self.assertLess(abs(added - expect), 5 * np.sqrt(expect))
        self.assertTrue(np.all(a.label[seq.n_events:] == 0))


class ConfigTests(unittest.TestCase):
    def test_e1_config_matches_v5(self):
        from utils.stream_common import load_flat_config
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        v5 = load_flat_config(os.path.join(root, "configs", "evisseg_stream_tif_comm_peak.yaml"))
        e1 = load_flat_config(os.path.join(root, "configs", "evisseg_stream_tif_comm_peak_noiseaug.yaml"))
        extra = {"noise_aug_prob", "noise_aug_rate_min", "noise_aug_rate_max"}
        self.assertEqual(set(e1) - set(v5), extra)
        for key in v5:
            if key != "save_root":
                self.assertEqual(e1[key], v5[key], key)
        self.assertEqual(e1["save_root"], "log/stream_tif_commpeak_noiseaug_seed37")
        self.assertEqual(noise_aug_settings(e1), (0.5, 0.002, 0.04))


if __name__ == "__main__":
    unittest.main()
