"""模块二 v2：学出来的杂波强度场（model/clutter_field.py）的测试（CPU、合成数据）。

    1. 初值：各候选估计等权；没有历史时等于参照率按权重的份额（不变接口取 max(背景率, 参照率)，起步时相当于恒等）
    2. 因果：第 t 窗的估计与第 t 窗及之后的计数无关
    3. 片段切分不变；新录像开始（keep = 0）时只清空该流
    4. 汇聚按传感器掩码归一化：补零区不把边缘的背景率拉低
    5. 学习：泊松负对数似然训练后，在恒定泊松背景上学到真实的率，背景率突变后跟得上（时间常数是学出来的）
    6. 事件计数图：通道 0 = ON、1 = OFF，按流、按窗放置，可按事件加权（只数非目标事件）
"""
import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.clutter_field import ClutterField, event_count_maps  # noqa: E402

H, W = 16, 20
REF = (0.002, 0.0001)


def poisson_counts(T, B, rate_on, rate_off, seed=0, h=H, w=W):
    g = torch.Generator().manual_seed(seed)
    on = torch.poisson(torch.full((T, B, 1, h, w), float(rate_on)), generator=g)
    off = torch.poisson(torch.full((T, B, 1, h, w), float(rate_off)), generator=g)
    return torch.cat([on, off], 2)


class FieldTests(unittest.TestCase):
    def test_init_is_a_convex_combination(self):
        field = ClutterField(H, W, ref_rate=REF)
        log_rate, _ = field.forward_chunk(poisson_counts(5, 2, 0.3, 0.1))
        J = field.J
        first = torch.exp(log_rate[0])                                   # no history: only the reference has weight
        self.assertTrue(torch.allclose(first[:, 0], torch.full_like(first[:, 0], REF[0] / J), rtol=1e-4))
        later = torch.exp(log_rate[4])
        self.assertTrue(bool((later >= REF[1] / J - 1e-9).all()))
        self.assertTrue(bool((later[:, 0] <= 4.0).all()))                 # bounded by the observed rates

    def _randomized(self, seed=0):
        torch.manual_seed(seed)
        field = ClutterField(H, W, ref_rate=REF)
        with torch.no_grad():
            for p in field.mix.parameters():
                p.normal_(0.0, 0.3)
        return field

    def test_causal(self):
        field = self._randomized()
        counts = poisson_counts(8, 1, 0.3, 0.05)
        changed = counts.clone()
        changed[5:] = changed[5:] * 3.0 + 1.0
        a, _ = field.forward_chunk(counts)
        b, _ = field.forward_chunk(changed)
        self.assertTrue(torch.equal(a[:6], b[:6]))                    # windows 0..5 only see windows < their index
        self.assertFalse(torch.equal(a[6:], b[6:]))

    def test_chunk_invariance_and_reset(self):
        field = self._randomized(1)
        counts = poisson_counts(8, 2, 0.2, 0.05, seed=3)
        keep = torch.ones(8, 2)
        keep[5, 1] = 0.0
        whole, _ = field.forward_chunk(counts, None, keep)
        first, st = field.forward_chunk(counts[:4], None, keep[:4])
        second, _ = field.forward_chunk(counts[4:], field.detach_state(st), keep[4:])
        self.assertTrue(torch.allclose(whole, torch.cat([first, second]), atol=1e-6))
        fresh, _ = field.forward_chunk(counts[5:, 1:], None)          # stream 1 restarts at window 5
        self.assertTrue(torch.allclose(whole[5:, 1:], fresh, atol=1e-6))

    def test_mask_normalized_pooling(self):
        field = ClutterField(H, W, sensor_hw=(12, 15), ref_rate=REF, scales=(4,))
        x = torch.zeros(1, 1, H, W)
        x[..., :12, :15] = 0.5                                        # uniform rate on the sensor, nothing on padding
        pooled = field._pool(x, 4)
        self.assertTrue(torch.allclose(pooled[..., :12, :15], torch.full((1, 1, 12, 15), 0.5), atol=1e-6))

    def test_learns_rate_and_tracks_a_step(self):
        torch.manual_seed(0)
        field = ClutterField(H, W, ref_rate=REF, taus=(2.0, 16.0), scales=(1, 4))
        opt = torch.optim.Adam(field.parameters(), lr=0.05)
        first = last = None
        g = torch.Generator().manual_seed(1)
        for step in range(200):
            r0, r1 = (0.05, 0.2) if step % 2 == 0 else (0.2, 0.05)     # recordings with a background step
            cut = int(torch.randint(8, 20, (1,), generator=g))
            counts = torch.cat([poisson_counts(cut, 2, r0, r0 / 4.0, seed=2 * step),
                                poisson_counts(32 - cut, 2, r1, r1 / 4.0, seed=2 * step + 1)])
            log_rate, _ = field.forward_chunk(counts)
            loss = field.nll(log_rate[2:], counts[2:])                 # after a short warm-up
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = float(loss) if first is None else first
            last = float(loss)
        self.assertLess(last, first)
        with torch.no_grad():
            quiet, noisy = poisson_counts(20, 1, 0.05, 0.0125, seed=999), poisson_counts(20, 1, 0.2, 0.05, seed=998)
            step_counts = torch.cat([quiet, noisy])                     # background jumps x4 at window 20
            log_rate, _ = field.forward_chunk(step_counts)
            before = float(torch.exp(log_rate[15:20, 0, 0]).mean())
            after = float(torch.exp(log_rate[32:40, 0, 0]).mean())
        self.assertLess(abs(before - 0.05) / 0.05, 0.35)
        self.assertLess(abs(after - 0.2) / 0.2, 0.35)

    def test_event_count_maps(self):
        events = {"t": torch.tensor([0, 0, 1, 1]), "b": torch.tensor([0, 1, 1, 1]), "y": torch.tensor([2, 3, 3, 3]),
                  "x": torch.tensor([4, 5, 5, 5]), "p": torch.tensor([1.0, 0.0, 1.0, 1.0])}
        c = event_count_maps(events, 2, 2, H, W)
        self.assertEqual(float(c.sum()), 4.0)
        self.assertEqual(float(c[0, 0, 0, 2, 4]), 1.0)                 # ON event, window 0, stream 0
        self.assertEqual(float(c[0, 1, 1, 3, 5]), 1.0)                 # OFF event, window 0, stream 1
        self.assertEqual(float(c[1, 1, 0, 3, 5]), 2.0)                 # two ON events, window 1, stream 1
        w = event_count_maps(events, 2, 2, H, W, weight=torch.tensor([1.0, 0.0, 1.0, 0.0]))
        self.assertEqual(float(w.sum()), 2.0)


if __name__ == "__main__":
    unittest.main()
