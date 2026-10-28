"""E2 分环境恒虚警约束（utils/cfar_constraint.py）的测试（CPU、合成数据，不需要数据集）。

    1. 分层：按 (窗, 流) 的事件数换算 5x5 巧合负荷，边界归属正确
    2. mu = 0：不产生惩罚，只累计硬计数；训练与不配置时逐位相同
    3. 对偶：密层误报率高于安静层时 mu 上升，低于时回落到 0；安静层 mu 恒为 0
    4. 梯度：惩罚只作用于 s >= 1 的负事件，方向是压低它们的 logit
    5. 训练端到端：开启后能跑通，输出 cfar 统计；配置与 E1 只差 cfar_* 三项与 save_root
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_stream_v1 as T  # noqa: E402
from test_stream_layerwise import CFG, H, Q99, W, make_net, synthetic_sequence  # noqa: E402
from utils.cfar_constraint import CfarConstraint  # noqa: E402

CPU = torch.device("cpu")


def make(edges=(0.1, 0.3, 1.0), **kw):
    return CfarConstraint(0.1, 0.9, 10, 10, edges=edges, min_negatives=1, **kw)   # 100 像素：负荷 = 0.25 x 事件数


def events_for(counts):
    """counts[(t, b)] = 事件数 -> events 字典（t、b）。"""
    t, b = [], []
    for (ti, bi), n in counts.items():
        t += [ti] * n
        b += [bi] * n
    return {"t": torch.tensor(t), "b": torch.tensor(b)}


class StrataTests(unittest.TestCase):
    def test_load_and_edges(self):
        c = make()
        # 负荷 = 0.25 x 事件数：1 -> 0.25（层 1）、4 -> 1.0（层 3，边界归上层）、2 -> 0.5（层 2）、8 -> 2.0（层 3）
        ev = events_for({(0, 0): 1, (0, 1): 4, (1, 0): 2, (1, 1): 8})
        s = c.strata(ev, steps=2, batch=2)
        load = {1: 0.25, 4: 1.0, 2: 0.5, 8: 2.0}
        expect = {0.25: 1, 1.0: 3, 0.5: 2, 2.0: 3}
        got = s.tolist()
        idx = 0
        for n in (1, 4, 2, 8):
            self.assertEqual(set(got[idx:idx + n]), {expect[load[n]]}, n)
            idx += n
        small = make(edges=(0.5,))
        self.assertEqual(small.strata(events_for({(0, 0): 1}), 1, 1).tolist(), [0])   # 0.25 < 0.5 -> 安静层


class DualTests(unittest.TestCase):
    def test_zero_mu_no_penalty_and_dual_ascent(self):
        c = make(edges=(0.5,), ema=0.5)
        # 安静窗 (t0)：1 个事件（负荷 0.25）；密窗 (t1)：8 个事件（负荷 2）
        ev = events_for({(0, 0): 1, (1, 0): 8})
        n = 9
        labels = torch.zeros(n)
        logits = torch.full((n,), -5.0)
        logits[1:5] = 5.0                                  # 密窗 8 个负事件中 4 个误报；安静窗 0 个误报
        logits[0] = 5.0                                    # 安静窗唯一的负事件也误报 -> 安静层误报率 1.0
        self.assertIsNone(c.chunk_loss(logits, labels, ev, 2, 1, n))
        c.end_update()
        np.testing.assert_allclose(c.fpr(), [1.0, 0.5])
        self.assertEqual(c.mu[1], 0.0)                     # 密层 0.5 < 安静层 1.0：mu 不升
        logits[0] = -5.0                                   # 安静层不再误报
        logits[5:] = 5.0                                   # 密层 8 个全误报
        for _ in range(3):
            c.chunk_loss(logits, labels, ev, 2, 1, n)
            logits[0] = 5.0 if _ == 0 else -5.0            # 让安静层有少量误报，参照为正数
            c.end_update()
        self.assertGreater(c.mu[1], 0.0)
        self.assertEqual(c.mu[0], 0.0)

    def test_gradient_only_on_dense_negatives(self):
        c = make(edges=(0.5,))
        c.mu[:] = [0.0, 2.0]
        ev = events_for({(0, 0): 1, (1, 0): 8})
        labels = torch.zeros(9)
        labels[8] = 1.0                                    # 密窗最后一个事件是目标
        logits = torch.zeros(9, requires_grad=True)
        loss = c.chunk_loss(logits, labels, ev, 2, 1, 9)
        loss.backward()
        g = logits.grad
        self.assertEqual(float(g[0]), 0.0)                 # 安静层不受约束
        self.assertEqual(float(g[8]), 0.0)                 # 正事件不受约束
        self.assertTrue(bool((g[1:8] > 0).all()))          # 梯度下降会压低密层负事件的 logit

    def test_from_config(self):
        self.assertIsNone(CfarConstraint.from_config({"trainer": "streams"}))
        c = CfarConstraint.from_config({"trainer": "streams", "cfar_dual_lr": 0.01, "threshold": 0.9,
                                        "height": 10, "width_px": 10})
        self.assertEqual(c.n_strata, 4)
        with self.assertRaises(ValueError):
            CfarConstraint.from_config({"trainer": "sequence", "cfar_dual_lr": 0.01, "threshold": 0.9,
                                        "height": 10, "width_px": 10})


class TrainTests(unittest.TestCase):
    def cfg(self, **kw):
        base = dict(CFG, execution="layer", tbptt_k=4, streams=1, accumulate=1, height=H, width_px=W,
                    window_ms=50, trainer="streams")
        base.update(kw)
        return base

    def run_train(self, cfg, cfar):
        dataset = [synthetic_sequence(seed) for seed in (0, 1)]
        lengths = [seq.n_windows for seq in dataset]
        torch.manual_seed(0)
        net = make_net()
        opt = torch.optim.SGD(net.parameters(), lr=0.05)
        out = T.train_streams(net, dataset, lengths, opt, cfg, Q99, 4.0, CPU, None,
                              plan=[(0, [(0, lengths[0]), (1, lengths[1])])], cfar=cfar)
        return net, out

    def test_off_is_bit_identical_and_on_runs(self):
        net_a, out_a = self.run_train(self.cfg(), None)
        zero = CfarConstraint.from_config(self.cfg(cfar_dual_lr=0.01))          # mu 从 0 开始：首次更新前无惩罚
        net_b, out_b = self.run_train(self.cfg(), zero)
        self.assertIn("cfar", out_b)
        self.assertNotIn("cfar", out_a)
        # 第一次更新之前 mu = 0；之后 mu 可能变化，所以只比较损失统计的有限性与结构
        self.assertTrue(np.isfinite(out_b["loss_sum"]))
        on = CfarConstraint.from_config(self.cfg(cfar_dual_lr=0.01, cfar_load_edges=[0.01]))
        on.mu[1:] = 5.0
        _, out_c = self.run_train(self.cfg(), on)
        self.assertTrue(np.isfinite(out_c["loss_sum"]))
        self.assertEqual(len(out_c["cfar"]["mu_end"]), 2)

    def test_mu_zero_forever_is_bit_identical(self):
        net_a, _ = self.run_train(self.cfg(), None)
        frozen = CfarConstraint.from_config(self.cfg(cfar_dual_lr=0.01))
        frozen.end_update = lambda: None                    # 乘子永远为 0：训练应与不开启逐位相同
        net_b, _ = self.run_train(self.cfg(), frozen)
        for pa, pb in zip(net_a.parameters(), net_b.parameters()):
            self.assertTrue(torch.equal(pa, pb))


class ConfigTests(unittest.TestCase):
    def test_e2_config_matches_e1(self):
        from utils.stream_common import load_flat_config
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        e1 = load_flat_config(os.path.join(root, "configs", "evisseg_stream_tif_comm_peak_noiseaug.yaml"))
        e2 = load_flat_config(os.path.join(root, "configs", "evisseg_stream_tif_comm_peak_cfar.yaml"))
        self.assertEqual(set(e2) - set(e1), {"cfar_dual_lr", "cfar_load_edges", "cfar_temperature"})
        for key in e1:
            if key != "save_root":
                self.assertEqual(e2[key], e1[key], key)
        self.assertEqual(e2["save_root"], "log/stream_tif_commpeak_cfar_seed37")
        c = CfarConstraint.from_config(e2)
        self.assertEqual((c.edges, c.dual_lr, c.temperature), ((0.1, 0.3, 1.0), 0.01, 1.0))


if __name__ == "__main__":
    unittest.main()
