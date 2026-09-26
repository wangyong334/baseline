"""tools/persistence_gate.py 的单元测试：3x3 膨胀、因果的持续性计数、两个版本的公式，
以及在小型合成导出上端到端跑通（原地闪烁的误检被压掉，运动目标不受影响）。"""
import contextlib
import io
import math
import os
import shutil
import tempfile
import unittest

import numpy as np

from tools import persistence_gate as pg

WINDOWS = 40


def event_rows(rng, windows=WINDOWS):
    """一条序列：运动目标（每窗右移 3 像素，融合概率 0.95）+ 原地闪烁像素（初判低但 F 大，融合概率 0.98，
    只靠阈值分不开）+ 稀疏背景噪声。"""
    rows = []                                                # (x, y, t, label, m, F)
    for k in range(windows):
        t = 50 * k + 10
        for dx in (0, 1):
            for dy in (0, 1):
                rows.append((10 + 3 * k + dx, 20 + dy, t, 1, 1.0, 2.0))
        rows.append((100, 100, t, 0, -1.0, 5.0))
        rows.append((100, 100, t + 5, 0, -1.0, 5.0))
        for _ in range(3):
            rows.append((int(rng.integers(150, 340)), int(rng.integers(0, 250)), t + 20, 0, -4.0, 0.0))
    return np.asarray(rows, dtype=np.float64)


def write_split(directory, seed, n_seq=2):
    os.makedirs(directory)
    rng = np.random.default_rng(seed)
    for i in range(n_seq):
        r = event_rows(rng)
        n = r.shape[0]
        locs = np.stack([np.zeros(n), r[:, 0], r[:, 1], r[:, 2]], 1)
        np.savez(os.path.join(directory, "seq_%03d.npz" % i), locs=locs, labels=r[:, 3].astype(np.float32),
                 target_id=r[:, 3], logit_net=r[:, 4].astype(np.float32), evidence_d2=r[:, 5].astype(np.float32),
                 prob_fused_d2=(1.0 / (1.0 + np.exp(-(r[:, 4] + r[:, 5])))).astype(np.float32))


class PersistenceGateTests(unittest.TestCase):
    def test_dilate3(self):
        D = np.zeros((1, 5, 5), dtype=bool)
        D[0, 0, 0] = True
        D[0, 4, 2] = True
        out = pg.dilate3(D)
        self.assertEqual(int(out.sum()), 4 + 6)
        self.assertTrue(out[0, 1, 1] and out[0, 3, 3] and not out[0, 2, 2])

    def test_history_is_causal_and_counts_windows(self):
        """原地像素每窗都为正：第 k 窗事件的 h = sum_{j<k} rho^(k-1-j)；同一窗的第二个事件不增加计数；
        运动目标每窗移 3 像素，3x3 邻域在过去没有为正，h = 0。"""
        r = event_rows(np.random.default_rng(0), windows=6)
        seq = {"locs": np.stack([np.zeros(len(r)), r[:, 0], r[:, 1], r[:, 2]], 1), "m": r[:, 4], "F": r[:, 5]}
        h = pg.persistence(seq, 0.5, [20.0], 1.0, 50.0, 6)[20.0]
        rho = math.exp(-1.0 / 20.0)
        flicker = (r[:, 0] == 100) & (r[:, 1] == 100)
        k = (r[:, 2] // 50).astype(int)
        expect = np.array([sum(rho ** (kk - 1 - j) for j in range(kk)) for kk in k[flicker]])
        self.assertLess(float(np.abs(h[flicker] - expect).max()), 1e-5)
        self.assertEqual(float(h[flicker][0]), 0.0)                     # 第 0 窗没有历史
        target = r[:, 3] == 1
        self.assertTrue(np.all(h[target & (k == 0)] == 0))
        self.assertLess(float(h[target].max()), 1e-6)

    def test_variant_formulas(self):
        m, F, q = np.array([0.0, 0.0, 1.0]), np.array([2.0, -1.0, 3.0]), np.array([0.25, 0.25, 1.0])
        self.assertTrue(np.allclose(pg.scores(m, F, 1.0, "A", q), [0.5, -1.0, 4.0]))   # 只削减正分
        self.assertTrue(np.allclose(pg.scores(m, F, 1.0, "B", q, beta=4.0), [-1.0, -4.0, 4.0]))
        self.assertTrue(np.allclose(pg.scores(m, F, 1.0, "fused"), m + F))
        self.assertAlmostEqual(float(pg.gate(np.array([8.0]), 8.0)[0]), 0.5)

    def test_end_to_end_suppresses_static_flicker(self):
        tmp = tempfile.mkdtemp()
        try:
            write_split(os.path.join(tmp, "val"), 1)
            write_split(os.path.join(tmp, "test"), 2)
            with contextlib.redirect_stdout(io.StringIO()):
                report = pg.main(["--val-dump", os.path.join(tmp, "val"), "--test-dump", os.path.join(tmp, "test"),
                                  "--n-windows", str(WINDOWS), "--tau-h", "0.5", "--tau-w", "20", "--h0", "4",
                                  "--beta", "4", "--thresholds", "0.5", "0.7", "0.9", "--seqs", "seq_000", "--no-pd"])
        finally:
            shutil.rmtree(tmp)
        fused, a, b = report["fused@val"]["test"], report["A@val"]["test"], report["B@val"]["test"]
        self.assertGreater(a["iou"], fused["iou"])
        self.assertLess(a["fp"], fused["fp"])
        self.assertEqual(a["fn"], fused["fn"])                          # 运动目标一个不丢
        self.assertGreater(b["iou"], fused["iou"])
        self.assertIn("seq_000", report["A@val"]["sequences"])


if __name__ == "__main__":
    unittest.main()
