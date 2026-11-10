"""DeCo V13 VoW unit tests (11-10; CPU, synthetic data, torch 1.9 compatible).

    1. Quantile head: q_lo <= q_50 <= q_hi for any network output
    2. Stop rule: stop as target / background only when the predicted range stays on the current side of tau; at the
       deadline (range 0) every event stops with its sign
    3. Split conformal (CQR): on exchangeable synthetic data the widened range covers each side at about 1 - alpha;
       the shifts per side follow their own risk levels
    4. Temporal BIR: the state stays in [0, L theta]; zero input gives zero output; it runs on any number of windows
    5. Trace features: finite, the right shape, windows without neighbours get the defaults
    6. Recording split (fit / calibration): both parts non-empty, disjoint, covering the rows
    7. fit_vow.Calibrated: per-side shifts follow their own risk levels
fit_vow end to end needs data files: tools/deco/smoke_v13.sh covers it.
"""
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "deco"))
import fit_vow as FV  # noqa: E402
import vow as VW  # noqa: E402

split_recordings = FV.split_recordings


class QuantileTests(unittest.TestCase):
    def test_ordered(self):
        torch.manual_seed(0)
        o = torch.randn(1000, 3) * 10.0
        lo, md, hi = VW.quantiles(o)
        self.assertTrue(bool((lo <= md).all()) and bool((md <= hi).all()))

    def test_heads_ordered(self):
        torch.manual_seed(1)
        gate = VW.VoWGate()
        lo, md, hi = gate(torch.randn(64, 11, VW.N_TRACE), torch.randn(64), torch.rand(64))
        self.assertTrue(bool((lo <= md).all()) and bool((md <= hi).all()))
        commit = VW.VoWCommit(8)
        lo, md, hi = commit(torch.randn(64, 8))
        self.assertTrue(bool((lo <= md).all()) and bool((md <= hi).all()))
        gz = VW.VoWGate(traces=False)
        lo, md, hi = gz(torch.randn(64, 11, VW.N_TRACE), torch.randn(64), torch.rand(64))
        self.assertTrue(bool((lo <= hi).all()))


class StopRuleTests(unittest.TestCase):
    def test_sides(self):
        tau = -1.0
        U = np.array([2.0, 2.0, -3.0, -3.0])
        q_lo = np.array([-1.0, -4.0, -1.0, -1.0])
        q_hi = np.array([1.0, 1.0, 1.0, 3.0])
        stop = VW.stop_now(U, tau, q_lo, q_hi)
        # target side: stays above tau with -1, crosses with -4; background side: stays below with +1, crosses with +3
        self.assertEqual(stop.tolist(), [True, False, True, False])

    def test_deadline(self):
        U = np.array([-5.0, -1.2, -0.8, 4.0])
        stop = VW.stop_now(U, -1.0, np.zeros(4), np.zeros(4))
        self.assertTrue(bool(stop.all()))


class ConformalTests(unittest.TestCase):
    def test_coverage(self):
        rng = np.random.RandomState(0)
        n = 20000
        y_cal, y_test = rng.standard_t(3, n), rng.standard_t(3, n)
        lo, hi = np.full(n, -1.0), np.full(n, 1.0)                    # a far too narrow range
        for a in (0.01, 0.05):
            s_lo, s_hi = VW.cqr_shift(lo, hi, y_cal, a)
            below = float((y_test < lo - s_lo).mean())
            above = float((y_test > hi + s_hi).mean())
            self.assertLess(below, a * 1.6)
            self.assertLess(above, a * 1.6)
            self.assertGreater(below, a * 0.4)

    def test_tighter_risk_wider_side(self):
        rng = np.random.RandomState(1)
        y = rng.normal(size=5000)
        lo, hi = np.full(5000, -0.5), np.full(5000, 0.5)
        self.assertGreater(VW.cqr_shift(lo, hi, y, 0.001)[1], VW.cqr_shift(lo, hi, y, 0.01)[1])


class TemporalBIRTests(unittest.TestCase):
    def test_bounded_and_silent(self):
        torch.manual_seed(2)
        bir = VW.TemporalBIR(VW.N_TRACE, channels=8)
        out = bir(torch.randn(32, 11, VW.N_TRACE) * 50.0)
        top = (bir.levels * torch.exp(bir.log_theta)).detach()
        self.assertTrue(bool((out >= 0).all()) and bool((out <= top + 1e-5).all()))
        with torch.no_grad():
            bir.inp.weight.zero_()
            bir.inp.bias.zero_()
        self.assertEqual(float(bir(torch.randn(4, 5, VW.N_TRACE)).abs().max()), 0.0)
        self.assertEqual(tuple(bir(torch.randn(3, 1, VW.N_TRACE)).shape), (3, 1, 8))
        m = bir.memory_windows()
        self.assertTrue(np.all(m >= 1.0))


class TraceTests(unittest.TestCase):
    def test_features(self):
        rng = np.random.RandomState(3)
        n, n_off, K, past, k_near = 50, 21, 16, 10, 8
        tok = rng.normal(size=(n, n_off, K, 6)).astype(np.float16) * 5
        valid = rng.rand(n, n_off, K) > 0.3
        valid[0] = False                                               # an event without any neighbour
        nall, nevid = rng.rand(n, n_off).astype(np.float32) * 100, rng.rand(n, n_off).astype(np.float32) * 10
        f = VW.trace_features(tok, valid, nall, nevid, past, k_near)
        self.assertEqual(f.shape, (n, past + 1, VW.N_TRACE))
        self.assertTrue(np.isfinite(f).all())
        self.assertAlmostEqual(float(f[0, 0, 0]), float(np.log1p(VW.FAR)), places=4)
        self.assertEqual(float(f[0, 0, 2]), 0.0)


class CalibratedTests(unittest.TestCase):
    def test_sides(self):
        rng = np.random.RandomState(4)
        y = rng.normal(size=20000)
        c = FV.Calibrated(None, np.full(len(y), -0.5), np.full(len(y), 0.5), y)
        s_lo_a, s_hi_a = c.shifts(0.01, 0.01)
        s_lo_b, s_hi_b = c.shifts(0.001, 0.01)                         # stricter background side only
        self.assertAlmostEqual(s_lo_a, s_lo_b)
        self.assertGreater(s_hi_b, s_hi_a)


class SplitTests(unittest.TestCase):
    def test_split(self):
        rec = np.repeat(np.arange(12), 5)
        rows = rec < 6
        a, b = split_recordings(rec, rows, 2.0 / 3.0, seed=1)
        self.assertTrue(a.any() and b.any())
        self.assertFalse(bool((a & b).any()))
        self.assertTrue(bool(((a | b) == rows).all()))
        a2, b2 = split_recordings(rec, rec < 2, 2.0 / 3.0, seed=1)        # two recordings: one each
        self.assertTrue(a2.any() and b2.any())


if __name__ == "__main__":
    unittest.main()
