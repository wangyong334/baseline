"""DeCo V11 tests (11-08; CPU, synthetic data, torch 1.9 compatible).

    1. Redundancy-corrected evidence: rho = 0 is the plain sum, rho = 1 the attributed mean, one window gives
       S / (1 + (m - 1) rho_w); duplicating an event adds nothing when rho = 1 and doubles the evidence when rho = 0;
       the effective count never exceeds the attributed mass
    2. V11 neurons: start exactly at z0; causal (U(d) ignores events of later windows) in both audit modes; events far
       outside every receptive field change nothing, near events do; ledger mode freezes old entries; softmax ablation
       runs; rho_t <= rho_w; no NaN when nothing has arrived
    3. Calibration slope diagnostic: about 1 on calibrated logits, about 1/2 when the logits are doubled
"""
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "deco"))
import sdn_v11 as S  # noqa: E402
import train_dnm as T  # noqa: E402


def inputs(B=6, n_off=7, K=4, D=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    T_ = n_off * K
    past = n_off - D - 1
    feat = torch.randn(B, T_, 6, generator=g)
    ok = torch.rand(B, T_, generator=g) > 0.2
    dw = (torch.arange(n_off) - past).repeat_interleave(K).float()[None].expand(B, T_).clone()
    valid = torch.ones(B, D + 1, dtype=torch.bool)
    z0 = torch.randn(B, generator=g) * 3.0
    z_in, p = z0 / 5.0, (torch.rand(B, generator=g) > 0.5).float()
    dist2 = torch.rand(B, T_, generator=g) * 400.0
    return feat, ok, dw, valid, z_in, z0, p, D, dist2


def make(n_off=7, K=4, nonzero=True, **kw):
    torch.manual_seed(1)
    net = S.SpikingDecisionNeuronsV11(n_off, K, sigma0=20.0, **kw)
    if nonzero:                                   # values switched on (they start at zero)
        with torch.no_grad():
            net.v_out.weight.normal_(0, 0.5)
            net.v_out.bias.normal_(0, 0.5)
    return net.eval()


class RedundancyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mw = torch.rand(5, 3, 4) * 3.0                        # [B, H, W]
        self.M = self.mw.sum(-1)
        self.Sv = torch.randn(5, 3)

    def test_limits(self):
        z, o = torch.zeros(3), torch.ones(3)
        lam0, _ = S.redundancy_evidence(self.Sv, self.M, self.mw, z, z)
        lam1, _ = S.redundancy_evidence(self.Sv, self.M, self.mw, o, o)
        torch.testing.assert_allclose(lam0, self.Sv, rtol=1e-4, atol=1e-5)
        torch.testing.assert_allclose(lam1, self.Sv / self.M, rtol=1e-4, atol=1e-5)

    def test_single_window(self):
        mw = torch.zeros(5, 3, 4)
        mw[..., 2] = self.mw[..., 2]
        M = mw.sum(-1)
        rw = torch.tensor([0.2, 0.5, 0.9])
        lam, _ = S.redundancy_evidence(self.Sv, M, mw, rw, rw * 0.3)
        torch.testing.assert_allclose(lam, self.Sv / (1.0 + (M - 1.0) * rw), rtol=1e-4, atol=1e-5)

    def test_duplicates(self):
        mw = torch.zeros(1, 1, 2)
        mw[0, 0, 0] = 1.0                                          # one fully attributed event, value 2
        Sv, M = torch.tensor([[2.0]]), torch.tensor([[1.0]])
        z, o = torch.zeros(1), torch.ones(1)
        a0, _ = S.redundancy_evidence(Sv, M, mw, z, z)
        a1, _ = S.redundancy_evidence(Sv, M, mw, o, o)
        mw2 = mw * 2.0                                             # the same event twice in the same window
        b0, _ = S.redundancy_evidence(Sv * 2.0, M * 2.0, mw2, z, z)
        b1, _ = S.redundancy_evidence(Sv * 2.0, M * 2.0, mw2, o, o)
        self.assertAlmostEqual(float(b1), float(a1), places=4)      # rho = 1: a copy adds nothing
        self.assertAlmostEqual(float(b0), 2.0 * float(a0), places=4)  # rho = 0: a copy counts again

    def test_effective_count_bounded(self):
        rw = torch.rand(3)
        _, ne = S.redundancy_evidence(self.Sv, self.M, self.mw, rw, rw * torch.rand(3))
        self.assertTrue(bool((ne <= self.M + 1e-4).all()))
        self.assertTrue(bool((ne >= 0).all()))


class NeuronTests(unittest.TestCase):
    def test_starts_at_z0(self):
        x = inputs()
        for audit in ("reaudit", "ledger"):
            net = make(nonzero=False, audit=audit)
            with torch.no_grad():
                U = net(*x)
            torch.testing.assert_allclose(U, x[5][:, None].expand_as(U))

    def test_causal(self):
        for audit in ("reaudit", "ledger"):
            net = make(audit=audit)
            x = list(inputs())
            with torch.no_grad():
                U = net(*x)
                later = x[2] > 1.5                                 # events of windows >= 2
                x[0] = x[0].clone()
                x[0][later] = torch.randn(int(later.sum()), 6)
                U2 = net(*x)
            torch.testing.assert_allclose(U[:, :2], U2[:, :2])
            self.assertGreater(float((U[:, 2:] - U2[:, 2:]).abs().max()), 1e-4)

    def test_far_events_change_nothing(self):
        net = make()
        x = list(inputs())
        with torch.no_grad():
            U = net(*x)
            far = x[2] > 0.5                                       # all future events moved out of reach
            x[8] = x[8].clone()
            x[8][far] = 1e12
            x[0] = x[0].clone()
            x[0][far] = torch.randn(int(far.sum()), 6)
            U_far = net(*x)
            x[0][far] = torch.randn(int(far.sum()), 6)
            U_far2 = net(*x)
        torch.testing.assert_allclose(U_far, U_far2)               # their content no longer matters
        self.assertGreater(float((U - U_far).abs().max()), 1e-4)   # but near events did

    def test_ledger_freezes_old_entries(self):
        net = make(audit="ledger")
        x = inputs()
        with torch.no_grad():
            U = net(*x)
        steps = U[:, 1:] - U[:, :-1]
        net2 = make(audit="ledger")
        with torch.no_grad():
            x2 = list(x)
            x2[1] = x[1] & (x[2] < 0.5)                            # remove every future event
            U2 = net2(*x2)
        torch.testing.assert_allclose(U2[:, 1:], U2[:, :1].expand(-1, U2.shape[1] - 1))   # nothing new -> no change
        self.assertGreater(float(steps.abs().max()), 1e-4)

    def test_options_and_constants(self):
        x = inputs()
        for kw in ({"gate": "softmax"}, {"rho": "sum"}, {"rho": "mean"}):
            net = make(**kw)
            with torch.no_grad():
                U = net(*x)
            self.assertTrue(bool(torch.isfinite(U).all()))
        c = make().learned_constants()
        self.assertTrue(all(t <= w + 1e-9 for t, w in zip(c["rho_t"], c["rho_w"])))

    def test_nothing_arrived(self):
        net = make()
        x = list(inputs())
        x[1] = torch.zeros_like(x[1])
        with torch.no_grad():
            U = net(*x)
        self.assertTrue(bool(torch.isfinite(U).all()))
        torch.testing.assert_allclose(U, x[5][:, None].expand_as(U))

    def test_gradients_flow(self):
        net = make()
        net.train()
        x = inputs()
        U = net(*x)
        U.pow(2).mean().backward()
        for name in ("r_w", "r_t", "log_sigma", "rf_bias"):
            self.assertIsNotNone(getattr(net, name).grad)
            self.assertTrue(bool(torch.isfinite(getattr(net, name).grad).all()))


class BandTests(unittest.TestCase):
    def test_martingale_band(self):
        import evidence as EV
        tau = T._logit(0.9) - float(np.log(30.0))                   # V5: tau = -1.20, p_tau = 0.23
        lo, hi = EV.band_bounds(tau, 0.01, "martingale")
        self.assertAlmostEqual(lo, -6.07, delta=0.01)
        self.assertAlmostEqual(hi, 4.86, delta=0.01)
        p_tau = 1.0 / (1.0 + np.exp(-tau))
        self.assertAlmostEqual(1.0 / (1.0 + np.exp(-lo)) / p_tau, 0.01, places=6)          # Ville bound = alpha
        self.assertAlmostEqual((1.0 / (1.0 + np.exp(hi))) / (1.0 - p_tau), 0.01, places=6)
        self.assertEqual(EV.band_bounds(tau, 0.01, "sym"), EV.bounds(tau, 0.01))

    def test_band_in_extraction(self):
        import tempfile
        import evidence as EV
        sys.path.insert(0, os.path.join(ROOT, "tests"))
        from test_deco_v10 import synthetic_recording
        with tempfile.TemporaryDirectory() as tmp:
            pz, pd = synthetic_recording(tmp)
            base = (pz, pd, 0.01, 0.01, 3, 3, 50, np.inf, 4, 4)
            sym, sym2 = EV.one_recording(base), EV.one_recording(base + ("sym",))
            mart = EV.one_recording(base + ("martingale",))
            np.testing.assert_array_equal(sym[0]["idx"], sym2[0]["idx"])
            self.assertGreater(mart[0]["idx"].size, sym[0]["idx"].size)     # the martingale band is wider here
            self.assertTrue(set(sym[0]["idx"].tolist()) <= set(mart[0]["idx"].tolist()))


class SlopeTests(unittest.TestCase):
    def test_slope(self):
        rng = np.random.RandomState(0)
        z = rng.randn(20000, 1) * 3.0
        y = rng.rand(20000) < 1.0 / (1.0 + np.exp(-z[:, 0]))
        F_ = np.zeros(20000, int)
        self.assertAlmostEqual(T.calibration_slopes(z, y, F_)[0], 1.0, delta=0.05)
        self.assertAlmostEqual(T.calibration_slopes(2.0 * z, y, F_)[0], 0.5, delta=0.03)


if __name__ == "__main__":
    unittest.main()
