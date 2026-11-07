"""DeCo V10 tests (11-07; CPU, synthetic data, torch 1.9 compatible).

    1. Evidence without a radius cap (tools/deco/evidence.py --radius 0) = a cap larger than the frame; counts become
       window totals
    2. V10 decision neurons (tools/deco/train_dnm.py --design v10): start exactly at z0; tokens far outside the learned
       receptive field change nothing (always-on empty key); near tokens do; receptive field per head or per offset
    3. V10 featurizer: z-scored token features, RMS token distance as the receptive-field start
    4. Change-risk head (tools/deco/fit_head.py): the state at step d is causal; commit rule; on synthetic trajectories
       whose change risk falls with the margin and towards the deadline, the learned head does the same
    5. Evidence too big for the GPU (EV-Flying, train_dnm.py --gpu-evidence-gb): the memory-mapped arrays equal np.load,
       and host and device evidence give identical v10 and v9 features
"""
import gc
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "deco"))
import evidence as EV  # noqa: E402
import fit_head as FH  # noqa: E402
import train_dnm as T  # noqa: E402


def synthetic_recording(tmp, n=600, n_win=8, window_ms=50, seed=0):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.randint(0, n_win * window_ms * 1000, n)) // 1000          # ms
    x, y = rng.randint(0, 346, n), rng.randint(0, 260, n)
    loc = np.stack([x, y, t], 1).astype(np.int64)
    norm = np.zeros((n, 6), np.float32)
    norm[:, 3] = rng.randint(0, 2, n)
    norm[:, 4] = (rng.rand(n) < 0.2).astype(np.float32)
    log_w = float(np.log(30.0))
    tau = T._logit(0.9) - log_w
    z0 = (tau + rng.randn(n) * 4.0).astype(np.float32)                          # many events inside the band
    path_d, path_z = os.path.join(tmp, "rec.npz"), os.path.join(tmp, "z0.npz")
    np.savez(path_d, ev_loc=loc, evs_norm=norm)
    np.savez(path_z, traj=z0[:, None], log_w=np.float32(log_w), window_ms=np.int64(window_ms))
    return path_z, path_d


class EvidenceRadiusTests(unittest.TestCase):
    def test_no_cap_equals_huge_radius(self):
        with tempfile.TemporaryDirectory() as tmp:
            pz, pd = synthetic_recording(tmp)
            job = lambda r: (pz, pd, 0.01, 0.01, 3, 3, 50, r, 4, 4)            # noqa: E731
            a, b = EV.one_recording(job(np.inf)), EV.one_recording(job(1e9))
            self.assertGreater(a[0]["idx"].size, 10)
            for k in a[0]:
                np.testing.assert_array_equal(a[0][k], b[0][k])
            for i in (1, 2, 3, 4):                                             # tokens, valid, counts
                np.testing.assert_array_equal(a[i], b[i])
            capped = EV.one_recording(job(5.0))
            self.assertLess(capped[2].sum(), a[2].sum())                      # a small cap drops tokens


def tiny_inputs(B=5, n_off=4, K=3, D=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    T_ = n_off * K
    feat = torch.randn(B, T_, T.N_FEAT_V10, generator=g)
    ok = torch.ones(B, T_, dtype=torch.bool)
    dw = torch.arange(n_off).repeat_interleave(K).float()[None].expand(B, T_) - 1.0
    valid = torch.ones(B, D + 1, dtype=torch.bool)
    z0 = torch.randn(B, generator=g)
    p = torch.randint(0, 2, (B,), generator=g).float()
    dist2 = torch.rand(B, T_, generator=g) * 100.0
    return feat, ok, dw, valid, (z0 - 0.1) / 2.0, z0, p, D, dist2


class V10NeuronTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = T.DecisionNeuronsV10(n_off=4, K=3, sigma0=10.0).eval()

    def test_starts_at_z0(self):
        args = tiny_inputs()
        lam = self.net(*args)
        self.assertEqual(tuple(lam.shape), (5, 3))
        np.testing.assert_allclose(lam.detach().numpy(), args[5][:, None].expand(5, 3).numpy(), rtol=0, atol=1e-6)

    def test_far_tokens_are_ignored_near_tokens_are_not(self):
        with torch.no_grad():                                                  # a "trained" readout
            self.net.delta[-1].weight.normal_(0, 1.0)
            self.net.wm.weight.normal_(0, 1.0)
        feat, ok, dw, valid, z_in, z0, p, D, dist2 = tiny_inputs()
        extra = 3                                                              # three more tokens per event
        feat2 = torch.cat([feat, torch.randn(5, extra, feat.shape[-1])], 1)
        ok2 = torch.cat([ok, torch.ones(5, extra, dtype=torch.bool)], 1)
        dw2 = torch.cat([dw, torch.zeros(5, extra)], 1)
        net2 = T.DecisionNeuronsV10(n_off=5, K=3, sigma0=10.0).eval()
        net2.load_state_dict({k: v for k, v in self.net.state_dict().items() if k != "off_idx"}, strict=False)
        ref = net2(torch.cat([feat, torch.zeros(5, extra, feat.shape[-1])], 1),
                   torch.cat([ok, torch.zeros(5, extra, dtype=torch.bool)], 1), dw2, valid, z_in, z0, p, D,
                   torch.cat([dist2, torch.zeros(5, extra)], 1))
        far = net2(feat2, ok2, dw2, valid, z_in, z0, p, D, torch.cat([dist2, torch.full((5, extra), 1e6)], 1))
        near = net2(feat2, ok2, dw2, valid, z_in, z0, p, D, torch.cat([dist2, torch.ones(5, extra)], 1))
        np.testing.assert_allclose(far.detach().numpy(), ref.detach().numpy(), rtol=0, atol=1e-5)
        self.assertGreater(float((near - ref).abs().max().detach()), 1e-3)

    def test_receptive_field_sharing(self):
        self.assertEqual(tuple(self.net.log_sigma.shape), (4, 1))
        per_off = T.DecisionNeuronsV10(n_off=4, K=3, sigma0=10.0, per="head_offset")
        self.assertEqual(tuple(per_off.log_sigma.shape), (4, 4))
        self.assertEqual(per_off.off_idx.tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])


class FeaturizerTests(unittest.TestCase):
    def test_zscores_and_rms_distance(self):
        rng = np.random.RandomState(1)
        tok = rng.randn(40, 3, 4, 6).astype(np.float32) * [5, 5, 20, 1, 0.5, 3] + [1, -2, 10, 0, 0.5, -1]
        ok = rng.rand(40, 3, 4) < 0.7
        g = {"tok": torch.as_tensor(tok.astype(np.float16)), "ok": torch.as_tensor(ok),
             "valid": torch.ones(40, 3, dtype=torch.bool), "z0": torch.as_tensor(rng.randn(40).astype(np.float32)),
             "p": torch.zeros(40), "y": torch.zeros(40), "win_s": 0.05}
        fz = T.FeaturizerV10()
        sigma0 = fz.fit(g, 40, chunk=16)
        t16 = tok.astype(np.float16).astype(np.float64)
        self.assertAlmostEqual(sigma0, float(np.sqrt((t16[..., 0] ** 2 + t16[..., 1] ** 2)[ok].mean())), places=4)
        feat = fz.batch(g, torch.arange(40))[0]
        x = feat[torch.as_tensor(ok).reshape(40, -1)].double()
        np.testing.assert_allclose(x.mean(0).numpy(), np.zeros(6), rtol=0, atol=1e-4)
        np.testing.assert_allclose(x.std(0, unbiased=False).numpy(), np.ones(6), rtol=0, atol=1e-3)


class ChangeRiskHeadTests(unittest.TestCase):
    def test_state_is_causal(self):
        rng = np.random.RandomState(2)
        z0, lt = rng.randn(30), rng.randn(30, 6) * 3
        X = FH.state_features(z0, lt, 0.2, 5)
        lt2 = lt.copy()
        lt2[:, 3:] += rng.randn(30, 3) * 10
        X2 = FH.state_features(z0, lt2, 0.2, 5)
        np.testing.assert_array_equal(X[:, :3], X2[:, :3])
        self.assertFalse(np.array_equal(X[:, 3:], X2[:, 3:]))

    def test_commit_rule(self):
        p = np.array([[0.5, 0.004, 0.0], [0.5, 0.5, 0.5], [0.001, 0.5, 0.5], [0.5, 0.5, 0.002]])
        F_ = np.array([2, 2, 2, 1])
        np.testing.assert_array_equal(FH.commit_by_prob(p, F_, 0.01), [1, 2, 0, 1])

    def test_head_learns_margin_and_deadline(self):
        rng = np.random.RandomState(3)
        N, D, tau = 4000, 6, 0.0
        z0 = rng.randn(N) * 3
        steps = rng.randn(N, D + 1) * np.linspace(1.5, 0.2, D + 1)            # less movement near the deadline
        lt = z0[:, None] + np.cumsum(steps, 1)
        F_ = np.full(N, D)
        X = FH.state_features(z0, lt, tau, D)
        predict, state = FH.fit(X, lt, F_, tau, D, np.ones(N, bool), steps=400)
        self.assertGreater(state["rows"], 0)
        med = np.median(X[:, :D].reshape(-1, FH.N_STATE), 0)
        grid = lambda m, d: float(predict(np.r_[np.log1p(m), med[1:5], d / D].astype(np.float32)[None])[0])  # noqa
        self.assertGreater(grid(0.3, 1), grid(4.0, 1))                         # risk falls with the margin
        self.assertGreater(grid(1.5, 1), grid(1.5, 5))                         # and towards the deadline


def write_evidence(tmp, n=37, n_off=3, K=4, D=2, seed=4):
    rng = np.random.RandomState(seed)
    arrays = {"tok": (rng.randn(n, n_off, K, 6) * 10).astype(np.float16), "tok_valid": rng.rand(n, n_off, K) < 0.7,
              "nall": rng.randint(0, 50, (n, n_off)).astype(np.int32),
              "nevid": rng.randint(0, 9, (n, n_off)).astype(np.int32),
              "z0": rng.randn(n).astype(np.float32), "p": rng.randint(0, 2, n).astype(np.float32),
              "label": rng.randint(0, 2, n).astype(np.int8), "F": rng.randint(0, D + 1, n).astype(np.int64),
              "D": np.int64(D), "k_near": np.int64(K // 2), "window_ms": np.int64(50)}
    path = os.path.join(tmp, "ev_val.npz")
    np.savez(path, **arrays)
    return path, arrays


class HostEvidenceTests(unittest.TestCase):
    def test_memmap_equals_np_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, arrays = write_evidence(tmp)
            for k in T.HOST_KEYS:
                mm = T.npz_memmap(path, k)
                self.assertIsNotNone(mm)
                self.assertEqual(mm.dtype, arrays[k].dtype)
                np.testing.assert_array_equal(np.asarray(mm), arrays[k])
                del mm
            self.assertGreater(T.evidence_bytes(path), arrays["tok"].nbytes)
            gc.collect()                                                       # release the mappings (Windows)

    def test_host_and_device_give_identical_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = write_evidence(tmp)
            _, g_dev, _ = T.load_evidence(path, torch.device("cpu"), -3.0, 2.0, host=False)
            m_host, g_host, _ = T.load_evidence(path, torch.device("cpu"), -3.0, 2.0, host=True)
            self.assertIsInstance(g_host["tok"], np.memmap)
            self.assertTrue(torch.is_tensor(g_dev["tok"]))
            idx = torch.tensor([5, 0, 36, 5, 12])
            fz_dev, fz_host = T.FeaturizerV10(), T.FeaturizerV10()
            self.assertEqual(fz_dev.fit(g_dev, 37, chunk=8), fz_host.fit(g_host, 37, chunk=8))
            for a, b in zip(fz_dev.batch(g_dev, idx), fz_host.batch(g_host, idx)):          # v10 inputs
                np.testing.assert_array_equal(a.numpy(), b.numpy())
            for a, b in zip(T.batch(g_dev, idx), T.batch(g_host, idx)):                    # v9 inputs
                np.testing.assert_array_equal(a.numpy(), b.numpy())
            del m_host, g_host
            gc.collect()


if __name__ == "__main__":
    unittest.main()
