"""Nested V11 tests (11-08; CPU, synthetic data, torch 1.9 compatible).

    1. GPU candidates (tools/deco/nested_v11.gpu_picks) = the staged rule (tools/deco/evidence.candidate_picks) when no
       two candidates are at the same distance
    2. Token building (nested_v11.build_tokens) = the staged evidence tokens (float16) for the same candidates
    3. The neurons with zero report columns compute exactly the staged neurons, whatever the reports are
    4. A small network + report head + neurons: the neurons' loss reaches the small network through the priors, and the
       report columns of the neurons through the reports
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "deco"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import evidence as EV  # noqa: E402
import nested_v11 as NV  # noqa: E402
import sdn_v11 as S  # noqa: E402


def float_recording(n=700, n_win=9, window_ms=50, seed=0):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.randint(0, n_win * window_ms, n)).astype(np.float64)
    loc = np.stack([rng.rand(n) * 300.0, rng.rand(n) * 200.0, t], 1)       # float positions: no distance ties
    zc = (rng.randn(n) * 4.0 - 1.0).astype(np.float32)
    return loc, zc, (loc[:, 2] // window_ms).astype(np.int64), n_win


class PickTests(unittest.TestCase):
    def test_gpu_equals_staged(self):
        loc, zc, win, n_win = float_recording()
        pend = np.nonzero((zc > -4.0) & (zc < 2.0))[0]
        offsets = list(range(-3, 4))
        kn, ke, flo = 4, 4, -3.0
        pick, ok, _, _ = EV.candidate_picks(loc, zc, win, n_win, pend, offsets, kn, ke, flo, np.inf)
        cq = win[pend][:, None] + np.asarray(offsets)[None, :]
        cq = np.where((cq >= 0) & (cq < n_win), cq, -1)
        gp, gok = NV.gpu_picks(torch.as_tensor(loc[:, 0]), torch.as_tensor(loc[:, 1]), torch.as_tensor(win),
                               torch.as_tensor(zc), torch.as_tensor(pend), torch.as_tensor(cq), kn, ke, flo)
        np.testing.assert_array_equal(gok.numpy(), ok)
        np.testing.assert_array_equal(np.where(ok, gp.numpy(), -1), np.where(ok, pick, -1))


class TokenTests(unittest.TestCase):
    def test_tokens_equal_evidence_files(self):
        from test_deco_v10 import synthetic_recording
        with tempfile.TemporaryDirectory() as tmp:
            pz, pd = synthetic_recording(tmp, n=800, n_win=8)
            meta, tok, tok_ok, _, _, log_w, _ = EV.one_recording((pz, pd, 0.01, 0.01, 3, 3, 50, np.inf, 4, 4))
            with np.load(pz) as tr:
                z0 = np.asarray(tr["traj"][:, 0], np.float32)
            with np.load(pd) as d:
                loc, norm = np.asarray(d["ev_loc"]), np.asarray(d["evs_norm"])
        win = (loc[:, 2] // 50).astype(np.int64)
        tau = EV._logit(0.9) - log_w
        flo, _ = EV.bounds(tau, 0.01)
        zc = np.clip(z0, -20.0, 20.0)
        offsets = list(range(-3, 4))
        pick, ok, _, _ = EV.candidate_picks(loc, zc, win, int(win.max()) + 1, meta["idx"], offsets, 4, 4, flo, np.inf)
        f = lambda a: torch.as_tensor(np.asarray(a, np.float32))                 # noqa: E731
        raw, rep, okT, dw, dist2 = NV.build_tokens(f(loc[:, 0]), f(loc[:, 1]), f(loc[:, 2]), f(norm[:, 3]), f(z0),
                                                   torch.zeros(len(z0), 2), torch.as_tensor(meta["idx"]),
                                                   torch.as_tensor(pick), torch.as_tensor(ok), offsets)
        ref = tok.astype(np.float32).reshape(raw.shape)
        np.testing.assert_array_equal(okT.numpy(), tok_ok.reshape(okT.shape))
        np.testing.assert_array_equal(raw.numpy(), ref)
        np.testing.assert_array_equal(dist2.numpy(), ref[..., 0] ** 2 + ref[..., 1] ** 2)


def neuron_inputs(B=5, n_off=7, K=4, D=3, extra=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    T_ = n_off * K
    feat = torch.randn(B, T_, 6 + extra, generator=g)
    ok = torch.rand(B, T_, generator=g) > 0.2
    dw = (torch.arange(n_off) - (n_off - D - 1)).repeat_interleave(K).float()[None].expand(B, T_).clone()
    valid = torch.ones(B, D + 1, dtype=torch.bool)
    z0 = torch.randn(B, generator=g) * 3.0
    return feat, ok, dw, valid, z0 / 5.0, z0, (torch.rand(B, generator=g) > 0.5).float(), D, \
        torch.rand(B, T_, generator=g) * 300.0


class StagedEquivalenceTests(unittest.TestCase):
    def test_zero_report_columns(self):
        torch.manual_seed(0)
        staged = S.SpikingDecisionNeuronsV11(7, 4, sigma0=20.0)
        with torch.no_grad():
            for p_ in staged.parameters():
                p_.normal_(0, 0.3)
        nested = S.SpikingDecisionNeuronsV11(7, 4, sigma0=20.0, n_extra=3)
        nested.load_stage1(staged.state_dict())
        x = list(neuron_inputs(extra=3))
        x6 = list(x)
        x6[0] = x[0][..., :6]
        with torch.no_grad():
            a = staged.eval()(*x6)
            b = nested.eval()(*x)
            x[0] = x[0].clone()
            x[0][..., 6:] = torch.randn_like(x[0][..., 6:]) * 5.0
            c = nested(*x)
        torch.testing.assert_allclose(a, b)
        torch.testing.assert_allclose(a, c)


class JointGradientTests(unittest.TestCase):
    def test_gradients_reach_small_network(self):
        import train_stream_v1 as T
        from utils.stream_common import load_flat_config
        cfg = load_flat_config(os.path.join(ROOT, "configs", "final", "bb_v5_50.yaml"))
        cfg.update(pad_height=32, pad_width=32, height=32, width_px=32, channels=[4, 4, 4, 4])
        torch.manual_seed(0)
        small = T.build_net(cfg, torch.device("cpu"))
        report = NV.ReportHead(small, 3)
        big = S.SpikingDecisionNeuronsV11(7, 4, sigma0=10.0, n_extra=3)
        with torch.no_grad():
            big.v_out.weight.normal_(0, 0.5)
        model = NV.NestedDeCo(small, big, report, [0.0] * 6, [1.0] * 6, 0.0, 5.0, 50)
        steps, n = 7, 300
        rng = np.random.RandomState(0)
        tt = torch.as_tensor(np.sort(rng.randint(0, steps, n))).long()
        events = {"t": tt, "b": torch.zeros(n, dtype=torch.long), "y": torch.as_tensor(rng.randint(0, 32, n)).long(),
                  "x": torch.as_tensor(rng.randint(0, 32, n)).long(), "p": torch.as_tensor(rng.randint(0, 2, n)).float(),
                  "t_local": torch.as_tensor(rng.rand(n)).float()}
        x = torch.rand(steps, 1, 12, 32, 32)
        logits, _, _ = small.forward_chunk(x, events, None)
        rep = report()
        z0 = logits - 1.0
        pend = torch.nonzero(tt == 3, as_tuple=True)[0][:6]
        offsets = list(range(-3, 4))
        cq = tt[pend][:, None] + torch.as_tensor(offsets)[None, :]
        xf, yf = events["x"].float(), events["y"].float()
        pick, ok = NV.gpu_picks(xf, yf, tt, z0.detach(), pend, cq, 2, 2, -100.0)
        t_ms = (tt.float() + events["t_local"]) * 50.0
        raw, rtok, okT, dw, dist2 = NV.build_tokens(xf, yf, t_ms, events["p"], z0, rep, pend, pick, ok, offsets)
        U = model.decide(raw, rtok, okT, dw, dist2, z0[pend], events["p"][pend], torch.ones(len(pend), 4, dtype=torch.bool), 3)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(U, torch.ones_like(U))
        loss.backward()
        g_small = sum(float(p_.grad.abs().sum()) for p_ in small.parameters() if p_.grad is not None)
        self.assertGreater(g_small, 0.0)
        g_cols = float(big.tok[0].weight.grad[:, 6:].abs().sum())
        self.assertGreater(g_cols, 0.0)                    # report columns learn first (they start at zero)


if __name__ == "__main__":
    unittest.main()
