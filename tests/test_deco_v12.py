"""DeCo-SNN V12 tests (11-09; CPU, synthetic data, torch 1.9 compatible).

    1. Streaming steps with one age per row (sdn_v11.stream_step) = the batch forward of the same neurons
    2. The split small network (layers at once, dec1 + readout window by window) = forward_chunk, with stream starts
       inside the chunk; the zero-initialised feedback convolution leaves it unchanged
    3. V12 candidates (store layout) = the nested / staged GPU rule
    4. The streaming engine (births, candidates as windows arrive, one step per window) = the batch forward of every
       neuron on its final candidates; carrying it across chunks (store pruning) changes nothing
    5. Feedback maps: innovation and belief channels
    6. One training epoch + export on synthetic recordings: finite loss, gradients reach the small network, the
       neurons and the feedback convolution; the export covers every event exactly once
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
import deco_v12 as V12  # noqa: E402
import nested_v11 as NV  # noqa: E402
import sdn_v11 as S  # noqa: E402

CPU = torch.device("cpu")
H, W = 16, 24


def small_cfg():
    import train_stream_v1 as T
    cfg = T.load_flat_config(os.path.join(ROOT, "configs", "evisseg_stream_tif_comm_peak.yaml"))
    cfg.update(channels=[2, 4, 4, 4], height=H, width_px=W, pad_height=H, pad_width=W, n_windows=12)
    return cfg


def small_net(seed=0, double=False):
    import train_stream_v1 as T
    torch.manual_seed(seed)
    net = T.build_net(small_cfg(), CPU)
    net.gain_calibrated.fill_(True)
    for block in net.blocks():                       # strong gains: an active network
        block.gain.set_gain(torch.full((block.out_ch,), 4.0))
    return net.double() if double else net


def neuron_inputs(B=6, n_off=5, K=3, seed=0):
    rng = np.random.RandomState(seed)
    T_ = n_off * K
    feat = torch.as_tensor(rng.randn(B, T_, 6), dtype=torch.float32)
    ok = torch.as_tensor(rng.rand(B, T_) > 0.3)
    P = (n_off - 1) // 2
    dw = torch.arange(-P, n_off - P).float().repeat_interleave(K)[None, :].expand(B, T_).contiguous()
    dist2 = torch.as_tensor(rng.rand(B, T_) * 400.0, dtype=torch.float32)
    z0 = torch.as_tensor(rng.randn(B) * 2.0, dtype=torch.float32)
    p = torch.as_tensor(rng.randint(0, 2, B), dtype=torch.float32)
    return feat, ok, dw, dist2, z0, p, P


class StreamStepTests(unittest.TestCase):
    def test_rows_with_their_own_age(self):
        for value, rho in (("signed", "mean"), ("ei", "mean"), ("signed", "learn")):
            torch.manual_seed(1)
            net = S.SpikingDecisionNeuronsV11(5, 3, sigma0=10.0, value=value, rho=rho)
            with torch.no_grad():
                net.v_out.weight.normal_(0, 0.5)
                net.wm.weight.normal_(0, 0.3)
            feat, ok, dw, dist2, z0, p, P = neuron_inputs()
            D = 5 - 1 - P
            valid = torch.ones(len(z0), D + 1, dtype=torch.bool)
            ref = net(feat, ok, dw, valid, z0 / 5.0, z0, p, D, dist2)
            birth = torch.tensor([0, 1, 1, 2, 3, 0])            # staggered births: one call holds several ages
            c0 = net.stream_start(z0 / 5.0, p)
            c, U = c0.clone(), z0.clone()
            got = torch.full_like(ref, float("nan"))
            for t in range(int(birth.max()) + D + 1):
                age = t - birth
                rows = torch.nonzero((age >= 0) & (age <= D), as_tuple=True)[0]
                with torch.no_grad():
                    u, cc, _, _ = net.stream_step(feat[rows], ok[rows], dw[rows], dist2[rows], z0[rows], c0[rows],
                                                  c[rows], U[rows], age[rows])
                U[rows], c[rows] = u, cc
                got[rows, age[rows]] = u
            self.assertLess(float((got - ref.detach()).abs().max()), 1e-4, (value, rho))       # float32 rounding


def chunk_events(T_, B, n, seed=0):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.randint(0, T_, n))
    b = rng.randint(0, B, n)
    o = np.lexsort((b, t))
    return {"t": torch.as_tensor(t[o]).long(), "b": torch.as_tensor(b[o]).long(),
            "y": torch.as_tensor(rng.randint(0, H, n)).long(), "x": torch.as_tensor(rng.randint(0, W, n)).long(),
            "p": torch.as_tensor(rng.randint(0, 2, n)).double(), "t_local": torch.as_tensor(rng.rand(n)).double()}


class SplitSmallNetworkTests(unittest.TestCase):
    def test_equals_forward_chunk(self):
        small = small_net(double=True)
        model = V12.DeCoV12(small, P=2, D=2, k_near=2, k_evid=2, feedback="innov").double()
        T_, B = 6, 2
        rng = np.random.RandomState(3)
        x = torch.as_tensor(rng.rand(T_, B, 12, H, W) * 3.0)
        keep = torch.ones(T_, B, dtype=torch.float64)
        keep[3, 1] = 0.0                                       # stream 1 starts a new recording at window 3
        ev = chunk_events(T_, B, 400)
        _, states, _ = small.forward_chunk(torch.as_tensor(rng.rand(4, B, 12, H, W) * 3.0), chunk_events(4, B, 50, 1),
                                           None)
        ref, ref_states, _ = small.forward_chunk(x, ev, states, keep=keep)
        s1, deep1, st6, _ = model.pre_dec1(x, states, keep)
        from model.lif2d_stream import mask_state
        fb_map = torch.rand(B, 2, H, W, dtype=torch.float64)    # any map: the convolution starts at zero
        state7, outs = states[6], []
        for w in range(T_):
            state7 = mask_state(state7, keep, w)
            u7, state7 = model.dec1_window(s1[w], deep1[w], state7, fb_map.float().double())
            sel = ev["t"] == w
            outs.append(model.readout_window(u7, deep1[w], {k_: v[sel] for k_, v in ev.items() if k_ != "t"})
                        + model.log_w)
        got = torch.cat(outs)
        self.assertGreater(float(ref.abs().sum()), 0.0)
        self.assertLess(float((got - ref).abs().max()), 1e-10)
        for a, b in zip(st6 + [state7], ref_states):
            a = a if isinstance(a, tuple) else (a,)
            b = b if isinstance(b, tuple) else (b,)
            for u, v in zip(a, b):
                self.assertLess(float((u - v).abs().max()), 1e-10)


def synthetic_store(n=600, n_win=9, window_ms=50, seed=0, B=1):
    """Engine columns of B streams of random float positions (no distance ties), sorted by (window, stream)."""
    rng = np.random.RandomState(seed)
    g = np.sort(rng.randint(0, n_win, n))
    b = rng.randint(0, B, n)
    o = np.lexsort((b, g))
    g, b = g[o], b[o]
    tl = rng.rand(n).astype(np.float32)
    xf, yf = rng.rand(n) * 300.0, rng.rand(n) * 200.0
    cols = {"g": torch.as_tensor(g).long(), "b": torch.as_tensor(b).long(), "xf": torch.as_tensor(xf).float(),
            "yf": torch.as_tensor(yf).float(), "y": torch.as_tensor(yf.astype(np.int64) % H),
            "x": torch.as_tensor(xf.astype(np.int64) % W), "pf": torch.as_tensor(rng.randint(0, 2, n)).float(),
            "tl": torch.as_tensor(tl), "lab": torch.as_tensor(rng.rand(n) < 0.3).float(),
            "z": torch.as_tensor(rng.randn(n) * 4.0 - 1.0).float(), "rec": torch.as_tensor(b).long(),
            "fidx": torch.arange(n), "rw": torch.as_tensor(g).long(), "glo": torch.zeros(n, dtype=torch.long),
            "ghi": torch.full((n,), n_win - 1, dtype=torch.long)}
    cols["tm"] = (cols["rw"].float() + cols["tl"]) * window_ms
    return cols


class PickTests(unittest.TestCase):
    def test_equals_nested_rule(self):
        cols = synthetic_store()
        st = V12.EventStore(1, CPU)
        st.append(cols)
        rng = np.random.RandomState(1)
        q = torch.as_tensor(np.sort(rng.choice(st.n, 80, replace=False))).long()
        offsets = list(range(-3, 4))
        cq = st.g[q][:, None] + torch.as_tensor(offsets)[None, :]
        cq = torch.where((cq >= 0) & (cq < 9), cq, torch.full_like(cq, -1))
        kn, ke, flo = 3, 3, -3.0
        ref_p, ref_ok = NV.gpu_picks(st.xf, st.yf, st.g, st.z, q, cq, kn, ke, flo)
        qi, qo = torch.nonzero(cq >= 0, as_tuple=True)
        pk, okk = V12.gpu_picks(st, q[qi], cq[qi, qo], kn, ke, flo, block=7)
        got_p = torch.full_like(ref_p, -1)
        got_ok = torch.zeros_like(ref_ok)
        got_p[qi, qo], got_ok[qi, qo] = pk, okk
        self.assertTrue(bool((got_ok == ref_ok).all()))
        self.assertTrue(bool((torch.where(ref_ok, got_p, -1) == torch.where(ref_ok, ref_p, -1)).all()))


def decision_model(P=2, D=3, feedback="innov", birth="none"):
    small = small_net()
    model = V12.DeCoV12(small, P=P, D=D, k_near=3, k_evid=2, feedback=feedback, birth=birth)
    with torch.no_grad():
        model.big.v_out.weight.normal_(0, 0.5)
        model.big.wm.weight.normal_(0, 0.3)
    return model


def run_engine(model, cols, n_win, B, chunks, gate=None):
    """Streams the synthetic columns through the engine with the given chunk boundaries; returns the retired neurons
    (eid -> fidx) and the engine of the last chunk."""
    eng = V12.Engine(model, B, H, W, CPU, keep_traj=True, sub_batch=5, pick_block=6, gate=gate)
    outs = []
    for g0, g1 in chunks:
        eng.begin_chunk(g0 - model.P - model.D)
        sel = (cols["g"] >= g0) & (cols["g"] < g1)
        if bool(sel.any()):
            eng.st.append({k: v[sel] for k, v in cols.items()})
        for g in range(g0, g1):
            a, e = eng.st.window(g)
            eng.birth(g, a, e)
            eng.advance(g)
            eng.feedback(g)
            eng.retire(g)
        for q in eng.retired:
            outs.append({"fidx": eng.st.fidx[q["eid"]], "traj": q["traj"], "pick": eng.st.fidx[q["pick"].long().clamp(
                min=0)] * q["ok"].long() - (~q["ok"]).long(), "ok": q["ok"], "gs": q["gs"]})
        eng.retired = []
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}, eng


class EngineTests(unittest.TestCase):
    def test_streaming_equals_batch_and_chunking(self):
        for birth in ("none", "band"):
            model = decision_model(birth=birth)
            n_win, B = 9, 2
            cols = synthetic_store(n=500, n_win=n_win, B=B, seed=4)
            whole, _ = run_engine(model, cols, n_win, B, [(0, n_win)])
            parts, _ = run_engine(model, cols, n_win, B, [(0, 3), (3, 5), (5, 9)])
            o1, o2 = torch.argsort(whole["fidx"]), torch.argsort(parts["fidx"])
            self.assertTrue(bool((whole["fidx"][o1] == parts["fidx"][o2]).all()))
            self.assertTrue(bool((whole["pick"][o1] == parts["pick"][o2]).all()))
            a, b = whole["traj"][o1], parts["traj"][o2]
            self.assertTrue(bool((torch.isnan(a) == torch.isnan(b)).all()))
            self.assertLess(float((torch.nan_to_num(a) - torch.nan_to_num(b)).abs().max()), 1e-5)
            if birth == "none":
                self.assertEqual(int(whole["fidx"].numel()), int(cols["g"].numel()))   # every event got a neuron
            # batch forward of every neuron on its final candidates
            st = V12.EventStore(B, CPU)
            st.append(cols)
            pick = whole["pick"][o1]
            ok = whole["ok"][o1]
            eid = whole["fidx"][o1]                            # fidx = store index in this one-shot store
            last = torch.clamp(st.ghi[eid] - whole["gs"][o1], max=model.D)
            run = torch.arange(model.D + 1)[None, :] <= last[:, None]
            with torch.no_grad():
                U = model.trajectories(st, st.z, eid, torch.where(ok, pick, torch.zeros_like(pick)), ok, run)
            ref = torch.where(run, U, torch.full_like(U, float("nan")))
            self.assertLess(float((torch.nan_to_num(ref) - torch.nan_to_num(a)).abs().max()), 1e-5)
            self.assertTrue(bool((torch.isnan(ref) == torch.isnan(a)).all()))


class GatedEngineTests(unittest.TestCase):
    def test_fire_and_stop(self):
        model = decision_model(feedback="none")
        n_win, B = 9, 2
        cols = synthetic_store(n=500, n_win=n_win, B=B, seed=5)
        full, ef = run_engine(model, cols, n_win, B, [(0, 4), (4, 9)])
        gated, eg = run_engine(model, cols, n_win, B, [(0, 4), (4, 9)], gate=(-1.0, 1.5))
        o1, o2 = torch.argsort(full["fidx"]), torch.argsort(gated["fidx"])
        a, b = full["traj"][o1], gated["traj"][o2]
        stopped = 0
        for i in range(a.shape[0]):
            for d in range(a.shape[1]):
                if torch.isnan(b[i, d]):
                    stopped += int(not torch.isnan(a[i, d]))
                    continue
                self.assertLess(abs(float(a[i, d] - b[i, d])), 1e-5)       # identical until it fired
                if d and not (-1.0 < float(b[i, d - 1]) < 1.5):
                    self.fail("a fired neuron kept computing")
        self.assertGreater(stopped, 0)
        self.assertLess(eg.stats["steps"], ef.stats["steps"])
        self.assertEqual(eg.stats["births"], ef.stats["births"])


class FeedbackMapTests(unittest.TestCase):
    def test_channels(self):
        b = torch.tensor([0, 0, 1])
        y = torch.tensor([1, 1, 2])
        x = torch.tensor([3, 3, 0])
        U = torch.tensor([2.0, -1.0, 0.0])
        z0 = torch.tensor([0.0, 0.0, 0.0])
        m = V12.feedback_map("innov", 2, 4, 5, b, y, x, U, z0)
        su = torch.sigmoid(U)
        self.assertAlmostEqual(float(m[0, 0, 1, 3]), float(torch.log1p(su[0] - 0.5)), places=6)
        self.assertAlmostEqual(float(m[0, 1, 1, 3]), float(torch.log1p(0.5 - su[1])), places=6)
        self.assertEqual(float(m[1].abs().sum()), 0.0)          # no update, no news
        m = V12.feedback_map("belief", 2, 4, 5, b, y, x, U, z0)
        self.assertAlmostEqual(float(m[0, 0, 1, 3]), float(torch.log1p(su[0] + su[1])), places=6)
        self.assertAlmostEqual(float(m[1, 1, 2, 0]), float(torch.log1p(1.0 - su[2])), places=6)


def write_dataset(root, n_win=12):
    from test_stream_layerwise import synthetic_sequence
    for split, seeds in (("train", (3, 4, 5)), ("val", (6, 7)), ("test", (8,))):
        os.makedirs(os.path.join(root, split))
        for seed in seeds:
            seq = synthetic_sequence(seed=seed)
            loc = np.stack((seq.x, seq.y, seq.t), axis=1)
            norm = np.column_stack((loc, seq.p, seq.label, seq.target_id))
            np.savez(os.path.join(root, split, "seq_%d.npz" % seed), ev_loc=loc, evs_norm=norm)


class TrainingTests(unittest.TestCase):
    def test_epoch_and_export(self):
        import argparse
        import train_v12 as TV
        import train_stream_v1 as T
        from dataset.ev_uav_stream import EvUAVStream
        with tempfile.TemporaryDirectory() as root:
            write_dataset(root)
            cfg = small_cfg()
            cfg.update(root=root, execution="layer", input_device="gpu", tbptt_k=4, streams=2, accumulate=1,
                       stream_groups=1, n_windows=12)
            torch.manual_seed(0)
            small = T.build_net(cfg, CPU)
            small.gain_calibrated.fill_(True)
            for block in small.blocks():
                block.gain.set_gain(torch.full((block.out_ch,), 4.0))
            model = V12.DeCoV12(small, P=2, D=2, k_near=2, k_evid=2, feedback="innov")
            q99 = np.linspace(0.5, 1.5, 12).astype(np.float32)
            args = argparse.Namespace(samples=64, strata=6, balance="on", prior_grad="on", own_prior_grad="off", max_chunks=0, sub_batch=64, profile=False,
                                      pick_block=32, log_every=0, eval_streams=2, eval_chunk=5)
            train_set = EvUAVStream(root, "train", cfg)
            lengths = [int(train_set[i].n_windows) for i in range(len(train_set))]
            seen = {}

            class Probe(torch.optim.SGD):
                def step(self, closure=None):
                    seen["small"] = sum(float(p.grad.abs().sum()) for p in model.small.parameters() if p.grad is not None)
                    seen["big"] = sum(float(p.grad.abs().sum()) for p in model.big.parameters() if p.grad is not None)
                    seen["fb"] = float(model.fb.weight.grad.abs().sum()) if model.fb.weight.grad is not None else 0.0
                    return super(Probe, self).step(closure)
            opt = Probe(model.parameters(), lr=0.01)
            rec = TV.train_epoch(model, cfg, q99, train_set, lengths, opt, None, CPU, np.random.RandomState(0), args)
            self.assertTrue(np.isfinite(rec["loss"]))
            self.assertGreater(rec["updates"], 0)
            self.assertGreater(seen["small"], 0.0)
            self.assertGreater(seen["big"], 0.0)
            self.assertGreater(seen["fb"], 0.0)
            with tempfile.TemporaryDirectory() as out:
                summ = TV.export_split(model, cfg, q99, "val", CPU, args, out=out)
                with np.load(os.path.join(out, "evm_val_meta.npz")) as m:
                    names = list(m["names"])
                    for r, name in enumerate(names):
                        with np.load(os.path.join(root, "val", str(name))) as d:
                            n_ev = int(d["ev_loc"].shape[0])
                        idx = m["idx"][m["rec"] == r]
                        self.assertEqual(sorted(idx.tolist()), list(range(n_ev)))       # every event once
                self.assertEqual(summ["neurons"], sum(int(np.load(os.path.join(root, "val", n_))["ev_loc"].shape[0])
                                                      for n_ in names))
                with np.load(os.path.join(out, "run", "raw_val.npz")) as r_:
                    self.assertFalse(bool(np.isnan(r_["lam_traj"]).any()))
            # the export (two recordings in parallel, chunks of 5 windows) = deployment (one recording, one window)
            names = sorted(os.listdir(os.path.join(root, "val")))
            za, na = TV.stream_split(model, cfg, q99, "val", CPU, names, 2, 5, 64, 32)
            zb, nb = TV.stream_split(model, cfg, q99, "val", CPU, names, 1, 1, 64, 32)
            for a, b in zip(za, zb):
                self.assertLess(float(np.abs(a - b).max()), 1e-5)
            oa, ob = np.lexsort((na["fidx"], na["rec"])), np.lexsort((nb["fidx"], nb["rec"]))
            self.assertTrue(bool((na["fidx"][oa] == nb["fidx"][ob]).all()))
            ta, tb = na["traj"][oa], nb["traj"][ob]
            self.assertTrue(bool((np.isnan(ta) == np.isnan(tb)).all()))
            self.assertLess(float(np.nanmax(np.abs(ta - tb))), 1e-4)


if __name__ == "__main__":
    unittest.main()
