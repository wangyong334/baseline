"""V18 decision-unit tests (11-13; CPU, synthetic data, torch 1.9 compatible).

    1. Foveated layout: 1 + 8 L boxes that tile the square of side 3^(L+1) px exactly once; box 4 holds the event
    2. Evidence maps: every box feature equals a brute-force count over the events (edge clipping, the in-frame area,
       the event itself removed from its centre box in its own window); polarity is relative, so flipping every
       polarity changes nothing; windows past the recording end are masked
    3. Decision units: start exactly at the prior (every pool and observation unit); causal (step k ignores later
       windows) in both audit modes; windows add up: with fixed gates and context the change from step k - 1 to k depends on
       window k only; steps past the last valid one keep the membrane; opponent channels keep their signs whatever the
       weights; the ablations run; step costs grow with the audited windows
    3b. Streaming: the batched window-by-window layer commits every event at the same column with the same value as
       the offline trajectories (boxes and events; re-audit and ledger units; every token form)
    4. Stopping rule: isotonic bound at the margin where changes stop, forced commit at the last column, entry column
       commits safe priors, unit load per window
"""
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "v18"))
from model import sdu as S  # noqa: E402
import stopping as P  # noqa: E402


def recording(n=400, W=40, H=30, n_win=5, seed=0):
    rng = np.random.RandomState(seed)
    return {"x": rng.randint(0, W, n), "y": rng.randint(0, H, n), "win": rng.randint(0, n_win, n),
            "pol": rng.randint(0, 2, n), "z0": rng.randn(n).astype(np.float32) * 4.0, "W": W, "H": H, "n_win": n_win}


def brute(rec, e, rect, n_steps):
    """Features of event e by direct counting."""
    q = np.round(1.0 / (1.0 + np.exp(-rec["z0"].astype(np.float64))) * S.Q_SCALE)
    out = np.zeros((n_steps, len(rect), S.N_FEAT))
    ok = np.zeros(n_steps, bool)
    for s in range(n_steps):
        w = rec["win"][e] + s
        if w >= rec["n_win"]:
            continue
        ok[s] = True
        for k, (x0, x1, y0, y1) in enumerate(rect):
            gx0, gx1, gy0, gy1 = rec["x"][e] + x0, rec["x"][e] + x1, rec["y"][e] + y0, rec["y"][e] + y1
            inside = ((rec["win"] == w) & (rec["x"] >= gx0) & (rec["x"] <= gx1) & (rec["y"] >= gy0) &
                      (rec["y"] <= gy1))
            if s == 0:
                inside[e] = False
            same = inside & (rec["pol"] == rec["pol"][e])
            opp = inside & (rec["pol"] != rec["pol"][e])
            cx = max(0, min(gx1, rec["W"] - 1) - max(gx0, 0) + 1)
            cy = max(0, min(gy1, rec["H"] - 1) - max(gy0, 0) + 1)
            area = cx * cy / float((x1 - x0 + 1) * (y1 - y0 + 1))
            out[s, k] = [np.log1p(same.sum()), np.log1p(opp.sum()), np.log1p(q[same].sum() / S.Q_SCALE),
                         np.log1p(q[opp].sum() / S.Q_SCALE), area]
    return out, ok


def maps_of(rec):
    return S.EvidenceMaps(rec["x"], rec["y"], rec["win"], rec["pol"], rec["z0"], rec["n_win"], rec["H"], rec["W"],
                          "cpu")


class LayoutTests(unittest.TestCase):
    def test_tiling(self):
        for L in (1, 2, 3, 4):
            rect, r, lev = S.box_layout(L)
            self.assertEqual(len(rect), 1 + 8 * L)
            h = (3 ** (L + 1) - 1) // 2
            cover = np.zeros((2 * h + 1, 2 * h + 1), int)
            for x0, x1, y0, y1 in rect:
                cover[y0 + h:y1 + h + 1, x0 + h:x1 + h + 1] += 1
            self.assertTrue((cover == 1).all(), "level %d" % L)
            x0, x1, y0, y1 = rect[S.CENTRE]
            self.assertTrue(x0 <= 0 <= x1 and y0 <= 0 <= y1)
            self.assertEqual(r[S.CENTRE], 0.0)
            self.assertEqual(lev.max(), L - 1)


class MapTests(unittest.TestCase):
    def setUp(self):
        self.rec = recording()
        self.rect, _, _ = S.box_layout(2)
        self.maps = maps_of(self.rec)

    def features(self, rec, maps, idx, n_steps=3):
        return maps.features(rec["x"][idx], rec["y"][idx], rec["win"][idx], rec["pol"][idx], rec["z0"][idx],
                             self.rect, n_steps)

    def test_brute_force(self):
        idx = np.arange(0, 400, 7)
        feat, ok = self.features(self.rec, self.maps, idx)
        for j, e in enumerate(idx):
            ref, ref_ok = brute(self.rec, e, self.rect, 3)
            np.testing.assert_array_equal(ok[j].numpy(), ref_ok)
            np.testing.assert_allclose(feat[j].numpy(), ref * ref_ok[:, None, None], rtol=1e-5, atol=1e-5)

    def test_polarity_relative(self):
        flipped = dict(self.rec, pol=1 - self.rec["pol"])
        idx = np.arange(0, 400, 5)
        a, _ = self.features(self.rec, self.maps, idx)
        b, _ = self.features(flipped, maps_of(flipped), idx)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_mask_past_end(self):
        last = np.nonzero(self.rec["win"] == self.rec["n_win"] - 1)[0][:5]
        feat, ok = self.features(self.rec, self.maps, last)
        self.assertTrue(ok[:, 0].all())
        self.assertFalse(ok[:, 1:].any())
        self.assertEqual(float(feat[:, 1:].abs().sum()), 0.0)


def unit(value="ei", gate="attribution", pool="window", evidence="boxes", context=True, audit="reaudit", n_steps=4,
         levels=2, nonzero=True, tokens="mlp", ctx_ff=True, n_feat=S.N_FEAT, seed=1):
    torch.manual_seed(seed)
    _, r, _ = S.box_layout(levels)
    net = S.SpikingDecisionUnits(n_steps, r, dim=16, heads=4, value=value, gate=gate, pool=pool, evidence=evidence,
                                 context=context, audit=audit, tokens=tokens, ctx_ff=ctx_ff, n_feat=n_feat)
    if nonzero:
        with torch.no_grad():
            if tokens == "glm":
                net.coef.normal_(0, 0.5)
                net.bias.normal_(0, 0.5)
                net.dcoef.weight.normal_(0, 0.3)
            else:
                net.v_out.weight.normal_(0, 0.5)
                net.v_out.bias.normal_(0, 0.5)
            net.wm.weight.normal_(0, 0.3)
            if evidence == "events":
                net.sink_val.weight.normal_(0, 0.5)
                net.sink_logit.normal_(0, 1.0)
    return net.eval()


def unit_inputs(B=5, n_steps=4, K=17, seed=0):
    g = torch.Generator().manual_seed(seed)
    feat = torch.rand(B, n_steps, K, S.N_FEAT, generator=g) * 3.0
    valid = torch.ones(B, n_steps, dtype=torch.bool)
    z0 = torch.randn(B, generator=g) * 3.0
    pol = (torch.rand(B, generator=g) > 0.5).long()
    return feat, valid, z0, pol


class UnitTests(unittest.TestCase):
    def test_starts_at_prior(self):
        feat, valid, z0, pol = unit_inputs()
        for tokens in S.TOKENS:
            for value in ("ei", "signed"):
                for audit in ("reaudit", "ledger"):
                    for pool, evidence in (("window", "boxes"), ("mean", "boxes"), ("window", "events")):
                        for gate in ("attribution", "distance"):
                            net = unit(value=value, audit=audit, pool=pool, evidence=evidence, nonzero=False,
                                       tokens=tokens, gate=gate)
                            with torch.no_grad():
                                U = net(feat, valid, z0, pol)
                            self.assertLess(float((U - z0[:, None]).abs().max()), 1e-5,
                                            "%s %s %s %s %s %s" % (tokens, value, audit, pool, evidence, gate))

    def test_windows_add_up(self):
        feat, valid, z0, pol = unit_inputs()
        for tokens in S.TOKENS:
            for evidence in ("boxes", "events"):                # fixed gates and context: each window's evidence is
                for audit in ("reaudit", "ledger"):             # the same at every step
                    net = unit(context=False, gate="distance", evidence=evidence, tokens=tokens, audit=audit)
                    with torch.no_grad():
                        U = net(feat, valid, z0, pol)
                        f2 = feat.clone()
                        f2[:, :2] = torch.rand_like(f2[:, :2]) * 5.0     # change windows 0 and 1 only
                        U2 = net(f2, valid, z0, pol)
                    np.testing.assert_allclose((U2[:, 3] - U2[:, 2]).numpy(), (U[:, 3] - U[:, 2]).numpy(),
                                               rtol=1e-4, atol=1e-4)
                    self.assertGreater(float((U2[:, 1] - U[:, 1]).abs().max()), 1e-3)

    def test_causal(self):
        feat, valid, z0, pol = unit_inputs()
        for tokens in S.TOKENS:
            for audit in ("reaudit", "ledger"):
                net = unit(audit=audit, tokens=tokens)
                with torch.no_grad():
                    U = net(feat, valid, z0, pol)
                    for k in range(3):
                        f2 = feat.clone()
                        f2[:, k + 1:] = torch.rand_like(f2[:, k + 1:]) * 5.0
                        U2 = net(f2, valid, z0, pol)
                        np.testing.assert_allclose(U2[:, :k + 1].numpy(), U[:, :k + 1].numpy(), rtol=1e-6, atol=1e-6)
                        self.assertGreater(float((U2[:, k + 1:] - U[:, k + 1:]).abs().max()), 1e-4)

    def test_ledger_context_as_reaudit(self):
        """Ledger units compute their context from running sums exactly as re-audit units do from one audit of the
        same windows: when the reading does not depend on the context (glm without context shift, distance gate),
        both give the same membrane and context at every step (only re-reading old windows differs)."""
        feat, valid, z0, pol = unit_inputs()
        nets = {}
        for audit in ("reaudit", "ledger"):
            net = unit(tokens="glm", gate="distance", evidence="events", audit=audit, ctx_ff=False)
            with torch.no_grad():
                net.dcoef.weight.zero_()
                net.dcoef.bias.zero_()
                net.sink_val.weight.zero_()
                net.wo.weight.normal_(0, 0.3)
            nets[audit] = net
        nets["ledger"].load_state_dict(nets["reaudit"].state_dict())
        K = nets["ledger"].n_box
        trace = {}
        with torch.no_grad():
            for audit, net in nets.items():
                tok, c0 = net.embed(feat), net.start(z0, pol)
                c, U = c0, z0
                sums = net.start_sums(len(z0), "cpu") if audit == "ledger" else None
                out = []
                for k in range(4):
                    sl, nw = (slice(0, (k + 1) * K), k + 1) if audit == "reaudit" else (slice(k * K, (k + 1) * K), 1)
                    U, c, _, _, _, sums = net.step(c, c0, U, z0, k, {n: t[:, sl] for n, t in tok.items()}, nw,
                                                   valid[:, k], sums=sums)
                    out.append(torch.cat([U[:, None], c], 1))
                trace[audit] = torch.stack(out, 1)
        self.assertGreater(float(trace["reaudit"][:, 1:, 1:].std()), 1e-2)       # the context does change
        np.testing.assert_allclose(trace["ledger"].numpy(), trace["reaudit"].numpy(), rtol=1e-4, atol=1e-4)

    def test_empty_boxes_cost_nothing(self):
        """Events as observations: a box without events has weight 0, so what it holds otherwise (area) changes no
        membrane, and the event-driven cost equals the dense one when every box holds events."""
        feat, valid, z0, pol = unit_inputs()
        empty = torch.rand(feat.shape[:3]) < 0.4
        feat[..., 0][empty] = 0.0
        feat[..., 1][empty] = 0.0
        feat[..., 2][empty] = 0.0
        feat[..., 3][empty] = 0.0
        for tokens in S.TOKENS:
            for gate in ("attribution", "distance"):
                net = unit(evidence="events", tokens=tokens, gate=gate)
                with torch.no_grad():
                    U = net(feat, valid, z0, pol)
                    f2 = feat.clone()
                    f2[..., 4][empty] = torch.rand(int(empty.sum()))
                    U2 = net(f2, valid, z0, pol)
                np.testing.assert_allclose(U2.numpy(), U.numpy(), rtol=1e-5, atol=1e-5, err_msg=tokens + gate)
                K = net.n_box
                k = np.arange(4)
                np.testing.assert_allclose(net.step_macs(k, np.full(4, K), (k + 1) * K),
                                           [net.step_macs(int(i)) for i in k])
                self.assertTrue((net.step_macs(k, np.full(4, 3), 3 * (k + 1)) < net.step_macs(k)).all())

    def test_frozen_after_last_valid(self):
        feat, valid, z0, pol = unit_inputs()
        valid[:, 2:] = False
        for kw in ({}, {"audit": "ledger", "tokens": "glm", "gate": "distance", "evidence": "events"}):
            net = unit(**kw)
            with torch.no_grad():
                U = net(feat, valid, z0, pol)
            np.testing.assert_array_equal(U[:, 2].numpy(), U[:, 1].numpy())
            np.testing.assert_array_equal(U[:, 3].numpy(), U[:, 1].numpy())

    def test_opponent_signs(self):
        feat, valid, z0, pol = unit_inputs()
        net = unit(value="ei")
        with torch.no_grad():
            net(feat, valid, z0, pol, keep=True)
        lam = net.extras["lam"]
        self.assertTrue(bool((lam[..., :2] >= 0).all()) and bool((lam[..., 2:] <= 0).all()))
        self.assertEqual(tuple(net.extras["gates"].shape), (5, 4, 4, 4 * 17))

    def test_ablations_run(self):
        feat, valid, z0, pol = unit_inputs()
        for kw in ({"gate": "distance"}, {"pool": "sum"}, {"pool": "mean"}, {"context": False},
                   {"value": "signed", "pool": "sum"}, {"evidence": "events"}, {"audit": "ledger", "evidence": "events"},
                   {"tokens": "linear", "gate": "distance", "ctx_ff": False, "evidence": "events"},
                   {"tokens": "glm", "gate": "distance", "ctx_ff": False, "evidence": "events"},
                   {"tokens": "glm", "value": "signed"}, {"tokens": "mlp", "ctx_ff": False}):
            net = unit(**kw)
            with torch.no_grad():
                U = net(feat, valid, z0, pol)
            self.assertTrue(bool(torch.isfinite(U).all()), str(kw))

    def test_step_cost_grows(self):
        net = unit()
        c = [net.step_macs(k) for k in range(4)]
        self.assertTrue(all(b > a for a, b in zip(c[1:], c[2:])))
        self.assertGreater(c[0], c[1] - (net.step_macs(2) - net.step_macs(1)))


class StreamingTests(unittest.TestCase):
    def test_stream_equals_offline(self):
        for evidence in ("boxes", "events"):
            self.check_stream(evidence)
        for tokens, gate in (("linear", "distance"), ("glm", "distance"), ("glm", "attribution")):
            self.check_stream("events", tokens=tokens, gate=gate, ctx_ff=False)
        for kw in ({"tokens": "glm", "gate": "distance", "ctx_ff": False}, {"tokens": "glm", "context": False},
                   {"tokens": "mlp"}):                          # ledger units: membrane + context only
            self.check_stream("events", audit="ledger", **kw)

    def check_stream(self, evidence, **kw):
        rec = recording(n=600, W=40, H=30, n_win=7, seed=3)
        rect, r, _ = S.box_layout(2)
        n_steps = 4
        net = unit(n_steps=n_steps, evidence=evidence, **kw)
        maps = maps_of(rec)
        idx = np.arange(len(rec["x"]))
        F_ = np.minimum(n_steps - 1, rec["n_win"] - 1 - rec["win"])
        with torch.no_grad():
            feat, ok = maps.features(rec["x"], rec["y"], rec["win"], rec["pol"], rec["z0"], rect, n_steps)
            U = net(feat, ok, torch.as_tensor(rec["z0"]), torch.as_tensor(rec["pol"])).numpy()
        tau = -1.2
        Z = np.concatenate([rec["z0"][:, None], U], 1).astype(np.float64)
        bounds = [2.5, 2.0, 1.5, 1.0, 0.0]
        L = P.commit(Z, 1 + F_, tau, bounds)
        layer = S.StreamingDecisionLayer(net, rect, bounds, tau, rec["H"], rec["W"], "cpu")
        col = np.full(len(idx), -1)
        val = np.zeros(len(idx))
        for t in range(rec["n_win"]):
            ids = idx[rec["win"] == t]
            e, c, v = layer.window(t, rec["x"][ids], rec["y"][ids], rec["pol"][ids], rec["z0"][ids], ids, F_[ids])
            col[e], val[e] = c, v
        self.assertEqual(layer.open_units(), 0)
        np.testing.assert_array_equal(col, L)
        np.testing.assert_allclose(val, Z[np.arange(len(idx)), L], rtol=1e-5, atol=1e-5)
        self.assertTrue(0 < (L == 0).sum() < len(L) and (L > 1).any())


def first_population(rec, net1, rect, n_steps, tau, bounds1):
    """Offline decision spikes of a first population: own step / type per event and the spikes (x, y, window, type)."""
    with torch.no_grad():
        f1, ok1 = maps_of(rec).features(rec["x"], rec["y"], rec["win"], rec["pol"], rec["z0"], rect, n_steps)
        U1 = net1(f1, ok1, torch.as_tensor(rec["z0"]), torch.as_tensor(rec["pol"])).numpy()
    F_ = np.minimum(n_steps - 1, rec["n_win"] - 1 - rec["win"])
    Z1 = np.concatenate([rec["z0"][:, None], U1], 1).astype(np.float64)
    L1 = P.commit(Z1, 1 + F_, tau, bounds1)
    typ = (Z1[np.arange(len(L1)), L1] >= tau).astype(np.int64)
    step = np.where(L1 >= 1, L1 - 1, -1)
    sel = np.nonzero(L1 >= 1)[0]
    return step, typ, (rec["x"][sel], rec["y"][sel], rec["win"][sel] + step[sel], typ[sel]), F_


class V18Tests(unittest.TestCase):
    """Decision spikes as messages (two populations), the ledger mean ablation, the martingale term."""

    def test_message_features(self):
        rec = recording(n=400, W=40, H=30, n_win=5, seed=4)
        rng = np.random.RandomState(5)
        mi = rng.choice(400, 80, replace=False)
        mstep, mtype = rng.randint(0, 3, 80), rng.randint(0, 2, 80)
        mwin = rec["win"][mi] + mstep
        keep = mwin < rec["n_win"]
        mi, mstep, mtype, mwin = mi[keep], mstep[keep], mtype[keep], mwin[keep]
        own_step, own_type = np.full(400, -1), np.zeros(400, np.int64)
        own_step[mi], own_type[mi] = mstep, mtype
        maps = S.EvidenceMaps(rec["x"], rec["y"], rec["win"], rec["pol"], rec["z0"], rec["n_win"], rec["H"], rec["W"],
                              "cpu", messages=(rec["x"][mi], rec["y"][mi], mwin, mtype))
        rect, _, _ = S.box_layout(2)
        idx = np.arange(0, 400, 3)
        args = (rec["x"][idx], rec["y"][idx], rec["win"][idx], rec["pol"][idx], rec["z0"][idx], rect, 3)
        feat, ok = maps.features(*args, own_msg=(own_step[idx], own_type[idx]))
        base, _ = maps_of(rec).features(*args)
        self.assertEqual(feat.shape[-1], S.N_FEAT + S.N_MSG)
        np.testing.assert_array_equal(feat[..., :S.N_FEAT].numpy(), base.numpy())
        for j, e in enumerate(idx):
            for s in range(3):
                w = rec["win"][e] + s
                if w >= rec["n_win"]:
                    continue
                for k, (x0, x1, y0, y1) in enumerate(rect):
                    inside = ((mwin == w) & (mi != e) & (rec["x"][mi] >= rec["x"][e] + x0) &
                              (rec["x"][mi] <= rec["x"][e] + x1) & (rec["y"][mi] >= rec["y"][e] + y0) &
                              (rec["y"][mi] <= rec["y"][e] + y1))
                    self.assertAlmostEqual(float(feat[j, s, k, 5]), np.log1p((inside & (mtype == 1)).sum()), 5)
                    self.assertAlmostEqual(float(feat[j, s, k, 6]), np.log1p((inside & (mtype == 0)).sum()), 5)

    def test_second_population_starts_at_prior(self):
        g = torch.Generator().manual_seed(2)
        feat = torch.rand(5, 4, 17, S.N_FEAT + S.N_MSG, generator=g) * 3.0
        z0 = torch.randn(5, generator=g) * 3.0
        pol = (torch.rand(5, generator=g) > 0.5).long()
        net = unit(tokens="glm", gate="distance", evidence="events", audit="ledger", ctx_ff=False,
                   n_feat=S.N_FEAT + S.N_MSG, nonzero=False)
        with torch.no_grad():
            U = net(feat, torch.ones(5, 4, dtype=torch.bool), z0, pol)
        self.assertLess(float((U - z0[:, None]).abs().max()), 1e-5)

    def test_ledger_mean_ablation(self):
        feat, valid, z0, pol = unit_inputs()
        net = unit(tokens="glm", gate="distance", evidence="events", audit="ledger", ctx_ff=False, context=False,
                   pool="mean")
        with torch.no_grad():
            U = net(feat, valid, z0, pol, keep=True)
        cur = net.extras["lam"].sum(-1)                             # signed current of each window
        mean = cur.cumsum(1) / torch.arange(1, 5, dtype=cur.dtype)[None]
        np.testing.assert_allclose(U.numpy(), (z0[:, None] + mean).numpy(), rtol=1e-5, atol=1e-5)

    def test_martingale_penalty(self):
        vm, w = torch.ones(64, 4), torch.ones(64)
        flat = torch.randn(64, 1).expand(64, 4)
        self.assertLess(float(S.martingale_penalty(flat, vm, w)), 1e-12)
        self.assertGreater(float(S.martingale_penalty(flat + torch.arange(4.0)[None], vm, w)), 1e-3)

    def test_cooperative_stream_equals_offline(self):
        rec = recording(n=700, W=40, H=30, n_win=8, seed=6)
        rect, _, _ = S.box_layout(2)
        n_steps, tau = 4, -1.2
        bounds1, bounds2 = [2.5, 2.0, 1.5, 1.0, 0.0], [2.2, 1.8, 1.4, 0.9, 0.0]
        kw = dict(n_steps=n_steps, tokens="glm", gate="distance", evidence="events", audit="ledger", ctx_ff=False)
        net1 = unit(**kw)
        net2 = unit(n_feat=S.N_FEAT + S.N_MSG, seed=2, **kw)
        step, typ, spikes, F_ = first_population(rec, net1, rect, n_steps, tau, bounds1)
        maps2 = S.EvidenceMaps(rec["x"], rec["y"], rec["win"], rec["pol"], rec["z0"], rec["n_win"], rec["H"],
                               rec["W"], "cpu", messages=spikes)
        with torch.no_grad():
            f2, ok2 = maps2.features(rec["x"], rec["y"], rec["win"], rec["pol"], rec["z0"], rect, n_steps,
                                     own_msg=(step, typ))
            U2 = net2(f2, ok2, torch.as_tensor(rec["z0"]), torch.as_tensor(rec["pol"])).numpy()
            f0 = f2.clone()
            f0[..., S.N_FEAT:] = 0.0
            U0 = net2(f0, ok2, torch.as_tensor(rec["z0"]), torch.as_tensor(rec["pol"])).numpy()
        self.assertGreater(len(spikes[0]), 20)
        self.assertGreater(float(np.abs(U2 - U0).max()), 1e-3)        # the spikes do change the second population
        Z2 = np.concatenate([rec["z0"][:, None], U2], 1).astype(np.float64)
        L2 = P.commit(Z2, 1 + F_, tau, bounds2)
        layer = S.CooperativeStreamingLayer(
            S.StreamingDecisionLayer(net1, rect, bounds1, tau, rec["H"], rec["W"], "cpu"),
            S.StreamingDecisionLayer(net2, rect, bounds2, tau, rec["H"], rec["W"], "cpu"))
        layer.reset(rec["x"], rec["y"])
        idx = np.arange(len(rec["x"]))
        col, val = np.full(len(idx), -1), np.zeros(len(idx))
        for t in range(rec["n_win"]):
            ids = idx[rec["win"] == t]
            e, c, v = layer.window(t, rec["x"][ids], rec["y"][ids], rec["pol"][ids], rec["z0"][ids], ids, F_[ids])
            col[e], val[e] = c, v
        self.assertEqual(layer.open_units(), 0)
        np.testing.assert_array_equal(col, L2)
        np.testing.assert_allclose(val, Z2[np.arange(len(idx)), L2], rtol=1e-5, atol=1e-5)


class StoppingTests(unittest.TestCase):
    def test_bound_where_changes_stop(self):
        rng = np.random.RandomState(0)
        m = rng.rand(20000) * 10.0
        change = (m < 3.0) & (rng.rand(20000) < 0.4)
        b = P.bound_from_changes(m, change, 0.01, 0.01)
        self.assertGreater(b, 2.95)                          # just after the last change below 3
        self.assertLess(b, 3.5)
        self.assertEqual(P.bound_from_changes(m, np.ones_like(change), 0.01, 0.01), float("inf"))

    def test_calibrate_and_commit(self):
        rng = np.random.RandomState(1)
        n, C, tau = 6000, 5, 0.0
        z0 = rng.randn(n) * 4.0
        drift = np.sign(rng.randn(n)) * 1.5
        Z = np.stack([z0 + drift * c * (np.abs(z0) < 2) for c in range(C)], 1)
        last = np.full(n, C - 1)
        last[:300] = 2
        b = P.calibrate(Z, last, tau, 0.01, 0.1)
        self.assertEqual(len(b), C)
        self.assertEqual(b[-1], 0.0)
        L = P.commit(Z, last, tau, b)
        self.assertTrue((L <= last).all())
        self.assertTrue((L[np.abs(z0) > b[0]] == 0).all())
        final = Z[np.arange(n), last] >= tau
        changed = (Z[np.arange(n), L] >= tau) != final
        self.assertLess(changed.mean(), 0.01)

    def test_unit_load(self):
        win = np.array([0, 0, 1, 3])
        L = np.array([0, 2, 3, 1])                  # columns: 0 = entry commit, k >= 1 ran steps 0 .. k - 1
        rec = np.zeros(4, np.int64)
        units, macs = P.unit_load(win, L, {0: 5}, rec, 3, lambda k: 10.0 * (k + 1))
        np.testing.assert_array_equal(units, [1, 2, 1, 2, 0])
        np.testing.assert_allclose(macs, [10, 30, 20, 40, 0])
        per_event = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        _, (a, b) = P.unit_load(win, L, {0: 5}, rec, 3, [per_event, lambda k: 1.0])
        np.testing.assert_allclose(a, [4, 5 + 7, 8, 9 + 10, 0])         # event 1: steps 0, 1; event 2: 0..2; event 3: 0
        np.testing.assert_allclose(b, units)


if __name__ == "__main__":
    unittest.main()
