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
    5. V18: decision spikes as messages (box counts, own spike left out, lateral synapses, cooperative stream), the
       ledger mean ablation, the martingale term; tracked fields (brute-force tracker and counts, the field follows a
       moving target, the own spike is removed from the box that holds the event, stream == offline)
"""
import math
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


def maps_of(rec, track=False, messages=None):
    return S.EvidenceMaps(rec["x"], rec["y"], rec["win"], rec["pol"], rec["z0"], rec["n_win"], rec["H"], rec["W"],
                          "cpu", messages=messages, track=track)


def moving_recording(n_noise=500, W=60, H=40, n_win=8, v=(4.0, 2.0), per_win=25, seed=0):
    """Uniform noise (low priors) plus a compact target (high priors) that moves v px per window."""
    rng = np.random.RandomState(seed)
    xs, ys, ws = [rng.randint(0, W, n_noise)], [rng.randint(0, H, n_noise)], [rng.randint(0, n_win, n_noise)]
    z = [rng.randn(n_noise) * 3.0 - 8.0]
    for t in range(n_win):
        cx, cy = 12.0 + v[0] * t, 10.0 + v[1] * t
        xs.append(np.clip(np.round(cx + rng.randn(per_win) * 1.2), 0, W - 1).astype(np.int64))
        ys.append(np.clip(np.round(cy + rng.randn(per_win) * 1.2), 0, H - 1).astype(np.int64))
        ws.append(np.full(per_win, t))
        z.append(rng.randn(per_win) + 3.0)
    n = n_noise + n_win * per_win
    return {"x": np.concatenate(xs), "y": np.concatenate(ys), "win": np.concatenate(ws), "pol": rng.randint(0, 2, n),
            "z0": np.concatenate(z).astype(np.float32), "W": W, "H": H, "n_win": n_win,
            "target": np.arange(n) >= n_noise, "v": v}


def brute_track(rec, e, rect, n_steps, spikes=None, mode="recentre"):
    """Moving-field features of event e: a numpy alpha-beta tracker on exact centroids and direct counting.
    spikes = (index of the spiking event, window, type) of first-population decision spikes (own spike excluded);
    mode = recentre (the field on the predicted source centre) or transport (on the event moved with its source)."""
    q = np.round(1.0 / (1.0 + np.exp(-rec["z0"].astype(np.float32).astype(np.float64))) * S.Q_SCALE)
    px, py, vx, vy = float(rec["x"][e]), float(rec["y"][e]), 0.0, 0.0
    ox = oy = 0.0
    sh = 1.0 if mode == "transport" else 0.0
    nf = S.N_FEAT + (0 if spikes is None else S.N_MSG)
    out, ok, centres = np.zeros((n_steps, len(rect), nf)), np.zeros(n_steps, bool), []
    for s in range(n_steps):
        w = rec["win"][e] + s
        cx = int(np.clip(np.round(px + vx + sh * ox), 0, rec["W"] - 1))           # where the field reads
        cy = int(np.clip(np.round(py + vy + sh * oy), 0, rec["H"] - 1))
        sx = int(np.clip(np.round(px + vx), 0, rec["W"] - 1))                     # where the tracker measures
        sy = int(np.clip(np.round(py + vy), 0, rec["H"] - 1))
        centres.append((cx, cy))
        has = False
        if w < rec["n_win"]:
            ok[s] = True
            for k, (x0, x1, y0, y1) in enumerate(rect):
                gx0, gx1, gy0, gy1 = cx + x0, cx + x1, cy + y0, cy + y1
                inside = ((rec["win"] == w) & (rec["x"] >= gx0) & (rec["x"] <= gx1) & (rec["y"] >= gy0) &
                          (rec["y"] <= gy1))
                if s == 0:
                    inside[e] = False
                same = inside & (rec["pol"] == rec["pol"][e])
                opp = inside & (rec["pol"] != rec["pol"][e])
                ax = max(0, min(gx1, rec["W"] - 1) - max(gx0, 0) + 1)
                ay = max(0, min(gy1, rec["H"] - 1) - max(gy0, 0) + 1)
                area = ax * ay / float((x1 - x0 + 1) * (y1 - y0 + 1))
                row = [np.log1p(same.sum()), np.log1p(opp.sum()), np.log1p(q[same].sum() / S.Q_SCALE),
                       np.log1p(q[opp].sum() / S.Q_SCALE), area]
                if spikes is not None:
                    mi, mw, mt = spikes
                    hit = ((mw == w) & (mi != e) & (rec["x"][mi] >= gx0) & (rec["x"][mi] <= gx1) &
                           (rec["y"][mi] >= gy0) & (rec["y"][mi] <= gy1))
                    row += [np.log1p((hit & (mt == 1)).sum()), np.log1p((hit & (mt == 0)).sum())]
                out[s, k] = row
            sq = ((rec["win"] == w) & (np.abs(rec["x"] - sx) <= S.TRACK_HALF) &
                  (np.abs(rec["y"] - sy) <= S.TRACK_HALF))
            if s == 0:
                sq[e] = False                                  # the own event is not its own source
            mass = q[sq].sum()
            has = mass >= S.TRACK_MIN_Q
        if has:
            mx, my = (q[sq] * rec["x"][sq]).sum() / mass, (q[sq] * rec["y"][sq]).sum() / mass
            if s > 0:
                vx, vy = mx - px, my - py
            else:
                ox, oy = rec["x"][e] - mx, rec["y"][e] - my
            px, py = mx, my
        else:
            px, py = px + vx, py + vy
    return out, ok, centres


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


def membership_ref(rec, e):
    """Membership cue of event e by direct counting: log(1 + distance from the prior-weighted centroid of the other
    events of its +-13 px square in its own window), and whether there is prior mass at all."""
    q = np.round(1.0 / (1.0 + np.exp(-rec["z0"].astype(np.float32).astype(np.float64))) * S.Q_SCALE)
    sq = ((rec["win"] == rec["win"][e]) & (np.abs(rec["x"] - rec["x"][e]) <= S.TRACK_HALF) &
          (np.abs(rec["y"] - rec["y"][e]) <= S.TRACK_HALF))
    sq[e] = False
    m = q[sq].sum()
    if m < S.TRACK_MIN_Q:
        return np.array([0.0, 0.0])
    cx, cy = (q[sq] * rec["x"][sq]).sum() / m, (q[sq] * rec["y"][sq]).sum() / m
    return np.array([np.log1p(np.hypot(cx - rec["x"][e], cy - rec["y"][e])), 1.0])


def unit(value="ei", gate="attribution", pool="window", evidence="boxes", context=True, audit="reaudit", n_steps=4,
         levels=2, nonzero=True, tokens="mlp", ctx_ff=True, n_feat=S.N_FEAT, seed=1, n_msg=0, track=False,
         memb=False, pseudo=True, dendrite="mean", ring=False, clutter=False, content="box"):
    torch.manual_seed(seed)
    _, r, _ = S.box_layout(levels)
    net = S.SpikingDecisionUnits(n_steps, r, dim=16, heads=4, value=value, gate=gate, pool=pool, evidence=evidence,
                                 context=context, audit=audit, tokens=tokens, ctx_ff=ctx_ff, n_feat=n_feat,
                                 n_msg=n_msg, track=track, memb=memb, pseudo=pseudo, dendrite=dendrite, ring=ring,
                                 clutter=clutter, content=content)
    if nonzero:
        with torch.no_grad():
            if ring:
                net.ring_v.normal_(0.5, 0.5)
                net.ring_w.normal_(0.0, 0.5)
            if clutter:
                net.clutter_k.normal_(0.5, 0.3)
            if dendrite == "existence":
                net.miss_logit.normal_(0, 1.0)
                net.miss_ctx.weight.normal_(0, 0.3)
            if n_msg:
                net.lat.normal_(0, 0.5)
            if tokens == "glm":
                net.coef.normal_(0, 0.5)
                net.bias.normal_(0, 0.5)
                net.dcoef.weight.normal_(0, 0.3)
            else:
                net.v_out.weight.normal_(0, 0.5)
                net.v_out.bias.normal_(0, 0.5)
            net.wm.weight.normal_(0, 0.3)
            if net.pseudo:
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
                   {"tokens": "glm", "value": "signed"}, {"tokens": "mlp", "ctx_ff": False},
                   {"tokens": "glm", "gate": "none", "ctx_ff": False, "evidence": "events", "audit": "ledger"}):
            net = unit(**kw)
            with torch.no_grad():
                U = net(feat, valid, z0, pol)
            self.assertTrue(bool(torch.isfinite(U).all()), str(kw))

    def test_no_attribution(self):
        """gate none (V18 ablation): every box of every window weighs the same, whatever its distance; starts at the
        prior like the other forms."""
        feat, valid, z0, pol = unit_inputs()
        for audit in ("ledger", "reaudit"):
            net = unit(tokens="glm", gate="none", evidence="events", audit=audit, ctx_ff=False)
            with torch.no_grad():
                net(feat, valid, z0, pol, keep=True)
                g = net.extras["gates"]                                  # [B, step, H, all tokens]
                k = 3
                read = g[:, k, :, :(k + 1) * 17] if audit == "reaudit" else g[:, k, :, k * 17:(k + 1) * 17]
                np.testing.assert_allclose(read.numpy(), torch.sigmoid(net.rf_bias)[None, :, None].expand_as(read)
                                           .numpy(), rtol=1e-6, atol=1e-6)
                U = unit(tokens="glm", gate="none", evidence="events", audit=audit, ctx_ff=False, nonzero=False)(
                    feat, valid, z0, pol)
            self.assertLess(float((U - z0[:, None]).abs().max()), 1e-5)

    def test_no_pseudo_observation(self):
        """pseudo=False (V18 ablation): no leak conductance; a window without events in the unit's boxes then adds
        nothing (with it, the pseudo-observation still moves the membrane); starts at the prior; costs d H less."""
        feat, valid, z0, pol = unit_inputs()
        feat[:, 2, :, :2] = 0.0                                        # window 2: no events in any box
        kw = dict(tokens="glm", gate="distance", evidence="events", audit="ledger", ctx_ff=False)
        with torch.no_grad():
            U_off = unit(pseudo=False, **kw)(feat, valid, z0, pol)
            U_on = unit(pseudo=True, **kw)(feat, valid, z0, pol)
            U0 = unit(pseudo=False, nonzero=False, **kw)(feat, valid, z0, pol)
        self.assertLess(float((U_off[:, 2] - U_off[:, 1]).abs().max()), 1e-5)
        self.assertGreater(float((U_on[:, 2] - U_on[:, 1]).abs().max()), 1e-3)
        self.assertLess(float((U0 - z0[:, None]).abs().max()), 1e-5)
        self.assertFalse(hasattr(unit(pseudo=False, **kw), "sink_logit"))
        on, off = unit(pseudo=True, **kw), unit(pseudo=False, **kw)
        self.assertEqual(on.step_macs(1, 0, 0) - off.step_macs(1, 0, 0), 16 * 4)

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

    def test_stream_equals_offline_tracked(self):
        for kw in ({"audit": "ledger", "tokens": "glm", "gate": "distance", "ctx_ff": False, "track": "transport",
                    "memb": True},
                   {"audit": "reaudit", "tokens": "glm", "gate": "distance", "ctx_ff": False, "track": "transport"},
                   {"audit": "ledger", "tokens": "glm", "gate": "distance", "ctx_ff": False},
                   {"audit": "reaudit", "tokens": "glm", "gate": "distance", "ctx_ff": False},
                   {"audit": "ledger", "tokens": "glm", "gate": "distance", "ctx_ff": False, "memb": True},
                   {"audit": "ledger", "tokens": "glm", "gate": "distance", "ctx_ff": False, "memb": True,
                    "track": False}):
            kw = dict({"track": True}, **kw)
            self.check_stream("events", rec=moving_recording(n_noise=500, n_win=9, seed=8), **kw)

    def check_stream(self, evidence, track=False, memb=False, rec=None, **kw):
        rec = recording(n=600, W=40, H=30, n_win=7, seed=3) if rec is None else rec
        rect, r, _ = S.box_layout(2)
        n_steps = 4
        net = unit(n_steps=n_steps, evidence=evidence, track=track, memb=memb, **kw)
        if memb:
            with torch.no_grad():                                   # the cue must change the start state
                net.c0[0].weight[:, 2:].normal_(0, 2.0)
        maps = maps_of(rec, track=bool(track) or memb)
        idx = np.arange(len(rec["x"]))
        F_ = np.minimum(n_steps - 1, rec["n_win"] - 1 - rec["win"])
        with torch.no_grad():
            feat, ok, cue = maps.features(rec["x"], rec["y"], rec["win"], rec["pol"], rec["z0"], rect, n_steps,
                                          follow=net.track_mode if net.track else False, return_memb=True)
            U = net(feat, ok, torch.as_tensor(rec["z0"]), torch.as_tensor(rec["pol"]), memb=cue).numpy()
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
        for audit in ("ledger", "reaudit"):
            net = unit(tokens="glm", gate="distance", evidence="events", audit=audit, ctx_ff=False,
                       n_msg=S.N_MSG, nonzero=False)
            with torch.no_grad():
                U = net(feat, torch.ones(5, 4, dtype=torch.bool), z0, pol)
            self.assertLess(float((U - z0[:, None]).abs().max()), 1e-5)

    def test_lateral_synapses(self):
        """Spikes of others reach the soma even in a box without events (their weight does not depend on the
        evidence of the box); with zero lateral weights the unit is exactly the V17 unit."""
        feat, valid, z0, pol = unit_inputs()
        spikes = torch.zeros(feat.shape[:3] + (S.N_MSG,))
        spikes[:, 1, 3, 0] = np.log1p(2.0)                          # two target spikes in box 3 at step 1
        feat[:, 1, 3, 0:2] = 0.0                                     # ... a box without events in that window
        kw = dict(tokens="glm", gate="distance", evidence="events", audit="ledger", ctx_ff=False)
        net2 = unit(n_msg=S.N_MSG, **kw)
        with torch.no_grad():
            U_with = net2(torch.cat([feat, spikes], -1), valid, z0, pol)
            U_none = net2(torch.cat([feat, torch.zeros_like(spikes)], -1), valid, z0, pol)
        diff = (U_with - U_none).numpy()
        np.testing.assert_allclose(diff[:, 0], 0.0, atol=1e-6)      # nothing before the spike arrives
        np.testing.assert_allclose(diff[:, 1:], float(net2.lat[1, 3, 0]) * np.log1p(2.0), rtol=1e-4, atol=1e-5)
        net1 = unit(**kw)
        net2.load_state_dict(net1.state_dict(), strict=False)          # same weights, lateral synapses aside
        with torch.no_grad():
            net2.lat.zero_()
            np.testing.assert_allclose(net2(torch.cat([feat, spikes], -1), valid, z0, pol).numpy(),
                                       net1(feat, valid, z0, pol).numpy(), rtol=1e-6, atol=1e-6)

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
        g = torch.Generator().manual_seed(3)
        vm, w = torch.ones(4000, 6), torch.ones(4000)
        start = torch.randn(4000, 1, generator=g) * 2.0
        flat = start.expand(4000, 6)
        self.assertLess(float(S.martingale_penalty(flat, vm, w)), 1e-12)
        drift = start - 0.3 * torch.arange(6.0)[None]                 # predictable downward revisions
        self.assertGreater(float(S.martingale_penalty(drift, vm, w)), 1e-4)
        # zero-mean revisions of the probability (a martingale): the noise of single revisions is not penalised
        P = 0.2 + 0.6 * torch.rand(4000, 1, generator=g)
        for _ in range(5):
            P = torch.cat([P, P[:, -1:] + 0.03 * (2 * (torch.rand(4000, 1, generator=g) > 0.5).float() - 1)], 1)
        noisy = torch.log(P / (1 - P))
        self.assertLess(float(S.martingale_penalty(noisy, vm, w)), 0.1 * float(S.martingale_penalty(drift, vm, w)))

    def test_cooperative_stream_equals_offline(self):
        """Both ways a unit hears decision spikes: lateral synapses on the soma (--spikes synapse) and spike counts as
        two more box statistics of the dendrite (--spikes feature, V18 first version)."""
        for second in ({"n_msg": S.N_MSG}, {"n_feat": S.N_FEAT + S.N_MSG}):
            self.check_cooperative(second)

    def check_cooperative(self, second, first=None):
        rec = recording(n=700, W=40, H=30, n_win=8, seed=6)
        rect, _, _ = S.box_layout(2)
        n_steps, tau = 4, -1.2
        bounds1, bounds2 = [2.5, 2.0, 1.5, 1.0, 0.0], [2.2, 1.8, 1.4, 0.9, 0.0]
        kw = dict(n_steps=n_steps, tokens="glm", gate="distance", evidence="events", audit="ledger", ctx_ff=False)
        net1 = unit(**dict(kw, **(first or {})))
        net2 = unit(seed=2, **dict(kw, **second))
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


EXIST = dict(tokens="glm", gate="distance", audit="ledger", ctx_ff=False, dendrite="existence")
V19 = dict(EXIST, ring=True, clutter=True, content="ring")


class V19Tests(unittest.TestCase):
    """V19 existence dendrite: detection-or-miss log-likelihood ratio, expanding association ring, divisive clutter
    normalisation, content coefficients shared around each ring."""

    def test_starts_at_prior(self):
        feat, valid, z0, pol = unit_inputs()
        for kw in ({}, {"ring": True}, {"ring": True, "clutter": True, "content": "ring"}):
            with torch.no_grad():
                U = unit(nonzero=False, evidence="events", **dict(EXIST, **kw))(feat, valid, z0, pol)
            self.assertLess(float((U - z0[:, None]).abs().max()), 1e-5, str(kw))

    def test_formula_one_window(self):
        """lambda_h = log(miss_h + sum_b n_b exp(prior + content - kappa_h log(1 + density x px))), the attributed
        mass is the detection probability D / (miss + D) and the gates are n_b exp(score) / (miss + D)."""
        net = unit(evidence="events", **V19)
        feat, valid, z0, pol = unit_inputs(B=3)
        Fn, H = S.N_FEAT, 4
        with torch.no_grad():
            tok = net.embed(feat[:, :1])
            c = net.start(z0, pol)
            lam, mass, _, resp = net.audit_tokens(c, 0, tok, 1)
            dc = net.dcoef(c).view(3, H, Fn + 1)
            content = tok["s"].permute(0, 2, 1) + torch.einsum("bhf,btf->bht", dc[..., :Fn], tok["x"]) + dc[..., Fn:]
            dens = tok["bg"].sum(-1) / tok["px"].sum(-1).clamp(min=1.0)
            score = (content + net.association_prior(0, 1)[None]
                     - torch.nn.functional.softplus(net.clutter_k)[None, :, None]
                     * torch.log1p(dens[:, None] * tok["px"])[:, None, :])
            n = tok["cnt"] * (tok["cnt"] > 0.5).float()
            D = (n[:, None, :] * torch.exp(score)).sum(-1)
            miss = torch.sigmoid(net.miss_logit[None] + net.miss_ctx(c))
        np.testing.assert_allclose(lam.numpy(), torch.log(miss + D).numpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(mass.numpy(), (D / (miss + D)).numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(resp.sum(-1).numpy(), mass.numpy(), rtol=1e-5, atol=1e-6)
        self.assertTrue(bool((resp >= 0).all()) and bool((resp.sum(-1) <= 1 + 1e-5).all()))

    def test_empty_window_is_evidence_of_absence(self):
        """A window without events in the unit's boxes adds sum_h sign_h log miss_h: when the target hypothesis
        expects detections (small miss), nothing seen pulls the membrane towards background."""
        feat, valid, z0, pol = unit_inputs()
        feat[:, 2, :, :2] = 0.0
        net = unit(evidence="events", **V19)
        with torch.no_grad():
            net.miss_ctx.weight.zero_()
            net.miss_ctx.bias.zero_()
            net.miss_logit.copy_(torch.tensor([-2.0, -1.0, 0.5, 1.0]))
            U = net(feat, valid, z0, pol)
            expect = float((torch.nn.functional.logsigmoid(net.miss_logit) * net.head_sign).sum())
        self.assertLess(expect, 0.0)
        np.testing.assert_allclose((U[:, 2] - U[:, 1]).numpy(), expect, rtol=1e-5, atol=1e-5)

    def test_ring_prior(self):
        """Without growth the ring is the fixed distance field; with speed v and no spread the prior of window offset
        k peaks at the box nearest to v k; per-unit ages give the same priors as one age at a time."""
        net = unit(evidence="events", **V19)
        with torch.no_grad():
            net.ring_v.fill_(-30.0)
            net.ring_w.fill_(-30.0)
            ref = net.rf_bias[:, None] - net.r2[None] / (2.0 * net.sigma()[:, None] ** 2)
            np.testing.assert_allclose(net.association_prior(5, 1).numpy(), ref.numpy(), rtol=1e-5, atol=1e-5)
            net.ring_v.fill_(math.log(math.expm1(3.0)))                   # 3 px per window
            net.log_sigma.fill_(math.log(1.5))
            for k, r_best in ((0, 0.0), (1, 3.0), (3, 9.0)):
                self.assertAlmostEqual(float(net.r[net.association_prior(k, 1)[0].argmax()]), r_best, places=4)
            per = net.association_prior(torch.tensor([0, 1, 3]), 1)
            one = torch.stack([net.association_prior(k, 1) for k in (0, 1, 3)])
            np.testing.assert_allclose(per.numpy(), one.numpy(), rtol=1e-6, atol=1e-6)
            many = net.association_prior(3, 4)                              # re-audit: windows 0 .. 3
            np.testing.assert_allclose(many.view(4, 4, -1)[:, 3].numpy(), one[2].numpy(), rtol=1e-6, atol=1e-6)

    def test_stream_equals_offline(self):
        for kw in (V19, dict(EXIST, ring=True), dict(EXIST, clutter=True), EXIST,
                   dict(EXIST, audit="reaudit", ring=True, clutter=True),
                   {"tokens": "glm", "gate": "distance", "audit": "ledger", "ctx_ff": False, "ring": True,
                    "clutter": True}):                          # the mean dendrite with ring and clutter (fallback)
            StreamingTests.check_stream(self, "events", **kw)

    def test_cooperative(self):
        """Both populations with the existence dendrite; the second reads the decision spikes as box statistics."""
        V18Tests.check_cooperative(self, dict(V19, n_feat=S.N_FEAT + S.N_MSG), first=V19)

    def test_cost_and_constants(self):
        net = unit(evidence="events", **V19)
        c = [net.step_macs(k, 5, 5) for k in range(4)]
        self.assertTrue(all(v > 0 for v in c))
        lc = net.learned_constants()
        for key in ("ring_speed_px_per_window", "ring_spread_px_per_window", "clutter_exponent", "miss_probability"):
            self.assertEqual(len(lc[key]), 4)
        self.assertFalse(hasattr(net, "sink_logit"))
        self.assertEqual(tuple(net.coef.shape[1:3]), (net.n_ring, 4))
        with self.assertRaises(ValueError):
            unit(evidence="boxes", **EXIST)


class TrackTests(unittest.TestCase):
    """V18 tracked fields: the boxes of a unit follow its source."""

    def test_brute_force(self):
        rec = moving_recording(seed=1)
        rect, _, _ = S.box_layout(2)
        idx = np.concatenate([np.nonzero(rec["target"])[0][::9], np.arange(0, 500, 23)])
        for mode in ("recentre", "transport"):
            feat, ok, state = maps_of(rec, track=True).features(rec["x"][idx], rec["y"][idx], rec["win"][idx],
                                                                rec["pol"][idx], rec["z0"][idx], rect, 5,
                                                                return_track=True, follow=mode)
            self.assertEqual(tuple(state.shape), (len(idx), 6))
            for j, e in enumerate(idx):
                ref, ref_ok, _ = brute_track(rec, e, rect, 5, mode=mode)
                np.testing.assert_array_equal(ok[j].numpy(), ref_ok)
                np.testing.assert_allclose(feat[j].numpy(), ref * ref_ok[:, None, None], rtol=1e-5, atol=1e-5,
                                           err_msg=mode)

    def test_transport_keeps_the_place_in_the_source(self):
        """An event 6 px beside a moving target: the transported field moves with the target and stays 6 px beside
        it (its centre box stays off the target); the re-centred field lands on the target."""
        rec = moving_recording(n_noise=200, seed=7, v=(4.0, 0.0))
        n = len(rec["x"])
        side = {k: np.append(rec[k], v) for k, v in (("x", 12 + 0), ("y", 10 + 7), ("win", 0), ("pol", 0))}
        side["z0"] = np.append(rec["z0"], np.float32(-2.0))
        side.update(W=rec["W"], H=rec["H"], n_win=rec["n_win"])
        rect, _, _ = S.box_layout(2)
        e = np.array([n])
        args = (side["x"][e], side["y"][e], side["win"][e], side["pol"][e], side["z0"][e], rect, 6)
        maps = maps_of(side, track=True)
        centre = {}
        for mode in ("recentre", "transport"):
            f, _ = maps.features(*args, follow=mode)
            centre[mode] = float((torch.expm1(f[0, 3:, S.CENTRE, 0]) + torch.expm1(f[0, 3:, S.CENTRE, 1])).mean())
        self.assertGreater(centre["recentre"], 2.0 * max(centre["transport"], 0.5))

    def test_follows_a_moving_target(self):
        """A target that moves 4 px per window leaves a fixed +-4 px square after two windows; the tracked field keeps
        it in its centre boxes."""
        rec = moving_recording(n_noise=300, seed=2)
        rect, _, _ = S.box_layout(2)
        e = np.nonzero(rec["target"] & (rec["win"] == 0))[0]
        args = (rec["x"][e], rec["y"][e], rec["win"][e], rec["pol"][e], rec["z0"][e], rect, 6)
        fixed, _ = maps_of(rec).features(*args)
        tracked, _ = maps_of(rec, track=True).features(*args)
        level0 = slice(0, 9)                                        # the +-4 px square
        n_fixed = (torch.expm1(fixed[:, 3:, level0, 0]) + torch.expm1(fixed[:, 3:, level0, 1])).sum(-1).mean()
        n_tracked = (torch.expm1(tracked[:, 3:, level0, 0]) + torch.expm1(tracked[:, 3:, level0, 1])).sum(-1).mean()
        self.assertLess(float(n_fixed), 2.0)
        self.assertGreater(float(n_tracked), 18.0)                  # of 25 target events per window

    def test_fixed_maps_unchanged(self):
        """Maps without tracking give the same features as before (V17), with and without returning the state."""
        rec = recording()
        rect, _, _ = S.box_layout(2)
        idx = np.arange(0, 400, 7)
        args = (rec["x"][idx], rec["y"][idx], rec["win"][idx], rec["pol"][idx], rec["z0"][idx], rect, 3)
        feat, ok, state = maps_of(rec).features(*args, return_track=True)
        self.assertIsNone(state)
        for j, e in enumerate(idx):
            ref, ref_ok = brute(rec, e, rect, 3)
            np.testing.assert_allclose(feat[j].numpy(), ref * ref_ok[:, None, None], rtol=1e-5, atol=1e-5)

    def test_own_spike_in_tracked_box(self):
        rec = moving_recording(seed=4)
        n = len(rec["x"])
        rng = np.random.RandomState(5)
        mi = rng.choice(n, 120, replace=False)
        mstep, mtype = rng.randint(0, 4, 120), rng.randint(0, 2, 120)
        mwin = rec["win"][mi] + mstep
        keep = mwin < rec["n_win"]
        mi, mstep, mtype, mwin = mi[keep], mstep[keep], mtype[keep], mwin[keep]
        own_step, own_type = np.full(n, -1), np.zeros(n, np.int64)
        own_step[mi], own_type[mi] = mstep, mtype
        rect, _, _ = S.box_layout(2)
        maps = maps_of(rec, track=True, messages=(rec["x"][mi], rec["y"][mi], mwin, mtype))
        idx = np.concatenate([mi[:40], np.nonzero(rec["target"])[0][::7]])
        for mode in ("recentre", "transport"):
            feat, _ = maps.features(rec["x"][idx], rec["y"][idx], rec["win"][idx], rec["pol"][idx], rec["z0"][idx],
                                    rect, 4, own_msg=(own_step[idx], own_type[idx]), follow=mode)
            for j, e in enumerate(idx):
                ref, ref_ok, _ = brute_track(rec, e, rect, 4, spikes=(mi, mwin, mtype), mode=mode)
                np.testing.assert_allclose(feat[j].numpy(), ref * ref_ok[:, None, None], rtol=1e-5, atol=1e-5)

    def test_tracker_cost(self):
        ne1 = np.array([[3, 5, 7]])
        np.testing.assert_array_equal(unit(track=True).track_macs(ne1), 2 * ne1 + S.TRACK_OPS)
        np.testing.assert_array_equal(unit(memb=True).track_macs(ne1), [[2 * 3 + S.TRACK_OPS, 0, 0]])
        np.testing.assert_array_equal(unit().track_macs(ne1), [[0, 0, 0]])

    def test_membership_cue(self):
        """The cue equals a direct computation; it is the same with fixed and tracked fields; units with the cue
        start at the prior; the cue changes the start state."""
        rec = moving_recording(n_noise=150, W=120, H=80, seed=6)
        rect, _, _ = S.box_layout(2)
        idx = np.concatenate([np.nonzero(rec["target"])[0][::5], np.arange(0, 150, 3)])
        args = (rec["x"][idx], rec["y"][idx], rec["win"][idx], rec["pol"][idx], rec["z0"][idx], rect, 3)
        maps = maps_of(rec, track=True)
        f_fix, ok, cue_fix = maps.features(*args, follow=False, return_memb=True)
        f_trk, _, cue_trk = maps.features(*args, return_memb=True)
        ref = np.stack([membership_ref(rec, e) for e in idx])
        np.testing.assert_allclose(cue_fix.numpy(), ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(cue_trk.numpy(), ref, rtol=1e-5, atol=1e-5)
        self.assertTrue(0 < ref[:, 1].sum() < len(idx))                     # both cases occur
        np.testing.assert_allclose(f_fix.numpy(), maps_of(rec).features(*args)[0].numpy())   # fixed boxes as V17
        net = unit(tokens="glm", gate="distance", evidence="events", audit="ledger", ctx_ff=False, memb=True,
                   nonzero=False, n_steps=3)
        with torch.no_grad():
            U = net(f_fix, ok, torch.as_tensor(rec["z0"][idx]), torch.as_tensor(rec["pol"][idx]), memb=cue_fix)
            self.assertLess(float((U - torch.as_tensor(rec["z0"][idx])[:, None]).abs().max()), 1e-5)
            c_a = net.start(torch.as_tensor(rec["z0"][idx]), torch.as_tensor(rec["pol"][idx]), cue_fix)
            c_b = net.start(torch.as_tensor(rec["z0"][idx]), torch.as_tensor(rec["pol"][idx]), torch.zeros_like(cue_fix))
        self.assertGreater(float((c_a - c_b).abs().max()), 1e-3)
        with self.assertRaises(ValueError):
            net.start(torch.as_tensor(rec["z0"][idx]), torch.as_tensor(rec["pol"][idx]))

    def test_cooperative_tracked_stream(self):
        """Second population with tracked fields reading the spikes of a fixed-field first population."""
        rec = moving_recording(n_noise=600, n_win=9, seed=9)
        rect, _, _ = S.box_layout(2)
        n_steps, tau = 4, -1.2
        bounds1, bounds2 = [2.5, 2.0, 1.5, 1.0, 0.0], [2.2, 1.8, 1.4, 0.9, 0.0]
        kw = dict(n_steps=n_steps, tokens="glm", gate="distance", evidence="events", audit="ledger", ctx_ff=False)
        net1 = unit(**kw)
        net2 = unit(n_msg=S.N_MSG, seed=2, track="transport", **kw)
        step, typ, spikes, F_ = first_population(rec, net1, rect, n_steps, tau, bounds1)
        with torch.no_grad():
            f2, ok2 = maps_of(rec, track=True, messages=spikes).features(
                rec["x"], rec["y"], rec["win"], rec["pol"], rec["z0"], rect, n_steps, own_msg=(step, typ),
                follow="transport")
            U2 = net2(f2, ok2, torch.as_tensor(rec["z0"]), torch.as_tensor(rec["pol"])).numpy()
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
        self.assertTrue((L2 > 1).any())


class HelperTests(unittest.TestCase):
    """tools/v18/common.py with every field option: features, cue and trajectories (with box counts) of a split."""

    def test_event_features_and_trajectories(self):
        import importlib.util                      # by path: the V17 tests import their own 'common'
        spec = importlib.util.spec_from_file_location("v18_common", os.path.join(ROOT, "tools", "v18", "common.py"))
        C = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(C)
        mv = moving_recording(n_noise=400, W=C.SENSOR[1], H=C.SENSOR[0], n_win=6, seed=11)
        rec = {"x": mv["x"].astype(np.int32), "y": mv["y"].astype(np.int32), "win": mv["win"].astype(np.int32),
               "pol": mv["pol"].astype(np.int8), "z0": mv["z0"], "n_win": mv["n_win"], "lab": mv["target"],
               "F": np.minimum(3, mv["n_win"] - 1 - mv["win"]).astype(np.int16)}
        idx = np.arange(0, len(rec["x"]), 5)
        for track in ("off", "recentre", "transport"):
            for memb in ("off", "on"):
                cfg = dict(levels=2, dim=16, heads=4, value="ei", gate="distance", pool="window", evidence="events",
                           context="on", context_ff="off", audit="ledger", tokens="glm", n_steps=4, track=track,
                           membership=memb)
                net, rect = C.build_net(cfg, "cpu")
                maps = C.rec_maps(rec, "cpu", track=C.needs_moments(net))
                feat, ok, cue = C.event_features(maps, rec, idx, rect, 4, None, net)
                self.assertEqual(tuple(feat.shape), (len(idx), 4, len(rect), S.N_FEAT))
                self.assertEqual(cue is None, memb == "off", (track, memb))
                U, nz, ne, ne1 = C.trajectories(net, [rec], rect, "cpu", counts=True)
                self.assertEqual(U[0].shape, (len(rec["x"]), 4))
                self.assertTrue(bool((ne1[0] <= ne[0]).all()))
                np.testing.assert_allclose(U[0], np.repeat(rec["z0"][:, None], 4, 1), atol=1e-5)   # starts at prior

class StoppingTests(unittest.TestCase):
    def test_bound_where_changes_stop(self):
        rng = np.random.RandomState(0)
        m = rng.rand(20000) * 10.0
        change = (m < 3.0) & (rng.rand(20000) < 0.4)
        b = P.bound_from_changes(m, change, 0.01, 0.01)
        self.assertGreater(b, 2.95)                          # just after the last change below 3
        self.assertLess(b, 3.5)
        self.assertEqual(P.bound_from_changes(m, np.ones_like(change), 0.01, 0.01), float("inf"))

    def test_window_bound_robust_to_a_late_flip(self):
        """One change among the most confident events: the isotonic rule cannot certify the 200 events above it
        (infinite bound); the window rule still finds the bound just above the margins that change often, and the
        same bound as the isotonic rule without that flip; frequent changes everywhere or too few events certify
        nothing; easy events far from the bound do not lower it (no averaging over a tail)."""
        rng = np.random.RandomState(2)
        m = rng.rand(30000) * 10.0
        base = (m < 2.0) & (rng.rand(30000) < 0.3)
        change = base.copy()
        change[np.argsort(-m)[200]] = True               # one of the 201 most confident events flips later
        level = 0.1 / 10
        self.assertEqual(P.bound_from_changes(m, change, 0.01, level), float("inf"))
        b = P.bound_from_window(m, change, 0.01, level)
        self.assertTrue(2.0 <= b < 2.8, b)
        self.assertLess(abs(b - P.bound_from_window(m, base, 0.01, level)), 0.05)
        self.assertLess(abs(b - P.bound_from_changes(m, base, 0.01, level)), 0.8)
        self.assertEqual(P.bound_from_window(m, rng.rand(30000) < 0.05, 0.01, level), float("inf"))
        self.assertEqual(P.bound_from_window(m[:100], change[:100], 0.01, level), float("inf"))
        easy = np.r_[m, 20.0 + rng.rand(10 ** 6)]          # a million easy events far above the bound
        self.assertLess(abs(P.bound_from_window(easy, np.r_[change, np.zeros(10 ** 6, bool)], 0.01, level) - b),
                        1e-9)

    def test_window_factor(self):
        """The window size is the rule's one hand-set constant: calibrate(factor=f) uses f x the fewest events that
        certify alpha; factor 4 is the default; a smaller window certifies a lower observed rate, so its bound is not
        below that of a larger window here, and every factor still keeps the realised change rate below alpha."""
        rng = np.random.RandomState(3)
        n, C, tau = 40000, 4, 0.0
        z0 = rng.randn(n) * 4.0
        flip = rng.rand(n) < 0.5 * np.exp(-np.abs(z0))       # change probability falls with the margin
        Z = np.stack([z0] + [np.where(flip, -z0, z0) * (1 + 0.1 * c) for c in range(1, C)], 1)
        last = np.full(n, C - 1)
        self.assertEqual(P.calibrate(Z, last, tau, 0.01, 0.1), P.calibrate(Z, last, tau, 0.01, 0.1, factor=4))
        b2, b8 = (P.calibrate(Z, last, tau, 0.01, 0.1, factor=f) for f in (2, 8))
        self.assertGreaterEqual(b2[0], b8[0] - 1e-9)
        for b in (b2, b8):
            L = P.commit(Z, last, tau, b)
            final = Z[np.arange(n), last] >= tau
            self.assertLess(((Z[np.arange(n), L] >= tau) != final).mean(), 0.01)

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
        self.assertEqual(len(P.calibrate(Z, last, tau, 0.01, 0.1, rule="isotonic")), C)
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
