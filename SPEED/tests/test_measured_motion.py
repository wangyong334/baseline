import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from speed.core.build import load_config, with_overrides  # noqa: E402
from speed.core.system import System  # noqa: E402
from speed.data.events import EventStream  # noqa: E402
from speed.eval.breakdown import target_kinematics  # noqa: E402
from speed.slots.publish.readouts import build_readouts  # noqa: E402
from speed.slots.verify.drift_cusum import DriftEvidence  # noqa: E402
from speed.slots.verify.measured_motion import MeasuredEvidence, MotionMoments  # noqa: E402
from test_system import CONFIG, SMALL, small_system, synthetic_stream  # noqa: E402

DT = 50.0


def random_inputs(rng, steps, H=12, W=14):
    """Per step: counts, mu0, log_g of the previous step (None at k = 0) as float64 tensors [1,1,H,W]."""
    out = []
    for k in range(steps):
        counts = torch.from_numpy(rng.poisson(0.3, (1, 1, H, W)).astype(np.float64))
        mu0 = torch.from_numpy(rng.uniform(0.05, 0.5, (1, 1, H, W)))
        log_g = None if k == 0 else torch.from_numpy(rng.normal(-4.0, 2.0, (1, 1, H, W)))
        out.append((counts, mu0, log_g))
    return out


def random_events(rng, n, H, W):
    z = torch.zeros(n, dtype=torch.long)
    return {"b": z, "y": torch.from_numpy(rng.randint(0, H, n)).long(), "x": torch.from_numpy(rng.randint(0, W, n)).long(),
            "age_ms": torch.from_numpy(rng.uniform(0, DT, n)), "idx": np.arange(n)}


def system_with(base, verifier, cfg):
    return System(base.clock, base.representation, base.network, verifier,
                  build_readouts(cfg["readouts"], verifier, cfg["evaluation"]["threshold"]),
                  base.canvas_multiple, base.chunk_steps, base.threshold)


class EquivalenceTests(unittest.TestCase):
    """A constant integer velocity field must reproduce DriftEvidence with hypotheses {v0, 0} bit for bit."""

    def step_both(self, v0, include_zero):
        decay, gate = math.exp(-DT / 250.0), 0.01
        hyps = [v0, (0.0, 0.0)] if include_zero else [v0]
        drift = DriftEvidence(hyps, 3, decay, gate)
        meas = MeasuredEvidence(DT, [20.0], [2], 5.0, 3, decay, gate, include_zero, fixed_velocity=v0)
        rng = np.random.RandomState(4)
        sd, sm = drift.init_state(1, 12, 14, "cpu", torch.float64), meas.init_state(1, 12, 14, "cpu", torch.float64)
        for k, (counts, mu0, log_g) in enumerate(random_inputs(rng, 9)):
            sd = drift.step(sd, counts, mu0, log_g)
            sm = meas.step(sm, counts, mu0, log_g)
            self.assertTrue(torch.equal(sd["G"], sm["G"]), k)
            self.assertTrue(torch.equal(sd["ell"], sm["ell"]), k)
            if k >= 1:
                n = 40
                b = torch.zeros(n, dtype=torch.long)
                y, x = torch.from_numpy(rng.randint(0, 12, n)).long(), torch.from_numpy(rng.randint(0, 14, n)).long()
                k_from = torch.from_numpy(rng.randint(0, k, n)).long()
                self.assertTrue(torch.equal(drift.gather_along_many(sd["ell"], k, b, y, x, k_from),
                                            meas.gather_along_many(sm["ell"], k, b, y, x, k_from)))

    def test_step_and_gather(self):
        for v0 in ((1.0, -2.0), (0.0, 3.0)):
            self.step_both(v0, True)
        self.step_both((-1.0, 1.0), False)

    def test_system_readouts(self):
        base = small_system(seed=1)
        with torch.no_grad():
            base.network.heads.net[2].bias[1] += 8.0                    # intensities above the gate: real evidence
        cfg = with_overrides(load_config(CONFIG), SMALL)
        decay = math.exp(-base.clock.step_ms / float(cfg["verifier"]["track_tau_ms"]))
        f, gate = int(cfg["verifier"]["footprint_px"]), float(cfg["verifier"]["gate_eps"])
        stream = synthetic_stream(seed=2)
        for v0 in ((0.0, 1.0), (-1.0, 2.0)):
            a = system_with(base, DriftEvidence([v0, (0.0, 0.0)], f, decay, gate), cfg)
            b = system_with(base, MeasuredEvidence(base.clock.step_ms, [20.0], [2], 5.0, f, decay, gate,
                                                   fixed_velocity=v0), cfg)
            with torch.no_grad():
                ra, _ = a.run_stream(stream)
                rb, _ = b.run_stream(stream)
            self.assertEqual(sorted(ra), sorted(rb))
            for name in ra:
                self.assertTrue(np.array_equal(ra[name][0], rb[name][0]), name)
                self.assertTrue(np.array_equal(ra[name][1], rb[name][1]), name)
            self.assertFalse(np.array_equal(ra["fused_d2"][0], ra["net"][0]))  # the comparison is not vacuous


def reference_fit(ev, qy, qx, taus, radii, min_weight=5.0):
    """Direct weighted regression (ev rows: y, x, age at the reference time) -> (vy, vx) px/ms or None."""
    best = None
    for tau in taus:
        for r in radii:
            m = np.maximum(np.abs(ev[:, 0] - qy), np.abs(ev[:, 1] - qx)) <= r
            w = np.exp(-ev[m, 2] / tau)
            Y, X, T = ev[m, 0], ev[m, 1], -ev[m, 2]
            S0 = w.sum()
            if S0 < min_weight:
                continue
            my, mx, mt = (w * Y).sum() / S0, (w * X).sum() / S0, (w * T).sum() / S0
            ctt = (w * T * T).sum() / S0 - mt * mt
            spread = (w * X * X).sum() / S0 - mx * mx + (w * Y * Y).sum() / S0 - my * my
            if ctt <= 1e-6 or spread <= 1e-9:
                continue
            cxt, cyt = (w * X * T).sum() / S0 - mx * mt, (w * Y * T).sum() / S0 - my * mt
            r2 = min(max((cxt * cxt + cyt * cyt) / (ctt * spread), 0.0), 1.0)
            if best is None or r2 > best[0]:
                best = (r2, cyt / ctt, cxt / ctt)
    return None if best is None else best[1:]


def moving_blob(steps, vy, vx, H, W, y0, x0, per_ms=3, noise=40, seed=5):
    """Event lists per step: a 3x3 blob moving at (vy, vx) px/ms plus uniform noise; ages to the step end."""
    rng = np.random.RandomState(seed)
    out = []
    for k in range(steps):
        t = rng.uniform(k * DT, (k + 1) * DT, int(per_ms * DT))
        y = np.clip(np.round(y0 + vy * t + rng.randint(-1, 2, t.size)), 0, H - 1)
        x = np.clip(np.round(x0 + vx * t + rng.randint(-1, 2, t.size)), 0, W - 1)
        tn = rng.uniform(k * DT, (k + 1) * DT, noise)
        y, x = np.r_[y, rng.randint(0, H, noise)], np.r_[x, rng.randint(0, W, noise)]
        age = (k + 1) * DT - np.r_[t, tn]
        out.append((y.astype(np.int64), x.astype(np.int64), age))
    return out


class EstimatorTests(unittest.TestCase):
    def test_moments_match_direct_sums(self):
        mm = MotionMoments(DT, [20.0, 100.0], [2])
        H, W, rng = 6, 7, np.random.RandomState(3)
        m, events = mm.init_state(1, H, W, "cpu"), []
        for k in range(6):
            n = rng.randint(0, 9)
            y, x, age = rng.randint(0, H, n), rng.randint(0, W, n), rng.uniform(0, DT, n)
            m = mm.advance(m, torch.zeros(n, dtype=torch.long), torch.from_numpy(y).long(), torch.from_numpy(x).long(),
                           torch.from_numpy(age))
            events += [(k, a, b, c) for a, b, c in zip(y, x, age)]
        for j, tau in enumerate(mm.taus):
            direct = np.zeros((4, H, W))
            for k, y, x, age in events:
                a = age + (5 - k) * DT
                w = math.exp(-a / tau)
                direct[:, y, x] += [w, w * a, w * a * a, w * w]
            np.testing.assert_allclose(m[0, j].numpy(), direct, rtol=1e-12, atol=1e-300)

    def test_estimate_matches_direct_regression(self):
        H, W = 60, 70
        mm = MotionMoments(DT, [20.0, 100.0, 500.0], [4, 8, 16])
        m, rows = mm.init_state(1, H, W, "cpu"), []
        steps = moving_blob(8, -0.04, 0.12, H, W, 40, 10)
        for k, (y, x, age) in enumerate(steps):
            m = mm.advance(m, torch.zeros(y.size, dtype=torch.long), torch.from_numpy(y), torch.from_numpy(x),
                           torch.from_numpy(age))
        for k, (y, x, age) in enumerate(steps):
            rows.append(np.stack([y, x, age + (len(steps) - 1 - k) * DT], 1).astype(np.float64))
        ev = np.concatenate(rows)
        qy, qx = np.array([24, 25, 30, 5]), np.array([57, 58, 40, 5])
        v, valid, _ = mm.estimate(m, torch.zeros(4, dtype=torch.long), torch.from_numpy(qy).long(),
                                  torch.from_numpy(qx).long())
        for i in range(4):
            ref = reference_fit(ev, qy[i], qx[i], mm.taus, mm.radii)
            self.assertEqual(bool(valid[i]), ref is not None)
            if ref is not None:
                np.testing.assert_allclose(v[i].numpy(), ref, rtol=1e-7, atol=1e-10)

    def test_recovers_motion_and_rest(self):
        H, W = 80, 120
        mm = MotionMoments(DT, [20.0, 100.0, 500.0], [4, 8, 16])
        for vy, vx in ((-0.05, 0.1), (0.0, 0.0)):
            m = mm.init_state(1, H, W, "cpu")
            for y, x, age in moving_blob(12, vy, vx, H, W, 50, 20):
                m = mm.advance(m, torch.zeros(y.size, dtype=torch.long), torch.from_numpy(y), torch.from_numpy(x),
                               torch.from_numpy(age))
            t_end = 12 * DT
            qy, qx = int(round(50 + vy * t_end)), int(round(20 + vx * t_end))
            v, valid, _ = mm.estimate(m, torch.zeros(1, dtype=torch.long), torch.tensor([qy]), torch.tensor([qx]))
            self.assertTrue(bool(valid[0]))
            est = v[0].numpy()
            if vx:
                ang = math.degrees(math.acos(np.dot(est, [vy, vx]) / (np.linalg.norm(est) * math.hypot(vy, vx))))
                self.assertLess(ang, 5.0)
                self.assertLess(abs(np.linalg.norm(est) / math.hypot(vy, vx) - 1.0), 0.15)
            else:
                self.assertLess(np.linalg.norm(est) * DT, 0.5)         # under half a pixel per step


class OrderTests(unittest.TestCase):
    def test_step_k_prediction_ignores_step_k_events(self):
        rng = np.random.RandomState(6)
        H, W = 12, 14
        inputs = random_inputs(rng, 6, H, W)
        evs = [random_events(rng, 30, H, W) for _ in range(6)]
        meas = MeasuredEvidence(DT, [20.0, 100.0], [2, 4], 2.0, 3, math.exp(-DT / 250.0), 0.01)
        runs = []
        for variant in range(2):
            s = meas.init_state(1, H, W, "cpu", torch.float64)
            for k in range(6):
                counts, mu0, log_g = inputs[k]
                ev = evs[k]
                if k == 4 and variant == 1:                             # change only what happens in step 4
                    counts = counts + 3.0
                    ev = random_events(np.random.RandomState(99), 30, H, W)
                s = meas.step(s, counts, mu0, log_g, ev)
                runs.append((variant, k, s["G"].clone(), s["ell"].clone(), {j: f.clone() for j, f in s["hist"].items()}))
        a = {(k): (G, e, h) for v, k, G, e, h in runs if v == 0}
        b = {(k): (G, e, h) for v, k, G, e, h in runs if v == 1}
        self.assertTrue(torch.equal(a[4][0], b[4][0]))                  # prediction for step 4 unchanged
        self.assertTrue(torch.equal(a[4][2][3], b[4][2][3]))            # proposal from steps <= 3 unchanged
        self.assertFalse(torch.equal(a[4][1], b[4][1]))                 # evidence of step 4 sees the new counts
        self.assertFalse(torch.equal(a[5][2][4], b[5][2][4]))           # proposal from steps <= 4 may change


def oracle_stream(vx_ms=0.08):
    """One target moving at vx_ms px/ms along a row plus noise, 10 steps on 24 x 60."""
    rng = np.random.RandomState(7)
    span = int(10 * DT * 1000)
    tt = np.sort(rng.randint(0, span, 600))
    xs = np.round(5 + vx_ms * tt / 1000.0).astype(np.int64)
    tn = rng.randint(0, span, 200)
    t = np.r_[tt, tn]
    x, y = np.r_[xs, rng.randint(0, 60, 200)], np.r_[np.full(600, 12), rng.randint(0, 24, 200)]
    tid = np.r_[np.ones(600, np.int64), np.zeros(200, np.int64)]
    return EventStream("oracle", x, y, t, np.zeros(t.size), tid > 0, tid, (tid > 0).astype(np.int16), 60, 24,
                       span_us=span).validate()


class SystemTests(unittest.TestCase):
    def test_runs_with_energy_and_oracle(self):
        from speed.core.accounting import EnergyStats, energy_parts
        base = small_system(seed=3)
        cfg = with_overrides(load_config(CONFIG), SMALL)
        decay = math.exp(-base.clock.step_ms / float(cfg["verifier"]["track_tau_ms"]))
        f, gate = int(cfg["verifier"]["footprint_px"]), float(cfg["verifier"]["gate_eps"])
        stream = synthetic_stream(seed=4)
        for zero in (True, False):
            v = MeasuredEvidence(base.clock.step_ms, [20.0, 100.0], [2, 4], 5.0, f, decay, gate, include_zero=zero)
            system = system_with(base, v, cfg)
            stats = EnergyStats()
            with torch.no_grad():
                res, _ = system.run_stream(stream, energy=stats)
            for prob, when in res.values():
                self.assertTrue(np.isfinite(prob).all())
            s = stats.summary()
            s["firing_rates"] = {n: 0.1 for n in ("enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1")}
            self.assertGreater(s["verifier_queries_per_step"], 0.0)
            H, W = system.canvas(stream)
            parts = energy_parts(system, H, W, s)
            self.assertGreater(parts["verifier"]["mac"], 0.0)

        # oracle: the proposal at a target event's pixel is the mean label velocity there (px/step)
        ostream = oracle_stream()
        v = MeasuredEvidence(base.clock.step_ms, [20.0], [2], 5.0, f, decay, gate, oracle=True)
        system = system_with(base, v, cfg)
        with torch.no_grad():
            system.run_stream(ostream)
        vx, vy, _ = target_kinematics(ostream)
        steps = system.clock.partition(ostream)
        k = 5
        sel = np.flatnonzero((ostream.label == 1) & (steps.step_of_events()[np.argsort(steps.order)] == k))
        sel = sel[np.isfinite(vx[sel])]
        px = ostream.x[sel]
        for p in np.unique(px):
            want = np.mean(vx[sel][px == p]) * DT
            got = float(v._hist[k][0, 1, 12, p])
            self.assertAlmostEqual(got, want, places=5)


if __name__ == "__main__":
    unittest.main()


class CloudTests(unittest.TestCase):
    """Round 2: uncertainty-shaped hypothesis cloud (sigma points) with an adaptive zero weight."""

    def test_sigma_points_reproduce_mean_and_covariance(self):
        from speed.slots.verify.measured_motion import UT_WEIGHTS, cholesky2, sigma_points
        rng = np.random.RandomState(8)
        a = rng.normal(size=(20, 2, 2))
        cov_m = a @ a.transpose(0, 2, 1) + 0.1 * np.eye(2)
        cov = torch.from_numpy(np.stack([cov_m[:, 0, 0], cov_m[:, 0, 1], cov_m[:, 1, 1]], 1))
        chol = cholesky2(cov)
        L = np.zeros((20, 2, 2))
        L[:, 0, 0], L[:, 1, 0], L[:, 1, 1] = chol[:, 0].numpy(), chol[:, 1].numpy(), chol[:, 2].numpy()
        np.testing.assert_allclose(L @ L.transpose(0, 2, 1), cov_m, atol=1e-12)
        centre = torch.from_numpy(rng.normal(size=(20, 2)))
        pts = sigma_points(centre, chol).numpy()
        w = np.array(UT_WEIGHTS).reshape(5, 1, 1)
        self.assertAlmostEqual(float(w.sum()), 1.0)
        np.testing.assert_allclose((w * pts).sum(0), centre.numpy(), atol=1e-12)
        d = pts - centre.numpy()[None]
        np.testing.assert_allclose(np.einsum("j,jni,jnk->nik", w[:, 0, 0], d, d), cov_m, atol=1e-12)
        self.assertTrue(torch.equal(cholesky2(torch.zeros(3, 3, dtype=torch.float64)), torch.zeros(3, 3, dtype=torch.float64)))

    def test_fixed_cloud_matches_six_velocity_grid(self):
        decay, gate = math.exp(-DT / 250.0), 0.01
        v0 = (1.0, -2.0)
        hyps = [v0, (2.0, -2.0), (0.0, -2.0), (1.0, -1.0), (1.0, -3.0), (0.0, 0.0)]
        drift = DriftEvidence(hyps, 3, decay, gate)
        meas = MeasuredEvidence(DT, [20.0], [2], 5.0, 3, decay, gate, fixed_velocity=v0, cloud=True)
        rng = np.random.RandomState(9)
        sd, sm = drift.init_state(1, 12, 14, "cpu", torch.float64), meas.init_state(1, 12, 14, "cpu", torch.float64)
        for k, (counts, mu0, log_g) in enumerate(random_inputs(rng, 8)):
            sd, sm = drift.step(sd, counts, mu0, log_g), meas.step(sm, counts, mu0, log_g)
            self.assertTrue(torch.equal(sd["G"], sm["G"]), k)
            self.assertTrue(torch.equal(sd["ell"], sm["ell"]), k)
            if k >= 1:
                n = 30
                b = torch.zeros(n, dtype=torch.long)
                y, x = torch.from_numpy(rng.randint(0, 12, n)).long(), torch.from_numpy(rng.randint(0, 14, n)).long()
                k_from = torch.from_numpy(rng.randint(0, k, n)).long()
                self.assertTrue(torch.equal(drift.gather_along_many(sd["ell"], k, b, y, x, k_from),
                                            meas.gather_along_many(sm["ell"], k, b, y, x, k_from)))

    def test_collapsed_cloud_equals_v4a_readouts(self):
        base = small_system(seed=1)
        with torch.no_grad():
            base.network.heads.net[2].bias[1] += 8.0
        cfg = with_overrides(load_config(CONFIG), SMALL)
        decay = math.exp(-base.clock.step_ms / float(cfg["verifier"]["track_tau_ms"]))
        f, gate = int(cfg["verifier"]["footprint_px"]), float(cfg["verifier"]["gate_eps"])
        stream = synthetic_stream(seed=2)
        a = system_with(base, MeasuredEvidence(base.clock.step_ms, [20.0], [2], 5.0, f, decay, gate,
                                               fixed_velocity=(0.0, 1.0)), cfg)
        b = system_with(base, MeasuredEvidence(base.clock.step_ms, [20.0], [2], 5.0, f, decay, gate,
                                               fixed_velocity=(0.0, 1.0), cloud=True, cloud_spacing=0.0), cfg)
        with torch.no_grad():
            ra, _ = a.run_stream(stream)
            rb, _ = b.run_stream(stream)
        for name in ra:
            np.testing.assert_allclose(ra[name][0], rb[name][0], atol=1e-6, err_msg=name)
            self.assertGreater(float(np.mean(ra[name][1] == rb[name][1])), 0.999, name)

    def test_adaptive_zero_weight(self):
        meas = MeasuredEvidence(DT, [20.0], [2], 5.0, 3, 0.5, 0.01, fixed_velocity=(0.0, 0.0), cloud=True,
                                zero_weight="adaptive")
        one = torch.zeros(1, dtype=torch.long)
        lw = meas.log_weights(one, one, one, one).exp()
        self.assertAlmostEqual(float(lw.sum()), 1.0, places=12)
        self.assertAlmostEqual(float(lw[-1, 0]), 0.5, places=12)            # zero inside the cloud: weight 1/2
        fast = MeasuredEvidence(DT, [20.0], [2], 5.0, 3, 0.5, 0.01, fixed_velocity=(0.0, 20.0), cloud=True,
                                zero_weight="adaptive")
        lw = fast.log_weights(one, one, one, one).exp()
        self.assertAlmostEqual(float(lw.sum()), 1.0, places=12)
        self.assertLess(float(lw[-1, 0]), 1e-100)                            # zero far outside: no log 2 cost

    def test_cloud_order_and_system(self):
        rng = np.random.RandomState(10)
        H, W = 12, 14
        inputs = random_inputs(rng, 6, H, W)
        evs = [random_events(rng, 30, H, W) for _ in range(6)]
        meas = MeasuredEvidence(DT, [20.0, 100.0], [2, 4], 2.0, 3, math.exp(-DT / 250.0), 0.01, cloud=True,
                                zero_weight="adaptive")
        G4, hist3 = [], []
        for variant in range(2):
            s = meas.init_state(1, H, W, "cpu", torch.float64)
            for k in range(5):
                counts, mu0, log_g = inputs[k]
                ev = evs[k]
                if k == 4 and variant == 1:
                    counts, ev = counts + 3.0, random_events(np.random.RandomState(98), 30, H, W)
                s = meas.step(s, counts, mu0, log_g, ev)
            G4.append(s["G"].clone())
            hist3.append(s["hist"][3].clone())
        self.assertTrue(torch.equal(G4[0], G4[1]))
        self.assertTrue(torch.equal(hist3[0], hist3[1]))
        base = small_system(seed=3)
        cfg = with_overrides(load_config(CONFIG), SMALL)
        decay = math.exp(-base.clock.step_ms / float(cfg["verifier"]["track_tau_ms"]))
        v = MeasuredEvidence(base.clock.step_ms, [20.0, 100.0], [2, 4], 5.0, int(cfg["verifier"]["footprint_px"]),
                             decay, float(cfg["verifier"]["gate_eps"]), cloud=True, zero_weight="adaptive")
        with torch.no_grad():
            res, _ = system_with(base, v, cfg).run_stream(synthetic_stream(seed=4))
        for prob, when in res.values():
            self.assertTrue(np.isfinite(prob).all())
        self.assertEqual(v.n_hypotheses, 6)


class CloudOracleTests(unittest.TestCase):
    def test_oracle_replaces_only_the_centre(self):
        base = small_system(seed=3)
        cfg = with_overrides(load_config(CONFIG), SMALL)
        decay = math.exp(-base.clock.step_ms / float(cfg["verifier"]["track_tau_ms"]))
        f, gate = int(cfg["verifier"]["footprint_px"]), float(cfg["verifier"]["gate_eps"])
        ostream = oracle_stream()
        v = MeasuredEvidence(base.clock.step_ms, [20.0], [2], 5.0, f, decay, gate, oracle=True, cloud=True,
                             zero_weight="adaptive")
        with torch.no_grad():
            res, _ = system_with(base, v, cfg).run_stream(ostream)
        for prob, when in res.values():
            self.assertTrue(np.isfinite(prob).all())
        vx, vy, _ = target_kinematics(ostream)
        steps = v_steps = base.clock.partition(ostream)
        k = 5
        sel = np.flatnonzero((ostream.label == 1) & (steps.step_of_events()[np.argsort(steps.order)] == k))
        sel = sel[np.isfinite(vx[sel])]
        for p in np.unique(ostream.x[sel]):
            entry = v._hist[k][0, :, 12, p]
            self.assertAlmostEqual(float(entry[1]), np.mean(vx[sel][ostream.x[sel] == p]) * DT, places=5)
            self.assertGreater(float(entry[2]), 0.0)                          # the cloud keeps its spread
