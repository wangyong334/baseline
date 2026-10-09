import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from speed.core.build import build_loss, build_system, load_config, with_overrides  # noqa: E402
from speed.core.splat import bilinear_splat  # noqa: E402
from speed.core.training import train_stream  # noqa: E402
from speed.slots.heads.mark_intensity import MarkIntensityHeads  # noqa: E402
from speed.slots.heads.motion import MarkIntensityMotionHeads  # noqa: E402
from speed.slots.transport.motion import MotionTransport  # noqa: E402
from speed.slots.verify.measured_motion import MeasuredEvidence  # noqa: E402
from test_system import SMALL, synthetic_stream  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTION = os.path.join(ROOT, "configs", "v4", "motion_evuav.yaml")
DT = 50.0


def motion_system(seed=0, transport=True, dtype=torch.float64):
    cfg = with_overrides(load_config(MOTION), SMALL + ([] if transport else ["network.transport.kind=none"]))
    torch.manual_seed(seed)
    system = build_system(cfg)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in system.network.named_parameters():
            if name.endswith("log_gain"):
                p.copy_(torch.rand(p.shape, generator=g) * 2.0 + 0.3)
            elif not name.startswith("heads"):
                p.copy_(torch.randn(p.shape, generator=g) * (2.0 / max(p[0].numel(), 1)) ** 0.5)
        system.network.heads.base.net[2].bias[1] += 6.0                    # intensities above the gate
    system.network.backbone.gain_calibrated.fill_(True)
    system.network.to(dtype)
    return system, cfg


class SplatTests(unittest.TestCase):
    def test_zero_and_integer_displacement(self):
        v = torch.randn(2, 3, 5, 6, dtype=torch.float64)
        z = torch.zeros(2, 1, 5, 6, dtype=torch.float64)
        self.assertTrue(torch.allclose(bilinear_splat(v, z, z), v, atol=0, rtol=0))
        out = bilinear_splat(v, z + 1.0, z - 2.0)
        want = torch.zeros_like(v)
        want[:, :, 1:, :4] = v[:, :, :4, 2:]
        self.assertTrue(torch.equal(out, want))

    def test_fraction_and_mass(self):
        v = torch.zeros(1, 1, 6, 6, dtype=torch.float64)
        v[0, 0, 2, 2] = 1.0
        dy = torch.full((1, 1, 6, 6), 0.25, dtype=torch.float64)
        dx = torch.full((1, 1, 6, 6), 0.5, dtype=torch.float64)
        out = bilinear_splat(v, dy, dx)[0, 0]
        self.assertAlmostEqual(float(out[2, 2]), 0.75 * 0.5)
        self.assertAlmostEqual(float(out[2, 3]), 0.75 * 0.5)
        self.assertAlmostEqual(float(out[3, 2]), 0.25 * 0.5)
        self.assertAlmostEqual(float(out[3, 3]), 0.25 * 0.5)
        self.assertAlmostEqual(float(out.sum()), 1.0)

    def test_gradients(self):
        torch.manual_seed(0)
        v = torch.randn(1, 2, 4, 5, dtype=torch.float64, requires_grad=True)
        dy = (torch.rand(1, 1, 4, 5, dtype=torch.float64) * 0.8 + 0.1).requires_grad_()
        dx = (torch.rand(1, 1, 4, 5, dtype=torch.float64) * 0.8 - 0.9).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(bilinear_splat, (v, dy, dx)))


class HeadAndTransportTests(unittest.TestCase):
    def test_heads_keep_base_initialisation(self):
        torch.manual_seed(5)
        base = MarkIntensityHeads(12, 16)
        torch.manual_seed(5)
        heads = MarkIntensityMotionHeads(12, 22, DT, 16, 16)
        for a, b in zip(base.parameters(), heads.base.parameters()):
            self.assertTrue(torch.equal(a, b))
        out = heads(torch.zeros(2, 12, 4, 5), torch.zeros(2, 22, 4, 5))
        self.assertEqual(tuple(out["motion"].shape), (2, 3, 4, 5))
        self.assertTrue(torch.allclose(out["motion"][:, 2], torch.tensor(math.log(1.0 / DT)), atol=0.5))

    def test_transport(self):
        tr = MotionTransport(DT)
        torch.manual_seed(1)
        states = [torch.randn(1, 3, 8, 8, dtype=torch.float64), torch.randn(1, 4, 4, 4, dtype=torch.float64), None]
        motion = torch.zeros(1, 3, 8, 8, dtype=torch.float64)
        motion[:, 1] = 2.0 / DT                                               # 2 px per step to the right
        off = tr.apply(states, {"motion": motion, "mark": torch.full((1, 1, 8, 8), -1e4, dtype=torch.float64)})
        self.assertTrue(torch.equal(off[0], states[0]) and torch.equal(off[1], states[1]) and off[2] is None)
        on = tr.apply(states, {"motion": motion, "mark": torch.full((1, 1, 8, 8), 1e4, dtype=torch.float64)})
        want = torch.zeros_like(states[0])
        want[..., 2:] = states[0][..., :6]
        self.assertTrue(torch.allclose(on[0], want, atol=1e-12))
        want1 = torch.zeros_like(states[1])
        want1[..., 1:] = states[1][..., :3]                                   # stride 2: one cell
        self.assertTrue(torch.allclose(on[1], want1, atol=1e-12))


class SparseTransportTests(unittest.TestCase):
    def test_sparse_push_equals_dense_splat_with_gradients(self):
        """The transport pushes only the moving cells; values and gradients equal the dense splat of p * u."""
        torch.manual_seed(7)
        tr = MotionTransport(DT)
        H, W = 16, 24
        mark = torch.randn(2, 1, H, W, dtype=torch.float64) * 3.0
        motion = torch.randn(2, 3, H, W, dtype=torch.float64) * 0.05
        outputs = {"mark": mark, "motion": motion}
        shapes = [(2, 3, 16, 24), (2, 4, 8, 12), (2, 6, 4, 6), (2, 6, 2, 3)]
        states = [torch.randn(*sh, dtype=torch.float64, requires_grad=True) for sh in shapes]
        sparse = tr.apply(states, outputs)
        p = torch.sigmoid(mark.float())                                     # the dense code of v4-1, line by line
        p = p * (p >= 0.5).to(p.dtype)
        v = motion[:, :2].float() * DT
        dense = []
        for st in states:
            s = H // int(st.shape[2])
            ps = p if s == 1 else torch.nn.functional.avg_pool2d(p, s)
            vs = v if s == 1 else torch.nn.functional.avg_pool2d(p * v, s) / ps.clamp(min=1e-6) / s
            moving = ps.to(st.dtype) * st
            dense.append(st - moving + bilinear_splat(moving, vs[:, 0:1].to(st.dtype), vs[:, 1:2].to(st.dtype)))
        for a, b in zip(sparse, dense):
            self.assertTrue(torch.equal(a, b))
        weights = [torch.randn_like(a) for a in sparse]
        ga = torch.autograd.grad(sum((a * w).sum() for a, w in zip(sparse, weights)), states)
        gb = torch.autograd.grad(sum((b * w).sum() for b, w in zip(dense, weights)), states)
        for a, b in zip(ga, gb):
            self.assertTrue(torch.allclose(a, b, atol=1e-14, rtol=0))


    def test_sparse_push_is_bit_exact_in_deterministic_mode(self):
        """float32 at the EV-UAV canvas, deterministic algorithms (as training): values and gradients bit-identical to
        the dense splat; also on CUDA when available (the server's test gate)."""
        import torch.nn.functional as F
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        was = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for device in devices:
                g = torch.Generator().manual_seed(3)
                H, W = 264, 352
                mark = (torch.randn(1, 1, H, W, generator=g) * 2.0 - 3.0).to(device)
                motion = (torch.randn(1, 3, H, W, generator=g) * 0.05).to(device)
                shapes = [(1, 12, 264, 352), (1, 24, 132, 176), (1, 48, 66, 88), (1, 48, 33, 44)]
                states = [torch.randn(*sh, generator=g).to(device).requires_grad_() for sh in shapes]
                out = MotionTransport(DT).apply(states, {"mark": mark, "motion": motion})
                p = torch.sigmoid(mark.float())
                p = p * (p >= 0.5).to(p.dtype)
                v = motion[:, :2].float() * DT
                ref = []
                for st in states:
                    s = H // int(st.shape[2])
                    ps = p if s == 1 else F.avg_pool2d(p, s)
                    vs = v if s == 1 else F.avg_pool2d(p * v, s) / ps.clamp(min=1e-6) / s
                    moving = ps.to(st.dtype) * st
                    ref.append(st - moving + bilinear_splat(moving, vs[:, 0:1].to(st.dtype), vs[:, 1:2].to(st.dtype)))
                weights = [torch.randn(a.shape, generator=g).to(device) for a in out]
                ga = torch.autograd.grad(sum((a * w).sum() for a, w in zip(out, weights)), states)
                gb = torch.autograd.grad(sum((b * w).sum() for b, w in zip(ref, weights)), states)
                for a, b in zip(out + list(ga), ref + list(gb)):
                    self.assertTrue(torch.equal(a, b), device)
        finally:
            torch.use_deterministic_algorithms(was)


class LossTests(unittest.TestCase):
    def test_forecast_and_motion_terms(self):
        system, cfg = motion_system(seed=2, transport=False)
        loss = build_loss(cfg["loss"], DT)
        T, H, W = 3, 8, 10
        log_g = torch.full((T, 1, 1, H, W), -3.0, dtype=torch.float64)
        motion = torch.zeros(T, 1, 3, H, W, dtype=torch.float64)
        F0 = loss.forecast(log_g, motion)
        self.assertTrue(torch.allclose(F0, torch.exp(log_g) + 2e-4))
        # one target event per step at (4, 5) with label velocity (0, 0.04) px/ms
        from speed.core.clock import EventBlocks, FixedClock
        from speed.data.events import EventStream
        t = np.array([10, 60, 110]) * 1000
        s = EventStream("s", np.full(3, 5), np.full(3, 4), t, np.zeros(3), np.ones(3), np.ones(3), np.ones(3), W, H,
                        span_us=150000).validate()
        steps = FixedClock(DT).partition(s)
        vel = np.tile([[0.0, 0.04]], (3, 1))
        blk = EventBlocks(s, steps, H, W, "cpu", torch.float64, {"vel": vel}).block(0, 3)
        good, bad = motion.clone(), motion.clone()
        good[:, 0, 1, 4, 5] = 0.04
        good[:, 0, 2], bad[:, 0, 2] = math.log(0.01), math.log(0.01)
        self.assertLess(float(loss.motion_nll(good, blk)), float(loss.motion_nll(bad, blk)))


class SystemTests(unittest.TestCase):
    def test_paths_and_training(self):
        system, cfg = motion_system(seed=3, transport=False)
        stream = synthetic_stream(seed=4)
        H, W = system.canvas(stream)
        steps = system.clock.partition(stream)
        from speed.core.clock import EventBlocks
        blk = EventBlocks(stream, steps, H, W, "cpu", torch.float64).block(0, 6)
        _, inputs, _ = system.representation.encode(system.representation.init_state(1, H, W, "cpu", torch.float64), blk)
        with torch.no_grad():
            a, _, _ = system.network.forward_chunk(inputs, None)
            b, _, _ = system.network.forward_chunk(inputs, None, force_stepwise=True)
        for k in a:
            self.assertTrue(torch.allclose(a[k], b[k], atol=1e-10), k)
        for transport in (False, True):
            system, cfg = motion_system(seed=3, transport=transport)
            opt = torch.optim.Adam(system.network.parameters(), lr=1e-3)
            out = train_stream(system, build_loss(cfg["loss"], DT), synthetic_stream(4), opt, 4, 1.0,
                               np.random.RandomState(0))
            self.assertTrue(math.isfinite(out["loss_sum"]))
            self.assertIn("motion_sum", out)
            self.assertFalse([n for n in out["missing_grads"] if "motion" in n])

    def test_inference_energy_and_motion_eval(self):
        from speed.core.accounting import EnergyStats, energy_parts
        from speed.eval.breakdown import MotionAccuracy
        from speed.slots.publish.readouts import MotionProbe
        system, cfg = motion_system(seed=5, transport=True)
        system.network.eval()
        stream = synthetic_stream(seed=6)
        probe, stats = MotionProbe(), EnergyStats()
        with torch.no_grad():
            res, _ = system.run_stream(stream, energy=stats, readouts=list(system.readouts) + [probe])
        for prob, when in res.values():
            self.assertTrue(np.isfinite(prob).all())
        self.assertNotIn("motion_probe", res)
        acc = MotionAccuracy(DT, 50000)
        acc.update(stream, probe.extra["motion"])
        self.assertGreater(acc.result()["events"], 0)
        s = stats.summary()
        s["firing_rates"] = {n: 0.1 for n in ("enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1")}
        parts = energy_parts(system, *system.canvas(stream), s)
        self.assertIn("motion_head", parts)
        self.assertIn("transport", parts)

    def test_network_source_matches_fixed_cloud(self):
        decay, gate = math.exp(-DT / 250.0), 0.01
        net = MeasuredEvidence(DT, footprint=3, track_decay=decay, gate_eps=gate, cloud=True, source="network")
        fix = MeasuredEvidence(DT, footprint=3, track_decay=decay, gate_eps=gate, cloud=True, fixed_velocity=(0.0, 1.0))
        rng = np.random.RandomState(11)
        H, W = 12, 14
        motion = torch.zeros(1, 3, H, W, dtype=torch.float64)
        motion[:, 1] = 1.0 / DT                                               # 1 px per step
        motion[:, 2] = -1000.0                                                # sigma = 0: only the spacing floor
        sn, sf = net.init_state(1, H, W, "cpu", torch.float64), fix.init_state(1, H, W, "cpu", torch.float64)
        for k in range(7):
            counts = torch.from_numpy(rng.poisson(0.3, (1, 1, H, W)).astype(np.float64))
            mu0 = torch.from_numpy(rng.uniform(0.05, 0.5, (1, 1, H, W)))
            log_g = None if k == 0 else torch.from_numpy(rng.normal(-3.0, 1.5, (1, 1, H, W)))
            n = 25
            ev = {"b": torch.zeros(n, dtype=torch.long), "y": torch.from_numpy(rng.randint(0, H, n)).long(),
                  "x": torch.from_numpy(rng.randint(0, W, n)).long(), "age_ms": torch.zeros(n, dtype=torch.float64),
                  "idx": np.arange(n)}
            sn = net.step(sn, counts, mu0, log_g, ev, motion if k > 0 else None)
            sf = fix.step(sf, counts, mu0, log_g, ev)
            self.assertTrue(torch.allclose(sn["G"], sf["G"], atol=1e-12), k)
            self.assertTrue(torch.allclose(sn["ell"], sf["ell"], atol=1e-12), k)


if __name__ == "__main__":
    unittest.main()
