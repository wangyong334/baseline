import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from speed.core.build import build_loss, build_system, load_config, with_overrides  # noqa: E402
from speed.core.clock import EventBlocks, FixedClock, canvas_size  # noqa: E402
from speed.core.training import calibrate, make_chunks, train_stream  # noqa: E402
from speed.data.events import EventStream  # noqa: E402
from speed.slots.publish.readouts import NetReadout  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "base_evuav.yaml")
H, W, STEPS, STEP_US = 24, 40, 14, 50000
SMALL = ["representation.taus_ms=[20, 100, 500]", "representation.dipole_taus_ms=[100]",
         "representation.dipole_radius_px=2", "representation.bg_smooth_radius_px=2",
         "network.backbone.channels=[4, 8, 8, 8]", "network.heads.hidden=6",
         "verifier.velocities_px_per_step=[-2, -1, 0, 1, 2]", "verifier.gate_eps=0.01",
         "readouts.1.delays_steps=[1, 2, 3]", "readouts.2.deadline_steps=3", "evaluation.chunk_steps=5",
         "evaluation.threshold=0.6"]


def synthetic_stream(seed=0, n_noise=900, height=H, width=W):
    rng = np.random.RandomState(seed)
    span = STEPS * STEP_US
    t = rng.randint(0, span, n_noise)
    x, y, p = rng.randint(0, width, n_noise), rng.randint(0, height, n_noise), rng.randint(0, 2, n_noise)
    tid = np.zeros(n_noise, np.int64)
    parts = [(t, x, y, p, tid)]
    for ident, (y0, vx) in enumerate(((8, 30e-6), (16, -20e-6)), start=1):
        tt = rng.randint(0, span, 350)
        dx = rng.randint(-2, 3, 350)
        cx = (4 if vx > 0 else width - 5) + tt * vx
        parts.append((tt, np.clip(np.round(cx + dx), 0, width - 1), np.clip(y0 + rng.randint(-1, 2, 350), 0, height - 1),
                      (dx < 0).astype(np.int64), np.full(350, ident, np.int64)))
    t, x, y, p, tid = (np.concatenate([q[i] for q in parts]).astype(np.int64) for i in range(5))
    perm = rng.permutation(t.size)
    t, x, y, p, tid = (a[perm] for a in (t, x, y, p, tid))
    return EventStream("syn%d" % seed, x, y, t, p, tid > 0, tid, tid > 0, width, height, span_us=span).validate()


def small_system(seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    system = build_system(with_overrides(load_config(CONFIG), SMALL))
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in system.network.named_parameters():
            if name.endswith("log_gain"):
                p.copy_(torch.rand(p.shape, generator=g) * 2.0 + 0.3)
            elif not name.startswith("heads"):
                p.copy_(torch.randn(p.shape, generator=g) * (2.0 / max(p[0].numel(), 1)) ** 0.5)
    system.network.backbone.gain_calibrated.fill_(True)
    system.network.to(dtype)
    system.network.eval()
    return system


class ClockTests(unittest.TestCase):
    def test_canvas(self):
        self.assertEqual(canvas_size(260, 346, 8), (264, 352))
        self.assertEqual(canvas_size(720, 1280, 8), (720, 1280))

    def test_steps_and_ages(self):
        s = synthetic_stream()
        steps = FixedClock(50).partition(s)
        self.assertEqual(steps.n_steps, STEPS)
        self.assertEqual(int(steps.t_end_us[-1]), STEPS * STEP_US)
        blocks = EventBlocks(s, steps, H, W, "cpu", torch.float64)
        age = blocks.age.numpy()
        t_sorted = s.t[steps.order]
        self.assertTrue(np.allclose(age, ((t_sorted // STEP_US + 1) * STEP_US - t_sorted) / 1000.0))
        self.assertTrue((age > 0).all() and (age <= 50).all())

    def test_chunks_cover_every_step_once(self):
        for first in (1, 3, 4):
            chunks = make_chunks(13, 4, first)
            covered = [k for a, b in chunks for k in range(a, b)]
            self.assertEqual(covered, list(range(13)))


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.system = small_system(1)
        self.stream = synthetic_stream(2)
        with torch.no_grad():
            self.results, self.steps = self.system.run_stream(self.stream)

    def test_readouts_and_publish_times(self):
        self.assertEqual(sorted(self.results), ["fused_d1", "fused_d2", "fused_d3", "net", "pub"])
        own_end = (self.stream.t // STEP_US + 1) * STEP_US
        last = STEPS * STEP_US
        for name, (prob, publish_us) in self.results.items():
            self.assertEqual(prob.shape, (self.stream.n_events,))
            self.assertTrue(np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all())
            self.assertTrue((publish_us >= own_end).all(), name)
            if name == "net":
                self.assertTrue(np.array_equal(publish_us, own_end))
            elif name.startswith("fused_d"):
                d = int(name[len("fused_d"):])
                self.assertTrue(np.array_equal(publish_us, np.minimum(own_end + d * STEP_US, last)))
            else:
                self.assertTrue((publish_us <= np.minimum(own_end + 3 * STEP_US, last)).all())

    def test_publishing_actually_waits_for_some_events(self):
        prob, publish_us = self.results["pub"]
        own_end = (self.stream.t // STEP_US + 1) * STEP_US
        self.assertTrue((publish_us > own_end).any() and (publish_us == own_end).any())

    def test_causal(self):
        """Removing every event after step K changes nothing that was published before the end of step K."""
        K = 8
        cut, index = self.stream.time_slice(0, K * STEP_US, rebase=False)
        with torch.no_grad():
            short, _ = self.system.run_stream(cut)
        for name, (prob, publish_us) in self.results.items():
            early = publish_us[index] < K * STEP_US
            self.assertTrue(early.any(), name)
            self.assertTrue(np.array_equal(short[name][0][early], prob[index][early]), name)
            self.assertTrue(np.array_equal(short[name][1][early], publish_us[index][early]), name)

    def test_deterministic(self):
        with torch.no_grad():
            again, _ = self.system.run_stream(self.stream)
        for name in self.results:
            self.assertTrue(np.array_equal(again[name][0], self.results[name][0]))

    def test_net_only_without_verifier(self):
        with torch.no_grad():
            only, _ = self.system.run_stream(self.stream, readouts=[NetReadout()])
        self.assertTrue(np.array_equal(only["net"][0], self.results["net"][0]))

    def test_execution_paths(self):
        x = torch.rand(5, 1, self.system.representation.n_features, H, W, dtype=torch.float64) * 2
        with torch.no_grad():
            a, _, ia = self.system.network.forward_chunk(x, None, collect=True)
            b, _, ib = self.system.network.forward_chunk(x, None, collect=True, force_stepwise=True)
        for u, v in zip(ia["spikes"], ib["spikes"]):
            self.assertTrue(torch.equal(u, v))
        for name in a:
            self.assertLess(float((a[name] - b[name]).abs().max()), 1e-12)


class TrainingTests(unittest.TestCase):
    def test_update_reaches_every_parameter(self):
        system = small_system(3)
        system.network.train()
        cfg = with_overrides(load_config(CONFIG), SMALL)
        opt = torch.optim.Adam(system.network.parameters(), lr=1e-3)
        before = {n: p.detach().clone() for n, p in system.network.named_parameters()}
        out = train_stream(system, build_loss(cfg["loss"]), synthetic_stream(4), opt, 4, 1.0, np.random.RandomState(0))
        self.assertTrue(np.isfinite(out["loss_sum"]) and out["loss_sum"] > 0)
        self.assertEqual(out["missing_grads"], [])
        for n, p in system.network.named_parameters():
            self.assertFalse(torch.equal(p, before[n]), n)

    def test_checkpointed_training_matches(self):
        """Activation checkpointing (sub-chunks of 2 steps) gives the same loss and parameter update up to rounding."""
        cfg = with_overrides(load_config(CONFIG), SMALL)
        results = []
        for ckpt in (0, 2):
            system = small_system(5)
            system.network.train()
            opt = torch.optim.SGD(system.network.parameters(), lr=0.1)
            out = train_stream(system, build_loss(cfg["loss"]), synthetic_stream(7), opt, 4, 1e9,
                               np.random.RandomState(1), checkpoint_steps=ckpt)
            results.append((out["loss_sum"], {n: p.detach().clone() for n, p in system.network.named_parameters()}))
        self.assertAlmostEqual(results[0][0], results[1][0], delta=1e-9 * abs(results[0][0]))
        for name, p in results[0][1].items():
            self.assertLess(float((p - results[1][1][name]).abs().max()), 1e-10, name)

    def test_calibration_sets_gains(self):
        torch.manual_seed(0)
        system = build_system(with_overrides(load_config(CONFIG), SMALL))
        calib = dict(steps=6, samples_per_channel=64, min_positive=32, quantile=0.99, gain_min=0.05, gain_max=20.0)
        reports = calibrate(system, [synthetic_stream(5), synthetic_stream(6)], calib, 37)
        self.assertEqual(len(reports), 7)
        self.assertTrue(bool(system.network.backbone.gain_calibrated))
        with self.assertRaises(RuntimeError):
            calibrate(system, [synthetic_stream(5)], calib, 37)


if __name__ == "__main__":
    unittest.main()
