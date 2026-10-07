"""Stage 2b gate (local, CPU): the SPEED base system must reproduce the legacy V2-1 / V3 pipeline bit for bit.

    inference   every readout (net, fused_d*, pub): probabilities and publish steps identical (float64 and float32)
    training    one train_stream update: loss sums, every parameter and the Adam state identical
    calibration gains identical
    execution   time-parallel and step-by-step network paths: same spikes, values equal to rounding
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "SPEED"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import train_stream_v2 as tv2  # noqa: E402
from convert_checkpoint import convert_config, convert_state_dict  # noqa: E402
from dataset import stream_windows as sw  # noqa: E402
from dataset.stream_features import EventChunkSource  # noqa: E402
from model.evspsegnet_stream import calibrate_gains as legacy_calibrate  # noqa: E402
from speed.core.build import build_loss, build_system  # noqa: E402
from speed.core.training import calibrate, train_stream  # noqa: E402
from speed.data.events import EventStream  # noqa: E402
from utils.stream_common import load_flat_config  # noqa: E402

H, W, WINDOWS = 24, 32, 14


def legacy_config():
    cfg = load_flat_config(os.path.join(ROOT, "configs", "evisseg_stream_v3.yaml"))
    cfg.update(pad_height=H, pad_width=W, height=H, width_px=W, n_windows=WINDOWS, eval_chunk=5,
               readout_delays=[1, 2, 3], publish_deadline=3, fe_taus_ms=[20, 100, 500], fe_dipole_taus_ms=[100],
               fe_dipole_radius=2, bg_smooth_radius=2, cusum_axis_velocities=[-2, -1, 0, 1, 2], tbptt_k=4,
               channels=[4, 8, 8, 8], head_hidden=6, threshold=0.6, cusum_gate_eps=0.01)
    return cfg


def make_pair(seed=0, n_noise=900):
    rng = np.random.RandomState(seed)
    span = 50 * WINDOWS
    t = rng.randint(0, span, n_noise)
    x, y, p = rng.randint(0, W, n_noise), rng.randint(0, H, n_noise), rng.randint(0, 2, n_noise)
    tid = np.zeros(n_noise, np.int64)
    parts = [(t, x, y, p, tid)]
    for ident, (y0, vx) in enumerate(((8, 0.03), (16, -0.02)), start=1):
        tt = rng.randint(0, span, 350)
        dx = rng.randint(-2, 3, 350)
        cx = (4 if vx > 0 else W - 5) + tt * vx
        parts.append((tt, np.clip(np.round(cx + dx), 0, W - 1).astype(np.int64),
                      np.clip(y0 + rng.randint(-1, 2, 350), 0, H - 1), (dx < 0).astype(np.int64),
                      np.full(350, ident, np.int64)))
    t, x, y, p, tid = (np.concatenate([q[i] for q in parts]) for i in range(5))
    perm = rng.permutation(t.size)
    t, x, y, p, tid = (a[perm].astype(np.int64) for a in (t, x, y, p, tid))
    label = (tid > 0).astype(np.float32)
    order, bounds = sw.split_windows(t, 50, WINDOWS)
    bins, local = sw.time_bin_and_local(t, 50, 5)
    legacy = sw.StreamSequence("syn%d.npz" % seed, x, y, t, p.astype(np.int8), label, tid.astype(np.float64), bins,
                               local, order, bounds)
    stream = EventStream("syn%d.npz" % seed, x, y, t * 1000, p, label.astype(np.uint8), tid, (tid > 0).astype(np.int16),
                         W, H, span_us=span * 1000).validate()
    return legacy, stream


def randomise(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("log_gain"):
                p.copy_(torch.rand(p.shape, generator=g) * 2.0 + 0.3)
            elif name.endswith("neuron.a"):
                p.copy_(torch.randn(p.shape, generator=g))
            elif name.startswith("head") and name.endswith("bias"):
                p.add_(torch.randn(p.shape, generator=g) * 0.5)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * (2.0 / max(p[0].numel(), 1)) ** 0.5)


def build_pair(cfg, dtype, seed=1):
    torch.manual_seed(0)
    frontend, model, cusum = tv2.build_all(cfg, torch.device("cpu"))
    randomise(model, seed)
    model.backbone.gain_calibrated.fill_(True)
    model, frontend = model.to(dtype), frontend
    system = build_system(convert_config(cfg))
    system.network.load_state_dict(convert_state_dict(model.state_dict()))
    system.network.to(dtype)
    return frontend, model, cusum, system


class InferenceEquivalence(unittest.TestCase):
    def check(self, dtype, seed):
        cfg = legacy_config()
        frontend, model, cusum, system = build_pair(cfg, dtype, seed)
        legacy_seq, stream = make_pair(seed)
        with torch.no_grad():
            model.eval()
            system.network.eval()
            probs, extra, _ = tv2.run_sequence(model, frontend, cusum, legacy_seq, cfg, torch.device("cpu"), "carry")
            results, steps = system.run_stream(stream)
        self.assertEqual(sorted(results), sorted(probs))
        birth = legacy_seq.t // 50
        for name, (prob, publish_us) in results.items():
            self.assertTrue(np.array_equal(prob, probs[name]), "%s probabilities differ (%s)" % (name, dtype))
            if name == "net":
                legacy_window = birth
            elif name == "pub":
                legacy_window = extra["publish_pub"]
            else:
                legacy_window = extra["publish_d%s" % name.rsplit("_d", 1)[1]][birth]
            self.assertTrue(np.array_equal(publish_us, (legacy_window + 1) * 50000), "%s publish times differ" % name)
        decided = results["pub"][0] >= np.float32(cfg["threshold"])
        self.assertTrue(0 < decided.sum() < decided.size)

    def test_float64(self):
        self.check(torch.float64, 3)

    def test_float32(self):
        self.check(torch.float32, 4)


class ExecutionPaths(unittest.TestCase):
    def test_stepwise_equals_time_parallel(self):
        """Same spikes; values equal up to float64 rounding (batched convs may sum in another order)."""
        cfg = legacy_config()
        _, _, _, system = build_pair(cfg, torch.float64, 5)
        x = torch.rand(6, 1, system.representation.n_features, H, W, dtype=torch.float64) * 2
        with torch.no_grad():
            a, sa, ia = system.network.forward_chunk(x, None, collect=True)
            b, sb, ib = system.network.forward_chunk(x, None, collect=True, force_stepwise=True)
        for u, v in zip(ia["spikes"], ib["spikes"]):
            self.assertTrue(torch.equal(u, v))
        for name in a:
            self.assertLess(float((a[name] - b[name]).abs().max()), 1e-12, name)
        for u, v in zip(sa, sb):
            self.assertLess(float((u - v).abs().max()), 1e-12)


class TrainingEquivalence(unittest.TestCase):
    def test_one_update(self):
        cfg = legacy_config()
        frontend, model, _, system = build_pair(cfg, torch.float64, 6)
        legacy_seq, stream = make_pair(7)
        opt_legacy = torch.optim.Adam(model.parameters(), lr=1e-2)
        opt_speed = torch.optim.Adam(system.network.parameters(), lr=1e-2)
        loss_fn = build_loss(convert_config(cfg)["loss"])
        for _ in range(2):
            out_legacy = tv2.train_sequence(model, frontend, legacy_seq, opt_legacy, cfg, torch.device("cpu"),
                                            np.random.RandomState(11))
            out_speed = train_stream(system, loss_fn, stream, opt_speed, cfg["tbptt_k"], cfg["grad_clip"],
                                     np.random.RandomState(11))
            for key in ("loss_sum", "mark_sum", "intensity_sum", "events"):
                self.assertEqual(out_legacy[key], out_speed[key], key)
        legacy_params = convert_state_dict(dict(model.named_parameters()))
        for name, p in system.network.named_parameters():
            self.assertTrue(torch.equal(p, legacy_params[name]), name)


class CalibrationEquivalence(unittest.TestCase):
    def test_gains(self):
        cfg = legacy_config()
        frontend, model, _, system = build_pair(cfg, torch.float32, 8)
        model.backbone.gain_calibrated.fill_(False)
        system.network.backbone.gain_calibrated.fill_(False)
        pairs = [make_pair(s) for s in (9, 10)]

        def factory():
            for legacy_seq, _ in pairs:
                def windows(seq=legacy_seq):
                    source = EventChunkSource(seq, cfg, torch.device("cpu"))
                    state = frontend.init_state(1, H, W, torch.device("cpu"))
                    _, feats, _, _ = frontend.run_chunk(state, source.chunk(0, min(6, seq.n_windows)))
                    for t in range(int(feats.shape[0])):
                        yield feats[t], None
                yield windows()

        legacy_calibrate(model.backbone, factory, 0.99, 64, 32, 0.05, 20.0, 37 + 4000)
        calib = dict(sequences=2, steps=6, samples_per_channel=64, min_positive=32, quantile=0.99, gain_min=0.05,
                     gain_max=20.0)
        calibrate(system, [s for _, s in pairs], calib, 37)
        for a, b in zip(model.backbone.blocks(), system.network.blocks()):
            self.assertTrue(torch.equal(a.gain.log_gain, b.gain.log_gain))


if __name__ == "__main__":
    unittest.main()
