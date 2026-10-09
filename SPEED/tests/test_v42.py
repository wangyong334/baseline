import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from speed.core.build import build_loss, build_system, load_config, with_overrides  # noqa: E402
from speed.core.splat import bilinear_splat, sample_bilinear, splat_points  # noqa: E402
from speed.core.training import train_stream  # noqa: E402
from speed.data.augment import augment_stream, perturb_stream  # noqa: E402
from speed.slots.heads.readout import EvidenceReadoutHead, log_epsilon_bound  # noqa: E402
from speed.slots.neuron.lif import ALIF2d, LIF2d  # noqa: E402
from speed.slots.publish.learned import LearnedDelayReadout, LearnedPublishReadout  # noqa: E402
from speed.slots.publish.readouts import FixedDelayReadout  # noqa: E402
from speed.slots.representation.evidence import EvidenceFrontEnd, LearnableEvidenceFrontEnd  # noqa: E402
from speed.slots.verify.learned_evidence import LearnedEvidence, sigma_cloud  # noqa: E402
from speed.slots.verify.measured_motion import MeasuredEvidence  # noqa: E402
from test_system import synthetic_stream  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V42 = os.path.join(ROOT, "configs", "v4", "v42_evuav.yaml")
DT = 50.0
SMALL = ["representation.taus_ms=[20, 100, 500]", "representation.dipole_taus_ms=[100]",
         "representation.dipole_radius_px=2", "representation.bg_smooth_radius_px=2",
         "network.backbone.channels=[4, 8, 8, 8]", "network.heads.hidden=6", "network.readout_head.max_delay_steps=3",
         "readouts.1.delays_steps=[1, 2, 3]", "evaluation.chunk_steps=5", "evaluation.threshold=0.6"]


def v42_system(seed=0, overrides=(), dtype=torch.float64):
    cfg = with_overrides(load_config(V42), SMALL + list(overrides))
    torch.manual_seed(seed)
    system = build_system(cfg)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in system.network.named_parameters():
            if name.endswith("log_gain"):
                p.copy_(torch.rand(p.shape, generator=g) * 2.0 + 0.3)
            elif name.startswith("backbone"):
                p.copy_(torch.randn(p.shape, generator=g) * (2.0 / max(p[0].numel(), 1)) ** 0.5)
        system.network.heads.base.net[2].bias[1] += 6.0                    # intensities above the source gate
    system.network.backbone.gain_calibrated.fill_(True)
    system.network.to(dtype)
    return system, cfg


class NeuronTests(unittest.TestCase):
    def test_alif_without_adaptation_is_lif(self):
        torch.manual_seed(0)
        lif, alif = LIF2d(3, DT, 200.0, 50.0, 2000.0, 1.0, -4.0), ALIF2d(3, DT, 200.0, 50.0, 2000.0, 1.0, -4.0)
        with torch.no_grad():
            alif.b_raw.fill_(-50.0)                                       # softplus -> ~0
        x = torch.randn(6, 2, 3, 4, 5, dtype=torch.float64) * 1.5
        lif, alif = lif.double(), alif.double()
        s1, u1, _ = lif.forward_steps(x, None)
        s2, st2, _ = alif.forward_steps(x, None)
        self.assertTrue(torch.equal(s1, s2))
        self.assertTrue(torch.allclose(u1, st2[:, :3], atol=1e-12))

    def test_adaptation_raises_threshold_and_steps_agree(self):
        alif = ALIF2d(1, DT, 200.0, 50.0, 2000.0, 1.0, adapt_init=1.0).double()
        x = torch.full((8, 1, 1, 1, 1), 1.2, dtype=torch.float64)
        s, state, _ = alif.forward_steps(x, None)
        lif = LIF2d(1, DT, 200.0, 50.0, 2000.0, 1.0).double()
        s_lif, _, _ = lif.forward_steps(x, None)
        self.assertLess(float(s.sum()), float(s_lif.sum()))               # spike-frequency adaptation
        st, outs = None, []
        for t in range(8):
            o, st, _ = alif(x[t], st)
            outs.append(o)
        self.assertTrue(torch.equal(torch.stack(outs), s) and torch.allclose(st, state))


class AugmentTests(unittest.TestCase):
    def test_identity_thinning_and_compression(self):
        s = synthetic_stream(1)
        rng = np.random.RandomState(0)
        self.assertIs(augment_stream(s, rng), s)
        thin = augment_stream(s, np.random.RandomState(1), keep_min=0.5)
        self.assertLess(thin.n_events, s.n_events)
        self.assertGreater(thin.n_events, 0.4 * s.n_events)
        fast = augment_stream(s, np.random.RandomState(2), time_scale_min=0.5)
        self.assertEqual(fast.n_events, s.n_events)
        self.assertLess(fast.span_us, s.span_us)
        self.assertTrue(np.array_equal(fast.label, s.label) and np.array_equal(fast.x, s.x))
        self.assertTrue((np.diff(fast.t[np.argsort(s.t, kind="stable")]) >= 0).all())
        self.assertIs(perturb_stream(s, rng), s)
        fixed = perturb_stream(s, np.random.RandomState(3), keep=0.7, time_scale=0.5)
        self.assertAlmostEqual(fixed.n_events / float(s.n_events), 0.7, delta=0.05)
        self.assertEqual(fixed.span_us, int(math.ceil(s.span_us * 0.5)))


class FrontEndAndHeadTests(unittest.TestCase):
    def test_learnable_front_end_starts_at_the_base_and_trains(self):
        from speed.core.clock import EventBlocks, FixedClock
        args = ([20.0, 100.0, 500.0], DT, [100.0], 2, 500.0, 5000.0, 0.01, 2.0, 1e-3, 2,
                ("count", "ratio", "age", "dipole", "logmu"))
        base, learn = EvidenceFrontEnd(*args).double(), LearnableEvidenceFrontEnd(*args).double()
        s = synthetic_stream(2)
        steps = FixedClock(DT).partition(s)
        blk = EventBlocks(s, steps, 24, 40, "cpu", torch.float64).block(0, 6)
        _, a, aux_a = base.encode(base.init_state(1, 24, 40, "cpu", torch.float64), blk)
        _, b, aux_b = learn.encode(learn.init_state(1, 24, 40, "cpu", torch.float64), blk)
        self.assertEqual(base.feature_names()[-1], "log_mu0")
        self.assertTrue(torch.allclose(a, b, rtol=1e-5, atol=1e-6))           # log tau is stored in float32
        self.assertTrue(torch.allclose(a[:, :, -1:], torch.log(aux_a["mu0"])))
        b.sum().backward()
        self.assertGreater(float(learn.log_tau.grad.abs().sum()), 0.0)

    def test_backbone_skips_the_dense_channel_and_background_head(self):
        system, _ = v42_system()
        n = system.representation.n_features
        self.assertEqual(system.network.backbone.enc1.conv.in_channels, n - 1)
        self.assertEqual(system.network.heads.background_channel, n - 1)
        x = torch.randn(2, n, 8, 8, dtype=torch.float64)
        u = torch.randn(2, 4, 8, 8, dtype=torch.float64)
        out = system.network.heads(u, x)
        self.assertLess(float((out["log_mu"] - x[:, -1:]).abs().max()), 0.5)    # starts near log mu0


class SplatAndCloudTests(unittest.TestCase):
    def test_sparse_splat_matches_dense_and_sampling_gradients(self):
        torch.manual_seed(3)
        v = torch.zeros(2, 1, 6, 7, dtype=torch.float64)
        v[0, 0, 2, 3], v[1, 0, 4, 1], v[1, 0, 0, 6] = 1.5, 0.7, 2.0
        dy, dx = torch.rand(2, 1, 6, 7, dtype=torch.float64) * 3 - 1.5, torch.rand(2, 1, 6, 7, dtype=torch.float64) * 3 - 1.5
        flat = v.reshape(-1).nonzero().view(-1)
        b, rem = torch.div(flat, 42, rounding_mode="floor"), flat % 42
        y, x = torch.div(rem, 7, rounding_mode="floor"), rem % 7
        sparse = splat_points(v.reshape(-1)[flat], b, y, x, dy.reshape(-1)[flat], dx.reshape(-1)[flat], (2, 6, 7))
        self.assertTrue(torch.allclose(sparse, bilinear_splat(v, dy, dx), atol=1e-12))
        maps = torch.randn(2, 3, 5, 6, dtype=torch.float64, requires_grad=True)
        py = (torch.rand(4, dtype=torch.float64) * 4).requires_grad_()
        px = (torch.rand(4, dtype=torch.float64) * 5).requires_grad_()
        bb, cc = torch.tensor([0, 1, 1, 0]), torch.tensor([2, 0, 1, 1])
        self.assertTrue(torch.autograd.gradcheck(lambda m, a, c: sample_bilinear(m, bb, cc, a, c), (maps, py, px)))

    def test_sigma_cloud(self):
        v = torch.tensor([[0.0, 2.0 / DT], [0.0, 0.0]], dtype=torch.float64)
        pts, logw = sigma_cloud(v, torch.full((2,), -1000.0, dtype=torch.float64), DT, 1.0)
        want = torch.tensor([[0.0, 2.0], [1.0, 2.0], [-1.0, 2.0], [0.0, 3.0], [0.0, 1.0]], dtype=torch.float64)
        self.assertTrue(torch.allclose(pts[:, 0], want, atol=1e-12))
        self.assertTrue(torch.allclose(torch.logsumexp(logw, 0), torch.zeros(2, dtype=torch.float64), atol=1e-12))
        self.assertAlmostEqual(float(torch.exp(logw[5, 1])), 0.5)                   # at rest: zero weight 1/2
        self.assertAlmostEqual(float(torch.exp(logw[5, 0])), 0.5 * math.exp(-0.5 * 4.0 * 3.0))   # d^2 = |v|^2 / s^2
        self.assertAlmostEqual(log_epsilon_bound(0.02), math.log(49.0))


def random_steps(rng, steps, H, W):
    counts = [torch.from_numpy(rng.poisson(0.3, (1, 1, H, W)).astype(np.float64)) for _ in range(steps)]
    mu0 = [torch.from_numpy(rng.uniform(0.05, 0.5, (1, 1, H, W))) for _ in range(steps)]
    log_g = [torch.from_numpy(rng.normal(-3.0, 1.5, (1, 1, H, W))) for _ in range(steps)]
    return counts, mu0, log_g


class EquivalenceTests(unittest.TestCase):
    """At integer velocities, sigma -> 0, the 3 x 3 mean and H0 = mu0 the learned verifier is the v4-1 one."""

    def test_maps_and_delayed_readouts_match_v41(self):
        H, W, steps, gate = 12, 14, 7, 0.01
        head = EvidenceReadoutHead(3, 2, 4).double()
        with torch.no_grad():
            head.mix_logits[head.offsets.abs().max(1)[0] > 1] = -1e9
        learned = LearnedEvidence(DT, head, 1.0, 1e-3, gate, background="front", anchor=False)
        v41 = MeasuredEvidence(DT, footprint=3, track_decay=0.0, gate_eps=gate, cloud=True, zero_weight="adaptive",
                               source="network")
        motion = torch.zeros(1, 3, H, W, dtype=torch.float64)
        motion[:, 1] = 1.0 / DT
        motion[:, 2] = -1000.0
        rng = np.random.RandomState(4)
        counts, mu0, log_g = random_steps(rng, steps, H, W)
        sl, sv = learned.init_state(1, H, W, "cpu", torch.float64), v41.init_state(1, H, W, "cpu", torch.float64)
        ra, rb = LearnedDelayReadout(learned, [1, 2, 3]), FixedDelayReadout(v41, [1, 2, 3])
        n_ev = 0
        for r in (ra, rb):
            r.begin(steps * 20)
        for k in range(steps):
            ev = {"b": torch.zeros(20, dtype=torch.long), "y": torch.from_numpy(rng.randint(0, H, 20)).long(),
                  "x": torch.from_numpy(rng.randint(0, W, 20)).long(), "age_ms": torch.zeros(20, dtype=torch.float64),
                  "idx": np.arange(n_ev, n_ev + 20)}
            prev = (None, None) if k == 0 else (log_g[k - 1], motion)
            sl = learned.step(sl, counts[k], mu0[k], prev[0], ev, prev[1])
            sv = v41.step(sv, counts[k], mu0[k], prev[0], ev, prev[1])
            if k > 0:
                self.assertTrue(torch.allclose(sl["E"], sv["ell"], atol=1e-9), k)
            ctx = {"k": k, "idx": ev["idx"], "b": ev["b"], "y": ev["y"], "x": ev["x"],
                   "logits": torch.from_numpy(rng.normal(0, 2, 20)), "motion": motion, "total": counts[k]}
            ra.step(dict(ctx, verifier=sl))
            rb.step(dict(ctx, verifier=sv))
            n_ev += 20
        ra.flush()
        rb.flush()
        a, b = ra.results(0.9), rb.results(0.9)
        for name in a:
            self.assertTrue(np.allclose(a[name][0], b[name][0], atol=1e-6), name)
            self.assertTrue(np.array_equal(a[name][1], b[name][1]), name)


class TrainInferConsistencyTests(unittest.TestCase):
    def test_loss_chains_equal_streaming_readout(self):
        """The evidence the loss trains is the evidence the streaming readouts use (same functions)."""
        system, cfg = v42_system(seed=1)
        loss = build_loss(cfg["loss"], DT, system)
        stream = synthetic_stream(seed=3)
        H, W = system.canvas(stream)
        steps = system.clock.partition(stream)
        from speed.core.clock import EventBlocks
        T = 9
        blk = EventBlocks(stream, steps, H, W, "cpu", torch.float64).block(0, T)
        system.network.eval()
        with torch.no_grad():
            rep = system.representation
            _, inputs, aux = rep.encode(rep.init_state(1, H, W, "cpu", torch.float64), blk)
            outputs, _, _ = system.network.forward_chunk(inputs, None)
            ev = blk["events"]
            logits = outputs["mark"][ev["t"], ev["b"], 0, ev["y"], ev["x"]]
            chains = loss.chains(outputs, aux, ev, T)
        D = system.network.readout_head.max_delay
        readout = LearnedDelayReadout(system.verifier, list(range(1, D + 1)))
        readout.begin(int(ev["t"].shape[0]))
        state = system.verifier.init_state(1, H, W, "cpu", torch.float64)
        order = np.arange(int(ev["t"].shape[0]))
        with torch.no_grad():
            for k in range(T):
                sel = (ev["t"] == k).nonzero().view(-1)
                prev = None if k == 0 else {n: v[k - 1] for n, v in outputs.items()}
                state = system.verifier.step(state, aux["total"][k], aux["mu0"][k],
                                             None if k == 0 else outputs["log_g"][k - 1], None,
                                             None if k == 0 else outputs["motion"][k - 1], prev)
                readout.step({"k": k, "idx": order[sel.numpy()], "b": ev["b"][sel], "y": ev["y"][sel],
                              "x": ev["x"][sel], "logits": logits[sel], "motion": outputs["motion"][k],
                              "verifier": state})
        readout.flush()
        res = readout.results(0.9)
        checked = 0
        for d, (Fd, valid) in enumerate(chains, 1):
            z = logits + system.network.readout_head.fusion_weight(d).detach() * Fd
            want = torch.sigmoid(z).numpy()
            got = res["fused_d%d" % d][0]
            m = valid.numpy()
            self.assertTrue(np.allclose(got[m], want[m], atol=1e-6), d)
            checked += int(m.sum())
        self.assertGreater(checked, 0)


class TrainingTests(unittest.TestCase):
    def test_every_learned_part_gets_gradients(self):
        for overrides in ([], ["network.neuron.kind=lif", "training.augment=null"]):
            system, cfg = v42_system(seed=2, overrides=overrides)
            opt = torch.optim.Adam(system.network.parameters(), lr=1e-3)
            out = train_stream(system, build_loss(cfg["loss"], DT, system), synthetic_stream(4), opt, 6, 1.0,
                               np.random.RandomState(0))
            self.assertTrue(math.isfinite(out["loss_sum"]))
            self.assertEqual(out["missing_grads"], [])
            for name in ("background_sum", "evidence_sum", "stability_sum"):
                self.assertGreater(out[name], 0.0, name)

    def test_evidence_weight_zero_leaves_fusion_untrained(self):
        system, cfg = v42_system(seed=2, overrides=["loss.evidence_weight=0"])
        opt = torch.optim.Adam(system.network.parameters(), lr=1e-3)
        out = train_stream(system, build_loss(cfg["loss"], DT, system), synthetic_stream(4), opt, 6, 1.0,
                           np.random.RandomState(0))
        self.assertIn("readout_head.fusion", out["missing_grads"])
        self.assertIn("readout_head.mix_logits", out["missing_grads"])
        self.assertNotIn("readout_head.stability.0.weight", out["missing_grads"])


class SystemTests(unittest.TestCase):
    def test_inference_publishing_and_energy(self):
        from speed.core.accounting import EnergyStats, energy_parts
        system, cfg = v42_system(seed=5)
        system.network.eval()
        stream = synthetic_stream(seed=6)
        stats = EnergyStats()
        with torch.no_grad():
            res, steps = system.run_stream(stream, energy=stats)
            again, _ = system.run_stream(stream)
        self.assertEqual(sorted(res), ["fused_d1", "fused_d2", "fused_d3", "net", "pub"])
        for name, (prob, when) in res.items():
            self.assertTrue(np.isfinite(prob).all(), name)
            self.assertTrue(np.array_equal(prob, again[name][0]), name)
        pub = [r for r in system.readouts if isinstance(r, LearnedPublishReadout)][0]
        self.assertLessEqual(int(pub.extra["age"].max()), 3)
        s = stats.summary()
        s["firing_rates"] = {n: 0.1 for n in ("enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1")}
        parts = energy_parts(system, *system.canvas(stream), s)
        for name in ("background_head", "verifier", "readout_learned_delay", "readout_pub"):
            self.assertIn(name, parts)
        head = system.network.readout_head
        theta = pub.theta
        net_dec = res["net"][0] >= np.float32(0.6)
        with torch.no_grad():
            head.stability[2].bias.fill_(1e4)                             # always sure: publish at once = net
            head.stability[2].weight.zero_()
            now, _ = system.run_stream(stream)
            head.stability[2].bias.fill_(-1e4)                            # never sure: wait to the deadline
            late, _ = system.run_stream(stream)
        self.assertTrue(np.array_equal(now["pub"][0] >= np.float32(0.6), net_dec))
        self.assertTrue(np.array_equal(now["pub"][1], steps.t_end_us[np.repeat(np.arange(steps.n_steps),
                                                                              np.diff(steps.bounds))][
            np.argsort(steps.order, kind="stable")]))
        z3 = np.log(late["fused_d3"][0].astype(np.float64)) - np.log1p(-late["fused_d3"][0].astype(np.float64))
        agree = (late["pub"][0] >= np.float32(0.6)) == (z3 >= theta)
        self.assertGreater(agree.mean(), 0.999)

    def test_front_background_ablation_runs_on_the_same_weights(self):
        system, cfg = v42_system(seed=5)
        system.network.eval()
        stream = synthetic_stream(seed=6)
        with torch.no_grad():
            head_res, _ = system.run_stream(stream)
            system.verifier.background = "front"
            front_res, _ = system.run_stream(stream)
        self.assertTrue(np.array_equal(head_res["net"][0], front_res["net"][0]))
        self.assertFalse(np.array_equal(head_res["fused_d2"][0], front_res["fused_d2"][0]))


if __name__ == "__main__":
    unittest.main()
