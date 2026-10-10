import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import torch  # noqa: E402

from speed.core.build import build_loss, build_system, load_config, with_overrides, with_sections  # noqa: E402
from speed.core.clock import EventBlocks  # noqa: E402
from speed.core.training import save_checkpoint, train_stream  # noqa: E402
from speed.eval.breakdown import MotionAccuracy  # noqa: E402
from speed.slots.heads.readout import EvidenceReadoutHead  # noqa: E402
from speed.slots.motion.anchors import AnchorMotion, AnchorTable  # noqa: E402
from speed.slots.publish.learned import LearnedDelayReadout, LearnedPublishReadout, MotionProbe  # noqa: E402
from speed.slots.publish.readouts import FixedDelayReadout  # noqa: E402
from speed.slots.verify.drift_cusum import DriftEvidence, velocity_grid  # noqa: E402
from speed.slots.verify.tube_evidence import TubeEvidence  # noqa: E402
from test_system import synthetic_stream  # noqa: E402

V43 = os.path.join(ROOT, "configs", "v4", "v43_evuav.yaml")
JOINT = os.path.join(ROOT, "configs", "v4", "v43_evuav_joint.yaml")
BASE = os.path.join(ROOT, "configs", "base_evuav.yaml")
HAND = os.path.join(ROOT, "configs", "v4", "hand_v2v3.yaml")
DT = 50.0
NET_SMALL = ["representation.taus_ms=[20, 100, 500]", "representation.dipole_taus_ms=[100]",
             "representation.dipole_radius_px=2", "representation.bg_smooth_radius_px=2",
             "network.backbone.channels=[4, 8, 8, 8]", "network.heads.hidden=6", "evaluation.chunk_steps=5"]
SMALL = NET_SMALL + ["network.readout_head.max_delay_steps=3", "readouts.1.delays_steps=[1, 2, 3]",
                     "evaluation.threshold=0.6"]


def randomise(system, seed, bias=4.0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in system.network.named_parameters():
            if name.endswith("log_gain"):
                p.copy_(torch.rand(p.shape, generator=g) * 2.0 + 0.3)
            elif name.startswith("backbone"):
                p.copy_(torch.randn(p.shape, generator=g) * (2.0 / max(p[0].numel(), 1)) ** 0.5)
        heads = system.network.heads
        (heads.base if hasattr(heads, "base") else heads).net[2].bias[1] += bias
    system.network.backbone.gain_calibrated.fill_(True)


def v43_system(path=V43, seed=0, extra=(), dtype=torch.float64):
    cfg = with_overrides(load_config(path), SMALL + list(extra))
    torch.manual_seed(seed)
    system = build_system(cfg)
    randomise(system, seed)
    system.network.to(dtype)
    return system, cfg


class AnchorTableTests(unittest.TestCase):
    def test_soft_target_and_residual_round_trip(self):
        table = AnchorTable((0.5, 1, 2, 4, 8, 16, 32, 64), 16)
        self.assertEqual(table.size, 129)
        rng = np.random.RandomState(0)
        speed = np.exp(rng.uniform(math.log(0.6), math.log(60.0), 400))
        ang = rng.uniform(0, 2 * math.pi, 400)
        v = torch.from_numpy(np.c_[speed * np.sin(ang), speed * np.cos(ang)])
        soft, dom, res = table.soft_target(v)
        self.assertTrue(torch.allclose(soft.sum(1), torch.ones(400, dtype=torch.float64)))
        back = table.displacement(dom, res)
        self.assertTrue(torch.allclose(back, v, atol=1e-9))
        still, dom0, _ = table.soft_target(torch.tensor([[0.0, 0.0], [0.1, -0.1]], dtype=torch.float64))
        self.assertTrue(torch.equal(dom0, torch.zeros(2, dtype=torch.long)))
        self.assertAlmostEqual(float(still[0, 0]), 1.0)
        half, _, _ = table.soft_target(torch.tensor([[0.0, 0.5 / math.sqrt(2.0)]], dtype=torch.float64))
        self.assertAlmostEqual(float(half[0, 0]), 0.5)                     # halfway between zero and the first ring


class AnchorMotionTests(unittest.TestCase):
    def test_matching_finds_the_displacement(self):
        torch.manual_seed(0)
        motion = AnchorMotion(4).double()
        with torch.no_grad():
            for p in motion.prior.parameters():
                p.zero_()
        H, W = 96, 128
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64))
        q = (48, 64)
        for disp in ((0.0, 2.0), (8.0, 0.0), (0.0, -32.0), (-16.0, 0.0)):
            blob = lambda cy, cx: torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 4.0)  # noqa: E731
            cur = blob(*q).expand(1, 4, H, W).contiguous()
            prev = blob(q[0] - disp[0], q[1] - disp[1]).expand(1, 4, H, W).contiguous()
            logits, res = motion.query(motion.pyramid(cur), motion.pyramid(prev), torch.zeros(1, dtype=torch.long),
                                       torch.tensor([q[0]]), torch.tensor([q[1]]))
            best = motion.table.disp[int(logits.argmax(1))]
            self.assertTrue(torch.allclose(best, torch.tensor(disp, dtype=torch.float64)), (disp, best))


def random_maps(rng, steps, H, W):
    counts = [torch.from_numpy(rng.poisson(0.3, (1, 1, H, W)).astype(np.float64)) for _ in range(steps)]
    mu0 = [torch.from_numpy(rng.uniform(0.05, 0.5, (1, 1, H, W))) for _ in range(steps)]
    log_g = [torch.from_numpy(rng.normal(-3.0, 1.5, (1, 1, H, W))) for _ in range(steps)]
    return counts, mu0, log_g


class EquivalenceTests(unittest.TestCase):
    """Grid hypotheses with uniform weights, uniform position weights, w = 1, H0 = mu0, no anchor: the tube evidence
    of V4-3 is V2's drift evidence (without track memory and gate)."""

    def test_grid_hypotheses_reproduce_v2(self):
        H, W, steps, n_ev = 16, 16, 7, 20                                  # canvases are multiples of 8
        grid = velocity_grid([-2, -1, 0, 1, 2])
        head = EvidenceReadoutHead(3).double()
        motion = AnchorMotion(2).double()
        tube = TubeEvidence(DT, head, motion, hypotheses=len(grid), background="front", anchor=False)
        v2 = DriftEvidence(grid, footprint=3, track_decay=0.0, gate_eps=0.0)

        def grid_hypotheses(cur, prev, frame, y, x):
            n = int(y.shape[0])
            disp = torch.tensor(grid, dtype=torch.float64).view(len(grid), 1, 2).expand(len(grid), n, 2)
            logw = torch.full((len(grid), n), -math.log(len(grid)), dtype=torch.float64)
            return disp, logw, None, None

        tube.hypotheses = grid_hypotheses
        rng = np.random.RandomState(4)
        counts, mu0, log_g = random_maps(rng, steps, H, W)
        st, sv = tube.init_state(1, H, W, "cpu", torch.float64), v2.init_state(1, H, W, "cpu", torch.float64)
        ra, rb = LearnedDelayReadout(tube, [1, 2, 3]), FixedDelayReadout(v2, [1, 2, 3])
        for r in (ra, rb):
            r.begin(steps * n_ev)
        phi = torch.zeros(1, 2, H, W, dtype=torch.float64)
        for k in range(steps):
            ev = {"b": torch.zeros(n_ev, dtype=torch.long), "y": torch.from_numpy(rng.randint(0, H, n_ev)).long(),
                  "x": torch.from_numpy(rng.randint(0, W, n_ev)).long(),
                  "age_ms": torch.zeros(n_ev, dtype=torch.float64), "idx": np.arange(k * n_ev, (k + 1) * n_ev)}
            prev_g = None if k == 0 else log_g[k - 1]
            st = tube.step(st, counts[k], mu0[k], prev_g, ev, None, None, {"phi": phi})
            sv = v2.step(sv, counts[k], mu0[k], prev_g, ev)
            ctx = {"k": k, "idx": ev["idx"], "b": ev["b"], "y": ev["y"], "x": ev["x"],
                   "logits": torch.from_numpy(rng.normal(0, 2, n_ev)), "total": counts[k]}
            ra.step(dict(ctx, verifier=st))
            rb.step(dict(ctx, verifier=sv))
        ra.flush()
        rb.flush()
        a, b = ra.results(0.9), rb.results(0.9)
        for name in a:
            self.assertTrue(np.allclose(a[name][0], b[name][0], atol=1e-6), name)
            self.assertTrue(np.array_equal(a[name][1], b[name][1]), name)


class TrainInferConsistencyTests(unittest.TestCase):
    def test_loss_chains_equal_streaming_readout(self):
        system, cfg = v43_system(seed=1)
        loss = build_loss(cfg["loss"], DT, system)
        stream = synthetic_stream(seed=3)
        H, W = system.canvas(stream)
        steps = system.clock.partition(stream)
        T = 9
        from speed.eval.breakdown import target_kinematics
        vx, vy, _ = target_kinematics(stream, 50000)
        blk = EventBlocks(stream, steps, H, W, "cpu", torch.float64, {"vel": np.stack([vy, vx], 1)}).block(0, T)
        system.network.eval()
        with torch.no_grad():
            rep = system.representation
            _, inputs, aux = rep.encode(rep.init_state(1, H, W, "cpu", torch.float64), blk)
            outputs, _, _ = system.network.forward_chunk(inputs, None)
            ev = blk["events"]
            logits = outputs["mark"][ev["t"], ev["b"], 0, ev["y"], ev["x"]]
            blk.update(aux=aux, prev_phi=None, rng=None)
            prep = loss.prepare(outputs, logits, blk)
            sel, _, chains = loss.chains(prep)
        D = system.network.readout_head.max_delay
        readout = LearnedDelayReadout(system.verifier, list(range(1, D + 1)))
        readout.begin(int(ev["t"].shape[0]))
        state = system.verifier.init_state(1, H, W, "cpu", torch.float64)
        order = np.arange(int(ev["t"].shape[0]))
        with torch.no_grad():
            for k in range(T):
                pick = (ev["t"] == k).nonzero().view(-1)
                events = {"b": ev["b"][pick], "y": ev["y"][pick], "x": ev["x"][pick]}
                prev = None if k == 0 else {n: v[k - 1] for n, v in outputs.items()}
                cur = {n: v[k] for n, v in outputs.items()}
                state = system.verifier.step(state, aux["total"][k], aux["mu0"][k],
                                             None if k == 0 else outputs["log_g"][k - 1], events, None, prev, cur)
                readout.step({"k": k, "idx": order[pick.numpy()], "b": ev["b"][pick], "y": ev["y"][pick],
                              "x": ev["x"][pick], "logits": logits[pick], "verifier": state})
        readout.flush()
        res = readout.results(0.9)
        checked = 0
        idx = prep["q"][sel].numpy()
        for d, (Fd, valid) in enumerate(chains, 1):
            z = logits[prep["q"][sel]] + system.network.readout_head.fusion_weight(d).detach() * Fd
            want = torch.sigmoid(z).numpy()
            m = valid.numpy()
            self.assertTrue(np.allclose(res["fused_d%d" % d][0][idx[m]], want[m], atol=1e-6), d)
            checked += int(m.sum())
        self.assertGreater(checked, 0)


class TrainingTests(unittest.TestCase):
    def run_update(self, path):
        system, cfg = v43_system(path, seed=2)
        frozen = tuple(cfg["training"].get("freeze") or ())
        for name, p in system.network.named_parameters():
            if name.startswith(frozen):
                p.requires_grad_(False)
        before = {n: p.detach().clone() for n, p in system.network.named_parameters()}
        opt = torch.optim.Adam([p for p in system.network.parameters() if p.requires_grad], lr=1e-3)
        out = train_stream(system, build_loss(cfg["loss"], DT, system), synthetic_stream(4), opt, 6, 1.0,
                           np.random.RandomState(0))
        after = dict(system.network.named_parameters())
        return out, before, after, frozen

    def test_trunk_from_base_stays_fixed(self):
        out, before, after, frozen = self.run_update(V43)
        self.assertTrue(math.isfinite(out["loss_sum"]))
        self.assertEqual(out["missing_grads"], [])
        for name, value in before.items():
            changed = not torch.equal(value, after[name].detach())
            self.assertEqual(changed, not name.startswith(frozen), name)
        for name in ("motion_sum", "residual_sum", "growth_sum", "evidence_sum", "stability_sum", "background_sum"):
            self.assertNotEqual(out[name], 0.0, name)

    def test_joint_training_reaches_the_trunk(self):
        out, before, after, _ = self.run_update(JOINT)
        self.assertEqual(out["missing_grads"], [])
        self.assertGreater(out["grad_norms"]["backbone.enc1"], 0.0)
        self.assertGreater(out["mark_sum"], 0.0)


class SystemTests(unittest.TestCase):
    def test_inference_publishing_probe_and_energy(self):
        from speed.core.accounting import EnergyStats, energy_parts
        system, cfg = v43_system(seed=5)
        system.network.eval()
        stream = synthetic_stream(seed=6)
        probe = MotionProbe(DT, system.verifier.n_hypotheses)
        stats = EnergyStats()
        with torch.no_grad():
            res, steps = system.run_stream(stream, energy=stats, readouts=list(system.readouts) + [probe])
            again, _ = system.run_stream(stream)
        self.assertEqual(sorted(res), ["fused_d1", "fused_d2", "fused_d3", "net", "pub"])
        for name, (prob, when) in res.items():
            self.assertTrue(np.isfinite(prob).all(), name)
            self.assertTrue(np.array_equal(prob, again[name][0]), name)
        pub = [r for r in system.readouts if isinstance(r, LearnedPublishReadout)][0]
        self.assertLessEqual(int(pub.extra["age"].max()), 3)
        acc = MotionAccuracy(system.verifier.motion.table, DT, 50000)
        acc.update(stream, probe.extra["velocity"], probe.extra["anchors"])
        self.assertGreater(acc.result()["events"], 0)
        s = stats.summary()
        s["firing_rates"] = {n: 0.1 for n in ("enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1")}
        parts = energy_parts(system, *system.canvas(stream), s)
        for name in ("motion_encoder", "background_head", "verifier", "readout_learned_delay", "readout_pub"):
            self.assertIn(name, parts)

    def test_base_checkpoint_loads_and_hand_readouts_reproduce_the_base(self):
        import tempfile
        from train import load_initial_weights
        cfg_b = with_overrides(load_config(BASE), NET_SMALL)
        torch.manual_seed(3)
        base = build_system(cfg_b)
        randomise(base, 3, bias=2.0)
        base.network.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "base.pt")
            save_checkpoint(path, base.network, None, 0, 0.0, cfg_b)
            cfg_v = with_sections(with_overrides(load_config(V43), NET_SMALL), [HAND])
            torch.manual_seed(9)
            v43 = build_system(cfg_v)
            loaded = load_initial_weights(v43.network, path, "cpu")
        self.assertIn("heads.base.net.0.weight", loaded)
        self.assertTrue(all(k.startswith(("backbone.", "heads.base.")) for k in loaded))
        v43.network.eval()
        stream = synthetic_stream(seed=7)
        with torch.no_grad():
            want, _ = base.run_stream(stream)
            got, _ = v43.run_stream(stream)
        self.assertEqual(sorted(want), sorted(got))
        for name in want:
            self.assertTrue(np.array_equal(want[name][0], got[name][0]), name)
            self.assertTrue(np.array_equal(want[name][1], got[name][1]), name)


if __name__ == "__main__":
    unittest.main()
