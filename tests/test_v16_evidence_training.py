"""Real-BIR integration for compact paired evidence and stream boundaries."""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from dataset.overlap_windows import OverlapSequence, OverlapWindowSource
from model.lif2d_stream import detach_states
from test_v14_training import synthetic_record, write_npz
from test_v16_training import tiny_config as dense_config, tensors_in
from tools.v16 import train as tr
from utils.comm_budget import CommBudget


def tiny_config():
    cfg = dense_config()
    cfg.update(revision_architecture="evidence", evidence_hidden=8,
               evidence_kappa=1., cache_record_indices=False,
               readout_hidden=32, threshold=.9, min_epochs=1,
               early_stop_patience=0)
    return cfg


class EvidenceTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.cfg = tiny_config()
        self.device = torch.device("cpu")
        self.q99 = np.full(12, np.log1p(2), dtype=np.float32)
        tr.seed_everything(37, True)
        self.net, self.head = tr.make_models(self.cfg, self.device)
        calibration = synthetic_record(37)

        def factory():
            source = OverlapWindowSource(calibration, self.cfg, self.q99, self.device)

            def windows():
                for k in range(calibration.n_windows):
                    x, events, _, _ = source.window(k)
                    yield x, events

            yield windows()

        tr.calibrate_gains(self.net, factory, .99, 32, 1, .05, 20., 4037)

    def activate(self, head=None):
        head = self.head if head is None else head
        with torch.no_grad():
            if head.has_revision:
                head.output.weight.normal_(0., .04)
            if head.has_context:
                head.reliability[-1].weight.normal_(0., .04)

    def record_forward(self, seq, lo=0, hi=None, states=None):
        hi = seq.n_windows if hi is None else hi
        source = OverlapWindowSource(seq, self.cfg, self.q99, self.device)
        x, ev, labels, ids = source.chunk(lo, hi)
        _, states, info = self.net.forward_chunk(x, ev, states, return_features=True, skip_readout=True)
        z, h = tr.readout_with_hidden(self.net, info["features"], ev)
        return info["features"], z, h, labels, ev, ids, states

    def loss_part(self, seq, features, z, h, labels, events, ids, lo, hi, pending):
        a, b = int(seq.bounds[lo]), int(seq.bounds[hi])
        ev = {k: v[a:b] for k, v in events.items()}
        ev["t"] = ev["t"] - lo
        return tr.classification_loss(self.net, self.head, features[lo:hi], z[a:b], labels[a:b], ev,
                                      [(0, lo, hi, 0, seq, ids[a:b])],
                                      [[(0, lo, hi, 0, lo == 0)]], pending, 1.,
                                      hidden=h[a:b], cfg=self.cfg)

    def test_common_backbone_and_remaining_modules_have_identical_initialization(self):
        states, rng = {}, {}
        for variant in ("full", "no_context", "no_revision", "second"):
            tr.seed_everything(37, True)
            net, head = tr.make_models(dict(self.cfg, revision_variant=variant), self.device)
            states[variant] = (net.state_dict(), head.state_dict())
            rng[variant] = torch.get_rng_state().clone()
        for variant, (net_state, head_state) in states.items():
            self.assertTrue(torch.equal(rng[variant], rng["full"]))
            for name, value in net_state.items():
                self.assertTrue(torch.equal(value, states["full"][0][name]), (variant, name))
            for name, value in head_state.items():
                self.assertTrue(torch.equal(value, states["full"][1][name]), (variant, name))

    def test_readout_reuses_exact_original_v5_hidden_activations(self):
        seq = synthetic_record(1)
        features, z, h, _, ev, _, _ = self.record_forward(seq)
        values = features[ev["t"], ev["b"], :, ev["y"], ev["x"]]
        original = self.net._readout(values, ev)
        self.assertTrue(torch.equal(z, original))
        self.assertEqual(tuple(h.shape), (seq.order.size, 32))

    def test_unique_supervision_and_same_losses_across_tbptt_boundaries(self):
        self.activate()
        seq = synthetic_record(2)
        features, z, h, labels, ev, ids, _ = self.record_forward(seq)
        pending = {}
        full, mass, count, cross = self.loss_part(seq, features, z, h, labels, ev, ids, 0, 5, pending)
        self.assertFalse(pending)
        self.assertEqual((mass, count, cross), (seq.n_events, seq.n_events, 0))
        losses, total_mass, total_count, total_cross = [], 0., 0, 0
        for lo, hi in ((0, 2), (2, 4), (4, 5)):
            loss, mass, count, cross = self.loss_part(seq, features, z, h, labels, ev, ids, lo, hi, pending)
            losses.append(loss)
            total_mass += mass; total_count += count; total_cross += cross
            pending = tr.detach_pending(pending)
        self.assertFalse(pending)
        self.assertEqual(total_mass, seq.n_events)
        self.assertEqual(total_count, seq.n_events)
        self.assertGreater(total_cross, 0)
        self.assertTrue(torch.allclose(sum(losses), full, atol=3e-5, rtol=1e-6))

    def test_pending_cache_is_only_uncommitted_hidden_and_detaches_gradient(self):
        self.activate()
        seq = synthetic_record(3)
        features, z, h, labels, ev, ids, states = self.record_forward(seq, 0, 2)
        h.retain_grad()
        pending = {}
        tr.classification_loss(self.net, self.head, features, z, labels, ev,
                               [(0, 0, 2, 0, seq, ids)], [[(0, 0, 2, 0, True)]],
                               pending, 1., hidden=h, cfg=self.cfg)
        self.assertEqual(set(pending[0]), {"window", "ids", "logits", "hidden"})
        wanted = np.flatnonzero((seq.first_window < 2) & (seq.last_window >= 2))
        np.testing.assert_array_equal(np.sort(pending[0]["ids"]), wanted)
        self.assertEqual(tuple(pending[0]["hidden"].shape), (len(wanted), 32))
        before = [t.detach().clone() for t in tensors_in(pending)]
        storage = {t.storage().data_ptr() for t in tensors_in(pending)}
        pending = tr.detach_pending(pending)
        for old, new in zip(before, tensors_in(pending)):
            self.assertTrue(torch.equal(old, new))
            self.assertFalse(new.requires_grad)
            self.assertIsNone(new.grad_fn)
            self.assertNotIn(new.storage().data_ptr(), storage)
            self.assertEqual(new.storage().size(), new.numel())
        f2, z2, h2, y2, ev2, ids2, _ = self.record_forward(seq, 2, 4, detach_states(states))
        loss, _, _, cross = tr.classification_loss(
            self.net, self.head, f2, z2, y2, ev2, [(0, 2, 4, 0, seq, ids2)],
            [[(0, 2, 4, 0, False)]], pending, 1., hidden=h2, cfg=self.cfg)
        loss.backward()
        self.assertGreater(cross, 0)
        self.assertIsNone(h.grad)
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any())
                            for p in self.head.parameters()))

    def test_empty_singleton_cold_start_and_tail_coverage(self):
        self.activate()
        for timestamp in (0, 49, 50, 99, 100, 199, 249, 250, 299):
            seq = OverlapSequence("one", [1], [2], [timestamp], [1], [1.], [1.], total_ms=300)
            output, _, _ = tr.infer_record(self.net, self.head, seq, self.cfg,
                                           self.q99, self.device, step=True)
            self.assertTrue(all(v.shape == (1,) and np.isfinite(v).all() for v in output.values()))
            if seq.coverage_count[0] == 1:
                np.testing.assert_array_equal(output["main"], output["second"])
        seq = OverlapSequence("empty", [], [], [], [], [], [], total_ms=300)
        output, _, _ = tr.infer_record(self.net, self.head, seq, self.cfg, self.q99, self.device, step=True)
        self.assertTrue(all(v.size == 0 for v in output.values()))

    def test_midchunk_record_transitions_match_independent_pairs(self):
        self.activate()
        records = [synthetic_record(seed) for seed in (10, 11, 12)]
        sources = tr.Sources(records, self.cfg, self.q99, self.device)
        pieces = [[(0, 0, 5, 0, True), (2, 0, 5, 5, True)], [(1, 0, 5, 2, True)]]
        x, ev, _, keep, segments = tr.assemble(sources, pieces, 10, self.cfg, self.device)
        with torch.no_grad():
            _, _, info = self.net.forward_chunk(x, ev, None, keep=keep, return_features=True, skip_readout=True)
            z, h = tr.readout_with_hidden(self.net, info["features"], ev)
            pending = {}
            fused, rows, _ = tr.evidence_pairs(self.head, z, h, segments, pending)
            self.assertFalse(pending)
            by_row = {int(row): value for row, value in zip(rows, fused)}
            for index, lo, hi, row, seq, ids in segments:
                _, iz, ih, _, _, _, _ = self.record_forward(seq)
                independent, local_rows, _ = tr.evidence_pairs(
                    self.head, iz, ih, [(index, lo, hi, 0, seq, ids)], {})
                expected = torch.stack([by_row[row + int(local)] for local in local_rows])
                self.assertTrue(torch.allclose(expected, independent, atol=3e-6, rtol=1e-5), index)

    def test_main_only_reads_each_appearance_once_and_matches_diagnostics(self):
        self.activate()
        seq = synthetic_record(13)
        diagnostic, _, _ = tr.infer_record(self.net, self.head, seq, self.cfg,
                                           self.q99, self.device, step=True)
        reads = []
        hook = self.net.readout[0].register_forward_pre_hook(lambda module, inputs: reads.append(len(inputs[0])))
        try:
            production, _, _ = tr.infer_record(self.net, self.head, seq, self.cfg,
                                               self.q99, self.device, modes=["main"], step=True)
        finally:
            hook.remove()
        self.assertEqual(sum(reads), seq.order.size)
        self.assertLessEqual(sum(reads), 2 * seq.n_events)
        np.testing.assert_array_equal(production["main"], diagnostic["main"])

    def test_step_layer_and_causality_with_active_modules(self):
        self.activate()
        seq = synthetic_record(14)
        layer, _, layer_messages = tr.infer_record(self.net, self.head, seq, self.cfg,
                                                  self.q99, self.device, diagnostics=True)
        step, timing, step_messages = tr.infer_record(self.net, self.head, seq, self.cfg,
                                                     self.q99, self.device, step=True, diagnostics=True)
        for mode in step:
            np.testing.assert_allclose(layer[mode], step[mode], rtol=1e-5, atol=3e-6)
        np.testing.assert_allclose(layer_messages, step_messages, rtol=1e-5, atol=1e-5)
        self.assertEqual(len(timing["all_step_ms"]), seq.n_windows)
        changed = copy.deepcopy(seq)
        late = changed.t >= 200
        changed.x[late] = (changed.x[late] + 3) % 8
        changed.y[late] = (changed.y[late] + 2) % 8
        changed.p[late] = 1 - changed.p[late]
        altered, _, _ = tr.infer_record(self.net, self.head, changed, self.cfg,
                                        self.q99, self.device, step=True)
        committed = seq.output_time_ms <= 200
        for mode in step:
            np.testing.assert_array_equal(step[mode][committed], altered[mode][committed])

    def test_real_multiple_updates_with_each_independent_ablation(self):
        records = [synthetic_record(seed) for seed in (20, 21, 22)]
        plan = [(0, [(0, 5), (2, 5)]), (1, [(1, 5)])]
        for variant in ("full", "no_context", "no_revision", "second"):
            cfg = dict(self.cfg, revision_variant=variant,
                       fusion_loss="excess_risk", excess_risk_weight=1.)
            net, head = tr.make_models(cfg, self.device)
            net.load_state_dict(self.net.state_dict())
            optimizer = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=1e-3)
            comm = CommBudget.from_config(cfg, net)
            result = tr.train_epoch(net, head, records, optimizer, cfg, self.q99, 1., self.device,
                                    np.random.RandomState(37), comm=comm, plan=plan)
            self.assertEqual(result["supervision_mass"], sum(seq.n_events for seq in records))
            self.assertEqual(result["committed_events"], sum(seq.n_events for seq in records))
            self.assertGreaterEqual(result["updates"], 2)
            self.assertGreater(result["cross_tbptt_pairs"], 0)
            self.assertEqual(result["comm"]["mode"], "peak")
            self.assertGreater(result["grad_groups_mean"]["backbone"], 0., variant)
            self.assertGreater(result["grad_groups_mean"]["readout"], 0., variant)
            if variant != "second":
                self.assertGreater(result["grad_groups_mean"]["fusion"], 0., variant)
            else:
                self.assertEqual(result["grad_groups_mean"]["fusion"], 0.)

    def test_excess_risk_corrects_each_class_and_does_not_train_reference(self):
        # Bad positive corrections must move upward; bad negative corrections
        # must move downward. Beneficial changes are not pulled back to teacher.
        reference = torch.tensor([1., -1., 1., -1., 0., 0.], requires_grad=True)
        logits = torch.tensor([0., 0., 2., -2., 0., 0.], requires_grad=True)
        labels = torch.tensor([1., 0., 1., 0., 1., 0.])
        penalty = tr.supervised_excess_risk(logits, reference, labels, 3.)
        self.assertTrue(bool((penalty[:2] > 0).all()))
        self.assertAlmostEqual(float(penalty[0] / penalty[1]), 3., places=6)
        self.assertTrue(torch.equal(penalty[2:], torch.zeros(4)))
        penalty.sum().backward()
        self.assertLess(float(logits.grad[0]), 0.)
        self.assertGreater(float(logits.grad[1]), 0.)
        self.assertTrue(torch.equal(logits.grad[2:], torch.zeros(4)))
        self.assertIsNone(reference.grad)
        extreme = torch.tensor([-1000., 1000.], requires_grad=True)
        loss = tr.supervised_excess_risk(extreme, -extreme.detach(), torch.tensor([1., 0.]), 30.)
        self.assertTrue(bool(torch.isfinite(loss).all()))
        loss.sum().backward()
        self.assertTrue(bool(torch.isfinite(extreme.grad).all()))

    def test_excess_risk_keeps_unique_supervision_mass_across_boundaries(self):
        self.activate()
        self.cfg.update(fusion_loss="excess_risk", excess_risk_weight=1.)
        seq = synthetic_record(24)
        features, z, h, labels, ev, ids, _ = self.record_forward(seq)
        full, mass, count, _ = self.loss_part(seq, features, z, h, labels, ev, ids, 0, 5, {})
        pending, losses, counts, masses = {}, [], [], []
        for lo, hi in ((0, 2), (2, 4), (4, 5)):
            loss, weight, committed, _ = self.loss_part(
                seq, features, z, h, labels, ev, ids, lo, hi, pending)
            losses.append(loss); counts.append(committed); masses.append(weight)
            pending = tr.detach_pending(pending)
        self.assertFalse(pending)
        self.assertEqual((mass, count, sum(masses), sum(counts)), (seq.n_events,) * 4)
        self.assertTrue(torch.allclose(sum(losses), full, rtol=1e-6, atol=3e-5))

    def training_files(self, directory, name="output", **updates):
        root = Path(directory)
        data, output = root / "data", root / name
        for split, count in (("train", 2), ("val", 1)):
            folder = data / split
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                path = folder / ("%s_%03d.npz" % (split, i))
                if not path.exists():
                    write_npz(path, synthetic_record(i + 40))
        cfg = dict(self.cfg, root=str(data), save_root=str(output),
                   stats_path=str(output / "train_stats.json"))
        cfg.update(updates)
        args = SimpleNamespace(out=str(output), device="cpu", resume=False,
                               limit_records=None, max_chunks=None)
        return cfg, args, output

    @staticmethod
    def constant_validation(*args, **kwargs):
        return {"metrics": {"iou": .5, "acc": .7},
                "readouts": {"main": {"iou": .5, "acc": .7}}}

    def test_plateau_resume_preserves_patience_and_finished_run_cannot_restart(self):
        # The same maximum can appear repeatedly; only strict improvements
        # reset patience, including after process interruption and resumption.
        with tempfile.TemporaryDirectory(prefix="v16_evidence_resume_") as directory:
            cfg, args, output = self.training_files(
                directory, epochs=7, min_epochs=2, early_stop_patience=3)
            original = tr.train_epoch
            calls = [0]

            def interrupted(*a, **kw):
                calls[0] += 1
                if calls[0] == 3:
                    raise RuntimeError("simulated process interruption")
                return original(*a, **kw)

            with mock.patch.object(tr, "evaluate", side_effect=self.constant_validation), contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(tr, "train_epoch", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "simulated process interruption"):
                        tr.train(copy.deepcopy(cfg), args)
                before = json.loads((output / "summary.json").read_text())
                self.assertEqual(before["epochs_completed"], 2)
                self.assertFalse(before["complete"])
                args.resume = True
                tr.train(copy.deepcopy(cfg), args)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["epochs_completed"], 4)
            self.assertEqual(summary["best_epoch"], 1)
            self.assertTrue(summary["early_stopped"])
            self.assertTrue(summary["complete"])
            history = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in history], [0, 1, 2, 3])
            with mock.patch.object(tr, "train_epoch") as training:
                with self.assertRaisesRegex(ValueError, "[Cc]omplet|[Ff]inish"):
                    tr.train(copy.deepcopy(cfg), args)
                training.assert_not_called()

    def test_resume_rejects_changed_training_conditions_before_update(self):
        with tempfile.TemporaryDirectory(prefix="v16_evidence_resume_config_") as directory:
            cfg, args, output = self.training_files(directory, epochs=3)
            original = tr.train_epoch
            calls = [0]

            def interrupted(*a, **kw):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError("simulated process interruption")
                return original(*a, **kw)

            with mock.patch.object(tr, "evaluate", side_effect=self.constant_validation), \
                    mock.patch.object(tr, "train_epoch", side_effect=interrupted), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "simulated process interruption"):
                    tr.train(copy.deepcopy(cfg), args)
            args.resume = True
            changed = dict(cfg, fusion_loss="excess_risk", excess_risk_weight=2.)
            with mock.patch.object(tr, "train_epoch") as training:
                with self.assertRaisesRegex(ValueError, "Resume config/design mismatch"):
                    tr.train(changed, args)
                training.assert_not_called()

    def test_warmstart_checks_seed_normalization_and_inherits_real_backbone(self):
        with tempfile.TemporaryDirectory(prefix="v16_evidence_init_") as directory:
            cfg, args, output = self.training_files(directory, epochs=1)
            stats = tr.compute_overlap_stats(tr.Records(cfg, "train").paths, cfg)
            path = Path(directory) / "initial.pt"
            initial = {"model": self.net.state_dict(), "config": copy.deepcopy(cfg),
                       "stats": stats, "comm_mu": .02}
            torch.save(initial, str(path))
            cfg["init_checkpoint"] = str(path)
            original = tr.train_epoch
            seen = []

            def check_initial(net, head, *a, **kw):
                for name, tensor in net.state_dict().items():
                    self.assertTrue(torch.equal(tensor, initial["model"][name]), name)
                self.assertTrue(torch.equal(head.output.weight, torch.zeros_like(head.output.weight)))
                self.assertTrue(torch.equal(head.reliability[-1].weight, torch.zeros_like(head.reliability[-1].weight)))
                seen.append(True)
                return original(net, head, *a, **kw)

            with mock.patch.object(tr, "calibrate") as calibration, \
                    mock.patch.object(tr, "evaluate", side_effect=self.constant_validation), \
                    mock.patch.object(tr, "train_epoch", side_effect=check_initial), contextlib.redirect_stdout(io.StringIO()):
                tr.train(copy.deepcopy(cfg), args)
                calibration.assert_not_called()
            self.assertEqual(seen, [True])
            self.assertEqual(json.loads((output / "calibration.json").read_text())["mode"], "inherited matched backbone")
            for case in ("seed", "q99_values", "q99_shape", "bir_levels", "input_clip",
                         "mu_nan", "mu_inf", "mu_negative"):
                bad = copy.deepcopy(initial)
                if case == "seed":
                    bad["config"]["seed"] = 38
                elif case == "q99_values":
                    bad["stats"]["q99"] = np.asarray(bad["stats"]["q99"]) + 1.
                elif case == "q99_shape":
                    # Values broadcast to the correct vector but must not be
                    # accepted as a valid channel-normalization shape.
                    bad["stats"]["q99"] = np.asarray(bad["stats"]["q99"]).reshape(1, -1).tolist()
                elif case == "bir_levels":
                    # These settings do not change state_dict tensor shapes,
                    # but they do change the meaning of inherited weights.
                    bad["config"]["tif"]["levels"] = 3
                elif case == "input_clip":
                    bad["config"]["input_clip"] = float(cfg["input_clip"]) + 1.
                else:
                    bad["comm_mu"] = {"mu_nan": float("nan"), "mu_inf": float("inf"),
                                      "mu_negative": -.1}[case]
                torch.save(bad, str(path))
                bad_cfg, bad_args, _ = self.training_files(directory, name="bad_" + case, epochs=1)
                bad_cfg["init_checkpoint"] = str(path)
                with mock.patch.object(tr, "train_epoch", side_effect=AssertionError("Invalid warmstart reached training")) as training:
                    with self.assertRaises(ValueError, msg=case):
                        tr.train(bad_cfg, bad_args)
                    training.assert_not_called()

    def test_cli_train_validate_test_and_profile(self):
        with tempfile.TemporaryDirectory(prefix="v16_evidence_cli_") as directory:
            root = Path(directory)
            data, output = root / "data", root / "output"
            for split, count in (("train", 3), ("val", 1), ("test", 1)):
                folder = data / split
                folder.mkdir(parents=True)
                for i in range(count):
                    write_npz(folder / ("%s_%03d.npz" % (split, i)), synthetic_record(i + 30))
            cfg = dict(self.cfg, root=str(data), save_root=str(output), stats_path=str(output / "train_stats.json"))
            config = root / "tiny.yaml"
            config.write_text(yaml.safe_dump({"EXPERIMENT": cfg}), encoding="utf-8")
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            command = [sys.executable, str(ROOT / "tools/v16/train.py"), "--config", str(config),
                       "--out", str(output), "--device", "cpu"]

            def run(cmd):
                result = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)

            run(command + ["--mode", "train"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["epochs_completed"], cfg["epochs"])
            checkpoint = output / "best_val_iou_seed37.pt"
            for split in ("val", "test"):
                run(command + ["--mode", "eval", "--checkpoint", str(checkpoint), "--split", split,
                               "--step", "--dump-predictions"])
                report = json.loads((output / ("eval_%s_step.json" % split)).read_text())
                self.assertEqual(report["split"], split)
                for metric in ("iou", "acc", "pd", "fa"):
                    self.assertTrue(np.isfinite(report["metrics"][metric]))
                self.assertEqual(report["production_event_readouts"], synthetic_record(30).order.size)
                pred = next((output / ("predictions_" + split)).glob("*.npz"))
                with np.load(str(pred)) as values:
                    np.testing.assert_array_equal(values["event_id"], np.arange(report["events"]))
            run([sys.executable, str(ROOT / "tools/v16/profile_inference.py"), "--config", str(config),
                 "--checkpoint", str(checkpoint), "--data-root", str(data), "--out", str(output),
                 "--device", "cpu", "--split", "test", "--records", "1", "--repeats", "2"])
            profile = json.loads((output / "profile.json").read_text())
            self.assertEqual(profile["event_reads"], 2 * synthetic_record(30).order.size)
            self.assertEqual(profile["events"], 2 * synthetic_record().n_events)
            self.assertEqual(profile["rows"][1]["repeat_max_abs_diff"], 0.)


if __name__ == "__main__":
    unittest.main()
