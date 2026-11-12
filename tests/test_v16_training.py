"""V16 integration contracts: real BIR, rolling IDs, TBPTT and CLI artifacts.

These tests deliberately include empty windows, record transitions and pairs
crossing an optimizer boundary.  They do not change the formal training model.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from dataset.overlap_windows import OverlapSequence, OverlapWindowSource
from model.lif2d_stream import detach_states
from test_v14_training import synthetic_record, tiny_config as v14_tiny_config, write_npz
from tools.v16 import train as tr
from utils.comm_budget import CommBudget


def tiny_config():
    cfg = v14_tiny_config()
    cfg.update(channels=[12, 12, 12, 12], readout_hidden=8,
               revision_dim=12, revision_hidden=24, revision_heads=3,
               revision_radius=2, revision_downsample=4,
               revision_variant="full")
    return cfg


def tensors_in(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors_in(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from tensors_in(child)


class V16TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

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

        tr.calibrate_gains(self.net, factory, 0.99, 32, 1, 0.05, 20., 4037)

    def make_nonzero_revision(self):
        with torch.no_grad():
            self.head.revision[-1].weight.normal_(0., 0.02)
            self.head.revision[-1].bias[:12].fill_(0.2)
            self.head.revision[-1].bias[12:].fill_(0.03)

    def test_variants_share_backbone_initialization_but_different_seeds_do_not(self):
        initializations = {}
        rng_after = {}
        for seed in (37, 38):
            for variant in ("full", "no_context", "no_revision"):
                cfg = dict(self.cfg, seed=seed, revision_variant=variant)
                tr.seed_everything(seed, True)
                net, _ = tr.make_models(cfg, self.device)
                initializations[(seed, variant)] = {name: value.clone() for name, value in net.state_dict().items()}
                rng_after[(seed, variant)] = torch.get_rng_state().clone()
        for seed in (37, 38):
            reference = initializations[(seed, "full")]
            for variant in ("no_context", "no_revision"):
                candidate = initializations[(seed, variant)]
                self.assertEqual(set(reference), set(candidate))
                for name in reference:
                    self.assertTrue(torch.equal(reference[name], candidate[name]), (seed, variant, name))
                self.assertTrue(torch.equal(rng_after[(seed, "full")], rng_after[(seed, variant)]))
        self.assertTrue(any(not torch.equal(value, initializations[(38, "full")][name])
                            for name, value in initializations[(37, "full")].items()))

    def forward_record(self, seq):
        source = OverlapWindowSource(seq, self.cfg, self.q99, self.device)
        x, events, labels, ids = source.chunk(0, seq.n_windows)
        logits, states, info = self.net.forward_chunk(x, events, None, collect=True)
        return info["u_pre"][-1], logits, labels, events, ids, states

    def loss_part(self, seq, features, logits, labels, events, ids, lo, hi, pending):
        a, b = int(seq.bounds[lo]), int(seq.bounds[hi])
        ev = {key: value[a:b] for key, value in events.items()}
        ev["t"] = ev["t"] - lo
        segments = [(0, lo, hi, 0, seq, ids[a:b])]
        pieces = [[(0, lo, hi, 0, lo == 0)]]
        return tr.classification_loss(self.net, self.head, features[lo:hi],
                                      logits[a:b], labels[a:b], ev, segments,
                                      pieces, pending, 1.0)

    def test_loss_mass_and_final_ids_do_not_duplicate_overlap(self):
        self.make_nonzero_revision()
        seq = synthetic_record(1)
        features, logits, labels, events, ids, _ = self.forward_record(seq)
        pending = {}
        full, mass, count, cross = self.loss_part(
            seq, features, logits, labels, events, ids, 0, 5, pending)
        self.assertEqual(mass, seq.n_events)
        self.assertEqual(count, seq.n_events)
        self.assertEqual(cross, 0)
        self.assertFalse(pending)
        losses, total_mass, total_count, total_cross = [], 0., 0, 0
        for lo, hi in ((0, 2), (2, 4), (4, 5)):
            loss, m, n, crossed = self.loss_part(
                seq, features, logits, labels, events, ids, lo, hi, pending)
            losses.append(loss)
            total_mass += m
            total_count += n
            total_cross += crossed
            pending = tr.detach_pending(pending)
        self.assertFalse(pending)
        self.assertEqual(total_mass, seq.n_events)
        self.assertEqual(total_count, seq.n_events)
        self.assertGreater(total_cross, 0)
        self.assertTrue(torch.allclose(sum(losses), full, atol=3e-5, rtol=1e-6))

    def test_singletons_empty_windows_and_missing_future(self):
        self.make_nonzero_revision()
        # Every event contributes mass one individually, including cold start,
        # tail fallback and an overlap pair with otherwise empty windows.
        for timestamp in (0, 49, 50, 99, 100, 199, 249, 250, 299):
            seq = OverlapSequence("one", [1], [2], [timestamp], [1], [1.], [1.], total_ms=300)
            features, logits, labels, events, ids, _ = self.forward_record(seq)
            pending, mass, committed = {}, 0., 0
            for lo, hi in ((0, 2), (2, 4), (4, 5)):
                _, m, n, _ = self.loss_part(seq, features, logits, labels, events, ids,
                                           lo, hi, pending)
                mass += m
                committed += n
                pending = tr.detach_pending(pending)
            self.assertEqual(mass, 1., timestamp)
            self.assertEqual(committed, 1, timestamp)
            self.assertFalse(pending)
            output, _, _ = tr.infer_record(self.net, self.head, seq, self.cfg,
                                           self.q99, self.device, step=True)
            for probability in output.values():
                self.assertEqual(probability.shape, (1,))
                self.assertTrue(np.isfinite(probability).all())
            if seq.coverage_count[0] == 1:
                np.testing.assert_array_equal(output["main"], output["second"])
        empty = OverlapSequence("empty", [], [], [], [], [], [], total_ms=300)
        output, _, _ = tr.infer_record(self.net, self.head, empty, self.cfg,
                                       self.q99, self.device, step=True)
        self.assertTrue(all(value.size == 0 for value in output.values()))

    def test_zero_revision_is_exact_second_readout_with_second_local_time(self):
        # Make the readout's dependence on t_local explicit.  Using AB time for
        # a BC output would shift every paired logit by exactly 0.5 here.
        with torch.no_grad():
            for parameter in self.net.readout.parameters():
                parameter.zero_()
            self.net.readout[0].weight[0, -1] = 1.
            self.net.readout[0].bias[0] = 1.
            self.net.readout[2].weight[0, 0] = 1.
        seq = synthetic_record(3)
        features, logits, _, events, ids, _ = self.forward_record(seq)
        fused, rows, crossed = tr.fuse_chunk(
            self.net, self.head, features, events,
            [(0, 0, 5, 0, seq, ids)], [[(0, 0, 5, 0, True)]], {})
        self.assertEqual(crossed, 0)
        index = torch.as_tensor(rows, dtype=torch.long)
        self.assertTrue(torch.equal(fused, logits[index]))
        self.assertTrue(torch.equal(fused, 1. + events["t_local"][index]))
        paired_ids = ids[rows]
        first_times = seq.expanded_t_local[seq.first_position[paired_ids]]
        second_times = seq.expanded_t_local[seq.last_position[paired_ids]]
        np.testing.assert_allclose(first_times - second_times, 0.5, atol=3e-8)

    def test_pending_detaches_and_owns_small_storage_across_chunks(self):
        self.make_nonzero_revision()
        seq = synthetic_record(4)
        source = OverlapWindowSource(seq, self.cfg, self.q99, self.device)
        x, events, labels, ids = source.chunk(0, 2)
        logits, states, info = self.net.forward_chunk(x, events, None, collect=True)
        old_features = info["u_pre"][-1]
        old_features.retain_grad()
        pending = {}
        tr.classification_loss(self.net, self.head, old_features, logits, labels,
                               events, [(0, 0, 2, 0, seq, ids)],
                               [[(0, 0, 2, 0, True)]], pending, 1.)
        self.assertIn(0, pending)
        before = [tensor.detach().clone() for tensor in tensors_in(pending)]
        old_storage = {tensor.storage().data_ptr() for tensor in tensors_in(pending)}
        pending = tr.detach_pending(pending)
        after = list(tensors_in(pending))
        self.assertEqual(len(before), len(after))
        for expected, tensor in zip(before, after):
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
            self.assertTrue(torch.equal(expected, tensor))
            self.assertNotIn(tensor.storage().data_ptr(), old_storage)
        x, events, labels, ids = source.chunk(2, 4)
        logits, _, info = self.net.forward_chunk(x, events, detach_states(states), collect=True)
        loss, _, _, cross = tr.classification_loss(
            self.net, self.head, info["u_pre"][-1], logits, labels, events,
            [(0, 2, 4, 0, seq, ids)], [[(0, 2, 4, 0, False)]], pending, 1.)
        loss.backward()
        self.assertGreater(cross, 0)
        self.assertIsNone(old_features.grad)
        self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0)
                            for p in self.head.parameters()))

    def test_parallel_streams_reset_records_and_match_independent_fusion(self):
        self.make_nonzero_revision()
        records = [synthetic_record(seed) for seed in (10, 11, 12)]
        sources = tr.Sources(records, self.cfg, self.q99, self.device)
        # Mid-chunk transition, a stream with delayed start, and a padded tail.
        pieces = [[(0, 0, 5, 0, True), (2, 0, 5, 5, True)], [(1, 0, 5, 2, True)]]
        x, events, _, keep, segments = tr.assemble(sources, pieces, 10, self.cfg, self.device)
        self.assertEqual(float(keep[5, 0]), 0.)
        self.assertEqual(float(keep[2, 1]), 0.)
        with torch.no_grad():
            _, _, info = self.net.forward_chunk(x, events, None, keep=keep, collect=True)
            pending = {}
            batched, rows, _ = tr.fuse_chunk(self.net, self.head, info["u_pre"][-1],
                                            events, segments, pieces, pending)
            self.assertFalse(pending)
            by_row = {int(row): value for row, value in zip(rows, batched)}
            for index, lo, hi, row, seq, ids in segments:
                xs, es, _, _ = sources.get(index).chunk(lo, hi)
                _, _, individual_info = self.net.forward_chunk(xs, es, None, collect=True)
                independent, local_rows, _ = tr.fuse_chunk(
                    self.net, self.head, individual_info["u_pre"][-1], es,
                    [(index, lo, hi, 0, seq, ids)], [[(index, lo, hi, 0, True)]], {})
                expected = torch.stack([by_row[row + int(local)] for local in local_rows])
                self.assertTrue(torch.allclose(expected, independent, atol=3e-6, rtol=1e-5), index)

    def test_online_layer_inference_and_future_causality_with_active_revision(self):
        self.make_nonzero_revision()
        self.net.eval()
        self.head.eval()
        seq = synthetic_record(13)
        layer, _, layer_messages = tr.infer_record(self.net, self.head, seq, self.cfg,
                                                   self.q99, self.device, diagnostics=True)
        step, timing, step_messages = tr.infer_record(self.net, self.head, seq, self.cfg,
                                                      self.q99, self.device, step=True, diagnostics=True)
        self.assertEqual(set(layer), set(step))
        for mode in layer:
            np.testing.assert_allclose(layer[mode], step[mode], rtol=1e-5, atol=3e-6, err_msg=mode)
        np.testing.assert_allclose(layer_messages, step_messages, rtol=1e-5, atol=1e-5)
        self.assertEqual(len(timing["all_step_ms"]), seq.n_windows)
        changed = copy.deepcopy(seq)
        later = changed.t >= 200
        changed.x[later] = (changed.x[later] + 3) % 8
        changed.y[later] = (changed.y[later] + 2) % 8
        changed.p[later] = 1 - changed.p[later]
        altered, _, _ = tr.infer_record(self.net, self.head, changed, self.cfg,
                                        self.q99, self.device, step=True)
        committed = seq.output_time_ms <= 200
        for mode in step:
            np.testing.assert_array_equal(step[mode][committed], altered[mode][committed], err_msg=mode)
        self.assertTrue(any(np.max(np.abs(step[mode][~committed] - altered[mode][~committed])) > 1e-8
                            for mode in step))

    def test_real_training_multiple_updates_with_communication_and_record_switch(self):
        records = [synthetic_record(seed) for seed in (20, 21, 22)]
        optimizer = torch.optim.Adam(list(self.net.parameters()) + list(self.head.parameters()), lr=1e-3)
        comm = CommBudget.from_config(self.cfg, self.net)
        plan = [(0, [(0, 5), (2, 5)]), (1, [(1, 5)])]
        result = tr.train_epoch(self.net, self.head, records, optimizer, self.cfg,
                                self.q99, 1., self.device, np.random.RandomState(37), comm=comm, plan=plan)
        self.assertEqual(result["supervision_mass"], sum(seq.n_events for seq in records))
        self.assertEqual(result["committed_events"], sum(seq.n_events for seq in records))
        self.assertGreaterEqual(result["updates"], 2)
        self.assertEqual(result["updates"], result["chunks"])
        self.assertGreater(result["cross_tbptt_pairs"], 0)
        self.assertEqual(result["comm"]["mode"], "peak")
        self.assertEqual(result["comm"]["budget"], 20.)
        self.assertGreater(result["comm"]["mu_end"], result["comm"]["mu_start"])
        for group in ("backbone", "readout", "fusion"):
            self.assertGreater(result["grad_groups_mean"][group], 0., group)

    def test_production_reads_each_event_once_and_matches_diagnostic_main(self):
        self.make_nonzero_revision()
        self.net.eval()
        self.head.eval()
        seq = synthetic_record(25)
        diagnostic, _, _ = tr.infer_record(self.net, self.head, seq, self.cfg,
                                           self.q99, self.device, step=True)
        read_rows = []
        hook = self.net.readout[0].register_forward_pre_hook(
            lambda module, inputs: read_rows.append(int(inputs[0].shape[0])))
        try:
            production, _, _ = tr.infer_record(self.net, self.head, seq, self.cfg,
                                               self.q99, self.device, modes=["main"], step=True)
        finally:
            hook.remove()
        self.assertEqual(set(production), {"main"})
        self.assertEqual(sum(read_rows), seq.n_events)
        self.assertLess(sum(read_rows), seq.order.size)
        # Singleton rows use a smaller GEMM batch in production, so FP32
        # rounding need not be bitwise equal to the diagnostic auxiliary pass.
        np.testing.assert_allclose(production["main"], diagnostic["main"], atol=1e-7, rtol=1e-6)
        np.testing.assert_array_equal(production["main"] >= self.cfg["threshold"],
                                      diagnostic["main"] >= self.cfg["threshold"])

    def test_cli_train_then_validation_and_test_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="v16_cli_") as directory:
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

            def run(extra):
                result = subprocess.run(command + extra, cwd=str(ROOT), env=env,
                                        capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)

            run(["--mode", "train"])
            for name in ("last.pt", "best_val_iou_seed37.pt", "train_stats.json", "run_config.json",
                         "calibration.json", "metrics.jsonl", "summary.json"):
                self.assertTrue((output / name).is_file(), name)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["epochs_completed"], cfg["epochs"])
            history = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(history), cfg["epochs"])
            for row in history:
                self.assertTrue(np.isfinite(row["val_iou"]))
                self.assertEqual(row["supervision_mass"], 3 * synthetic_record().n_events)
            checkpoint = output / "best_val_iou_seed37.pt"
            for split in ("val", "test"):
                run(["--mode", "eval", "--checkpoint", str(checkpoint), "--split", split,
                     "--step", "--dump-predictions"])
                report = json.loads((output / ("eval_%s_step.json" % split)).read_text())
                self.assertEqual(report["split"], split)
                self.assertEqual(report["timing"]["execution"], "step")
                for metric in ("iou", "acc", "pd", "fa"):
                    self.assertTrue(np.isfinite(report["metrics"][metric]), (split, metric))
                self.assertEqual(report["timing"]["step_ms"]["count"], 5)
                prediction = next((output / ("predictions_" + split)).glob("*.npz"))
                with np.load(str(prediction)) as values:
                    np.testing.assert_array_equal(values["event_id"], np.arange(report["events"]))
                    self.assertTrue(all(np.isfinite(values[key]).all() for key in values.files if key.startswith("prob_")))
                    self.assertIn("output_time_ms", values.files)

            profile_command = [sys.executable, str(ROOT / "tools/v16/profile_inference.py"),
                               "--config", str(config), "--checkpoint", str(checkpoint),
                               "--data-root", str(data), "--out", str(output),
                               "--device", "cpu", "--split", "test", "--records", "1", "--repeats", "2"]
            profile_run = subprocess.run(profile_command, cwd=str(ROOT), env=env,
                                         capture_output=True, text=True, timeout=180)
            self.assertEqual(profile_run.returncode, 0, profile_run.stdout + "\n" + profile_run.stderr)
            profile = json.loads((output / "profile.json").read_text())
            self.assertEqual(profile["event_reads"], profile["events"])
            self.assertEqual(profile["events"], 2 * synthetic_record().n_events)
            self.assertEqual(profile["windows"], 10)
            self.assertEqual(len(profile["rows"]), 2)
            self.assertIsNone(profile["energy"]["board_joules"])
            self.assertEqual(profile["rows"][0]["prediction_sha256"],
                             profile["rows"][1]["prediction_sha256"])
            self.assertEqual(profile["rows"][1]["repeat_max_abs_diff"], 0.)
            self.assertTrue(np.isfinite(profile["total_wall_seconds"]))


if __name__ == "__main__":
    unittest.main()
