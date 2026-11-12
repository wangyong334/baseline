"""Integration tests for V14's real BIR backbone, TBPTT and CLI lifecycle.

Uses tiny deterministic event recordings, not a mock backbone or a duplicate
implementation of the fusion/loss. Run with Python 3.8 / torch 1.9:
    python -m unittest discover -s tests -p test_v14_training.py -v
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

from dataset.overlap_windows import OverlapSequence, OverlapWindowSource
from model.lif2d_stream import detach_states
from tools.v14 import train_overlap as tr
from utils.comm_budget import CommBudget
from utils.stream_common import load_flat_config


def tiny_config():
    cfg = load_flat_config(ROOT / "configs/v14/overlap_conditional.yaml")
    cfg.update(height=8, width_px=8, pad_height=8, pad_width=8,
               total_ms=300, n_windows=5, channels=[4, 8, 8, 8], readout_hidden=8,
               streams=2, tbptt_k=2, epochs=2, eval_chunk=2, threshold=0.5,
               calib_sequences=2, calib_windows=3, calib_samples_per_channel=32,
               calib_min_positive=1, train_subset_size=2, train_subset_every=1,
               comm_budget_messages=20.0, timing_warmup_windows=0,
               num_workers=0, deterministic=True)
    return cfg


def synthetic_record(seed=0, n=180, total_ms=300):
    rng = np.random.RandomState(seed)
    # The prefix ensures every temporal/polarity channel has nonzero counts,
    # every evaluation frame has targets, and timestamps include boundaries.
    times = np.concatenate((np.repeat(np.arange(0, total_ms, 10), 2),
                            rng.randint(0, total_ms, size=n)))
    size = times.size
    x, y = rng.randint(0, 8, size=size), rng.randint(0, 8, size=size)
    polarity = np.arange(size) % 2
    label = (np.arange(size) % 3 == 0).astype(np.float32)
    permutation = rng.permutation(size)
    return OverlapSequence("record_%d.npz" % seed, x[permutation], y[permutation],
                           times[permutation], polarity[permutation], label[permutation],
                           label[permutation], total_ms=total_ms)


def write_npz(path, seq):
    norm = np.zeros((seq.n_events, 6), dtype=np.float32)
    norm[:, 3], norm[:, 4], norm[:, 5] = seq.p, seq.label, seq.target_id
    np.savez_compressed(str(path), ev_loc=np.stack((seq.x, seq.y, seq.t), 1), evs_norm=norm)


class TrainingIntegrationTests(unittest.TestCase):
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
        # Match real training's gain calibration; an uncalibrated tiny BIR
        # network can be entirely silent and make state/gradient checks vacuous.
        calibration = synthetic_record(37)
        def factory():
            source = OverlapWindowSource(calibration, self.cfg, self.q99, self.device)
            def windows():
                for k in range(calibration.n_windows):
                    x, events, _, _ = source.window(k)
                    yield x, events
            yield windows()
        tr.calibrate_gains(self.net, factory, 0.99, 32, 1, 0.05, 20., 4037)

    def forward_record(self, seq):
        source = OverlapWindowSource(seq, self.cfg, self.q99, self.device)
        x, events, labels, ids = source.chunk(0, seq.n_windows)
        logits, _, _ = self.net.forward_chunk(x, events, None)
        return source, logits, labels, ids

    def test_unique_event_supervision_mass_across_tbptt_boundary(self):
        seq = synthetic_record()
        source, logits, labels, ids = self.forward_record(seq)
        full_pending = {}
        full, mass, count, cross = tr.classification_loss(
            logits, labels, [(0, 0, seq.n_windows, 0, seq, ids)],
            full_pending, self.head, 1.0)
        self.assertEqual(mass, seq.n_events)
        self.assertEqual(count, seq.n_events)
        self.assertEqual(cross, 0)
        self.assertFalse(full_pending)

        pending, pieces, total_mass, total_count, total_cross = {}, [], 0., 0, 0
        for lo, hi in ((0, 2), (2, 4), (4, 5)):
            a, b = seq.bounds[lo], seq.bounds[hi]
            loss, part_mass, committed, crossed = tr.classification_loss(
                logits[a:b], labels[a:b], [(0, lo, hi, 0, seq, ids[a:b])],
                pending, self.head, 1.0)
            pieces.append(loss)
            total_mass += part_mass
            total_count += committed
            total_cross += crossed
            pending = {key: (event_ids, value.detach()) for key, (event_ids, value) in pending.items()}
        self.assertFalse(pending)
        self.assertEqual(total_mass, seq.n_events)
        self.assertEqual(total_count, seq.n_events)
        self.assertGreater(total_cross, 0)
        self.assertTrue(torch.allclose(sum(pieces), full, atol=2e-5, rtol=1e-6))

        # Single-event recordings test the mass per ID, rather than allowing
        # erroneous positive and negative weights to cancel in a global total.
        for timestamp in (0, 49, 50, 100, 199, 249, 250, 299):
            one = OverlapSequence("one", [1], [2], [timestamp], [1], [1.0], [1.0], total_ms=300)
            _, output, target, one_ids = self.forward_record(one)
            one_pending, one_mass, one_count = {}, 0., 0
            for lo, hi in ((0, 2), (2, 4), (4, 5)):
                a, b = one.bounds[lo], one.bounds[hi]
                _, m, c, _ = tr.classification_loss(
                    output[a:b], target[a:b], [(0, lo, hi, 0, one, one_ids[a:b])],
                    one_pending, self.head, 1.0)
                one_mass += m
                one_count += c
                one_pending = {key: (eid, v.detach()) for key, (eid, v) in one_pending.items()}
            self.assertEqual(one_mass, 1.0, timestamp)
            self.assertEqual(one_count, 1, timestamp)
            self.assertFalse(one_pending)

    def test_final_loss_reaches_both_views_readout_and_backbone(self):
        seq = synthetic_record(1)
        _, logits, labels, ids = self.forward_record(seq)
        logits.retain_grad()
        r1, r2, rows, _, _ = tr.segment_pairs(
            logits, (0, 0, seq.n_windows, 0, seq, ids), {})
        fused = self.head(r1, r2)
        F.binary_cross_entropy_with_logits(fused, labels[torch.as_tensor(rows)]).backward()
        paired_ids = ids[rows]
        first_grad = logits.grad[torch.as_tensor(seq.first_position[paired_ids])]
        second_grad = logits.grad[torch.as_tensor(seq.last_position[paired_ids])]
        self.assertTrue(torch.all(first_grad != 0))
        self.assertTrue(torch.all(second_grad != 0))
        self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0) for p in self.net.readout.parameters()))
        self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0)
                            for name, p in self.net.named_parameters() if not name.startswith("readout.")))
        self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0) for p in self.head.parameters()))

    def test_pending_crosses_tbptt_as_values_without_old_autograd_graph(self):
        seq = synthetic_record(2)
        source = OverlapWindowSource(seq, self.cfg, self.q99, self.device)
        x, ev, _, ids = source.chunk(0, 2)
        old_logits, states, _ = self.net.forward_chunk(x, ev, None)
        old_logits.retain_grad()
        pending = {}
        tr.segment_pairs(old_logits, (0, 0, 2, 0, seq, ids), pending)
        self.assertIn(0, pending)
        pending = {key: (event_ids, values.detach()) for key, (event_ids, values) in pending.items()}
        self.assertFalse(pending[0][1].requires_grad)
        x, ev, label, ids = source.chunk(2, 4)
        logits, _, _ = self.net.forward_chunk(x, ev, detach_states(states))
        logits.retain_grad()
        loss, _, _, cross = tr.classification_loss(
            logits, label, [(0, 2, 4, 0, seq, ids)], pending, self.head, 1.0)
        loss.backward()
        self.assertGreater(cross, 0)
        self.assertIsNone(old_logits.grad)
        self.assertTrue(torch.any(logits.grad != 0))

    def test_parallel_streams_and_record_reset_match_independent_execution(self):
        records = [synthetic_record(seed) for seed in (10, 11, 12)]
        sources = tr.Sources(records, self.cfg, self.q99, self.device)
        # Stream 0 switches recording inside this chunk; stream 1 ends early.
        pieces = [[(0, 0, 5, 0, True), (2, 0, 5, 5, True)], [(1, 0, 5, 0, True)]]
        x, ev, _, keep, segments = tr.assemble(sources, pieces, 10, self.cfg, self.device)
        self.assertEqual(float(keep[0, 0]), 0.)
        self.assertEqual(float(keep[5, 0]), 0.)
        self.assertEqual(float(keep[0, 1]), 0.)
        self.assertTrue(torch.all(keep[1:5] == 1))
        with torch.no_grad():
            batched, _, _ = self.net.forward_chunk(x, ev, None, keep=keep)
            for index, lo, hi, row, seq, ids in segments:
                xs, es, _, expected_ids = sources.get(index).chunk(lo, hi)
                independent, _, _ = self.net.forward_chunk(xs, es, None)
                np.testing.assert_array_equal(ids, expected_ids)
                self.assertTrue(torch.allclose(batched[row:row + len(ids)], independent,
                                               atol=2e-6, rtol=1e-5), index)
            # The comparison is non-vacuous: leaking the previous recording's
            # BIR state must change at least one second-record prediction.
            leaked_keep = keep.clone()
            leaked_keep[5, 0] = 1
            leaked, _, _ = self.net.forward_chunk(x, ev, None, keep=leaked_keep)
            second = segments[1]
            row, size = second[3], len(second[5])
            self.assertGreater(float((leaked[row:row + size] - batched[row:row + size]).abs().max()), 1e-8)

    def test_layer_and_online_step_inference_match_all_readouts(self):
        self.net.eval()
        self.head.eval()
        seq = synthetic_record(13)
        layer, _, layer_messages = tr.infer_record(
            self.net, self.head, seq, self.cfg, self.q99, self.device, diagnostics=True)
        step, timing, step_messages = tr.infer_record(
            self.net, self.head, seq, self.cfg, self.q99, self.device, step=True, diagnostics=True)
        self.assertEqual(set(layer), {"first", "second", "mean", "conditional", "calibration_only"})
        for mode in layer:
            np.testing.assert_allclose(layer[mode], step[mode], rtol=1e-5, atol=2e-6, err_msg=mode)
        np.testing.assert_allclose(layer_messages, step_messages, atol=1e-5, rtol=1e-5)
        self.assertEqual(len(timing["all_step_ms"]), seq.n_windows)

    def test_future_changes_cannot_change_already_committed_outputs(self):
        self.net.eval()
        self.head.eval()
        seq = synthetic_record(14)
        changed = copy.deepcopy(seq)
        later = changed.t >= 200
        changed.x[later] = (changed.x[later] + 3) % 8
        changed.y[later] = (changed.y[later] + 2) % 8
        changed.p[later] = 1 - changed.p[later]
        old, _, _ = tr.infer_record(self.net, self.head, seq, self.cfg, self.q99, self.device)
        new, _, _ = tr.infer_record(self.net, self.head, changed, self.cfg, self.q99, self.device)
        committed = seq.output_time_ms <= 200
        self.assertTrue(np.any(committed) and np.any(~committed))
        for mode in old:
            np.testing.assert_array_equal(old[mode][committed], new[mode][committed], err_msg=mode)
        self.assertGreater(float(np.max(np.abs(old["conditional"][~committed] -
                                                 new["conditional"][~committed]))), 1e-8)

    def test_original_event_polarity_and_local_time_reach_readout(self):
        seq = synthetic_record(15)
        source = OverlapWindowSource(seq, self.cfg, self.q99, self.device)
        x, events, _, _ = source.chunk(0, 2)
        events["p"] = events["p"].clone().requires_grad_(True)
        events["t_local"] = events["t_local"].clone().requires_grad_(True)
        logits, _, _ = self.net.forward_chunk(x, events, None)
        logits.sum().backward()
        self.assertTrue(torch.any(events["p"].grad != 0))
        self.assertTrue(torch.any(events["t_local"].grad != 0))
        self.assertEqual(self.net.readout[0].in_features, self.cfg["channels"][0] + 2)

    def test_nonrolling_state_layout_is_rejected(self):
        cfg = dict(self.cfg, state_layout="dual_interleaved")
        with self.assertRaisesRegex(ValueError, "rolling state chain"):
            tr.validate_config(cfg)

    def test_full_train_epoch_accounts_each_event_and_communication_budget(self):
        records = [synthetic_record(seed) for seed in (20, 21, 22)]
        optimizer = torch.optim.Adam(list(self.net.parameters()) + list(self.head.parameters()), lr=1e-3)
        budget = CommBudget.from_config(self.cfg, self.net)
        # Force differing stream phases and a recording transition mid-chunk.
        plan = [(0, [(0, 5), (2, 5)]), (1, [(1, 5)])]
        result = tr.train_epoch(self.net, self.head, records, optimizer, self.cfg,
                                self.q99, 1.0, self.device, np.random.RandomState(37),
                                comm=budget, plan=plan)
        self.assertEqual(result["supervision_mass"], sum(r.n_events for r in records))
        self.assertEqual(result["committed_events"], sum(r.n_events for r in records))
        self.assertGreater(result["cross_tbptt_pairs"], 0)
        self.assertEqual(result["updates"], result["chunks"])
        self.assertEqual(result["comm"]["mode"], "peak")
        self.assertEqual(result["comm"]["budget"], 20.0)
        self.assertIsNotNone(result["comm"]["messages_est_mean"])
        self.assertGreater(result["comm"]["mu_end"], result["comm"]["mu_start"])
        self.assertGreater(result["comm"]["violation_mean"], 0.)
        for key in ("backbone", "readout", "fusion"):
            self.assertGreater(result["grad_groups_mean"][key], 0., key)

    def test_cli_train_validate_test_checkpoint_and_metric_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="v14_cli_") as directory:
            root = Path(directory)
            data, output = root / "data", root / "output"
            for split, count in (("train", 3), ("val", 1), ("test", 1)):
                folder = data / split
                folder.mkdir(parents=True)
                for i in range(count):
                    write_npz(folder / ("%s_%03d.npz" % (split, i)), synthetic_record(i + 30))
            cfg = dict(self.cfg, root=str(data), save_root=str(output),
                       stats_path=str(output / "train_stats.json"))
            config = root / "tiny.yaml"
            config.write_text(yaml.safe_dump({"EXPERIMENT": cfg}), encoding="utf-8")
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            base = [sys.executable, str(ROOT / "tools/v14/train_overlap.py"),
                    "--config", str(config), "--out", str(output), "--device", "cpu"]

            def run(extra, output_dir=None, expected_error=None):
                command = list(base)
                if output_dir is not None:
                    command[command.index("--out") + 1] = str(output_dir)
                result = subprocess.run(command + extra, cwd=str(ROOT), env=env,
                                        capture_output=True, text=True, timeout=120)
                if expected_error is None:
                    self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
                    self.assertIn(expected_error, result.stdout + "\n" + result.stderr)
                return result

            run(["--mode", "train"])
            for name in ("last.pt", "best_val_iou_seed37.pt", "train_stats.json", "run_config.json",
                         "calibration.json", "metrics.jsonl", "summary.json"):
                self.assertTrue((output / name).is_file(), name)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertFalse(summary["probe_only"])
            self.assertEqual(summary["training_scope"], {"limit_records": None, "max_chunks": None})
            self.assertEqual(summary["epochs_completed"], cfg["epochs"])
            history = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(history), cfg["epochs"])
            for row in history:
                self.assertTrue(np.isfinite(row["val_iou"]))
                self.assertEqual(row["comm"]["mode"], "peak")
                self.assertEqual(row["supervision_mass"], 3 * synthetic_record().n_events)
                self.assertGreater(row["grad_groups_mean"]["fusion"], 0.)
            checkpoint = output / "best_val_iou_seed37.pt"
            saved = tr.load_checkpoint(checkpoint, self.device)
            self.assertEqual(saved["design"], tr.DESIGN)
            self.assertIn("model", saved)
            self.assertIn("fusion", saved)
            self.assertEqual(saved["config"]["streams"], 2)
            self.assertEqual(saved["config"]["tbptt_k"], 2)
            for split in ("val", "test"):
                run(["--mode", "eval", "--checkpoint", str(checkpoint), "--split", split,
                     "--step", "--dump-predictions"])
                report = json.loads((output / ("eval_%s_step.json" % split)).read_text())
                self.assertEqual(report["split"], split)
                self.assertEqual(report["timing"]["execution"], "step")
                for metric in ("iou", "acc", "pd", "fa"):
                    self.assertTrue(np.isfinite(report["metrics"][metric]), (split, metric))
                self.assertEqual(report["timing"]["step_ms"]["count"], 5)
                self.assertEqual(report["communication"]["budget"], 20.0)
                self.assertEqual(report["operations"]["energy_joules"], None)
                self.assertEqual(len(report["checkpoint_sha256"]), 64)
                prediction = next((output / ("predictions_" + split)).glob("*.npz"))
                with np.load(str(prediction)) as values:
                    np.testing.assert_array_equal(values["event_id"], np.arange(report["events"]))
                    self.assertTrue(np.isfinite(values["prob_conditional"]).all())
                    self.assertIn("output_time_ms", values.files)

            # A completed full run may load its checkpoint and resume as a
            # no-op; it must not duplicate history or replace the trained model.
            history_before = (output / "metrics.jsonl").read_bytes()
            last_sha = tr.sha256(output / "last.pt")
            run(["--mode", "train", "--resume"])
            self.assertEqual((output / "metrics.jsonl").read_bytes(), history_before)
            self.assertEqual(tr.sha256(output / "last.pt"), last_sha)
            self.assertTrue(json.loads((output / "summary.json").read_text())["complete"])

            # Reuse this tiny dataset for one genuinely truncated run. Merely
            # reaching its requested epoch count must not label it full/complete.
            probe = root / "probe_output"
            run(["--mode", "train", "--max-chunks", "1", "--limit-records", "1"], output_dir=probe)
            probe_summary = json.loads((probe / "summary.json").read_text())
            self.assertEqual(probe_summary["epochs_completed"], cfg["epochs"])
            self.assertTrue(probe_summary["probe_only"])
            self.assertFalse(probe_summary["complete"])
            self.assertEqual(probe_summary["training_scope"], {"limit_records": 1, "max_chunks": 1})
            probe_saved = tr.load_checkpoint(probe / "last.pt", self.device)
            self.assertEqual(probe_saved["config"]["_training_scope"], probe_summary["training_scope"])
            probe_sha = tr.sha256(probe / "last.pt")
            probe_history = (probe / "metrics.jsonl").read_bytes()
            # Removing both limits, or either one alone, changes the experiment
            # scope and must fail before mutating checkpoints/history.
            for limits in ([], ["--max-chunks", "1"], ["--limit-records", "1"]):
                run(["--mode", "train", "--resume"] + limits, output_dir=probe,
                    expected_error="Resume config/design mismatch")
                self.assertEqual(tr.sha256(probe / "last.pt"), probe_sha)
                self.assertEqual((probe / "metrics.jsonl").read_bytes(), probe_history)
            run(["--mode", "train", "--max-chunks", "0"], output_dir=root / "invalid_probe",
                expected_error="--max-chunks must be positive")
            self.assertFalse((root / "invalid_probe").exists())


if __name__ == "__main__":
    unittest.main()
