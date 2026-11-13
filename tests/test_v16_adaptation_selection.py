"""Checkpoint selection uses the fixed initial four-metric validation reference.

Exercise the real train/resume/checkpoint lifecycle with tiny recordings and
mocked optimizer/evaluation work. No actual backbone retraining is required.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_v14_training import synthetic_record, write_npz
from test_v16_evidence_training import tiny_config
from tools.v16 import train as tr


class AdaptationSelectionTests(unittest.TestCase):
    reference = {"iou": .8, "acc": .9, "pd": .85, "fa": 1e-5}

    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def files(self, directory, **updates):
        root = Path(directory)
        data, out = root / "data", root / "output"
        for split, count in (("train", 2), ("val", 1)):
            folder = data / split
            folder.mkdir(parents=True)
            for index in range(count):
                write_npz(folder / ("%s_%03d.npz" % (split, index)),
                          synthetic_record(100 + index))
        cfg = tiny_config()
        cfg.update(root=str(data), save_root=str(out),
                   stats_path=str(out / "train_stats.json"),
                   validate_initial=True,
                   selection_rule="four_metric_noninferiority",
                   epochs=4, early_stop_patience=0)
        cfg.update(updates)
        tr.validate_config(cfg)
        args = SimpleNamespace(out=str(out), device="cpu", resume=False,
                               limit_records=None, max_chunks=None)
        return cfg, args, out

    @staticmethod
    def result(metrics):
        return {"metrics": dict(metrics), "readouts": {"main": dict(metrics)}}

    def training(self, cfg, args, rows, stop_after=None):
        calls = [0]

        def epoch(net, *unused, **kwargs):
            if stop_after is not None and calls[0] == stop_after:
                raise RuntimeError("simulated interruption")
            calls[0] += 1
            with torch.no_grad():
                next(net.parameters()).add_(.001)
            return {"train_seconds": .001, "loss_per_event": .1}

        with mock.patch.object(tr, "calibrate"), \
                mock.patch.object(tr, "subset_due", return_value=False), \
                mock.patch.object(tr, "train_epoch", side_effect=epoch), \
                mock.patch.object(tr, "evaluate", side_effect=[self.result(r) for r in rows]) as evaluate, \
                contextlib.redirect_stdout(io.StringIO()):
            if stop_after is None:
                tr.train(copy.deepcopy(cfg), args)
            else:
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    tr.train(copy.deepcopy(cfg), args)
            # Pd/Fa must be computed for the initial reference and every epoch.
            self.assertEqual(evaluate.call_count, len(rows))
            self.assertTrue(all(call.kwargs.get("full") is True for call in evaluate.call_args_list))

    @staticmethod
    def history(out):
        return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]

    def best(self, out):
        return tr.load_checkpoint(out / "best_val_iou_seed37.pt", torch.device("cpu"))

    def test_rejects_any_metric_regression_and_accepts_four_metric_noninferiority(self):
        with tempfile.TemporaryDirectory(prefix="v16_selection_") as directory:
            cfg, args, out = self.files(directory)
            rows = [self.reference,
                    dict(self.reference, iou=.91, acc=.89),
                    dict(self.reference, iou=.92, pd=.84),
                    dict(self.reference, iou=.93, fa=1.1e-5),
                    dict(self.reference, iou=.85)]
            self.training(cfg, args, rows)
            self.assertEqual([r["selection_eligible"] for r in self.history(out)],
                             [False, False, False, True])
            best = self.best(out)
            self.assertEqual(best["epoch"], 3)
            self.assertEqual(best["best_val_iou"], .85)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["selection_reference"], self.reference)
            self.assertEqual(summary["best_epoch"], 4)

    def test_no_eligible_epoch_preserves_initial_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="v16_selection_fallback_") as directory:
            cfg, args, out = self.files(directory, epochs=2)
            self.training(cfg, args, [self.reference,
                                     dict(self.reference, iou=.94, acc=.89),
                                     dict(self.reference, iou=.95, pd=.84)])
            self.assertEqual(self.best(out)["epoch"], -1)
            self.assertEqual(self.best(out)["best_val_iou"], .8)
            self.assertEqual(tr.load_checkpoint(out / "last.pt", torch.device("cpu"))["epoch"], 1)
            summary = json.loads((out / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["best_epoch"], 0)

    def test_resume_keeps_initial_reference_not_previous_selected_epoch(self):
        with tempfile.TemporaryDirectory(prefix="v16_selection_resume_") as directory:
            cfg, args, out = self.files(directory, epochs=3)
            improved = {"iou": .82, "acc": .95, "pd": .93, "fa": .8e-5}
            self.training(cfg, args, [self.reference, improved], stop_after=1)
            initial_bytes = (out / "initial_validation.json").read_bytes()
            args.resume = True
            # Better IoU, but lower ACC/Pd and higher Fa than epoch 0: still
            # eligible relative to the immutable initial reference.
            next_metrics = {"iou": .83, "acc": .91, "pd": .86, "fa": .9e-5}
            self.training(cfg, args, [next_metrics, dict(next_metrics, iou=.84)])
            self.assertEqual((out / "initial_validation.json").read_bytes(), initial_bytes)
            self.assertTrue(all(r["selection_eligible"] for r in self.history(out)))
            self.assertEqual(self.best(out)["epoch"], 2)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["selection_reference"], self.reference)

    def test_ineligible_tie_does_not_reset_patience_after_resume(self):
        with tempfile.TemporaryDirectory(prefix="v16_selection_patience_") as directory:
            cfg, args, out = self.files(directory, epochs=5, early_stop_patience=2)
            # Same IoU as the initial checkpoint, but ineligible by ACC.
            self.training(cfg, args, [self.reference, dict(self.reference, acc=.89)], stop_after=1)
            args.resume = True
            self.training(cfg, args, [dict(self.reference, iou=.81, acc=.89)])
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertTrue(summary["early_stopped"])
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["best_epoch"], 0)
            self.assertEqual(self.best(out)["epoch"], -1)


if __name__ == "__main__":
    unittest.main()
