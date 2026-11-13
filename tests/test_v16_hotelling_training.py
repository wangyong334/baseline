"""Real-BIR tests for live-backbone, train-only Hotelling statistics."""
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_v14_training import synthetic_record, write_npz
from test_v16_evidence_training import tiny_config as evidence_config
from tools.v16 import runtime_preflight, train as tr, train_hotelling as ho
from tools.v16.train_covariance import observations
from utils.comm_budget import CommBudget


def tiny_config():
    cfg = evidence_config()
    cfg.update(revision_architecture="hotelling", covariance_feature_source="dec1",
               hotelling_shrinkage=.05, from_scratch=True, require_initialization=False,
               freeze_backbone=False, backbone_adaptation=False, fusion_loss="bce",
               selection_rule="val_iou", validate_initial=False)
    return cfg


class HotellingTrainingTests(unittest.TestCase):
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

    def models(self):
        net, head = tr.make_models(self.cfg, self.device)
        seq = synthetic_record(37)

        def factory():
            source = tr.OverlapWindowSource(seq, self.cfg, self.q99, self.device)

            def windows():
                for k in range(seq.n_windows):
                    x, events, _, _ = source.window(k)
                    yield x, events
            yield windows()

        tr.calibrate_gains(net, factory, .99, 32, 1, .05, 20., 4037)
        return net, head

    def training_files(self, directory):
        data, output = Path(directory) / "data", Path(directory) / "output"
        for split, count in (("train", 2), ("val", 1)):
            folder = data / split
            folder.mkdir(parents=True)
            for i in range(count):
                write_npz(folder / ("%s_%03d.npz" % (split, i)), synthetic_record(i + 40))
        cfg = dict(self.cfg, root=str(data), save_root=str(output),
                   stats_path=str(output / "train_stats.json"))
        return cfg, SimpleNamespace(out=str(output), device="cpu", resume=False,
                                    limit_records=None, max_chunks=None), output

    def test_scratch_protocol_rejects_inheritance_freezing_and_reference_selection(self):
        tr.validate_config(self.cfg)
        for change in ({"from_scratch": False}, {"init_checkpoint": "old.pt"},
                       {"require_initialization": True}, {"freeze_backbone": True},
                       {"backbone_adaptation": True}, {"initialization_policy": "matched"},
                       {"covariance_feature_source": "multilevel"}, {"threshold": .5},
                       {"fusion_loss": "rate_constrained"}, {"validate_initial": True},
                       {"selection_rule": "four_metric_noninferiority"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                tr.validate_config(dict(self.cfg, **change))

    def test_real_two_updates_and_online_repeat_for_all_variants(self):
        for variant in ("full", "no_context", "no_revision", "second"):
            with self.subTest(variant=variant), contextlib.redirect_stdout(io.StringIO()):
                result = runtime_preflight.check(dict(self.cfg, revision_variant=variant), self.device)
            self.assertTrue(result["passed"])
            self.assertEqual(result["training"]["updates"], 2)
            groups = result["training"]["grad_groups_mean"]
            self.assertGreater(groups["backbone"], 0)
            self.assertGreater(groups["readout"], 0)
            if variant != "second":
                self.assertGreater(groups["fusion"], 0)

    def test_refresh_preserves_weights_calibration_and_uses_only_training_data(self):
        net, head = self.models()
        records = [synthetic_record(12), synthetic_record(13)]
        with torch.no_grad():
            head.scale_raw.add_(.7)
            head.bias.add_(.3)
        before_net = ho.backbone_sha256(net)
        before_params = {name: value.detach().clone() for name, value in head.named_parameters()}
        net.train(); head.eval()
        with tempfile.TemporaryDirectory(prefix="v16_hotelling_refresh_") as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                report = ho.refresh_statistics(net, head, records, self.cfg, self.q99,
                                               self.device, directory, -1, "train")
            self.assertEqual(report["backbone_sha256"], before_net)
            self.assertEqual(ho.backbone_sha256(net), before_net)
            self.assertTrue(net.training)
            self.assertFalse(head.training)
            self.assertTrue(all(torch.equal(before_params[name], value)
                                for name, value in head.named_parameters()))
            self.assertEqual(len(report["records"]), len(records))
            self.assertEqual(report["statistics_sha256"], ho.statistics_sha256(head))
            for split in ("val", "test"):
                with self.subTest(split=split), self.assertRaisesRegex(ValueError, "training split"):
                    ho.refresh_statistics(net, head, records, self.cfg, self.q99,
                                          self.device, directory, 0, split)
            wrong_dataset = SimpleNamespace(paths=[Path(directory) / "val" / "record.npz"])
            with self.assertRaisesRegex(ValueError, "non-training path"):
                ho.refresh_statistics(net, head, wrong_dataset, self.cfg, self.q99,
                                      self.device, directory, 0, "train")
            first = json.loads((Path(directory) / "hotelling_stats_initial.json").read_text())
            self.assertEqual(first, report)

    def test_online_scores_equal_train_observations_and_never_fit(self):
        net, head = self.models()
        seq = synthetic_record(14)
        z1, z2, h1, h2 = observations(net, seq, self.cfg, self.q99, self.device)
        paired = seq.coverage_count == 2
        head.fit_statistics(z1[paired], z2[paired], h1[paired], h2[paired], seq.label[paired])
        inputs = [torch.tensor(value[paired]) for value in (z1, z2, h1, h2)]
        expected = z2.copy()
        expected[paired] = head(*inputs).detach().numpy()
        identity = ho.statistics_sha256(head)
        with mock.patch.object(head, "fit_statistics", side_effect=AssertionError("inference fitted statistics")):
            step, _, _ = tr.infer_record(net, head, seq, self.cfg, self.q99,
                                         self.device, modes=["main"], step=True)
            layer, _, _ = tr.infer_record(net, head, seq, self.cfg, self.q99,
                                          self.device, modes=["main"], step=False)
        # infer_record exports probabilities; singleton events use original z2.
        expected = torch.sigmoid(torch.as_tensor(expected)).numpy()
        np.testing.assert_allclose(step["main"], expected, rtol=1e-5, atol=3e-6)
        np.testing.assert_allclose(layer["main"], step["main"], rtol=1e-5, atol=3e-6)
        self.assertEqual(ho.statistics_sha256(head), identity)

    def test_zero_separation_still_trains_backbone_and_original_readout(self):
        net, head = self.models()
        n = 12
        labels = torch.arange(n) % 2
        zeros_h, zeros_z = torch.zeros(n, 12), torch.zeros(n)
        head.fit_statistics(zeros_z, zeros_z, zeros_h, zeros_h, labels)
        self.assertTrue(head.statistics_summary()["zero_separation"])
        before = ho.backbone_sha256(net)
        stats_before = ho.statistics_sha256(head)
        optimizer = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=.001)
        result = tr.train_epoch(net, head, [synthetic_record(20), synthetic_record(21)],
                                optimizer, self.cfg, self.q99, 1., self.device,
                                np.random.RandomState(37), CommBudget.from_config(self.cfg, net))
        self.assertGreater(result["updates"], 1)
        self.assertTrue(np.isfinite(result["loss_per_event"]))
        self.assertGreater(result["grad_groups_mean"]["backbone"], 0)
        self.assertGreater(result["grad_groups_mean"]["readout"], 0)
        self.assertNotEqual(before, ho.backbone_sha256(net))
        self.assertEqual(stats_before, ho.statistics_sha256(head))

    def test_real_scratch_training_resume_and_selected_checkpoint_statistics(self):
        with tempfile.TemporaryDirectory(prefix="v16_hotelling_train_") as directory:
            cfg, args, output = self.training_files(directory)
            original = tr.train_epoch
            calls = [0]

            def interrupt_second_epoch(*a, **kw):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError("simulated interruption")
                return original(*a, **kw)

            with contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch("tools.v16.train_covariance.train_cached", side_effect=AssertionError("frozen route")), \
                    mock.patch.object(ho, "refresh_statistics", wraps=ho.refresh_statistics) as refresh:
                with mock.patch.object(tr, "train_epoch", side_effect=interrupt_second_epoch):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        tr.train(copy.deepcopy(cfg), args)
                self.assertEqual([call.kwargs["epoch"] for call in refresh.call_args_list], [-1, 0])
                # An uncommitted latest report cannot override last.pt's bound report.
                (output / "hotelling_stats_latest.json").write_text('{"uncommitted": true}')
                args.resume = True
                tr.train(copy.deepcopy(cfg), args)
                self.assertEqual([call.kwargs["epoch"] for call in refresh.call_args_list], [-1, 0, 1])
                self.assertTrue(all(call.kwargs["split"] == "train" for call in refresh.call_args_list))
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["epochs_completed"], 2)
            audit = json.loads((output / "hotelling_training_audit.json").read_text())
            self.assertTrue(audit["from_scratch"])
            self.assertIsNone(audit["initial_checkpoint"])
            self.assertTrue(audit["backbone_changed_from_initial"])
            self.assertTrue(audit["selected_backbone_changed_from_initial"])
            self.assertTrue(audit["statistics_refreshed_after_final_epoch"])
            self.assertFalse(audit["test_access_during_training"])
            self.assertEqual(len(audit["statistics_artifact_sha256"]), 3)
            history = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in history], [0, 1])
            for row in history:
                self.assertTrue({"iou", "acc", "pd", "fa"} <= set(row["val_readouts"]["main"]))
                self.assertEqual(row["val_pd"], row["val_readouts"]["main"]["pd"])
                self.assertEqual(row["val_fa"], row["val_readouts"]["main"]["fa"])
                self.assertTrue(all(value > 0 for value in row["grad_groups_mean"].values()))
            ckpt = tr.load_checkpoint(output / "best_val_iou_seed37.pt", self.device)
            net, head = tr.make_models(cfg, self.device)
            net.load_state_dict(ckpt["model"])
            head.load_state_dict(ckpt["fusion"])
            selected = json.loads((output / "hotelling_stats_selected.json").read_text())
            self.assertEqual(ckpt["epoch"] + 1, selected["epoch"])
            self.assertEqual(ho.backbone_sha256(net), selected["backbone_sha256"])
            self.assertEqual(ho.statistics_sha256(head), selected["statistics_sha256"])
            last = tr.load_checkpoint(output / "last.pt", self.device)
            last_net, last_head = tr.make_models(cfg, self.device)
            last_net.load_state_dict(last["model"])
            last_head.load_state_dict(last["fusion"])
            with torch.no_grad():
                last_head.score_bias.add_(.1)
            with self.assertRaisesRegex(ValueError, "do not match"):
                ho.resume_statistics(last_net, last_head, output, int(last["epoch"]))
            net.eval(); head.eval()
            seq = synthetic_record(99)
            q99 = np.asarray(ckpt["stats"]["q99"], np.float32)
            before, _, _ = tr.infer_record(net, head, seq, cfg, q99, self.device, modes=["main"], step=True)
            copied_net, copied_head = tr.make_models(cfg, self.device)
            copied_net.load_state_dict(ckpt["model"])
            copied_head.load_state_dict(ckpt["fusion"])
            after, _, _ = tr.infer_record(copied_net, copied_head, seq, cfg, q99, self.device, modes=["main"], step=True)
            np.testing.assert_array_equal(before["main"], after["main"])
            with mock.patch.object(tr, "train_epoch") as training:
                with self.assertRaisesRegex(ValueError, "Completed training"):
                    tr.train(copy.deepcopy(cfg), args)
                training.assert_not_called()


if __name__ == "__main__":
    unittest.main()
