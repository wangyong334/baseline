"""Frozen-backbone covariance release contracts; no server or GPU required."""
import copy
import io
from pathlib import Path
import shutil
import unittest
from contextlib import redirect_stdout
from unittest import mock

from tests import test_v16_queue as legacy


ROOT, queue, runner = legacy.ROOT, legacy.queue, legacy.runner


class V16R4PipelineTests(unittest.TestCase):
    plan = legacy.V16QueueTests.plan
    enqueue = legacy.V16QueueTests.enqueue
    args = legacy.V16QueueTests.args

    def setUp(self):
        legacy.V16QueueTests.setUp(self)
        self.release = "context_20261113_r4"
        config = self.project / "configs/v16/r4"
        config.mkdir(parents=True)
        for variant in queue.VARIANTS:
            shutil.copyfile(str(ROOT / "configs/v16/r4" / (variant + ".yaml")),
                            str(config / (variant + ".yaml")))
        (self.project / "model/covariance_revision.py").write_text("# covariance fixture\n", encoding="utf-8")
        import torch
        self.initial = self.project / "initial"
        self.initial.mkdir()
        torch.save({"config": {"seed": 37}, "model": {"weight": torch.ones(1)},
                    "stats": {"q99": [1.]}}, str(self.initial / "seed37.pt"))

    def r4_plan(self, **kwargs):
        options = dict(config_dir="configs/v16/r4", variants=("full",),
                       seeds=(37,), init_root=self.initial)
        options.update(kwargs)
        return self.plan(**options)

    def test_matched_ablations_preserve_frozen_v5_and_fixed_threshold(self):
        configs = runner.validate_configs(self.project, "configs/v16/r4")
        normalized = []
        for variant, raw in configs.items():
            flat = runner.flatten(raw)
            self.assertTrue(flat["freeze_backbone"])
            self.assertEqual(flat["fusion_loss"], "rate_constrained")
            self.assertEqual(flat["threshold"], .9)
            self.assertEqual(flat["selection_rule"], "four_metric_noninferiority")
            self.assertEqual(flat["constraint"], "smooth_fn_reference")
            self.assertEqual(flat["state_mode"], "carry")
            self.assertEqual(flat["comm_budget_messages"], 157700.)
            self.assertEqual((flat["streams"], flat["tbptt_k"]), (4, 16))
            self.assertEqual(flat["revision_variant"], variant)
            item = copy.deepcopy(raw)
            item["CONTEXT"].pop("revision_variant")
            item["DATA"].pop("stats_path")
            item["TRAIN"].pop("save_root")
            normalized.append(item)
        self.assertTrue(all(item == normalized[0] for item in normalized))

    def test_protocol_rejects_unfrozen_backbone_or_invalid_statistics_settings(self):
        flat = runner.flatten(runner.load_yaml(self.project / "configs/v16/r4/full.yaml"))
        changes = ({"freeze_backbone": False}, {"require_initialization": False},
                   {"validate_initial": False}, {"threshold": .8},
                   {"fusion_loss": "bce"}, {"covariance_channels": 0},
                   {"covariance_shrinkage": 0.}, {"covariance_shrinkage": float("nan")},
                   {"recall_tolerance": -.1}, {"recall_dual_lr": 0.},
                   {"recall_tolerance": .001}, {"pd_tolerance": .001},
                   {"selection_rule": "iou_only"}, {"constraint": "binary_fnr"},
                   {"min_epochs": 21}, {"early_stop_patience": 0})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_covariance_protocol(dict(flat, **change))

    def test_requires_seed_matched_initialization_before_publication(self):
        with self.assertRaisesRegex(ValueError, "requires --init-root"):
            self.r4_plan(init_root=None)
        with self.assertRaisesRegex(ValueError, "seed38"):
            self.r4_plan(seeds=(38,))
        self.assertFalse((self.project / "run").exists())
        self.assertFalse((self.scheduler / "run").exists())

    def test_single_full37_release_snapshots_model_and_initialization(self):
        arguments = ["enqueue", "--project-root", str(self.project),
                     "--scheduler-root", str(self.scheduler), "--data-root", str(self.data),
                     "--release", self.release, "--config-dir", "configs/v16/r4",
                     "--variants", "full", "--seeds", "37", "--init-root", str(self.initial)]
        with mock.patch.object(queue.sys, "platform", "linux"), \
             mock.patch.object(queue, "dependency_check", return_value="torch=fixture"), \
             mock.patch.object(queue, "live_pool", return_value=1234), \
             redirect_stdout(io.StringIO()) as output:
            queue.main(arguments + ["--dry-run"])
            self.assertFalse((self.project / "run").exists())
            queue.main(arguments)
        self.assertIn("DRY_RUN_OK: 1 training job(s); GPU 1", output.getvalue())
        manifest, plan = queue.release_plan(self.project, self.release)
        self.assertEqual([(j["variant"], j["seed"]) for j in plan["jobs"]], [("full", 37)])
        self.assertTrue(plan["jobs"][0]["name"].startswith("1280_"))
        self.assertIn("model/covariance_revision.py", plan["files"])
        self.assertIn("configs/v16/r4/full.yaml", plan["files"])
        self.assertEqual(set(plan["initializations"]), {"37"})
        scripts = list((self.scheduler / "run/queue/pool/pending").glob("*.sh"))
        self.assertEqual(len(scripts), 1)
        script = scripts[0].read_text(encoding="utf-8")
        for token in ("configs/v16/r4", "--init-sha256", "CUDA_VISIBLE_DEVICES=1"):
            self.assertIn(token, script)
        self.assertEqual(queue.verify_source(manifest)["status"], "queued")

    def test_runner_keeps_preflight_train_val_test_profile_order(self):
        plan = self.r4_plan()
        self.enqueue(plan)
        initialize = plan["initializations"]["37"]
        args = self.args()
        args.config_dir = "configs/v16/r4"
        args.init_checkpoint = initialize["path"]
        args.init_sha256 = initialize["sha256"]
        def execute(command, source):
            legacy.V16QueueTests.fake_execute(self, command, source)
            if "--mode" in command and command[-1] == "train":
                out = Path(command[command.index("--out") + 1])
                runner.write_json(out / "summary.json", {"epochs_completed": 20,
                                  "complete": True, "probe_only": False})
        with mock.patch.object(runner, "ROOT", self.project), \
             mock.patch.object(runner, "execute", side_effect=execute) as run, \
             redirect_stdout(io.StringIO()):
            result = runner.run(args)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["trained_from_scratch"])
        self.assertEqual(run.call_count, 5)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(commands[0][3].endswith("runtime_preflight.py"))
        self.assertEqual(commands[1][-2:], ["--mode", "train"])
        self.assertEqual(commands[2][-2:], ["--split", "val"])
        self.assertEqual(commands[3][-2:], ["--split", "test"])
        self.assertTrue(commands[4][3].endswith("profile_inference.py"))

    def test_preflight_statistics_are_fitted_before_nonzero_head_gradient_check(self):
        import torch
        from model.covariance_revision import CovarianceRevision
        from tools.v16.runtime_preflight import fit_preflight_statistics, activate_head_for_preflight
        head = CovarianceRevision(feature_dim=60, channels=8, hidden=8)
        report = fit_preflight_statistics(head, torch.device("cpu"))
        self.assertTrue(report["fitted"])
        self.assertEqual(report["training_pairs"], 256)
        self.assertEqual(report["class_counts"], [128, 128])
        before = {name: value.clone() for name, value in head.named_buffers()}
        activate_head_for_preflight(head, "covariance")
        self.assertGreater(float(head.output.weight.detach().abs().sum()), 0.)
        z1, z2 = torch.randn(8), torch.randn(8)
        h1 = torch.randn(8, 60, requires_grad=True)
        h2 = torch.randn(8, 60, requires_grad=True)
        head(z1, z2, h1, h2).sum().backward()
        self.assertGreater(float(h1.grad.abs().sum()), 0.)
        self.assertGreater(float(h2.grad.abs().sum()), 0.)
        self.assertTrue(all(torch.equal(value, dict(head.named_buffers())[name])
                            for name, value in before.items()))

    def test_smooth_constraint_uses_same_risk_in_primal_and_dual(self):
        import torch
        from tools.v16.train_covariance import constrained_loss, update_recall_multiplier
        logits = torch.tensor([0., 3., 1., 4.], requires_grad=True)
        labels = torch.tensor([0., 0., 1., 1.])
        boundary = torch.log(torch.tensor(9.))
        fp = torch.nn.functional.softplus(logits[:2] - boundary).mean()
        fn = torch.nn.functional.softplus(boundary - logits[2:]).mean()
        multiplier, reference = .2, .4
        loss = constrained_loss(logits, labels, multiplier, reference_fn=reference)
        self.assertTrue(torch.allclose(loss, fp + multiplier * (fn-reference)))
        loss.backward()
        self.assertTrue(bool((logits.grad[:2] > 0).all()))
        self.assertTrue(bool((logits.grad[2:] < 0).all()))
        self.assertGreater(update_recall_multiplier(.2, .5, .4, .5), .2)
        self.assertEqual(update_recall_multiplier(.2, .4, .4, .5), .2)
        self.assertEqual(update_recall_multiplier(.01, .1, .4, .5), 0.)

    def test_selection_rejects_each_metric_regression_and_nonfinite_value(self):
        from tools.v16.train_covariance import four_metric_noninferiority
        reference = dict(iou=.86, acc=.91, pd=.92, fa=6e-6)
        self.assertTrue(four_metric_noninferiority(reference, reference))
        better = dict(iou=.87, acc=.92, pd=.93, fa=5e-6)
        self.assertTrue(four_metric_noninferiority(better, reference))
        for key, value in [('iou', .85), ('acc', .909999), ('pd', .919999), ('fa', 6.00001e-6)]:
            with self.subTest(key=key):
                self.assertFalse(four_metric_noninferiority(dict(better, **{key: value}), reference))
                self.assertFalse(four_metric_noninferiority(dict(better, **{key: float('nan')}), reference))


if __name__ == "__main__":
    unittest.main()
