"""Overlap deployment contract and isolated runner tests, without a GPU/server."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = load("test_overlap_queue", ROOT / "tools/v14/overlap_queue.py")
runner = load("test_overlap_runner", ROOT / "tools/v14/run_overlap.py")


class OverlapQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.project = base / "overlap project"
        self.scheduler = base / "existing V13 pool"
        self.data = base / "data"
        self.release = "overlap_20261112_r1"
        for relative in ("train_stream_v1.py", "tools/v14/train_overlap.py", "tools/v14/profile_overlap.py",
                         "tools/v14/queue.py", "tools/v14/overlap_queue.py", "model/lif.py",
                         "dataset/stream.py", "utils/config.py"):
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n", encoding="utf-8")
        for relative in ("tools/v14/run_overlap.py", "tools/v14/run_experiment.py",
                         "configs/evisseg_stream_tif_comm_peak.yaml",
                         "configs/v14/overlap_conditional.yaml", "configs/v14/overlap_mean.yaml",
                         "configs/v14/overlap_concat.yaml"):
            target = self.project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(ROOT / relative), str(target))
        for relative in ("tools/final/gpu_pool.sh", "tools/final/gpu_queue.sh"):
            path = self.scheduler / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# existing pool\n", encoding="utf-8")
        for split, count in queue.SPLIT_COUNTS.items():
            directory = self.data / split
            directory.mkdir(parents=True)
            for index in range(count):
                (directory / ("%s_%03d.npz" % (split, index))).write_bytes(b"fixture")

    def plan(self, **kwargs):
        return queue.prepare_plan(self.project, self.scheduler, self.data, self.release,
                                  runtime_checks=False, **kwargs)

    def test_default_is_only_conditional_seed37_and_preflight_is_read_only(self):
        plan = self.plan()
        self.assertEqual([(j["variant"], j["seed"]) for j in plan["jobs"]], [("conditional", 37)])
        self.assertEqual(plan["physical_gpu"], 1)
        self.assertFalse((self.project / "run").exists())
        self.assertFalse((self.scheduler / "run").exists())

    def test_ablation_training_requires_explicit_request(self):
        plan = self.plan(variants=("mean", "concat"))
        self.assertEqual([j["variant"] for j in plan["jobs"]], ["mean", "concat"])
        self.assertNotEqual(plan["jobs"][0]["out"], plan["jobs"][1]["out"])

    def test_snapshot_excludes_sftp_and_uses_existing_gpu1_pool(self):
        secret = self.project / ".vscode/sftp.json"
        secret.parent.mkdir()
        secret.write_text("private credentials", encoding="utf-8")
        plan = self.plan()
        manifest = queue.enqueue(plan)
        self.assertEqual(queue.verify_source(manifest)["status"], "queued")
        self.assertFalse((Path(plan["source"]) / ".vscode").exists())
        scripts = list((self.scheduler / "run/queue/pool/pending").glob("*.sh"))
        self.assertEqual(len(scripts), 1)
        text = scripts[0].read_text(encoding="utf-8")
        for item in ("CUDA_VISIBLE_DEVICES=1", "verify-source", "run_overlap.py", "--variant conditional"):
            self.assertIn(item, text)
        for item in ("nohup", "kill ", "gpu_pool.sh", "gpu_queue.sh", "--overwrite", "--resume"):
            self.assertNotIn(item, text)

    def test_duplicate_release_and_completed_queue_marker_are_rejected(self):
        plan = self.plan()
        manifest = queue.enqueue(plan)
        for path in (self.scheduler / "run/queue/pool/pending").glob("*.sh"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "release already exists"):
            self.plan()
        self.assertTrue(manifest.is_file())

    def test_given_failed_done_markers_and_existing_outputs_are_rejected(self):
        for part in ("pool/given", "gpu1/done", "gpu1/failed"):
            directory = self.scheduler / "run/queue" / part
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / ("1100_v14_overlap_%s_conditional_s37.sh" % self.release)
            path.touch()
            with self.assertRaisesRegex(ValueError, "release jobs already exist"):
                self.plan()
            path.unlink()
        out = self.project / "log/v14_overlap" / self.release / "conditional_s37"
        out.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "output already exists"):
            self.plan()

    def test_upload_mutation_before_enqueue_publishes_nothing(self):
        plan = self.plan()
        (self.project / "model/lif.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source changed after preflight"):
            queue.enqueue(plan)
        self.assertEqual(list((self.scheduler / "run/queue/pool/pending").glob("*.sh")), [])

    def test_snapshot_mutation_is_detected(self):
        plan = self.plan()
        manifest = queue.enqueue(plan)
        (Path(plan["source"]) / "model/lif.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            queue.verify_source(manifest)

    def test_representation_stats_and_v5_budget_contract(self):
        raw = runner.runtime_config("conditional", 37, self.release, self.data, self.project, self.project)
        self.assertEqual(raw["DATA"]["n_windows"], 159)
        self.assertEqual(raw["DATA"]["stride_ms"], 50)
        self.assertEqual(raw["NET"]["state_layout"], "single_rolling")
        self.assertEqual(raw["TRAIN"]["comm_budget_messages"], 157700.0)
        self.assertEqual(raw["TRAIN"]["execution"], "layer")
        self.assertEqual(raw["TRAIN"]["eval_execution"], "step")
        self.assertEqual(Path(raw["DATA"]["stats_path"]).parent, Path(raw["TRAIN"]["save_root"]))
        path = self.project / "configs/v14/overlap_conditional.yaml"
        path.write_text(path.read_text(encoding="utf-8").replace("157700.0", "0.0"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changes V5 TRAIN.comm_budget_messages"):
            runner.validate_configs(self.project)

    def test_incomplete_results_cannot_be_packaged_as_complete(self):
        queue.enqueue(self.plan())
        with self.assertRaisesRegex(ValueError, "not complete"):
            queue.pack(self.project, self.release)
        self.assertFalse((self.project / "Res").exists())

    def test_complete_package_contains_weights_metrics_source_and_scheduler_log(self):
        plan = self.plan()
        queue.enqueue(plan)
        out = Path(plan["jobs"][0]["out"])
        out.mkdir(parents=True)
        checkpoint = out / "best_val_iou_seed37.pt"
        checkpoint.write_bytes(b"trained checkpoint fixture")
        for name in ("eval_val.json", "eval_test.json", "profile.json", "train_stats.json"):
            (out / name).write_text("{}", encoding="utf-8")
        (out / "metrics.jsonl").write_text('{"epoch": 1}\n', encoding="utf-8")
        (out / "comparison_result.json").write_text(json.dumps({"status": "complete", "checkpoint_sha256": queue.digest(checkpoint)}), encoding="utf-8")
        log = self.scheduler / "run/queue/gpu1/logs" / (plan["jobs"][0]["name"] + ".log")
        log.parent.mkdir(parents=True)
        log.write_text("training complete\n", encoding="utf-8")
        archive = queue.pack(self.project, self.release)
        with tarfile.open(str(archive)) as handle:
            names = handle.getnames()
        self.assertTrue(any(n.endswith("best_val_iou_seed37.pt") for n in names))
        self.assertTrue(any(n.endswith("metrics.jsonl") for n in names))
        self.assertTrue(any(n.startswith("scheduler_logs/") for n in names))
        self.assertTrue(any("/source/tools/v14/train_overlap.py" in n for n in names))
        self.assertFalse(any(".vscode" in n or n.endswith(".npz") for n in names))

    def test_runner_trains_then_evaluates_val_test_then_profiles_and_prevents_rerun(self):
        calls = []
        metrics = {"iou": .8, "acc": .9, "pd": .85, "fa": .00001}
        def execute(command, source):
            calls.append(command)
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            if "profile_overlap.py" in command[3]:
                profile = {"records": 3, "checkpoint_sha256": runner.sha256(out / "best_val_iou_seed37.pt"),
                           "readout_policy": "conditional", "split": "test", "total_wall_seconds": 2.,
                           "energy": {"board_joules": None}}
                (out / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
            elif command[command.index("--mode") + 1] == "train":
                (out / "summary.json").write_text('{"epochs_completed": 50, "complete": true, "probe_only": false}', encoding="utf-8")
                (out / "best_val_iou_seed37.pt").write_bytes(b"weights")
            else:
                split = command[command.index("--split") + 1]
                value = {"metrics": metrics, "timing": {"execution": "step"},
                         "checkpoint_sha256": runner.sha256(out / "best_val_iou_seed37.pt"),
                         "readouts": {m: metrics for m in ("first", "second", "mean", "conditional", "calibration_only")}}
                (out / ("eval_%s.json" % split)).write_text(json.dumps(value), encoding="utf-8")
        args = argparse.Namespace(project_root=self.project, data_root=self.data, release=self.release,
                                  variant="conditional", seed=37, device="cpu")
        with mock.patch.object(runner, "ROOT", self.project), mock.patch.object(runner, "execute", side_effect=execute):
            result = runner.run(args)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[1][calls[1].index("--split") + 1], "val")
            self.assertEqual(calls[2][calls[2].index("--split") + 1], "test")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                runner.run(args)

    def test_profile_cannot_be_attributed_to_different_weights_or_readout(self):
        path = self.project / "profile.json"
        value = {"checkpoint_sha256": "wrong", "readout_policy": "conditional", "split": "test",
                 "total_wall_seconds": 2., "energy": {"board_joules": None}}
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checkpoint identity"):
            runner.validate_profile(path, "conditional", "selected")
        value.update(checkpoint_sha256="selected", readout_policy="mean")
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "trained main readout"):
            runner.validate_profile(path, "conditional", "selected")

    def test_formal_runner_rejects_incomplete_probe_or_unmarked_summary(self):
        for index, fields in enumerate(({"complete": False, "probe_only": False},
                                       {"complete": True, "probe_only": True},
                                       {"complete": True}, {})):
            with self.subTest(fields=fields):
                def execute(command, source):
                    out = Path(command[command.index("--out") + 1])
                    out.mkdir(parents=True)
                    summary = dict(epochs_completed=50, **fields)
                    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                args = argparse.Namespace(project_root=self.project, data_root=self.data,
                                          release=self.release + "_bad%d" % index,
                                          variant="conditional", seed=37, device="cpu")
                with mock.patch.object(runner, "ROOT", self.project), \
                     mock.patch.object(runner, "execute", side_effect=execute) as call:
                    with self.assertRaisesRegex(ValueError, "complete, non-probe"):
                        runner.run(args)
                    self.assertEqual(call.call_count, 1)

    def test_formal_evaluation_must_use_step_but_training_stays_layer(self):
        configs = runner.validate_configs(self.project)
        for raw in configs.values():
            self.assertEqual(raw["TRAIN"]["execution"], "layer")
            self.assertEqual(raw["TRAIN"]["eval_execution"], "step")
        path = self.project / "eval_test.json"
        for timing in ({}, {"execution": "layer"}):
            path.write_text(json.dumps({"timing": timing}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must use step"):
                runner.validate_evaluation(path, "conditional")
        config = self.project / "configs/v14/overlap_conditional.yaml"
        config.write_text(config.read_text(encoding="utf-8").replace("eval_execution: step", "eval_execution: layer"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "requires eval_execution='step'"):
            self.plan()

    def test_invalid_dataset_stop_and_duplicate_choices_are_rejected(self):
        with self.assertRaises(ValueError):
            self.plan(seeds=(37, 37))
        with self.assertRaises(ValueError):
            self.plan(variants=("conditional", "conditional"))
        stop = self.scheduler / "run/queue/gpu1/STOP"
        stop.parent.mkdir(parents=True)
        stop.touch()
        with self.assertRaisesRegex(ValueError, "STOP request"):
            self.plan()
        stop.unlink()
        (self.data / "val/val_000.npz").unlink()
        with self.assertRaisesRegex(ValueError, "24 nonempty"):
            self.plan()


if __name__ == "__main__":
    unittest.main()
