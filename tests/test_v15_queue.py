"""V15 release lifecycle tests, without contacting or mutating a server."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.v15 import package_results as packaging
from tools.v15 import queue_jobs as queue
from tools.v15 import run_experiment as runner


class ReleaseFixture(unittest.TestCase):
    def setUp(self):
        import yaml
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.project = self.base / "EV-UAV-v15"
        self.scheduler = self.base / "EV-UAV-v13"
        self.data = self.base / "dataset"
        self.release = "feature_test_r1"
        required = runner.RUNTIME_FILES + ("model/__init__.py", "dataset/__init__.py", "utils/__init__.py")
        for relative in required:
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# source fixture\n", encoding="utf-8")
        raw = runner.load_yaml(ROOT / "configs/evisseg_stream_tif_comm_peak.yaml")
        base_config = self.project / "configs/evisseg_stream_tif_comm_peak.yaml"
        base_config.parent.mkdir(parents=True)
        base_config.write_text(yaml.safe_dump(raw), encoding="utf-8")
        for mode in runner.MODES:
            cfg = copy.deepcopy(raw)
            cfg["REVISION"] = {"revision_source": mode}
            path = self.project / ("configs/v15/%s.yaml" % mode)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        for name in ("gpu_pool.sh", "gpu_queue.sh"):
            path = self.scheduler / "tools/final" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/bin/bash\n", encoding="utf-8")
        for split, count in runner.SPLIT_COUNTS.items():
            directory = self.data / split
            directory.mkdir(parents=True)
            for index in range(count):
                (directory / ("%s_%03d.npz" % (split, index))).write_bytes(b"nonempty fixture")

    def plan(self, **kwargs):
        return queue.prepare_plan(self.project, self.scheduler, self.data, self.release,
                                  runtime_checks=False, **kwargs)

    def test_source_freeze_excludes_obsolete_sftp_files_and_requires_dependencies(self):
        stale = self.project / "model/retired_experiment.py"
        stale.write_text("raise RuntimeError('old source')\n", encoding="utf-8")
        paths = runner.source_files(self.project)
        self.assertNotIn("model/retired_experiment.py", paths)
        dependency = self.project / "utils/comm_budget.py"
        dependency.unlink()
        with self.assertRaisesRegex(ValueError, "missing source dependency"):
            runner.source_files(self.project)

    def evaluation(self, split, digest):
        return {"complete": True, "probe_only": False, "split": split,
                "checkpoint_sha256": digest, "timing": {"execution": "step"},
                "metrics": {"iou": 0.85, "acc": 0.91, "pd": 0.93, "fa": 0.00001},
                "diagnostics": {"unique_events": 12, "readout_events": 12}}

    def simulate_trainer(self, command, source):
        """Stub numerical work; exercise real orchestration, files and validation."""
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        cfg = runner.load_yaml(command[command.index("--config") + 1])
        seed = cfg["TRAIN"]["seed"]
        checkpoint = out / ("best_val_iou_seed%d.pt" % seed)
        if any(item.endswith("profile.py") for item in command):
            runner.write_json(out / "profile.json", {"checkpoint_sha256": runner.digest(checkpoint), "split": "test",
                              "variants": {name: {"events_per_second": 1000.0, "records": [{"record": "test_000"}]}
                                           for name in ("v15_final", "v5_same_backbone")}})
        elif command[command.index("--mode") + 1] == "train":
            checkpoint.write_bytes(b"selected weights")
            (out / "last.pt").write_bytes(b"last weights")
            (out / "metrics.jsonl").write_text('{"epoch": 50}\n', encoding="utf-8")
            runner.write_json(out / "calibration.json", {"done": True})
            runner.write_json(out / "train_stats.json", {"q99": [1.0] * 12})
            runner.write_json(out / "summary.json", {"complete": True, "probe_only": False,
                                                      "epochs_completed": cfg["TRAIN"]["epochs"]})
        else:
            split = command[command.index("--split") + 1]
            self.assertIn("--step", command)
            runner.write_json(out / ("eval_%s.json" % split), self.evaluation(split, runner.digest(checkpoint)))

    def run_job(self):
        manifest = queue.enqueue(self.plan())
        plan = runner.read_json(manifest)
        args = argparse.Namespace(project_root=self.project, data_root=self.data,
                                  release=self.release, variant="observation", seed=37, device="cpu")
        with mock.patch.object(runner, "ROOT", Path(plan["source"])), \
                mock.patch.object(runner, "execute", side_effect=self.simulate_trainer) as execute, \
                mock.patch.object(runner, "hardware_inventory", return_value={"device": "test"}):
            result = runner.run(args)
        return plan, result, execute.call_args_list


class QueueTests(ReleaseFixture):
    def test_default_is_one_observation_seed37_job_gpu1(self):
        plan = self.plan()
        self.assertEqual(len(plan["jobs"]), 1)
        self.assertEqual(plan["jobs"][0]["name"], "1200_v15_feature_feature_test_r1_observation_s37")
        self.assertEqual(plan["physical_gpu"], 1)
        script = queue.job_script(plan, plan["jobs"][0], self.project / "release.json")
        self.assertIn("CUDA_VISIBLE_DEVICES=1", script)
        self.assertIn("--device cuda:0", script)
        self.assertNotIn("gpu_pool.sh", script)
        self.assertNotIn("nohup", script)
        self.assertFalse((self.project / "run").exists())

    def test_explicit_controls_are_separate_training_jobs(self):
        plan = self.plan(modes=("observation", "deep", "causal"))
        self.assertEqual([job["variant"] for job in plan["jobs"]], ["observation", "deep", "causal"])
        self.assertEqual(len({job["out"] for job in plan["jobs"]}), 3)
        self.assertTrue(all(job["seed"] == 37 for job in plan["jobs"]))

    def test_enqueue_freezes_source_and_refuses_duplicates_and_tampering(self):
        manifest = queue.enqueue(self.plan())
        plan = queue.verify_source(manifest)
        queued = list((self.scheduler / "run/queue/pool/pending").glob("*.sh"))
        self.assertEqual(len(queued), 1)
        self.assertTrue(queued[0].name.endswith("@gpu1.sh"))
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.plan()
        frozen = Path(plan["source"]) / "model/observation_revision.py"
        frozen.write_text("# modified\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "modified"):
            queue.verify_source(manifest)

    def test_atomic_publish_never_overwrites_existing_job(self):
        target = self.base / "job.sh"
        queue.publish_job(target, "first\n")
        with self.assertRaises(FileExistsError):
            queue.publish_job(target, "second\n")
        self.assertEqual(target.read_text(), "first\n")
        self.assertFalse(list(self.base.glob("*.tmp")))

    def test_reject_stop_request_and_existing_output(self):
        stop = self.scheduler / "run/queue/gpu1/STOP"
        stop.parent.mkdir(parents=True)
        stop.write_text("stop")
        with self.assertRaisesRegex(ValueError, "STOP"):
            self.plan()
        stop.unlink()
        out = self.project / "log/v15" / self.release / "observation_s37"
        out.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "output already"):
            self.plan()

    def test_config_rejects_reset_or_removed_communication_budget(self):
        import yaml
        path = self.project / "configs/v15/observation.yaml"
        raw = runner.load_yaml(path)
        raw["NET"]["state_mode"] = "reset_each_window"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "state_mode"):
            self.plan()
        raw["NET"]["state_mode"] = "carry"
        raw["TRAIN"]["comm_budget_messages"] = 0.0
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "comm_budget_messages"):
            self.plan()

    def test_incomplete_dataset_rejected(self):
        next((self.data / "test").glob("*.npz")).unlink()
        with self.assertRaisesRegex(ValueError, "test needs 24"):
            self.plan()

    def test_queue_filename_dedup_even_without_release_directory(self):
        pending = self.scheduler / "run/queue/gpu1/running"
        pending.mkdir(parents=True)
        (pending / "1200_v15_feature_feature_test_r1_observation_s37.sh").write_text("running")
        with self.assertRaisesRegex(ValueError, "already present"):
            self.plan()


class RunnerAndPackageTests(ReleaseFixture):
    def test_train_then_val_test_profile_and_download_complete_bundle(self):
        plan, result, calls = self.run_job()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(calls), 4)
        out = Path(plan["jobs"][0]["out"])
        self.assertEqual(runner.read_json(out / "train_stats_provenance.json")["statistics_split"], "train")
        cache = out / "cache"
        cache.mkdir()
        (cache / "large.pt").write_bytes(b"excluded cache")
        (out / "predictions.npz").write_bytes(b"excluded event array")
        archive = packaging.pack(self.project, self.release)
        with tarfile.open(str(archive), "r:gz") as handle:
            names = handle.getnames()
            bundle = json.load(handle.extractfile("bundle_manifest.json"))
        self.assertTrue(bundle["complete"])
        self.assertFalse(bundle["partial"])
        self.assertTrue(any(name.endswith("eval_val.json") for name in names))
        self.assertTrue(any(name.endswith("eval_test.json") for name in names))
        self.assertTrue(any(name.endswith("profile.json") for name in names))
        self.assertFalse(any("large.pt" in name or "predictions.npz" in name for name in names))
        self.assertEqual(set(bundle["files_sha256"]), set(names) - {"bundle_manifest.json"})

    def test_partial_release_is_explicit_and_marked(self):
        queue.enqueue(self.plan())
        with self.assertRaisesRegex(ValueError, "not complete"):
            packaging.pack(self.project, self.release)
        archive = packaging.pack(self.project, self.release, allow_partial=True)
        self.assertIn("partial", archive.name)
        with tarfile.open(str(archive), "r:gz") as handle:
            payload = json.load(handle.extractfile("bundle_manifest.json"))
        self.assertFalse(payload["complete"])
        self.assertTrue(payload["partial"])

    def test_bad_eval_identity_prevents_completion(self):
        manifest = queue.enqueue(self.plan())
        plan = runner.read_json(manifest)
        args = argparse.Namespace(project_root=self.project, data_root=self.data,
                                  release=self.release, variant="observation", seed=37, device="cpu")
        def corrupted(command, source):
            self.simulate_trainer(command, source)
            if "--split" in command:
                out = Path(command[command.index("--out") + 1])
                split = command[command.index("--split") + 1]
                payload = self.evaluation(split, "wrong weights")
                runner.write_json(out / ("eval_%s.json" % split), payload)
        with mock.patch.object(runner, "ROOT", Path(plan["source"])), \
                mock.patch.object(runner, "execute", side_effect=corrupted), \
                mock.patch.object(runner, "hardware_inventory", return_value={}):
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                runner.run(args)
        metadata = manifest.parent / "jobs/observation_s37/manifest.json"
        self.assertEqual(runner.read_json(metadata)["status"], "failed")
        self.assertFalse((Path(plan["jobs"][0]["out"]) / "comparison_result.json").exists())

    def test_complete_bundle_refuses_changed_checkpoint(self):
        plan, result, calls = self.run_job()
        checkpoint = Path(plan["jobs"][0]["out"]) / "best_val_iou_seed37.pt"
        checkpoint.write_bytes(b"new unverified weights")
        with self.assertRaisesRegex(ValueError, "checkpoint is missing or changed"):
            packaging.pack(self.project, self.release)


if __name__ == "__main__":
    unittest.main()
