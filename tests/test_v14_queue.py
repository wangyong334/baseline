"""Release/queue safety tests; no CUDA or server connection is required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v14_queue", str(ROOT / "tools/v14/queue.py"))
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.project = base / "new project"
        self.scheduler = base / "existing pool"
        self.data = base / "data"
        for relative in ("train_stream_v1.py", "tools/stream_train_stats.py",
                         "tools/v14/queue.py", "model/lif.py", "dataset/stream.py",
                         "utils/config.py", "configs/evisseg_stream_tif_notransport.yaml",
                         "configs/v14/bir_carry.yaml", "configs/v14/bir_reset.yaml"):
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n", encoding="utf-8")
        (self.project / "tools/v14/run_experiment.py").write_text(
            "def validate_pair(source_root):\n    return {}\n", encoding="utf-8")
        for relative in ("tools/final/gpu_pool.sh", "tools/final/gpu_queue.sh"):
            path = self.scheduler / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# existing scheduler\n", encoding="utf-8")
        for split, count in queue.SPLIT_COUNTS.items():
            directory = self.data / split
            directory.mkdir(parents=True)
            for index in range(count):
                (directory / ("%s_%03d.npz" % (split, index))).write_bytes(b"fixture")

    def plan(self, **kwargs):
        return queue.prepare_plan(self.project, self.scheduler, self.data,
                                  "mem_ablation_20261112_r1", runtime_checks=False, **kwargs)

    def test_default_pair_is_two_seed37_jobs_and_read_only(self):
        (self.project / "model/__init__.py").write_bytes(b"")
        plan = self.plan()
        self.assertEqual([(j["mode"], j["seed"]) for j in plan["jobs"]],
                         [("carry", 37), ("reset", 37)])
        self.assertEqual(plan["physical_gpu"], 1)
        self.assertIn("model/__init__.py", plan["files"])
        self.assertFalse((self.project / "run").exists())
        self.assertFalse((self.scheduler / "run").exists())

    def test_snapshot_and_outputs_are_separate_from_existing_pool(self):
        secret = self.project / ".vscode/sftp.json"
        secret.parent.mkdir()
        secret.write_text("credential must not be copied", encoding="utf-8")
        plan = self.plan()
        manifest = queue.enqueue(plan)
        saved = queue.verify_source(manifest)
        self.assertEqual(saved["status"], "queued")
        self.assertFalse((Path(plan["source"]) / ".vscode").exists())
        scripts = list((self.scheduler / "run/queue/pool/pending").glob("*.sh"))
        self.assertEqual(len(scripts), 2)
        self.assertTrue(all(p.name.endswith("@gpu1.sh") for p in scripts))
        for path in scripts:
            content = path.read_text(encoding="utf-8")
            self.assertIn("CUDA_VISIBLE_DEVICES=1", content)
            self.assertIn("verify-source", content)
            self.assertIn("--project-root", content)
            self.assertIn(str(self.project.resolve()), content)
            self.assertNotIn("nohup", content)
            self.assertNotIn("gpu_pool.sh", content)
            self.assertNotIn("gpu_queue.sh", content)
        for job in plan["jobs"]:
            Path(job["out"]).relative_to(self.project)  # raises if output escapes V14

    def test_existing_release_cannot_be_requeued_after_job_is_taken(self):
        plan = self.plan()
        queue.enqueue(plan)
        for script in (self.scheduler / "run/queue/pool/pending").glob("*.sh"):
            script.unlink()  # simulate the dispatcher moving the jobs elsewhere
        with self.assertRaisesRegex(ValueError, "release already exists"):
            self.plan()

    def test_done_failed_and_given_jobs_prevent_duplicate_release(self):
        for part in ("gpu1/done", "gpu1/failed", "pool/given"):
            with self.subTest(part=part):
                directory = self.scheduler / "run/queue" / part
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "1000_v14_bir_mem_ablation_20261112_r1_carry_s37.sh"
                path.write_text("finished", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "release jobs already exist"):
                    self.plan()
                path.unlink()

    def test_existing_output_is_never_overwritten(self):
        out = self.project / "log/v14/mem_ablation_20261112_r1/reset_s37"
        out.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "output already exists"):
            self.plan()

    def test_stop_request_or_wrong_dataset_refuses_preflight(self):
        stop = self.scheduler / "run/queue/gpu1/STOP"
        stop.parent.mkdir(parents=True)
        stop.touch()
        with self.assertRaisesRegex(ValueError, "STOP request"):
            self.plan()
        stop.unlink()
        (self.data / "val/val_000.npz").unlink()
        with self.assertRaisesRegex(ValueError, "24 nonempty"):
            self.plan()

    def test_modified_snapshot_detected_before_training(self):
        plan = self.plan()
        manifest = queue.enqueue(plan)
        (Path(plan["source"]) / "model/lif.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            queue.verify_source(manifest)

    def test_upload_change_during_preflight_publishes_no_jobs(self):
        plan = self.plan()
        (self.project / "model/lif.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source changed after preflight"):
            queue.enqueue(plan)
        self.assertEqual(list((self.scheduler / "run/queue/pool/pending").glob("*.sh")), [])
        manifest = Path(plan["source"]).parent / "release.json"
        self.assertEqual(json.loads(manifest.read_text())["status"], "enqueue_failed")

    def test_atomic_publish_never_replaces_existing_job(self):
        path = self.project / "job.sh"
        queue.publish_job(path, "original\n")
        with self.assertRaises(FileExistsError):
            queue.publish_job(path, "replacement\n")
        self.assertEqual(path.read_text(), "original\n")

    def test_invalid_release_and_duplicate_seeds_refused(self):
        with self.assertRaises(ValueError):
            queue.check_release("../outside")
        with self.assertRaises(ValueError):
            self.plan(seeds=(37, 37))

    def test_live_pool_ignores_unreadable_unrelated_pids(self):
        unreadable = mock.MagicMock()
        unreadable.name = "1"
        unreadable.__truediv__.return_value.read_bytes.side_effect = PermissionError("foreign PID")
        dispatcher = mock.MagicMock()
        dispatcher.name = "42"
        cmdline = mock.Mock()
        cmdline.read_bytes.return_value = b"bash\0tools/final/gpu_pool.sh\0"
        cwd = mock.Mock()
        cwd.resolve.return_value = self.scheduler.resolve()
        environ = mock.Mock()
        environ.read_bytes.return_value = b"GPUS=1\0KEEP=1\0"
        dispatcher.__truediv__.side_effect = {"cmdline": cmdline, "cwd": cwd,
                                             "environ": environ}.__getitem__
        with mock.patch.object(queue.Path, "iterdir", return_value=iter([unreadable, dispatcher])):
            self.assertEqual(queue.live_pool(self.scheduler), 42)


if __name__ == "__main__":
    unittest.main()
