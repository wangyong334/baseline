"""Final release publication and reboot recovery preserve existing work."""
import json
from pathlib import Path
import tempfile
import tarfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.bst.queue_final import archive_stale, backbone_seed, checked_child, plan, pool_preflight, publish
from tools.bst import pack_final


class FinalQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/"project"
        self.root.mkdir()
        self.data = Path(self.temp.name)/"data"
        for split in ("train", "val", "test"):
            (self.data/split).mkdir(parents=True)
            (self.data/split/"one.npz").write_bytes(b"data")
        (self.root/"tools/bst").mkdir(parents=True)
        for name in ("run_final.sh", "export_priors.py", "train.py", "fit_policy.py", "evaluate.py",
                     "final_metrics.py", "audit_reconsideration.py"):
            (self.root/"tools/bst"/name).write_text("# source\n")
        (self.root/"configs").mkdir()
        self.cfg = self.root/"configs/bst.json"
        self.cfg.write_text(json.dumps({"train": {"seed": 37}, "recording_n_windows": 160}))
        self.backbones = {}
        for seed in (37, 38):
            self.backbones[seed] = Path(self.temp.name)/("seed%d.pt" % seed)
            self.backbones[seed].write_bytes(str(seed).encode())
        self.seed_reader = patch("tools.bst.queue_final.backbone_seed",
                                 side_effect=lambda path: int(Path(path).stem[4:]))
        self.seed_reader.start()
        self.addCleanup(self.seed_reader.stop)

    def tearDown(self):
        self.temp.cleanup()

    def make_plan(self, release="final_test"):
        return plan(self.root, release, self.data, self.backbones, self.cfg)

    def test_both_backbones_required_before_writes(self):
        self.backbones[38].unlink()
        with self.assertRaises(FileNotFoundError):
            self.make_plan()
        self.assertFalse((self.root/"run").exists())

    def test_audit_entrypoint_required_before_writes(self):
        (self.root/"tools/bst/audit_reconsideration.py").unlink()
        with self.assertRaises(FileNotFoundError):
            self.make_plan()
        self.assertFalse((self.root/"run").exists())

    def test_backbones_cannot_be_same_seed_bytes(self):
        self.backbones[38].write_bytes(self.backbones[37].read_bytes())
        with self.assertRaisesRegex(ValueError, "identical"):
            self.make_plan()

    def test_wrong_checkpoint_seed_fails_before_any_publication(self):
        with patch("tools.bst.queue_final.backbone_seed", side_effect=[37, 37]):
            with self.assertRaisesRegex(ValueError, "BIR seed mismatch"):
                self.make_plan()
        self.assertFalse((self.root/"run").exists())

    def test_checkpoint_seed_is_required_and_loaded_on_cpu(self):
        calls = []
        def load(path, **kwargs):
            calls.append(kwargs)
            return {"config": {"seed": 38}}
        with patch.dict(sys.modules, {"torch": SimpleNamespace(load=load)}):
            self.assertEqual(backbone_seed("trusted.pt"), 38)
        self.assertEqual(calls, [{"map_location": "cpu", "weights_only": False}])
        for invalid in ({}, {"seed": True}, {"seed": "38"}):
            with patch.dict(sys.modules, {"torch": SimpleNamespace(load=lambda *a, **k: {"config": invalid})}):
                with self.assertRaisesRegex(ValueError, "cannot verify BIR seed"):
                    backbone_seed("trusted.pt")

    def test_atomic_jobs_and_matching_configs(self):
        spec = self.make_plan()
        publish(spec)
        jobs = list((self.root/"run/queue/pool/pending").glob("*.sh"))
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(p.name.endswith("@gpu1.sh") for p in jobs))
        self.assertTrue(all("export BST_PYTHON=" in p.read_text() for p in jobs))
        self.assertEqual(spec["python_executable"], str(Path(sys.executable).resolve()))
        for job in spec["jobs"]:
            cfg = json.loads(Path(job["config"]).read_text())
            self.assertEqual(cfg["train"]["seed"], job["seed"])
        self.assertFalse(list((self.root/"run/queue/pool/pending").glob("*.tmp")))
        with self.assertRaises(FileExistsError):
            self.make_plan()

    def test_snapshot_does_not_follow_later_source_edits(self):
        spec = self.make_plan()
        publish(spec)
        source = self.root/"tools/bst/run_final.sh"
        source.write_text("changed")
        frozen = Path(spec["release_dir"])/"source/tools/bst/run_final.sh"
        self.assertEqual(frozen.read_text(), "# source\n")

    def test_reboot_archive_preserves_logs_and_other_gpu(self):
        q = self.root/"run/queue"
        for name in ["pool/pending/old.sh", "gpu1/running/old.sh", "gpu1/pending/next.sh",
                     "gpu1/STOP", "gpu1/runner.pid", "gpu1/logs/old.log", "gpu2/running/other.sh"]:
            path = q/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        archive, moved = archive_stale(self.root)
        self.assertEqual(len(moved), 5)
        self.assertEqual((archive/"gpu1/running/old.sh").read_text(), "gpu1/running/old.sh")
        self.assertTrue((q/"gpu1/logs/old.log").exists())
        self.assertTrue((q/"gpu2/running/other.sh").exists())
        self.assertFalse((q/"gpu1/running/old.sh").exists())

    def test_existing_dispatcher_prevents_duplicate_startup(self):
        records = [{"kind": "gpu_pool.sh", "gpus": ["1"], "ours": True}]
        with patch("tools.bst.queue_final.os", SimpleNamespace(name="posix")), \
             patch("tools.bst.queue_final.process_inventory", return_value=records), \
             patch.object(Path, "is_dir", return_value=True):
            self.assertEqual(pool_preflight(self.root, False), 3)

    def test_custom_worker_queue_cannot_silently_redirect_final_jobs(self):
        records = [{"kind": "gpu_queue.sh", "gpus": ["1"], "ours": True, "queue_name": "old_gpu1"}]
        with patch("tools.bst.queue_final.os", SimpleNamespace(name="posix")), \
             patch("tools.bst.queue_final.process_inventory", return_value=records), \
             patch.object(Path, "is_dir", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "custom QNAME"):
                pool_preflight(self.root, False)

    def test_paths_and_release_names_cannot_escape(self):
        with self.assertRaises(ValueError):
            checked_child(self.root, self.root/"../outside")
        with self.assertRaises(ValueError):
            self.make_plan("../../outside")
        with self.assertRaises(ValueError):
            self.make_plan("a; touch bad")

    def test_package_marks_incomplete_and_excludes_weights(self):
        spec = self.make_plan()
        publish(spec)
        output = Path(spec["jobs"][0]["out"])
        output.mkdir(parents=True)
        (output/"model.pt").write_bytes(b"weights")
        (output/"summary.json").write_text("{}")
        (output/"events.npz").write_bytes(b"events")
        with patch.object(pack_final, "ROOT", self.root), patch("sys.argv", ["pack", "--release", "final_test"]):
            pack_final.main()
        archive = self.root/"Res/bst_final_final_test.tgz"
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            self.assertFalse(any(n.endswith("model.pt") for n in names))
            self.assertTrue(any(n.endswith("events.npz") for n in names))
            status = json.load(tar.extractfile("PACK_STATUS.json"))
            self.assertFalse(status["runs"][0]["complete"])
            self.assertFalse(status["runs"][1]["exists"])
        original = archive.read_bytes()
        with patch.object(pack_final, "ROOT", self.root), patch("sys.argv", ["pack", "--release", "final_test"]):
            with self.assertRaises(FileExistsError):
                pack_final.main()
        self.assertEqual(original, archive.read_bytes())


if __name__ == "__main__":
    unittest.main()
