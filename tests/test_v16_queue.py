"""V16 deployment contracts, using isolated files and no GPU/server."""
import argparse
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = load("test_v16_queue_module", ROOT / "tools/v16/queue_jobs.py")
runner = load("test_v16_runner_module", ROOT / "tools/v16/run_experiment.py")


class V16QueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name) / "V16 project with spaces"
        self.scheduler = Path(temporary.name) / "existing V13 pool"
        self.data = Path(temporary.name) / "data"
        self.release = "context_20261112_r1"
        for relative in ("train_stream_v1.py", "tools/v16/train.py",
                         "tools/v16/profile_inference.py", "tools/v16/runtime_preflight.py",
                         "tools/v14/train_overlap.py", "model/context_revision.py",
                         "dataset/overlap_windows.py", "utils/stream_common.py"):
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n", encoding="utf-8")
        for relative in ("tools/v14/queue.py", "tools/v14/run_experiment.py",
                         "tools/v16/queue_jobs.py", "tools/v16/run_experiment.py",
                         "configs/evisseg_stream_tif_comm_peak.yaml",
                         "configs/v14/overlap_conditional.yaml",
                         "configs/v16/full.yaml", "configs/v16/no_revision.yaml",
                         "configs/v16/no_context.yaml"):
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

    def enqueue(self, plan):
        with redirect_stdout(io.StringIO()):
            return queue.enqueue(plan)

    def pack(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return queue.pack(self.project, self.release, **kwargs)

    def args(self, variant="full", seed=37, release=None):
        return argparse.Namespace(project_root=self.project, data_root=self.data,
                                  release=release or self.release, variant=variant,
                                  seed=seed, device="cpu")

    def fake_execute(self, command, source):
        out = Path(command[command.index("--out") + 1])
        if command[3].endswith("runtime_preflight.py"):
            runner.write_json(out, {"passed": True, "device": "fixture"})
            return
        out.mkdir(parents=True, exist_ok=True)
        raw = runner.load_yaml(Path(command[command.index("--config") + 1]))
        seed, variant = raw["TRAIN"]["seed"], raw["CONTEXT"]["revision_variant"]
        checkpoint = out / ("best_val_iou_seed%d.pt" % seed)
        if command[3].endswith("profile_inference.py"):
            runner.write_json(out / "profile.json", {
                "checkpoint_sha256": runner.sha256(checkpoint), "readout_policy": variant,
                "split": "test", "total_wall_seconds": 2., "energy": {"board_joules": None}})
        elif command[command.index("--mode") + 1] == "train":
            runner.write_json(out / "summary.json", {
                "epochs_completed": 50, "complete": True, "probe_only": False})
            runner.write_json(out / "train_stats.json", {"q99": 1})
            checkpoint.write_bytes(b"trained weights fixture")
        else:
            split = command[command.index("--split") + 1]
            metrics = {"iou": .8, "acc": .9, "pd": .85, "fa": .00001}
            runner.write_json(out / ("eval_%s.json" % split), {
                "metrics": metrics, "timing": {"execution": "step"}, "readout_policy": variant,
                "checkpoint_sha256": runner.sha256(checkpoint),
                "readouts": {name: metrics for name in ("first", "second", "main")}})

    def complete_job(self, job):
        with mock.patch.object(runner, "ROOT", self.project), \
             mock.patch.object(runner, "execute", side_effect=self.fake_execute), \
             redirect_stdout(io.StringIO()):
            result = runner.run(self.args(job["variant"], job["seed"]))
        log = self.scheduler / "run/queue/gpu1/logs" / (job["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("complete fixture\n", encoding="utf-8")
        return result

    def test_default_six_jobs_order_and_read_only_plan(self):
        plan = self.plan()
        self.assertEqual([(j["variant"], j["seed"]) for j in plan["jobs"]],
                         [(v, s) for v in queue.VARIANTS for s in (37, 38)])
        self.assertEqual(len({j["out"] for j in plan["jobs"]}), 6)
        self.assertEqual(plan["physical_gpu"], 1)
        self.assertFalse((self.project / "run").exists())
        self.assertFalse((self.scheduler / "run").exists())

    def test_cli_dry_run_checks_runtime_without_creating_jobs(self):
        with mock.patch.object(queue.sys, "platform", "linux"), \
             mock.patch.object(queue, "dependency_check", return_value="torch=fixture") as dependencies, \
             mock.patch.object(queue, "live_pool", return_value=1234) as pool, \
             redirect_stdout(io.StringIO()) as output:
            queue.main(["enqueue", "--project-root", str(self.project), "--scheduler-root", str(self.scheduler),
                        "--data-root", str(self.data), "--release", self.release,
                        "--config-dir", "configs/v16", "--dry-run"])
        dependencies.assert_called_once_with(self.project.resolve())
        pool.assert_called_once_with(self.scheduler.resolve())
        self.assertIn("DRY_RUN_OK: 6 training job(s)", output.getvalue())
        self.assertFalse((self.project / "run").exists())
        self.assertFalse((self.scheduler / "run").exists())

    def test_configs_change_only_module_variant_and_output_statistics(self):
        reference = None
        for variant, raw in runner.validate_configs(self.project).items():
            self.assertEqual(raw["CONTEXT"]["revision_variant"], variant)
            normalized = copy.deepcopy(raw)
            normalized["CONTEXT"].pop("revision_variant")
            normalized["DATA"].pop("stats_path")
            normalized["TRAIN"].pop("save_root")
            if reference is not None:
                self.assertEqual(normalized, reference)
            reference = normalized
        raw = runner.runtime_config("no_revision", 38, self.release, self.data, self.project, self.project)
        self.assertEqual(raw["TRAIN"]["seed"], 38)
        self.assertEqual(raw["TRAIN"]["epochs"], 50)
        self.assertEqual(raw["TRAIN"]["comm_budget_messages"], 157700.0)
        self.assertEqual(raw["TRAIN"]["streams"], 4)
        self.assertEqual(raw["TRAIN"]["tbptt_k"], 16)
        self.assertEqual(raw["TRAIN"]["eval_execution"], "step")
        self.assertEqual(raw["NET"]["state_layout"], "single_rolling")
        self.assertEqual((raw["DATA"]["window_ms"], raw["DATA"]["stride_ms"]), (100, 50))
        self.assertEqual(Path(raw["DATA"]["stats_path"]).parent, Path(raw["TRAIN"]["save_root"]))
        path = self.project / "configs/v16/full.yaml"
        path.write_text(path.read_text(encoding="utf-8").replace("157700.0", "0.0"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changes V14 TRAIN.comm_budget_messages"):
            self.plan()

    def test_source_freeze_covers_dependencies_and_excludes_credentials(self):
        secret = self.project / ".vscode/sftp.json"
        secret.parent.mkdir()
        secret.write_text("private", encoding="utf-8")
        plan = self.plan()
        for relative in ("train_stream_v1.py", "model/context_revision.py", "dataset/overlap_windows.py",
                         "utils/stream_common.py", "tools/v14/train_overlap.py", "tools/v14/queue.py",
                         "tools/v14/run_experiment.py", "tools/v16/train.py",
                         "tools/v16/runtime_preflight.py", "tools/v16/profile_inference.py"):
            self.assertIn(relative, plan["files"])
        manifest = self.enqueue(plan)
        self.assertEqual(queue.verify_source(manifest)["status"], "queued")
        self.assertFalse((Path(plan["source"]) / ".vscode").exists())
        scripts = sorted((self.scheduler / "run/queue/pool/pending").glob("*@gpu1.sh"))
        self.assertEqual(len(scripts), 6)
        for path in scripts:
            text = path.read_text(encoding="utf-8")
            for item in ("CUDA_VISIBLE_DEVICES=1", '${GPU:-}', "verify-source", "run_experiment.py"):
                self.assertIn(item, text)
            for item in ("nohup", "kill ", "gpu_pool.sh", "gpu_queue.sh", "--overwrite", "--resume"):
                self.assertNotIn(item, text)

    def test_generated_six_scripts_parse_in_bash(self):
        bash = shutil.which("bash")
        if os.name == "nt":
            candidate = Path("C:/Program Files/Git/bin/bash.exe")
            if candidate.is_file():
                bash = str(candidate)
        if not bash:
            self.skipTest("bash unavailable for Linux shell syntax validation")
        plan = self.plan()
        # Exercise quoting of valid Linux paths as well as scheduler routing.
        plan.update(source="/tmp/V16 user's project/source", project_root="/tmp/V16 user's project",
                    data_root="/tmp/data with spaces", python="/opt/evuav/bin/python")
        scripts = [queue.job_script(plan, job, Path("/tmp/V16 user's project/release.json"))
                   for job in plan["jobs"]]
        commands = (ROOT / "docs/V16_SERVER_COMMANDS.md").read_text(encoding="utf-8")
        command_blocks = re.findall(r"```bash\n(.*?)\n```", commands, flags=re.S)
        self.assertEqual(len(scripts), 6)
        self.assertGreaterEqual(len(command_blocks), 5)  # mkdir/enqueue/status/pack/partial-pack
        scripts.extend(command_blocks)
        for script in scripts:
            parsed = subprocess.run([bash, "-n"], input=script, universal_newlines=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(parsed.returncode, 0, parsed.stderr)

    def test_dependency_check_hides_gpu_and_does_not_write_bytecode(self):
        value = subprocess.CompletedProcess([], 0, "torch=fixture\n", "")
        with mock.patch.object(queue.subprocess, "run", return_value=value) as call:
            self.assertEqual(queue.dependency_check(self.project), "torch=fixture")
        self.assertEqual(call.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(call.call_args.kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertIn("-B", call.call_args.args[0])

    def test_duplicate_release_stays_blocked_after_dispatcher_takes_scripts(self):
        manifest = self.enqueue(self.plan())
        for path in (self.scheduler / "run/queue/pool/pending").glob("*.sh"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "release already exists"):
            self.plan()
        self.assertTrue(manifest.is_file())

    def test_partial_publication_retains_evidence_and_cannot_be_reentered(self):
        publish = queue.publish_job
        calls = []
        def second_fails(path, script):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("fixture publish failure")
            return publish(path, script)
        plan = self.plan()
        with mock.patch.object(queue, "publish_job", side_effect=second_fails):
            with self.assertRaisesRegex(OSError, "publish failure"):
                self.enqueue(plan)
        manifest = queue.release_plan(self.project, self.release)[1]
        self.assertEqual(manifest["status"], "enqueue_failed")
        self.assertEqual(len(manifest["queued"]), 1)
        self.assertEqual(len(list((self.scheduler / "run/queue/pool/pending").glob("*@gpu1.sh"))), 1)
        with self.assertRaisesRegex(ValueError, "release already exists"):
            self.plan()

    def test_status_recognizes_actual_pool_given_suffix(self):
        plan = self.plan(variants=("full",), seeds=(37,))
        self.enqueue(plan)
        name = plan["jobs"][0]["name"]
        pending = self.scheduler / "run/queue/pool/pending" / (name + "@gpu1.sh")
        given = self.scheduler / "run/queue/pool/given" / (name + "__gpu1.sh")
        given.parent.mkdir()
        pending.rename(given)
        with redirect_stdout(io.StringIO()) as output:
            queue.show_status(self.project, self.release, tail=0)
        self.assertIn("pool/given; stage=not_started", output.getvalue())

    def test_scheduler_markers_outputs_stop_and_duplicate_options_rejected(self):
        for part in ("pool/given", "gpu1/done", "gpu1/failed"):
            directory = self.scheduler / "run/queue" / part
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / ("1100_v16_context_%s_full_s37.sh" % self.release)
            path.touch()
            with self.assertRaisesRegex(ValueError, "release jobs already exist"):
                self.plan()
            path.unlink()
        with self.assertRaises(ValueError):
            self.plan(seeds=(37, 37))
        with self.assertRaises(ValueError):
            self.plan(variants=("full", "full"))
        stop = self.scheduler / "run/queue/gpu1/STOP"
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.touch()
        with self.assertRaisesRegex(ValueError, "STOP request"):
            self.plan()
        stop.unlink()
        out = self.project / "log/v16_context" / self.release / "full_s37"
        out.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "output already exists"):
            self.plan()

    def test_source_mutation_blocks_publication_and_snapshot_mutation_is_detected(self):
        plan = self.plan()
        changed = self.project / "model/context_revision.py"
        changed.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source changed after preflight"):
            self.enqueue(plan)
        self.assertFalse(list((self.scheduler / "run/queue/pool/pending").glob("*.sh")))

    def test_extra_or_changed_snapshot_source_is_detected(self):
        plan = self.plan()
        manifest = self.enqueue(plan)
        extra = Path(plan["source"]) / "tools/v16/injected.py"
        extra.write_text("# extra", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            queue.verify_source(manifest)
        extra.unlink()
        (Path(plan["source"]) / "model/context_revision.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            queue.verify_source(manifest)

    def test_runner_preflights_trains_evaluates_profiles_and_refuses_rerun(self):
        with mock.patch.object(runner, "ROOT", self.project), \
             mock.patch.object(runner, "execute", side_effect=self.fake_execute) as call, \
             redirect_stdout(io.StringIO()):
            result = runner.run(self.args())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(call.call_count, 5)
            commands = [c.args[0] for c in call.call_args_list]
            self.assertTrue(commands[0][3].endswith("runtime_preflight.py"))
            self.assertEqual(commands[1][-2:], ["--mode", "train"])
            self.assertEqual(commands[2][-2:], ["--split", "val"])
            self.assertEqual(commands[3][-2:], ["--split", "test"])
            self.assertTrue(commands[4][3].endswith("profile_inference.py"))
            self.assertEqual(len(result["artifact_sha256"]), 8)
            self.assertTrue(result["trained_from_scratch"])
            self.assertFalse(result["resume"])
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                runner.run(self.args())

    def test_preflight_failure_prevents_training_and_records_failure(self):
        def fail(command, source):
            runner.write_json(Path(command[command.index("--out") + 1]), {"passed": False})
        with mock.patch.object(runner, "ROOT", self.project), \
             mock.patch.object(runner, "execute", side_effect=fail) as call:
            with self.assertRaisesRegex(ValueError, "preflight"):
                runner.run(self.args())
            self.assertEqual(call.call_count, 1)
        meta = self.project / "run/v16_context" / self.release / "jobs/full_s37/manifest.json"
        self.assertEqual(runner.read_json(meta)["status"], "failed")
        self.assertFalse((self.project / "log/v16_context" / self.release).exists())

    def test_incomplete_probe_or_unmarked_training_cannot_continue(self):
        for index, fields in enumerate(({"complete": False, "probe_only": False},
                                       {"complete": True, "probe_only": True},
                                       {"complete": True}, {})):
            def execute(command, source):
                self.fake_execute(command, source)
                if "--mode" in command and command[-1] == "train":
                    out = Path(command[command.index("--out") + 1])
                    runner.write_json(out / "summary.json", dict(epochs_completed=50, **fields))
            with self.subTest(fields=fields), mock.patch.object(runner, "ROOT", self.project), \
                 mock.patch.object(runner, "execute", side_effect=execute) as call:
                with self.assertRaisesRegex(ValueError, "complete, non-probe"):
                    runner.run(self.args(release=self.release + "_bad%d" % index))
                self.assertEqual(call.call_count, 2)

    def test_evaluation_and_profile_identity_and_step_contract(self):
        path = self.project / "eval_test.json"
        runner.write_json(path, {"timing": {"execution": "layer"}})
        with self.assertRaisesRegex(ValueError, "must use step"):
            runner.validate_evaluation(path, "full")
        runner.write_json(path, {"timing": {"execution": "step"}, "readout_policy": "no_context"})
        with self.assertRaisesRegex(ValueError, "main readout"):
            runner.validate_evaluation(path, "full")
        runner.write_json(path, {"timing": {"execution": "step"}, "readout_policy": "full", "readouts": {}})
        with self.assertRaisesRegex(ValueError, "missing readout"):
            runner.validate_evaluation(path, "full")
        runner.write_json(path, {"checkpoint_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "checkpoint identity"):
            runner.validate_profile(path, "full", "selected")

    def test_incomplete_pack_rejects_and_partial_is_explicit(self):
        self.enqueue(self.plan())
        with self.assertRaisesRegex(ValueError, "not complete"):
            self.pack()
        self.assertFalse((self.project / "Res").exists())
        archive = self.pack(allow_partial=True)
        self.assertIn("_partial.tgz", archive.name)
        with tarfile.open(str(archive)) as handle:
            payload = json.load(handle.extractfile("package_manifest.json"))
        self.assertFalse(payload["complete"])
        self.assertEqual(len(payload["missing_results"]), 6)

    def test_complete_pack_has_all_six_results_weights_logs_and_source(self):
        plan = self.plan()
        self.enqueue(plan)
        for job in plan["jobs"]:
            self.complete_job(job)
        archive = self.pack()
        with tarfile.open(str(archive)) as handle:
            names = handle.getnames()
            payload = json.load(handle.extractfile("package_manifest.json"))
        self.assertTrue(payload["complete"])
        self.assertEqual(sum(name.endswith("comparison_result.json") for name in names), 6)
        self.assertEqual(sum(name.endswith(".pt") for name in names), 6)
        self.assertEqual(sum(name.startswith("scheduler_logs/") for name in names), 6)
        self.assertTrue(any("/source/tools/v16/train.py" in name for name in names))
        self.assertFalse(any(".vscode" in name or name.endswith(".npz") for name in names))
        out = Path(plan["jobs"][0]["out"])
        (out / "profile.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact is changed"):
            self.pack()

    def test_completed_result_missing_summary_requires_explicit_partial(self):
        plan = self.plan(variants=("full",), seeds=(37,))
        self.enqueue(plan)
        self.complete_job(plan["jobs"][0])
        (Path(plan["jobs"][0]["out"]) / "summary.json").unlink()
        with self.assertRaisesRegex(ValueError, "summary.json"):
            self.pack()
        self.assertIn("_partial", self.pack(allow_partial=True).name)

    def test_early_stop_requires_configured_policy_and_complete_training(self):
        raw = {"TRAIN": {"epochs": 40, "min_epochs": 15, "early_stop_patience": 8}}
        summary = {"epochs_completed": 23, "early_stopped": True,
                   "complete": True, "probe_only": False}
        self.assertEqual(runner.validate_training_summary(summary, raw), 23)
        for changed in ({"early_stopped": False}, {"epochs_completed": 14},
                        {"probe_only": True}, {"complete": False}, {"epochs_completed": 41}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                runner.validate_training_summary(dict(summary, **changed), raw)
        with self.assertRaises(ValueError):
            runner.validate_training_summary(summary, {"TRAIN": {"epochs": 40}})

    def test_initial_checkpoint_seed_and_statistics_are_checked(self):
        import torch
        init = self.project / "initial"
        init.mkdir()
        path = init / "seed37.pt"
        torch.save({"config": {"seed": 38}, "model": {}, "stats": {"q99": [1.]}}, str(path))
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            self.plan(seeds=(37,), init_root=init)
        torch.save({"config": {"seed": 37}, "model": {}, "stats": {}}, str(path))
        with self.assertRaisesRegex(ValueError, "q99"):
            self.plan(seeds=(37,), init_root=init)
        torch.save({"config": {"seed": 37}, "model": {}, "stats": {"q99": [1.]}}, str(path))
        plan = self.plan(seeds=(37,), init_root=init)
        frozen = Path(plan["initializations"]["37"]["path"])
        self.assertFalse(frozen.exists())
        manifest = self.enqueue(plan)
        self.assertEqual(queue.digest(frozen), queue.digest(path))
        for job in plan["jobs"]:
            self.assertIn("--init-sha256", queue.job_script(plan, job, manifest))
        frozen.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "initial checkpoint"):
            queue.verify_source(manifest)

    def test_nested_r2_configs_are_frozen(self):
        path = self.project / "configs/v16/r2/full.yaml"
        path.parent.mkdir()
        path.write_text("# new architecture\n", encoding="utf-8")
        plan = self.plan()
        self.assertIn("configs/v16/r2/full.yaml", plan["files"])
        manifest = self.enqueue(plan)
        changed = Path(plan["source"]) / "configs/v16/r2/full.yaml"
        changed.write_text("# mutated\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            queue.verify_source(manifest)

    def evidence_configs(self):
        import yaml
        for variant in queue.VARIANTS:
            raw = runner.load_yaml(self.project / ("configs/v16/%s.yaml" % variant))
            raw["CONTEXT"] = {"revision_architecture": "evidence", "revision_variant": variant,
                              "evidence_hidden": 8, "evidence_kappa": 1.0}
            raw["TRAIN"].update(epochs=40, min_epochs=15, early_stop_patience=8)
            path = self.project / "configs/v16/r2" / (variant + ".yaml")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    def test_r2_config_family_uses_isolated_sources_and_same_seed_protocol(self):
        self.evidence_configs()
        plan = self.plan(config_dir="configs/v16/r2")
        self.assertTrue(plan["jobs"][0]["name"].startswith("1280_"))
        self.assertFalse(plan["initializations"])
        self.assertEqual(plan["config_dir"], "configs/v16/r2")
        raw = runner.runtime_config("no_context", 38, self.release, self.data, self.project,
                                    self.project, "configs/v16/r2")
        self.assertEqual(raw["TRAIN"]["seed"], 38)
        self.assertEqual(raw["TRAIN"]["epochs"], 40)
        self.assertEqual(raw["CONTEXT"]["revision_architecture"], "evidence")
        path = self.project / "configs/v16/r2/no_revision.yaml"
        path.write_text(path.read_text(encoding="utf-8").replace("evidence_hidden: 8", "evidence_hidden: 16"),
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ablations may differ only"):
            self.plan(config_dir="configs/v16/r2")

    def test_cli_default_points_to_r2_and_same_existing_pool(self):
        self.evidence_configs()
        with mock.patch.object(queue.sys, "platform", "linux"), \
             mock.patch.object(queue, "dependency_check", return_value="torch=fixture"), \
             mock.patch.object(queue, "live_pool", return_value=1234), \
             redirect_stdout(io.StringIO()) as output:
            queue.main(["enqueue", "--project-root", str(self.project), "--scheduler-root", str(self.scheduler),
                        "--data-root", str(self.data), "--dry-run"])
        self.assertIn("context_20261113_r2", output.getvalue())
        self.assertIn("configs/v16/r2", output.getvalue())
        self.assertFalse((self.project / "run").exists())


if __name__ == "__main__":
    unittest.main()
