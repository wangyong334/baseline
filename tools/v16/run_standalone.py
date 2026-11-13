"""Run one frozen-source V16 full/seed37 experiment on physical GPU 2.

No dispatcher, queue, or GPU holder is used. Python 3.8 compatible. The runner
performs the configured adaptation, head training, validation, test and profile;
the separate pack command verifies the completed artifacts before packaging.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.v16 import queue_jobs as common


def gpu2_identity():
    """Resolve the requested physical board, without creating a CUDA context."""
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    output = subprocess.check_output(
        ["nvidia-smi", "-i", "2", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
        stderr=subprocess.STDOUT, universal_newlines=True, timeout=15, **flags)
    rows = list(csv.reader(io.StringIO(output.strip())))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("Cannot uniquely identify physical GPU 2")
    index, gpu_uuid, name = [value.strip() for value in rows[0]]
    if index != "2" or not gpu_uuid.startswith("GPU-"):
        raise ValueError("nvidia-smi did not identify physical GPU 2")
    return {"index": 2, "uuid": gpu_uuid, "name": name}


def prepare_plan(project_root, data_root, release, config_dir, init_checkpoint,
                 runtime_checks=True):
    project, data = Path(project_root).resolve(), Path(data_root).resolve()
    common.check_release(release)
    release_root = project / "run/v16_context" / release
    out = project / "log/v16_context" / release / "full_s37"
    if release_root.exists() or out.exists():
        raise ValueError("release or output already exists; use a new unique release")
    configs = common.validate_configs(project, config_dir)
    if "full" not in configs:
        raise ValueError("standalone requires a full configuration")
    initial = common.initial_checkpoint(init_checkpoint, 37)
    initial["path"] = str(release_root / "initialization/seed37.pt")
    plan = {"format_version": 1, "execution": "standalone", "release": release,
            "project_root": str(project), "data_root": str(data), "data": common.dataset_summary(data),
            "config_dir": config_dir, "variant": "full", "seed": 37,
            "physical_gpu": 2, "device": "cuda:0", "out": str(out),
            "source": str(release_root / "source"), "files": common.source_files(project),
            # A venv executable can be a symlink to system Python: resolving it
            # would silently discard the environment used for dependency checks.
            "initializations": {"37": initial}, "python": os.path.abspath(sys.executable),
            "python_version": sys.version, "log": str(release_root / "standalone.log")}
    if runtime_checks:
        if sys.platform != "linux":
            raise ValueError("physical-GPU-2 standalone execution is for the Linux server")
        plan["runtime"] = {"dependencies": common.dependency_check(project), "gpu": gpu2_identity()}
    return plan


def freeze(plan):
    source, project = Path(plan["source"]), Path(plan["project_root"])
    release_root = source.parent
    release_root.parent.mkdir(parents=True, exist_ok=True)
    release_root.mkdir()  # Exclusive reservation, including failed releases.
    manifest = release_root / "standalone.json"
    state = dict(plan, status="preparing", created_at_unix=time.time())
    common.write_json(manifest, state)
    try:
        source.mkdir()
        for relative, expected in plan["files"].items():
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(project / relative), str(target))
            if common.digest(target) != expected:
                raise ValueError("source changed while freezing: " + relative)
        for item in plan["initializations"].values():
            target = Path(item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["original"], str(target))
            if common.digest(target) != item["sha256"]:
                raise ValueError("initial checkpoint changed while freezing")
        common.verify_source(manifest)
        state["status"] = "prepared"
        common.write_json(manifest, state)
    except BaseException as error:
        state.update(status="failed", error=str(error) or type(error).__name__)
        common.write_json(manifest, state)
        raise
    return manifest


def runner_command(plan):
    initial = plan["initializations"]["37"]
    return [plan["python"], "-B", "-u", str(Path(plan["source"]) / "tools/v16/run_experiment.py"),
            "--project-root", plan["project_root"], "--data-root", plan["data_root"],
            "--release", plan["release"], "--config-dir", plan["config_dir"],
            "--variant", "full", "--seed", "37", "--device", "cuda:0",
            "--init-checkpoint", initial["path"], "--init-sha256", initial["sha256"]]


def child_environment(plan):
    # Selecting the verified board UUID also avoids PCI/ordinal ambiguity. CUDA
    # sees this single physical GPU 2 as logical cuda:0 in every child process.
    gpu = plan.get("runtime", {}).get("gpu", {})
    gpu_uuid = gpu.get("uuid")
    if gpu.get("index") != 2 or not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise ValueError("standalone execution requires a verified physical GPU 2 UUID")
    return dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu_uuid,
                OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")


def _stop_launched_group(process):
    """On interruption stop only our newly created runner process group."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    except ProcessLookupError:
        pass


def stream_runner(command, source, environment, log):
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    with Path(log).open("x", encoding="utf8", newline="\n") as output:
        output.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        output.flush()
        process = subprocess.Popen(command, cwd=str(source), env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True, bufsize=1,
                                   start_new_session=os.name != "nt", **flags)
        try:
            for line in process.stdout:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        except BaseException:
            _stop_launched_group(process)
            raise
        finally:
            process.stdout.close()


def completed_result(plan):
    """Validate a complete ordinary runner result, independent of a scheduler."""
    project, source, out = (Path(plan[key]).resolve() for key in ("project_root", "source", "out"))
    expected_out = project / "log/v16_context" / plan["release"] / "full_s37"
    expected_source = project / "run/v16_context" / plan["release"] / "source"
    if (out != expected_out or source != expected_source or plan.get("physical_gpu") != 2 or
            plan.get("execution") != "standalone" or plan.get("variant") != "full" or
            plan.get("seed") != 37 or plan.get("device") != "cuda:0"):
        raise ValueError("standalone manifest identity mismatch")
    runner = common.load_module("v16_standalone_result", source / "tools/v16/run_experiment.py")
    result = runner.read_json(out / "comparison_result.json")
    for key, expected in (("status", "complete"), ("release", plan["release"]), ("variant", "full"), ("seed", 37)):
        if result.get(key) != expected:
            raise ValueError("standalone result identity mismatch: " + key)
    if result.get("source_sha256") != plan["files"]:
        raise ValueError("result source identity does not match the frozen standalone release")
    if result.get("cuda_visible_devices") != child_environment(plan)["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("result GPU identity does not match physical GPU 2")
    recorded = common.validated_completed_artifacts(project, result)
    meta = source.parent / "jobs/full_s37"
    checkpoint = out / "best_val_iou_seed37.pt"
    required = [out / name for name in ("summary.json", "train_stats.json", "eval_val.json",
                                        "eval_test.json", "profile.json")]
    required += [checkpoint, meta / "config.yaml", meta / "runtime_preflight.json"]
    if any(path.relative_to(project).as_posix() not in recorded for path in required):
        raise ValueError("completed result lacks required artifact hashes")
    if common.digest(checkpoint) != result.get("checkpoint_sha256"):
        raise ValueError("standalone selected checkpoint identity mismatch")
    if runner.read_json(meta / "manifest.json").get("status") != "complete":
        raise ValueError("runner job metadata has not completed")
    runner.validate_training_summary(runner.read_json(out / "summary.json"), runner.load_yaml(meta / "config.yaml"))
    runner.validate_preflight(meta / "runtime_preflight.json")
    for split in ("val", "test"):
        evaluation = runner.validate_evaluation(out / ("eval_%s.json" % split), "full")
        if evaluation.get("checkpoint_sha256") != result["checkpoint_sha256"]:
            raise ValueError("standalone evaluation checkpoint identity mismatch")
    runner.validate_profile(out / "profile.json", "full", result["checkpoint_sha256"])
    return result


def run(plan):
    environment = child_environment(plan)
    manifest = freeze(plan)
    state = json.loads(manifest.read_text(encoding="utf8"))
    started = time.time()
    state.update(status="running", started_at_unix=started,
                 cuda_visible_devices=environment["CUDA_VISIBLE_DEVICES"])
    common.write_json(manifest, state)
    print("STANDALONE GPU 2 -> cuda:0; LOG " + plan["log"], flush=True)
    try:
        common.verify_source(manifest)
        stream_runner(runner_command(plan), plan["source"], environment, plan["log"])
        common.verify_source(manifest)
        result = completed_result(plan)
        state.update(status="complete", finished_at_unix=time.time(),
                     elapsed_seconds=time.time() - started, checkpoint_sha256=result["checkpoint_sha256"])
        common.write_json(manifest, state)
        print("STANDALONE COMPLETE " + plan["out"], flush=True)
        return result
    except BaseException as error:
        state.update(status="failed", finished_at_unix=time.time(),
                     error=str(error) or type(error).__name__)
        common.write_json(manifest, state)
        raise


def pack(project_root, release):
    project = Path(project_root).resolve()
    common.check_release(release)
    manifest = project / "run/v16_context" / release / "standalone.json"
    plan = common.verify_source(manifest)
    if Path(plan["project_root"]).resolve() != project or plan.get("release") != release:
        raise ValueError("standalone release/project identity mismatch")
    if plan.get("status") != "complete":
        raise ValueError("standalone experiment is not complete; inspect its log and job manifest")
    result = completed_result(plan)
    files = {}
    allowed = {".json", ".jsonl", ".yaml", ".log", ".txt", ".pt", ".csv", ".tsv"}
    for directory in (Path(plan["out"]), manifest.parent / "jobs"):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed:
                files[path.relative_to(project).as_posix()] = path
    for path in [manifest, Path(plan["log"])] + [Path(item["path"]) for item in plan["initializations"].values()]:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError("standalone provenance or log is missing: " + str(path))
        files[path.relative_to(project).as_posix()] = path
    for relative in plan["files"]:
        path = Path(plan["source"]) / relative
        files[path.relative_to(project).as_posix()] = path
    before = {}
    for name, path in files.items():
        if path.is_symlink():
            raise ValueError("unsafe standalone archive symlink: " + name)
        try:
            path.resolve().relative_to(project)
        except ValueError:
            raise ValueError("standalone archive path escapes project: " + name)
        before[name] = (path.stat().st_size, path.stat().st_mtime_ns)
    destination = project / "Res"
    destination.mkdir(exist_ok=True)
    target = destination / ("v16_%s_%s_%s.tgz" % (release, time.strftime("%Y%m%d_%H%M%S"), uuid.uuid4().hex[:6]))
    temporary = target.with_suffix(".tmp")
    try:
        with temporary.open("xb") as handle:
            with tarfile.open(fileobj=handle, mode="w:gz") as archive:
                for name, path in sorted(files.items()):
                    archive.add(str(path), arcname=name, recursive=False)
                payload = json.dumps({"execution": "standalone", "physical_gpu": 2,
                                      "release": release, "complete": True,
                                      "checkpoint_sha256": result["checkpoint_sha256"]}, indent=2).encode("utf8")
                info = tarfile.TarInfo("package_manifest.json")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        if any((path.stat().st_size, path.stat().st_mtime_ns) != before[name] for name, path in files.items()):
            raise ValueError("standalone artifacts changed while packaging")
        os.link(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()
    print("DOWNLOAD " + str(target))
    print("SHA256 " + common.digest(target))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--project-root", type=Path, default=ROOT)
    launch.add_argument("--data-root", type=Path, default=Path(common.DEFAULT_DATA))
    launch.add_argument("--release", required=True)
    launch.add_argument("--config-dir", required=True)
    launch.add_argument("--init-checkpoint", type=Path, required=True)
    launch.add_argument("--dry-run", action="store_true")
    package = commands.add_parser("pack")
    package.add_argument("--project-root", type=Path, default=ROOT)
    package.add_argument("--release", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            pack(args.project_root, args.release)
        else:
            plan = prepare_plan(args.project_root, args.data_root, args.release,
                                args.config_dir, args.init_checkpoint)
            if args.dry_run:
                print(json.dumps({key: value for key, value in plan.items() if key != "files"}, indent=2))
                print("DRY_RUN_OK: one standalone full/seed37 job on physical GPU 2")
            else:
                run(plan)
    except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, "ERROR: %s\n" % error)


if __name__ == "__main__":
    main()
