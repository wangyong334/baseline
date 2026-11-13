"""Queue, inspect and package V16 context runs through the existing GPU-1 pool.

Default: full, no_revision, no_context, each independently trained at seeds 37/38. No server
process is started, killed or restarted. Python 3.8 compatible.
"""
import argparse
from collections import deque
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE = "context_20261113_r2"
DEFAULT_DATA = "/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset"
VARIANTS = ("full", "no_revision", "no_context")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_module("v16_context_queue_common", ROOT / "tools/v14/queue.py")
digest = common.digest
check_release = common.check_release
dataset_summary = common.dataset_summary
live_pool = common.live_pool
queue_scripts = common.queue_scripts
write_json = common.write_json
publish_job = common.publish_job
SPLIT_COUNTS = common.SPLIT_COUNTS


def source_files(root):
    root = Path(root).resolve()
    required = {root / p for p in (
        "train_stream_v1.py", "tools/v16/train.py", "tools/v16/run_experiment.py",
        "tools/v16/profile_inference.py",
        "tools/v14/run_experiment.py", "tools/v14/queue.py", "tools/v16/queue_jobs.py",
        "configs/evisseg_stream_tif_comm_peak.yaml", "configs/v14/overlap_conditional.yaml", "configs/v16/full.yaml",
        "configs/v16/no_revision.yaml", "configs/v16/no_context.yaml",
        "tools/v16/runtime_preflight.py")}
    paths = set(required)
    for directory in ("model", "dataset", "utils", "tools/v14", "tools/v16"):
        found = list((root / directory).rglob("*.py"))
        if not found:
            raise ValueError("required source directory is missing or empty: " + directory)
        paths.update(found)
    paths.update((root / "configs/v14").glob("*.yaml"))
    paths.update((root / "configs/v16").rglob("*.yaml"))
    result = {}
    for path in sorted(paths):
        if not path.is_file() or (path in required and not path.stat().st_size):
            raise ValueError("required source file missing or empty: %s" % path)
        try:
            path.resolve().relative_to(root)
        except ValueError:
            raise ValueError("source symlink escapes project: %s" % path)
        result[path.relative_to(root).as_posix()] = digest(path)
    return result


def verify_source(manifest):
    plan = common.verify_source(manifest)
    if source_files(plan["source"]) != plan["files"]:
        raise ValueError("source snapshot changed: source inventory differs from release")
    for item in plan.get("initializations", {}).values():
        path = Path(item["path"])
        if not path.is_file() or digest(path) != item["sha256"]:
            raise ValueError("frozen initial checkpoint is missing or changed: " + str(path))
    return plan


def validate_configs(root, config_dir="configs/v16"):
    module = load_module("v16_context_preflight", Path(root) / "tools/v16/run_experiment.py")
    return module.validate_configs(source_root=Path(root), config_dir=config_dir)


def initial_checkpoint(path, seed):
    """Load metadata on CPU; never substitute a different seed's backbone."""
    import torch
    path = Path(path).resolve()
    if not path.is_file() or not path.stat().st_size:
        raise ValueError("initial checkpoint is missing or empty: " + str(path))
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    config = checkpoint.get("config", checkpoint.get("cfg", {}))
    recorded_seed = config.get("seed", config.get("TRAIN", {}).get("seed"))
    if recorded_seed != seed:
        raise ValueError("initial checkpoint seed mismatch: requested %d, found %r" % (seed, recorded_seed))
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError("initial checkpoint lacks model weights")
    stats = checkpoint.get("stats", {})
    if "q99" not in stats:
        raise ValueError("initial checkpoint lacks training q99 statistics")
    return {"original": str(path), "sha256": digest(path), "seed": seed}


def dependency_check(root):
    command = [sys.executable, "-B", "-c",
               "import tools.v16.train, tools.v16.profile_inference, tools.v16.runtime_preflight, "
               "tools.v16.run_experiment, yaml; import torch; print('torch=' + torch.__version__)"]
    # Import checks must not acquire the busy GPU or create __pycache__ files.
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(command, cwd=str(root), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True, env=environment)
    if result.returncode:
        raise ValueError("V16 dependency check failed:\n" + result.stderr.strip())
    return result.stdout.strip()


def prepare_plan(project_root, scheduler_root, data_root, release, seeds=(37, 38),
                 variants=VARIANTS, runtime_checks=True, config_dir="configs/v16", init_root=None):
    project = Path(project_root).resolve()
    scheduler = Path(scheduler_root).resolve()
    data_root = Path(data_root).resolve()
    check_release(release)
    seeds = tuple(int(s) for s in seeds)
    variants = tuple(variants)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError("variants must be unique choices from " + ", ".join(VARIANTS))
    if scheduler == project:
        raise ValueError("scheduler root must be the existing V13 project, separate from V16")
    for relative in ("tools/final/gpu_pool.sh", "tools/final/gpu_queue.sh"):
        if not (scheduler / relative).is_file():
            raise ValueError("existing scheduler file missing: %s" % (scheduler / relative))
    for relative in ("run/queue/pool/STOP", "run/queue/gpu1/STOP"):
        if (scheduler / relative).exists():
            raise ValueError("existing scheduler has a STOP request: " + relative)
    release_root = project / "run/v16_context" / release
    if release_root.exists():
        raise ValueError("release already exists; refusing duplicate or partial enqueue: %s" % release_root)
    hashes = source_files(project)
    configs = validate_configs(project, config_dir)
    if any(v not in configs for v in variants):
        raise ValueError("requested variant configuration unavailable")
    if init_root is None and any(configs[v]["TRAIN"].get("require_initialization", False)
                                 for v in variants):
        raise ValueError("selected configuration requires --init-root with matched seed checkpoints")
    data = dataset_summary(data_root)
    scripts = queue_scripts(scheduler)
    prefix = "v16_context_%s_" % release
    if any(prefix in path.name for path in scripts):
        raise ValueError("release jobs already exist in the scheduler")
    paired = configs[variants[0]]["CONTEXT"].get("revision_architecture") in ("evidence", "anchored", "covariance")
    number = max([1270 if paired else 1090] + [int(m.group(1)) for path in scripts
                          for m in [re.match(r"^(\d+)_", path.name)] if m])
    initializations = {}
    if init_root is not None:
        for seed in seeds:
            item = initial_checkpoint(Path(init_root) / ("seed%d.pt" % seed), seed)
            item["path"] = str(release_root / "initialization" / ("seed%d.pt" % seed))
            initializations[str(seed)] = item
    jobs = []
    for variant in variants:
        for seed in seeds:
            number += 10
            out = project / "log/v16_context" / release / ("%s_s%d" % (variant, seed))
            if out.exists():
                raise ValueError("experiment output already exists; refusing overwrite: %s" % out)
            jobs.append({"name": "%d_%s%s_s%d" % (number, prefix, variant, seed),
                         "variant": variant, "seed": seed, "out": str(out)})
    runtime = {}
    if runtime_checks:
        if sys.platform != "linux":
            raise ValueError("enqueue is for the Linux server after SFTP upload")
        runtime = {"dependencies": dependency_check(project), "dispatcher_pid": live_pool(scheduler)}
    return {"format_version": 1, "release": release, "project_root": str(project),
            "scheduler_root": str(scheduler), "data_root": str(data_root), "physical_gpu": 1,
            "seeds": list(seeds), "variants": list(variants), "jobs": jobs, "data": data,
            "config_dir": str(config_dir), "initializations": initializations,
            "source": str(release_root / "source"), "files": hashes,
            "python": str(Path(sys.executable).resolve()), "python_version": sys.version,
            "runtime": runtime}


def job_script(plan, job, manifest):
    source = Path(plan["source"])
    verify = [plan["python"], "-B", str(source / "tools/v16/queue_jobs.py"),
              "verify-source", "--manifest", str(manifest)]
    command = [plan["python"], "-B", "-u", str(source / "tools/v16/run_experiment.py"),
               "--project-root", plan["project_root"], "--variant", job["variant"],
               "--seed", str(job["seed"]), "--release", plan["release"],
               "--data-root", plan["data_root"], "--device", "cuda:0",
               "--config-dir", plan.get("config_dir", "configs/v16")]
    initial = plan.get("initializations", {}).get(str(job["seed"]))
    if initial:
        command.extend(["--init-checkpoint", initial["path"], "--init-sha256", initial["sha256"]])
    return "\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail",
        '[ "${GPU:-}" = 1 ] || { echo "ERROR: this job requires pool GPU=1" >&2; exit 1; }',
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1",
        "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1",
        "cd " + shlex.quote(str(source)), " ".join(shlex.quote(x) for x in verify),
        "exec " + " ".join(shlex.quote(x) for x in command), ""])


def enqueue(plan):
    source = Path(plan["source"])
    project = Path(plan["project_root"])
    release_root = source.parent
    release_root.parent.mkdir(parents=True, exist_ok=True)
    release_root.mkdir()
    manifest = release_root / "release.json"
    plan = dict(plan, status="preparing", queued=[])
    write_json(manifest, plan)
    try:
        source.mkdir()
        for relative, expected in plan["files"].items():
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(project / relative), str(target))
            if digest(target) != expected:
                raise ValueError("source changed after preflight: " + relative)
        for initial in plan.get("initializations", {}).values():
            target = Path(initial["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(initial["original"], str(target))
            if digest(target) != initial["sha256"]:
                raise ValueError("initial checkpoint changed after preflight: " + initial["original"])
        verify_source(manifest)
        for relative in ("run/queue/pool/STOP", "run/queue/gpu1/STOP"):
            if (Path(plan["scheduler_root"]) / relative).exists():
                raise ValueError("existing scheduler has a STOP request: " + relative)
        if plan.get("runtime", {}).get("dispatcher_pid"):
            live_pool(Path(plan["scheduler_root"]))
        pending = Path(plan["scheduler_root"]) / "run/queue/pool/pending"
        pending.mkdir(parents=True, exist_ok=True)
        plan["status"] = "publishing"
        write_json(manifest, plan)
        for job in plan["jobs"]:
            path = pending / (job["name"] + "@gpu1.sh")
            publish_job(path, job_script(plan, job, manifest))
            plan["queued"].append(str(path))
            write_json(manifest, plan)
            print("QUEUED " + str(path), flush=True)
        plan["status"] = "queued"
        write_json(manifest, plan)
    except Exception as error:
        plan.update(status="enqueue_failed", error=str(error))
        write_json(manifest, plan)
        raise
    print("RELEASE " + str(manifest))
    return manifest


def release_plan(project_root, release):
    manifest = Path(project_root).resolve() / "run/v16_context" / check_release(release) / "release.json"
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    if Path(plan["project_root"]).resolve() != Path(project_root).resolve() or plan["release"] != release:
        raise ValueError("release identity does not match project/release")
    return manifest, plan


def show_status(project_root, release, tail=12):
    manifest, plan = release_plan(project_root, release)
    queue = Path(plan["scheduler_root"]) / "run/queue"
    print("RELEASE %s (%s)" % (release, plan["status"]))
    for job in plan["jobs"]:
        state = "unknown"
        for part in ("pool/pending", "pool/given", "gpu1/pending", "gpu1/done", "gpu1/failed", "gpu1/running"):
            suffix = {"pool/pending": "@gpu1.sh", "pool/given": "__gpu1.sh"}.get(part, ".sh")
            if (queue / part / (job["name"] + suffix)).exists():
                state = part
        metadata = manifest.parent / "jobs" / ("%s_s%d" % (job["variant"], job["seed"])) / "manifest.json"
        stage = json.loads(metadata.read_text(encoding="utf-8")).get("status") if metadata.is_file() else "not_started"
        print("%s: %s; stage=%s\n  output: %s" % (job["name"], state, stage, job["out"]))
        log = queue / "gpu1/logs" / (job["name"] + ".log")
        if log.is_file() and tail:
            with log.open(encoding="utf-8", errors="replace") as handle:
                for line in deque(handle, maxlen=tail):
                    print("  " + line.rstrip())


def validated_completed_artifacts(project_root, result):
    """Validate every recorded artifact without assuming a queue or new stage."""
    project = Path(project_root).resolve()
    recorded = result.get("artifact_sha256", {})
    if not isinstance(recorded, dict):
        raise ValueError("completed artifact hash inventory must be a mapping")
    paths = {}
    for name, expected in recorded.items():
        if not isinstance(name, str) or not name:
            raise ValueError("unsafe completed artifact path")
        relative = PurePosixPath(name)
        if (not relative.parts or relative.is_absolute() or ".." in relative.parts or
                PureWindowsPath(name).anchor or "\\" in name):
            raise ValueError("unsafe completed artifact path: " + name)
        path = project / Path(*relative.parts)
        try:
            path.resolve().relative_to(project)
        except ValueError:
            raise ValueError("completed artifact path escapes project: " + name)
        if path.is_symlink():
            raise ValueError("unsafe completed artifact symlink: " + name)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError("completed artifact is missing or empty: " + name)
        if digest(path) != expected:
            raise ValueError("completed artifact is changed: " + name)
        paths[name] = path
    return paths


def pack(project_root, release, allow_partial=False):
    """Package a stable artifact snapshot, excluding data, credentials and caches."""
    project = Path(project_root).resolve()
    manifest, plan = release_plan(project, release)
    verify_source(manifest)
    runner = load_module("v16_package_validation", Path(plan["source"]) / "tools/v16/run_experiment.py")
    files = {}
    complete = True
    missing_results = {}
    for job in plan["jobs"]:
        out = Path(job["out"]).resolve()
        out.relative_to(project / "log/v16_context" / release)
        meta = manifest.parent / "jobs" / ("%s_s%d" % (job["variant"], job["seed"]))
        log = Path(plan["scheduler_root"]) / "run/queue/gpu1/logs" / (job["name"] + ".log")
        result_path = out / "comparison_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        checkpoint = out / ("best_val_iou_seed%d.pt" % job["seed"])
        artifacts = [out / name for name in ("summary.json", "train_stats.json", "eval_val.json",
                                             "eval_test.json", "profile.json")]
        artifacts.extend((checkpoint, meta / "config.yaml", meta / "runtime_preflight.json"))
        required = artifacts + [result_path, meta / "manifest.json", log]
        missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
        if result.get("status") != "complete":
            missing.append("comparison_result.status=complete")
        done = not missing
        complete = complete and done
        if missing:
            missing_results[job["name"]] = missing
        if not done and not allow_partial:
            raise ValueError("experiment is not complete: %s; missing: %s; use status or --allow-partial" %
                             (job["name"], ", ".join(missing)))
        # A partial package may omit missing results, but must not hide corruption
        # in an experiment that claimed completion.
        if result.get("status") == "complete":
            if any(result.get(key) != value for key, value in
                   (("release", release), ("variant", job["variant"]), ("seed", job["seed"]))):
                raise ValueError("completed experiment identity mismatch: " + job["name"])
            if checkpoint.is_file() and digest(checkpoint) != result.get("checkpoint_sha256"):
                raise ValueError("selected checkpoint is missing or changed: %s" % checkpoint)
            recorded = validated_completed_artifacts(project, result)
            for path in artifacts:
                if path.is_file() and path.relative_to(project).as_posix() not in recorded:
                    raise ValueError("completed artifact is changed or lacks its recorded hash: %s" % path)
        if done:
            summary = runner.read_json(out / "summary.json")
            if hasattr(runner, "validate_training_summary"):
                runner.validate_training_summary(summary, runner.load_yaml(meta / "config.yaml"))
            elif (summary.get("epochs_completed") != 50 or summary.get("complete") is not True
                    or summary.get("probe_only") is not False):
                raise ValueError("completed legacy experiment lacks a formal 50-epoch summary")
            if runner.read_json(meta / "manifest.json").get("status") != "complete":
                raise ValueError("job metadata has not reached complete: " + job["name"])
            runner.validate_preflight(meta / "runtime_preflight.json")
            for split in ("val", "test"):
                evaluation = runner.validate_evaluation(out / ("eval_%s.json" % split), job["variant"])
                if evaluation.get("checkpoint_sha256") != result["checkpoint_sha256"]:
                    raise ValueError("evaluation checkpoint identity mismatch: " + split)
            runner.validate_profile(out / "profile.json", job["variant"], result["checkpoint_sha256"])
        if out.is_dir():
            for path in out.rglob("*"):
                if path.is_file() and path.suffix.lower() in (".json", ".jsonl", ".yaml", ".log", ".txt", ".pt", ".csv", ".tsv"):
                    files[path.relative_to(project).as_posix()] = path
        if log.is_file():
            files["scheduler_logs/" + log.name] = log
    files[manifest.relative_to(project).as_posix()] = manifest
    for initial in plan.get("initializations", {}).values():
        path = Path(initial["path"])
        files[path.relative_to(project).as_posix()] = path
    for relative in plan["files"]:
        path = Path(plan["source"]) / relative
        files[path.relative_to(project).as_posix()] = path
    for path in (manifest.parent / "jobs").rglob("*"):
        if path.is_file() and path.suffix in (".json", ".yaml"):
            files[path.relative_to(project).as_posix()] = path
    before = {}
    for name, path in files.items():
        if path.is_symlink() or ".." in Path(name).parts or Path(name).is_absolute():
            raise ValueError("unsafe archive path: " + name)
        before[name] = (path.stat().st_size, path.stat().st_mtime_ns)
    result_dir = project / "Res"
    result_dir.mkdir(exist_ok=True)
    target = result_dir / ("v16_%s_%s_%s%s.tgz" %
                           (release, time.strftime("%Y%m%d_%H%M%S"), uuid.uuid4().hex[:6],
                            "_partial" if not complete else ""))
    temporary = target.with_suffix(".tmp")
    try:
        with temporary.open("xb") as handle:
            with tarfile.open(fileobj=handle, mode="w:gz") as archive:
                for name, path in sorted(files.items()):
                    archive.add(str(path), arcname=name, recursive=False)
                report = json.dumps({"release": release, "complete": complete,
                                     "missing_results": missing_results, "jobs": plan["jobs"]},
                                    indent=2, ensure_ascii=False).encode("utf-8")
                info = tarfile.TarInfo("package_manifest.json")
                info.size = len(report)
                archive.addfile(info, io.BytesIO(report))
        if any((path.stat().st_size, path.stat().st_mtime_ns) != before[name] for name, path in files.items()):
            raise ValueError("files changed while packaging; retry after the current write finishes")
        os.link(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()
    print("DOWNLOAD: " + str(target))
    print("SIZE_MiB: %.2f" % (target.stat().st_size / 1024.0 ** 2))
    print("SHA256: " + digest(target))
    if not complete:
        print("PARTIAL: see package_manifest.json for missing results")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("enqueue")
    add.add_argument("--project-root", type=Path, default=ROOT)
    add.add_argument("--scheduler-root", type=Path)
    add.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA))
    add.add_argument("--release", default=DEFAULT_RELEASE)
    add.add_argument("--config-dir", default="configs/v16/r2")
    add.add_argument("--init-root", type=Path,
                     help="Optional directory containing seed37.pt and seed38.pt; weights are frozen into this release")
    add.add_argument("--seeds", nargs="+", type=int, default=[37, 38])
    add.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    add.add_argument("--dry-run", action="store_true")
    for name in ("status", "pack"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--project-root", type=Path, default=ROOT)
        cmd.add_argument("--release", default=DEFAULT_RELEASE)
        if name == "status":
            cmd.add_argument("--tail", type=int, default=12)
        else:
            cmd.add_argument("--allow-partial", action="store_true")
    verify = sub.add_parser("verify-source")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-source":
            verify_source(args.manifest)
            print("SOURCE_VERIFIED")
        elif args.command == "status":
            if args.tail < 0:
                raise ValueError("tail must be nonnegative")
            show_status(args.project_root, args.release, args.tail)
        elif args.command == "pack":
            pack(args.project_root, args.release, args.allow_partial)
        else:
            scheduler = args.scheduler_root or args.project_root.resolve().parent / "EV-UAV-v13"
            plan = prepare_plan(args.project_root, scheduler, args.data_root,
                                args.release, args.seeds, args.variants,
                                config_dir=args.config_dir, init_root=args.init_root)
            if args.dry_run:
                print(json.dumps({k: v for k, v in plan.items() if k != "files"}, indent=2))
                print("DRY_RUN_OK: %d training job(s); GPU 1; %d frozen source files" %
                      (len(plan["jobs"]), len(plan["files"])))
            else:
                enqueue(plan)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, "ERROR: %s\n" % error)


if __name__ == "__main__":
    main()
