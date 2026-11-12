"""Queue, inspect and package V14 overlap runs through the existing GPU-1 pool.

Default: one independently trained conditional-fusion job, seed 37. No server
process is started, killed or restarted. Python 3.8 compatible.
"""
import argparse
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE = "overlap_20261112_r1"
DEFAULT_DATA = "/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset"
VARIANTS = ("conditional", "mean", "concat")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_module("v14_overlap_queue_common", ROOT / "tools/v14/queue.py")
digest = common.digest
check_release = common.check_release
dataset_summary = common.dataset_summary
live_pool = common.live_pool
queue_scripts = common.queue_scripts
write_json = common.write_json
publish_job = common.publish_job
verify_source = common.verify_source
SPLIT_COUNTS = common.SPLIT_COUNTS


def source_files(root):
    root = Path(root).resolve()
    required = {root / p for p in (
        "train_stream_v1.py", "tools/v14/train_overlap.py", "tools/v14/run_overlap.py",
        "tools/v14/profile_overlap.py",
        "tools/v14/run_experiment.py", "tools/v14/queue.py", "tools/v14/overlap_queue.py",
        "configs/evisseg_stream_tif_comm_peak.yaml", "configs/v14/overlap_conditional.yaml")}
    paths = set(required)
    for directory in ("model", "dataset", "utils", "tools/v14"):
        paths.update((root / directory).rglob("*.py"))
    paths.update((root / "configs/v14").glob("*.yaml"))
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


def validate_configs(root):
    module = load_module("v14_overlap_preflight", Path(root) / "tools/v14/run_overlap.py")
    return module.validate_configs(source_root=Path(root))


def dependency_check(root):
    command = [sys.executable, "-B", "-c",
               "import tools.v14.train_overlap, tools.v14.profile_overlap, "
               "tools.v14.run_overlap, yaml; import torch; print('torch=' + torch.__version__)"]
    result = subprocess.run(command, cwd=str(root), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode:
        raise ValueError("overlap dependency check failed:\n" + result.stderr.strip())
    return result.stdout.strip()


def prepare_plan(project_root, scheduler_root, data_root, release, seeds=(37,),
                 variants=("conditional",), runtime_checks=True):
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
        raise ValueError("scheduler root must be the existing V13 project, separate from overlap")
    for relative in ("tools/final/gpu_pool.sh", "tools/final/gpu_queue.sh"):
        if not (scheduler / relative).is_file():
            raise ValueError("existing scheduler file missing: %s" % (scheduler / relative))
    for relative in ("run/queue/pool/STOP", "run/queue/gpu1/STOP"):
        if (scheduler / relative).exists():
            raise ValueError("existing scheduler has a STOP request: " + relative)
    release_root = project / "run/v14_overlap" / release
    if release_root.exists():
        raise ValueError("release already exists; refusing duplicate or partial enqueue: %s" % release_root)
    hashes = source_files(project)
    configs = validate_configs(project)
    if any(v not in configs for v in variants):
        raise ValueError("requested variant configuration unavailable")
    data = dataset_summary(data_root)
    scripts = queue_scripts(scheduler)
    prefix = "v14_overlap_%s_" % release
    if any(prefix in path.name for path in scripts):
        raise ValueError("release jobs already exist in the scheduler")
    number = max([1090] + [int(m.group(1)) for path in scripts
                          for m in [re.match(r"^(\d+)_", path.name)] if m])
    jobs = []
    for seed in seeds:
        for variant in variants:
            number += 10
            out = project / "log/v14_overlap" / release / ("%s_s%d" % (variant, seed))
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
            "source": str(release_root / "source"), "files": hashes,
            "python": str(Path(sys.executable).resolve()), "python_version": sys.version,
            "runtime": runtime}


def job_script(plan, job, manifest):
    source = Path(plan["source"])
    verify = [plan["python"], "-B", str(source / "tools/v14/overlap_queue.py"),
              "verify-source", "--manifest", str(manifest)]
    command = [plan["python"], "-B", "-u", str(source / "tools/v14/run_overlap.py"),
               "--project-root", plan["project_root"], "--variant", job["variant"],
               "--seed", str(job["seed"]), "--release", plan["release"],
               "--data-root", plan["data_root"], "--device", "cuda:0"]
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
        verify_source(manifest)
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
    manifest = Path(project_root).resolve() / "run/v14_overlap" / check_release(release) / "release.json"
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
        for part in ("pool/pending", "gpu1/pending", "gpu1/done", "gpu1/failed", "gpu1/running"):
            suffix = "@gpu1.sh" if part == "pool/pending" else ".sh"
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


def pack(project_root, release, allow_partial=False):
    """Package a stable artifact snapshot, excluding data, credentials and caches."""
    project = Path(project_root).resolve()
    manifest, plan = release_plan(project, release)
    verify_source(manifest)
    files = {}
    complete = True
    for job in plan["jobs"]:
        out = Path(job["out"]).resolve()
        out.relative_to(project / "log/v14_overlap" / release)
        result_path = out / "comparison_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        done = result.get("status") == "complete"
        complete = complete and done
        if not done and not allow_partial:
            raise ValueError("experiment is not complete: %s; use status first" % job["name"])
        if done:
            checkpoint = out / ("best_val_iou_seed%d.pt" % job["seed"])
            if not checkpoint.is_file() or digest(checkpoint) != result.get("checkpoint_sha256"):
                raise ValueError("selected checkpoint is missing or changed: %s" % checkpoint)
            for item in ("eval_val.json", "eval_test.json", "profile.json"):
                if not (out / item).is_file():
                    raise ValueError("completed experiment is missing evaluation: " + item)
        if out.is_dir():
            for path in out.rglob("*"):
                if path.is_file() and path.suffix.lower() in (".json", ".jsonl", ".yaml", ".log", ".txt", ".pt", ".csv", ".tsv"):
                    files[path.relative_to(project).as_posix()] = path
        log = Path(plan["scheduler_root"]) / "run/queue/gpu1/logs" / (job["name"] + ".log")
        if log.is_file():
            files["scheduler_logs/" + log.name] = log
    files[manifest.relative_to(project).as_posix()] = manifest
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
    target = result_dir / ("v14_%s_%s_%s%s.tgz" %
                           (release, time.strftime("%Y%m%d_%H%M%S"), uuid.uuid4().hex[:6],
                            "_partial" if not complete else ""))
    temporary = target.with_suffix(".tmp")
    try:
        with temporary.open("xb") as handle:
            with tarfile.open(fileobj=handle, mode="w:gz") as archive:
                for name, path in sorted(files.items()):
                    archive.add(str(path), arcname=name, recursive=False)
        if any((path.stat().st_size, path.stat().st_mtime_ns) != before[name] for name, path in files.items()):
            raise ValueError("files changed while packaging; retry after the current write finishes")
        os.link(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()
    print("DOWNLOAD: " + str(target))
    print("SIZE_MiB: %.2f" % (target.stat().st_size / 1024.0 ** 2))
    print("SHA256: " + digest(target))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("enqueue")
    add.add_argument("--project-root", type=Path, default=ROOT)
    add.add_argument("--scheduler-root", type=Path)
    add.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA))
    add.add_argument("--release", default=DEFAULT_RELEASE)
    add.add_argument("--seeds", nargs="+", type=int, default=[37])
    add.add_argument("--variants", nargs="+", choices=VARIANTS, default=["conditional"])
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
                                args.release, args.seeds, args.variants)
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
