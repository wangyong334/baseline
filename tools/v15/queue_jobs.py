"""Publish frozen V15 jobs to the already-running V13 GPU-1 scheduler.

Never starts, restarts or kills a pool, worker, holder or training process.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.v15.run_experiment import (DEFAULT_DATA, DEFAULT_RELEASE, MODES, checked_name,
                                     dataset_inventory, digest, read_json, source_files,
                                     validate_configs, write_json)


def live_pool(scheduler_root, proc_root=Path("/proc")):
    matches = []
    for proc in Path(proc_root).iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
            if not any(arg.endswith(b"tools/final/gpu_pool.sh") for arg in args):
                continue
            if (proc / "cwd").resolve() != Path(scheduler_root).resolve():
                continue
            env = dict(item.split(b"=", 1) for item in (proc / "environ").read_bytes().split(b"\0") if b"=" in item)
            if b"1" in env.get(b"GPUS", b"1").replace(b",", b" ").split():
                matches.append(int(proc.name))
        except (OSError, ValueError):
            continue
    if len(matches) != 1:
        raise ValueError("expected one existing GPU-1 dispatcher in %s, found %s; inspect it, do not start another" %
                         (scheduler_root, matches))
    return matches[0]


def queue_scripts(scheduler_root):
    result = []
    for part in ("pool/pending", "pool/given", "gpu1/pending", "gpu1/running", "gpu1/done", "gpu1/failed"):
        result.extend((Path(scheduler_root) / "run/queue" / part).glob("*.sh"))
    return result


def dependency_check(root):
    result = subprocess.run([sys.executable, "-B", "-c",
                             "import tools.v15.train, tools.v15.profile, yaml, torch; print('torch=' + torch.__version__)"],
                            cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    if result.returncode:
        raise ValueError("training dependency check failed:\n" + result.stderr.strip())
    return result.stdout.strip()


def prepare_plan(project_root, scheduler_root, data_root, release, seeds=(37,),
                 modes=("observation",), runtime_checks=True):
    project, scheduler, data = (Path(path).resolve() for path in (project_root, scheduler_root, data_root))
    checked_name(release)
    seeds, modes = tuple(int(s) for s in seeds), tuple(modes)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique and nonnegative")
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("modes must be unique choices from " + ", ".join(MODES))
    if project == scheduler:
        raise ValueError("V15 project and existing V13 scheduler must be separate directories")
    for relative in ("tools/final/gpu_pool.sh", "tools/final/gpu_queue.sh"):
        if not (scheduler / relative).is_file():
            raise ValueError("existing scheduler file missing: " + relative)
    for relative in ("run/queue/pool/STOP", "run/queue/gpu1/STOP"):
        if (scheduler / relative).exists():
            raise ValueError("scheduler has a STOP request: " + relative)
    release_root = project / "run/v15" / release
    if release_root.exists():
        raise ValueError("release already exists; refusing duplicate/partial enqueue: %s" % release_root)
    hashes = source_files(project)
    validate_configs(project)
    inventory = dataset_inventory(data)
    paths = queue_scripts(scheduler)
    prefix = "v15_feature_%s_" % release
    if any(prefix in path.name for path in paths):
        raise ValueError("release is already present in the scheduler")
    number = max([1190] + [int(match.group(1)) for path in paths for match in [re.match(r"^(\d+)_", path.name)] if match])
    jobs = []
    for seed in seeds:
        for mode in modes:
            number += 10
            out = project / "log/v15" / release / ("%s_s%d" % (mode, seed))
            if out.exists():
                raise ValueError("output already exists: %s" % out)
            jobs.append({"name": "%d_%s%s_s%d" % (number, prefix, mode, seed),
                         "variant": mode, "seed": seed, "out": str(out)})
    runtime = {}
    if runtime_checks:
        if sys.platform != "linux":
            raise ValueError("enqueue must run on the Linux server after SFTP upload")
        runtime = {"dependencies": dependency_check(project), "dispatcher_pid": live_pool(scheduler)}
    return {"format_version": 1, "release": release, "project_root": str(project),
            "scheduler_root": str(scheduler), "data_root": str(data), "physical_gpu": 1,
            "seeds": list(seeds), "modes": list(modes), "jobs": jobs, "dataset": inventory,
            "source": str(release_root / "source"), "files": hashes,
            "python": str(Path(sys.executable).resolve()), "python_version": sys.version, "runtime": runtime}


def job_script(plan, job, manifest):
    source = Path(plan["source"])
    verify = [plan["python"], "-B", str(source / "tools/v15/queue_jobs.py"), "verify-source", "--manifest", str(manifest)]
    command = [plan["python"], "-B", "-u", str(source / "tools/v15/run_experiment.py"),
               "--project-root", plan["project_root"], "--data-root", plan["data_root"],
               "--release", plan["release"], "--variant", job["variant"], "--seed", str(job["seed"]), "--device", "cuda:0"]
    return "\n".join(["#!/usr/bin/env bash", "set -euo pipefail",
                      '[ "${GPU:-}" = 1 ] || { echo "ERROR: physical GPU 1 required" >&2; exit 1; }',
                      "export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1",
                      "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1",
                      "cd " + shlex.quote(str(source)), " ".join(shlex.quote(arg) for arg in verify),
                      "exec " + " ".join(shlex.quote(arg) for arg in command), ""])


def publish_job(path, script):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(script)
            handle.flush()
            os.fsync(handle.fileno())
        # A complete script appears atomically; a competing existing job wins.
        os.link(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_source(manifest):
    plan = read_json(manifest)
    root = Path(plan["source"]).resolve()
    for relative, expected in plan["files"].items():
        path = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
            raise ValueError("unsafe source path: " + relative)
        path.resolve().relative_to(root)
        if not path.is_file() or digest(path) != expected:
            raise ValueError("frozen source missing or modified: " + relative)
    return plan


def enqueue(plan):
    source = Path(plan["source"])
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
            shutil.copyfile(str(Path(plan["project_root"]) / relative), str(target))
            if digest(target) != expected:
                raise ValueError("source changed since preflight: " + relative)
        verify_source(manifest)
        if plan.get("runtime", {}).get("dependencies"):
            # Import from the snapshot, not the editable checkout. This catches
            # forgotten local imports before the GPU worker sees any task.
            dependency_check(source)
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
    project = Path(project_root).resolve()
    manifest = project / "run/v15" / checked_name(release) / "release.json"
    plan = read_json(manifest)
    if Path(plan["project_root"]).resolve() != project or plan["release"] != release:
        raise ValueError("release identity does not match project/release")
    if Path(plan["source"]).resolve() != manifest.parent / "source":
        raise ValueError("release source is outside its snapshot directory")
    return manifest, plan


def show_status(project_root, release, tail=12):
    manifest, plan = release_plan(project_root, release)
    print("RELEASE %s (%s)" % (release, plan["status"]))
    queue = Path(plan["scheduler_root"]) / "run/queue"
    for job in plan["jobs"]:
        state = "unknown"
        for part in ("pool/pending", "gpu1/pending", "gpu1/done", "gpu1/failed", "gpu1/running"):
            suffix = "@gpu1.sh" if part == "pool/pending" else ".sh"
            if (queue / part / (job["name"] + suffix)).exists():
                state = part
        metadata = manifest.parent / "jobs" / ("%s_s%d" % (job["variant"], job["seed"])) / "manifest.json"
        stage = read_json(metadata).get("status") if metadata.is_file() else "not_started"
        print("%s: %s; stage=%s\n  output: %s" % (job["name"], state, stage, job["out"]))
        log = queue / "gpu1/logs" / (job["name"] + ".log")
        if log.is_file() and tail:
            with log.open(encoding="utf-8", errors="replace") as handle:
                for line in deque(handle, maxlen=tail):
                    print("  " + line.rstrip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("enqueue")
    add.add_argument("--project-root", type=Path, default=ROOT)
    add.add_argument("--scheduler-root", type=Path)
    add.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA))
    add.add_argument("--release", default=DEFAULT_RELEASE)
    add.add_argument("--seeds", nargs="+", type=int, default=[37])
    add.add_argument("--modes", nargs="+", choices=MODES, default=["observation"])
    add.add_argument("--dry-run", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--project-root", type=Path, default=ROOT)
    status.add_argument("--release", default=DEFAULT_RELEASE)
    status.add_argument("--tail", type=int, default=12)
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
        else:
            scheduler = args.scheduler_root or args.project_root.resolve().parent / "EV-UAV-v13"
            plan = prepare_plan(args.project_root, scheduler, args.data_root, args.release, args.seeds, args.modes)
            if args.dry_run:
                print(json.dumps({k: v for k, v in plan.items() if k not in ("files", "dataset")}, indent=2))
                print("DRY_RUN_OK: %d training job(s); physical GPU 1; %d source files" % (len(plan["jobs"]), len(plan["files"])))
            else:
                enqueue(plan)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, "ERROR: %s\n" % error)


if __name__ == "__main__":
    main()
