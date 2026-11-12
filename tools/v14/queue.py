"""Queue isolated V14 BIR experiments in the existing V13 GPU-1 pool.

This tool never starts/stops a dispatcher, worker, holder, or training process.
Python 3.8 compatible. A release contains a frozen source snapshot and jobs that
write their results under the V14 project, not under the scheduler project.
"""
import argparse
import hashlib
import importlib.util
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
DEFAULT_RELEASE = "mem_ablation_20261112_r1"
DEFAULT_DATA = "/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset"
MODES = ("carry", "reset")
SPLIT_COUNTS = {"train": 99, "val": 24, "test": 24}


def digest(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_release(release):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", release):
        raise ValueError("release must contain only letters, digits, _, . and -")
    return release


def source_files(root):
    """Explicit code/config allowlist: no weights, data, logs or credentials."""
    root = Path(root).resolve()
    required = ["train_stream_v1.py", "tools/stream_train_stats.py",
                "tools/v14/run_experiment.py", "tools/v14/queue.py",
                "configs/evisseg_stream_tif_notransport.yaml",
                "configs/v14/bir_carry.yaml", "configs/v14/bir_reset.yaml"]
    required_paths = {root / item for item in required}
    paths = set(required_paths)
    for directory in ("model", "dataset", "utils", "tools/v14"):
        paths.update((root / directory).rglob("*.py"))
    paths.update((root / "configs/v14").glob("*.yaml"))
    result = {}
    for path in sorted(paths):
        if not path.is_file() or (path in required_paths and path.stat().st_size == 0):
            raise ValueError("required source file missing or empty: %s" % path)
        try:
            path.resolve().relative_to(root)
        except ValueError:
            raise ValueError("source symlink escapes project: %s" % path)
        relative = path.relative_to(root).as_posix()
        result[relative] = digest(path)
    return result


def validate_configs(root):
    path = Path(root) / "tools/v14/run_experiment.py"
    spec = importlib.util.spec_from_file_location("v14_runner_preflight", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_pair(source_root=Path(root))


def dataset_summary(data_root):
    result = {}
    for split, expected in SPLIT_COUNTS.items():
        directory = Path(data_root) / split
        files = sorted(directory.glob("*.npz"))
        if len(files) != expected or any(p.stat().st_size == 0 for p in files):
            raise ValueError("%s must contain %d nonempty .npz files (found %d)" %
                             (directory, expected, len(files)))
        result[split] = {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}
    return result


def dependency_check(root):
    # Importing the actual training entry point checks its dependencies without
    # constructing a model or initializing CUDA. -B avoids source mutations.
    command = [sys.executable, "-B", "-c",
               "import train_stream_v1, yaml, cv2, pandas, tqdm; "
               "import torch; print('torch=' + torch.__version__)"]
    result = subprocess.run(command, cwd=str(root), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode:
        raise ValueError("training dependency check failed:\n" + result.stderr.strip())
    return result.stdout.strip()


def live_pool(scheduler_root, proc_root=Path("/proc")):
    """Inspect only matching processes; unreadable unrelated PIDs are ignored."""
    expected = Path(scheduler_root).resolve()
    matches = []
    for proc in Path(proc_root).iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
            if not any(a.endswith(b"tools/final/gpu_pool.sh") for a in args):
                continue
            if (proc / "cwd").resolve() != expected:
                continue
            env = dict(item.split(b"=", 1) for item in
                       (proc / "environ").read_bytes().split(b"\0") if b"=" in item)
            if b"1" not in env.get(b"GPUS", b"1").split():
                continue
            matches.append(int(proc.name))
        except (OSError, ValueError):
            continue
    if len(matches) != 1:
        raise ValueError("expected exactly one live GPU-1 dispatcher in %s; found %s. "
                         "Inspect the existing pool; this tool will not start another one." %
                         (expected, matches))
    return matches[0]


def queue_scripts(scheduler_root):
    queue = Path(scheduler_root) / "run/queue"
    paths = []
    for part in ("pool/pending", "pool/given", "gpu1/pending", "gpu1/running",
                 "gpu1/done", "gpu1/failed"):
        paths.extend((queue / part).glob("*.sh"))
    return paths


def prepare_plan(project_root, scheduler_root, data_root, release, seeds=(37,),
                 runtime_checks=True):
    project_root = Path(project_root).resolve()
    scheduler_root = Path(scheduler_root).resolve()
    data_root = Path(data_root).resolve()
    check_release(release)
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    if scheduler_root == project_root:
        raise ValueError("scheduler root must be the existing V13 project, separate from V14")
    for item in ("tools/final/gpu_pool.sh", "tools/final/gpu_queue.sh"):
        if not (scheduler_root / item).is_file():
            raise ValueError("existing scheduler file missing: %s" % (scheduler_root / item))
    for item in ("run/queue/pool/STOP", "run/queue/gpu1/STOP"):
        if (scheduler_root / item).exists():
            raise ValueError("existing scheduler has a STOP request: %s" % item)
    release_root = project_root / "run/v14" / release
    if release_root.exists():
        raise ValueError("release already exists; refusing duplicate or partial enqueue: %s" % release_root)
    hashes = source_files(project_root)
    validate_configs(project_root)
    data = dataset_summary(data_root)
    paths = queue_scripts(scheduler_root)
    prefix = "v14_bir_%s_" % release
    duplicates = [str(p) for p in paths if prefix in p.name]
    if duplicates:
        raise ValueError("release jobs already exist: " + ", ".join(duplicates))
    number = max([990] + [int(m.group(1)) for p in paths
                         for m in [re.match(r"^(\d+)_", p.name)] if m])
    jobs = []
    for seed in seeds:
        for mode in MODES:
            number += 10
            name = "%d_%s%s_s%d" % (number, prefix, mode, seed)
            out = project_root / "log/v14" / release / ("%s_s%d" % (mode, seed))
            if out.exists():
                raise ValueError("experiment output already exists; refusing retraining over it: %s" % out)
            jobs.append({"name": name, "mode": mode, "seed": seed, "out": str(out)})
    runtime = {}
    if runtime_checks:
        if sys.platform != "linux":
            raise ValueError("enqueue is for the Linux server; upload the project before running it")
        runtime["dependencies"] = dependency_check(project_root)
        runtime["dispatcher_pid"] = live_pool(scheduler_root)
    return {"format_version": 1, "release": release, "project_root": str(project_root),
            "scheduler_root": str(scheduler_root), "data_root": str(data_root),
            "physical_gpu": 1, "seeds": list(seeds), "jobs": jobs, "data": data,
            "source": str(release_root / "source"), "files": hashes,
            "python": str(Path(sys.executable).resolve()), "python_version": sys.version,
            "runtime": runtime}


def job_script(plan, job, manifest):
    source = Path(plan["source"])
    executable = plan["python"]
    verify = [executable, "-B", str(source / "tools/v14/queue.py"),
              "verify-source", "--manifest", str(manifest)]
    command = [executable, "-B", "-u", str(source / "tools/v14/run_experiment.py"),
               "--project-root", plan["project_root"], "--mode", job["mode"],
               "--seed", str(job["seed"]), "--release", plan["release"],
               "--data-root", plan["data_root"], "--device", "cuda:0"]
    return "\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail",
        '[ "${GPU:-}" = 1 ] || { echo "ERROR: this job requires pool GPU=1" >&2; exit 1; }',
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1",
        "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1",
        "cd " + shlex.quote(str(source)),
        " ".join(shlex.quote(x) for x in verify),
        "exec " + " ".join(shlex.quote(x) for x in command), ""])


def write_json(path, obj):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temp), str(path))


def publish_job(path, text):
    # A complete .sh becomes visible atomically. Hard linking refuses to
    # replace an existing job even if another enqueue races with this one.
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def enqueue(plan):
    project = Path(plan["project_root"])
    source = Path(plan["source"])
    release_root = source.parent
    release_root.parent.mkdir(parents=True, exist_ok=True)
    release_root.mkdir()  # exclusive release reservation; duplicate calls fail
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
                raise ValueError("source changed after preflight: %s" % relative)
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
        plan["status"] = "enqueue_failed"
        plan["error"] = str(error)
        write_json(manifest, plan)
        raise
    print("RELEASE " + str(manifest))
    return manifest


def verify_source(manifest):
    plan = json.loads(Path(manifest).read_text(encoding="utf-8"))
    root = Path(plan["source"]).resolve()
    for relative, expected in plan["files"].items():
        path = root / relative
        try:
            path.resolve().relative_to(root)
        except ValueError:
            raise ValueError("invalid snapshot path: %s" % relative)
        if not path.is_file() or digest(path) != expected:
            raise ValueError("source snapshot changed or missing: %s" % relative)
    return plan


def show_status(project_root, release, tail=8):
    manifest = Path(project_root) / "run/v14" / check_release(release) / "release.json"
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    queue = Path(plan["scheduler_root"]) / "run/queue"
    print("RELEASE %s (%s)" % (release, plan["status"]))
    for job in plan["jobs"]:
        name = job["name"]
        state = "unknown"
        for part in ("pool/pending", "gpu1/pending", "gpu1/done", "gpu1/failed", "gpu1/running"):
            suffix = "@gpu1.sh" if part == "pool/pending" else ".sh"
            if (queue / part / (name + suffix)).exists():
                state = part
        print("%s: %s\n  output: %s" % (name, state, job["out"]))
        log = queue / "gpu1/logs" / (name + ".log")
        if log.is_file() and tail:
            print("  log: %s" % log)
            # Logs are bounded here by line count at display time; training
            # errors remain in the full file in the existing scheduler.
            from collections import deque
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
    add.add_argument("--dry-run", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--project-root", type=Path, default=ROOT)
    status.add_argument("--release", default=DEFAULT_RELEASE)
    status.add_argument("--tail", type=int, default=8)
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
            plan = prepare_plan(args.project_root, scheduler, args.data_root,
                                args.release, args.seeds)
            if args.dry_run:
                print(json.dumps({k: v for k, v in plan.items() if k != "files"},
                                 ensure_ascii=False, indent=2))
                print("DRY_RUN_OK: %d independent training jobs; GPU 1; %d frozen source files" %
                      (len(plan["jobs"]), len(plan["files"])))
            else:
                enqueue(plan)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, "ERROR: %s\n" % error)


if __name__ == "__main__":
    main()
