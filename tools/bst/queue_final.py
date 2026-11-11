"""Publish GPU-1-only final runs, or safely prepare the existing pool after reboot.

Run on the Linux server after SFTP upload. No SSH, process termination, or
automatic retry of interrupted experiments. Python 3.8 compatible.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def backbone_seed(path):
    """Read the recorded BIR seed, rather than trusting the filename.

    These are the user's trusted local training checkpoints. Explicit loading
    supports both recent PyTorch and the server's older weights_only-less API.
    """
    import torch
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    seed = config.get("seed") if isinstance(config, dict) else None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("cannot verify BIR seed: checkpoint.config.seed must be an integer: " + str(path))
    return seed


def checked_child(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("path escapes project: " + str(path))
    if path == root:
        raise ValueError("operation must target a child of the project")
    return path


def process_inventory(root):
    """Only inspect exact script argv entries; never match our command text."""
    records = []
    for proc in Path("/proc").glob("[0-9]*"):
        kind = None
        try:
            args = (proc/"cmdline").read_bytes().decode(errors="replace").split("\0")
            kind = next((Path(arg).name for arg in args[1:]
                         if arg and Path(arg).name in ("gpu_pool.sh", "gpu_queue.sh")), None)
            if kind is None:
                continue
            cwd = (proc/"cwd").resolve(strict=True)
            env = dict(item.split("=", 1) for item in (proc/"environ").read_bytes().decode(errors="replace").split("\0") if "=" in item)
            gpus = env.get("GPUS", "1").split() if kind == "gpu_pool.sh" else [env.get("GPU", "unknown")]
            records.append({"pid": int(proc.name), "kind": kind, "cwd": str(cwd),
                            "ours": cwd == Path(root).resolve(), "gpus": gpus,
                            "queue_name": env.get("QNAME", "gpu"+env.get("GPU", "unknown"))})
        except (FileNotFoundError, ProcessLookupError):
            # A process can exit while /proc is read; no action is taken on it.
            continue
        except (OSError, ValueError) as error:
            if kind is not None:
                raise RuntimeError("cannot inspect existing scheduler %s: %s" % (proc, error))
            continue
    return records


def archive_stale(root):
    """Move only queue control/job files inside this project; keep all logs."""
    root = Path(root).resolve()
    queue = checked_child(root, root/"run/queue")
    files = []
    for folder in ["pool/pending", "gpu1/pending", "gpu1/running"]:
        files.extend(sorted((queue/folder).glob("*.sh")))
    for name in ["pool/STOP", "gpu1/STOP", "gpu1/HOLD_STOP", "gpu1/runner.pid", "gpu1/hold.pid"]:
        if (queue/name).exists():
            files.append(queue/name)
    if not files:
        return None, []
    archive = checked_child(root, queue/"reboot_archive"/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]))
    moved = []
    for path in files:
        if path.is_symlink():
            raise ValueError("refusing symlink queue file " + str(path))
        source = checked_child(queue, path)
        dest = checked_child(archive, archive/source.relative_to(queue))
        if not source.is_file() or source.is_symlink():
            raise ValueError("refusing nonregular queue file " + str(path))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
        moved.append(str(source.relative_to(queue)))
    (archive/"manifest.json").write_text(json.dumps({"archived": moved, "requeued": False,
        "reason": "explicit reboot recovery; interrupted work is preserved, not resumed"}, indent=2), encoding="utf-8")
    return archive, moved


def pool_preflight(root, recover):
    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("pool startup is for the Linux server only")
    records = process_inventory(root)
    for item in records:
        if "1" in item["gpus"] and not item["ours"]:
            raise RuntimeError("GPU 1 already has a dispatcher/runner from another directory: " + str(item))
    ours = [item for item in records if item["ours"]]
    if any(item["kind"] == "gpu_queue.sh" and "1" in item["gpus"] and
           item["queue_name"] != "gpu1" for item in ours):
        raise RuntimeError("existing GPU 1 runner uses a custom QNAME; inspect it before starting the final pool")
    if any(item["kind"] == "gpu_pool.sh" and item["gpus"] != ["1"] for item in ours):
        raise RuntimeError("existing project dispatcher is not GPU-1-only; inspect it before restarting")
    if any(item["kind"] == "gpu_pool.sh" for item in ours):
        print("Project dispatcher already running; no second dispatcher started.")
        return 3
    if any(item["kind"] == "gpu_queue.sh" for item in ours) and recover:
        raise RuntimeError("A project runner is alive: refusing reboot recovery; start without --recover-reboot")
    if recover:
        archive, moved = archive_stale(root)
        print("Reboot recovery: archived %d control/job files to %s; logs preserved; no old job requeued." % (len(moved), archive))
    else:
        stale = list((Path(root)/"run/queue/gpu1/running").glob("*.sh"))
        if stale and not any(item["kind"] == "gpu_queue.sh" for item in ours):
            raise RuntimeError("Stale running jobs found. Use --recover-reboot to archive interrupted work first.")
        for path in [Path(root)/"run/queue/pool/STOP", Path(root)/"run/queue/gpu1/STOP"]:
            if path.exists():
                raise RuntimeError("STOP marker found; explicit --recover-reboot required: " + str(path))
    return 0


def source_files(root):
    root = Path(root)
    files = list(root.glob("*.py"))
    for folder in ["model", "dataset", "utils", "tools", "configs"]:
        files.extend(p for p in (root/folder).rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".json", ".yaml", ".yml"))
    return sorted(set(files))


def plan(root, release, data_root, backbones, config):
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", release):
        raise ValueError("release must be 1-80 letters/digits/underscore/dot/dash and start alphanumeric")
    data_root = Path(data_root).resolve()
    for split in ("train", "val", "test"):
        if not list((data_root/split).glob("*.npz")):
            raise FileNotFoundError("missing dataset split: " + str(data_root/split))
    required = [root/"tools/bst"/name for name in ("run_final.sh", "export_priors.py", "train.py",
                "fit_policy.py", "evaluate.py", "final_metrics.py", "audit_reconsideration.py")]
    required.append(Path(config))
    for path in required + [Path(backbones[s]) for s in (37, 38)]:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError("required nonempty file missing: " + str(path))
    hashes = {s: digest(backbones[s]) for s in (37, 38)}
    if hashes[37] == hashes[38]:
        raise ValueError("37 and 38 point to identical backbone bytes; independent BIR seeds were requested")
    for seed in (37, 38):
        recorded = backbone_seed(backbones[seed])
        if recorded != seed:
            raise ValueError("BIR seed mismatch: requested %d, checkpoint.config.seed=%d: %s" %
                             (seed, recorded, backbones[seed]))
    release_dir = checked_child(root, root/"run/bst_final"/release)
    if release_dir.exists():
        raise FileExistsError("release already exists; choose a NEW release tag: " + str(release_dir))
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    if cfg.get("recording_n_windows") != 160:
        raise ValueError("final EV-UAV release must explicitly declare 160 recording windows")
    jobs = []
    for ordinal, seed in [(900, 37), (910, 38)]:
        name = "%d_bst_final_%s_v5s%d_s%d" % (ordinal, release, seed, seed)
        if list((root/"run/queue").rglob(name+"*.sh")):
            raise FileExistsError("job name already used: " + name)
        output = checked_child(root, root/"log"/("bst_final_%s_v5s%d_s%d" % (release, seed, seed)))
        if output.exists():
            raise FileExistsError("output already exists: " + str(output))
        jobs.append({"name": name, "seed": seed, "backbone": str(Path(backbones[seed]).resolve()),
                     "backbone_sha256": hashes[seed], "out": str(output),
                     "priors": str(release_dir/"priors"/("v5s%d" % seed)),
                     "config": str(release_dir/("config_s%d.json" % seed)), "physical_gpu": 1})
    return {"schema": 1, "release": release, "release_dir": str(release_dir), "root": str(root),
            "data_root": str(data_root), "base_config": cfg, "jobs": jobs,
            "python_executable": str(Path(sys.executable).resolve()), "python_version": sys.version}


def publish(spec):
    root, release = Path(spec["root"]), Path(spec["release_dir"])
    release.mkdir(parents=True, exist_ok=False)
    snapshot = release/"source"
    hashes = {}
    for source in source_files(root):
        if source.is_symlink():
            raise ValueError("refusing symlink source file " + str(source))
        relative = source.relative_to(root)
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source), str(target))
        hashes[str(relative).replace("\\", "/")] = digest(target)
    spec["source_sha256"] = hashes
    for job in spec["jobs"]:
        cfg = json.loads(json.dumps(spec["base_config"]))
        cfg["train"]["seed"] = job["seed"]
        Path(job["config"]).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (release/"release.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    pending = root/"run/queue/pool/pending"
    pending.mkdir(parents=True, exist_ok=True)
    staged = []
    for job in spec["jobs"]:
        fields = {"DATA_ROOT": spec["data_root"], "BACKBONE": job["backbone"], "OUT": job["out"],
                  "Z0_ROOT": job["priors"], "CONFIG": job["config"], "RELEASE_MANIFEST": str(release/"release.json"),
                  "SEED": str(job["seed"]), "BACKBONE_SHA256": job["backbone_sha256"],
                  "BST_PYTHON": spec["python_executable"]}
        text = "#!/usr/bin/env bash\nset -euo pipefail\n"
        text += "\n".join("export %s=%s" % (k, shlex.quote(v)) for k, v in fields.items()) + "\n"
        text += "exec bash " + shlex.quote(str(snapshot/"tools/bst/run_final.sh")) + "\n"
        temporary = pending/("."+job["name"]+".tmp")
        with open(temporary, "x", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        staged.append((temporary, pending/(job["name"]+"@gpu1.sh")))
    for temporary, destination in staged:
        # Atomic no-clobber publication; dispatcher never sees a partial script.
        os.link(str(temporary), str(destination))
        temporary.unlink()
        print("QUEUED", destination)
    print("RELEASE", release/"release.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    q = sub.add_parser("enqueue")
    q.add_argument("--release", required=True)
    q.add_argument("--data-root", default=str(ROOT.parent/"datasets/EV-UAV-dataset"))
    q.add_argument("--config", default=str(ROOT/"configs/bst_shared.json"))
    for seed in (37, 38):
        q.add_argument("--backbone-%d" % seed, default=str(ROOT.parent/("EV-UAV-v8-rcl41/log/stream_tif_commpeak_msg_seed%d/best_val_iou_seed%d.pt" % (seed, seed))))
    q.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("pool-preflight")
    p.add_argument("--recover-reboot", action="store_true")
    args = ap.parse_args()
    if args.command == "pool-preflight":
        return pool_preflight(ROOT, args.recover_reboot)
    spec = plan(ROOT, args.release, args.data_root, {37: args.backbone_37, 38: args.backbone_38}, args.config)
    if args.dry_run:
        print(json.dumps(spec, indent=2))
    else:
        publish(spec)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print("ERROR:", error, file=sys.stderr)
        sys.exit(1)
