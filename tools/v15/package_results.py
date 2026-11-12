"""Export a stable V15 result bundle; never includes data, caches or credentials."""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.v15.queue_jobs import release_plan, verify_source
from tools.v15.run_experiment import (DEFAULT_RELEASE, digest, read_json,
                                     validate_evaluation, validate_profile)


def pack(project_root, release, allow_partial=False):
    project = Path(project_root).resolve()
    manifest, plan = release_plan(project, release)
    verify_source(manifest)
    files = {}
    job_states = []
    all_complete = True
    for job in plan["jobs"]:
        out = Path(job["out"]).resolve()
        expected = project / "log/v15" / release / ("%s_s%d" % (job["variant"], job["seed"]))
        if out != expected:
            raise ValueError("job output is outside the expected release directory")
        metadata = manifest.parent / "jobs" / ("%s_s%d" % (job["variant"], job["seed"])) / "manifest.json"
        result_path = out / "comparison_result.json"
        result = read_json(result_path) if result_path.is_file() else {}
        stage = read_json(metadata).get("status") if metadata.is_file() else "not_started"
        complete = result.get("complete") is True and result.get("status") == "complete" and stage == "complete"
        all_complete = all_complete and complete
        job_states.append({"name": job["name"], "variant": job["variant"], "seed": job["seed"],
                           "stage": stage, "complete": complete})
        if not complete and not allow_partial:
            raise ValueError("%s is not complete (stage=%s); inspect status or explicitly use --allow-partial" %
                             (job["name"], stage))
        if complete:
            if result.get("variant") != job["variant"] or result.get("seed") != job["seed"] or result.get("release") != release:
                raise ValueError("completed result has wrong job identity")
            checkpoint = out / ("best_val_iou_seed%d.pt" % job["seed"])
            if not checkpoint.is_file() or digest(checkpoint) != result.get("checkpoint_sha256"):
                raise ValueError("selected checkpoint is missing or changed: %s" % checkpoint)
            for split in ("val", "test"):
                validate_evaluation(out / ("eval_%s.json" % split), result["checkpoint_sha256"], split)
            validate_profile(out / "profile.json", result["checkpoint_sha256"])
        if out.is_dir():
            # Only top-level known artifact classes: no cached feature/data tree.
            for path in out.iterdir():
                if path.is_file() and path.suffix.lower() in (".json", ".jsonl", ".yaml", ".log", ".txt", ".pt", ".csv", ".tsv"):
                    files[path.relative_to(project).as_posix()] = path
        log = Path(plan["scheduler_root"]) / "run/queue/gpu1/logs" / (job["name"] + ".log")
        if log.is_file():
            files["scheduler_logs/" + log.name] = log
        # Exact submitted shell is useful even after the scheduler moved it.
        for location, suffix in (("pool/pending", "@gpu1.sh"), ("gpu1/pending", ".sh"),
                                 ("gpu1/running", ".sh"), ("gpu1/done", ".sh"), ("gpu1/failed", ".sh")):
            script = Path(plan["scheduler_root"]) / "run/queue" / location / (job["name"] + suffix)
            if script.is_file():
                files["scheduler_jobs/" + script.name] = script
    files[manifest.relative_to(project).as_posix()] = manifest
    for relative in plan["files"]:
        path = Path(plan["source"]) / relative
        files[path.relative_to(project).as_posix()] = path
    for path in (manifest.parent / "jobs").rglob("*"):
        if path.is_file() and path.suffix.lower() in (".json", ".yaml"):
            files[path.relative_to(project).as_posix()] = path
    before = {}
    hashes = {}
    for name, path in files.items():
        if path.is_symlink() or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe archive path: " + name)
        stat = path.stat()
        before[name] = (stat.st_size, stat.st_mtime_ns)
        hashes[name] = digest(path)
    bundle = {"format_version": 1, "release": release, "complete": all_complete,
              "partial": not all_complete, "created_at_unix": time.time(), "jobs": job_states,
              "files_sha256": hashes,
              "excluded": ["dataset", "feature caches", "prediction arrays", "SFTP credentials"]}
    result_dir = project / "Res"
    result_dir.mkdir(exist_ok=True)
    target = result_dir / ("v15_%s_%s_%s%s.tgz" %
                           (release, time.strftime("%Y%m%d_%H%M%S"), uuid.uuid4().hex[:6],
                            "_partial" if not all_complete else ""))
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("xb") as handle:
            with tarfile.open(fileobj=handle, mode="w:gz") as archive:
                payload = json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8")
                info = tarfile.TarInfo("bundle_manifest.json")
                info.size = len(payload)
                info.mtime = int(time.time())
                archive.addfile(info, io.BytesIO(payload))
                for name, path in sorted(files.items()):
                    archive.add(str(path), arcname=name, recursive=False)
        for name, path in files.items():
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != before[name] or digest(path) != hashes[name]:
                raise ValueError("artifacts changed during packaging; retry after writes finish")
        os.link(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()
    print("BUNDLE_COMPLETE: " + str(all_complete))
    print("DOWNLOAD: " + str(target))
    print("SIZE_MiB: %.2f" % (target.stat().st_size / 1024.0 ** 2))
    print("SHA256: " + digest(target))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        pack(args.project_root, args.release, args.allow_partial)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, "ERROR: %s\n" % error)


if __name__ == "__main__":
    main()
