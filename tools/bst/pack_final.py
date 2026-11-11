"""Pack one immutable final release and both result directories for download.

Default includes metrics, logs, event outputs and source snapshot, not model
weights or full BIR prior caches. Incomplete/missing runs are recorded explicitly.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tarfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.bst.queue_final import ROOT, checked_child, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out")
    ap.add_argument("--include-weights", action="store_true")
    args = ap.parse_args()
    release = checked_child(ROOT, ROOT/"run/bst_final"/args.release)
    manifest = release/"release.json"
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    output = Path(args.out).resolve() if args.out else ROOT/"Res"/("bst_final_"+spec["release"]+".tgz")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("preserving existing package: " + str(output))
    temporary = output.with_name("."+output.name+"."+uuid.uuid4().hex[:8]+".tmp")
    files = {manifest}
    files.update(release.glob("config_s*.json"))
    files.update(path for path in (release/"source").rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    allowed = {".json", ".jsonl", ".log", ".md", ".csv", ".tsv", ".npz", ".npy", ".png", ".svg", ".txt"}
    if args.include_weights:
        allowed.add(".pt")
    status = []
    for job in spec["jobs"]:
        out = checked_child(ROOT, job["out"])
        status.append({"seed": job["seed"], "out": str(out), "exists": out.is_dir(),
                       "complete": (out/"COMPLETE").is_file(), "failed": (out/"FAILED").is_file()})
        if out.is_dir():
            files.update(path for path in out.rglob("*") if path.is_file() and
                         (path.suffix in allowed or path.name in {"COMPLETE", "FAILED"}))
        log = ROOT/"run/queue/gpu1/logs"/(job["name"]+".log")
        if log.is_file():
            files.add(log)
    import io
    status_bytes = json.dumps({"runs": status, "weights_included": args.include_weights,
                               "prior_caches_included": False}, indent=2).encode()
    with tarfile.open(str(temporary), "x:gz") as tar:
        for path in sorted(files):
            if path.is_symlink():
                raise ValueError("refusing symlink in results: " + str(path))
            path = checked_child(ROOT, path)
            tar.add(str(path), arcname=str(path.relative_to(ROOT)), recursive=False)
        item = tarfile.TarInfo("PACK_STATUS.json")
        item.size = len(status_bytes)
        tar.addfile(item, io.BytesIO(status_bytes))
    os.link(str(temporary), str(output))
    temporary.unlink()
    print(json.dumps(status, indent=2))
    print("PACKED", output, "bytes", output.stat().st_size)
    print("SHA256", digest(output))


if __name__ == "__main__":
    main()
