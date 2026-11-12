"""Run an isolated V15 training job and complete held-out evaluation.

This runner never manages the shared GPU pool and never resumes/overwrites an
existing experiment. It runs from the immutable release snapshot (Python 3.8).
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE = "feature_20261112_r3"
DEFAULT_DATA = "/media/stephen/nvme0n1/wy_data/datasets/EV-UAV-dataset"
MODES = ("observation", "deep", "causal")
SPLIT_COUNTS = {"train": 99, "val": 24, "test": 24}
RUNTIME_FILES = (
    "train_stream_v1.py", "tools/stream_train_stats.py",
    "tools/v15/train.py", "tools/v15/run_experiment.py", "tools/v15/queue_jobs.py",
    "tools/v15/package_results.py", "tools/v15/profile.py", "tools/v15/runtime_check.py",
    "model/evspsegnet_stream.py", "model/lif2d_stream.py", "model/observation_revision.py",
    "dataset/ev_uav_stream.py", "dataset/stream_source.py", "dataset/stream_windows.py",
    "utils/comm_budget.py", "utils/eval.py", "utils/stream_common.py", "utils/stream_metrics.py",
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checked_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value):
        raise ValueError("invalid release name")
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def load_yaml(path):
    import yaml
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping: %s" % path)
    return value


def flatten(raw):
    result = {}
    for section in raw.values():
        if not isinstance(section, dict):
            raise ValueError("configuration sections must be mappings")
        for key, value in section.items():
            if key in result:
                raise ValueError("duplicate configuration key: " + key)
            result[key] = value
    return result


def validate_configs(root=ROOT):
    """Freeze the published V5 contract, leaving revision details to the model."""
    configs = {}
    base = load_yaml(Path(root) / "configs/evisseg_stream_tif_comm_peak.yaml")
    for mode in MODES:
        raw = load_yaml(Path(root) / ("configs/v15/%s.yaml" % mode))
        flat = flatten(raw)
        contract = {"window_ms": 50, "n_windows": 160, "time_bins": 5,
                    "state_mode": "carry", "neuron": "tif", "trainer": "streams",
                    "streams": 4, "tbptt_k": 16, "comm_budget_mode": "peak",
                    "revision_source": mode}
        for key, expected in contract.items():
            if flat.get(key) != expected:
                raise ValueError("%s requires %s=%r" % (mode, key, expected))
        for section, keys in {"NET": ("channels", "tif", "readout_hidden"),
                              "TRAIN": ("comm_budget_messages", "comm_peak_alpha", "comm_dual_lr")}.items():
            for key in keys:
                if raw.get(section, {}).get(key) != base[section][key]:
                    raise ValueError("%s must preserve V5 %s.%s" % (mode, section, key))
        if int(flat.get("epochs", 0)) <= 0 or int(flat.get("seed", -1)) < 0:
            raise ValueError("invalid training epochs or seed")
        configs[mode] = raw
    return configs


def dataset_inventory(data_root, strict=True):
    inventory = {}
    root = Path(data_root).resolve()
    for split, expected in SPLIT_COUNTS.items():
        paths = sorted((root / split).glob("*.npz"))
        if not paths or (strict and len(paths) != expected):
            raise ValueError("%s needs %d records; found %d" % (split, expected, len(paths)))
        rows = []
        for path in paths:
            stat = path.stat()
            if not stat.st_size:
                raise ValueError("empty data file: %s" % path)
            rows.append({"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        inventory[split] = rows
    return inventory


def source_files(root):
    """Freeze runtime packages, the actual training entry points and configs.

    The dependency list is explicit: an older SFTP upload may leave obsolete
    files on the server, and those must not enter the frozen release. Import
    checks in the frozen snapshot verify this list before publication.
    """
    root = Path(root).resolve()
    required = list(RUNTIME_FILES) + ["configs/evisseg_stream_tif_comm_peak.yaml"]
    required += ["configs/v15/%s.yaml" % mode for mode in MODES]
    paths = {root / relative for relative in required}
    for relative in ("model/__init__.py", "dataset/__init__.py", "utils/__init__.py",
                     "tools/__init__.py", "tools/v15/__init__.py"):
        if (root / relative).is_file():
            paths.add(root / relative)
    result = {}
    for path in sorted(paths):
        if not path.is_file() or (path.relative_to(root).as_posix() in required and not path.stat().st_size):
            raise ValueError("missing source dependency: %s" % path)
        if path.is_symlink():
            raise ValueError("source symlinks are not supported: %s" % path)
        path.resolve().relative_to(root)
        result[path.relative_to(root).as_posix()] = digest(path)
    return result


def runtime_config(mode, seed, release, data_root, project, root=ROOT):
    checked_name(release)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    raw = copy.deepcopy(validate_configs(root)[mode])
    out = Path(project).resolve() / "log/v15" / release / ("%s_s%d" % (mode, seed))
    raw["DATA"]["root"] = str(Path(data_root).resolve())
    # Fresh training-only statistics. No cache of unknown origin is reused.
    raw["DATA"]["stats_path"] = str(out / "train_stats.json")
    raw["TRAIN"]["seed"] = seed
    raw["TRAIN"]["save_root"] = str(out)
    return raw


def execute(command, root):
    print("RUN " + json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=str(root), check=True)


def hardware_inventory():
    result = {"platform": platform.platform(), "processor": platform.processor(),
              "python": sys.version, "executable": sys.executable,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
              "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
              "mkl_num_threads": os.environ.get("MKL_NUM_THREADS")}
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                                "--format=csv,noheader"], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True, timeout=10)
        result["nvidia_smi"] = query.stdout.strip() if query.returncode == 0 else query.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        result["nvidia_smi_unavailable"] = str(error)
    return result


def validate_evaluation(path, checkpoint_hash, split):
    payload = read_json(path)
    if payload.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("evaluation used a different checkpoint: %s" % path)
    if payload.get("split") != split or payload.get("timing", {}).get("execution") != "step":
        raise ValueError("evaluation must be full %s in step mode" % split)
    if payload.get("complete") is not True:
        raise ValueError("evaluation is incomplete: %s" % path)
    for key in ("iou", "acc", "pd", "fa"):
        value = payload.get("metrics", {}).get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("invalid %s in %s" % (key, path))
        if value < 0 or value > 1:
            raise ValueError("metric out of range: " + key)
    if not isinstance(payload.get("diagnostics"), dict) or not payload["diagnostics"]:
        raise ValueError("evaluation diagnostics are missing: %s" % path)
    return payload


def validate_profile(path, checkpoint_hash):
    payload = read_json(path)
    if payload.get("checkpoint_sha256") != checkpoint_hash or payload.get("split") != "test":
        raise ValueError("profile must use the selected checkpoint on test inputs")
    variants = payload.get("variants", {})
    if not {"v15_final", "v5_same_backbone"}.issubset(variants):
        raise ValueError("profile must measure both the final model and the backbone readout")
    for name in ("v15_final", "v5_same_backbone"):
        row = variants[name]
        speed = row.get("events_per_second")
        if not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0 or not row.get("records"):
            raise ValueError("profile timing is incomplete: " + name)
    return payload


def run(args):
    source = ROOT.resolve()
    project = Path(args.project_root).resolve()
    raw = runtime_config(args.variant, args.seed, args.release, args.data_root, project, source)
    out = Path(raw["TRAIN"]["save_root"])
    metadata = project / "run/v15" / args.release / "jobs" / ("%s_s%d" % (args.variant, args.seed))
    if out.exists() or metadata.exists():
        raise ValueError("experiment already exists; use a new release: %s" % out)
    inventory = dataset_inventory(args.data_root)
    hashes = source_files(source)
    metadata.mkdir(parents=True, exist_ok=False)
    config = metadata / "config.yaml"
    import yaml
    with config.open("x", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False, allow_unicode=True)
    identity = {"format_version": 1, "release": args.release, "variant": args.variant,
                "seed": args.seed, "config": raw, "config_sha256": digest(config),
                "dataset_root": str(Path(args.data_root).resolve()), "dataset": inventory,
                "dataset_identity_note": "File names/sizes/mtimes, not content hashes.",
                "source": str(source), "source_sha256": hashes, "out": str(out),
                "hardware": hardware_inventory(), "trained_from_scratch": True}
    manifest = metadata / "manifest.json"
    started = time.time()
    state = dict(identity, status="training", started_at_unix=started)
    write_json(manifest, state)
    checkpoint = out / ("best_val_iou_seed%d.pt" % args.seed)
    command = [sys.executable, "-B", "-u", str(source / "tools/v15/train.py"),
               "--config", str(config), "--data-root", str(Path(args.data_root).resolve()),
               "--out", str(out), "--device", args.device]
    try:
        execute(command + ["--mode", "train"], source)
        summary = read_json(out / "summary.json")
        if summary.get("complete") is not True or summary.get("probe_only") is not False:
            raise ValueError("formal run requires complete non-probe training")
        if summary.get("epochs_completed") != int(raw["TRAIN"]["epochs"]):
            raise ValueError("training did not complete all configured epochs")
        for relative in (checkpoint.name, "last.pt", "calibration.json", "train_stats.json", "metrics.jsonl"):
            if not (out / relative).is_file() or not (out / relative).stat().st_size:
                raise ValueError("missing training artifact: " + relative)
        write_json(out / "train_stats_provenance.json", {
            "statistics_split": "train", "generated_by": "tools/v15/train.py",
            "reused_external_statistics": False, "dataset_root": identity["dataset_root"],
            "training_files": inventory["train"], "config_sha256": identity["config_sha256"],
            "statistics_sha256": digest(out / "train_stats.json")})
        selected_hash = digest(checkpoint)
        results = {}
        for split in ("val", "test"):
            state.update(status="evaluating_" + split)
            write_json(manifest, state)
            execute(command + ["--mode", "eval", "--split", split, "--checkpoint", str(checkpoint), "--step"], source)
            results[split] = validate_evaluation(out / ("eval_%s.json" % split), selected_hash, split)
        state.update(status="profiling")
        write_json(manifest, state)
        execute([sys.executable, "-B", "-u", str(source / "tools/v15/profile.py"),
                 "--config", str(config), "--checkpoint", str(checkpoint),
                 "--data-root", str(Path(args.data_root).resolve()), "--out", str(out),
                 "--device", args.device, "--split", "test", "--records", "3", "--repeats", "2"], source)
        profile = validate_profile(out / "profile.json", selected_hash)
        if digest(checkpoint) != selected_hash or digest(config) != identity["config_sha256"]:
            raise ValueError("checkpoint or runtime configuration changed during execution")
        if source_files(source) != hashes or dataset_inventory(args.data_root) != inventory:
            raise ValueError("source or dataset inventory changed during execution")
        result = dict(identity, status="complete", complete=True,
                      epochs_completed=summary["epochs_completed"], checkpoint_sha256=selected_hash,
                      evaluations=results, elapsed_seconds=time.time() - started,
                      profile=profile, profiling_status="complete")
        write_json(out / "comparison_result.json", result)
        state.update(status="complete", finished_at_unix=time.time(), checkpoint_sha256=selected_hash)
        write_json(manifest, state)
        print("V15_FEATURE_FINISHED " + str(out), flush=True)
        return result
    except Exception as error:
        state.update(status="failed", error=str(error), finished_at_unix=time.time())
        write_json(manifest, state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--variant", choices=MODES, default="observation")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, "ERROR: %s\n" % error)


if __name__ == "__main__":
    main()
