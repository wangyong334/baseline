"""Train one isolated V14 overlap model, then evaluate both held-out splits.

The existing V13 dispatcher owns the GPU. This runner neither starts processes
in that dispatcher nor overwrites an earlier experiment. Python 3.8 compatible.
"""
import argparse
import copy
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE = "overlap_20261112_r1"
VARIANTS = ("conditional", "mean", "concat")


def _load_common():
    spec = importlib.util.spec_from_file_location(
        "v14_overlap_runner_common", str(ROOT / "tools/v14/run_experiment.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load_common()
read_json = common.read_json
write_json = common.write_json
sha256 = common.sha256
load_yaml = common.load_yaml
flatten = common.flatten


def validate_configs(source_root=ROOT):
    """Keep the V5 backbone/optimizer/budget contract visible and enforce it."""
    source_root = Path(source_root)
    base = load_yaml(source_root / "configs/evisseg_stream_tif_comm_peak.yaml")
    configs = {}
    allowed = {"DATA": {"root", "stats_path", "window_ms", "n_windows"},
               "NET": set(), "TRAIN": {"save_root", "seed"},
               "EVAL": {"segments"}, "CALIB": set()}
    for variant in VARIANTS:
        path = source_root / ("configs/v14/overlap_%s.yaml" % variant)
        if not path.is_file():
            if variant == "conditional":
                raise ValueError("required configuration missing: %s" % path)
            continue
        raw = load_yaml(path)
        flat = flatten(raw)
        for section, values in base.items():
            for key, expected in values.items():
                if key not in allowed[section] and raw.get(section, {}).get(key) != expected:
                    raise ValueError("%s changes V5 %s.%s" % (variant, section, key))
        contract = {"window_ms": 100, "stride_ms": 50, "total_ms": 8000,
                    "n_windows": 159, "state_mode": "carry",
                    "state_layout": "single_rolling", "fusion_mode": variant,
                    "fusion_hidden": 8, "trainer": "streams", "streams": 4,
                    "tbptt_k": 16, "stream_layout": "greedy", "eval_execution": "step"}
        for key, expected in contract.items():
            if flat.get(key) != expected:
                raise ValueError("%s requires %s=%r" % (variant, key, expected))
        if int(flat.get("seed", -1)) < 0 or int(flat.get("epochs", 0)) <= 0:
            raise ValueError("seed/epochs invalid")
        configs[variant] = raw
    return configs


def runtime_config(variant, seed, release, data_root, project_root, source_root=ROOT):
    common.checked_name(release)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    configs = validate_configs(source_root)
    if variant not in configs:
        raise ValueError("configuration unavailable: " + variant)
    raw = copy.deepcopy(configs[variant])
    out = Path(project_root).resolve() / "log/v14_overlap" / release / ("%s_s%d" % (variant, seed))
    raw["DATA"]["root"] = str(Path(data_root).resolve())
    # Statistics belong to this 100/50 representation, never the old 50 ms run.
    raw["DATA"]["stats_path"] = str(out / "train_stats.json")
    raw["TRAIN"]["seed"] = seed
    raw["TRAIN"]["save_root"] = str(out)
    return raw


def source_hashes(source_root):
    source_root = Path(source_root)
    paths = {source_root / "train_stream_v1.py",
             source_root / "configs/evisseg_stream_tif_comm_peak.yaml"}
    for directory in ("model", "dataset", "utils", "tools/v14"):
        paths.update((source_root / directory).rglob("*.py"))
    paths.update((source_root / "configs/v14").glob("*.yaml"))
    return {p.relative_to(source_root).as_posix(): sha256(p) for p in sorted(paths)}


def execute(command, source_root):
    import json
    print("RUN " + json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=str(source_root), check=True)


def validate_evaluation(path, variant):
    payload = read_json(path)
    if payload.get("timing", {}).get("execution") != "step":
        raise ValueError("formal evaluation must use step execution: %s" % path)
    expected = {"first", "second", "mean", variant}
    if variant == "conditional":
        expected.add("calibration_only")
    readouts = payload.get("readouts", {})
    if not expected.issubset(readouts):
        raise ValueError("missing readout evaluations in %s: %s" %
                         (path, sorted(expected.difference(readouts))))
    groups = [("main", payload.get("metrics", {}))]
    for name in expected:
        value = readouts[name]
        groups.append((name, value.get("metrics", value)))
    for group, metrics in groups:
        for key in ("iou", "acc", "pd", "fa"):
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("%s %s has invalid %s" % (path, group, key))
            if value < 0 or (key != "fa" and value > 1):
                raise ValueError("%s %s has out-of-range %s" % (path, group, key))
    return payload


def validate_profile(path, variant, checkpoint_sha256):
    payload = read_json(path)
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("profile checkpoint identity does not match trained weights")
    if payload.get("readout_policy") != variant or payload.get("split") != "test":
        raise ValueError("profile must measure the trained main readout on test inputs")
    elapsed = payload.get("total_wall_seconds")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("profile has invalid wall time")
    joules = payload.get("energy", {}).get("board_joules")
    if joules is not None and (not isinstance(joules, (int, float)) or not math.isfinite(joules) or joules < 0):
        raise ValueError("profile has invalid optional board energy")
    return payload


def run(args):
    source = ROOT.resolve()
    project = Path(args.project_root).resolve()
    raw = runtime_config(args.variant, args.seed, args.release,
                         args.data_root, project, source)
    out = Path(raw["TRAIN"]["save_root"])
    meta = project / "run/v14_overlap" / args.release / "jobs" / ("%s_s%d" % (args.variant, args.seed))
    if out.exists() or meta.exists():
        raise ValueError("experiment already exists; refusing overwrite: %s" % out)
    inventory = common.dataset_inventory(args.data_root)
    meta.mkdir(parents=True, exist_ok=False)
    config = meta / "config.yaml"
    import yaml
    with config.open("x", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False, allow_unicode=True)
    identity = {"format_version": 1, "release": args.release, "variant": args.variant,
                "seed": args.seed, "config": raw, "config_sha256": sha256(config),
                "dataset": inventory, "source": str(source),
                "source_sha256": source_hashes(source), "out": str(out),
                "python": sys.executable, "python_version": sys.version,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "trained_from_scratch": True, "resume": False}
    manifest = meta / "manifest.json"
    started = time.time()
    state = dict(identity, status="training", started_at_unix=started)
    write_json(manifest, state)
    checkpoint = out / ("best_val_iou_seed%d.pt" % args.seed)
    base_command = [sys.executable, "-B", "-u", str(source / "tools/v14/train_overlap.py"),
                    "--config", str(config), "--data-root", str(Path(args.data_root).resolve()),
                    "--out", str(out), "--device", args.device]
    try:
        execute(base_command + ["--mode", "train"], source)
        summary = read_json(out / "summary.json")
        completed = int(summary.get("epochs_completed", -1))
        if completed != int(raw["TRAIN"]["epochs"]):
            raise ValueError("training ended before all configured epochs: %d" % completed)
        if summary.get("complete") is not True or summary.get("probe_only") is not False:
            raise ValueError("formal run requires a complete, non-probe training summary")
        if not checkpoint.is_file() or not checkpoint.stat().st_size:
            raise ValueError("trained best checkpoint missing: %s" % checkpoint)
        selected_hash = sha256(checkpoint)
        results = {}
        for split in ("val", "test"):
            state.update(status="evaluating_" + split)
            write_json(manifest, state)
            execute(base_command + ["--mode", "eval", "--checkpoint", str(checkpoint),
                                    "--split", split], source)
            results[split] = validate_evaluation(out / ("eval_%s.json" % split), args.variant)
            if results[split].get("checkpoint_sha256") != selected_hash:
                raise ValueError("evaluation checkpoint identity mismatch: " + split)
        state.update(status="profiling")
        write_json(manifest, state)
        execute([sys.executable, "-B", "-u", str(source / "tools/v14/profile_overlap.py"),
                 "--config", str(config), "--checkpoint", str(checkpoint),
                 "--data-root", str(Path(args.data_root).resolve()), "--out", str(out),
                 "--device", args.device, "--split", "test", "--records", "3", "--repeats", "2"], source)
        profile = validate_profile(out / "profile.json", args.variant, selected_hash)
        if sha256(checkpoint) != selected_hash:
            raise ValueError("selected checkpoint changed during evaluation")
        if sha256(config) != identity["config_sha256"] or source_hashes(source) != identity["source_sha256"]:
            raise ValueError("source or runtime configuration changed during training")
        result = dict(identity, status="complete", epochs_completed=completed,
                      checkpoint=str(checkpoint), checkpoint_sha256=selected_hash,
                      elapsed_seconds=time.time() - started, evaluations=results, profile=profile,
                      note="Readout ablations share the selected weights; mean/concat training ablations are separate jobs.")
        write_json(out / "comparison_result.json", result)
        state.update(status="complete", finished_at_unix=time.time(),
                     checkpoint_sha256=result["checkpoint_sha256"])
        write_json(manifest, state)
        print("V14_OVERLAP_FINISHED " + str(out), flush=True)
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
    parser.add_argument("--variant", choices=VARIANTS, default="conditional")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, "ERROR: %s\n" % error)


if __name__ == "__main__":
    main()
