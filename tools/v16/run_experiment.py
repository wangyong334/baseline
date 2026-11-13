"""Train one isolated V16 context model, then evaluate both held-out splits.

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
DEFAULT_RELEASE = "context_20261113_r2"
VARIANTS = ("full", "no_revision", "no_context")


def _load_common():
    spec = importlib.util.spec_from_file_location(
        "v16_context_runner_common", str(ROOT / "tools/v14/run_experiment.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load_common()
read_json = common.read_json
write_json = common.write_json
sha256 = common.sha256
load_yaml = common.load_yaml
flatten = common.flatten


def _validate_legacy_configs(source_root=ROOT):
    """Keep V14/V5 settings fixed; only the two V16 modules may be ablated."""
    source_root = Path(source_root)
    base = load_yaml(source_root / "configs/v14/overlap_conditional.yaml")
    configs = {}
    allowed = {"DATA": {"stats_path"}, "NET": set(),
               "TRAIN": {"save_root"}, "EVAL": set(), "CALIB": set()}
    reference = None
    for variant in VARIANTS:
        path = source_root / ("configs/v16/%s.yaml" % variant)
        if not path.is_file():
            raise ValueError("required configuration missing: %s" % path)
        raw = load_yaml(path)
        flat = flatten(raw)
        if set(raw) != set(allowed) | {"CONTEXT"}:
            raise ValueError("%s requires DATA/NET/CALIB/TRAIN/EVAL/CONTEXT only" % variant)
        for section, values in base.items():
            if section == "FUSION":
                continue
            if set(raw.get(section, {})) != set(values):
                raise ValueError("%s changes V14 %s keys" % (variant, section))
            for key, expected in values.items():
                if key not in allowed[section] and raw.get(section, {}).get(key) != expected:
                    raise ValueError("%s changes V14 %s.%s" % (variant, section, key))
        context = {"revision_variant": variant, "revision_dim": 12,
                   "revision_hidden": 24, "revision_heads": 3,
                   "revision_radius": 2, "revision_downsample": 4}
        if raw.get("CONTEXT") != context:
            raise ValueError("%s requires the fixed V16 CONTEXT contract" % variant)
        contract = {"window_ms": 100, "stride_ms": 50, "total_ms": 8000,
                    "n_windows": 159, "state_mode": "carry",
                    "state_layout": "single_rolling", "trainer": "streams", "streams": 4,
                    "tbptt_k": 16, "stream_layout": "greedy", "eval_execution": "step",
                    "execution": "layer", "epochs": 50, "seed": 37,
                    "comm_budget_messages": 157700.0, "comm_budget_mode": "peak"}
        for key, expected in contract.items():
            if flat.get(key) != expected:
                raise ValueError("%s requires %s=%r" % (variant, key, expected))
        normalized = copy.deepcopy(raw)
        normalized["CONTEXT"].pop("revision_variant")
        normalized["DATA"].pop("stats_path")
        normalized["TRAIN"].pop("save_root")
        if reference is not None and normalized != reference:
            raise ValueError("V16 configs may differ only in variant/output/statistics paths")
        reference = normalized
        configs[variant] = raw
    return configs


def validate_configs(source_root=ROOT, config_dir="configs/v16"):
    """Validate matched ablations without silently changing the V5 protocol."""
    source_root = Path(source_root).resolve()
    relative = Path(config_dir)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("config-dir must be a project-relative directory")
    if relative.as_posix() == "configs/v16":
        return _validate_legacy_configs(source_root)
    base = load_yaml(source_root / "configs/v14/overlap_conditional.yaml")
    configs, reference = {}, None
    for variant in VARIANTS:
        path = source_root / relative / (variant + ".yaml")
        if not path.is_file():
            raise ValueError("required configuration missing: %s" % path)
        raw = load_yaml(path)
        flat = flatten(raw)
        if set(raw) != {"DATA", "NET", "TRAIN", "CALIB", "EVAL", "CONTEXT"}:
            raise ValueError("paired V16 requires DATA/NET/TRAIN/CALIB/EVAL/CONTEXT")
        for section in ("NET", "CALIB", "EVAL"):
            if raw[section] != base[section]:
                raise ValueError("paired V16 changes the V5 %s protocol" % section)
        for key, value in base["DATA"].items():
            if key not in ("root", "stats_path") and raw["DATA"].get(key) != value:
                raise ValueError("paired V16 changes DATA." + key)
        if flat.get("revision_architecture") not in ("evidence", "anchored", "covariance"):
            raise ValueError("paired V16 requires evidence, anchored or covariance revision_architecture")
        required = {"revision_variant": variant,
                    "state_mode": "carry", "state_layout": "single_rolling",
                    "trainer": "streams", "streams": 4, "tbptt_k": 16,
                    "stream_layout": "greedy", "eval_execution": "step",
                    "execution": "layer", "seed": 37,
                    "comm_budget_messages": 157700.0, "comm_budget_mode": "peak"}
        for key, value in required.items():
            if flat.get(key) != value:
                raise ValueError("paired V16 requires %s=%r" % (key, value))
        if int(flat.get("epochs", 0)) < 1:
            raise ValueError("paired V16 epochs must be positive")
        if flat.get("revision_architecture") == "covariance":
            validate_covariance_protocol(flat)
        normalized = copy.deepcopy(raw)
        normalized["CONTEXT"].pop("revision_variant")
        normalized["DATA"].pop("stats_path")
        normalized["TRAIN"].pop("save_root")
        if reference is not None and normalized != reference:
            raise ValueError("paired V16 ablations may differ only in variant/output/statistics paths")
        reference = normalized
        configs[variant] = raw
    return configs


def validate_covariance_protocol(flat):
    """Keep train-fitted statistics and the fixed operating point explicit."""
    required = {"freeze_backbone": True, "require_initialization": True,
                "validate_initial": True, "fusion_loss": "rate_constrained",
                "threshold": .9, "recall_tolerance": 0., "pd_tolerance": 0.,
                "selection_rule": "four_metric_noninferiority",
                "constraint": "smooth_fn_reference"}
    for key, value in required.items():
        if flat.get(key) != value:
            raise ValueError("covariance V16 requires %s=%r" % (key, value))
    channels = flat.get("covariance_channels")
    if isinstance(channels, bool) or not isinstance(channels, int) or not 1 <= channels <= 60:
        raise ValueError("covariance_channels must be an integer between 1 and 60")
    for key, low, high, closed_low in (
            ("covariance_shrinkage", 0., 1., False),
            ("covariance_regularizer", 0., float("inf"), True),
            ("recall_tolerance", 0., 1., True),
            ("recall_dual_lr", 0., float("inf"), False),
            ("recall_dual_init", 0., float("inf"), True)):
        value = flat.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value > high or
                (value < low if closed_low else value <= low)):
            raise ValueError("invalid covariance training setting: " + key)
    minimum, maximum = int(flat.get("min_epochs", 0)), int(flat.get("epochs", 0))
    if not 1 <= minimum <= maximum or int(flat.get("early_stop_patience", -1)) < 1:
        raise ValueError("covariance V16 requires a valid minimum/maximum/patience schedule")


def runtime_config(variant, seed, release, data_root, project_root, source_root=ROOT,
                   config_dir="configs/v16", init_checkpoint=None):
    common.checked_name(release)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    configs = validate_configs(source_root, config_dir)
    if variant not in configs:
        raise ValueError("configuration unavailable: " + variant)
    raw = copy.deepcopy(configs[variant])
    out = Path(project_root).resolve() / "log/v16_context" / release / ("%s_s%d" % (variant, seed))
    raw["DATA"]["root"] = str(Path(data_root).resolve())
    # Statistics belong to this 100/50 representation, never the old 50 ms run.
    raw["DATA"]["stats_path"] = str(out / "train_stats.json")
    raw["TRAIN"]["seed"] = seed
    raw["TRAIN"]["save_root"] = str(out)
    if init_checkpoint:
        raw["TRAIN"]["init_checkpoint"] = str(Path(init_checkpoint).resolve())
    return raw


def validate_training_summary(summary, raw):
    completed = int(summary.get("epochs_completed", -1))
    maximum = int(raw["TRAIN"]["epochs"])
    if summary.get("complete") is not True or summary.get("probe_only") is not False:
        raise ValueError("formal run requires a complete, non-probe training summary")
    if completed == maximum:
        return completed
    stopped = summary.get("early_stopped") is True
    minimum = int(raw["TRAIN"].get("min_epochs", maximum))
    patience = int(raw["TRAIN"].get("early_stop_patience", 0))
    if not (stopped and patience > 0 and minimum <= completed < maximum):
        raise ValueError("training ended before all configured epochs without a valid early stop: %d" % completed)
    return completed


def source_hashes(source_root):
    source_root = Path(source_root)
    spec = importlib.util.spec_from_file_location(
        "v16_source_inventory", str(source_root / "tools/v16/queue_jobs.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source_files(source_root)


def execute(command, source_root):
    import json
    print("RUN " + json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=str(source_root), check=True)


def validate_evaluation(path, variant):
    payload = read_json(path)
    if payload.get("timing", {}).get("execution") != "step":
        raise ValueError("formal evaluation must use step execution: %s" % path)
    if payload.get("readout_policy") != variant:
        raise ValueError("evaluation must identify its trained main readout: %s" % path)
    expected = {"first", "second", "main"}
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


def validate_preflight(path):
    payload = read_json(path)
    if payload.get("passed") is not True:
        raise ValueError("GPU backward preflight did not report passed=true")
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
                         args.data_root, project, source,
                         getattr(args, "config_dir", "configs/v16"),
                         getattr(args, "init_checkpoint", None))
    initialization = None
    if raw["TRAIN"].get("init_checkpoint"):
        init_path = Path(raw["TRAIN"]["init_checkpoint"])
        if not init_path.is_file() or not init_path.stat().st_size:
            raise ValueError("initial checkpoint is missing or empty: " + str(init_path))
        initialization = {"path": str(init_path), "sha256": sha256(init_path)}
        expected = getattr(args, "init_sha256", None)
        if expected and initialization["sha256"] != expected:
            raise ValueError("initial checkpoint changed since enqueue")
    out = Path(raw["TRAIN"]["save_root"])
    meta = project / "run/v16_context" / args.release / "jobs" / ("%s_s%d" % (args.variant, args.seed))
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
                "trained_from_scratch": initialization is None, "initialization": initialization,
                "resume": False}
    manifest = meta / "manifest.json"
    started = time.time()
    state = dict(identity, status="preflight", started_at_unix=started)
    write_json(manifest, state)
    checkpoint = out / ("best_val_iou_seed%d.pt" % args.seed)
    base_command = [sys.executable, "-B", "-u", str(source / "tools/v16/train.py"),
                    "--config", str(config), "--data-root", str(Path(args.data_root).resolve()),
                    "--out", str(out), "--device", args.device]
    try:
        preflight_path = meta / "runtime_preflight.json"
        execute([sys.executable, "-B", "-u", str(source / "tools/v16/runtime_preflight.py"),
                 "--config", str(config), "--device", args.device,
                 "--out", str(preflight_path)], source)
        preflight = validate_preflight(preflight_path)
        state.update(status="training")
        write_json(manifest, state)
        execute(base_command + ["--mode", "train"], source)
        summary = read_json(out / "summary.json")
        completed = validate_training_summary(summary, raw)
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
        execute([sys.executable, "-B", "-u", str(source / "tools/v16/profile_inference.py"),
                 "--config", str(config), "--checkpoint", str(checkpoint),
                 "--data-root", str(Path(args.data_root).resolve()), "--out", str(out),
                 "--device", args.device, "--split", "test", "--records", "3", "--repeats", "2"], source)
        profile = validate_profile(out / "profile.json", args.variant, selected_hash)
        if sha256(checkpoint) != selected_hash:
            raise ValueError("selected checkpoint changed during evaluation")
        if sha256(config) != identity["config_sha256"] or source_hashes(source) != identity["source_sha256"]:
            raise ValueError("source or runtime configuration changed during training")
        if initialization and sha256(Path(initialization["path"])) != initialization["sha256"]:
            raise ValueError("initial checkpoint changed during training")
        result = dict(identity, status="complete", epochs_completed=completed,
                      checkpoint=str(checkpoint), checkpoint_sha256=selected_hash,
                      elapsed_seconds=time.time() - started, evaluations=results, profile=profile,
                      preflight=preflight,
                      note="first/second are diagnostics of these weights; each revision variant is independently trained.")
        artifacts = (out / "summary.json", out / "train_stats.json",
                     out / "eval_val.json", out / "eval_test.json", out / "profile.json",
                     checkpoint, config, preflight_path)
        result["artifact_sha256"] = {
            p.relative_to(project).as_posix(): sha256(p) for p in artifacts}
        write_json(out / "comparison_result.json", result)
        state.update(status="complete", finished_at_unix=time.time(),
                     checkpoint_sha256=result["checkpoint_sha256"])
        write_json(manifest, state)
        print("V16_CONTEXT_FINISHED " + str(out), flush=True)
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
    parser.add_argument("--config-dir", default="configs/v16/r2")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--init-sha256")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, "ERROR: %s\n" % error)


if __name__ == "__main__":
    main()
