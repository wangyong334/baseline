"""V14 paired, independently trained BIR memory ablation (Python 3.8 compatible).

The existing trainer is unchanged. Only the trained state mode is evaluated;
cross-mode interventions are deliberately excluded from the comparison.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MODES = {"carry": "carry", "reset": "reset_each_window"}
DEFAULT_RELEASE = "mem_ablation_20261112_r1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def json_finite(value):
    """Empty diagnostic groups use NaN upstream; encode those as JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(json_finite(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def load_yaml(path):
    import yaml
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def flatten(raw):
    result = {}
    for section in raw.values():
        for key, value in section.items():
            if key in result:
                raise ValueError("Duplicate config key: " + key)
            result[key] = value
    return result


def validate_pair(source_root=ROOT):
    """Reject any extra mechanism or unmatched hyperparameter in either arm."""
    source_root = Path(source_root)
    base = load_yaml(source_root / "configs/evisseg_stream_tif_notransport.yaml")
    pair = {}
    for mode, state_mode in MODES.items():
        raw = load_yaml(source_root / ("configs/v14/bir_%s.yaml" % mode))
        expected = copy.deepcopy(base)
        expected["NET"]["state_mode"] = state_mode
        expected["TRAIN"]["save_root"] = "log/v14/%s_seed37" % mode
        expected["TRAIN"]["stream_layout"] = "greedy"  # Historical V5 default, made explicit.
        if raw != expected:
            raise ValueError("V14 %s must match basic V5 except state_mode/save_root" % mode)
        flat = flatten(raw)
        if flat["neuron"] != "tif" or any(k.startswith("comm_") for k in flat):
            raise ValueError("V14 requires basic BIR without communication loss")
        for mechanism in ("dual_memory", "subthreshold_residual", "invariant_input", "comod",
                          "noise_aug", "spiking_decoder"):
            if flat.get(mechanism):
                raise ValueError("Unexpected mechanism: " + mechanism)
        pair[mode] = raw
    return pair


def dataset_inventory(data_root):
    records = {}
    for split in ("train", "val", "test"):
        paths = sorted((Path(data_root) / split).glob("*.npz"))
        if not paths:
            raise ValueError("Missing nonempty split: " + str(Path(data_root) / split))
        records[split] = [{"name": p.name, "bytes": p.stat().st_size,
                           "mtime_ns": p.stat().st_mtime_ns} for p in paths]
        if any(item["bytes"] == 0 for item in records[split]):
            raise ValueError("Empty dataset file in " + split)
    return records


def checked_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Invalid release name")
    return value


def runtime_config(mode, seed, release, data_root, project_root, source_root=ROOT):
    checked_name(release)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    raw = copy.deepcopy(validate_pair(source_root)[mode])
    project_root = Path(project_root).resolve()
    raw["DATA"]["root"] = str(Path(data_root).resolve())
    raw["DATA"]["stats_path"] = str(project_root / "run/v14" / release / "train_stats.json")
    raw["TRAIN"]["seed"] = seed
    raw["TRAIN"]["save_root"] = str(project_root / "log/v14" / release / ("%s_s%d" % (mode, seed)))
    return raw


def execute(command, source_root):
    print("RUN " + json.dumps(command, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=str(source_root), check=True)


def source_hashes(source_root):
    root = Path(source_root)
    paths = [root / "train_stream_v1.py", root / "tools/stream_train_stats.py",
             root / "configs/evisseg_stream_tif_notransport.yaml"]
    for directory, suffix in (("model", "*.py"), ("dataset", "*.py"), ("utils", "*.py"),
                              ("tools/v14", "*.py"), ("configs/v14", "*.yaml")):
        paths.extend((root / directory).rglob(suffix))
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(set(paths))}


def ensure_stats(raw, config_path, inventory, source_root):
    stats_path = Path(raw["DATA"]["stats_path"])
    meta_path = stats_path.with_name("train_stats_provenance.json")
    expected = {"data_root": raw["DATA"]["root"], "inventory": inventory,
                "data_config": {k: v for k, v in raw["DATA"].items() if k != "stats_path"},
                "pos_weight_cap": raw["TRAIN"]["pos_weight_cap"]}
    if stats_path.exists():
        if not meta_path.exists():
            raise RuntimeError("Stats exist without provenance; use a new release: " + str(stats_path))
        previous = read_json(meta_path)
        if previous.get("inputs") != expected or previous.get("sha256") != sha256(stats_path):
            raise RuntimeError("Dataset/stats changed within this release")
        return
    if meta_path.exists():
        raise RuntimeError("Stats provenance exists but stats are missing")
    execute([sys.executable, "-u", str(Path(source_root) / "tools/stream_train_stats.py"),
             "--config", str(config_path)], source_root)
    stats = read_json(stats_path)
    if len(stats.get("q99", [])) != 12 or stats.get("train_pos", 0) <= 0:
        raise RuntimeError("Invalid generated training statistics")
    write_json(meta_path, {"inputs": expected, "sha256": sha256(stats_path),
                           "statistics_from": "train only; val/test audited without collecting statistics"})


def validate_checkpoint_config(actual, expected):
    """Permit trainer-added defaults, but never a different arm/seed/data/protocol."""
    wanted = flatten(expected)
    for key, value in wanted.items():
        if actual.get(key) != value:
            raise RuntimeError("Checkpoint/config mismatch at %s" % key)
    if any(k.startswith("comm_") for k in actual):
        raise RuntimeError("Checkpoint unexpectedly contains communication loss")


def evaluate_trained_mode(config_path, checkpoint, device):
    """Reuse the established metrics and step/cpu timing path, with one state mode."""
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    import train_stream_v1 as trainer

    raw = load_yaml(config_path)
    try:
        ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    except TypeError:  # torch 1.9 server
        ckpt = torch.load(str(checkpoint), map_location=device)
    validate_checkpoint_config(ckpt["config"], raw)
    cfg = trainer.eval_train_config(ckpt["config"], flatten(raw))
    cfg["execution"], cfg["input_device"] = "step", "cpu"
    trainer.seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    device = torch.device(device)
    net = trainer.build_net(cfg, device)
    net.load_state_dict(ckpt["model"])
    if not bool(net.gain_calibrated):
        raise RuntimeError("Checkpoint gains are not calibrated")
    q99 = np.asarray(ckpt["stats"]["q99"], dtype=np.float32)
    mode = cfg["state_mode"]
    outputs = {}
    for split in ("val", "test"):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = trainer.evaluate_split(net, cfg, q99, device, split, mode, full=True)
        result["peak_memory_gib"] = trainer.peak_memory_gib(device)
        result["tau"] = trainer.tau_statistics(net)
        outputs[split] = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                          "epoch": int(ckpt["epoch"]), "trained_state_mode": mode,
                          "threshold": float(cfg["threshold"]), "seed": int(cfg["seed"]),
                          "neuron": cfg["neuron"], "parameters": trainer.count_parameters(net),
                          "evaluation": "trained mode only; original step/cpu implementation",
                          mode: result}
        trained_metrics(outputs[split], mode, int(cfg["seed"]))
        target = Path(checkpoint).parent / ("eval_%s_trained_mode.json" % split)
        if target.exists():
            raise RuntimeError("Refusing to overwrite evaluation: " + str(target))
        write_json(target, outputs[split])
        print("EVAL %s mode=%s IoU=%.6f ACC=%.6f Pd=%.6f Fa=%.8g" %
              (split, mode, result["iou"], result["acc"], result["pd"], result["fa"]), flush=True)
    return outputs


def trained_metrics(evaluation, expected_mode, expected_seed):
    if evaluation.get("trained_state_mode") != expected_mode or evaluation.get("seed") != expected_seed:
        raise RuntimeError("Evaluation belongs to a different trained arm or seed")
    result = evaluation[expected_mode]
    for key in ("iou", "acc", "pd", "fa"):
        if not math.isfinite(float(result[key])):
            raise RuntimeError("Nonfinite metric: " + key)
    return {key: result.get(key) for key in ("iou", "acc", "pd", "fa", "timing", "latency",
                                            "operations", "peak_memory_gib", "segment_iou",
                                            "messages_mean", "messages_per_window")}


def run(args):
    source_root = ROOT
    project_root = Path(args.project_root).resolve()
    raw = runtime_config(args.mode, args.seed, args.release, args.data_root, project_root, source_root)
    out = Path(raw["TRAIN"]["save_root"])
    meta = project_root / "run/v14" / args.release / "jobs" / ("%s_s%d" % (args.mode, args.seed))
    config_path = meta / "config.yaml"
    manifest_path = meta / "manifest.json"
    checkpoint = out / ("best_val_iou_seed%d.pt" % args.seed)
    if args.stage == "evaluate":
        if not config_path.exists():
            raise RuntimeError("Missing prepared runtime config")
        evaluate_trained_mode(config_path, checkpoint, args.device)
        return
    inventory = dataset_inventory(args.data_root)
    identity = {"mode": args.mode, "state_mode": MODES[args.mode], "seed": args.seed,
                "release": args.release, "config": raw, "dataset_inventory": inventory,
                "source_sha256": source_hashes(source_root)}
    done = out / "comparison_result.json"
    if out.exists() or meta.exists():
        if done.exists() and manifest_path.exists() and read_json(manifest_path).get("identity") == identity:
            result = read_json(done)
            if result.get("checkpoint_sha256") == sha256(checkpoint) and result.get("complete") is True:
                print("ALREADY COMPLETE " + str(out), flush=True)
                return
        raise RuntimeError("Existing/partial run is protected; use a new release (no automatic resume): " + str(out))
    import yaml
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.mkdir()  # Exclusive claim: concurrent/direct invocations cannot overwrite a run.
    with config_path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(raw, stream, sort_keys=False, allow_unicode=True)
    manifest = {"identity": identity, "status": "preparing", "started_unix": time.time(),
                "python": sys.executable, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "independent_training": True, "resume": False}
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    try:
        ensure_stats(raw, config_path, inventory, source_root)
        manifest["stats_sha256"] = sha256(raw["DATA"]["stats_path"])
        manifest["status"] = "training"
        write_json(manifest_path, manifest)
        execute([sys.executable, "-u", str(source_root / "train_stream_v1.py"), "--mode", "train",
                 "--config", str(config_path), "--device", args.device], source_root)
        summary = read_json(out / "summary.json")
        if int(summary["epochs_completed"]) != int(raw["TRAIN"]["epochs"]):
            raise RuntimeError("Training did not complete its configured epoch budget")
        if not checkpoint.is_file():
            raise RuntimeError("Best validation checkpoint missing")
        manifest["status"] = "evaluating"
        write_json(manifest_path, manifest)
        execute([sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "evaluate",
                 "--mode", args.mode, "--seed", str(args.seed), "--release", args.release,
                 "--data-root", str(Path(args.data_root).resolve()), "--project-root", str(project_root),
                 "--device", args.device], source_root)
        evaluations = {split: read_json(out / ("eval_%s_trained_mode.json" % split)) for split in ("val", "test")}
        result = {"complete": True, "mode": args.mode, "trained_state_mode": MODES[args.mode],
                  "seed": args.seed, "release": args.release, "checkpoint": str(checkpoint),
                  "checkpoint_sha256": sha256(checkpoint), "best_epoch_zero_based": evaluations["val"]["epoch"],
                  "epochs_completed": summary["epochs_completed"], "threshold": raw["EVAL"]["threshold"],
                  "communication_budget_loss": False, "elapsed_seconds": time.perf_counter() - started,
                  "metrics": {split: trained_metrics(value, MODES[args.mode], args.seed)
                              for split, value in evaluations.items()},
                  "cost_note": "operations are model estimates, not measured board energy"}
        write_json(done, result)
        manifest["status"] = "complete"
        manifest["finished_unix"] = time.time()
        write_json(manifest_path, manifest)
        print("V14 FINISHED " + str(done), flush=True)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["finished_unix"] = time.time()
        write_json(manifest_path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("full", "evaluate"), default="full", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
