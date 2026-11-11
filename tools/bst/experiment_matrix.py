"""Plan retrained ablations and backbone transfer without launching experiments.

Every backbone/seed has its own identity-checked arrival cache; each SDU variant
is retrained and its VoW refitted. Missing checkpoints stay BLOCKED. LIF/TIF/
graded/ReLU variants of EvSpSegNetStream are one architecture family, not four
independent backbone architectures. Run this planner on the experiment host so
the generated absolute paths name that host's files.
"""
import argparse
import copy
import json
import math
import pickle
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bst.pipeline import file_digest, torch_load


POLICIES = (("band", "arrival"), ("band", "fixed"), ("band", "learned"), ("learned", "learned"))
NEURONS = ("lif", "graded", "relu", "tif")


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("experiment identifiers must be 1-64 letters/digits/underscore/dash")
    return value


def variants(base):
    """Retraining contrasts. A parent full config defines all non-ablated choices."""
    patches = {"full": {}, "no_reweight": {"reweight": False},
               "no_revalue": {"revalue": False},
               "no_reaudit": {"reweight": False, "revalue": False}}
    if base["model"]["normalization"] != "mean":
        patches["legacy_flat_mean"] = {"normalization": "mean", "window_gate": False, "bounded_values": False}
    if base["model"].get("window_gate"):
        patches["no_window_gate"] = {"window_gate": False}
    if base["model"].get("bounded_values"):
        patches["unbounded_values"] = {"bounded_values": False}
    output = {}
    for name, patch in patches.items():
        config = copy.deepcopy(base)
        config["model"].update(patch)
        output[name] = config
    return output


def inspect_backbone(path, seed, spec, config):
    """Validate the actual supported streaming checkpoint, not its file name."""
    if not path or not Path(path).is_file():
        raise ValueError("checkpoint not supplied or does not exist")
    if spec.get("adapter") != "evspsegnet_stream":
        raise ValueError("adapter is not implemented; a causal event-logit adapter is required")
    if spec.get("architecture_family") != "EvSpSegNetStream":
        raise ValueError("this adapter builds EvSpSegNetStream; do not label it as another architecture family")
    checkpoint = torch_load(str(path), "cpu")
    cfg, stats = checkpoint["config"], checkpoint["stats"]
    recorded_seed = cfg.get("seed")
    if isinstance(recorded_seed, bool) or recorded_seed != seed:
        raise ValueError("checkpoint.config.seed does not match the requested seed")
    neuron = cfg.get("neuron")
    if neuron not in NEURONS or (spec.get("neuron") and spec["neuron"] != neuron):
        raise ValueError("checkpoint neuron differs from the declared supported variant")
    if float(cfg["window_ms"]) != float(config["model"]["window_ms"]):
        raise ValueError("backbone and SDU window durations differ")
    if int(cfg.get("n_windows", 0)) != int(config["recording_n_windows"]):
        raise ValueError("backbone and SDU recording horizons differ")
    if neuron != "relu" and cfg.get("state_mode") != "carry":
        raise ValueError("live adapter carries state; this backbone was not trained with state_mode=carry")
    pos_weight = float(stats["pos_weight"])
    if not math.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError("invalid backbone pos_weight")
    import numpy as np
    q99 = np.asarray(stats["q99"])
    if q99.size != 2 + 2 * int(cfg["time_bins"]) or not np.isfinite(q99).all() or (q99 <= 0).any():
        raise ValueError("q99 must provide positive finite scales for every input channel")
    import train_stream_v1 as T
    net = T.build_net(cfg, "cpu")
    net.load_state_dict(checkpoint["model"])
    if not bool(net.gain_calibrated):
        raise ValueError("backbone layer gains have not been calibrated")
    return {"checkpoint": str(Path(path).resolve()), "sha256": file_digest(path), "seed": seed,
            "architecture_family": "EvSpSegNetStream", "neuron": neuron,
            "window_ms": cfg["window_ms"], "n_windows": cfg["n_windows"],
            "log_w": math.log(pos_weight), "input_channels": int(q99.size),
            "channels": list(cfg["channels"]), "state_mode_trained": cfg.get("state_mode"),
            "live_adapter": "BackboneArrivalProvider", "source_features": "[f-log(pos_weight), polarity], f is the raw weighted-BCE backbone logit"}


def command(script, *arguments):
    return " ".join(shlex.quote(str(x)) for x in [sys.executable, str(ROOT/"tools/bst"/script)] + list(arguments))


def build_plan(spec, base, destination, data_root, seeds=(37, 38), inspector=inspect_backbone):
    destination, data_root = Path(destination).resolve(), Path(data_root).resolve()
    if base["model"].get("source_features", 2) != 2:
        raise ValueError("the supported live adapter supplies [z0, polarity] only")
    if len(seeds) != len(set(seeds)) or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("unique uint32 seeds are required")
    backbones = spec.get("backbones", [])
    if not backbones:
        raise ValueError("at least one declared backbone is required")
    names = [safe_name(x["id"]) for x in backbones]
    if len(names) != len(set(names)):
        raise ValueError("duplicate backbone id")
    data = {split: sorted(p.name for p in (data_root/split).glob("*.npz")) for split in ("train", "val", "test")}
    data_ready = all(data.values())
    entries, cases, exports, hashes = [], [], [], {}
    for backbone in backbones:
        name = backbone["id"]
        for seed in seeds:
            entry = {"id": name, "seed": seed, "declared_family": backbone.get("architecture_family"),
                     "declared_adapter": backbone.get("adapter"), "ready": False, "errors": []}
            path = backbone.get("checkpoints", {}).get(str(seed))
            try:
                details = inspector(path, seed, backbone, base)
                if details["sha256"] in hashes:
                    raise ValueError("checkpoint bytes reused across backbone/seed entries: " + hashes[details["sha256"]])
                hashes[details["sha256"]] = "%s/s%d" % (name, seed)
                entry.update(details)
                entry["ready"] = data_ready
            except (OSError, ValueError, KeyError, RuntimeError, TypeError, pickle.UnpicklingError) as error:
                entry["errors"].append(str(error))
            if not data_ready:
                entry["errors"].append("raw train/val/test split missing or empty")
            entries.append(entry)
            cache = destination/"priors"/name/("s%d_%s" % (seed, entry.get("sha256", "UNVERIFIED")[:12]))
            if entry["ready"]:
                for split in ("train", "val", "test"):
                    exports.append({"backbone": name, "seed": seed, "split": split,
                        "command": command("export_priors.py", "--checkpoint", entry["checkpoint"], "--data-root", data_root,
                                           "--split", split, "--out", cache/("z0_"+split), "--device", "cuda:0")})
            for variant, cfg in variants(base).items():
                cfg["train"]["seed"] = seed
                key = "%s_s%d_%s" % (name, seed, variant)
                config_path, output = destination/"configs"/(key+".json"), destination/"runs"/key
                commands = []
                if entry["ready"]:
                    common = ["--data-root", data_root, "--z0-root", cache, "--device", "cuda:0"]
                    commands.append(command("train.py", "--config", config_path, *common, "--out", output,
                                            "--seed", seed, "--require-complete-priors"))
                    commands.append(command("fit_policy.py", "--checkpoint", output/"model.pt", *common,
                                            "--out", output/"policy.pt", "--require-complete-priors"))
                    for split in ("val", "test"):
                        for birth, commit in POLICIES:
                            commands.append(command("evaluate.py", "--checkpoint", output/"model.pt", "--policy", output/"policy.pt",
                                                    *common, "--split", split, "--birth", birth, "--commit", commit,
                                                    "--out", output/(split+"_"+birth+"_"+commit)))
                    commands.append(command("final_metrics.py", "--checkpoint", output/"model.pt", "--policy", output/"policy.pt",
                                            "--backbone", entry["checkpoint"], "--data-root", data_root,
                                            "--out", output/"final_metrics", "--physical-gpu", 1))
                cases.append({"id": key, "backbone": name, "seed": seed, "variant": variant,
                              "ready": entry["ready"], "config_path": str(config_path), "config": cfg,
                              "out": str(output), "priors": str(cache), "commands": commands,
                              "retrained_sdu": True, "refitted_vow": True, "test_selects_variant": False})
    families = sorted({entry["architecture_family"] for entry in entries if entry.get("ready")})
    return {"schema": 1, "mode": "plan_only_no_training_or_queue_submission", "seeds": list(seeds),
            "data_root": str(data_root), "raw_split_membership": data, "destination": str(destination),
            "all_backbones_ready": all(entry["ready"] for entry in entries),
            "ready_architecture_families": families, "distinct_architecture_transfer_ready": len(families) >= 2,
            "limits": "same-family neuron changes are variant transfer, not independent architecture transfer; retrain per backbone, not zero-shot transfer",
            "evaluation": "predeclared BIR/band-arrival/band-fixed/band-learned/learned-learned; arrival is age0 history refinement without waiting; test does not select a winner",
            "backbones": entries, "prior_export_commands": exports, "experiments": cases}


def write_plan(plan):
    root = Path(plan["destination"])
    root.mkdir(parents=True, exist_ok=False)
    for case in plan["experiments"]:
        path = Path(case["config_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(case["config"], indent=2), encoding="utf-8")
    (root/"matrix.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Retrained ablations and backbone transfer", "", "Planning only. No job was launched or queued.",
             "LIF/TIF/graded/ReLU use the same EvSpSegNetStream architecture family.",
             "A different architecture needs a real causal adapter before a cross-architecture claim.", "",
             "| Backbone | Seed | Ready | Detail |", "|---|---:|---|---|"]
    for entry in plan["backbones"]:
        lines.append("| %s | %d | %s | %s |" % (entry["id"], entry["seed"], entry["ready"],
                     "; ".join(entry["errors"]) or entry["neuron"]))
    lines += ["", "All commands use the paths on the machine where this plan was generated.",
              "Use the existing GPU 1 queue; do not start these cases concurrently for energy measurements.",
              "Run each backbone's prior export once, then retrain each declared variant.",
              "The plan records metadata, not an immutable source snapshot; freeze the source before launch.", "", "## Prior export", "", "```bash"]
    lines += [entry["command"] for entry in plan["prior_export_commands"]]
    lines += ["```"]
    for case in plan["experiments"]:
        lines += ["", "## " + case["id"], ""]
        lines += ["```bash"] + case["commands"] + ["```"] if case["ready"] else ["BLOCKED: validate the declared backbone and raw dataset first."]
    (root/"COMMANDS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbones", required=True, help="JSON with backbones[] declarations")
    parser.add_argument("--config", required=True, help="full SDU/VoW config; ablations inherit it")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True, help="new plan directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[37, 38])
    parser.add_argument("--require-ready", action="store_true", help="fail before writing if any declared backbone is unavailable")
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.backbones).read_text(encoding="utf-8-sig"))
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    plan = build_plan(spec, config, args.out, args.data_root, tuple(args.seeds))
    if args.require_ready and not plan["all_backbones_ready"]:
        parser.error("matrix has blocked backbones; fix checkpoint paths/metadata before final experiments")
    write_plan(plan)
    print("MATRIX", Path(args.out)/"matrix.json", "all_backbones_ready=", plan["all_backbones_ready"],
          "distinct_architecture_transfer_ready=", plan["distinct_architecture_transfer_ready"])


if __name__ == "__main__":
    main()
