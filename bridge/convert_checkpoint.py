"""Legacy V2-1 checkpoint (train_stream_v2.py) -> SPEED checkpoint with the base variant of every slot.

Network / representation / training settings come from the checkpoint's own config; the evaluation-time settings
(verifier, readouts, threshold, chunking) come from the frozen V3 YAML, exactly as legacy `--mode eval` merged them.

    python bridge/convert_checkpoint.py --legacy log/v21_floor4_seed37/best_val_iou_seed37.pt \
        --out log/speed_ckpt/v21_s37.pt
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "SPEED"))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from speed.core.training import save_checkpoint  # noqa: E402

LEGACY_V3_YAML = os.path.join(ROOT, "configs", "evisseg_stream_v3.yaml")
# Keys legacy --mode eval took from the current YAML instead of the checkpoint.
EVAL_KEYS = ("threshold", "pd_detT", "correct_thresh", "readout_delays", "eval_chunk", "fusion_weight",
             "cusum_axis_velocities", "cusum_footprint", "cusum_compensator", "cusum_nb_kappa", "cusum_aggregate",
             "cusum_track_tau_ms", "cusum_memory_gain", "cusum_gate_eps", "cusum_reset_radius", "publish",
             "publish_deadline", "publish_theta", "publish_upper", "publish_lower", "publish_collapse", "publish_gate",
             "publish_anchor", "attr")


def require(cfg, key, allowed, default=None):
    value = cfg.get(key, default)
    if value not in allowed:
        raise ValueError("legacy config %s=%r has no SPEED base variant (allowed %s)" % (key, value, allowed))


def convert_config(c):
    """Flat legacy config -> nested SPEED config (base variants only; anything else is rejected)."""
    require(c, "neuron", ("lif",))
    require(c, "norm", ("none",))
    require(c, "merged_decoder", (True,))
    require(c, "bg_mode", ("adaptive",), "adaptive")
    require(c, "cusum_compensator", ("poisson",))
    require(c, "cusum_aggregate", ("lme",), "lme")
    require(c, "cusum_memory_gain", (1.0, 1), 1.0)
    require(c, "attr", (False, None), False)
    require(c, "state_mode", ("carry",))
    readouts = [{"kind": "net"},
                {"kind": "fixed_delay", "delays_steps": [int(d) for d in c["readout_delays"]],
                 "fusion_weight": float(c.get("fusion_weight", 1.0))}]
    if c.get("publish"):
        readouts.append({"kind": "publish", "deadline_steps": int(c.get("publish_deadline", 5)),
                         "theta": c.get("publish_theta"), "upper": float(c.get("publish_upper", 1.0)),
                         "lower": float(c.get("publish_lower", 2.0)), "collapse": c.get("publish_collapse", "linear"),
                         "gate": c.get("publish_gate"), "anchor": bool(c.get("publish_anchor")),
                         "fusion_weight": float(c.get("fusion_weight", 1.0))})
    return {
        "dataset": "evuav",
        "clock": {"kind": "fixed", "step_ms": float(c["window_ms"])},
        "canvas": {"multiple": 8},
        "representation": {"kind": "evidence", "taus_ms": list(c["fe_taus_ms"]),
                           "dipole_taus_ms": list(c["fe_dipole_taus_ms"]), "dipole_radius_px": c["fe_dipole_radius"],
                           "bg_fast_ms": c["bg_fast_ms"], "bg_slow_ms": c["bg_slow_ms"],
                           "bg_prior_per_px_step": c["bg_prior"], "bg_prior_steps": c["bg_prior_windows"],
                           "bg_floor_per_px_step": c["bg_floor"], "bg_smooth_radius_px": c["bg_smooth_radius"],
                           "features": list(c.get("fe_features", ["count", "ratio", "age", "dipole"]))},
        "network": {"neuron": {"kind": "lif", "v_threshold": c["v_threshold"], "tau_init_ms": c["tau_init_ms"],
                               "tau_min_ms": c["tau_min_ms"], "tau_max_ms": c["tau_max_ms"],
                               "u_floor": c.get("neuron_u_floor"), "u_ceil": c.get("neuron_u_ceil")},
                    "backbone": {"kind": "merged_unet", "channels": list(c["channels"])},
                    "heads": {"kind": "mark_intensity", "hidden": c["head_hidden"], "mark_prior": c["mark_prior"],
                              "intensity_prior_per_px_step": c["intensity_prior"], "log_g_max": c["log_g_max"]},
                    "transport": {"kind": "none"}},
        "verifier": {"kind": "drift_evidence", "velocities_px_per_step": list(c["cusum_axis_velocities"]),
                     "footprint_px": c["cusum_footprint"], "track_tau_ms": c.get("cusum_track_tau_ms", 0.0),
                     "gate_eps": c.get("cusum_gate_eps", 0.0)},
        "readouts": readouts,
        "loss": {"kind": "mark_intensity", "mark_weight": c["loss_mark_weight"],
                 "intensity_weight": c["loss_intensity_weight"],
                 "intensity_smooth_px": c.get("loss_intensity_smooth", 3)},
        "training": {"kind": "tbptt", "seed": c["seed"], "epochs": c["epochs"], "lr": c["lr"], "lr_end": c["lr_end"],
                     "tbptt_steps": c["tbptt_k"], "grad_clip": c["grad_clip"], "num_workers": c.get("num_workers", 2),
                     "deterministic": c.get("deterministic", True), "carry": True,
                     "calibration": {"sequences": c["calib_sequences"], "steps": c["calib_windows"],
                                     "samples_per_channel": c["calib_samples_per_channel"],
                                     "min_positive": c["calib_min_positive"], "quantile": c["calib_quantile"],
                                     "gain_min": c["calib_gain_min"], "gain_max": c["calib_gain_max"]}},
        "evaluation": {"threshold": c["threshold"], "chunk_steps": c["eval_chunk"], "frame_ms": c.get("pd_detT", 50),
                       "correct_thresh": c.get("correct_thresh", 1e-4)},
    }


def convert_state_dict(state_dict):
    out = {}
    for key, value in state_dict.items():
        out[("heads.net." + key[len("head."):]) if key.startswith("head.") else key] = value
    return out


def legacy_eval_config(ckpt_config, yaml_path=LEGACY_V3_YAML):
    from utils.stream_common import load_flat_config
    cfg = dict(ckpt_config)
    current = load_flat_config(yaml_path)
    for key in EVAL_KEYS:
        if key in current:
            cfg[key] = current[key]
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--yaml", default=LEGACY_V3_YAML)
    args = parser.parse_args()
    ckpt = torch.load(args.legacy, map_location="cpu")
    config = convert_config(legacy_eval_config(ckpt["config"], args.yaml))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_checkpoint(args.out, _StateDictHolder(convert_state_dict(ckpt["model"])), None, ckpt.get("epoch"),
                    ckpt.get("best_val_iou"), config, {"converted_from": os.path.abspath(args.legacy)})
    print("converted %s (epoch %s) -> %s" % (args.legacy, ckpt.get("epoch"), args.out))


class _StateDictHolder(object):
    def __init__(self, sd):
        self.sd = sd

    def state_dict(self):
        return self.sd


if __name__ == "__main__":
    main()
