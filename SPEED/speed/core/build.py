"""Build a system (and its loss) from a nested config; every slot is chosen by its `kind`.

Config sections: clock, canvas, representation, network {neuron, backbone, heads, transport}, verifier, readouts,
loss, training, evaluation. See configs/base_evuav.yaml for the base variant of every slot.
"""
import copy
import math

import yaml

from speed.core.clock import build_clock
from speed.core.network import Network
from speed.core.system import System
from speed.slots.backbone.merged_unet import DOWNSAMPLE, MergedUNet
from speed.slots.heads.mark_intensity import MarkIntensityHeads
from speed.slots.heads.motion import MarkIntensityMotionHeads
from speed.slots.loss.mark_intensity import MarkIntensityLoss
from speed.slots.loss.motion import MarkForecastMotionLoss
from speed.slots.neuron.lif import build_neuron_factory
from speed.slots.publish.readouts import build_readouts
from speed.slots.representation.evidence import EvidenceFrontEnd
from speed.slots.transport.identity import NoTransport
from speed.slots.transport.motion import MotionTransport
from speed.slots.verify.drift_cusum import DriftEvidence, velocity_grid
from speed.slots.verify.measured_motion import MeasuredEvidence


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_representation(cfg, dt_ms):
    if cfg["kind"] != "evidence":
        raise ValueError("unknown representation kind %s" % cfg["kind"])
    return EvidenceFrontEnd(cfg["taus_ms"], dt_ms, cfg["dipole_taus_ms"], cfg["dipole_radius_px"], cfg["bg_fast_ms"],
                            cfg["bg_slow_ms"], cfg["bg_prior_per_px_step"], cfg["bg_prior_steps"],
                            cfg["bg_floor_per_px_step"], cfg["bg_smooth_radius_px"],
                            cfg.get("features", ("count", "ratio", "age", "dipole")))


def build_network(cfg, in_channels, dt_ms):
    neuron = build_neuron_factory(cfg["neuron"], dt_ms)
    if cfg["backbone"]["kind"] != "merged_unet":
        raise ValueError("unknown backbone kind %s" % cfg["backbone"]["kind"])
    backbone = MergedUNet(in_channels, cfg["backbone"]["channels"], neuron)
    h = cfg["heads"]
    if h["kind"] == "mark_intensity":
        heads = MarkIntensityHeads(backbone.out_channels, h["hidden"], h["mark_prior"],
                                   h["intensity_prior_per_px_step"], h["log_g_max"])
    elif h["kind"] == "mark_intensity_motion":
        heads = MarkIntensityMotionHeads(backbone.out_channels, in_channels, dt_ms, h["hidden"], h["motion_hidden"],
                                         h["mark_prior"], h["intensity_prior_per_px_step"], h["log_g_max"],
                                         h.get("motion_sigma_init_px_per_step", 1.0))
    else:
        raise ValueError("unknown heads kind %s" % h["kind"])
    kind = cfg.get("transport", {"kind": "none"})["kind"]
    if kind == "none":
        transport = NoTransport()
    elif kind == "motion":
        if "motion" not in heads.outputs:
            raise ValueError("transport 'motion' needs heads with a motion output")
        transport = MotionTransport(dt_ms, float(cfg["transport"].get("gate_probability", 0.5)))
    else:
        raise ValueError("unknown transport kind %s" % kind)
    return Network(backbone, heads, transport)


def build_verifier(cfg, dt_ms):
    if cfg is None:
        return None
    tau = float(cfg.get("track_tau_ms", 0.0))
    decay = math.exp(-float(dt_ms) / tau) if tau > 0 else 0.0
    if cfg["kind"] == "drift_evidence":
        return DriftEvidence(velocity_grid(cfg["velocities_px_per_step"]), int(cfg["footprint_px"]), decay,
                             float(cfg.get("gate_eps", 0.0)))
    if cfg["kind"] == "measured_motion":
        return MeasuredEvidence(dt_ms, cfg.get("taus_ms", (20.0,)), cfg.get("radii_px", (4,)),
                                float(cfg.get("min_weight", 5.0)),
                                int(cfg["footprint_px"]), decay, float(cfg.get("gate_eps", 0.0)),
                                bool(cfg.get("include_zero", True)), cfg.get("fixed_velocity_px_per_step"),
                                bool(cfg.get("oracle", False)), cloud=bool(cfg.get("cloud", False)),
                                cloud_spacing=float(cfg.get("cloud_spacing_px_per_step", 1.0)),
                                zero_weight=cfg.get("zero_weight", "equal"), source=cfg.get("source", "moments"))
    raise ValueError("unknown verifier kind %s" % cfg["kind"])


def build_system(cfg):
    clock = build_clock(cfg["clock"])
    dt_ms = clock.step_ms
    representation = build_representation(cfg["representation"], dt_ms)
    network = build_network(cfg["network"], representation.n_features, dt_ms)
    verifier = build_verifier(cfg.get("verifier"), dt_ms)
    ev = cfg["evaluation"]
    readouts = build_readouts(cfg["readouts"], verifier, ev["threshold"])
    return System(clock, representation, network, verifier, readouts, cfg["canvas"].get("multiple", DOWNSAMPLE),
                  ev["chunk_steps"], ev["threshold"], bool(ev.get("amp", False)))


def build_loss(cfg, dt_ms=None):
    if cfg["kind"] == "mark_intensity":
        return MarkIntensityLoss(cfg["mark_weight"], cfg["intensity_weight"], cfg["intensity_smooth_px"])
    if cfg["kind"] == "mark_forecast_motion":
        if dt_ms is None:
            raise ValueError("the forecast loss needs the clock step (dt_ms)")
        return MarkForecastMotionLoss(dt_ms, cfg["mark_weight"], cfg["forecast_weight"], cfg["motion_weight"],
                                      cfg.get("current_weight", 0.0), cfg["intensity_smooth_px"],
                                      cfg["forecast_floor_per_px_step"])
    raise ValueError("unknown loss kind %s" % cfg["kind"])


def with_sections(cfg, paths):
    """Replace whole top-level sections of cfg by those of the YAML files in paths (later files win)."""
    out = copy.deepcopy(cfg)
    for path in paths or []:
        out.update(copy.deepcopy(load_config(path)))
    return out


def with_overrides(cfg, overrides):
    """overrides: ["section.key=value", ...] with YAML-parsed values."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        path, value = item.split("=", 1)
        node = out
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[int(key)] if isinstance(node, list) else node[key]
        node[keys[-1]] = yaml.safe_load(value)
    return out
