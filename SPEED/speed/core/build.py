"""Build a system (and its loss) from a nested config; every slot is chosen by its `kind`.

Config sections: clock, canvas, representation, network {neuron, backbone, heads, transport, motion, readout_head},
verifier, readouts, loss, training, evaluation. configs/base_evuav.yaml has the base variant of every slot,
configs/v4/v43_*.yaml the V4-3 variants (learned motion anchors + tube evidence + learned fusion and publishing).
"""
import copy
import math

import yaml

from speed.core.clock import build_clock
from speed.core.network import Network
from speed.core.system import System
from speed.slots.backbone.merged_unet import DOWNSAMPLE, MergedUNet
from speed.slots.heads.mark_intensity import MarkIntensityHeads
from speed.slots.heads.readout import EvidenceReadoutHead
from speed.slots.heads.v43 import V43Heads
from speed.slots.loss.mark_intensity import MarkIntensityLoss
from speed.slots.loss.v43 import V43Loss
from speed.slots.motion.anchors import AnchorMotion
from speed.slots.neuron.lif import build_neuron_factory
from speed.slots.publish.readouts import build_readouts
from speed.slots.representation.evidence import EvidenceFrontEnd
from speed.slots.transport.identity import NoTransport
from speed.slots.verify.drift_cusum import DriftEvidence, velocity_grid
from speed.slots.verify.tube_evidence import TubeEvidence


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


def build_network(cfg, in_channels, dt_ms, feature_names=None):
    """feature_names: the representation's channels; a trailing dense log_mu0 channel reaches the heads only."""
    neuron = build_neuron_factory(cfg["neuron"], dt_ms)
    if cfg["backbone"]["kind"] != "merged_unet":
        raise ValueError("unknown backbone kind %s" % cfg["backbone"]["kind"])
    names = list(feature_names or [])
    backbone_channels = None
    if "log_mu0" in names:
        if names[-1] != "log_mu0":
            raise ValueError("log_mu0 must be the last input channel")
        backbone_channels = in_channels - 1
    backbone = MergedUNet(in_channels if backbone_channels is None else backbone_channels,
                          cfg["backbone"]["channels"], neuron)
    h = cfg["heads"]
    motion = None
    if h["kind"] == "mark_intensity":
        heads = MarkIntensityHeads(backbone.out_channels, h["hidden"], h["mark_prior"],
                                   h["intensity_prior_per_px_step"], h["log_g_max"])
    elif h["kind"] == "v43":
        if "log_mu0" not in names:
            raise ValueError("heads v43 need the log_mu0 input channel (representation feature logmu)")
        heads = V43Heads(backbone.out_channels, in_channels, names.index("log_mu0"), h["hidden"], h["mark_prior"],
                         h["intensity_prior_per_px_step"], h["log_g_max"], h["motion_channels"], h["motion_hidden"],
                         h["background_hidden"], bool(h.get("detach_trunk", True)))
        m = cfg["motion"]
        motion = AnchorMotion(h["motion_channels"], m["rings_px_per_step"], m["directions"], m["hidden"])
    else:
        raise ValueError("unknown heads kind %s" % h["kind"])
    if cfg.get("transport", {"kind": "none"})["kind"] != "none":
        raise ValueError("only transport kind none is available")
    rh = cfg.get("readout_head")
    readout_head = None if rh is None else EvidenceReadoutHead(rh["max_delay_steps"], rh["stability_hidden"])
    return Network(backbone, heads, NoTransport(), readout_head, backbone_channels, motion)


def build_verifier(cfg, dt_ms, network=None):
    if cfg is None:
        return None
    if cfg["kind"] == "tube_evidence":
        if getattr(network, "readout_head", None) is None or getattr(network, "motion", None) is None:
            raise ValueError("verifier tube_evidence needs network.readout_head and the v43 heads (network.motion)")
        return TubeEvidence(dt_ms, network.readout_head, network.motion, int(cfg["hypotheses"]),
                            float(cfg["mu_floor_per_px_step"]), cfg.get("background", "head"),
                            bool(cfg.get("anchor", True)))
    tau = float(cfg.get("track_tau_ms", 0.0))
    decay = math.exp(-float(dt_ms) / tau) if tau > 0 else 0.0
    if cfg["kind"] == "drift_evidence":
        return DriftEvidence(velocity_grid(cfg["velocities_px_per_step"]), int(cfg["footprint_px"]), decay,
                             float(cfg.get("gate_eps", 0.0)))
    raise ValueError("unknown verifier kind %s" % cfg["kind"])


def build_system(cfg):
    clock = build_clock(cfg["clock"])
    dt_ms = clock.step_ms
    representation = build_representation(cfg["representation"], dt_ms)
    network = build_network(cfg["network"], representation.n_features, dt_ms, representation.feature_names())
    verifier = build_verifier(cfg.get("verifier"), dt_ms, network)
    ev = cfg["evaluation"]
    readouts = build_readouts(cfg["readouts"], verifier, ev["threshold"])
    return System(clock, representation, network, verifier, readouts, cfg["canvas"].get("multiple", DOWNSAMPLE),
                  ev["chunk_steps"], ev["threshold"], bool(ev.get("amp", False)))


def build_loss(cfg, dt_ms=None, system=None):
    if cfg["kind"] == "mark_intensity":
        return MarkIntensityLoss(cfg["mark_weight"], cfg["intensity_weight"], cfg["intensity_smooth_px"])
    if cfg["kind"] == "v43":
        if system is None or not isinstance(system.verifier, TubeEvidence):
            raise ValueError("loss v43 needs the system with its tube_evidence verifier")
        theta = math.log(system.threshold / (1.0 - system.threshold))
        return V43Loss(dt_ms, system.verifier, theta, bool(cfg.get("train_base", True)), cfg["mark_weight"],
                       cfg["intensity_weight"], cfg["intensity_smooth_px"], cfg["background_weight"],
                       cfg["motion_weight"], cfg["residual_weight"], cfg["growth_weight"], cfg["evidence_weight"],
                       cfg["stability_weight"], cfg["max_queries"])
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
