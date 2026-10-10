"""Full-system operation counts of a system, per step, from statistics measured during inference.

EnergyStats collects what the counts depend on: events per step, non-zero fraction of the network input, firing rates
(LayerMonitor), the verifier's active fraction (tube intensity >= gate eps, dilated by the footprint, i.e. the
(hypothesis, pixel) pairs a sparse implementation must compute) and the publishing units' occupancy.
energy_parts() turns them into speed.eval.energy parts for every slot; speed.eval.energy.system_energy prices them.
"""
import torch
import torch.nn.functional as F

from speed.eval.energy import part, pixel_head
from speed.slots.publish.learned import LearnedDelayReadout, LearnedPublishReadout
from speed.slots.publish.readouts import FixedDelayReadout, NetReadout, PublishReadout


class EnergyStats(object):
    def __init__(self):
        self.events = 0
        self.steps = 0
        self.nonzero = 0
        self.elements = 0
        self.active = 0.0
        self.active_steps = 0
        self.unit_steps = 0
        self.queries = 0

    def update_inputs(self, inputs):
        self.nonzero += int((inputs != 0).sum())
        self.elements += int(inputs.numel())

    def update_verifier(self, state, verifier):
        self.queries += int(state.get("queried", 0)) + int(state.get("queries", 0))
        if getattr(verifier, "gate_eps", 0.0) <= 0:
            return
        act = (state["G"] >= verifier.gate_eps).to(state["G"].dtype)
        f = verifier.footprint
        dil = act if f == 1 else F.max_pool2d(act, f, stride=1, padding=f // 2)
        self.active += float(dil.mean())
        self.active_steps += 1

    def summary(self, monitor_summary=None):
        return {"events_per_step": self.events / float(max(self.steps, 1)),
                "input_density": self.nonzero / float(max(self.elements, 1)),
                "verifier_active_fraction": (self.active / self.active_steps) if self.active_steps else 1.0,
                "publish_unit_steps_per_step": self.unit_steps / float(max(self.steps, 1)),
                "verifier_queries_per_step": self.queries / float(max(self.steps, 1)),
                "firing_rates": {k: v["firing_rate"] for k, v in (monitor_summary or {}).items()}}


def _sum(parts):
    return part(sum(p["mac"] for p in parts), sum(p["elementwise"] for p in parts),
                sum(p["transcendental"] for p in parts))


def energy_parts(system, height, width, stats):
    """{slot: part} per step. stats: EnergyStats.summary()."""
    e = float(stats["events_per_step"])
    out = {"representation": _sum(system.representation.operations(height, width, e))}
    layers = system.network.backbone.operations(height, width, stats["firing_rates"], stats["input_density"])
    out["backbone"] = part(sum(l["mac"] for l in layers), sum(l["ac"] for l in layers))
    heads = system.network.heads
    out["heads"] = pixel_head(heads.in_channels, heads.hidden, 2, height * width, transcendental_per_position=1,
                              note="dense, as implemented")
    if hasattr(heads, "motion_channels"):
        out["motion_encoder"] = part(mac=height * width * heads.encoder_macs(),
                                     ac=height * width * (heads.motion_hidden + heads.motion_channels),
                                     note="dense, as implemented")
        out["background_head"] = pixel_head(heads.in_channels + heads.input_channels, heads.background_hidden, 1,
                                            height * width, note="dense, as implemented")
    transport = system.network.transport
    if not transport.is_identity and hasattr(transport, "operations"):
        ops = transport.operations(system.network.backbone, height, width)
        out["transport"] = part(ops["mac"], ops["ac"], ops["transcendental"])
    v = system.verifier
    if v is not None:
        out["verifier"] = _sum(v.operations(height, width, min(1.0, stats["verifier_active_fraction"]),
                                            events_per_step=e,
                                            queries_per_step=float(stats.get("verifier_queries_per_step", 0.0))))
        V = float(v.n_hypotheses)
    for r in system.readouts:
        if isinstance(r, NetReadout):
            out["readout_" + r.name] = part(transcendental=e)
        elif isinstance(r, FixedDelayReadout):
            ops = [part(ac=e * (2 * float(d) * V + V), transcendental=e * (V + 1)) for d in r.delays]
            out["readout_fixed_delay"] = part(sum(p["mac"] for p in ops), sum(p["ac"] for p in ops),
                                              sum(p["transcendental"] for p in ops))
        elif isinstance(r, PublishReadout):
            u = float(stats["publish_unit_steps_per_step"])
            f2 = float(v.footprint ** 2)
            out["readout_" + r.name] = part(ac=u * (3 * V if r.anchor else 2 * V) + (f2 * height * width if r.anchor else 0),
                                            transcendental=u * (V + 1))
        elif isinstance(r, LearnedDelayReadout):
            # per event and step of its wait, per hypothesis: tube positions, 9 offsets x (3 reads, ratio, product,
            # difference) with one log1p, position mixture (9 exp + 1 log), anchored sum; hypothesis mixture
            u = e * float(r.max_delay)
            out["readout_learned_delay"] = part(mac=u * V * (4 + 9 * 2), ac=u * V * (9 * 4 + 12),
                                                transcendental=u * (V * (9 + 9 + 1) + V + 1) + e * len(r.delays))
        elif isinstance(r, LearnedPublishReadout):
            u = float(stats["publish_unit_steps_per_step"])
            h = r.head.operations_per_unit()
            out["readout_" + r.name] = part(mac=u * (V * (4 + 9 * 2) + 1 + h["mac"]) + e * h["mac"],
                                            ac=u * (V * (9 * 4 + 12) + h["ac"]) + e * h["ac"],
                                            transcendental=u * (V * (9 + 9 + 1) + V + 1) + e)
    return out

