"""Paired, stepwise memory diagnostics. No training or checkpoint mutation.

Event-labelled measurements sample the feature cell containing each event;
these are not claims that the entire receptive field has that event's label.
Delayed history uses only past states and is zero during warmup.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
import argparse
import copy
import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from train_stream_v1 import build_net, seed_everything
from dataset.ev_uav_stream import EvUAVStream
from dataset.stream_source import make_window_source
from model.evspsegnet_stream import LAYER_NAMES

STRIDE = dict(enc1=1, enc2=2, enc3=4, enc4=8, dec3=4, dec2=2, dec1=1)


class ReadIntervention:
    """Forward hook changes readout only; retains original memory write result."""
    def __init__(self, mode, delay):
        self.mode, self.delay = mode, delay
        self.history = deque(maxlen=delay)

    def __call__(self, neuron, inputs, result):
        current, state = inputs
        spike, (soma, memory), pre = result
        old = torch.zeros_like(current) if state is None else state[1]
        if self.mode == "zero":
            read = torch.zeros_like(old)
        elif self.mode == "delay":
            read = self.history[0] if len(self.history) == self.delay else torch.zeros_like(old)
        elif self.mode == "spatial_mean":
            read = old.mean(dim=(-2, -1), keepdim=True).expand_as(old)
        else:
            raise ValueError(self.mode)
        if self.mode == "delay":
            self.history.append(old.detach().clone())
        beta, _, gain = neuron._coefficients(current)
        # Reconstruct instead of subtracting the old read (avoids cancellation).
        new_pre = current + gain * read if state is None else beta * state[0] + current + gain * read
        output, fired = neuron.activate(new_pre)
        return output, (new_pre - neuron.v_threshold * fired, memory), new_pre


def add_stat(store, key, values):
    if not values.numel():
        return
    values = values.detach().double()
    r = store.setdefault(key, {"sum": 0., "count": 0, "max": 0.})
    r["sum"] += float(values.sum())
    r["count"] += values.numel()
    r["max"] = max(r["max"], float(values.max()))


def finish_stats(store):
    return {k: dict(v, mean=v["sum"] / v["count"]) for k, v in store.items()}


@torch.no_grad()
def diagnose_sequence(net, seq, cfg, q99, device, mode="zero", delay=4):
    other = copy.deepcopy(net).eval()
    handles = [getattr(other, name).neuron.register_forward_hook(ReadIntervention(mode, delay))
               for name in other.dual_memory_layers]
    source = make_window_source(seq, cfg, q99, device, next(net.parameters()).dtype)
    states = changed_states = None
    stats, counts = {}, {"real": [0, 0, 0], "intervention": [0, 0, 0]}
    try:
        for k in range(seq.n_windows):
            x, events, labels, _ = source.window(k)
            logits, new_states, info = net(x, events, states, collect=True)
            altered, changed_states, alt = other(x, events, changed_states, collect=True)
            prob, alt_prob = logits.sigmoid(), altered.sigmoid()
            target = labels > .5
            pred, alt_pred = prob >= cfg["threshold"], alt_prob >= cfg["threshold"]
            for key, prediction in [("real", pred), ("intervention", alt_pred)]:
                for i, value in enumerate([(prediction & target).sum(), (prediction & ~target).sum(), (~prediction & target).sum()]):
                    counts[key][i] += int(value)
            for label, mask in [("target", target), ("background", ~target)]:
                add_stat(stats, label + "/prediction_flip", (pred != alt_pred)[mask].float())
                add_stat(stats, label + "/probability_abs_change", (prob - alt_prob).abs()[mask])
            for i, name in enumerate(LAYER_NAMES):
                add_stat(stats, name + "/all_spike_flip", (info["spikes"][i] != alt["spikes"][i]).float())
                if name not in net.dual_memory_layers:
                    continue
                neuron = getattr(net, name).neuron
                current = info["current"][i]
                _, _, gain = neuron._coefficients(current)
                read = torch.zeros_like(current) if states is None else gain * states[i][1]
                pre = info["u_pre"][i]
                margin = (pre - neuron.v_threshold).abs() / neuron.v_threshold
                # Direct counterfactual fixes this window's U and upstream current.
                direct_flip = ((pre >= neuron.v_threshold) != ((pre - read) >= neuron.v_threshold)).float()
                yy, xx = events["y"] // STRIDE[name], events["x"] // STRIDE[name]
                for label, mask in [("target", target), ("background", ~target)]:
                    for metric, value in [("read_abs_over_theta", read.abs() / neuron.v_threshold),
                                          ("margin_abs_over_theta", margin), ("direct_spike_flip", direct_flip),
                                          ("trajectory_spike_flip", (info["spikes"][i] != alt["spikes"][i]).float())]:
                        sampled = value[events["b"], :, yy, xx]
                        add_stat(stats, name + "/" + label + "/" + metric, sampled[mask])
            states = new_states
    finally:
        for handle in handles:
            handle.remove()
    return {"counts": counts, "statistics": stats}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--intervention", choices=["zero", "delay", "spatial_mean"], default="zero")
    p.add_argument("--delay-windows", type=int, default=4)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.delay_windows < 1:
        p.error("delay-windows must be positive")
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("Refusing to overwrite " + str(output))
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = dict(ckpt["config"], root=args.root, execution="step", input_device="cpu")
    seed_everything(cfg["seed"], cfg["deterministic"])
    net = build_net(cfg, device).eval()
    net.load_state_dict(ckpt["model"])
    if not net.dual_memory_layers or not bool(net.gain_calibrated):
        raise RuntimeError("Requires calibrated dual-memory checkpoint")
    q99 = np.asarray(ckpt["stats"]["q99"], dtype=np.float32)
    dataset = EvUAVStream(cfg["root"], args.split, cfg)
    aggregate, total, sequences = {}, {"real": [0, 0, 0], "intervention": [0, 0, 0]}, []
    for i in range(len(dataset)):
        result = diagnose_sequence(net, dataset[i], cfg, q99, device, args.intervention, args.delay_windows)
        for key, values in result["counts"].items():
            total[key] = [a + b for a, b in zip(total[key], values)]
        for key, values in result["statistics"].items():
            acc = aggregate.setdefault(key, {"sum": 0., "count": 0, "max": 0.})
            acc["sum"] += values["sum"]
            acc["count"] += values["count"]
            acc["max"] = max(acc["max"], values["max"])
        sequences.append({"index": i, "counts": result["counts"]})
        print("sequence %d/%d %s" % (i + 1, len(dataset), result["counts"]), flush=True)
    payload = {"arguments": vars(args), "epoch": ckpt["epoch"], "config": cfg,
               "counts": total, "iou": {k: v[0] / max(sum(v), 1) for k, v in total.items()},
               "statistics": finish_stats(aggregate), "sequences": sequences,
               "notes": "Event-weighted feature-cell samples, not semantic RF labels. All interventions read-only, causal, reset per sequence. Delay warmup uses zero. No timing benchmark. Trajectory changes include downstream feedback; direct_spike_flip holds current/U fixed."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SAVED", output, payload["iou"])


if __name__ == "__main__":
    main()
