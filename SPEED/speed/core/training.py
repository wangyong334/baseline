"""Slot 10 base variant: stateful truncated BPTT over whole streams, one parameter update per stream.

train_stream: chunks of tbptt_steps (first chunk length random in 1..k so no step is always a chunk start);
representation under no_grad, network + heads + loss with gradients, states detached at chunk borders;
loss divided by the stream's event count; gradient clipping; one optimiser step.
Also: gain calibration driver, learning-rate schedule, layer monitor, checkpoint I/O.
"""
import math

import numpy as np
import torch

from speed.core.clock import EventBlocks
from speed.slots.backbone.merged_unet import LAYER_NAMES, calibrate_gains
from speed.slots.neuron.lif import LIF2d, detach_states


def make_chunks(n_steps, k, first_len):
    if not 1 <= first_len <= k:
        raise ValueError("first_len must be in [1, k]")
    chunks, start, end = [], 0, min(first_len, n_steps)
    while start < n_steps:
        chunks.append((start, end))
        start, end = end, min(end + k, n_steps)
    return chunks


def linear_epoch_lr(epoch, epochs, start_lr, end_lr):
    if epochs < 2:
        raise ValueError("the linear schedule needs at least 2 epochs")
    if epoch <= 0:
        return float(start_lr)
    if epoch >= epochs - 1:
        return float(end_lr)
    return float(start_lr + (end_lr - start_lr) * epoch / (epochs - 1))


def select_subset(names, size, seed):
    names = sorted(names)
    if size >= len(names):
        return names
    picked = np.random.RandomState(int(seed)).choice(len(names), int(size), replace=False)
    return sorted(names[i] for i in picked)


def gradient_group_norms(network):
    squares = {}
    for name, param in network.named_parameters():
        parts = name.split(".")
        group = ".".join(parts[:2]) if parts[0] == "backbone" else parts[0]
        value = 0.0 if param.grad is None else float(param.grad.detach().double().pow(2).sum())
        squares[group] = squares.get(group, 0.0) + value
    return {k: math.sqrt(v) for k, v in squares.items()}


def train_stream(system, loss_fn, stream, optimizer, tbptt_steps, grad_clip, rng, max_steps=None, carry=True):
    network, rep = system.network, system.representation
    device, dtype = system.device, system.dtype
    H, W = system.canvas(stream)
    steps = system.clock.partition(stream)
    n = steps.n_steps if max_steps is None else min(int(max_steps), steps.n_steps)
    k = int(tbptt_steps)
    chunks = make_chunks(n, k, int(rng.randint(1, k + 1)))
    total_events = int(steps.bounds[n] - steps.bounds[0])
    denominator = float(max(total_events, 1))
    blocks = EventBlocks(stream, steps, H, W, device, dtype)
    rep_state = rep.init_state(1, H, W, device, dtype)
    states = None
    sums = {"loss_sum": 0.0, "mark_sum": 0.0, "intensity_sum": 0.0}
    optimizer.zero_grad(set_to_none=True)
    for start, end in chunks:
        blk = blocks.block(start, end)
        with torch.no_grad():
            rep_state, inputs, _ = rep.encode(rep_state, blk)
        outputs, states, _ = network.forward_chunk(inputs, states, carry)
        ev = blk["events"]
        logits = outputs["mark"][ev["t"], ev["b"], 0, ev["y"], ev["x"]]
        loss, parts = loss_fn(outputs, logits, blk)
        sums["mark_sum"] += parts["mark"]
        sums["intensity_sum"] += parts["intensity"]
        if loss is not None:
            (loss / denominator).backward()
            value = float(loss.detach())
            if not math.isfinite(value):
                raise RuntimeError("non-finite loss: %s steps %d-%d" % (stream.name, start, end))
            sums["loss_sum"] += value
        states = detach_states(states)
    norms = gradient_group_norms(network)
    missing = [name for name, p in network.named_parameters() if p.requires_grad and p.grad is None]
    torch.nn.utils.clip_grad_norm_(network.parameters(), float(grad_clip))
    optimizer.step()
    sums.update({"events": total_events, "grad_norms": norms, "missing_grads": missing})
    return sums


def calibrate(system, streams, cfg, seed, carry=True):
    """Gain calibration on the first cfg['steps'] steps of the given training streams."""
    device = system.device
    n_steps = int(cfg["steps"])

    def inputs_of(stream):
        H, W = system.canvas(stream)
        steps = system.clock.partition(stream)
        # The legacy calibration ran the front end in float32 regardless of the network dtype.
        blocks = EventBlocks(stream, steps, H, W, device, torch.float32)
        state = system.representation.init_state(1, H, W, device)
        _, feats, _ = system.representation.encode(state, blocks.block(0, min(n_steps, steps.n_steps)))
        for t in range(int(feats.shape[0])):
            yield feats[t]

    def factory():
        for stream in streams:
            yield inputs_of(stream)

    return calibrate_gains(system.network.backbone, factory, float(cfg["quantile"]), int(cfg["samples_per_channel"]),
                           int(cfg["min_positive"]), float(cfg["gain_min"]), float(cfg["gain_max"]), int(seed) + 4000,
                           carry)


class LayerMonitor(object):
    """Firing rate, silent / saturated neurons and membrane magnitude per backbone layer (accumulated on device)."""

    def __init__(self, v_threshold):
        self.v_threshold = float(v_threshold)
        self.data = {}

    def update(self, info):
        with torch.no_grad():
            for name, spikes, u_pre in zip(LAYER_NAMES, info["spikes"], info["u_pre"]):
                if spikes.dim() == 4:
                    spikes, u_pre = spikes.unsqueeze(0), u_pre.unsqueeze(0)
                active = spikes > 0
                d = self.data.get(name)
                if d is None:
                    d = {"spike_sum": torch.zeros((), device=spikes.device, dtype=torch.float64), "elements": 0,
                         "windows": 0,
                         "channel_spikes": torch.zeros(spikes.shape[2], device=spikes.device, dtype=torch.float64),
                         "neuron_windows": torch.zeros(spikes.shape[2:], device=spikes.device),
                         "u_abs_max": torch.zeros((), device=spikes.device),
                         "big": torch.zeros((), device=spikes.device, dtype=torch.float64)}
                    self.data[name] = d
                d["spike_sum"] += active.sum().double()
                d["elements"] += active.numel()
                d["windows"] += int(spikes.shape[0])
                d["channel_spikes"] += active.sum(dim=(0, 1, 3, 4)).double()
                d["neuron_windows"] += (active.sum(1) > 0).sum(0).float()
                u_abs = u_pre.abs()
                d["u_abs_max"] = torch.maximum(d["u_abs_max"], u_abs.max())
                d["big"] += (u_abs > 10.0 * self.v_threshold).sum().double()

    def summary(self):
        out = {}
        for name, d in self.data.items():
            out[name] = {"firing_rate": float(d["spike_sum"]) / max(d["elements"], 1),
                         "silent_channel_frac": float((d["channel_spikes"] == 0).float().mean()),
                         "always_on_frac": float(((d["neuron_windows"] / max(d["windows"], 1)) > 0.9).float().mean()),
                         "u_abs_max": float(d["u_abs_max"]),
                         "big_membrane_frac": float(d["big"]) / max(d["elements"], 1)}
        return out


def tau_statistics(network):
    out = {}
    for name, block in zip(LAYER_NAMES, network.blocks()):
        neuron = block.neuron
        if not isinstance(neuron, LIF2d):
            continue
        with torch.no_grad():
            tau, beta = neuron.tau().double().cpu(), neuron.beta().double().cpu()
        margin = 0.02 * (neuron.tau_max_ms - neuron.tau_min_ms)
        out[name] = {"tau_min": float(tau.min()), "tau_mean": float(tau.mean()), "tau_max": float(tau.max()),
                     "beta_mean": float(beta.mean()),
                     "near_min_frac": float((tau < neuron.tau_min_ms + margin).double().mean()),
                     "near_max_frac": float((tau > neuron.tau_max_ms - margin).double().mean())}
    return out


def save_checkpoint(path, network, optimizer, epoch, best_val_iou, config, extra=None):
    payload = {"network": network.state_dict(), "optimizer": None if optimizer is None else optimizer.state_dict(),
               "epoch": epoch, "best_val_iou": best_val_iou, "config": config, "format": "speed-1"}
    payload.update(extra or {})
    torch.save(payload, str(path))


def load_checkpoint(path, device):
    ckpt = torch.load(str(path), map_location=device)
    if ckpt.get("format") != "speed-1":
        raise ValueError("%s is not a SPEED checkpoint (convert legacy ones with bridge/convert_checkpoint.py)" % path)
    return ckpt
