"""Slot 10 base variant: stateful truncated BPTT over whole streams, one parameter update per stream.

train_stream: chunks of tbptt_steps (first chunk length random in 1..k so no step is always a chunk start);
representation under no_grad, network + heads + loss with gradients, states detached at chunk borders;
loss divided by the stream's event count; gradient clipping; one optimiser step.
checkpoint_steps > 0 recomputes activations in sub-chunks during backward (large sensors).
Also: gain calibration driver, learning-rate schedule, layer monitor, checkpoint I/O.
"""
import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

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


def forward_checkpointed(network, inputs, states, carry, sub_steps, amp=False):
    """network.forward_chunk over sub-chunks of sub_steps with activation checkpointing: same gradients (up to
    float rounding from smaller conv batches), activation memory of one sub-chunk instead of the whole chunk."""
    names = list(network.heads.outputs)
    dummy = torch.ones(1, requires_grad=True)          # torch 1.9 checkpoint needs an input that requires grad

    def run(x, _dummy, *flat):
        st = list(flat) if flat else None
        with torch.cuda.amp.autocast(enabled=amp):            # also active during the recomputation
            out, new_states, _ = network.forward_chunk(x, st, carry)
        return tuple(out[n] for n in names) + tuple(new_states)

    parts = {n: [] for n in names}
    for a in range(0, int(inputs.shape[0]), int(sub_steps)):
        flat = tuple(states) if states is not None else ()
        res = checkpoint(run, inputs[a:a + int(sub_steps)], dummy, *flat)
        for i, n in enumerate(names):
            parts[n].append(res[i])
        states = list(res[len(names):])
    return {n: torch.cat(v, 0) for n, v in parts.items()}, states


def train_stream(system, loss_fn, stream, optimizer, tbptt_steps, grad_clip, rng, max_steps=None, carry=True,
                 checkpoint_steps=0, update_steps=None, scaler=None):
    """update_steps=None: one update per stream (EV-UAV clips). Otherwise the stream is processed continuously
    (state never reset) with one update every update_steps steps, each normalised by its own event count.
    scaler: torch.cuda.amp.GradScaler for mixed-precision development runs (None = float32, the reference)."""
    amp = scaler is not None and scaler.is_enabled()
    network, rep = system.network, system.representation
    device, dtype = system.device, system.dtype
    H, W = system.canvas(stream)
    steps = system.clock.partition(stream)
    n = steps.n_steps if max_steps is None else min(int(max_steps), steps.n_steps)
    k = int(tbptt_steps)
    span = n if not update_steps else int(update_steps)
    segments = [(a, min(a + span, n)) for a in range(0, n, span)]
    blocks = EventBlocks(stream, steps, H, W, device, dtype)
    rep_state = rep.init_state(1, H, W, device, dtype)
    states = None
    sums = {"loss_sum": 0.0, "mark_sum": 0.0, "intensity_sum": 0.0}
    norms, missing = {}, []
    for seg_start, seg_end in segments:
        chunks = [(seg_start + a, seg_start + b) for a, b in make_chunks(seg_end - seg_start, k, int(rng.randint(1, k + 1)))]
        denominator = float(max(int(steps.bounds[seg_end] - steps.bounds[seg_start]), 1))
        optimizer.zero_grad(set_to_none=True)
        for start, end in chunks:
            blk = blocks.block(start, end)
            with torch.no_grad():
                rep_state, inputs, _ = rep.encode(rep_state, blk)
            if checkpoint_steps and int(inputs.shape[0]) > int(checkpoint_steps):
                outputs, states = forward_checkpointed(network, inputs, states, carry, checkpoint_steps, amp)
            else:
                with torch.cuda.amp.autocast(enabled=amp):
                    outputs, states, _ = network.forward_chunk(inputs, states, carry)
            if amp:                                            # loss and carried state in float32
                outputs = {k: v.float() for k, v in outputs.items()}
                states = [None if st is None else st.float() for st in states]
            ev = blk["events"]
            logits = outputs["mark"][ev["t"], ev["b"], 0, ev["y"], ev["x"]]
            loss, parts = loss_fn(outputs, logits, blk)
            sums["mark_sum"] += parts["mark"]
            sums["intensity_sum"] += parts["intensity"]
            if loss is not None:
                if amp:
                    scaler.scale(loss / denominator).backward()
                else:
                    (loss / denominator).backward()
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise RuntimeError("non-finite loss: %s steps %d-%d" % (stream.name, start, end))
                sums["loss_sum"] += value
            states = detach_states(states)
        if amp:
            scaler.unscale_(optimizer)
        norms = gradient_group_norms(network)
        missing = [name for name, p in network.named_parameters() if p.requires_grad and p.grad is None]
        torch.nn.utils.clip_grad_norm_(network.parameters(), float(grad_clip))
        if amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    sums.update({"events": int(steps.bounds[n] - steps.bounds[0]), "grad_norms": norms, "missing_grads": missing,
                 "updates": len(segments)})
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
                # Reductions written to avoid full-resolution temporaries (int64 sums / abs copies at 1280x720).
                d["spike_sum"] += active.sum().double()
                d["elements"] += active.numel()
                d["windows"] += int(spikes.shape[0])
                d["channel_spikes"] += active.sum(dim=(0, 1, 3, 4)).double()
                d["neuron_windows"] += active.any(1).sum(0).float()
                big = 10.0 * self.v_threshold
                d["u_abs_max"] = torch.maximum(d["u_abs_max"], torch.maximum(u_pre.max(), -u_pre.min()))
                d["big"] += ((u_pre > big).sum() + (u_pre < -big).sum()).double()

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
