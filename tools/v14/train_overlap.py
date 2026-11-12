"""V14 rolling-context training; formulas and limitations in V14_OVERLAP_METHOD.md.

The V5 backbone is unchanged. Observation length, state clock and evaluation
frame length are distinct. Event IDs only address scalar readouts; there is no
event-neighbour graph or event-specific recurrent network. Python 3.8 / torch 1.9.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
# tools/v14/queue.py is a project CLI, not Python's stdlib queue. Executing this
# file directly must not let that sibling shadow torch's import of Queue.
sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from dataset.overlap_windows import (load_overlap_npz, OverlapWindowSource,
                                     compute_overlap_stats)
from model.event_score_revision import EventScoreRevision, ConcatScoreFusion
from model.evspsegnet_stream import calibrate_gains, count_parameters
from model.lif2d_stream import detach_states
from train_stream_v1 import (build_net, seed_everything, plan_streams, stream_pieces,
                             stream_total_windows, stream_valid, tau_statistics)
from utils.comm_budget import CommBudget, window_messages
from utils.stream_common import load_flat_config, linear_epoch_lr, select_subset, subset_due
from utils.stream_metrics import FastRoc, window_confusion, threshold_counts, sweep_summary
from tools.v14.run_experiment import write_json, sha256

DESIGN = "v14-overlap-conditional-r1"
MODES = ("conditional", "mean", "concat")


def validate_config(cfg):
    if int(cfg["window_ms"]) != 2 * int(cfg["stride_ms"]):
        raise ValueError("V14 requires window_ms = 2 * stride_ms (two observations)")
    if int(cfg["total_ms"]) < int(cfg["window_ms"]) or int(cfg["total_ms"]) % int(cfg["stride_ms"]):
        raise ValueError("total_ms must be a stride multiple, at least one full window")
    expected = (int(cfg["total_ms"]) - int(cfg["window_ms"])) // int(cfg["stride_ms"]) + 1
    if int(cfg["n_windows"]) != expected:
        raise ValueError("n_windows does not match rolling observation times")
    if cfg.get("dt_ms", cfg["stride_ms"]) != cfg["stride_ms"]:
        raise ValueError("BIR dt_ms must equal stride_ms, not observation length")
    if cfg["state_mode"] != "carry" or cfg["neuron"] != "tif" or cfg["norm"] != "none":
        raise ValueError("V14 requires V5 BIR, carry and no temporal batch normalization")
    if cfg.get("state_layout", "single_rolling") != "single_rolling":
        raise ValueError("V14 requires one chronological rolling state chain")
    if cfg.get("trainer") != "streams" or cfg.get("execution") != "layer":
        raise ValueError("V14 requires parallel-stream layerwise TBPTT")
    if cfg.get("eval_execution", "step") not in ("step", "layer"):
        raise ValueError("eval_execution must be step or layer")
    if int(cfg.get("accumulate", 1)) != 1 or cfg.get("stream_layout", "greedy") != "greedy":
        raise ValueError("This release preserves greedy whole-record streams, accumulate=1")
    if int(cfg.get("stream_groups", 1)) != 1:
        raise ValueError("Use streams=4 without extra stream grouping in this release")
    if cfg.get("fusion_mode", "conditional") not in MODES or int(cfg.get("fusion_hidden", 8)) != 8:
        raise ValueError("Unsupported fusion architecture")
    if int(cfg["streams"]) < 1 or int(cfg["tbptt_k"]) < 1 or int(cfg["epochs"]) < 1:
        raise ValueError("streams, tbptt_k and epochs must be positive")
    if cfg.get("comm_budget_messages") is None:
        raise ValueError("V14 formal model retains the V5 communication budget")
    for key in ("dual_memory", "subthreshold_residual", "invariant_input", "noise_aug", "comod"):
        if cfg.get(key):
            raise ValueError("Unrelated mechanism enabled: " + key)
    return cfg


def make_models(cfg, device):
    # Quantized BIR states can magnify convolution rounding differences.
    # Use the same FP32 policy for training, selection, evaluation and profiling.
    # Determinism does not imply batched and unbatched kernels are bitwise equal;
    # formal checkpoint selection and reports therefore use online step inference.
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    # Construct the unchanged backbone first: head initialization must not shift
    # the V5 convolution/readout RNG sequence across independent ablation runs.
    clock_cfg = dict(cfg, window_ms=cfg["stride_ms"])
    net = build_net(clock_cfg, device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(cfg["seed"]) + 14000)
        head = (ConcatScoreFusion() if cfg.get("fusion_mode") == "concat"
                else EventScoreRevision(mode=cfg.get("fusion_mode", "conditional"), hidden=8))
    return net, head.to(device)


class Records(object):
    def __init__(self, cfg, split, names=None):
        self.cfg = cfg
        directory = Path(cfg["root"]) / split
        self.paths = ([directory / n for n in names] if names is not None
                      else sorted(directory.glob("*.npz")))
        if not self.paths:
            raise ValueError("No records in " + str(directory))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        c = self.cfg
        return load_overlap_npz(self.paths[index], c["height"], c["width_px"],
                                c["window_ms"], c["stride_ms"], c["total_ms"], c["time_bins"])


class Sources(object):
    def __init__(self, dataset, cfg, q99, device):
        self.dataset, self.cfg, self.q99, self.device = dataset, cfg, q99, device
        self.active = {}

    def get(self, index):
        if index not in self.active:
            self.active[index] = OverlapWindowSource(self.dataset[index], self.cfg, self.q99, self.device)
        return self.active[index]


def assemble(sources, pieces, steps, cfg, device):
    """Keep the original V5 layer-parallel [T,B,C,H,W] training layout."""
    x = torch.zeros(steps, len(pieces), 2 + 2 * int(cfg["time_bins"]),
                    int(cfg["pad_height"]), int(cfg["pad_width"]), device=device)
    keep = torch.ones(steps, len(pieces), device=device)
    keys = ("t", "b", "y", "x", "p", "t_local")
    parts, labels, segments, offset_rows = {k: [] for k in keys}, [], [], 0
    for b, stream in enumerate(pieces):
        for index, lo, hi, offset, new in stream:
            source = sources.get(index)
            xs, ev, lab, ids = source.chunk(lo, hi)
            x[offset:offset + hi - lo, b] = xs[:, 0]
            if new:
                keep[offset, b] = 0
            parts["t"].append(ev["t"] + int(offset))
            parts["b"].append(torch.full_like(ev["t"], b))
            for key in ("y", "x", "p", "t_local"):
                parts[key].append(ev[key])
            labels.append(lab)
            segments.append((index, lo, hi, offset_rows, source.seq, np.asarray(ids)))
            offset_rows += int(lab.numel())
    if labels:
        events = {key: torch.cat(value) for key, value in parts.items()}
        labels = torch.cat(labels)
    else:
        empty = torch.empty(0, dtype=torch.long, device=device)
        events = {key: empty for key in ("t", "b", "y", "x")}
        events.update(p=empty.float(), t_local=empty.float())
        labels = empty.float()
    return x, events, labels, keep, segments


def _index(array, device):
    return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.long, device=device)


def segment_pairs(logits, segment, pending):
    """Exact O(N) ID addressing, with at most one half-window of scalar memory.

    First/last occurrence positions were precomputed once at input loading.
    Numeric pending values cross TBPTT boundaries; no old autograd graph does.
    """
    index, lo, hi, row, seq, ids = segment
    a = int(seq.bounds[lo])
    positions = np.arange(a, a + len(ids), dtype=np.int64)
    first = positions == seq.first_position[ids]
    last = positions == seq.last_position[ids]
    paired = seq.coverage_count[ids] == 2
    second_rows = np.flatnonzero(last & paired)
    prior_positions = seq.first_position[ids[second_rows]] - a
    cross = prior_positions < 0
    r2 = logits[row + _index(second_rows, logits.device)]
    r1 = torch.empty_like(r2)
    if np.any(~cross):
        r1[_index(np.flatnonzero(~cross), logits.device)] = logits[
            row + _index(prior_positions[~cross], logits.device)]
    old = pending.pop(index, None)
    if np.any(cross):
        needed = ids[second_rows[cross]]
        if old is None or not np.array_equal(old[0], needed):
            raise RuntimeError("Missing/misaligned first predictions at TBPTT boundary")
        r1[_index(np.flatnonzero(cross), logits.device)] = old[1]
    elif old is not None and len(old[0]):
        raise RuntimeError("Unconsumed pending events at new segment")
    cache_rows = np.flatnonzero(first & paired & (seq.last_position[ids] >= a + len(ids)))
    if cache_rows.size:
        pending[index] = (ids[cache_rows].copy(), logits[row + _index(cache_rows, logits.device)])
    return r1, r2, second_rows, paired, int(cross.sum())


def classification_loss(logits, labels, segments, pending, head, pos_weight):
    """Eq. task loss: .5 final + .25 each view; singleton weight is one.

    Normalize by the accumulated supervision mass, not duplicated event count.
    Each original event contributes total mass one over the complete record.
    """
    weight = logits.new_tensor([float(pos_weight)])
    per_read = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight, reduction="none")
    loss = logits.sum() * 0
    mass, final_count, cross_count = 0.0, 0, 0
    for segment in segments:
        _, _, _, row, _, ids = segment
        r1, r2, rows, paired, cross = segment_pairs(logits, segment, pending)
        view_weight = np.where(paired, 0.25, 1.0).astype(np.float32)
        loss = loss + (per_read[row:row + len(ids)] * torch.as_tensor(view_weight, device=logits.device)).sum()
        mass += float(view_weight.sum())
        if len(rows):
            rf = head(r1, r2)
            target = labels[row + _index(rows, logits.device)]
            loss = loss + 0.5 * F.binary_cross_entropy_with_logits(rf, target, pos_weight=weight, reduction="sum")
            mass += 0.5 * len(rows)
        final_count += len(rows) + int((~paired).sum())
        cross_count += cross
    return loss, mass, final_count, cross_count


def train_epoch(net, head, dataset, optimizer, cfg, q99, pos_weight, device, rng,
                comm=None, max_chunks=None, plan=None):
    k = int(cfg["tbptt_k"])
    if plan is None:
        plan = plan_streams([int(cfg["n_windows"])] * len(dataset), int(cfg["streams"]), k, rng, "greedy")
    chunks = int(math.ceil(stream_total_windows(plan) / float(k)))
    if max_chunks is not None:
        chunks = min(chunks, int(max_chunks))
    sources, pending, states = Sources(dataset, cfg, q99, device), {}, None
    params = list(net.parameters()) + list(head.parameters())
    total_loss, total_mass, final_count, cross_count, norms = 0., 0., 0, 0, []
    grad_groups = {"backbone": 0., "readout": 0., "fusion": 0.}
    if comm is not None:
        comm.start_epoch()
    t_start = time.perf_counter()
    net.train()
    head.train()
    for chunk in range(chunks):
        pieces = stream_pieces(plan, chunk * k, (chunk + 1) * k)
        x, events, labels, keep, segments = assemble(sources, pieces, k, cfg, device)
        valid = stream_valid(pieces, k, device, x.dtype)
        optimizer.zero_grad(set_to_none=True)
        prev = states
        logits, states, info = net.forward_chunk(x, events, prev, keep=keep, collect=comm is not None)
        raw_loss, mass, nfinal, ncross = classification_loss(logits, labels, segments, pending, head, pos_weight)
        terms = [raw_loss / mass] if mass else []
        if comm is not None:
            penalty = comm.chunk_loss(net, info["u_pre"], prev, keep, 1, valid)
            if penalty is not None:
                terms.append(penalty)
        if terms:
            loss = sum(terms)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite loss at chunk %d" % chunk)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, float(cfg["grad_clip"]))
            if not math.isfinite(float(norm)):
                raise RuntimeError("Nonfinite gradients at chunk %d" % chunk)
            norms.append(float(norm))
            grad_sums = {key: logits.new_zeros(()) for key in grad_groups}
            for name, p in net.named_parameters():
                if p.grad is not None:
                    group = "readout" if name.startswith("readout.") else "backbone"
                    grad_sums[group] += p.grad.detach().norm()
            for p in head.parameters():
                if p.grad is not None:
                    grad_sums["fusion"] += p.grad.detach().norm()
            for key in grad_groups:
                grad_groups[key] += float(grad_sums[key])
            optimizer.step()
        if comm is not None:
            comm.end_update()
        states = detach_states(states)
        pending = {key: (ids, value.detach()) for key, (ids, value) in pending.items()}
        total_loss += float(raw_loss.detach())
        total_mass += mass
        final_count += nfinal
        cross_count += ncross
        for index, _, hi, _, seq, _ in segments:
            if hi == seq.n_windows:
                if index in pending:
                    raise RuntimeError("Record ended with uncommitted predictions")
                sources.active.pop(index, None)
        # Drop full chunk graphs before the next convolution allocation.
        del logits, info, x, events, labels, raw_loss, terms
        if mass or (comm is not None and penalty is not None):
            del loss
        if comm is not None:
            del penalty
        if (chunk + 1) % 25 == 0 or chunk + 1 == chunks:
            print("TRAIN chunks %d/%d committed %d | %.1fs" %
                  (chunk + 1, chunks, final_count, time.perf_counter() - t_start), flush=True)
    if max_chunks is None and pending:
        raise RuntimeError("Epoch ended with pending events")
    return {"loss_per_event": total_loss / max(total_mass, 1.), "supervision_mass": total_mass,
            "committed_events": final_count, "cross_tbptt_pairs": cross_count,
            "chunks": chunks, "updates": len(norms), "grad_norm_mean": float(np.mean(norms)) if norms else 0.,
            "grad_norm_max": max(norms) if norms else 0.,
            "grad_groups_mean": {key: value / max(len(norms), 1) for key, value in grad_groups.items()},
            "comm": comm.summary() if comm is not None else None,
            "train_seconds": time.perf_counter() - t_start}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {"count": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def readout_modes(cfg):
    modes = ["first", "second", "mean"]
    if cfg.get("fusion_mode") == "conditional":
        modes += ["conditional", "calibration_only"]
    elif cfg.get("fusion_mode") == "concat":
        modes += ["concat"]
    return modes


def fuse_modes(head, r1, r2, modes):
    outputs = {"first": r1, "second": r2, "mean": (r1 + r2) * .5}
    if "conditional" in modes:
        outputs["conditional"] = head(r1, r2)
        # Same trained head; intervention, not an independently fitted calibrator.
        outputs["calibration_only"] = head(r1, r1)
    if "concat" in modes:
        outputs["concat"] = head(r1, r2)
    return {key: outputs[key] for key in modes}


@torch.no_grad()
def infer_record(net, head, seq, cfg, q99, device, modes=None, step=False, diagnostics=False):
    """Only completed windows can emit predictions; all ID outputs occur once.

    Chunked inference is for fast validation; only step=True is latency timing.
    Diagnostic modes share one backbone pass and are not independent retrainings.
    """
    modes = readout_modes(cfg) if modes is None else modes
    sync(device)
    source_start = time.perf_counter()
    source = OverlapWindowSource(seq, cfg, q99, device)
    sync(device)
    source_seconds = time.perf_counter() - source_start
    probs = {m: np.full(seq.n_events, np.nan, dtype=np.float32) for m in modes}
    hits = np.zeros(seq.n_events, dtype=np.int8)
    states, pending = None, {}
    durations, net_times, message_counts, event_counts, completions = [], [], [], [], []
    warmup = int(cfg.get("timing_warmup_windows", 20))
    chunk_size = 1 if step else int(cfg.get("eval_chunk", 32))
    for lo in range(0, seq.n_windows, chunk_size):
        hi = min(seq.n_windows, lo + chunk_size)
        if step:
            sync(device)
        t0 = time.perf_counter()
        x, events, _, ids = source.chunk(lo, hi)
        if step:
            sync(device)
        t1 = time.perf_counter()
        prev = states
        if step:
            ev = {key: value for key, value in events.items() if key != "t"}
            logits, states, info = net(x[0], ev, prev, collect=diagnostics)
        else:
            logits, states, info = net.forward_chunk(x, events, prev, collect=diagnostics)
        if step:
            sync(device)
        t2 = time.perf_counter()
        segment = (0, lo, hi, 0, seq, np.asarray(ids))
        r1, r2, second_rows, paired, _ = segment_pairs(logits, segment, pending)
        values = fuse_modes(head, r1, r2, modes)
        single_rows = np.flatnonzero(~paired)
        final_rows = np.concatenate([single_rows, second_rows])
        final_ids = np.asarray(ids)[final_rows]
        # Batch one device-to-host transfer for all diagnostic readouts.
        cpu = torch.stack([torch.sigmoid(torch.cat([logits[_index(single_rows, device)], values[m]]))
                           for m in modes]).cpu().numpy()
        for j, mode in enumerate(modes):
            probs[mode][final_ids] = cpu[j]
        np.add.at(hits, final_ids, 1)
        if step:
            sync(device)
        elapsed = (time.perf_counter() - t0) * 1000.
        if step:
            completions.append(elapsed)
            if lo >= warmup:
                durations.append(elapsed)
                net_times.append((t2 - t1) * 1000.)
                event_counts.append(len(ids))
        if diagnostics:
            memories = ([m.unsqueeze(0) for m in info["u_pre"]] if step else info["u_pre"])
            message_counts.extend(window_messages(net, memories, prev, None).reshape(-1).cpu().tolist())
        del x, logits, info
    if pending or np.any(hits != 1) or any(not np.isfinite(p).all() for p in probs.values()):
        raise RuntimeError("Output coverage/finite-value check failed for " + seq.name)
    timing = {"execution": "step" if step else "layer", "step_ms": durations,
              "network_ms": net_times, "events_per_measured_step": event_counts,
              "all_step_ms": completions, "source_prepare_seconds": source_seconds,
              "input_storage": "whole-record sparse event arrays preloaded; dense representations built per chunk"}
    return probs, timing, message_counts


def metric_row(counts, roc):
    tp, fp, fn = [int(x) for x in counts]
    pd, fa = roc.result() if roc is not None else (None, None)
    return {"iou": tp / float(tp + fp + fn) if tp + fp + fn else None,
            "acc": tp / float(tp + fn) if tp + fn else None, "pd": pd, "fa": fa,
            "tp": tp, "fp": fp, "fn": fn}


@torch.no_grad()
def evaluate(net, head, dataset, cfg, q99, device, full=False, step=None, dump_dir=None):
    if step is None:
        step = cfg.get("eval_execution", "step") == "step"
    was_training = net.training
    net.eval()
    head.eval()
    modes = readout_modes(cfg)
    main = cfg.get("fusion_mode", "conditional")
    counts = {m: np.zeros(3, dtype=np.int64) for m in modes}
    roc = {m: FastRoc(cfg["pd_detT"], cfg["correct_thresh"], cfg["height"], cfg["width_px"])
           if full else None for m in modes}
    sweeps = {m: np.zeros((8, 3), dtype=np.int64) for m in modes}
    per_record, changes, total_events, total_reads = [], {}, 0, 0
    timing_all, net_all, msg_all, acquire_delays, system_delays = [], [], [], [], []
    source_seconds = 0.
    t_start = time.perf_counter()
    for i in range(len(dataset)):
        seq = dataset[i]
        prob, timing, msg = infer_record(net, head, seq, cfg, q99, device, modes, step, full)
        row = {"record": seq.name, "events": seq.n_events, "readouts": {}}
        for m in modes:
            c = np.asarray(window_confusion(prob[m], seq.label, np.float32(cfg["threshold"]))[:3])
            counts[m] += c
            sweeps[m] += threshold_counts(prob[m], seq.label)
            one_roc = None
            if full:
                one_roc = FastRoc(cfg["pd_detT"], cfg["correct_thresh"], cfg["height"], cfg["width_px"])
                for r in (roc[m], one_roc):
                    r.update(seq.t, prob[m], seq.target_id, seq.label, seq.x, seq.y, cfg["threshold"])
            row["readouts"][m] = metric_row(c, one_roc)
        if full:
            target = seq.label == 1
            full_pred = prob[main] >= np.float32(cfg["threshold"])
            for m in modes:
                base = prob[m] >= np.float32(cfg["threshold"])
                cc = changes.setdefault(m, {"corrected": 0, "harmed": 0, "rescued_target": 0,
                                             "suppressed_false_positive": 0})
                cc["corrected"] += int(np.sum((base != target) & (full_pred == target)))
                cc["harmed"] += int(np.sum((base == target) & (full_pred != target)))
                cc["rescued_target"] += int(np.sum(target & ~base & full_pred))
                cc["suppressed_false_positive"] += int(np.sum(~target & base & ~full_pred))
        if dump_dir is not None:
            Path(dump_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(Path(dump_dir) / seq.name),
                                event_id=np.arange(seq.n_events), t=seq.t, x=seq.x, y=seq.y,
                                label=seq.label, target_id=seq.target_id,
                                output_time_ms=seq.output_time_ms,
                                **{"prob_" + m: prob[m] for m in modes})
        total_events += seq.n_events
        total_reads += len(seq.order)
        per_record.append(row)
        timing_all.extend(timing["step_ms"])
        source_seconds += timing["source_prepare_seconds"]
        net_all.extend(timing["network_ms"])
        msg_all.extend(msg)
        acquire_delays.append(seq.output_time_ms - seq.t)
        if step:
            # Replay a single-device arrival queue using measured online steps.
            completion = 0.
            done = []
            for end, cost in zip(seq.window_ends, timing["all_step_ms"]):
                completion = max(completion, float(end)) + cost
                done.append(completion)
            system_delays.append(np.asarray(done)[seq.last_window] - seq.t)
        if full:
            print("EVAL %d/%d %s IoU %.5f" %
                  (i + 1, len(dataset), seq.name, row["readouts"][main]["iou"] or 0.), flush=True)
    result = {"design": DESIGN, "readouts": {m: metric_row(counts[m], roc[m]) for m in modes},
              "threshold": cfg["threshold"], "n_sequences": len(dataset), "events": total_events,
              "event_readouts": total_reads, "per_record": per_record,
              "threshold_sweeps": {m: sweep_summary(sweeps[m]) for m in modes},
              "diagnostics": {"relative_to_final": changes,
                              "ablation_type": "same-checkpoint intervention; not independent retraining"},
              "seconds_including_metrics": time.perf_counter() - t_start,
              "metric_definitions": {"acc": "target-event recall TP/(TP+FN)",
                                     "pd_fa": "original FastRoc, fixed pd_detT; includes original boundary conventions"},
              "timing": {"execution": "step" if step else "layer",
                         "step_ms": distribution(timing_all), "network_ms": distribution(net_all),
                         "state_stride_ms": cfg["stride_ms"], "observation_ms": cfg["window_ms"],
                         "source_prepare_seconds": source_seconds,
                         "input_storage": "whole-record sparse arrays preloaded; not an acquisition driver benchmark",
                         "acquisition_latency_ms": distribution(np.concatenate(acquire_delays)),
                         "single_device_replay_latency_ms": distribution(np.concatenate(system_delays)) if system_delays else None,
                         "note": "layer timing is throughput only; same-pass ablation overhead included in step timing"},
              "communication": {"messages": distribution(msg_all), "budget": cfg["comm_budget_messages"],
                                "violation_fraction": float(np.mean(np.asarray(msg_all) > cfg["comm_budget_messages"])) if msg_all else None},
              "operations": {"readout_linear_macs": total_reads * ((int(cfg["channels"][0]) + 2) * int(cfg["readout_hidden"]) + int(cfg["readout_hidden"])),
                             "fusion_linear_macs": (total_reads - total_events) * (24 if main != "mean" else 0),
                             "scope": "event heads only, excludes shared convolution and scalar operations",
                             "energy_joules": None}}
    result["metrics"] = result["readouts"][main]
    if cfg.get("fusion_mode") == "conditional":
        grid = torch.linspace(-10., 10., 81, device=device)
        _, detail = head(grid, grid, return_details=True)
        result["diagnostics"]["conditional_response"] = {
            "first_logit": grid.cpu().tolist(), "a": detail["a"].cpu().tolist(), "c": detail["c"].cpu().tolist(),
            "note": "functional response of trained head; not empirical correlation or exact B subtraction"}
    if msg_all:
        result["communication"]["messages_per_second_steady"] = float(np.mean(msg_all)) * 1000. / cfg["stride_ms"]
    if device.type == "cuda":
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
    net.train(was_training)
    head.train(was_training)
    return result


def calibrate(net, dataset, cfg, q99, device, out):
    names = select_subset([p.name for p in dataset.paths], int(cfg["calib_sequences"]), int(cfg["seed"]) + 1000)
    calibration_set = Records(cfg, "train", names)
    def factory():
        def windows(seq):
            source = OverlapWindowSource(seq, cfg, q99, device)
            for k in range(min(int(cfg["calib_windows"]), seq.n_windows)):
                x, ev, _, _ = source.window(k)
                yield x, ev
        for i in range(len(calibration_set)):
            yield windows(calibration_set[i])
    reports = calibrate_gains(net, factory, float(cfg["calib_quantile"]),
                              int(cfg["calib_samples_per_channel"]), int(cfg["calib_min_positive"]),
                              float(cfg["calib_gain_min"]), float(cfg["calib_gain_max"]), int(cfg["seed"]) + 4000)
    write_json(out / "calibration.json", {"sequences": names, "layers": reports})
    print("CALIBRATION COMPLETE", flush=True)


def save_checkpoint(path, net, head, optimizer, epoch, best, cfg, stats, comm):
    payload = {"design": DESIGN, "model": net.state_dict(), "fusion": head.state_dict(),
               "optimizer": optimizer.state_dict(), "epoch": epoch, "best_val_iou": best,
               "config": cfg, "stats": stats, "comm_mu": comm.mu if comm else 0.}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def load_checkpoint(path, device):
    # New local torch defaults to weights_only; project checkpoints include numpy
    # statistics. Files are generated by this project, never fetched remotely here.
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def train(cfg, args):
    out, device = Path(args.out), torch.device(args.device)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((out / n).exists() for n in ("last.pt", "metrics.jsonl", "run_config.json")):
        raise ValueError("Output exists; use a new release or --resume")
    seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    dataset = Records(cfg, "train")
    if args.limit_records:
        dataset.paths = dataset.paths[:args.limit_records]
    cfg["_training_scope"] = {"limit_records": args.limit_records, "max_chunks": args.max_chunks}
    cfg["_training_inventory"] = {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                                  for p in dataset.paths}
    source_paths = [Path(__file__), ROOT / "dataset/overlap_windows.py", ROOT / "model/event_score_revision.py",
                    ROOT / "model/evspsegnet_stream.py", ROOT / "model/lif2d_stream.py", ROOT / "utils/comm_budget.py",
                    ROOT / "train_stream_v1.py", ROOT / "utils/stream_metrics.py"]
    cfg["_source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    stats_path = out / "train_stats.json"
    if args.resume:
        ckpt = load_checkpoint(out / "last.pt", torch.device("cpu"))
        if ckpt["design"] != DESIGN or ckpt["config"] != cfg:
            raise ValueError("Resume config/design mismatch")
        stats = ckpt["stats"]
    elif stats_path.exists():
        raise ValueError("Stale stats without checkpoint; choose a clean output")
    else:
        stats = compute_overlap_stats(dataset.paths, cfg)
        write_json(stats_path, stats)
    q99, pos_weight = np.asarray(stats["q99"], dtype=np.float32), float(stats["pos_weight"])
    net, head = make_models(cfg, device)
    optimizer = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=float(cfg["lr"]))
    comm = CommBudget.from_config(cfg, net)
    start, best, history = 0, -float("inf"), []
    if args.resume:
        net.load_state_dict(ckpt["model"])
        head.load_state_dict(ckpt["fusion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        comm.mu = ckpt["comm_mu"]
        start, best = int(ckpt["epoch"]) + 1, float(ckpt["best_val_iou"])
        history = [json.loads(line) for line in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        history = [row for row in history if row["epoch"] < start]
        (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history), encoding="utf-8")
        del ckpt
    else:
        write_json(out / "run_config.json", {"design": DESIGN, "config": cfg,
                   "backbone_parameters": count_parameters(net), "fusion_parameters": count_parameters(head),
                   "precision": "float32; CUDA matmul and cuDNN TF32 disabled",
                   "torch": torch.__version__, "python": sys.version,
                   "data": {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in dataset.paths},
                   "source": {str(p.relative_to(ROOT)): sha256(p) for p in
                              [Path(__file__), ROOT / "dataset/overlap_windows.py", ROOT / "model/event_score_revision.py",
                               ROOT / "model/evspsegnet_stream.py", ROOT / "model/lif2d_stream.py", ROOT / "utils/comm_budget.py"]}})
        calibrate(net, dataset, cfg, q99, device, out)
    validation = Records(cfg, "val")
    if args.limit_records:
        validation.paths = validation.paths[:args.limit_records]
    subset = select_subset([p.name for p in dataset.paths], int(cfg["train_subset_size"]), int(cfg["seed"]) + 2000)
    subset_data = Records(cfg, "train", subset)
    for epoch in range(start, int(cfg["epochs"])):
        t0 = time.perf_counter()
        lr = (float(cfg["lr"]) if int(cfg["epochs"]) == 1 else
              linear_epoch_lr(epoch, int(cfg["epochs"]), float(cfg["lr"]), float(cfg["lr_end"])))
        for group in optimizer.param_groups:
            group["lr"] = lr
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = train_epoch(net, head, dataset, optimizer, cfg, q99, pos_weight, device,
                             np.random.RandomState(int(cfg["seed"]) * 1000 + epoch + 7), comm,
                             max_chunks=args.max_chunks)
        val = evaluate(net, head, validation, cfg, q99, device)
        sub = evaluate(net, head, subset_data, cfg, q99, device) if subset_due(epoch, int(cfg["epochs"]), int(cfg["train_subset_every"])) else None
        record = dict(result, epoch=epoch, lr=lr, val_iou=val["metrics"]["iou"], val_acc=val["metrics"]["acc"],
                      val_readouts=val["readouts"], train_subset_readouts=sub["readouts"] if sub else None,
                      epoch_seconds=time.perf_counter() - t0, tau=tau_statistics(net),
                      peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None)
        if record["val_iou"] is None or not math.isfinite(record["val_iou"]):
            raise RuntimeError("Validation IoU is not finite")
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        history.append(record)
        if record["val_iou"] > best:
            best = record["val_iou"]
            save_checkpoint(out / ("best_val_iou_seed%d.pt" % cfg["seed"]), net, head, optimizer, epoch, best, cfg, stats, comm)
        save_checkpoint(out / "last.pt", net, head, optimizer, epoch, best, cfg, stats, comm)
        write_json(out / "summary.json", {"design": DESIGN, "epochs_completed": epoch + 1, "epochs_requested": cfg["epochs"],
                   "best_val_iou": best,
                   "complete": epoch + 1 == int(cfg["epochs"]) and args.limit_records is None and args.max_chunks is None,
                   "training_scope": cfg["_training_scope"],
                   "probe_only": args.limit_records is not None or args.max_chunks is not None,
                   "train_seconds_total": sum(r["train_seconds"] for r in history),
                   "epoch_seconds_total": sum(r["epoch_seconds"] for r in history),
                   "seed": cfg["seed"], "fusion_mode": cfg["fusion_mode"]})
        print("EPOCH %d/%d train %.1fs total %.1fs loss %.5f val IoU %.5f ACC %.5f comm_mu %.4f" %
              (epoch + 1, cfg["epochs"], result["train_seconds"], record["epoch_seconds"], result["loss_per_event"],
               record["val_iou"], record["val_acc"], comm.mu), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v14/overlap_conditional.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("train", "eval", "stats"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--step", action="store_true", help="Online inference timing; no batched lookahead")
    parser.add_argument("--dump-predictions", action="store_true")
    args = parser.parse_args()
    for name in ("limit_records", "max_chunks", "epochs"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error("--%s must be positive" % name.replace("_", "-"))
    cfg = load_flat_config(args.config)
    if args.data_root:
        cfg["root"] = str(Path(args.data_root).resolve())
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    cfg["save_root"] = str(Path(args.out).resolve())
    cfg["stats_path"] = str(Path(args.out).resolve() / "train_stats.json")
    validate_config(cfg)
    if args.mode == "train":
        train(cfg, args)
    elif args.mode == "stats":
        destination = Path(args.out) / "stats_preview.json"
        if destination.exists() and not args.overwrite:
            raise ValueError("Stats preview exists")
        write_json(destination, compute_overlap_stats(Records(cfg, "train").paths, cfg))
    else:
        device = torch.device(args.device)
        if not args.checkpoint:
            raise ValueError("--checkpoint required for eval")
        ckpt = load_checkpoint(args.checkpoint, torch.device("cpu"))
        if ckpt.get("design") != DESIGN:
            raise ValueError("Not a V14 overlap checkpoint")
        original = ckpt["config"]
        # Architecture, thresholds and time bases come from the trained model.
        for key in original:
            if not key.startswith("_") and key not in ("root", "save_root", "stats_path") and cfg.get(key) != original[key]:
                raise ValueError("Evaluation config differs from checkpoint: " + key)
        net, head = make_models(cfg, device)
        net.load_state_dict(ckpt["model"])
        head.load_state_dict(ckpt["fusion"])
        destination = Path(args.out) / ("eval_%s%s.json" % (args.split, "_step" if args.step else ""))
        if destination.exists() and not args.overwrite:
            raise ValueError("Evaluation exists: " + str(destination))
        dataset = Records(cfg, args.split)
        if args.limit_records:
            dataset.paths = dataset.paths[:args.limit_records]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = evaluate(net, head, dataset, cfg, np.asarray(ckpt["stats"]["q99"], dtype=np.float32), device,
                          full=True, step=True if args.step else None,
                          dump_dir=Path(args.out) / ("predictions_" + args.split) if args.dump_predictions else None)
        result.update(checkpoint_sha256=sha256(args.checkpoint), selected_epoch=int(ckpt["epoch"]) + 1,
                      seed=cfg["seed"], split=args.split)
        write_json(destination, result)
        print(json.dumps(result["readouts"], indent=2), flush=True)
        print("EVALUATION COMPLETE " + str(destination), flush=True)


if __name__ == "__main__":
    main()
