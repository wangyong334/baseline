"""V15 native-window delayed feature readout, Python 3.8 / PyTorch 1.9.

Each 50 ms input drives the carry backbone exactly once. Training uses the V5
four-stream, layer-parallel TBPTT implementation. A one-window feature cache
is detached at chunk boundaries and never crosses a recording boundary.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
# Executing this file must not let the sibling profile.py shadow the stdlib
# module loaded by cProfile / newer torch optimizers.
sys.path[:] = [p for p in sys.path
               if Path(p or os.getcwd()).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from dataset.ev_uav_stream import EvUAVStream
from dataset.stream_source import make_window_source
from dataset import stream_windows as sw
from model.evspsegnet_stream import count_parameters
from model.observation_revision import ObservationRevision
from tools.v15.runtime_check import check_revision_backward
from train_stream_v1 import (build_net, seed_everything, load_stats, run_calibration,
                             plan_streams, stream_pieces, stream_total_windows,
                             StreamSources, assemble_streams, stream_valid, tau_statistics)
from utils.comm_budget import CommBudget, window_messages
from utils.stream_common import load_flat_config, linear_epoch_lr
from utils.stream_metrics import FastRoc, window_confusion, threshold_counts, sweep_summary

DESIGN = "v15-native-observation-revision-r3"
MODES = ("observation", "deep", "causal")
SOURCE_FILES = ("tools/v15/train.py", "train_stream_v1.py", "dataset/ev_uav_stream.py",
                "dataset/stream_windows.py", "dataset/stream_source.py",
                "model/evspsegnet_stream.py", "model/lif2d_stream.py",
                "model/observation_revision.py", "tools/v15/runtime_check.py", "utils/comm_budget.py",
                "utils/stream_common.py", "utils/stream_metrics.py")


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_plain(value), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def validate_config(cfg):
    if int(cfg["window_ms"]) != 50 or cfg.get("stride_ms", 50) != 50:
        raise ValueError("V15 uses nonoverlapping native 50 ms windows")
    if cfg["state_mode"] != "carry" or cfg["neuron"] != "tif" or cfg["norm"] != "none":
        raise ValueError("V15 requires the V5 BIR carry backbone without temporal normalization")
    if cfg.get("trainer") != "streams" or cfg.get("execution") != "layer":
        raise ValueError("Training must use parallel-stream, layerwise TBPTT")
    if cfg.get("eval_execution", "step") != "step":
        raise ValueError("Formal validation/evaluation must use FP32 step inference")
    if int(cfg.get("accumulate", 1)) != 1 or cfg.get("stream_layout", "greedy") != "greedy":
        raise ValueError("This release preserves greedy whole-record streams and accumulate=1")
    if int(cfg.get("stream_groups", 1)) != 1:
        raise ValueError("Use one layer-parallel stream group")
    if cfg.get("comm_budget_messages") is None:
        raise ValueError("All V15 arms retain the V5 communication budget")
    if cfg.get("revision_source", "observation") not in MODES:
        raise ValueError("Unknown revision_source")
    for key in ("streams", "tbptt_k", "epochs", "n_windows", "revision_pair_batch"):
        if int(cfg[key]) < 1:
            raise ValueError(key + " must be positive")
    patch = int(cfg["revision_patch"])
    if patch < 1:
        raise ValueError("revision_patch must be positive")
    if cfg["pad_height"] % patch or cfg["pad_width"] % patch:
        raise ValueError("Padded canvas must be divisible by revision_patch")
    if not (0 < int(cfg["height"]) <= int(cfg["pad_height"]) and
            0 < int(cfg["width_px"]) <= int(cfg["pad_width"])):
        raise ValueError("Sensor dimensions must be positive and fit the padded canvas")
    for key in ("dual_memory", "subthreshold_residual", "invariant_input", "noise_aug", "comod"):
        if cfg.get(key):
            raise ValueError("Unrelated mechanism enabled: " + key)
    return cfg


def make_models(cfg, device):
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    net = build_net(cfg, device)
    head = None
    if cfg["revision_source"] != "causal":
        # Backbone/readout initialization stays identical across ablation arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(cfg["seed"]) + 15000)
            head = ObservationRevision(
                channels=int(cfg["channels"][0]), width=int(cfg["revision_width"]),
                heads=int(cfg["revision_heads"]), patch=int(cfg["revision_patch"]),
                radius=int(cfg["revision_radius"]), source=cfg["revision_source"],
                pair_batch=int(cfg["revision_pair_batch"]),
                sensor_hw=(int(cfg["height"]), int(cfg["width_px"])))
        head = head.to(device)
    return net, head


class Records(EvUAVStream):
    @property
    def paths(self):
        return [Path(self.directory) / name for name in self.names]


def records(cfg, split, limit=None):
    dataset = Records(cfg["root"], split, cfg)
    if limit is not None:
        dataset.names = dataset.names[:int(limit)]
    return dataset


def file_inventory(paths):
    """Experiment identity by file name/size/mtime; not a content digest."""
    return {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in paths}


def validate_statistics(stats, cfg):
    """Refuse stale normalization/class weights even when their shape matches."""
    expected_config = {key: cfg[key] for key in
                       ("window_ms", "n_windows", "time_bins", "height", "width_px",
                        "pad_height", "pad_width")}
    if stats.get("config_used") != expected_config:
        raise ValueError("Statistics data configuration mismatch")
    expected_source = {"statistics_split": "train", "root": str(Path(cfg["root"]).resolve()),
                       "records": file_inventory(records(cfg, "train").paths)}
    if stats.get("provenance") != expected_source:
        raise ValueError("Statistics training-data provenance differs; regenerate train_stats.json")
    q99 = np.asarray(stats.get("q99", []), dtype=np.float64)
    if q99.shape != (int(cfg["time_bins"]) * 2 + 2,) or not np.all(np.isfinite(q99)) or np.any(q99 <= 0):
        raise ValueError("Statistics q99 must contain finite positive values for every channel")
    npos, nneg = float(stats.get("train_pos", 0)), float(stats.get("train_neg", 0))
    if not (math.isfinite(npos) and math.isfinite(nneg) and npos > 0 and nneg > 0):
        raise ValueError("Statistics require finite positive train foreground/background counts")
    expected_weight = sw.pos_weight_from_counts(npos, nneg, cfg["pos_weight_cap"])
    if (stats.get("pos_weight_cap") != cfg["pos_weight_cap"] or
            not math.isclose(float(stats.get("pos_weight", float("nan"))), expected_weight,
                             rel_tol=1e-7, abs_tol=1e-8)):
        raise ValueError("Statistics positive-class weight differs from training configuration")


def compute_stats(cfg):
    """Only train contributes q99/class weights; other splits are audited."""
    from tools.stream_train_stats import audit_and_collect
    audits = {}
    audits["train"], hist, npos, nneg = audit_and_collect(cfg, "train", True)
    for split in ("val", "test"):
        audits[split] = audit_and_collect(cfg, split, False)[0]
    paths = records(cfg, "train").paths
    return {"q99": [sw.quantile_from_histogram(h, 0.99) for h in hist],
            "channel_names": sw.channel_names(cfg["time_bins"]),
            "train_pos": npos, "train_neg": nneg,
            "pos_weight": sw.pos_weight_from_counts(npos, nneg, cfg["pos_weight_cap"]),
            "pos_weight_cap": cfg["pos_weight_cap"], "audit": audits,
            "config_used": {key: cfg[key] for key in
                           ("window_ms", "n_windows", "time_bins", "height", "width_px",
                            "pad_height", "pad_width")},
            "provenance": {"statistics_split": "train", "root": str(Path(cfg["root"]).resolve()),
                           "records": file_inventory(paths)}}


def encode_observations(head, info, cfg):
    if head is None:
        return None
    source = info["features"] if cfg["revision_source"] == "deep" else info["observation"]
    if source.dim() == 5:
        t, b = source.shape[:2]
        compact = head.encode_observation(source.reshape((-1,) + tuple(source.shape[2:])))
        return compact.reshape((t, b) + tuple(compact.shape[1:]))
    return head.encode_observation(source)


def _event_slice(events, start, end):
    return {key: events[key][start:end] for key in ("x", "y", "p", "t_local")}


def _detach_frame(frame):
    # clone is essential: a detached view would retain the entire [K,B] storage.
    return dict(frame, feature=frame["feature"].detach().clone(),
                observation=None if frame["observation"] is None else frame["observation"].detach().clone(),
                events={k: v.detach().clone() for k, v in frame["events"].items()},
                labels=frame["labels"].detach().clone())


def _detach_state_copy(state):
    """Do not retain temporal-stack storage through a last-state view."""
    if state is None:
        return None
    if isinstance(state, list):
        return [_detach_state_copy(value) for value in state]
    if isinstance(state, tuple):
        return tuple(_detach_state_copy(value) for value in state)
    return state.detach().clone()


def chunk_targets(features, compact, events, labels, sources, pieces, pending,
                  chunk_index, causal=False):
    """Create old/new-window pairs by recording identity, never by event matching.

    Returns [(old_frame, new_compact_or_None, cross_chunk), ...]. A valid empty
    next window is a pair; only a true recording tail lacks a future window.
    """
    targets, row = [], 0
    counts = {"input_events": int(labels.numel()), "input_windows": 0,
              "valid_empty_windows": 0, "paired_windows": 0, "tail_windows": 0,
              "paired_events": 0, "tail_events": 0, "crosschunk_events": 0,
              "crosschunk_pairs": 0, "valid_empty_future_windows": 0}
    for stream_id, stream in enumerate(pieces):
        for index, lo, hi, offset, new in stream:
            bounds = sources.bounds[index]
            if new and stream_id in pending:
                raise RuntimeError("Pending features would cross a recording boundary")
            if new and lo != 0:
                raise RuntimeError("V15 only supports whole-record greedy streams")
            start = int(bounds[lo])
            for window in range(lo, hi):
                a = row + int(bounds[window]) - start
                b = row + int(bounds[window + 1]) - start
                pos = int(offset) + window - lo
                frame = {"record": index, "window": window, "chunk": chunk_index,
                         "row": a, "end": b,
                         "feature": features[pos, stream_id],
                         "observation": None if compact is None else compact[pos, stream_id],
                         "events": _event_slice(events, a, b), "labels": labels[a:b]}
                counts["input_windows"] += 1
                counts["valid_empty_windows"] += int(a == b)
                if causal:
                    targets.append((frame, None, False))
                    continue
                old = pending.pop(stream_id, None)
                if old is not None:
                    if old["record"] != index or old["window"] + 1 != window:
                        raise RuntimeError("Missing/nonconsecutive old window in delayed cache")
                    cross = old["chunk"] != chunk_index
                    targets.append((old, frame["observation"], cross))
                    n = int(old["labels"].numel())
                    counts["paired_windows"] += 1
                    counts["paired_events"] += n
                    counts["crosschunk_pairs"] += int(cross)
                    counts["crosschunk_events"] += n if cross else 0
                    counts["valid_empty_future_windows"] += int(a == b)
                elif window != 0:
                    raise RuntimeError("Old window disappeared at a TBPTT boundary")
                if window == len(bounds) - 2:
                    targets.append((frame, None, False))
                    counts["tail_windows"] += 1
                    counts["tail_events"] += b - a
                else:
                    pending[stream_id] = frame
            row += int(bounds[hi] - bounds[lo])
    if row != labels.numel():
        raise RuntimeError("Event slicing disagrees with assemble_streams")
    counts["final_events"] = sum(int(old["labels"].numel()) for old, _, _ in targets)
    return targets, counts


def pack_events(frames):
    events = {key: torch.cat([frame["events"][key] for frame in frames])
              for key in ("x", "y", "p", "t_local")}
    events["b"] = torch.cat([torch.full_like(frame["events"]["x"], i)
                              for i, frame in enumerate(frames)])
    return events, torch.cat([frame["labels"] for frame in frames])


def classification_loss(net, head, targets, pos_weight, pair_batch,
                        current_logits, current_labels):
    """Immediate causal supervision plus delayed final supervision.

    With revision, .5 BCE is charged when the current feature is computed and
    .5 BCE when that event is finalized. Normalize each update by supervision
    mass (.5 N_current + .5 N_final). Over a complete recording each event has
    total mass one. In particular K=1 still trains the causal backbone.
    Only the delayed old-feature gradient is truncated at a TBPTT boundary.
    The causal-only arm uses one ordinary current-window BCE.
    """
    weight = current_logits.new_tensor([float(pos_weight)])
    causal = F.binary_cross_entropy_with_logits(current_logits, current_labels,
                                                pos_weight=weight, reduction="sum")
    current_count = int(current_labels.numel())
    if head is None:
        causal_value = float(causal.detach())
        return causal, float(current_count), {
            "causal_bce_sum": causal_value, "final_bce_sum": causal_value,
            "causal_events": current_count, "final_events": current_count}
    total = 0.5 * causal
    revised_sum, final_count = causal.detach().new_zeros(()), 0
    active = [target for target in targets if target[0]["labels"].numel()]
    # Pair/fallback grouping avoids evaluating attention on recording tails.
    for paired in (True, False):
        group = [target for target in active if (target[1] is not None) == paired]
        for start in range(0, len(group), pair_batch):
            batch = group[start:start + pair_batch]
            frames = [item[0] for item in batch]
            if paired:
                feature = torch.stack([frame["feature"] for frame in frames])
                ev, lab = pack_events(frames)
                old_obs = torch.stack([frame["observation"] for frame in frames])
                new_obs = torch.stack([item[1] for item in batch])
                revised = head.revise(feature, old_obs, new_obs)
                final_logits = net.read_features(revised, ev)
            else:
                # A recording tail occurs in this chunk; reuse its current
                # causal readout rather than perform a duplicate MLP operation.
                final_logits = torch.cat([current_logits[frame["row"]:frame["end"]] for frame in frames])
                lab = torch.cat([frame["labels"] for frame in frames])
            final = F.binary_cross_entropy_with_logits(final_logits, lab, pos_weight=weight, reduction="sum")
            total = total + 0.5 * final
            # Keep scalar diagnostics on-device across pair microbatches.
            # Converting each batch to float would serialize the CUDA stream.
            revised_sum = revised_sum + final.detach()
            final_count += int(lab.numel())
    causal_value, revised_value = torch.stack((causal.detach(), revised_sum)).cpu().tolist()
    return total, 0.5 * (current_count + final_count), {
        "causal_bce_sum": causal_value, "final_bce_sum": revised_value,
        "causal_events": current_count, "final_events": final_count}


def train_epoch(net, head, dataset, optimizer, cfg, q99, pos_weight, device, rng,
                comm=None, max_chunks=None, plan=None):
    k = int(cfg["tbptt_k"])
    if plan is None:
        plan = plan_streams([int(cfg["n_windows"])] * len(dataset), int(cfg["streams"]),
                            k, rng, "greedy")
    all_chunks = int(math.ceil(stream_total_windows(plan) / float(k)))
    chunks = all_chunks if max_chunks is None else min(all_chunks, int(max_chunks))
    sources = StreamSources(dataset, cfg, q99, device, torch.float32)
    pending, states, totals = {}, None, {}
    params = list(net.parameters()) + ([] if head is None else list(head.parameters()))
    loss_sum, causal_sum, final_sum, supervision_mass, norms, updates = 0., 0., 0., 0., [], 0
    if comm is not None:
        comm.start_epoch()
    net.train()
    if head is not None:
        head.train()
    start_time = time.perf_counter()
    for chunk in range(chunks):
        pieces = stream_pieces(plan, chunk * k, (chunk + 1) * k)
        x, events, labels, keep = assemble_streams(sources, pieces, k, cfg, device, torch.float32)
        valid = stream_valid(pieces, k, device, x.dtype)
        optimizer.zero_grad(set_to_none=True)
        prev = states
        _, states, info = net.forward_chunk(x, events, prev, keep=keep,
                                           collect=comm is not None,
                                           return_features=True, skip_readout=True)
        compact = encode_observations(head, info, cfg)
        current_logits = net.read_features(info["features"], events)
        targets, counts = chunk_targets(info["features"], compact, events, labels,
                                         sources, pieces, pending, chunk, head is None)
        raw, mass, losses = classification_loss(net, head, targets, pos_weight,
                                                int(cfg["revision_pair_batch"]), current_logits, labels)
        terms = [raw / float(mass)] if mass else []
        loss = penalty = None
        if comm is not None:
            penalty = comm.chunk_loss(net, info["u_pre"], prev, keep, 1, valid)
            if penalty is not None:
                terms.append(penalty)
        if terms:
            loss = sum(terms)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite loss at chunk %d" % chunk)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["grad_clip"])))
            if not math.isfinite(norm):
                raise RuntimeError("Nonfinite gradients at chunk %d" % chunk)
            norms.append(norm)
            optimizer.step()
            updates += 1
        if comm is not None:
            comm.end_update()
        states = _detach_state_copy(states)
        pending = {key: _detach_frame(frame) for key, frame in pending.items()}
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
        loss_sum += float(raw.detach())
        supervision_mass += mass
        causal_sum += losses["causal_bce_sum"]
        final_sum += losses["final_bce_sum"]
        for stream in pieces:
            for index, lo, hi, offset, new in stream:
                if hi == len(sources.bounds[index]) - 1:
                    sources.release(index)
        if (chunk + 1) % int(cfg.get("log_every_chunks", 25)) == 0 or chunk + 1 == chunks:
            print("TRAIN chunks %d/%d committed %d | %.1fs" %
                  (chunk + 1, chunks, totals.get("final_events", 0),
                   time.perf_counter() - start_time), flush=True)
        # Assignment evaluates the next forward before replacing its old result.
        # Explicitly release the completed [K,B] graph/storage before that call.
        del info, compact, targets, raw, terms, loss, penalty, prev, x, events, labels, keep, valid, current_logits
    pending_events = sum(int(frame["labels"].numel()) for frame in pending.values())
    if chunks == all_chunks:
        if pending or totals.get("input_events", 0) != totals.get("final_events", 0):
            raise RuntimeError("Full epoch did not finalize every event exactly once")
        if supervision_mass != totals.get("input_events", 0):
            raise RuntimeError("Complete epoch supervision mass differs from original event count")
    return dict(totals, loss_per_event=loss_sum / max(supervision_mass, 1.),
                supervision_mass=supervision_mass,
                causal_bce_per_event=causal_sum / max(totals.get("input_events", 0), 1),
                final_bce_per_event=final_sum / max(totals.get("final_events", 0), 1),
                train_seconds=time.perf_counter() - start_time, chunks=chunks, steps=updates,
                pending_events_at_truncation=pending_events, pending_windows_at_truncation=len(pending),
                grad_norm_mean=float(np.mean(norms)) if norms else None,
                clipped_steps=sum(n > float(cfg["grad_clip"]) for n in norms),
                comm=comm.summary() if comm else None)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    return {"count": int(values.size), "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": float(values.max())}


@torch.no_grad()
def infer_record(net, head, seq, cfg, q99, device, modes=None, step=True, diagnostics=False):
    """Strict online replay. modes=('final',) avoids auxiliary causal readouts.

    Returns (probabilities_by_mode, timing, detail). Output arrays use original
    NPZ event order. Only the last recording window lacks future observation.
    Input construction is preloaded-record replay, not a live camera benchmark.
    """
    if not step:
        raise ValueError("V15 formal inference only supports chronological step execution")
    modes = tuple(("final", "causal") if modes is None else modes)
    if not modes or any(mode not in ("final", "causal") for mode in modes):
        raise ValueError("Modes must be final and/or causal")
    _sync(device)
    source_start = time.perf_counter()
    source = make_window_source(seq, cfg, q99, device, torch.float32)
    _sync(device)
    prepare_seconds = time.perf_counter() - source_start
    pred = {mode: np.empty(seq.n_events, dtype=np.float32) for mode in modes}
    seen = {mode: np.zeros(seq.n_events, dtype=np.uint8) for mode in modes}
    states, old = None, None
    all_ms, network_ms, prepare_ms, messages = [], [], [], []
    diag = {"null_probability": [], "gate_mean": [], "correction_rms": []}
    counts = {"final_events": 0, "paired_events": 0, "tail_events": 0,
              "paired_windows": 0, "tail_windows": 0, "valid_empty_windows": 0,
              "valid_empty_future_windows": 0, "crosschunk_events": 0}
    reads = 0

    def put(mode, ids, logits):
        if mode in pred:
            if np.any(seen[mode][ids]):
                raise RuntimeError("Event finalized more than once")
            probabilities = torch.sigmoid(logits).float().cpu().numpy()
            # NaN comparisons otherwise become false and produce plausible but
            # invalid IoU/ACC, potentially selecting a corrupt checkpoint.
            if not np.all(np.isfinite(probabilities)):
                raise RuntimeError("Nonfinite %s predictions in %s window %d" %
                                   (mode, seq.name, window))
            pred[mode][ids] = probabilities
            seen[mode][ids] += 1

    for window in range(seq.n_windows):
        _sync(device)
        t0 = time.perf_counter()
        x, ev, lab, ids = source.window(window)
        _sync(device)
        t1 = time.perf_counter()
        prev = states
        _, states, info = net(x, ev, prev, return_features=True, skip_readout=True,
                              collect=diagnostics)
        compact = encode_observations(head, info, cfg)
        feature = info["features"]
        causal = None
        if "causal" in modes or head is None or window == seq.n_windows - 1:
            causal = net.read_features(feature, ev)
            reads += int(lab.numel())
            put("causal", ids, causal)
        if head is None:
            put("final", ids, causal)
            counts["final_events"] += int(lab.numel())
        else:
            if old is not None:
                old_feature, old_compact, old_ev, old_ids = old
                if len(old_ids):
                    revised = head.revise(old_feature, old_compact, compact,
                                          return_info=diagnostics)
                    if diagnostics:
                        revised, detail = revised
                        for key in diag:
                            value = detail.get("null_mass_mean", detail[key]) if key == "null_probability" else detail[key]
                            diag[key].append(float(value.mean().cpu()))
                    if "final" in modes:
                        logits = net.read_features(revised, old_ev)
                        reads += len(old_ids)
                        put("final", old_ids, logits)
                counts["paired_windows"] += 1
                counts["paired_events"] += len(old_ids)
                counts["final_events"] += len(old_ids)
                counts["valid_empty_future_windows"] += int(lab.numel() == 0)
            if window == seq.n_windows - 1:
                put("final", ids, causal)
                counts["tail_windows"] += 1
                counts["tail_events"] += int(lab.numel())
                counts["final_events"] += int(lab.numel())
                old = None
            else:
                old = (feature, compact, ev, ids)
        counts["valid_empty_windows"] += int(lab.numel() == 0)
        _sync(device)
        t2 = time.perf_counter()
        all_ms.append((t2 - t0) * 1000.)
        network_ms.append((t2 - t1) * 1000.)
        prepare_ms.append((t1 - t0) * 1000.)
        if diagnostics:
            memories = [value.unsqueeze(0) for value in info["u_pre"]]
            msg = window_messages(net, memories, prev, None)
            messages.append(float(msg[0, 0].cpu()))
    for mode, value in seen.items():
        if not np.all(value == 1):
            raise RuntimeError("Inference missed events in " + mode)
    if counts["final_events"] != seq.n_events or old is not None:
        raise RuntimeError("Delayed inference failed to flush recording tail")
    event_window = np.empty(seq.n_events, dtype=np.int64)
    for window in range(seq.n_windows):
        event_window[seq.window_index(window)] = window
    release_window = (event_window if head is None else
                      np.minimum(event_window + 1, seq.n_windows - 1))
    output_time = (release_window + 1) * float(cfg["window_ms"])
    warmup = min(int(cfg.get("timing_warmup_windows", 20)), len(all_ms))
    timing = {"all_step_ms": all_ms, "network_ms": network_ms, "prepare_ms": prepare_ms,
              "step_ms": all_ms[warmup:], "source_prepare_seconds": prepare_seconds,
              "execution": "step", "includes_auxiliary_causal_readout": "causal" in modes and head is not None}
    detail = {"counts": counts, "event_readouts": reads, "release_window": release_window,
              "output_time_ms": output_time, "messages": messages,
              "revision": {key: distribution(value) for key, value in diag.items()}}
    return pred, timing, detail


def metric_row(counts, roc=None):
    tp, fp, fn = (int(value) for value in counts)
    pd, fa = roc.result() if roc is not None else (None, None)
    return {"iou": tp / float(tp + fp + fn) if tp + fp + fn else None,
            "acc": tp / float(tp + fn) if tp + fn else None,
            "pd": pd, "fa": fa, "tp": tp, "fp": fp, "fn": fn}


@torch.no_grad()
def evaluate(net, head, dataset, cfg, q99, device, full=False, dump_dir=None):
    was_training = net.training
    head_training = head.training if head is not None else False
    net.eval()
    if head is not None:
        head.eval()
    try:
        if not len(dataset):
            raise ValueError("Evaluation split contains no recordings")
        return _evaluate_impl(net, head, dataset, cfg, q99, device, full, dump_dir)
    finally:
        net.train(was_training)
        if head is not None:
            head.train(head_training)


def _evaluate_impl(net, head, dataset, cfg, q99, device, full, dump_dir):
    modes = ("final", "causal") if head is not None else ("final",)
    counts = {mode: np.zeros(3, dtype=np.int64) for mode in modes}
    roc = {mode: FastRoc(cfg["pd_detT"], cfg["correct_thresh"], cfg["height"], cfg["width_px"])
           if full else None for mode in modes}
    sweeps = {mode: np.zeros((8, 3), dtype=np.int64) for mode in modes}
    aggregate, per_record, timing_rows, network_rows, msg_rows, latencies = {}, [], [], [], [], []
    replay_latencies, read_count, source_seconds = [], 0, 0.
    changes = {"corrected": 0, "harmed": 0, "rescued_target": 0, "suppressed_false_positive": 0}
    revision_rows = []
    start_time = time.perf_counter()
    for i in range(len(dataset)):
        seq = dataset[i]
        prob, timing, detail = infer_record(net, head, seq, cfg, q99, device, modes,
                                            diagnostics=full)
        row = {"record": seq.name, "events": seq.n_events, "readouts": {}}
        for mode in modes:
            count = np.asarray(window_confusion(prob[mode], seq.label, np.float32(cfg["threshold"]))[:3])
            counts[mode] += count
            sweeps[mode] += threshold_counts(prob[mode], seq.label)
            if full:
                roc[mode].update(seq.t, prob[mode], seq.target_id, seq.label, seq.x, seq.y, cfg["threshold"])
            row["readouts"][mode] = metric_row(count)
        if "causal" in prob:
            target = seq.label == 1
            baseline = prob["causal"] >= np.float32(cfg["threshold"])
            final = prob["final"] >= np.float32(cfg["threshold"])
            changes["corrected"] += int(np.sum((baseline != target) & (final == target)))
            changes["harmed"] += int(np.sum((baseline == target) & (final != target)))
            changes["rescued_target"] += int(np.sum(target & ~baseline & final))
            changes["suppressed_false_positive"] += int(np.sum(~target & baseline & ~final))
        if dump_dir is not None:
            Path(dump_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(Path(dump_dir) / seq.name), event_id=np.arange(seq.n_events),
                                t=seq.t, x=seq.x, y=seq.y, label=seq.label, target_id=seq.target_id,
                                output_time_ms=detail["output_time_ms"],
                                **{"prob_" + mode: prob[mode] for mode in modes})
        for key, value in detail["counts"].items():
            aggregate[key] = aggregate.get(key, 0) + value
        per_record.append(row)
        timing_rows.extend(timing["step_ms"])
        network_rows.extend(timing["network_ms"])
        msg_rows.extend(detail["messages"])
        read_count += detail["event_readouts"]
        source_seconds += timing["source_prepare_seconds"]
        latencies.append(detail["output_time_ms"] - seq.t)
        completion, finishes = 0., []
        for window, cost in enumerate(timing["all_step_ms"]):
            completion = max(completion, (window + 1) * float(cfg["window_ms"])) + cost
            finishes.append(completion)
        replay_latencies.append(np.asarray(finishes)[detail["release_window"]] - seq.t)
        revision_rows.append({"record": seq.name, **detail["revision"]})
        if full:
            print("EVAL %d/%d %s IoU %.5f" %
                  (i + 1, len(dataset), seq.name, row["readouts"]["final"]["iou"] or 0.), flush=True)
    result = {"design": DESIGN, "metrics": metric_row(counts["final"], roc["final"]),
              "readouts": {mode: metric_row(counts[mode], roc[mode]) for mode in modes},
              "threshold": cfg["threshold"], "n_sequences": len(dataset),
              "counts": aggregate, "events": aggregate.get("final_events", 0),
              "event_readouts": read_count, "per_record": per_record,
              "threshold_sweeps": {mode: sweep_summary(sweeps[mode]) for mode in modes},
              "diagnostics": {"relative_to_causal": changes, "revision_per_record": revision_rows,
                              "ablation_type": "same-checkpoint causal intervention, not independently retrained",
                              "empty_future": "observed empty windows are valid pairs; missing tails use causal fallback"},
              "metric_definitions": {"acc": "positive-event recall TP/(TP+FN), historical project ACC",
                                     "pd_fa": "original FastRoc including original strict time-boundary conventions"},
              "seconds_including_metrics": time.perf_counter() - start_time,
              "timing": {"execution": "step", "precision": "FP32; TF32 disabled",
                         "step_ms": distribution(timing_rows), "network_ms": distribution(network_rows),
                         "source_prepare_seconds": source_seconds,
                         "input_storage": "whole-record sparse arrays preloaded; not a live acquisition benchmark",
                         "includes_same_checkpoint_causal_readout": head is not None,
                         "acquisition_latency_ms": distribution(np.concatenate(latencies)),
                         "single_device_replay_latency_ms": distribution(np.concatenate(replay_latencies))},
              "communication": {"messages": distribution(msg_rows), "budget": cfg["comm_budget_messages"],
                                "violation_fraction": float(np.mean(np.asarray(msg_rows) > cfg["comm_budget_messages"]))
                                if msg_rows else None,
                                "scope": "BIR backbone only; revision is additional dense computation"},
              "operations": {"revision_dense_macs_per_full_pair": head.dense_macs(cfg["pad_height"], cfg["pad_width"])
                             if head is not None else 0,
                             "event_readout_macs": read_count * ((int(cfg["channels"][0]) + 2) *
                                                                 int(cfg["readout_hidden"]) + int(cfg["readout_hidden"])),
                             "scope": "revision/readout only; excludes backbone, softmax, interpolation and memory traffic",
                             "energy_joules": None}}
    return _plain(result)


def save_checkpoint(path, net, head, optimizer, epoch, best, cfg, stats, comm):
    payload = {"design": DESIGN, "model": net.state_dict(),
               "revision": head.state_dict() if head is not None else None,
               "optimizer": optimizer.state_dict(), "epoch": epoch, "best_val_iou": best,
               "config": cfg, "stats": stats, "comm_mu": comm.mu if comm else 0.}
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def load_checkpoint(path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def training_is_complete_scope(cfg):
    """Only an explicitly declared unrestricted run is a formal experiment.

    Unknown/old provenance and local implementation probes remain evaluable,
    but cannot be packaged as complete formal accuracy results.
    """
    scope = cfg.get("_training_scope")
    return (isinstance(scope, dict) and set(scope) == {"limit_records", "max_chunks"}
            and all(value is None for value in scope.values()))


def train(cfg, args):
    out, device = Path(args.out), torch.device(args.device)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((out / name).exists() for name in ("last.pt", "metrics.jsonl", "run_config.json")):
        raise ValueError("Output already contains a run; use a new release or --resume")
    seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    dataset = records(cfg, "train", args.limit_records)
    validation = records(cfg, "val", args.limit_records)
    cfg["_training_scope"] = {"limit_records": args.limit_records, "max_chunks": args.max_chunks}
    cfg["_source_sha256"] = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    cfg["_training_inventory"] = file_inventory(dataset.paths)
    # Validation determines checkpoint selection and is part of resume identity.
    cfg["_validation_inventory"] = file_inventory(validation.paths)
    if args.resume:
        ckpt = load_checkpoint(out / "last.pt", torch.device("cpu"))
        if ckpt.get("design") != DESIGN or ckpt["config"] != cfg:
            raise ValueError("Resume config/source/data differs from checkpoint")
    if cfg["revision_source"] != "causal":
        # Check the actual CUDA backward before statistics/calibration. The
        # isolated probe restores RNG/settings and never touches trained weights.
        write_json(out / "runtime_check.json", check_revision_backward(device))
    if args.resume:
        stats = ckpt["stats"]
    elif Path(cfg["stats_path"]).is_file():
        stats, _, _ = load_stats(cfg)
    else:
        stats = compute_stats(cfg)
    validate_statistics(stats, cfg)
    write_json(out / "train_stats.json", stats)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    pos_weight = min(float(stats["train_neg"]) / float(stats["train_pos"]), float(cfg["pos_weight_cap"]))
    net, head = make_models(cfg, device)
    params = list(net.parameters()) + ([] if head is None else list(head.parameters()))
    optimizer = torch.optim.Adam(params, lr=float(cfg["lr"]))
    comm = CommBudget.from_config(cfg, net)
    start, best, history = 0, -float("inf"), []
    if args.resume:
        net.load_state_dict(ckpt["model"])
        if head is not None:
            head.load_state_dict(ckpt["revision"])
        optimizer.load_state_dict(ckpt["optimizer"])
        comm.mu = float(ckpt["comm_mu"])
        start, best = int(ckpt["epoch"]) + 1, float(ckpt["best_val_iou"])
        history = [json.loads(line) for line in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        history = [row for row in history if row["epoch"] < start]
        (out / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history), encoding="utf-8")
    else:
        write_json(out / "run_config.json",
                   {"design": DESIGN, "config": cfg, "backbone_parameters": count_parameters(net),
                    "revision_parameters": count_parameters(head) if head is not None else {"total": 0},
                    "precision": "FP32; TF32 disabled", "torch": torch.__version__, "python": sys.version})
        run_calibration(net, cfg, q99, device, out)
        print("CALIBRATION COMPLETE", flush=True)
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
        row = dict(result, epoch=epoch, lr=lr, val_iou=val["metrics"]["iou"],
                   val_acc=val["metrics"]["acc"], val_readouts=val["readouts"],
                   epoch_seconds=time.perf_counter() - t0, tau=tau_statistics(net),
                   peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30
                   if device.type == "cuda" else None)
        if row["val_iou"] is None or not math.isfinite(row["val_iou"]):
            raise RuntimeError("Validation IoU is not finite")
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_plain(row), allow_nan=False) + "\n")
        history.append(row)
        if row["val_iou"] > best:
            best = row["val_iou"]
            save_checkpoint(out / ("best_val_iou_seed%d.pt" % cfg["seed"]), net, head,
                            optimizer, epoch, best, cfg, stats, comm)
        save_checkpoint(out / "last.pt", net, head, optimizer, epoch, best, cfg, stats, comm)
        probe = args.limit_records is not None or args.max_chunks is not None
        write_json(out / "summary.json", {"design": DESIGN, "epochs_completed": epoch + 1,
                   "epochs_requested": cfg["epochs"], "best_val_iou": best,
                   "complete": epoch + 1 == int(cfg["epochs"]) and not probe, "probe_only": probe,
                   "training_scope": cfg["_training_scope"], "seed": cfg["seed"],
                   "revision_source": cfg["revision_source"],
                   "train_seconds_total": sum(row["train_seconds"] for row in history),
                   "epoch_seconds_total": sum(row["epoch_seconds"] for row in history)})
        print("EPOCH %d/%d train %.1fs total %.1fs loss %.5f val IoU %.5f ACC %.5f comm_mu %.4f" %
              (epoch + 1, cfg["epochs"], result["train_seconds"], row["epoch_seconds"],
               result["loss_per_event"], row["val_iou"], row["val_acc"], comm.mu), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v15/observation.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", required=True, choices=("train", "eval", "stats"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--step", action="store_true", help="Compatibility flag: all evaluation is online step")
    parser.add_argument("--dump-predictions", action="store_true")
    args = parser.parse_args()
    for key in ("epochs", "limit_records", "max_chunks"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error("--%s must be positive" % key.replace("_", "-"))
    cfg = load_flat_config(args.config)
    if args.data_root:
        cfg["root"] = str(Path(args.data_root).resolve())
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    cfg["save_root"] = str(Path(args.out).resolve())
    validate_config(cfg)
    if args.mode == "stats":
        path = Path(args.out) / "train_stats.json"
        if path.exists() and not args.overwrite:
            raise ValueError("Statistics already exist")
        write_json(path, compute_stats(cfg))
    elif args.mode == "train":
        train(cfg, args)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for evaluation")
        device = torch.device(args.device)
        ckpt = load_checkpoint(args.checkpoint, torch.device("cpu"))
        if ckpt.get("design") != DESIGN:
            raise ValueError("Checkpoint is not a V15 native revision model")
        for key, value in ckpt["config"].items():
            if not key.startswith("_") and key not in ("root", "save_root", "stats_path") and cfg.get(key) != value:
                raise ValueError("Evaluation config differs from checkpoint: " + key)
        seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
        net, head = make_models(cfg, device)
        net.load_state_dict(ckpt["model"])
        if head is not None:
            head.load_state_dict(ckpt["revision"])
        target = Path(args.out) / ("eval_%s.json" % args.split)
        if target.exists() and not args.overwrite:
            raise ValueError("Evaluation exists: " + str(target))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        evaluation_data = records(cfg, args.split, args.limit_records)
        result = evaluate(net, head, evaluation_data, cfg,
                          np.asarray(ckpt["stats"]["q99"], dtype=np.float32), device, full=True,
                          dump_dir=Path(args.out) / ("predictions_" + args.split) if args.dump_predictions else None)
        training_scope = ckpt["config"].get("_training_scope", {})
        probe = args.limit_records is not None or not training_is_complete_scope(ckpt["config"])
        result.update(checkpoint_sha256=sha256(args.checkpoint), selected_epoch=int(ckpt["epoch"]) + 1,
                      seed=cfg["seed"], split=args.split, revision_source=cfg["revision_source"],
                      complete=not probe, probe_only=probe, checkpoint_training_scope=training_scope,
                      evaluation_dataset={"root": str(Path(cfg["root"]).resolve()),
                                          "split": args.split,
                                          "records": file_inventory(evaluation_data.paths)},
                      peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30
                      if device.type == "cuda" else None)
        write_json(target, result)
        print(json.dumps(result["metrics"], indent=2), flush=True)
        print("EVALUATION COMPLETE " + str(target), flush=True)


if __name__ == "__main__":
    main()
