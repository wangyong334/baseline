"""V16 context revision; r2 design in docs/V16_R2_METHOD.md.

V5 state dynamics and readout are preserved. Observation length, state clock
and evaluation frame length are distinct. r2 caches paired readout observations;
r1 caches dense context. No event neighbour graph or recurrent event state.
Python 3.8 / torch 1.9.
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
from model.context_revision import ContextRevision
from model.context_evidence import ContextEvidence
from model.anchored_revision import AnchoredRevision
from model.covariance_revision import CovarianceRevision
from model.evspsegnet_stream import calibrate_gains, count_parameters, tif_settings
from model.lif2d_stream import detach_states
from train_stream_v1 import (build_net, seed_everything, plan_streams, stream_pieces,
                             stream_total_windows, stream_valid, tau_statistics)
from utils.comm_budget import CommBudget, window_messages
from utils.stream_common import load_flat_config, linear_epoch_lr, select_subset, subset_due
from utils.stream_metrics import FastRoc, window_confusion, threshold_counts, sweep_summary
from tools.v14.run_experiment import write_json, sha256

DESIGN = "v16-context-revision-r1"
EVIDENCE_DESIGN = "v16-context-evidence-r2"
ANCHORED_DESIGN = "v16-anchored-revision-r3"
MODES = ("full", "no_context", "no_revision", "second")


def evidence_model(cfg):
    return cfg.get("revision_architecture", "dense") in ("evidence", "anchored", "covariance")


def covariance_feature_source(cfg):
    source = cfg.get("covariance_feature_source", "multilevel")
    if source not in ("multilevel", "dec1"):
        raise ValueError("Unknown covariance_feature_source: " + str(source))
    return source


def covariance_feature_dim(cfg):
    """Descriptor width before appending the original event logit."""
    if covariance_feature_source(cfg) == "dec1":
        return int(cfg["channels"][0])
    return 2 * int(cfg["channels"][1]) + int(cfg["channels"][0])


def readout_feature_source(cfg):
    # Adaptation switches to evidence/second while retaining covariance options.
    # Its original MLP activation must remain independent of those options.
    return covariance_feature_source(cfg) if cfg.get("revision_architecture") == "covariance" else "mlp"


def design_name(cfg):
    if cfg.get("revision_architecture") == "covariance":
        if cfg.get("constraint") == "record_fn_reference":
            return "v16-covariance-revision-r5"
        return "v16-covariance-revision-r4"
    if cfg.get("revision_architecture") == "anchored":
        return ANCHORED_DESIGN
    return EVIDENCE_DESIGN if evidence_model(cfg) else DESIGN


def boundary_retention_loss(logits, reference, labels, threshold=.9, retention_weight=5., margin=1.):
    """Fixed operating-point classification plus supervised positive retention.

    BCE(z-logit(tau), y) locates the classification boundary at tau without
    changing inference. The reference is detached and only correct training
    positives above that boundary are protected; wrong background scores remain
    free to fall. This is a soft training objective, not a recall guarantee.
    """
    boundary = math.log(float(threshold) / (1. - float(threshold)))
    reference = reference.detach()
    base = F.binary_cross_entropy_with_logits(logits - boundary, labels, reduction="none")
    protected = labels * (reference >= boundary).to(labels.dtype)
    target = torch.clamp(reference, max=boundary + float(margin))
    return base + float(retention_weight) * protected * F.relu(target - logits).square()


def supervised_excess_risk(logits, reference, labels, pos_weight):
    """Penalize worsening either class relative to the detached second view.

    This is a train-label loss, not a test-time veto or an error-rate guarantee.
    Beneficial corrections pay zero extra cost. Both classes use the same
    weighting as the task BCE; no threshold or validation labels are fitted.
    """
    reference = reference.detach()
    positive = F.relu(F.softplus(-logits) - F.softplus(-reference))
    negative = F.relu(F.softplus(logits) - F.softplus(reference))
    return labels * float(pos_weight) * positive + (1. - labels) * negative


def validate_config(cfg):
    if int(cfg["window_ms"]) != 2 * int(cfg["stride_ms"]):
        raise ValueError("V16 requires window_ms = 2 * stride_ms (two observations)")
    if int(cfg["total_ms"]) < int(cfg["window_ms"]) or int(cfg["total_ms"]) % int(cfg["stride_ms"]):
        raise ValueError("total_ms must be a stride multiple, at least one full window")
    expected = (int(cfg["total_ms"]) - int(cfg["window_ms"])) // int(cfg["stride_ms"]) + 1
    if int(cfg["n_windows"]) != expected:
        raise ValueError("n_windows does not match rolling observation times")
    if cfg.get("dt_ms", cfg["stride_ms"]) != cfg["stride_ms"]:
        raise ValueError("BIR dt_ms must equal stride_ms, not observation length")
    if cfg["state_mode"] != "carry" or cfg["neuron"] != "tif" or cfg["norm"] != "none":
        raise ValueError("V16 requires V5 BIR, carry and no temporal batch normalization")
    if cfg.get("state_layout", "single_rolling") != "single_rolling":
        raise ValueError("V16 requires one chronological rolling state chain")
    if cfg.get("trainer") != "streams" or cfg.get("execution") != "layer":
        raise ValueError("V16 requires parallel-stream layerwise TBPTT")
    if cfg.get("eval_execution", "step") not in ("step", "layer"):
        raise ValueError("eval_execution must be step or layer")
    if int(cfg.get("accumulate", 1)) != 1 or cfg.get("stream_layout", "greedy") != "greedy":
        raise ValueError("This release preserves greedy whole-record streams, accumulate=1")
    if int(cfg.get("stream_groups", 1)) != 1:
        raise ValueError("Use streams=4 without extra stream grouping in this release")
    if cfg.get("revision_variant") not in MODES:
        raise ValueError("Unsupported V16 variant")
    if cfg.get("revision_architecture", "dense") not in ("dense", "evidence", "anchored", "covariance"):
        raise ValueError("Unknown revision architecture")
    if not evidence_model(cfg) and int(cfg["revision_dim"]) != int(cfg["channels"][0]):
        raise ValueError("revision_dim must match the V5 readout channels")
    if evidence_model(cfg):
        if float(cfg.get("threshold", .9)) != .9:
            raise ValueError("Evidence experiments keep threshold=0.9")
        if int(cfg.get("early_stop_patience", 0)) < 0 or int(cfg.get("min_epochs", 1)) < 1:
            raise ValueError("Invalid stopping schedule")
        if cfg.get("fusion_loss", "bce") not in ("bce", "excess_risk", "boundary_retention", "rate_constrained"):
            raise ValueError("Unknown evidence loss")
        if not math.isfinite(float(cfg.get("excess_risk_weight", 1.))) or float(cfg.get("excess_risk_weight", 1.)) < 0:
            raise ValueError("excess_risk_weight must be finite and nonnegative")
        for key, default in (("retention_weight", 5.), ("retention_margin", 1.), ("backbone_lr_scale", 1.)):
            if not math.isfinite(float(cfg.get(key, default))) or float(cfg.get(key, default)) <= 0:
                raise ValueError(key + " must be finite and positive")
    if int(cfg["pad_height"]) % 8 or int(cfg["pad_width"]) % 8:
        raise ValueError("V5 padded canvas must be divisible by 8")
    if int(cfg["streams"]) < 1 or int(cfg["tbptt_k"]) < 1 or int(cfg["epochs"]) < 1:
        raise ValueError("streams, tbptt_k and epochs must be positive")
    if cfg.get("comm_budget_messages") is None:
        raise ValueError("V16 formal model retains the V5 communication budget")
    for key in ("dual_memory", "subthreshold_residual", "invariant_input", "noise_aug", "comod"):
        if cfg.get(key):
            raise ValueError("Unrelated mechanism enabled: " + key)
    if cfg.get("revision_architecture") == "covariance":
        from tools.v16.run_experiment import validate_covariance_protocol
        validate_covariance_protocol(cfg)
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
        torch.manual_seed(int(cfg["seed"]) + 16000)
        if cfg.get("revision_architecture") == "covariance":
            # Keep the validated r4 head seed independent of backbone creation.
            torch.manual_seed(int(cfg["seed"]))
            head = CovarianceRevision(feature_dim=covariance_feature_dim(cfg),
                        hidden=int(cfg.get("evidence_hidden", 8)), kappa=float(cfg.get("evidence_kappa", 2.)),
                        variant=cfg["revision_variant"], channels=int(cfg.get("covariance_channels", 8)),
                        shrinkage=float(cfg.get("covariance_shrinkage", .05)),
                        channel_basis=cfg.get("covariance_channel_basis", "shared_fisher"))
        elif evidence_model(cfg):
            head_type = AnchoredRevision if cfg.get("revision_architecture") == "anchored" else ContextEvidence
            common_head = head_type(feature_dim=int(cfg["readout_hidden"]),
                                   hidden=int(cfg.get("evidence_hidden", 8)),
                                   kappa=float(cfg.get("evidence_kappa", 1.)),
                                   variant="full")
            head = head_type(feature_dim=int(cfg["readout_hidden"]),
                                   hidden=int(cfg.get("evidence_hidden", 8)),
                                   kappa=float(cfg.get("evidence_kappa", 1.)),
                                   variant=cfg["revision_variant"])
            head.load_state_dict({k: v for k, v in common_head.state_dict().items()
                                  if k in head.state_dict()})
        else:
            head = ContextRevision(channels=int(cfg["channels"][0]), dim=int(cfg["revision_dim"]),
                               hidden=int(cfg["revision_hidden"]), heads=int(cfg["revision_heads"]),
                               radius=int(cfg["revision_radius"]), downsample=int(cfg["revision_downsample"]),
                               height=int(cfg["height"]), width=int(cfg["width_px"]),
                               variant=cfg["revision_variant"])
    return net, head.to(device)


class Records(object):
    def __init__(self, cfg, split, names=None):
        self.cfg = cfg
        directory = Path(cfg["root"]) / split
        self.paths = ([directory / n for n in names] if names is not None
                      else sorted(directory.glob("*.npz")))
        if not self.paths:
            raise ValueError("No records in " + str(directory))
        self._cache = {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        c = self.cfg
        path = self.paths[index]
        if not c.get("cache_record_indices", False):
            return load_overlap_npz(path, c["height"], c["width_px"], c["window_ms"],
                                    c["stride_ms"], c["total_ms"], c["time_bins"])
        if path not in self._cache:
            self._cache[path] = load_overlap_npz(path, c["height"], c["width_px"],
                                                c["window_ms"], c["stride_ms"], c["total_ms"], c["time_bins"])
        return self._cache[path]


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


def detach_pending(pending):
    # Clone a single slot: detach alone retains storage for the complete TBPTT batch.
    def own(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: own(child) for key, child in value.items()}
        return value
    return own(pending)


def readout_with_hidden(net, features, events, context=None, source=None):
    """Return unchanged V5 logits with an explicitly selected descriptor.

    Legacy callers without a source retain MLP activations (or the multilevel
    descriptor when context is supplied). Internal paths specify the source.
    """
    source = ("multilevel" if context is not None else "mlp") if source is None else source
    if source not in ("mlp", "multilevel", "dec1"):
        raise ValueError("Unknown readout feature source: " + str(source))
    if source == "multilevel" and (context is None or len(context) != 2):
        raise ValueError("multilevel descriptors require enc2 and dec2 feature taps")
    values = features[events["t"], events["b"], :, events["y"], events["x"]]
    extra = torch.stack([events["p"], events["t_local"]], 1).to(values.dtype)
    if net.readout_ablation != "none":
        raise ValueError("Evidence training requires the original V5 readout")
    hidden = net.readout[1](net.readout[0](torch.cat([values, extra], 1)))
    logits = net.readout[2](hidden).squeeze(1)
    if source == "multilevel":
        taps = [f[events["t"], events["b"], :, events["y"]//2, events["x"]//2] for f in context]
        hidden = torch.cat(taps + [values], 1)
    elif source == "dec1":
        hidden = values
    return logits, hidden


def evidence_pairs(head, logits, hidden, segments, pending):
    """Match original event IDs. Cache only the uncommitted half-window.

    Every appearance is read once. A pair inside TBPTT has gradients through
    both views; only the old half-window is detached at the TBPTT boundary.
    """
    first_z, first_h, second_rows = [], [], []
    cross = 0
    for index, lo, hi, row, seq, ids in segments:
        old = pending.pop(index, None)
        if lo and (old is None or old["window"] != lo - 1):
            raise RuntimeError("Missing preceding evidence window")
        if not lo and old is not None:
            raise RuntimeError("Evidence cache leaked across recordings")
        start = int(seq.bounds[lo])
        positions = np.arange(start, int(seq.bounds[hi]))
        final = np.flatnonzero((seq.coverage_count[ids] == 2) &
                              (seq.last_position[ids] == positions))
        prior = seq.first_position[ids[final]] - start
        inside = prior >= 0
        for mask in (inside, ~inside):
            selected = final[mask]
            if not len(selected):
                continue
            second_rows.append(selected + row)
            if mask is inside:
                rid = _index(prior[mask] + row, logits.device)
                first_z.append(logits[rid]); first_h.append(hidden[rid])
            else:
                if old is None or not np.array_equal(old["ids"], ids[selected]):
                    raise RuntimeError("Original event IDs do not match pending evidence")
                first_z.append(old["logits"]); first_h.append(old["hidden"])
                cross += len(selected)
        if hi < seq.n_windows:
            keep = np.flatnonzero((seq.coverage_count[ids] == 2) &
                                 (seq.first_position[ids] == positions) &
                                 (seq.last_window[ids] >= hi))
            rid = _index(keep + row, logits.device)
            pending[index] = {"window": hi - 1, "ids": ids[keep].copy(),
                              "logits": logits[rid], "hidden": hidden[rid]}
    if not second_rows:
        return logits.new_empty((0,)), np.empty(0, dtype=np.int64), cross
    rows = np.concatenate(second_rows)
    rid = _index(rows, logits.device)
    final = head(torch.cat(first_z), logits[rid], torch.cat(first_h), hidden[rid])
    return final, rows, cross


def fuse_chunk(net, head, features, events, segments, pieces, pending):
    """Shared dense context work, followed by one batched readout of completed pairs.

    Window coordinates come from stream metadata, never from events (empty
    windows are real observations). A cache is indexed by recording, not stream.
    """
    steps, batch = features.shape[:2]
    flat = features.reshape((-1,) + tuple(features.shape[2:]))
    enc = head.encode(flat)
    locations = {(index, k): (offset + k - lo) * batch + b
                 for b, stream in enumerate(pieces)
                 for index, lo, hi, offset, new in stream for k in range(lo, hi)}
    first, second, e1, e2, rows, pair_numbers = [], [], [], [], [], []
    cross = 0
    for index, lo, hi, row, seq, ids in segments:
        old = pending.pop(index, None)
        if lo and (old is None or old["window"] != lo - 1):
            raise RuntimeError("Missing/misaligned preceding feature window")
        if lo == 0 and old is not None:
            raise RuntimeError("Feature cache leaked across recordings")
        for k in range(lo, hi):
            a, z = int(seq.bounds[k] - seq.bounds[lo]), int(seq.bounds[k+1] - seq.bounds[lo])
            local_ids = ids[a:z]
            local_rows = np.flatnonzero((seq.coverage_count[local_ids] == 2) &
                                        (seq.last_window[local_ids] == k)) + a + row
            if not len(local_rows):
                continue
            current = locations[(index, k)]
            if k == lo:
                f1, c1 = old["feature"], old["encoded"]
                cross += len(local_rows)
            else:
                previous = locations[(index, k-1)]
                f1, c1 = flat[previous], {key: v[previous] for key, v in enc.items()}
            first.append(f1); second.append(flat[current])
            e1.append(c1); e2.append({key: v[current] for key, v in enc.items()})
            rows.append(local_rows)
            pair_numbers.append(np.full(len(local_rows), len(first)-1, dtype=np.int64))
        if hi < seq.n_windows:
            last = locations[(index, hi-1)]
            pending[index] = {"window": hi-1, "feature": flat[last],
                              "encoded": {key: v[last] for key, v in enc.items()}}
    if not rows:
        return features.new_empty((0,)), np.empty(0, dtype=np.int64), cross
    rows = np.concatenate(rows)
    rid = _index(rows, features.device)
    f1, f2 = torch.stack(first), torch.stack(second)
    c1 = {key: torch.stack([c[key] for c in e1]) for key in enc}
    c2 = {key: torch.stack([c[key] for c in e2]) for key in enc}
    revised = head.fuse(f1, f2, c1, c2)
    values = revised[_index(np.concatenate(pair_numbers), features.device), :, events["y"][rid], events["x"][rid]]
    logits = net._readout(values, {key: events[key][rid] for key in ("p", "t_local")})
    return logits, rows, cross


def classification_loss(net, head, features, logits, labels, events, segments, pieces, pending, pos_weight,
                        hidden=None, cfg=None):
    weight = logits.new_tensor([float(pos_weight)])
    per_read = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight, reduction="none")
    loss, mass, singles = logits.sum() * 0, 0., 0
    for index, lo, hi, row, seq, ids in segments:
        paired = seq.coverage_count[ids] == 2
        weights = np.where(paired, .25, 1.).astype(np.float32)
        loss = loss + (per_read[row:row+len(ids)] * torch.as_tensor(weights, device=logits.device)).sum()
        mass += float(weights.sum())
        singles += int((~paired).sum())
    if isinstance(head, ContextEvidence):
        if hidden is None:
            source = covariance_feature_source(cfg or {}) if isinstance(head, CovarianceRevision) else "mlp"
            logits, hidden = readout_with_hidden(net, features, events, source=source)
        final, rows, cross = evidence_pairs(head, logits, hidden, segments, pending)
    else:
        final, rows, cross = fuse_chunk(net, head, features, events, segments, pieces, pending)
    if len(rows):
        rid = _index(rows, logits.device)
        if (cfg or {}).get("fusion_loss") == "rate_constrained":
            # This path exercises end-to-end gradients in preflight. Formal r4
            # training freezes V5 and uses the population-cached rate objective.
            from tools.v16.train_covariance import constrained_loss
            risk = constrained_loss(final, labels[rid], float(cfg.get("recall_dual_init", 1.)), cfg["threshold"])
            final_loss = (risk + cfg["covariance_regularizer"] * head.variance_regularizer()) * len(rows)
        elif (cfg or {}).get("fusion_loss") == "boundary_retention":
            final_loss = boundary_retention_loss(final, logits[rid], labels[rid], cfg["threshold"],
                                                cfg.get("retention_weight", 5.), cfg.get("retention_margin", 1.)).sum()
        else:
            final_loss = F.binary_cross_entropy_with_logits(final, labels[rid], pos_weight=weight, reduction="sum")
        loss = loss + .5 * final_loss
        if isinstance(head, ContextEvidence) and (cfg or {}).get("fusion_loss") == "excess_risk":
            rid = _index(rows, logits.device)
            loss = loss + .5 * float(cfg.get("excess_risk_weight", 1.)) * supervised_excess_risk(
                final, logits[rid], labels[rid], pos_weight).sum()
        mass += .5 * len(rows)
    return loss, mass, singles + len(rows), cross


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
        feature_source = readout_feature_source(cfg)
        logits, states, info = net.forward_chunk(x, events, prev, keep=keep, collect=comm is not None,
                                                return_features=True, skip_readout=evidence_model(cfg),
                                                feature_taps=feature_source == "multilevel")
        hidden = None
        if evidence_model(cfg):
            logits, hidden = readout_with_hidden(net, info["features"], events, info.get("context_features"),
                                                source=feature_source)
        raw_loss, mass, nfinal, ncross = classification_loss(net, head, info["features"], logits, labels, events,
                                                           segments, pieces, pending, pos_weight, hidden, cfg)
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
        pending = detach_pending(pending)
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
        del logits, info, x, events, labels, raw_loss, terms, hidden
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
    return ["first", "second", "main"]


@torch.no_grad()
def infer_record(net, head, seq, cfg, q99, device, modes=None, step=False, diagnostics=False):
    modes = readout_modes(cfg) if modes is None else modes
    if not set(modes).issubset({"first", "second", "main"}):
        raise ValueError("Unknown readout policy")
    sync(device)
    start = time.perf_counter()
    source = OverlapWindowSource(seq, cfg, q99, device)
    sync(device)
    source_seconds = time.perf_counter() - start
    probs = {m: np.full(seq.n_events, np.nan, dtype=np.float32) for m in modes}
    hits = np.zeros(seq.n_events, dtype=np.int8)
    states, pending = None, {}
    # Diagnostic first scores only; production modes=['main'] allocates no score cache.
    first_scores = np.full(seq.n_events, np.nan, dtype=np.float32) if "first" in modes else None
    durations, net_times, message_counts, event_counts, completions = [], [], [], [], []
    chunk_size = 1 if step else int(cfg.get("eval_chunk", 32))
    for lo in range(0, seq.n_windows, chunk_size):
        hi = min(seq.n_windows, lo + chunk_size)
        if step: sync(device)
        t0 = time.perf_counter()
        x, events, _, ids = source.chunk(lo, hi)
        if step: sync(device)
        t1, prev = time.perf_counter(), states
        need_aux = "first" in modes or "second" in modes
        sparse_evidence = isinstance(head, ContextEvidence)
        feature_source = (covariance_feature_source(cfg)
                          if isinstance(head, CovarianceRevision) and "main" in modes else "mlp")
        covariance_taps = feature_source == "multilevel"
        if step:
            ev = {k: v for k, v in events.items() if k != "t"}
            logits, states, info = net(x[0], ev, prev, collect=diagnostics,
                                      return_features=True, skip_readout=sparse_evidence or not need_aux,
                                      feature_taps=covariance_taps)
            features = info["features"].unsqueeze(0)
            context = tuple(v.unsqueeze(0) for v in info["context_features"]) if "context_features" in info else None
        else:
            logits, states, info = net.forward_chunk(x, events, prev, collect=diagnostics,
                                                     return_features=True, skip_readout=sparse_evidence or not need_aux,
                                                     feature_taps=covariance_taps)
            features = info["features"]
            context = info.get("context_features")
        hidden = None
        if sparse_evidence:
            logits, hidden = readout_with_hidden(net, features, events, context, source=feature_source)
        if step: sync(device)
        t2 = time.perf_counter()
        segment = (0, lo, hi, 0, seq, np.asarray(ids))
        pieces = [[(0, lo, hi, 0, lo == 0)]]
        if "main" in modes:
            if sparse_evidence:
                revised, pair_rows, _ = evidence_pairs(head, logits, hidden, [segment], pending)
            else:
                revised, pair_rows, _ = fuse_chunk(net, head, features, events, [segment], pieces, pending)
            pending = detach_pending(pending)
        else:
            pair_rows = np.flatnonzero((seq.coverage_count[ids] == 2) &
                                       (seq.last_position[ids] == np.arange(seq.bounds[lo], seq.bounds[hi])))
        single_rows = np.flatnonzero(seq.coverage_count[ids] == 1)
        final_rows = np.concatenate([single_rows, pair_rows])
        final_ids = ids[final_rows]
        rid = _index(final_rows, device)
        if first_scores is not None:
            positions = np.arange(seq.bounds[lo], seq.bounds[hi])
            first_rows = np.flatnonzero(positions == seq.first_position[ids])
            first_scores[ids[first_rows]] = torch.sigmoid(logits[_index(first_rows, device)]).cpu().numpy()
        for mode in modes:
            if mode == "first":
                value = first_scores[final_ids]
            elif mode == "second":
                value = torch.sigmoid(logits[rid]).cpu().numpy()
            else:
                sr = _index(single_rows, device)
                if need_aux or sparse_evidence:
                    single_logits = logits[sr]
                else:
                    f = features[events["t"][sr], events["b"][sr], :, events["y"][sr], events["x"][sr]]
                    single_logits = net._readout(f, {key: events[key][sr] for key in ("p", "t_local")})
                value = torch.sigmoid(torch.cat([single_logits, revised])).cpu().numpy()
            probs[mode][final_ids] = value
        np.add.at(hits, final_ids, 1)
        if step: sync(device)
        elapsed = (time.perf_counter() - t0) * 1000.
        if step:
            completions.append(elapsed)
            if lo >= int(cfg.get("timing_warmup_windows", 20)):
                durations.append(elapsed); net_times.append((t2-t1)*1000.); event_counts.append(len(ids))
        if diagnostics:
            memories = [m.unsqueeze(0) for m in info["u_pre"]] if step else info["u_pre"]
            message_counts.extend(window_messages(net, memories, prev, None).reshape(-1).cpu().tolist())
        del x, logits, info, features, hidden
    if pending or np.any(hits != 1) or any(not np.isfinite(v).all() for v in probs.values()):
        raise RuntimeError("Output coverage/finite-value check failed: " + seq.name)
    return probs, {"execution": "step" if step else "layer", "step_ms": durations,
                   "network_ms": net_times, "events_per_measured_step": event_counts,
                   "all_step_ms": completions, "source_prepare_seconds": source_seconds,
                   "input_storage": "whole-record sparse arrays; per-chunk dense representation"}, message_counts


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
    main = "main"
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
                                             "suppressed_false_positive": 0, "lost_target": 0,
                                             "new_false_positive": 0})
                cc["corrected"] += int(np.sum((base != target) & (full_pred == target)))
                cc["harmed"] += int(np.sum((base == target) & (full_pred != target)))
                cc["rescued_target"] += int(np.sum(target & ~base & full_pred))
                cc["suppressed_false_positive"] += int(np.sum(~target & base & ~full_pred))
                cc["lost_target"] += int(np.sum(target & base & ~full_pred))
                cc["new_false_positive"] += int(np.sum(~target & ~base & full_pred))
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
    result = {"design": design_name(cfg), "readouts": {m: metric_row(counts[m], roc[m]) for m in modes},
              "threshold": cfg["threshold"], "n_sequences": len(dataset), "events": total_events,
              "production_event_readouts": total_reads if evidence_model(cfg) else total_events,
              "diagnostic_view_readouts": total_reads, "per_record": per_record,
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
              "operations": {}}
    result["metrics"] = result["readouts"][main]
    result["readout_policy"] = cfg["revision_variant"]
    reads = total_reads if evidence_model(cfg) else total_events
    result["operations"] = {"production_event_readout_linear_macs": reads *
                            ((int(cfg["channels"][0])+2)*int(cfg["readout_hidden"])+int(cfg["readout_hidden"])),
                            "context_dense_macs_per_window": (0 if evidence_model(cfg) else head.dense_macs(int(cfg["pad_height"]), int(cfg["pad_width"]))),
                            "paired_fusion_linear_macs": ((total_reads - total_events) * head.macs_per_pair() if evidence_model(cfg) else 0),
                            "scope": "head MAC estimates exclude backbone, softmax, interpolation, indexing; no MAC-to-energy claim",
                            "energy_joules": None}
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
    payload = {"design": design_name(cfg), "model": net.state_dict(), "fusion": head.state_dict(),
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


def validate_backbone_initialization(initial, cfg, q99):
    """Only an explicit trainable adaptation stage may change the window clock.

    Loading weights does not migrate the event representation. New windows use
    freshly computed train normalization and recalibrated train-only gains.
    The frozen statistical head must still match its source representation.
    """
    source = initial.get("config", {})
    adaptation = cfg.get("initialization_policy") == "window_adaptation"
    if cfg.get("initialization_policy") not in (None, "matched", "window_adaptation"):
        raise ValueError("Unknown initialization policy")
    if adaptation and (cfg.get("revision_architecture") != "evidence" or
                       cfg.get("revision_variant") != "second" or cfg.get("freeze_backbone", False)):
        raise ValueError("Window adaptation requires trainable backbone and second-only readout")
    for key in ("seed", "channels", "neuron", "state_mode", "time_bins", "input_clip",
                "height", "width_px", "pad_height", "pad_width", "norm", "v_threshold",
                "tau_init_ms", "tau_min_ms", "tau_max_ms", "readout_hidden"):
        if source.get(key) != cfg.get(key):
            raise ValueError("Initial backbone configuration mismatch: " + key)
    if tif_settings(source.get("tif")) != tif_settings(cfg.get("tif")):
        raise ValueError("Initial backbone BIR configuration mismatch")
    changed = any(source.get(k) != cfg.get(k) for k in ("window_ms", "stride_ms"))
    if changed and not adaptation:
        raise ValueError("Initial backbone window mismatch; explicit trainable window_adaptation is required")
    if source.get("state_layout", "single_rolling") != cfg.get("state_layout", "single_rolling"):
        raise ValueError("Initial backbone state layout mismatch")
    previous = np.asarray(initial.get("stats", {}).get("q99", []), dtype=np.float32)
    current = np.asarray(q99, dtype=np.float32)
    if previous.shape != current.shape or previous.ndim != 1 or not np.isfinite(previous).all():
        raise ValueError("Initial normalization must be finite and have the matched channel shape")
    if not changed and not np.allclose(previous, current, rtol=0, atol=1e-6):
        raise ValueError("Initial checkpoint must use this training-set normalization")
    mu = float(initial.get("comm_mu", 0.))
    if not math.isfinite(mu) or mu < 0:
        raise ValueError("Initial communication multiplier must be finite and nonnegative")
    return changed, mu


def train(cfg, args):
    if cfg.get("revision_architecture") == "covariance":
        from tools.v16.train_covariance import train_cached
        return train_cached(cfg, args)
    if cfg.get("require_initialization", False) and not cfg.get("init_checkpoint"):
        raise ValueError("This fine-tuning experiment requires --init-root at enqueue (matched seed backbone)")
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
    source_paths = [Path(__file__), ROOT / "dataset/overlap_windows.py", ROOT / "model/context_revision.py",
                    ROOT / "model/context_evidence.py", ROOT / "model/anchored_revision.py",
                    ROOT / "model/deterministic_resize.py", ROOT / "dataset/stream_windows.py",
                    ROOT / "utils/stream_common.py",
                    ROOT / "model/evspsegnet_stream.py", ROOT / "model/lif2d_stream.py", ROOT / "utils/comm_budget.py",
                    ROOT / "train_stream_v1.py", ROOT / "utils/stream_metrics.py"]
    cfg["_source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    stats_path = out / "train_stats.json"
    if args.resume:
        ckpt = load_checkpoint(out / "last.pt", torch.device("cpu"))
        if ckpt["design"] != design_name(cfg) or ckpt["config"] != cfg:
            raise ValueError("Resume config/design mismatch")
        summary_path = out / "summary.json"
        if summary_path.is_file():
            previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (previous_summary.get("complete") is True and
                    previous_summary.get("epochs_completed") == int(ckpt["epoch"]) + 1):
                raise ValueError("Completed training cannot be resumed; use a new release for a new experiment")
        stats = ckpt["stats"]
    elif stats_path.exists():
        raise ValueError("Stale stats without checkpoint; choose a clean output")
    else:
        stats = compute_overlap_stats(dataset.paths, cfg)
        write_json(stats_path, stats)
    q99, pos_weight = np.asarray(stats["q99"], dtype=np.float32), float(stats["pos_weight"])
    net, head = make_models(cfg, device)
    groups = [
        {"params": list(net.parameters()), "lr_scale": float(cfg.get("backbone_lr_scale", 1.))},
        {"params": list(head.parameters()), "lr_scale": 1.}]
    if cfg.get("revision_architecture") != "anchored":
        groups = list(net.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(groups, lr=float(cfg["lr"]))
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
        write_json(out / "run_config.json", {"design": design_name(cfg), "config": cfg,
                   "backbone_parameters": count_parameters(net), "fusion_parameters": count_parameters(head),
                   "precision": "float32; CUDA matmul and cuDNN TF32 disabled",
                   "actual_optimizer_step": "per parallel-stream TBPTT chunk, as in V14",
                   "torch": torch.__version__, "python": sys.version,
                   "data": {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in dataset.paths},
                   "source": {str(p.relative_to(ROOT)): sha256(p) for p in
                              [Path(__file__), ROOT / "dataset/overlap_windows.py", ROOT / "model/context_revision.py", ROOT / "model/context_evidence.py", ROOT / "model/anchored_revision.py",
                               ROOT / "model/evspsegnet_stream.py", ROOT / "model/lif2d_stream.py", ROOT / "utils/comm_budget.py"]}})
        if cfg.get("init_checkpoint"):
            initial = load_checkpoint(cfg["init_checkpoint"], torch.device("cpu"))
            changed_window, initial_mu = validate_backbone_initialization(initial, cfg, q99)
            net.load_state_dict(initial["model"], strict=True)
            comm.mu = initial_mu
            provenance = {"mode": "window adaptation" if changed_window else "inherited matched backbone",
                          "checkpoint": cfg["init_checkpoint"], "sha256": sha256(cfg["init_checkpoint"]),
                          "seed": cfg["seed"], "source_window_ms": initial["config"]["window_ms"],
                          "window_ms": cfg["window_ms"], "stride_ms": cfg["stride_ms"],
                          "normalization": "current training set", "gain_recalibrated": changed_window}
            if changed_window:
                net.gain_calibrated.fill_(False)
                calibrate(net, dataset, cfg, q99, device, out)
                report = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
                report["initialization"] = provenance
                write_json(out / "calibration.json", report)
            else:
                write_json(out / "calibration.json", provenance)
            write_json(out / "initialization.json", provenance)
            del initial
        else:
            calibrate(net, dataset, cfg, q99, device, out)
    validation = Records(cfg, "val")
    if args.limit_records:
        validation.paths = validation.paths[:args.limit_records]
    subset = select_subset([p.name for p in dataset.paths], int(cfg["train_subset_size"]), int(cfg["seed"]) + 2000)
    subset_data = Records(cfg, "train", subset)
    guarded_selection = cfg.get("selection_rule") == "four_metric_noninferiority"
    selection_reference = None
    if guarded_selection and not cfg.get("validate_initial", False):
        raise ValueError("Four-metric selection requires an initial validation reference")
    if guarded_selection and args.resume:
        selection_reference = json.loads((out / "initial_validation.json").read_text(encoding="utf-8"))["metrics"]
    best_epoch = min((r["epoch"] for r in history if r["val_iou"] == best and
                      r.get("selection_eligible", True)), default=-1)
    # A warm start must beat its own online validation result, rather than being
    # declared improved merely because every trained epoch is accepted.
    if not args.resume and cfg.get("validate_initial", False):
        initial_val = evaluate(net, head, validation, cfg, q99, device, full=guarded_selection)
        selection_reference = initial_val["metrics"]
        best = float(initial_val["metrics"]["iou"])
        if not math.isfinite(best):
            raise RuntimeError("Initial validation IoU is not finite")
        write_json(out / "initial_validation.json", initial_val)
        save_checkpoint(out / ("best_val_iou_seed%d.pt" % cfg["seed"]), net, head, optimizer, -1, best, cfg, stats, comm)
        print("INITIAL VALIDATION IoU %.6f ACC %.6f" % (best, initial_val["metrics"]["acc"]), flush=True)
    for epoch in range(start, int(cfg["epochs"])):
        t0 = time.perf_counter()
        lr = (float(cfg["lr"]) if int(cfg["epochs"]) == 1 else
              linear_epoch_lr(epoch, int(cfg["epochs"]), float(cfg["lr"]), float(cfg["lr_end"])))
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = train_epoch(net, head, dataset, optimizer, cfg, q99, pos_weight, device,
                             np.random.RandomState(int(cfg["seed"]) * 1000 + epoch + 7), comm,
                             max_chunks=args.max_chunks)
        val = evaluate(net, head, validation, cfg, q99, device, full=guarded_selection)
        sub = evaluate(net, head, subset_data, cfg, q99, device) if subset_due(epoch, int(cfg["epochs"]), int(cfg["train_subset_every"])) else None
        record = dict(result, epoch=epoch, lr=lr, val_iou=val["metrics"]["iou"], val_acc=val["metrics"]["acc"],
                      val_readouts=val["readouts"], train_subset_readouts=sub["readouts"] if sub else None,
                      epoch_seconds=time.perf_counter() - t0, tau=tau_statistics(net),
                      peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None)
        if record["val_iou"] is None or not math.isfinite(record["val_iou"]):
            raise RuntimeError("Validation IoU is not finite")
        if guarded_selection:
            from tools.v16.train_covariance import four_metric_noninferiority
            record["selection_eligible"] = four_metric_noninferiority(val["metrics"], selection_reference)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        history.append(record)
        if record.get("selection_eligible", True) and record["val_iou"] > best:
            best = record["val_iou"]
            best_epoch = epoch
            save_checkpoint(out / ("best_val_iou_seed%d.pt" % cfg["seed"]), net, head, optimizer, epoch, best, cfg, stats, comm)
        save_checkpoint(out / "last.pt", net, head, optimizer, epoch, best, cfg, stats, comm)
        patience = int(cfg.get("early_stop_patience", 0))
        early_stopped = bool(patience and epoch + 1 >= int(cfg.get("min_epochs", 1)) and epoch - best_epoch >= patience)
        write_json(out / "summary.json", {"design": design_name(cfg), "epochs_completed": epoch + 1, "epochs_requested": cfg["epochs"],
                   "best_val_iou": best,
                   "best_epoch": best_epoch + 1, "early_stopped": early_stopped,
                   "stopping_rule": ("eligible validation IoU patience; all four metrics noninferior to initial validation; test never used"
                                     if guarded_selection else "validation IoU patience only; test never used"),
                   "selection_reference": selection_reference if guarded_selection else None,
                   "complete": (epoch + 1 == int(cfg["epochs"]) or early_stopped) and args.limit_records is None and args.max_chunks is None,
                   "training_scope": cfg["_training_scope"],
                   "probe_only": args.limit_records is not None or args.max_chunks is not None,
                   "train_seconds_total": sum(r["train_seconds"] for r in history),
                   "epoch_seconds_total": sum(r["epoch_seconds"] for r in history),
                   "seed": cfg["seed"], "revision_variant": cfg["revision_variant"]})
        print("EPOCH %d/%d train %.1fs total %.1fs loss %.5f val IoU %.5f ACC %.5f comm_mu %.4f" %
              (epoch + 1, cfg["epochs"], result["train_seconds"], record["epoch_seconds"], result["loss_per_event"],
               record["val_iou"], record["val_acc"], comm.mu), flush=True)
        if early_stopped:
            print("EARLY STOP after %d epochs without an eligible higher validation IoU" % patience, flush=True)
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v16/full.yaml")
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
        if ckpt.get("design") != design_name(ckpt.get("config", {})):
            raise ValueError("Not a V16 context checkpoint")
        original = ckpt["config"]
        # Architecture, thresholds and time bases come from the trained model.
        for key in original:
            if not key.startswith("_") and key not in ("root", "save_root", "stats_path") and cfg.get(key) != original[key]:
                raise ValueError("Evaluation config differs from checkpoint: " + key)
        seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
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
