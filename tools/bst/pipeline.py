"""Shared training utilities. Source caches live for one optimizer graph only."""
import dataclasses
import glob
import hashlib
import json
import math
import os
import warnings
import uuid

import numpy as np
import torch

from model.bst import BSTConfig, SharedSequentialModel
from tools.bst.data import NeighborIndex, compact_neighbors, load_recording, sample_queries
from tools.bst.training_geometry import GeometryBatches
from tools.deco.evidence import band_bounds


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    BSTConfig(**cfg["model"])
    if cfg["model"].get("normalization") == "window" and cfg["model"].get("past_windows", 10) != cfg["evidence"]["past_windows"]:
        raise ValueError("window aggregation and evidence past horizons differ")
    if int(cfg["evidence"]["max_wait"]) != int(cfg["model"]["max_wait_steps"]):
        raise ValueError("model and evidence horizons differ")
    if not 0 < cfg["threshold"] < 1:
        raise ValueError("threshold must be a probability")
    return cfg


def save_json(path, value):
    def finite_json(item):
        if isinstance(item, dict):
            return {key: finite_json(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [finite_json(val) for val in item]
        if isinstance(item, np.ndarray):
            return finite_json(item.tolist())
        if isinstance(item, np.generic):
            return finite_json(item.item())
        return None if isinstance(item, float) and not math.isfinite(item) else item
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = str(path) + "." + uuid.uuid4().hex[:8] + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as fh:
            json.dump(finite_json(value), fh, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prior_provenance(z0_root, splits=("train", "val")):
    metadata = {}
    for split in splits:
        path = os.path.join(z0_root, "z0_" + split, "prior_provenance.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                metadata[split] = json.load(fh)
    hashes = {entry["backbone_sha256"] for entry in metadata.values()}
    if len(hashes) > 1:
        raise ValueError("prior splits were exported from different BIR checkpoints")
    verified = len(metadata) == len(splits)
    complete = verified and all(entry.get("complete", False) for entry in metadata.values())
    if not verified:
        warnings.warn("Legacy prior cache has no complete BIR identity metadata; provenance is unverified.")
    return {"verified": verified, "complete": complete,
            "backbone_sha256": next(iter(hashes)) if hashes else None, "splits": metadata}


def checkpoint_scope(checkpoint, data_root):
    """Keep a shortened training run labelled as a probe during full inference."""
    manifest = checkpoint.get("manifest", checkpoint)
    observed = {"train": set(manifest.get("train", [])),
                "val": set(manifest.get("policy_fit", [])) | set(manifest.get("selection", []))}
    coverage = {}
    for split, names in observed.items():
        available = {os.path.basename(p) for p in glob.glob(os.path.join(data_root, split, "*.npz"))}
        coverage[split] = {"used": len(names), "available": len(available),
                           "complete": bool(available) and names == available}
    return {"probe": not all(entry["complete"] for entry in coverage.values()), "coverage": coverage}


def torch_load(path, device="cpu"):
    # Explicit weights_only=False for our trusted local checkpoint dictionary;
    # keep compatibility with the server's torch 1.9 API.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(path, device="cpu"):
    ck = torch_load(path, device)
    model = SharedSequentialModel(BSTConfig(**ck["config"]["model"])).to(device)
    model.load_state_dict(ck["model"])
    return model, ck


def paths(data_root, z0_root, split, names=None, limit=0):
    folder = os.path.join(z0_root, "z0_" + split)
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if names is not None:
        wanted = set(names)
        files = [p for p in files if os.path.basename(p) in wanted]
        missing = wanted - {os.path.basename(p) for p in files}
        if missing:
            raise ValueError("missing prior recordings: %s" % sorted(missing))
    if limit:
        files = files[:int(limit)]
    if not files:
        raise ValueError("no recordings in %s" % folder)
    pairs = [(os.path.join(data_root, split, os.path.basename(p)), p) for p in files]
    for raw, _ in pairs:
        if not os.path.isfile(raw):
            raise ValueError("missing raw recording %s" % raw)
    return pairs


def validation_partition(pairs):
    """Disjoint recordings for VoW fitting and SDU/policy selection; never test."""
    if len(pairs) < 2:
        raise ValueError("at least two validation recordings are required for disjoint fitting/selection")
    return pairs[::2], pairs[1::2]


def get_record(pair, config):
    options = {"window_ms": config["model"]["window_ms"]}
    if config.get("recording_n_windows") is not None:
        options["n_windows"] = config["recording_n_windows"]
    rec = load_recording(pair[0], pair[1], **options)
    provenance_path = os.path.join(os.path.dirname(pair[1]), "prior_provenance.json")
    if os.path.isfile(provenance_path):
        with open(provenance_path, encoding="utf-8") as fh:
            provenance = json.load(fh)
        if rec.backbone_sha256 != provenance.get("backbone_sha256"):
            raise ValueError("cached recording identity differs from its directory provenance")
    return rec, NeighborIndex(rec, **config["evidence"])


def choose_queries(rec, background_budget, rng):
    # Every positive is retained. Uniform negatives have known inclusion
    # probabilities; raw sources are NEVER sampled or removed.
    if not background_budget:
        return sample_queries(rec, rng=rng)
    return sample_queries(rec, rng=rng, strata=rec.label.astype(np.int64),
                          per_stratum={0: int(background_budget), 1: max(len(rec), 1)})


def threshold_costs(threshold, log_w):
    """Match the official report decision, rather than silently using p=0.5."""
    tau = math.log(threshold / (1.0 - threshold)) - float(log_w)
    cost_fp = 1.0 / (1.0 + math.exp(-tau))
    return tau, cost_fp, 1.0 - cost_fp


def rollout(model, rec, index, ids, device="cpu", collect_features=False,
            prepared_batch=None, trim_prefix=True):
    """Full causal prefixes for TRAINING; not an early-exit timing benchmark.

    Only selected queries construct neighbours. Each unique source is encoded
    once per minibatch and reused across prefixes; future slots are masked in
    the model. LayerNorm is per source, so future rows cannot affect past rows.
    """
    D = model.config.max_wait_steps
    batch = index.query(ids, age=D) if prepared_batch is None else prepared_batch
    if prepared_batch is not None and not np.array_equal(batch["query_ids"], ids):
        raise ValueError("prepared geometry must preserve query identity and order")
    sources, local = compact_neighbors(batch)
    source = torch.as_tensor(rec.source_features[sources], device=device)
    encoded = model.encode_sources(source)
    neighbors = torch.as_tensor(local, dtype=torch.long, device=device)
    relative = torch.as_tensor(batch["relative"], device=device)
    pair_cache = model.prepare_pairs(encoded, neighbors, relative)
    prior = torch.as_tensor(rec.z0[ids], device=device)
    polarity = torch.as_tensor(rec.p[ids], device=device)
    maximum = torch.as_tensor(batch["max_age"], dtype=torch.long, device=device)
    state = model.make_state(prior, polarity)
    scores, features, masks = [], [], []
    for age in range(D + 1):
        valid = maximum >= age
        # Future slots are identically masked. Do not run their dynamic heads
        # before they arrive; retain every old slot for re-audit and gradients.
        end = min(neighbors.shape[1], (index.past_windows + age + 1) * index.K) if trim_prefix else neighbors.shape[1]
        visible_pairs = {key: value[:, :end] for key, value in pair_cache.items()}
        # This owned state is initialized at age zero above and never reordered;
        # no per-age host synchronization is needed for the public API guard.
        score, state, _ = model.reaudit(encoded, neighbors[:, :end], relative[:, :end], state, prior, polarity, age,
                                        valid=valid, pair_cache=visible_pairs, validate_birth=False)
        scores.append(score)
        masks.append(valid)
        if collect_features:
            features.append(model.value_features(state, age=age))
    return {"logits": torch.stack(scores, 1), "valid": torch.stack(masks, 1),
            "features": torch.stack(features, 1) if features else None,
            "source_count": len(sources), "pair_count": int(batch["mask"].sum()),
            "max_age": maximum}


def weighted_event_loss(logits, valid, labels, weights):
    """Per-event prefix average; inverse inclusion weights remain explicit."""
    ell = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels[:, None].expand_as(logits), reduction="none")
    per_event = (ell * valid.to(ell.dtype)).sum(1) / valid.sum(1).clamp(min=1)
    return (per_event * weights).sum()


def evaluate_sampled(model, pairs, config, device, background_budget=0, seed=37):
    """Validation proxy only. Final reported metrics use all events."""
    model.eval()
    total_loss, population = 0.0, 0
    tp = fp = fn = 0.0
    band_tp = band_fp = band_fn = 0.0
    expected_log_w = None
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for pair in pairs:
            rec, idx = get_record(pair, config)
            if expected_log_w is None:
                expected_log_w = rec.log_w
            elif not np.isclose(rec.log_w, expected_log_w, atol=1e-6, rtol=0):
                raise ValueError("mixed backbone score scales in validation")
            qs = choose_queries(rec, background_budget, rng)
            population += len(rec)
            tau, _, _ = threshold_costs(config["threshold"], rec.log_w)
            geometry = GeometryBatches(idx, qs.ids, config["train"]["batch_size"],
                                       config["train"].get("geometry_cache_mb", 64))
            for start, stop, prepared in geometry:
                ids = qs.ids[start:stop]
                weights = torch.as_tensor(qs.weights[start:start + len(ids)], dtype=torch.float32, device=device)
                y = torch.as_tensor(rec.label[ids], dtype=torch.float32, device=device)
                result = rollout(model, rec, idx, ids, device, prepared_batch=prepared)
                total_loss += float(weighted_event_loss(result["logits"], result["valid"], y, weights))
                z = result["logits"].gather(1, result["max_age"][:, None]).squeeze(1)
                pred, pos = z >= tau, y > 0.5
                tp += float(weights[pred & pos].sum())
                fp += float(weights[pred & ~pos].sum())
                fn += float(weights[~pred & pos].sum())
                prior = torch.as_tensor(rec.z0[ids], device=device)
                alpha = config["policy"]["band_alpha"]
                lo, hi = band_bounds(tau, alpha, config["policy"].get("band_kind", "sym"))
                opened = (prior > lo) & (prior < hi)
                band_pred = torch.where(opened, z, prior) >= tau
                band_tp += float(weights[band_pred & pos].sum())
                band_fp += float(weights[band_pred & ~pos].sum())
                band_fn += float(weights[~band_pred & pos].sum())
    return {"weighted_prefix_bce": total_loss / max(population, 1),
            "weighted_deadline_iou": tp / max(tp + fp + fn, 1),
            "weighted_band_iou": band_tp / max(band_tp + band_fp + band_fn, 1),
            "sampled_proxy": bool(background_budget), "population": population}
