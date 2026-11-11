"""Real stopped-query execution over a chronological event stream.

The input Recording may hold an offline evaluation file, but source features
are encoded lazily only after their acquisition window has completed. Committed
queries never re-enter SDU computation. Their observations remain eligible as
evidence for other live queries. A bounded history cache is shared by all live
queries, and fixed old-neighbour selections are retained between updates.

This executor does not turn sparse arithmetic into a claim of measured GPU
speed-up. It reports actual query updates, source encodings and pair visits.
Raw input/returned event arrays are O(N); retained learned features are bounded
by the explicit past-plus-wait horizon and observed event rate.
"""

import math
import time

import numpy as np
import torch
from torch import nn

from model.bst import concat_states, posterior_decision, take_state
from tools.bst.data import NeighborIndex, Recording


def _logit(probability):
    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    return math.log(p / (1.0 - p))


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))


def _birth_mask(prior, tau, alpha, kind):
    alpha = float(alpha)
    if not 0.0 < alpha < 0.5:
        raise ValueError("band_alpha must lie in (0, 0.5)")
    if kind == "sym":
        half = _logit(1.0 - alpha)
        return (prior > tau - half) & (prior < tau + half)
    if kind == "martingale":
        threshold = _sigmoid(tau)
        lo = _logit(alpha * threshold)
        hi = _logit(1.0 - alpha * (1.0 - threshold))
        return (prior > lo) & (prior < hi)
    raise ValueError("band_kind must be sym or martingale")


class CausalSourceCache:
    """One learned encoding per observed source, with temporal eviction."""

    def __init__(self, model, recording, device, batch_size, source_features=None):
        self.model, self.recording = model, recording
        self.device, self.batch_size = torch.device(device), int(batch_size)
        self.features = recording.source_features if source_features is None else np.asarray(source_features, np.float32)
        if self.features.shape != (len(recording), model.config.source_features):
            raise ValueError("source_features must match recording and model input dimension")
        self.source_ids = np.empty(0, np.int64)
        self.lookup = np.full(len(recording), -1, np.int64)
        self.encoded = model.encode_sources(torch.empty((0, model.config.source_features), device=self.device))
        self.encoded_total = 0
        self.peak_sources = 0
        self.peak_bytes = 0
        self.max_encoded_window = -1

    def ensure(self, source_ids, current_window):
        ids = np.unique(np.asarray(source_ids, np.int64))
        if ((ids < 0) | (ids >= len(self.recording))).any():
            raise IndexError("source id outside recording")
        if len(ids) and (self.recording.window[ids] > int(current_window)).any():
            raise ValueError("attempted to encode an unobserved future source")
        missing = ids[self.lookup[ids] < 0]
        if not len(missing):
            return
        parts = []
        for start in range(0, len(missing), self.batch_size):
            rows = missing[start:start + self.batch_size]
            x = torch.as_tensor(self.features[rows], dtype=torch.float32, device=self.device)
            parts.append(self.model.encode_sources(x))
        fresh = {key: torch.cat([p[key] for p in parts], 0) for key in self.encoded}
        offset = len(self.source_ids)
        self.lookup[missing] = np.arange(offset, offset + len(missing))
        self.source_ids = np.concatenate([self.source_ids, missing])
        self.encoded = {key: torch.cat([self.encoded[key], fresh[key]], 0) for key in self.encoded}
        self.encoded_total += len(missing)
        self.peak_sources = max(self.peak_sources, len(self.source_ids))
        self.peak_bytes = max(self.peak_bytes, sum(v.numel() * v.element_size() for v in self.encoded.values()))
        self.max_encoded_window = max(self.max_encoded_window, int(self.recording.window[missing].max()))

    def local_indices(self, global_indices):
        global_indices = np.asarray(global_indices, np.int64)
        mask = global_indices >= 0
        result = np.full(global_indices.shape, -1, np.int64)
        result[mask] = self.lookup[global_indices[mask]]
        if (result[mask] < 0).any():
            raise RuntimeError("required historical source was not encoded or was evicted too early")
        return result

    def evict_before(self, window):
        keep = self.recording.window[self.source_ids] >= int(window)
        if keep.all():
            return
        self.lookup[self.source_ids] = -1
        self.source_ids = self.source_ids[keep]
        self.lookup[self.source_ids] = np.arange(len(self.source_ids))
        select = torch.as_tensor(np.flatnonzero(keep), dtype=torch.long, device=self.device)
        self.encoded = {key: value.index_select(0, select) for key, value in self.encoded.items()}


def _replace_state_rows(state, rows, replacement):
    for key in state:
        state[key][rows] = replacement[key]


class CausalPairCache:
    """Inference-only, bounded by peak live queries, not total stream length.

    Cache ONLY immutable source/relative encodings. Gate weights, evidence
    values, posterior corrections and context are recomputed by reaudit.
    Stable reusable slots avoid copying all old pairs when the queue changes.
    """

    def __init__(self, model, source_cache, slots, batch_size):
        self.model, self.source = model, source_cache
        self.slots, self.batch_size = int(slots), int(batch_size)
        self.capacity, self.free, self.storage = 0, [], {}
        self.prepared_pairs = 0
        self.prepared_slots = 0
        c = self.model.config
        self.layout, offset = {}, 0
        for name, tail in (("key", (c.heads, c.relation_dim)), ("value", (c.heads, c.relation_dim)),
                           ("context", (c.state_dim,)), ("bias", (c.heads,)), ("dw", ()), ("relative", (4,))):
            width = int(np.prod(tail)) if tail else 1
            self.layout[name] = (offset, offset + width, tail)
            offset += width
        self.width = offset

    def allocate(self, count):
        if len(self.free) < count:
            old = self.capacity
            capacity = max(old * 2, old + count - len(self.free), min(256, self.batch_size))
            device = self.source.device
            shapes = {"packed": (self.width,), "indices": ()}
            dtype = next(self.model.parameters()).dtype
            for name, tail in shapes.items():
                kind = torch.long if name == "indices" else dtype
                data = torch.full((capacity, self.slots) + tail, -1 if name == "indices" else 0, dtype=kind, device=device)
                if old:
                    data[:old].copy_(self.storage[name])
                self.storage[name] = data
            self.free.extend(range(capacity - 1, old - 1, -1))
            self.capacity = capacity
        ids = np.asarray([self.free.pop() for _ in range(count)], np.int64)
        if len(ids):
            rows = torch.as_tensor(ids, device=self.source.device)
            # Invalid slots must be clean when a committed query's slot is reused.
            self.storage["packed"][rows] = 0
            self.storage["indices"][rows] = -1
        return ids

    def append(self, pool_rows, global_indices, relative, starts):
        """Encode an observed append, preserving the exact original slot order."""
        width = global_indices.shape[1]
        starts = np.broadcast_to(np.asarray(starts, np.int64), (len(pool_rows),)).copy()
        useful = (global_indices >= 0).any(axis=1)
        pool_rows, global_indices, relative, starts = (value[useful] for value in
                                                      (pool_rows, global_indices, relative, starts))
        for begin in range(0, len(pool_rows), self.batch_size):
            sl = slice(begin, begin + self.batch_size)
            ids = global_indices[sl]
            local = torch.as_tensor(self.source.local_indices(ids), device=self.source.device)
            rel = torch.as_tensor(relative[sl], device=self.source.device)
            pairs = self.model.prepare_pairs(self.source.encoded, local, rel)
            rows = torch.as_tensor(pool_rows[sl], device=self.source.device)[:, None]
            cols = torch.as_tensor(starts[sl], device=self.source.device)[:, None] + torch.arange(width, device=self.source.device)
            # One contiguous transfer/scatter instead of a kernel per field.
            values = dict(pairs, relative=rel)
            packed = torch.cat([values[name].reshape(len(ids), width, -1) for name in self.layout], dim=-1)
            self.storage["packed"][rows, cols] = packed
            self.storage["indices"][rows, cols] = torch.as_tensor(ids, device=self.source.device)
            self.prepared_pairs += int((ids >= 0).sum())
            self.prepared_slots += int(ids.size)

    def read(self, pool_rows, end):
        rows = torch.as_tensor(pool_rows, device=self.source.device)
        # Slice before gathering: do not move impossible future slots.
        packed = self.storage["packed"][:, :end][rows]
        indices = self.storage["indices"][:, :end][rows]
        result = {name: packed[..., start:stop].reshape((len(pool_rows), end) + tail)
                  for name, (start, stop, tail) in self.layout.items()}
        result.update(indices=indices, present=indices >= 0)
        return result

    def release(self, rows):
        self.free.extend(np.asarray(rows, np.int64).tolist())


def run_recording(model, rec, evidence_config=None, policy=None, birth="band", threshold=0.9,
                  band_alpha=0.01, batch_size=256, device="cpu", wait_cost=0.0,
                  fp_cost=None, fn_cost=None, birth_policy=None, band_kind="sym",
                  return_traces=False, prior_logits=None, source_features=None,
                  arrival_provider=None, source_batch_size=4096):
    """Run chronological inference and return ``logits``, ``lags`` and ``stats``.

    ``logits`` use the repository's z0 scale. Official reporting applies
    sigmoid(logits + rec.log_w) and ``threshold``. The implied model-probability
    threshold determines FP/FN costs unless both are explicitly supplied.

    ``policy=None`` waits until the deadline. An nn.Module predicts future
    classification loss; commit when risk <= wait_cost + its prediction. A
    callable ``policy(features, stop_risk, age)`` must instead return a bool
    tensor with True meaning commit. No policy sees labels or future states.

    ``birth='learned'`` requires ``birth_policy(ids_numpy, rec, current_window)``
    returning bool admissions from already available features; it runs BEFORE
    SDU state creation or neighbour search. The callback is responsible for
    respecting the supplied availability window (the recording is offline).

    Optional traces are event-query snapshots, not a precomputed full trajectory.
    This routine is inference-only; training unrolls are handled separately.

    ``arrival_provider(window, original_ids)`` optionally executes a live BIR
    window and returns {"prior": float[B], "source_features": float[B,C]}.
    It runs before any access to that window, including empty windows. Cached
    trajectory scores are then unnecessary. The caller's Recording is not
    mutated, and future source features remain unavailable to the cache.
    """
    if birth not in ("all", "band", "learned"):
        raise ValueError("birth must be all, band or learned")
    if birth == "learned" and birth_policy is None:
        raise ValueError("learned birth requires a pre-SDU birth_policy")
    if int(batch_size) != batch_size or int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if int(source_batch_size) != source_batch_size or int(source_batch_size) < 1:
        raise ValueError("source_batch_size must be positive")
    if not np.isfinite(wait_cost) or wait_cost < 0:
        raise ValueError("wait_cost must be finite and nonnegative")
    if arrival_provider is not None:
        if prior_logits is not None:
            raise ValueError("arrival_provider supplies priors; do not also pass prior_logits")
        # Copy mutable predictions, not the original-order coordinate arrays.
        # The initial future scores are irrelevant: no tree/encoder reads them.
        rec = Recording(rec.name, rec.x, rec.y, rec.t, rec.p, rec.z0.copy(), rec.label,
                        rec.window_ms, rec.log_w, rec.n_windows, rec.target_id)
        if source_features is not None:
            source_features = np.asarray(source_features, np.float32).copy()
    batch_size = int(batch_size)
    config = {} if evidence_config is None else dict(evidence_config)
    index = NeighborIndex(rec, **config)
    if index.max_wait != model.config.max_wait_steps:
        raise ValueError("evidence max_wait must match model max_wait_steps")
    if not math.isclose(float(rec.window_ms), float(model.config.window_ms)):
        raise ValueError("recording window_ms disagrees with the model")
    tau = _logit(threshold) - rec.log_w
    if fp_cost is None and fn_cost is None:
        fp_cost = _sigmoid(tau)
        fn_cost = 1.0 - fp_cost
    elif fp_cost is None or fn_cost is None:
        raise ValueError("supply both fp_cost and fn_cost")
    if fp_cost <= 0 or fn_cost <= 0 or not np.isfinite(fp_cost + fn_cost):
        raise ValueError("decision costs must be positive and finite")
    if not math.isclose(math.log(fp_cost / fn_cost), tau, abs_tol=1e-5):
        raise ValueError("decision cost ratio must match official reporting threshold")
    prior = rec.z0 if prior_logits is None else np.asarray(prior_logits, np.float32)
    if prior.shape != (len(rec),) or not np.isfinite(prior).all():
        raise ValueError("prior_logits must be finite with one row per event")
    device = torch.device(device)
    model_training = model.training
    policy_training = policy.training if isinstance(policy, nn.Module) else None
    model.to(device).eval()
    if isinstance(policy, nn.Module):
        policy.to(device).eval()
    logits = np.asarray(prior, np.float32).copy()
    lags = np.zeros(len(rec), np.int64)
    reasons = np.full(len(rec), "birth", dtype="<U8")
    active = np.empty(0, np.int64)
    pair_rows = np.empty(0, np.int64)
    counts = np.empty(0, np.int64)
    state = None
    traces = []
    stats = {"events": len(rec), "births": 0, "queries_updated": 0, "decision_checks": 0,
             "pair_visits": 0, "max_active": 0, "source_encoded": 0,
             "deadline_commits": 0, "policy_commits": 0, "birth_commits": 0,
             "empty_prefix_checks": 0, "cache_peak_sources": 0, "cache_peak_bytes": 0}
    started = time.perf_counter()
    try:
        with torch.no_grad():
            # Source MLPs are small and per-row: larger encoding batches reduce
            # launches without changing evidence budgets or SDU query batches.
            cache = CausalSourceCache(model, rec, device, source_batch_size, source_features)
            pairs_cache = CausalPairCache(model, cache, index.slots_per_query, batch_size)
            for window, arrived in rec.iter_windows(include_empty=True):
                if arrival_provider is not None:
                    arrival = arrival_provider(window, arrived.copy())
                    if not isinstance(arrival, dict) or "prior" not in arrival:
                        raise ValueError("arrival_provider must return a dict containing prior")
                    supplied_prior = np.asarray(arrival["prior"], np.float32)
                    if supplied_prior.shape != (len(arrived),) or not np.isfinite(supplied_prior).all():
                        raise ValueError("arrival prior must be finite float[events_arriving_now]")
                    supplied_features = arrival.get("source_features")
                    if supplied_features is None:
                        supplied_features = np.stack([supplied_prior, rec.p[arrived]], axis=1)
                    supplied_features = np.asarray(supplied_features, np.float32)
                    if supplied_features.shape != (len(arrived), model.config.source_features):
                        raise ValueError("arrival source_features have an incompatible shape")
                    if not np.isfinite(supplied_features).all():
                        raise ValueError("arrival source_features must be finite")
                    if (cache.lookup[arrived] >= 0).any():
                        raise RuntimeError("arrival callback attempted to change an encoded source")
                    rec.z0[arrived] = supplied_prior
                    prior[arrived] = supplied_prior
                    logits[arrived] = supplied_prior
                    # Neighbor filtering and cheap birth see the live arrival score.
                    rec._source_features[arrived] = np.stack([supplied_prior, rec.p[arrived]], axis=1)
                    cache.features[arrived] = supplied_features
                earliest = window - index.past_windows - index.max_wait
                cache.evict_before(earliest)
                index.evict_before(earliest)
                appends = []
                previous_counts = counts.copy()
                # Every live query retains exactly its old candidate identities.
                if len(active):
                    state["previous_score"] = state["score"].clone()
                    new = index.query_arriving(active, current_window=window)
                    counts += new["mask"].sum(axis=1)
                    appends.append((pair_rows, new["idx"], new["relative"], new["slot"]))
                if birth == "all":
                    admit = np.ones(len(arrived), bool)
                elif birth == "band":
                    admit = _birth_mask(prior[arrived], tau, band_alpha, band_kind)
                else:
                    admission = birth_policy(arrived, rec, window)
                    if torch.is_tensor(admission):
                        admission = admission.detach().cpu().numpy()
                    admit = np.asarray(admission)
                    if admit.shape != (len(arrived),) or admit.dtype != np.bool_:
                        raise ValueError("birth_policy must return bool[events_arriving_now]")
                born = arrived[admit]
                stats["birth_commits"] += int((~admit).sum())
                if len(born):
                    initial = index.query(born, age=0)
                    p0 = torch.as_tensor(prior[born], device=device)
                    polarity = torch.as_tensor(rec.p[born], device=device)
                    born_state = model.make_state(p0, polarity)
                    state = born_state if state is None else concat_states([state, born_state])
                    active = np.concatenate([active, born])
                    new_rows = pairs_cache.allocate(len(born))
                    pair_rows = np.concatenate([pair_rows, new_rows])
                    counts = np.concatenate([counts, initial["mask"].sum(axis=1)])
                    previous_counts = np.concatenate([previous_counts, np.full(len(born), -1, np.int64)])
                    end = (index.past_windows + 1) * index.K
                    appends.append((new_rows, initial["idx"][:, :end], initial["relative"][:, :end], 0))
                    stats["births"] += len(born)
                stats["max_active"] = max(stats["max_active"], len(active))
                if not len(active):
                    if return_traces:
                        traces.append({"window": window, "ids": active.copy(), "logits": np.empty(0),
                                       "features": np.empty((0, model.value_feature_dim), np.float32),
                                       "committed": np.empty(0, bool), "updated": np.empty(0, bool),
                                       "max_encoded_window": cache.max_encoded_window})
                    continue
                ages = window - rec.window[active]
                age_tensor = torch.as_tensor(ages, dtype=state["score"].dtype, device=device)
                state["age"] = age_tensor
                changed = (counts > previous_counts) & (counts > 0)
                stats["empty_prefix_checks"] += int((counts == 0).sum())
                rows_to_update = np.flatnonzero(changed)
                # Only actually updated queries cause source encoding and SDU work.
                if appends:
                    # Old sources were encoded already; never rescan the old prefix.
                    needed = np.concatenate([ids[ids >= 0] for _, ids, _, _ in appends])
                    cache.ensure(needed, window)
                    for pool, ids, rel, starts in appends:
                        pairs_cache.append(pool, ids, rel, starts)
                for begin in range(0, len(rows_to_update), batch_size):
                    rows_np = rows_to_update[begin:begin + batch_size]
                    rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
                    substate = take_state(state, rows)
                    # Fixed queue storage, but no kernels over impossible future slots.
                    last_slot = (index.past_windows + int(ages[rows_np].max()) + 1) * index.K
                    pairs = pairs_cache.read(pair_rows[rows_np], last_slot)
                    _, updated, _ = model.reaudit(cache.encoded, pairs["indices"], pairs["relative"], substate,
                                                  substate["prior"], substate["polarity"], age_tensor[rows], pair_cache=pairs)
                    _replace_state_rows(state, rows, updated)
                    stats["queries_updated"] += len(rows_np)
                    stats["pair_visits"] += int(counts[rows_np].sum())
                _, risks = posterior_decision(state["score"], fp_cost, fn_cost)
                deadline = (ages >= index.max_wait) | (window == rec.n_windows - 1)
                stop = deadline.copy()
                if policy is not None:
                    candidates = np.flatnonzero(~deadline)
                    decisions, invalid_predictions = [], []
                    for begin in range(0, len(candidates), batch_size):
                        rows_np = candidates[begin:begin + batch_size]
                        rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
                        substate = take_state(state, rows)
                        features = model.value_features(substate, age=age_tensor[rows])
                        if isinstance(policy, nn.Module):
                            prediction = policy(features)
                            if prediction.shape != risks[rows].shape:
                                raise ValueError("continuation head must return finite nonnegative loss[B]")
                            invalid_predictions.append((~torch.isfinite(prediction) | (prediction < 0)).any().reshape(1))
                            decision = risks[rows] <= float(wait_cost) + prediction
                        else:
                            decision = policy(features, risks[rows], age_tensor[rows])
                        decision = torch.as_tensor(decision, device=device)
                        if decision.shape != risks[rows].shape or decision.dtype != torch.bool:
                            raise ValueError("policy callable must return bool[B], True means commit")
                        decisions.append(decision)
                    if decisions:
                        # One D2H synchronization per window, not per minibatch
                        # and per validity check. Invalid outputs still fail.
                        checked = torch.cat(decisions + invalid_predictions).cpu().numpy()
                        if checked[len(candidates):].any():
                            raise ValueError("continuation head must return finite nonnegative loss[B]")
                        stop[candidates] = checked[:len(candidates)]
                stats["decision_checks"] += len(active)
                if return_traces:
                    traces.append({"window": window, "ids": active.copy(),
                                   "logits": state["score"].cpu().numpy().copy(),
                                   "features": model.value_features(state, age=age_tensor).cpu().numpy().copy(),
                                   "committed": stop.copy(), "updated": changed.copy(),
                                   "max_encoded_window": cache.max_encoded_window,
                                   "cached_sources": len(cache.source_ids)})
                emit = np.flatnonzero(stop)
                logits[active[emit]] = state["score"][torch.as_tensor(emit, device=device)].cpu().numpy()
                lags[active[emit]] = ages[emit]
                reasons[active[emit]] = np.where(deadline[emit], "deadline", "policy")
                stats["deadline_commits"] += int(deadline[emit].sum())
                stats["policy_commits"] += int((~deadline[emit]).sum())
                keep = np.flatnonzero(~stop)
                pairs_cache.release(pair_rows[emit])
                active, pair_rows, counts = active[keep], pair_rows[keep], counts[keep]
                state = take_state(state, torch.as_tensor(keep, dtype=torch.long, device=device)) if len(keep) else None
            stats["source_encoded"] = cache.encoded_total
            stats["cache_peak_sources"] = cache.peak_sources
            stats["cache_peak_bytes"] = cache.peak_bytes
            stats["trees_built"] = index.trees_built
            stats["remaining_active"] = len(active)
            stats["pair_cache_capacity"] = pairs_cache.capacity
            stats["pair_cache_bytes"] = sum(v.numel() * v.element_size() for v in pairs_cache.storage.values())
            stats["pairs_prepared"] = pairs_cache.prepared_pairs
            stats["pair_slots_prepared"] = pairs_cache.prepared_slots
    finally:
        model.train(model_training)
        if isinstance(policy, nn.Module):
            policy.train(policy_training)
    stats["seconds"] = time.perf_counter() - started
    stats["mean_extra_wait_ms"] = float(lags.mean() * rec.window_ms) if len(rec) else 0.0
    stats["max_extra_wait_ms"] = int(lags.max() * rec.window_ms) if len(rec) else 0
    # Window accumulation delay is separate from extra waiting and compute time.
    window_wait = rec.window_ms - (rec.t % rec.window_ms)
    stats["mean_acquisition_wait_ms"] = float(window_wait.mean()) if len(rec) else 0.0
    result = {"logits": logits, "committed_logits": logits, "lags": lags, "reasons": reasons,
              "stats": stats, "decision_logit": tau}
    if return_traces:
        result["traces"] = traces
    return result
