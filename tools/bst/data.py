"""Indexed evidence for shared sequential inference.

Each source event is stored once. Query batches hold integer references rather
than duplicated learned features. A query becomes available at the END of its
arrival window, matching the BIR backbone; this is window-causal, not an
assertion of event-timestamp causality within that window.

Only currently observed windows participate in neighbour selection. The fixed
per-window neighbourhoods grow monotonically with query age, which lets the
decision model re-audit old evidence without replacing it or counting it twice.
All code is compatible with Python 3.8 and NumPy/SciPy used by the legacy repo.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional, Tuple
import warnings

import numpy as np
from scipy.spatial import cKDTree


def _integer_vector(value, name):
    value = np.asarray(value)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("%s must be a finite vector" % name)
    if not np.equal(value, np.rint(value)).all():
        raise ValueError("%s must contain integers (timestamps are milliseconds)" % name)
    return value.astype(np.int64, copy=False)


@dataclass
class Recording:
    """One original-order event stream and its frozen arrival scores.

    ``z0`` is the backbone logit with log(class_weight) subtracted. This shift
    alone does not establish empirical probability calibration. ``label`` is
    retained for training/evaluation only and never read by NeighborIndex.
    """

    name: str
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray
    z0: np.ndarray
    label: np.ndarray
    window_ms: int
    log_w: float = 0.0
    n_windows: Optional[int] = None
    target_id: Optional[np.ndarray] = None
    backbone_sha256: Optional[str] = None
    window: np.ndarray = field(init=False, repr=False)
    order: np.ndarray = field(init=False, repr=False)
    bounds: np.ndarray = field(init=False, repr=False)
    _source_features: np.ndarray = field(init=False, repr=False)
    horizon_source: str = field(init=False, default="explicit")

    def __post_init__(self):
        self.name = str(self.name)
        self.x = _integer_vector(self.x, "x")
        self.y = _integer_vector(self.y, "y")
        self.t = _integer_vector(self.t, "t")
        self.p = np.asarray(self.p, dtype=np.float32)
        self.z0 = np.asarray(self.z0, dtype=np.float32)
        self.label = np.asarray(self.label, dtype=np.float32)
        n = len(self.t)
        for key in ("x", "y", "p", "z0", "label"):
            v = getattr(self, key)
            if v.ndim != 1 or len(v) != n:
                raise ValueError("%s must have one row per original event" % key)
            if not np.isfinite(v).all():
                raise ValueError("%s contains non-finite values" % key)
        if (self.x < 0).any() or (self.y < 0).any() or (self.t < 0).any():
            raise ValueError("coordinates and timestamps must be nonnegative")
        if not np.isin(self.label, [0, 1]).all():
            raise ValueError("labels must be binary")
        if not np.isin(self.p, [-1, 0, 1]).all():
            raise ValueError("polarities must use -1/+1 or 0/1 encoding")
        if int(self.window_ms) != self.window_ms or int(self.window_ms) <= 0:
            raise ValueError("window_ms must be a positive integer")
        self.window_ms = int(self.window_ms)
        self.log_w = float(self.log_w)
        if not np.isfinite(self.log_w):
            raise ValueError("log_w must be finite")
        self.window = self.t // self.window_ms
        minimum = int(self.window.max()) + 1 if n else 1
        if self.n_windows is None:
            self.n_windows = minimum
            self.horizon_source = "last_event"
        elif int(self.n_windows) != self.n_windows or int(self.n_windows) < minimum:
            raise ValueError("n_windows excludes recorded events")
        self.n_windows = int(self.n_windows)
        self.order = np.argsort(self.window, kind="stable")
        self.bounds = np.searchsorted(self.window[self.order], np.arange(self.n_windows + 1))
        self._source_features = np.stack([self.z0, self.p], axis=1)
        if self.target_id is not None:
            self.target_id = np.asarray(self.target_id)
            if self.target_id.ndim != 1 or len(self.target_id) != n:
                raise ValueError("target_id must have one row per original event")

    def __len__(self):
        return len(self.t)

    @property
    def num_events(self):
        return len(self)

    @property
    def labels(self):
        return self.label

    @property
    def source_features(self):
        """The shared source encoder input, [N, 2], with no label channels."""
        return self._source_features

    def window_ids(self, window):
        window = int(window)
        if not 0 <= window < self.n_windows:
            return np.empty(0, dtype=np.int64)
        return self.order[self.bounds[window]:self.bounds[window + 1]]

    def iter_windows(self, include_empty=True):
        # Empty windows matter for deadlines even though they add no evidence.
        for w in range(self.n_windows):
            ids = self.window_ids(w)
            if include_empty or len(ids):
                yield w, ids


def load_recording(data_path, z0_path, window_ms=None, n_windows=None):
    """Join raw events and arrival scores using their original row indices.

    Old trajectory files without ``window_ms`` require an explicit override;
    time units must never be guessed from the recording's event count.
    """
    with np.load(str(data_path), allow_pickle=False) as data:
        loc = np.asarray(data["ev_loc"])
        norm = np.asarray(data["evs_norm"])
    if loc.ndim != 2 or loc.shape[1] < 3 or norm.ndim != 2 or norm.shape[1] < 5:
        raise ValueError("raw recording requires ev_loc[N,3] and evs_norm[N,>=5]")
    if len(loc) != len(norm):
        raise ValueError("raw event arrays have different row counts")
    with np.load(str(z0_path), allow_pickle=False) as data:
        traj = np.asarray(data["traj"])
        if traj.ndim != 2 or traj.shape[1] < 1 or len(traj) != len(loc):
            raise ValueError("trajectory must have one original-order row per event")
        z0 = np.asarray(traj[:, 0], dtype=np.float32)
        cached_ms = int(np.asarray(data["window_ms"]).item()) if "window_ms" in data else None
        if window_ms is None:
            if cached_ms is None:
                raise ValueError("trajectory has no window_ms; pass it explicitly")
            window_ms = cached_ms
        elif cached_ms is not None and int(window_ms) != cached_ms:
            raise ValueError("window_ms override disagrees with cached trajectory")
        cached_windows = np.asarray(data["n_windows"]).item() if "n_windows" in data else None
        if cached_windows is not None:
            if int(cached_windows) != cached_windows or int(cached_windows) < 1:
                raise ValueError("cached n_windows must be a positive integer")
            if n_windows is not None and int(n_windows) != int(cached_windows):
                raise ValueError("n_windows override disagrees with cached trajectory")
            n_windows = int(cached_windows)
        elif n_windows is None:
            warnings.warn("Legacy prior cache has no n_windows; using last-event window + 1. "
                          "Pass an explicit recording duration to preserve trailing empty windows.",
                          RuntimeWarning, stacklevel=2)
        if "labels" in data:
            cached_labels = np.asarray(data["labels"]).reshape(-1)
            if cached_labels.shape != norm[:, 4].shape or not np.array_equal(cached_labels, norm[:, 4]):
                raise ValueError("trajectory labels do not match original event rows")
        log_w = float(np.asarray(data["log_w"]).item())
        backbone_sha256 = str(data["backbone_sha256"].item()) if "backbone_sha256" in data else None
    result = Recording(Path(data_path).name, loc[:, 0], loc[:, 1], loc[:, 2], norm[:, 3], z0,
                       norm[:, 4], window_ms, log_w, n_windows,
                       norm[:, 5] if norm.shape[1] > 5 else None, backbone_sha256)
    if cached_windows is not None:
        result.horizon_source = "cache_metadata"
    return result


def save_indexed_recording(path, recording):
    """Optional compact disk cache: one row per event, no query-token copies."""
    payload = {k: getattr(recording, k) for k in ("x", "y", "t", "p", "z0", "label")}
    payload.update(name=np.asarray(recording.name), window_ms=np.int64(recording.window_ms),
                   log_w=np.float64(recording.log_w), n_windows=np.int64(recording.n_windows),
                   schema=np.int64(1))
    if recording.target_id is not None:
        payload["target_id"] = recording.target_id
    if recording.backbone_sha256 is not None:
        payload["backbone_sha256"] = np.asarray(recording.backbone_sha256)
    np.savez_compressed(str(path), **payload)


def load_indexed_recording(path):
    with np.load(str(path), allow_pickle=False) as data:
        if int(data["schema"]) != 1:
            raise ValueError("unsupported indexed-recording schema")
        return Recording(str(data["name"].item()), *(data[k] for k in ("x", "y", "t", "p", "z0", "label")),
                         int(data["window_ms"]), float(data["log_w"]), int(data["n_windows"]),
                         data["target_id"] if "target_id" in data else None,
                         str(data["backbone_sha256"].item()) if "backbone_sha256" in data else None)


class NeighborIndex:
    """Lazy per-window KD trees and original-order source references.

    ``k_evidence=0`` selects only spatial neighbours. A positive value retains
    the legacy second list selected by arrival score, and therefore requires
    an explicit ``filter_logit``. Neither path uses ground-truth labels.
    Radius 0 or infinity means unbounded, matching legacy DeCo semantics.
    """

    def __init__(self, recording, past_windows=10, max_wait=10, k_near=8,
                 k_evidence=0, radius=32.0, filter_logit=None):
        self.recording = recording
        for key, value in (("past_windows", past_windows), ("max_wait", max_wait),
                           ("k_near", k_near), ("k_evidence", k_evidence)):
            if int(value) != value or int(value) < 0:
                raise ValueError("%s must be a nonnegative integer" % key)
            setattr(self, key, int(value))
        self.K = self.k_near + self.k_evidence
        if self.K < 1:
            raise ValueError("at least one neighbour slot per window is required")
        radius = float(radius)
        if np.isnan(radius) or radius < 0:
            raise ValueError("radius must be nonnegative")
        self.radius = np.inf if radius == 0 else radius
        if self.k_evidence and (filter_logit is None or not np.isfinite(filter_logit)):
            raise ValueError("k_evidence > 0 requires a finite filter_logit")
        self.filter_logit = None if filter_logit is None else float(filter_logit)
        self.offsets = np.arange(-self.past_windows, self.max_wait + 1, dtype=np.int64)
        self._trees = {}  # type: Dict[int, Tuple]
        self.trees_built = 0

    @property
    def cached_windows(self):
        return tuple(sorted(self._trees))

    @property
    def slots_per_query(self):
        return len(self.offsets) * self.K

    def evict_before(self, window):
        """Release trees no live query needs. The caller owns that invariant."""
        for key in list(self._trees):
            if key < int(window):
                del self._trees[key]

    def _window_tree(self, w):
        if w not in self._trees:
            r = self.recording
            ids = r.window_ids(w)
            pts = np.stack([r.x[ids], r.y[ids]], axis=1).astype(np.float64)
            tree = cKDTree(pts) if len(ids) else None
            evid = ids[r.z0[ids] > self.filter_logit] if self.k_evidence else np.empty(0, np.int64)
            epts = np.stack([r.x[evid], r.y[evid]], axis=1).astype(np.float64)
            etree = cKDTree(epts) if len(evid) else None
            self._trees[w] = (tree, ids, etree, evid)
            self.trees_built += 1
        return self._trees[w]

    def _knn(self, tree, ids, xy, count):
        if tree is None or count == 0:
            return np.empty((len(xy), 0), dtype=np.int64)
        count = min(count, len(ids))
        dist, loc = tree.query(xy, k=count, distance_upper_bound=self.radius)
        if count == 1:
            dist, loc = dist[:, None], loc[:, None]
        valid = np.isfinite(dist)
        return np.where(valid, ids[np.minimum(loc, len(ids) - 1)], -1)

    @staticmethod
    def _first(candidates, count):
        result = np.full((len(candidates), count), -1, dtype=np.int64)
        if count and candidates.shape[1]:
            order = np.argsort(candidates < 0, axis=1, kind="stable")
            take = np.take_along_axis(candidates, order, axis=1)[:, :count]
            result[:, :take.shape[1]] = take
        return result

    def query(self, query_ids, age=None, current_window=None, new_only=False):
        """Return ``idx``, ``relative``, ``mask``, ``age`` and ``max_age``.

        ``age`` and ``current_window`` are exclusive scalar or [B] arguments.
        Future slots are -1 and their relative features are zero. Calling with
        a larger age exposes extra windows without changing earlier picks.
        End-of-recording ages are clamped to the final observed window.
        ``new_only`` returns only the currently arriving window. A live queue
        can append these slots to its saved prefix, avoiding repeated searches.
        """
        q = _integer_vector(query_ids, "query_ids")
        r = self.recording
        if ((q < 0) | (q >= len(r))).any():
            raise IndexError("query id outside the recording")
        if age is not None and current_window is not None:
            raise ValueError("provide age or current_window, not both")
        raw_age = 0 if age is None else age
        if current_window is not None:
            raw_age = np.asarray(current_window) - r.window[q]
        raw_age = np.broadcast_to(np.asarray(raw_age), (len(q),))
        ages = _integer_vector(raw_age, "age")
        if ((ages < 0) | (ages > self.max_wait)).any():
            raise ValueError("query age must lie in [0, max_wait]")
        maximum = np.minimum(self.max_wait, r.n_windows - 1 - r.window[q])
        ages = np.minimum(ages, maximum)
        picks = np.full((len(q), len(self.offsets), self.K), -1, dtype=np.int64)
        xy = np.stack([r.x[q], r.y[q]], axis=1).astype(np.float64)
        for oi, dw in enumerate(self.offsets):
            target = r.window[q] + dw
            available = (dw <= ages) & (target >= 0) & (target < r.n_windows)
            if new_only:
                available &= dw == ages
            for w in np.unique(target[available]):
                rows = np.flatnonzero(available & (target == w))
                tree, ids, etree, evid = self._window_tree(int(w))
                near = self._knn(tree, ids, xy[rows], self.k_near + 1) if self.k_near else np.empty((len(rows), 0), np.int64)
                near[near == q[rows, None]] = -1
                near = self._first(near, self.k_near)
                selected = self._knn(etree, evid, xy[rows], self.k_evidence + self.k_near + 1) if self.k_evidence else np.empty((len(rows), 0), np.int64)
                if selected.shape[1]:
                    selected[selected == q[rows, None]] = -1
                    repeated = (selected[:, :, None] == near[:, None, :]).any(axis=2)
                    selected[repeated] = -1
                selected = self._first(selected, self.k_evidence)
                picks[rows, oi] = np.concatenate([near, selected], axis=1)
        idx = picks.reshape(len(q), self.slots_per_query)
        mask = idx >= 0
        relative = np.zeros(idx.shape + (4,), dtype=np.float32)
        if mask.any():
            rows, cols = np.nonzero(mask)
            src = idx[rows, cols]
            dest = q[rows]
            relative[rows, cols, 0] = r.x[src] - r.x[dest]
            relative[rows, cols, 1] = r.y[src] - r.y[dest]
            relative[rows, cols, 2] = r.t[src] - r.t[dest]
            relative[rows, cols, 3] = r.window[src] - r.window[dest]
        return {"idx": idx, "relative": relative, "mask": mask, "query_ids": q,
                "age": ages, "max_age": maximum}


def compact_neighbors(batch):
    """Map global source ids to one compact, unique source-encoder batch.

    Returns ``(source_ids, local_idx)``. Metadata and relative coordinates are
    unchanged. No gradients are cached across optimizer updates: callers must
    encode these source rows again after model parameters change.
    """
    idx = np.asarray(batch["idx"] if isinstance(batch, dict) else batch, dtype=np.int64)
    valid = idx >= 0
    source_ids, inverse = np.unique(idx[valid], return_inverse=True)
    local_idx = np.full(idx.shape, -1, dtype=np.int64)
    local_idx[valid] = inverse
    return source_ids, local_idx


@dataclass
class QuerySample:
    ids: np.ndarray
    probabilities: np.ndarray
    weights: np.ndarray
    population_size: int

    def estimate_mean(self, values):
        """Horvitz-Thompson event mean; dividing by sum(weights) is different."""
        values = np.asarray(values)
        if values.shape != self.weights.shape:
            raise ValueError("values must have one element per sampled query")
        return float(np.sum(values * self.weights) / max(self.population_size, 1))


def sample_queries(recording, max_queries=None, rng=None, strata=None, per_stratum=None):
    """Sample BEFORE neighbourhood construction without removing source events.

    Default is all queries. Uniform sampling uses an explicit ``max_queries``.
    Optional stratification requires ``strata[N]`` and an explicit positive
    ``per_stratum`` quota (scalar or mapping). Every stratum has positive
    inclusion probability. No score interval or target-dependent cap is hidden.
    """
    n = len(recording)
    rng = np.random.default_rng() if rng is None else rng
    if strata is None:
        if per_stratum is not None:
            raise ValueError("per_stratum requires strata")
        if max_queries is not None and (int(max_queries) != max_queries or int(max_queries) < 1):
            raise ValueError("max_queries must be positive or None")
        take = n if max_queries is None else min(n, int(max_queries))
        ids = np.arange(n, dtype=np.int64) if take == n else np.sort(rng.choice(n, take, replace=False))
        prob = np.full(take, float(take) / n if n else 1.0, dtype=np.float64)
    else:
        if max_queries is not None or per_stratum is None:
            raise ValueError("stratified sampling needs per_stratum, not max_queries")
        strata = np.asarray(strata)
        if strata.ndim != 1 or len(strata) != n:
            raise ValueError("strata must have one entry per event")
        ids_parts, prob_parts = [], []
        for key in np.unique(strata):
            group = np.flatnonzero(strata == key)
            quota = per_stratum[key] if isinstance(per_stratum, Mapping) else per_stratum
            if int(quota) != quota or int(quota) < 1:
                raise ValueError("every stratum needs a positive integer quota")
            take = min(len(group), int(quota))
            ids_parts.append(group if take == len(group) else rng.choice(group, take, replace=False))
            prob_parts.append(np.full(take, float(take) / len(group), dtype=np.float64))
        ids = np.concatenate(ids_parts) if ids_parts else np.empty(0, np.int64)
        prob = np.concatenate(prob_parts) if prob_parts else np.empty(0, np.float64)
        order = np.argsort(ids)
        ids, prob = ids[order], prob[order]
    return QuerySample(ids, prob, 1.0 / prob, n)
