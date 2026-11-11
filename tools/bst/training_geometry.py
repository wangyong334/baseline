"""Bounded, parameter-independent training geometry over an unchanged batch order.

Only global source IDs are cached. Learned source/relation projections must be
recomputed after every optimizer update. The cache is local to one iterator and
one recording; it never accumulates a dataset-sized bank across epochs.
"""
import math
import time

import numpy as np

from tools.bst.data import _integer_vector


class GeometryBatches:
    """Yield ``(start, stop, batch)`` with the original SGD minibatch boundaries.

    ``cache_mb`` limits resident ID-cache bytes, not total process memory. Age
    vectors, returned minibatch tensors, KD trees and query-construction
    temporaries are additional. A cached block is constructed with at most 2048
    query rows at a time; ``NeighborIndex.query`` temporarily also constructs
    int64 IDs, masks, relative features and grouping arrays for those rows.
    Those temporaries are released before the next construction chunk. Yielded
    IDs are independent int64 copies so a consumer retaining one minibatch does
    not retain an entire cache block.

    Zero budget, or a budget smaller than one full SGD batch of IDs, selects the
    existing per-minibatch query path. Thus a deliberately oversized SGD batch
    can still need its ordinary query memory; the cache budget is not a global
    memory cap. Stats reset for each iteration and describe work actually done.
    """

    BUILD_QUERY_LIMIT = 2048

    def __init__(self, index, ordered_ids, batch_size, cache_mb=64):
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not math.isfinite(float(cache_mb)) or float(cache_mb) < 0:
            raise ValueError("cache_mb must be finite and nonnegative")
        self.index = index
        self.ordered_ids = _integer_vector(ordered_ids, "ordered_ids").copy()
        if ((self.ordered_ids < 0) | (self.ordered_ids >= len(index.recording))).any():
            raise IndexError("query id outside the recording")
        self.batch_size = int(batch_size)
        self.cache_mb = float(cache_mb)
        self.cache_bytes = int(self.cache_mb * 1024 ** 2)
        self.id_dtype = np.dtype(np.int32 if len(index.recording) - 1 <= np.iinfo(np.int32).max else np.int64)
        self.bytes_per_query = index.slots_per_query * self.id_dtype.itemsize
        rows = self.cache_bytes // self.bytes_per_query
        self.block_queries = (rows // self.batch_size) * self.batch_size
        self.stats = self._new_stats()

    @staticmethod
    def _new_stats():
        return {"query_build_seconds": 0.0, "geometry_materialize_seconds": 0.0,
                "peak_id_cache_bytes": 0, "blocks": 0, "query_calls": 0,
                "peak_build_queries": 0, "cached_batches": 0, "uncached_batches": 0}

    def _query(self, ids):
        started = time.perf_counter()
        result = self.index.query(ids, age=self.index.max_wait)
        self.stats["query_build_seconds"] += time.perf_counter() - started
        self.stats["query_calls"] += 1
        self.stats["peak_build_queries"] = max(self.stats["peak_build_queries"], len(ids))
        return result

    def _materialize(self, ids, cached):
        started = time.perf_counter()
        r = self.index.recording
        # Copy even for int64 caches: no returned view may retain the full block.
        idx = cached.astype(np.int64, copy=True)
        mask = idx >= 0
        relative = np.zeros(idx.shape + (4,), dtype=np.float32)
        rows, cols = np.nonzero(mask)
        src, dest = idx[rows, cols], ids[rows]
        relative[rows, cols, 0] = r.x[src] - r.x[dest]
        relative[rows, cols, 1] = r.y[src] - r.y[dest]
        relative[rows, cols, 2] = r.t[src] - r.t[dest]
        relative[rows, cols, 3] = r.window[src] - r.window[dest]
        maximum = np.minimum(self.index.max_wait, r.n_windows - 1 - r.window[ids])
        result = {"idx": idx, "relative": relative, "mask": mask, "query_ids": ids.copy(),
                  "age": maximum.copy(), "max_age": maximum}
        self.stats["geometry_materialize_seconds"] += time.perf_counter() - started
        return result

    def __iter__(self):
        self.stats = self._new_stats()
        n, batch_size = len(self.ordered_ids), self.batch_size
        if self.block_queries == 0:
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                self.stats["blocks"] += 1
                self.stats["uncached_batches"] += 1
                yield start, stop, self._query(self.ordered_ids[start:stop])
            return
        for block_start in range(0, n, self.block_queries):
            block_stop = min(block_start + self.block_queries, n)
            cached = np.empty((block_stop - block_start, self.index.slots_per_query), dtype=self.id_dtype)
            self.stats["blocks"] += 1
            self.stats["peak_id_cache_bytes"] = max(self.stats["peak_id_cache_bytes"], cached.nbytes)
            for offset in range(0, len(cached), self.BUILD_QUERY_LIMIT):
                stop = min(offset + self.BUILD_QUERY_LIMIT, len(cached))
                prepared = self._query(self.ordered_ids[block_start + offset:block_start + stop])
                cached[offset:stop] = prepared["idx"]
                del prepared
            for start in range(block_start, block_stop, batch_size):
                stop = min(start + batch_size, block_stop)
                self.stats["cached_batches"] += 1
                batch = self._materialize(self.ordered_ids[start:stop], cached[start - block_start:stop - block_start])
                yield start, stop, batch
            # A yielded batch owns its own IDs, so this releases the old block
            # before allocation of the next one, even when callers retain it.
            del cached

