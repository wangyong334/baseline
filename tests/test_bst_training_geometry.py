"""Static geometry plans preserve exact SGD rows and enforce their ID budget."""
import unittest

import numpy as np

from tools.bst.data import NeighborIndex, Recording
from tools.bst.training_geometry import GeometryBatches


def make_index():
    rng = np.random.RandomState(37)
    n = 240
    return NeighborIndex(Recording("geometry", rng.randint(0, 7, n), rng.randint(0, 7, n),
                                   rng.randint(0, 12 * 50, n), rng.randint(0, 2, n),
                                   rng.randn(n), rng.randint(0, 2, n), 50, n_windows=14),
                         past_windows=3, max_wait=4, k_near=3, k_evidence=2,
                         filter_logit=0, radius=0)


class TrainingGeometryTests(unittest.TestCase):
    def assert_same(self, a, b):
        self.assertEqual(set(a), set(b))
        for key in b:
            self.assertEqual(a[key].dtype, b[key].dtype, key)
            self.assertEqual(a[key].shape, b[key].shape, key)
            self.assertEqual(a[key].tobytes(), b[key].tobytes(), key)

    def test_cached_batches_preserve_duplicate_queries_order_and_final_partial_batch(self):
        index = make_index()
        ordered = np.random.RandomState(1).randint(0, len(index.recording), 97)
        plan = GeometryBatches(index, ordered, batch_size=13, cache_mb=64)
        starts = []
        for start, stop, actual in plan:
            starts.append(start)
            self.assert_same(actual, index.query_reference(ordered[start:stop], age=index.max_wait))
        self.assertEqual(starts, list(range(0, len(ordered), 13)))
        self.assertEqual(plan.stats["cached_batches"], 8)
        self.assertEqual(plan.stats["uncached_batches"], 0)
        self.assertEqual(plan.stats["blocks"], 1)
        self.assertEqual(plan.id_dtype, np.dtype(np.int32))
        self.assertEqual(plan.stats["peak_id_cache_bytes"], len(ordered) * index.slots_per_query * 4)

    def test_tiny_budget_falls_back_without_changing_the_batch(self):
        index = make_index()
        ordered = np.arange(29)
        one_batch_bytes = 8 * index.slots_per_query * 4
        for cache_mb in (0, (one_batch_bytes - 1) / 1024 ** 2):
            plan = GeometryBatches(index, ordered, batch_size=8, cache_mb=cache_mb)
            for start, stop, actual in plan:
                self.assert_same(actual, index.query_reference(ordered[start:stop], age=index.max_wait))
            self.assertEqual(plan.stats["peak_id_cache_bytes"], 0)
            self.assertEqual(plan.stats["uncached_batches"], 4)
            self.assertEqual(plan.stats["query_calls"], 4)

    def test_budget_splits_only_at_sgd_boundaries_and_releases_cache_views(self):
        index = make_index()
        ordered = np.array([5, 2, 2, 99, 16, 8, 230, 7, 200, 19, 20])
        budget = 2 * 4 * index.slots_per_query * 4
        plan = GeometryBatches(index, ordered, batch_size=4, cache_mb=budget / 1024 ** 2)
        result = list(plan)
        self.assertEqual([(start, stop) for start, stop, _ in result], [(0, 4), (4, 8), (8, 11)])
        self.assertEqual(plan.stats["blocks"], 2)
        self.assertLessEqual(plan.stats["peak_id_cache_bytes"], budget)
        for start, stop, batch in result:
            self.assertTrue(batch["idx"].flags.owndata)
            self.assert_same(batch, index.query_reference(ordered[start:stop], age=index.max_wait))
        result[0][2]["idx"][:] = -1
        self.assert_same(result[1][2], index.query_reference(ordered[4:8], age=index.max_wait))

    def test_build_chunk_limit_does_not_change_large_sgd_batch_order(self):
        index = make_index()
        ordered = np.random.RandomState(2).randint(0, len(index.recording), 61)
        plan = GeometryBatches(index, ordered, batch_size=20, cache_mb=64)
        plan.BUILD_QUERY_LIMIT = 7
        for start, stop, batch in plan:
            self.assert_same(batch, index.query_reference(ordered[start:stop], age=index.max_wait))
        self.assertEqual(plan.stats["query_calls"], 9)
        self.assertEqual(plan.stats["peak_build_queries"], 7)
        self.assertEqual(plan.stats["cached_batches"], 4)

    def test_empty_iterator_stats_reset_and_input_is_owned(self):
        index = make_index()
        empty = GeometryBatches(index, [], batch_size=3)
        self.assertEqual(list(empty), [])
        self.assertEqual(empty.stats["blocks"], 0)
        self.assertEqual(empty.stats["peak_id_cache_bytes"], 0)
        ordered = np.arange(9)
        plan = GeometryBatches(index, ordered, batch_size=4)
        ordered[:] = 100
        first = list(plan)
        previous_stats = dict(plan.stats)
        second = list(plan)
        self.assertEqual(plan.stats["blocks"], previous_stats["blocks"])
        self.assertEqual(plan.stats["query_calls"], previous_stats["query_calls"])
        for a, b in zip(first, second):
            self.assert_same(a[2], b[2])
        np.testing.assert_array_equal(first[0][2]["query_ids"], np.arange(4))

    def test_bad_budget_size_and_query_ids_are_rejected(self):
        index = make_index()
        for budget in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                GeometryBatches(index, [0], 4, cache_mb=budget)
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                GeometryBatches(index, [0], size)
        for ids in ([-1], [len(index.recording)], [.5]):
            with self.assertRaises((ValueError, IndexError)):
                GeometryBatches(index, ids, 4)


if __name__ == "__main__":
    unittest.main()
