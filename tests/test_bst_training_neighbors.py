"""Source-window batching preserves the complete training neighborhood exactly."""
import unittest

import numpy as np

from tools.bst.data import NeighborIndex, Recording


def recording(seed=37):
    rng = np.random.RandomState(seed)
    # Repeated integer coordinates exercise KNN distance ties and self removal.
    n = 180
    order = rng.permutation(n)
    window = np.repeat(np.array([0, 1, 3, 4, 6, 8]), n // 6)[order]
    return Recording("tied", rng.randint(0, 4, n), rng.randint(0, 4, n),
                     50 * window + rng.randint(0, 50, n), rng.randint(0, 2, n),
                     rng.randn(n), rng.randint(0, 2, n), 50, n_windows=11)


class TrainingNeighborBatchingTests(unittest.TestCase):
    def assert_same(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            self.assertEqual(actual[key].dtype, expected[key].dtype, key)
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)

    def test_all_outputs_match_reference_for_mixed_ages_ties_and_radii(self):
        rec = recording()
        rng = np.random.RandomState(12)
        queries = rng.randint(0, len(rec), 37)
        ages = rng.randint(0, 5, len(queries))
        for radius in (0, .1, 1.01, 3.):
            for near, evidence in ((1, 0), (0, 3), (3, 4), (40, 40)):
                index = NeighborIndex(rec, past_windows=3, max_wait=4,
                                      k_near=near, k_evidence=evidence,
                                      filter_logit=0., radius=radius)
                for new_only in (False, True):
                    with self.subTest(radius=radius, near=near, evidence=evidence, new_only=new_only):
                        self.assert_same(index.query(queries, age=ages, new_only=new_only),
                                         index.query_reference(queries, age=ages, new_only=new_only))

    def test_source_tree_is_queried_once_and_original_rows_slots_are_preserved(self):
        rec = recording()
        queries = np.array([rec.window_ids(w)[0] for w in (8, 1, 4, 0, 6, 3, 1)])
        index = NeighborIndex(rec, past_windows=3, max_wait=4, k_near=4, k_evidence=3,
                              filter_logit=0., radius=0)
        original, calls = index._select_window, []
        def counted(q, xy, window):
            calls.append(int(window))
            return original(q, xy, window)
        index._select_window = counted
        actual = index.query(queries, age=4)
        grouped_calls = list(calls)
        calls.clear()
        expected = index.query_reference(queries, age=4)
        self.assert_same(actual, expected)
        self.assertEqual(len(grouped_calls), len(set(grouped_calls)))
        self.assertGreater(len(calls), len(grouped_calls))
        np.testing.assert_array_equal(actual["query_ids"], queries)

    def test_current_window_and_record_end_clamping_are_exact(self):
        rec = recording()
        queries = np.array([rec.window_ids(w)[0] for w in (6, 8, 6, 8)])
        index = NeighborIndex(rec, past_windows=2, max_wait=4, k_near=5, radius=0)
        result = index.query(queries, current_window=10)
        self.assert_same(result, index.query_reference(queries, current_window=10))
        np.testing.assert_array_equal(result["age"], [4, 2, 4, 2])
        self.assert_same(index.query(queries, age=4), index.query_reference(queries, age=4))

    def test_causal_prefix_picks_never_change_and_new_only_matches_arriving(self):
        rec = recording()
        queries = rec.window_ids(3)[:8]
        index = NeighborIndex(rec, past_windows=2, max_wait=4, k_near=3, k_evidence=2,
                              filter_logit=.1, radius=0)
        previous = index.query(queries, age=0)
        for age in range(1, 5):
            actual = index.query(queries, age=age)
            np.testing.assert_array_equal(actual["idx"][previous["mask"]], previous["idx"][previous["mask"]])
            np.testing.assert_array_equal(actual["relative"][previous["mask"]], previous["relative"][previous["mask"]])
            only = index.query(queries, age=age, new_only=True)
            arrived = index.query_arriving(queries, 3 + age)
            slot = (index.past_windows + age) * index.K
            np.testing.assert_array_equal(only["idx"][:, slot:slot + index.K], arrived["idx"])
            np.testing.assert_array_equal(only["relative"][:, slot:slot + index.K], arrived["relative"])
            self.assertTrue(np.all(rec.window[actual["idx"][actual["mask"]]] <= 3 + age))
            previous = actual

    def test_zero_horizon_empty_queries_and_empty_recording(self):
        for rec in (recording(), Recording("empty", [], [], [], [], [], [], 50, n_windows=5)):
            for past, future in ((0, 0), (2, 3)):
                index = NeighborIndex(rec, past_windows=past, max_wait=future, k_near=1, radius=0)
                self.assert_same(index.query([], age=future), index.query_reference([], age=future))
                if len(rec):
                    self.assert_same(index.query([0, 0], age=future), index.query_reference([0, 0], age=future))

    def test_invalid_query_and_age_arguments_still_fail(self):
        index = NeighborIndex(recording(), past_windows=2, max_wait=4)
        invalid = [([.5], {}), ([-1], {}), ([999], {}), ([0], {"age": -1}),
                   ([0], {"age": 5}), ([0], {"age": .5}),
                   ([0], {"age": 0, "current_window": 1})]
        for queries, kwargs in invalid:
            for method in (index.query, index.query_reference):
                with self.subTest(queries=queries, kwargs=kwargs, method=method.__name__):
                    with self.assertRaises((ValueError, IndexError)):
                        method(queries, **kwargs)


if __name__ == "__main__":
    unittest.main()
