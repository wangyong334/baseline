"""Causal evidence indexing and unbiased query selection."""
import itertools
import os
import tempfile
import unittest

import numpy as np

from tools.bst.data import (Recording, NeighborIndex, compact_neighbors, load_recording,
                            load_indexed_recording, save_indexed_recording, sample_queries)


def make_recording():
    # Original order intentionally differs from chronological order.
    return Recording("small.npz", np.array([1, 0, 2, 0, 1, 2]), np.zeros(6),
                     np.array([110, 0, 55, 15, 60, 160]), np.array([1, -1, 1, -1, 1, 1]),
                     np.array([2, -4, 1, -2, 3, 5]), np.array([1, 0, 0, 0, 1, 1]), 50)


class TestIndexedEvidence(unittest.TestCase):
    def test_window_order_and_empty_windows(self):
        r = make_recording()
        self.assertEqual([ids.tolist() for _, ids in r.iter_windows()], [[1, 3], [2, 4], [0], [5]])
        empty = Recording("empty", [], [], [], [], [], [], 50, n_windows=3)
        self.assertEqual(len(list(empty.iter_windows())), 3)
        self.assertEqual(len(list(empty.iter_windows(False))), 0)

    def test_prefix_causality_and_reaudit_indices(self):
        r = make_recording()
        idx = NeighborIndex(r, past_windows=1, max_wait=2, k_near=3, radius=0)
        first = idx.query([2], age=0)
        self.assertEqual(idx.cached_windows, (0, 1))
        self.assertTrue(np.all(r.window[first["idx"][first["mask"]]] <= 1))
        self.assertFalse(np.any(first["idx"] == 2))
        later = idx.query([2], age=2)
        np.testing.assert_array_equal(first["idx"][first["mask"]], later["idx"][first["mask"]])
        self.assertTrue(np.all(later["idx"][first["mask"]] >= 0))
        self.assertEqual(idx.cached_windows, (0, 1, 2, 3))
        self.assertTrue(np.all(first["relative"][~first["mask"]] == 0))

    def test_query_age_vector_and_record_end(self):
        r = make_recording()
        idx = NeighborIndex(r, past_windows=1, max_wait=2, k_near=3, radius=0)
        batch = idx.query([2, 5], age=[1, 2])
        np.testing.assert_array_equal(batch["age"], [1, 0])
        np.testing.assert_array_equal(batch["max_age"], [2, 0])
        a = idx.query([2], current_window=2)
        np.testing.assert_array_equal(a["idx"], batch["idx"][:1])
        with self.assertRaises(ValueError):
            idx.query([2], current_window=0)
        with self.assertRaises(ValueError):
            idx.query([2], age=0, current_window=1)

    def test_relative_units_and_label_independence(self):
        r = make_recording()
        a = NeighborIndex(r, past_windows=1, max_wait=0, k_near=4, radius=0).query([2])
        where = np.flatnonzero(a["idx"][0] == 4)[0]
        np.testing.assert_array_equal(a["relative"][0, where], [-1, 0, 5, 0])
        r.label[:] = 1 - r.label
        b = NeighborIndex(r, past_windows=1, max_wait=0, k_near=4, radius=0).query([2])
        np.testing.assert_array_equal(a["idx"], b["idx"])
        self.assertEqual(r.source_features.shape, (6, 2))

    def test_legacy_filtered_list_deduplicates_and_excludes_self(self):
        r = make_recording()
        idx = NeighborIndex(r, past_windows=1, max_wait=1, k_near=1, k_evidence=2,
                            filter_logit=0, radius=0)
        result = idx.query([2], age=1)
        ids = result["idx"][result["mask"]]
        self.assertNotIn(2, ids)
        self.assertEqual(len(ids), len(set(ids.tolist())))
        self.assertIn(4, ids)
        self.assertIn(0, ids)
        with self.assertRaises(ValueError):
            NeighborIndex(r, k_evidence=1)

    def test_compact_references_share_source_rows(self):
        original = np.array([[5, 2, -1], [2, 8, 5]])
        source, local = compact_neighbors(original)
        np.testing.assert_array_equal(source, [2, 5, 8])
        np.testing.assert_array_equal(source[local[local >= 0]], original[original >= 0])
        self.assertEqual(local[0, 1], local[1, 0])
        source, local = compact_neighbors(np.full((2, 3), -1))
        self.assertEqual(len(source), 0)

    def test_empty_query_and_no_neighbours(self):
        r = make_recording()
        idx = NeighborIndex(r, past_windows=0, max_wait=0, k_near=1, radius=0.1)
        a = idx.query([0])
        self.assertFalse(a["mask"].any())
        self.assertEqual(idx.query([])["relative"].shape, (0, 1, 4))

    def test_tree_eviction_and_rebuild(self):
        r = make_recording()
        idx = NeighborIndex(r, past_windows=1, max_wait=1)
        a = idx.query([2], age=0)
        idx.evict_before(1)
        self.assertEqual(idx.cached_windows, (1,))
        b = idx.query([2], age=0)
        np.testing.assert_array_equal(a["idx"], b["idx"])

    def test_io_alignment_and_explicit_time(self):
        r = make_recording()
        with tempfile.TemporaryDirectory() as temp:
            raw, cached, compact = [os.path.join(temp, f) for f in ("raw.npz", "cached.npz", "compact.npz")]
            norm = np.zeros((len(r), 6), np.float32)
            norm[:, 3], norm[:, 4] = r.p, r.label
            np.savez(raw, ev_loc=np.stack([r.x, r.y, r.t], 1), evs_norm=norm)
            np.savez(cached, traj=r.z0[:, None], labels=r.label, log_w=3.0)
            with self.assertRaisesRegex(ValueError, "window_ms"):
                load_recording(raw, cached)
            result = load_recording(raw, cached, window_ms=50, n_windows=4)
            np.testing.assert_array_equal(result.source_features, r.source_features)
            save_indexed_recording(compact, result)
            restored = load_indexed_recording(compact)
            np.testing.assert_array_equal(restored.t, r.t)
            np.testing.assert_array_equal(restored.label, r.label)
            np.savez(cached, traj=r.z0[:, None], labels=1-r.label, log_w=3.0, window_ms=50)
            with self.assertRaisesRegex(ValueError, "labels"):
                load_recording(raw, cached, n_windows=4)

    def test_cached_horizon_preserves_trailing_empty_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, cached = [os.path.join(temp, f) for f in ("raw.npz", "cached.npz")]
            norm = np.array([[0, 0, 0, 1, 1, 1]], np.float32)
            np.savez(raw, ev_loc=np.array([[0, 0, 1]]), evs_norm=norm)
            np.savez(cached, traj=np.array([[.2]]), labels=np.array([1]), log_w=0,
                     window_ms=50, n_windows=6)
            result = load_recording(raw, cached)
            self.assertEqual(result.n_windows, 6)
            self.assertEqual(result.horizon_source, "cache_metadata")
            self.assertEqual(len(list(result.iter_windows())), 6)
            with self.assertRaisesRegex(ValueError, "n_windows"):
                load_recording(raw, cached, n_windows=5)
            np.savez(cached, traj=np.array([[.2]]), labels=np.array([1]), log_w=0, window_ms=50)
            with self.assertWarnsRegex(RuntimeWarning, "Legacy"):
                legacy = load_recording(raw, cached)
            self.assertEqual(legacy.n_windows, 1)
            self.assertEqual(legacy.horizon_source, "last_event")
            explicit = load_recording(raw, cached, n_windows=6)
            self.assertEqual(explicit.n_windows, 6)

    def test_invalid_timestamps_and_nonfinite_scores(self):
        with self.assertRaisesRegex(ValueError, "integers"):
            Recording("bad", [0], [0], [0.2], [1], [0], [1], 50)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            Recording("bad", [0], [0], [0], [1], [np.nan], [1], 50)


class TestQuerySampling(unittest.TestCase):
    def test_no_implicit_cap_or_band(self):
        r = make_recording()
        s = sample_queries(r)
        np.testing.assert_array_equal(s.ids, np.arange(len(r)))
        np.testing.assert_array_equal(s.weights, 1)
        self.assertAlmostEqual(s.estimate_mean(r.z0), r.z0.mean(), places=6)

    def test_uniform_known_inclusion_probability(self):
        r = make_recording()
        s = sample_queries(r, max_queries=2, rng=np.random.RandomState(37))
        self.assertEqual(len(set(s.ids)), 2)
        np.testing.assert_allclose(s.probabilities, 1.0 / 3)
        np.testing.assert_allclose(s.weights, 3)
        self.assertEqual(len(r), 6)  # Source stream remains complete.

    def test_stratified_estimator_preserves_population_mean(self):
        r = make_recording()
        strata = np.array([0, 0, 0, 0, 1, 1])
        values = np.array([2, 2, 2, 2, 10, 10])
        s = sample_queries(r, strata=strata, per_stratum=1, rng=np.random.RandomState(2))
        self.assertAlmostEqual(s.estimate_mean(values[s.ids]), values.mean())
        self.assertNotAlmostEqual(float(np.mean(values[s.ids])), values.mean())
        with self.assertRaises(ValueError):
            sample_queries(r, strata=strata, per_stratum={0: 1, 1: 0})


if __name__ == "__main__":
    unittest.main()
