"""V14 temporal identity, boundaries, representations and train-only stats."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import stream_windows as sw
from dataset.overlap_windows import (
    OverlapSequence, OverlapWindowSource, build_overlap_window_input,
    compute_overlap_stats, load_overlap_npz, update_overlap_histogram,
)


def sequence(times, total_ms=8000):
    times = np.asarray(times, dtype=np.int64)
    ids = np.arange(times.size)
    return OverlapSequence("synthetic.npz", ids % 4, ids % 3, times, ids % 2,
                           (ids % 3 == 0).astype(np.float32), ids, total_ms=total_ms)


class OverlapIndexTests(unittest.TestCase):
    def test_complete_windows_and_half_open_boundaries(self):
        seq = sequence([7999, 0, 49, 50, 99, 100, 7949, 7950, 50])
        self.assertEqual(seq.n_windows, 159)
        np.testing.assert_array_equal(seq.window_starts[[0, -1]], [0, 7900])
        np.testing.assert_array_equal(seq.window_ends[[0, -1]], [100, 8000])
        np.testing.assert_array_equal(seq.window_index(0), [1, 2, 3, 8, 4])
        np.testing.assert_array_equal(seq.window_index(1), [3, 8, 4, 5])
        np.testing.assert_array_equal(seq.coverage_count, [1, 1, 1, 2, 2, 2, 2, 1, 2])
        np.testing.assert_array_equal(seq.order[seq.first_position], np.arange(seq.n_events))
        np.testing.assert_array_equal(seq.order[seq.last_position], np.arange(seq.n_events))
        self.assertTrue(np.all(seq.first_position <= seq.last_position))
        self.assertTrue(np.all(seq.first_position[seq.coverage_count == 1] ==
                               seq.last_position[seq.coverage_count == 1]))

    def test_first_second_final_and_output_time(self):
        seq = sequence([0, 49, 50, 99, 100, 7949, 7950, 7999])
        first = np.zeros(seq.n_events, dtype=np.int64)
        second = first.copy()
        final = first.copy()
        single = first.copy()
        for k in range(seq.n_windows):
            meta = seq.window_metadata(k)
            for field, target in (("first_mask", first), ("second_mask", second),
                                  ("final_mask", final), ("single_mask", single)):
                np.add.at(target, meta["id"][meta[field]], 1)
            if np.any(meta["final_mask"]):
                self.assertTrue(np.all(meta["output_time_ms"][meta["final_mask"]] == seq.window_ends[k]))
        np.testing.assert_array_equal(first, np.ones(seq.n_events))
        np.testing.assert_array_equal(final, np.ones(seq.n_events))
        np.testing.assert_array_equal(second, seq.coverage_count - 1)
        np.testing.assert_array_equal(single, seq.coverage_count == 1)
        np.testing.assert_array_equal(seq.output_time_ms, [100, 100, 150, 150, 200, 8000, 8000, 8000])

    def test_local_time_is_per_appearance(self):
        seq = sequence([50, 59, 60, 99, 100], total_ms=200)
        first = seq.first_position[0]
        second = seq.last_position[0]
        self.assertEqual(seq.expanded_inner_bin[first], 2)
        self.assertEqual(seq.expanded_inner_bin[second], 0)
        self.assertEqual(seq.expanded_t_local[first], 0.5)
        self.assertEqual(seq.expanded_t_local[second], 0.0)
        self.assertEqual(seq.expanded_inner_bin[seq.first_position[2]], 3)
        self.assertEqual(seq.expanded_inner_bin[seq.last_position[2]], 0)
        self.assertTrue(np.all((seq.expanded_t_local >= 0) & (seq.expanded_t_local < 1)))

    def test_empty_and_short_recordings_preserve_windows(self):
        seq = sequence([], total_ms=200)
        self.assertEqual(seq.n_windows, 3)
        self.assertEqual(seq.n_events, 0)
        np.testing.assert_array_equal(seq.bounds, [0, 0, 0, 0])
        seq = sequence([0, 50, 99], total_ms=100)
        np.testing.assert_array_equal(seq.coverage_count, [1, 1, 1])
        for kwargs in ({"window_ms": 90}, {"total_ms": 225}, {"total_ms": 50}, {"time_bins": 0}):
            with self.assertRaises(ValueError):
                OverlapSequence("bad", [], [], [], [], [], [], **kwargs)


class OverlapInputTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"time_bins": 5, "pad_height": 4, "pad_width": 5, "input_clip": 3.0}
        self.q99 = np.log1p(np.arange(1, 13)).astype(np.float32)

    def test_torch_chunk_matches_numpy_and_single_windows(self):
        seq = sequence([199, 0, 50, 50, 79, 101, 20, 99], total_ms=200)
        source = OverlapWindowSource(seq, self.cfg, self.q99, "cpu")
        x, ev, labels, ids = source.chunk(0, seq.n_windows, batch_index=2)
        self.assertEqual(tuple(x.shape), (3, 1, 12, 4, 5))
        np.testing.assert_array_equal(ids, seq.order)
        np.testing.assert_array_equal(ev["id"].numpy(), seq.order)
        np.testing.assert_array_equal(labels.numpy(), seq.label[seq.order])
        self.assertTrue(torch.all(ev["b"] == 2))
        self.assertEqual(ev["batch_idx"].data_ptr(), ev["b"].data_ptr())
        for k in range(seq.n_windows):
            expected, idx = build_overlap_window_input(seq, k, self.q99, 4, 5, 3.0)
            np.testing.assert_array_equal(x[k, 0].numpy(), expected)
            single_x, single_ev, single_labels, single_ids = source.window(k)
            self.assertNotIn("t", single_ev)
            np.testing.assert_array_equal(single_x[0].numpy(), expected)
            np.testing.assert_array_equal(single_ids, idx)
            np.testing.assert_array_equal(single_labels.numpy(), seq.label[idx])
        part_x, part_ev, _, part_ids = source.chunk(1, 3)
        self.assertTrue(torch.equal(part_x, x[1:]))
        np.testing.assert_array_equal(part_ids, seq.order[seq.bounds[1]:])
        np.testing.assert_array_equal(part_ev["t"].numpy(), seq.window_of[seq.bounds[1]:] - 1)

    def test_empty_source_and_metadata_dtype(self):
        seq = sequence([], total_ms=200)
        source = OverlapWindowSource(seq, self.cfg, self.q99, "cpu", dtype=torch.float64)
        x, ev, labels, ids = source.chunk(0, 3)
        self.assertEqual(x.dtype, torch.float64)
        self.assertEqual(labels.dtype, torch.float64)
        self.assertEqual(ev["id"].dtype, torch.int64)
        self.assertEqual(ev["first_mask"].dtype, torch.bool)
        self.assertEqual(ids.size, 0)
        self.assertEqual(int(torch.count_nonzero(x)), 0)
        for a, b in ((0, 0), (-1, 2), (1, 4)):
            with self.assertRaises(ValueError):
                source.chunk(a, b)

    def test_sparse_histogram_is_dense_equivalent(self):
        rng = np.random.RandomState(37)
        seq = sequence(rng.randint(0, 200, size=500), total_ms=200)
        sparse, dense = sw.new_count_histogram(12), sw.new_count_histogram(12)
        for k in range(seq.n_windows):
            update_overlap_histogram(sparse, seq, k, 4, 5)
            ids = seq.window_index(k)
            bins = seq.expanded_inner_bin[seq.bounds[k]:seq.bounds[k + 1]]
            counts = sw.count_channels(seq.x[ids], seq.y[ids], seq.p[ids], bins, 5, 4, 5)
            sw.update_count_histogram(dense, counts)
        for a, b in zip(sparse, dense):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(sw.quantile_from_histogram(a, 0.99), sw.quantile_from_histogram(b, 0.99))

    def test_npz_loader_and_unique_training_counts(self):
        times = np.repeat(np.arange(0, 200, 10), 2)
        seq = sequence(times, total_ms=200)
        ev_loc = np.stack((seq.x, seq.y, seq.t), axis=1)
        norm = np.zeros((seq.n_events, 6), dtype=np.float64)
        norm[:, 3], norm[:, 4], norm[:, 5] = seq.p, seq.label, seq.target_id
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_000.npz"
            np.savez(path, ev_loc=ev_loc, evs_norm=norm)
            loaded = load_overlap_npz(path, 4, 5, total_ms=200)
            np.testing.assert_array_equal(loaded.order, seq.order)
            cfg = dict(self.cfg, height=4, width_px=5, total_ms=200, window_ms=100,
                       stride_ms=50, pos_weight_cap=10.0)
            stats = compute_overlap_stats([path], cfg)
            self.assertEqual(stats["train_pos"] + stats["train_neg"], seq.n_events)
            self.assertEqual(stats["appearances"], seq.order.size)
            self.assertEqual(len(stats["q99"]), 12)
            self.assertEqual(stats["config_used"]["total_ms"], 200)
            with self.assertRaises(ValueError):
                compute_overlap_stats([path], cfg, split="val")


if __name__ == "__main__":
    unittest.main()
