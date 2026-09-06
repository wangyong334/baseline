import unittest
import numpy as np
from utils.semantic_cpu import ForegroundMetrics


class TestForegroundMetrics(unittest.TestCase):
    def test_threshold_counts_and_no_mutation(self):
        p = np.array([.9, .89, .99, .01], dtype=np.float32)
        y = np.array([1, 1, 0, 0])
        original = p.copy()
        m = ForegroundMetrics()
        m.update(p, y)
        self.assertEqual((m.tp, m.fp, m.fn, m.tn), (1, 1, 1, 1))
        self.assertEqual(m.compute(), {"iou": 1 / 3, "seg_acc": .5})
        np.testing.assert_array_equal(p, original)

    def test_streaming_matches_global_counts(self):
        rng = np.random.RandomState(37)
        p, y = rng.rand(103), rng.randint(0, 2, 103)
        m = ForegroundMetrics()
        for start in range(0, 103, 7):
            m.update(p[start:start+7], y[start:start+7])
        positive = p >= .9
        tp = np.count_nonzero(positive & (y == 1))
        expected = {"iou": tp / np.count_nonzero(positive | (y == 1)),
                    "seg_acc": tp / np.count_nonzero(y == 1)}
        self.assertEqual(m.compute(), expected)

    def test_all_background_predictions(self):
        m = ForegroundMetrics()
        m.update([0, 0], [1, 0])
        self.assertEqual(m.compute(), {"iou": 0, "seg_acc": 0})

    def test_invalid_inputs(self):
        for p, y in (([.5], [0, 1]), ([float('nan')], [1]), ([.5], [2]), ([], [])):
            with self.assertRaises(ValueError):
                ForegroundMetrics().update(p, y)

    def test_no_foreground_split(self):
        m = ForegroundMetrics()
        m.update([0, 1], [0, 0])
        with self.assertRaises(ValueError):
            m.compute()


if __name__ == '__main__':
    unittest.main()
