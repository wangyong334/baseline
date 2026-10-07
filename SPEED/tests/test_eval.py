import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speed.data.events import EventStream  # noqa: E402
from speed.eval.metrics import BenchmarkMetrics  # noqa: E402
from speed.eval.results import load_result, save_result  # noqa: E402

MS = 1000


def make(events, width=20, height=20, name="s"):
    """events: list of (x, y, t_ms, target_id)."""
    a = np.array(events, dtype=np.int64).reshape(-1, 4)
    tid = a[:, 3]
    return EventStream(name, a[:, 0], a[:, 1], a[:, 2] * MS, np.zeros(len(a)), tid > 0, tid, (tid > 0) * 1,
                       width, height).validate()


class MetricTests(unittest.TestCase):
    def test_iou_acc(self):
        s = make([(0, 0, 1, 1), (1, 0, 2, 1), (2, 0, 3, 0), (3, 0, 4, 0)])
        m = BenchmarkMetrics(20, 20)
        m.update(s, np.array([0.95, 0.1, 0.95, 0.2], np.float32))
        r = m.result()
        self.assertAlmostEqual(r["iou"], 1 / 3.0)
        self.assertAlmostEqual(r["acc"], 0.5)

    def test_pd_frames_and_edges(self):
        # target events in frame 0 (hit), frame 1 (miss) and exactly on the 100 ms edge (skipped)
        s = make([(1, 1, 10, 1), (1, 1, 60, 1), (1, 1, 100, 1)])
        m = BenchmarkMetrics(20, 20)
        m.update(s, np.array([0.99, 0.5, 0.99], np.float32))
        r = m.result()
        self.assertEqual(r["counts"]["objects"], 2)
        self.assertEqual(r["counts"]["detected"], 1)
        self.assertEqual(r["counts"]["frames"], 1)  # (100 - 10) // 50
        self.assertAlmostEqual(r["pd"], 0.5)

    def test_fa_blobs(self):
        ev = [(0, 0, 10, 0), (1, 1, 11, 0), (5, 5, 12, 0), (9, 9, 120, 0), (2, 2, 130, 1)]
        s = make(ev)
        m = BenchmarkMetrics(20, 20)
        m.update(s, np.array([0.95, 0.95, 0.95, 0.95, 0.95], np.float32))
        r = m.result()
        self.assertEqual(r["counts"]["false_blobs"], 3)  # frame 0: diagonal pair + single, frame 2: single
        self.assertAlmostEqual(r["fa"], 3 / (2 * 400.0))

    def test_fa_uint8_wrap(self):
        for n, blobs in ((256, 0), (257, 1)):
            s = make([(3, 3, 10 + (i % 30), 0) for i in range(n)])
            m = BenchmarkMetrics(20, 20)
            m.update(s, np.ones(n, np.float32))
            self.assertEqual(m.result()["counts"]["false_blobs"], blobs)

    def test_correct_thresh_ratio(self):
        n = 20000
        for hits, expected in ((1, 0), (2, 1)):
            s = make([(4, 4, 10 + (i % 30), 1) for i in range(n)])
            prob = np.zeros(n, np.float32)
            prob[:hits] = 0.95
            m = BenchmarkMetrics(20, 20)
            m.update(s, prob)
            self.assertEqual(m.result()["counts"]["detected"], expected)

    def test_decision_overrides_prob(self):
        s = make([(0, 0, 1, 1), (1, 0, 2, 0)])
        m = BenchmarkMetrics(20, 20)
        m.update(s, np.array([0.1, 0.1], np.float32), decision=np.array([1, 0]))
        self.assertAlmostEqual(m.result()["iou"], 1.0)

    def test_latency(self):
        s = make([(0, 0, 10, 1), (0, 0, 30, 1), (0, 0, 70, 1), (5, 5, 20, 0)])
        publish = np.array([50, 50, 100, 50]) * MS
        m = BenchmarkMetrics(20, 20)
        m.update(s, np.array([0.1, 0.95, 0.95, 0.95], np.float32), publish_us=publish)
        r = m.result()
        self.assertEqual(r["publish_latency"]["n"], 2)
        self.assertAlmostEqual(r["publish_latency"]["mean_ms"], (20 + 30) / 2.0)
        self.assertAlmostEqual(r["first_detection"]["mean_ms"], 40.0)  # published at 50 ms, first event at 10 ms
        self.assertEqual(r["first_detection"]["detection_rate"], 1.0)

    def test_rejects_late_start(self):
        s = make([(0, 0, 60, 0)])
        with self.assertRaises(ValueError):
            BenchmarkMetrics(20, 20).update(s, np.zeros(1, np.float32))

    def test_per_class(self):
        a = np.array([[0, 0, 10, 1, 1], [1, 1, 10, 2, 2], [2, 2, 10, 0, 0]])
        s = EventStream("c", a[:, 0], a[:, 1], a[:, 2] * MS, np.zeros(3), a[:, 3] > 0, a[:, 3], a[:, 4], 20, 20)
        m = BenchmarkMetrics(20, 20)
        m.update(s.validate(), np.array([0.95, 0.1, 0.1], np.float32))
        pc = m.result()["per_class"]
        self.assertAlmostEqual(pc["1"]["acc"], 1.0)
        self.assertAlmostEqual(pc["2"]["pd"], 0.0)


class ResultFileTests(unittest.TestCase):
    def test_roundtrip_and_fingerprint(self):
        s = make([(0, 0, 1, 1), (1, 0, 2, 0)], name="test/a.npz")
        other = make([(0, 0, 1, 1), (2, 0, 2, 0)], name="test/a.npz")
        with tempfile.TemporaryDirectory() as d:
            save_result(d, s, [0.9, 0.1], publish_us=[50000, 50000], meta={"method": "x"})
            r = load_result(d, s)
            np.testing.assert_array_equal(r["publish_us"], [50000, 50000])
            self.assertEqual(r["meta"]["method"], "x")
            with self.assertRaises(ValueError):
                load_result(d, other)


if __name__ == "__main__":
    unittest.main()
