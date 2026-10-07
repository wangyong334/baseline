import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from legacy_eval import compare, legacy_first_detection, original_metrics, speed_metrics  # noqa: E402
from speed.data.events import EventStream  # noqa: E402
from speed.eval.metrics import BenchmarkMetrics  # noqa: E402


def synthetic(rng, n=6000, n_targets=4, hot=256):
    x = rng.integers(0, 346, n)
    y = rng.integers(0, 260, n)
    t = rng.integers(0, 8000, n)
    t[:6] = [0, 50, 100, 7950, 7999, 3000]  # frame edges
    tid = np.where(rng.random(n) < 0.15, rng.integers(1, n_targets + 1, n), 0)
    if hot:  # many background events on one pixel in one frame (256 exercises the uint8 wrap)
        x[-hot:], y[-hot:], t[-hot:], tid[-hot:] = 7, 9, 4210 + np.arange(hot) % 40, 0
    s = EventStream("syn", x, y, t * 1000, rng.integers(0, 2, n), tid > 0, tid, tid > 0, 346, 260, span_us=8000000)
    return s.validate()


class LegacyEvalEquivalence(unittest.TestCase):
    def test_benchmark_metrics_bit_exact(self):
        rng = np.random.default_rng(3)
        streams = [synthetic(rng) for _ in range(3)]
        streams.append(synthetic(rng, hot=300))
        probs = [np.clip(s.label + rng.normal(0, 0.35, s.n_events), 0, 1).astype(np.float32) for s in streams]
        probs[1][:] = 1.0  # all positive
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orig = original_metrics(streams, probs)
        new = speed_metrics(streams, probs)
        self.assertEqual(compare(orig, new), [], (orig, new["counts"]))

    def test_first_detection_matches_legacy(self):
        rng = np.random.default_rng(4)
        s = synthetic(rng, hot=0)
        prob = np.clip(s.label + rng.normal(0, 0.4, s.n_events), 0, 1).astype(np.float32)
        birth = (s.t // 1000) // 50
        publish_window = np.minimum(birth + rng.integers(0, 6, s.n_events), 159)
        legacy = {r["target_id"]: r["latency_ms"] for r in legacy_first_detection(s, prob, publish_window)}
        m = BenchmarkMetrics(346, 260)
        m.update(s, prob, publish_us=(publish_window + 1) * 50000)
        m.result()
        for r in m.first:
            old = legacy[float(r["target_id"])]
            self.assertEqual(old is None, r["latency_us"] is None)
            if old is not None:
                self.assertEqual(old * 1000, r["latency_us"])


if __name__ == "__main__":
    unittest.main()
