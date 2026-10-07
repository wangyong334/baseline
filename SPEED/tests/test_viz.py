import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speed.data.events import EventStream  # noqa: E402
from speed.viz.trajectory import auto_zoom, panel, render  # noqa: E402


def synthetic():
    rng = np.random.default_rng(0)
    n_bg, n_t = 3000, 400
    t_t = np.sort(rng.integers(0, 2000000, n_t))
    x = np.r_[rng.integers(0, 120, n_bg), 20 + t_t // 25000]
    y = np.r_[rng.integers(0, 90, n_bg), 40 + rng.integers(0, 4, n_t)]
    t = np.r_[rng.integers(0, 2000000, n_bg), t_t]
    tid = np.r_[np.zeros(n_bg, int), np.ones(n_t, int)]
    return EventStream("syn", x, y, t, np.zeros(x.size), tid > 0, tid, tid > 0, 120, 90).validate()


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
class VizTests(unittest.TestCase):
    def test_auto_zoom_and_render(self):
        s = synthetic()
        gt = panel(s, s.label == 1, title="GT")
        x0, x1, y0, y1, t0, t1 = auto_zoom(gt, span_s=0.5)
        self.assertAlmostEqual(t1 - t0, 0.5)
        inside = gt["red"] & (gt["t"] >= t0) & (gt["t"] <= t1)
        self.assertTrue(inside.sum() > 50)
        self.assertTrue(((gt["x"][inside] >= x0) & (gt["x"][inside] <= x1)).all())
        self.assertTrue(x1 - x0 < 60)
        pred = panel(s, (s.label == 1) | (np.arange(s.n_events) % 97 == 0), 0, 1500000, "method")
        self.assertTrue(pred["t"].max() < 1.5)
        with tempfile.TemporaryDirectory() as d:
            out = render([[pred, gt]], os.path.join(d, "fig.png"), zoom="auto")
            self.assertGreater(os.path.getsize(out), 1000)


if __name__ == "__main__":
    unittest.main()
