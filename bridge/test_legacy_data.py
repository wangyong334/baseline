import io
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # legacy V1-V3 code
sys.path.insert(0, os.path.join(ROOT, "SPEED"))

from dataset.stream_windows import load_npz_events, split_windows  # noqa: E402
from speed.data.dataset_card import load_card  # noqa: E402
from speed.data.events import EventStream  # noqa: E402
from speed.data.readers import read_evuav_npz  # noqa: E402


def evuav_blob(rng, n=5000, n_targets=3):
    x = rng.integers(0, 346, n)
    y = rng.integers(0, 260, n)
    t = rng.integers(0, 8000, n)
    t[:5] = [0, 50, 100, 7950, 7999]
    p = rng.integers(0, 2, n)
    tid = np.where(rng.random(n) < 0.1, rng.integers(1, n_targets + 1, n), 0)
    loc = np.stack([x, y, t], 1).astype(np.int64)
    norm = np.stack([x / 352.0, y / 288.0, t / 8192.0, p, tid > 0, tid], 1).astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, ev=loc, ev_loc=loc, evs_norm=norm)
    return buf.getvalue()


class LegacyDataEquivalence(unittest.TestCase):
    def test_window_partition_matches_legacy(self):
        t_ms = np.random.default_rng(0).integers(0, 8000, 20000)
        order_old, bounds_old = split_windows(t_ms, 50, 160)
        n = t_ms.size
        z = np.zeros(n, int)
        s = EventStream("s", z, z, t_ms * 1000, z, z, z, z, 346, 260, span_us=8000000)
        order, bounds = s.window_partition(50000, 160)
        np.testing.assert_array_equal(order, order_old)
        np.testing.assert_array_equal(bounds, bounds_old)

    def test_evuav_reader_matches_legacy_loader(self):
        blob = evuav_blob(np.random.default_rng(1))
        new = read_evuav_npz(io.BytesIO(blob), load_card("evuav"), "x.npz")
        old = load_npz_events(io.BytesIO(blob), 260, 346, 50, 160, 5)
        for a, b in ((new.x, old.x), (new.y, old.y), (new.t, old.t * 1000), (new.p, old.p),
                     (new.label, old.label), (new.target_id, old.target_id)):
            np.testing.assert_array_equal(a, b)
        order, bounds = new.window_partition(50000, 160)
        np.testing.assert_array_equal(order, old.order)
        np.testing.assert_array_equal(bounds, old.bounds)


if __name__ == "__main__":
    unittest.main()
