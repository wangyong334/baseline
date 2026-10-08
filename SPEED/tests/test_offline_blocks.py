import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from offline_blocks import export_stream, stitch_stream  # noqa: E402
from speed.data.events import EventStream  # noqa: E402


def stream(seed=0, n=5000, span_s=19.5):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, int(span_s * 1e6), n)
    tid = (rng.random(n) < 0.1) * rng.integers(1, 4, n)
    return EventStream("Train/9/9.npy", rng.integers(0, 1280, n), rng.integers(0, 720, n), t, rng.integers(0, 2, n),
                       tid > 0, tid, tid > 0, 1280, 720).validate()


class OfflineBlockTests(unittest.TestCase):
    def test_round_trip(self):
        s = stream()
        with tempfile.TemporaryDirectory() as blocks, tempfile.TemporaryDirectory() as dumps:
            entries = export_stream(s, blocks, 8_000_000, (1280, 736, 8192))
            self.assertEqual([e["start_us"] for e in entries], [0, 8_000_000, 16_000_000])
            self.assertEqual(sum(e["n"] for e in entries), s.n_events)
            for e in entries:
                with np.load(os.path.join(blocks, e["file"])) as b:
                    loc, norm, idx = b["ev_loc"], b["evs_norm"], b["index"]
                self.assertTrue(np.array_equal(loc[:, 0], s.x[idx]) and np.array_equal(loc[:, 1], s.y[idx]))
                self.assertTrue((loc[:, 2] >= 0).all() and (loc[:, 2] < 8000).all())
                self.assertTrue(np.all(np.diff(idx) > 0))               # file order inside the block
                self.assertTrue(np.allclose(norm[:, 0], s.x[idx] / 1280.0) and np.array_equal(norm[:, 4], s.label[idx]))
                fake = (loc[:, 0] * 7 + loc[:, 1] * 3 + loc[:, 2]) % 1000 / 1000.0
                np.savez(os.path.join(dumps, e["file"]), locs=np.c_[np.zeros(len(idx), np.int64), loc],
                         probabilities=fake.astype(np.float32))
            prob, publish = stitch_stream(s, entries, blocks, dumps)
            t_ms = np.floor((s.t % 8_000_000) / 1000.0).astype(np.int64)
            want = ((s.x * 7 + s.y * 3 + t_ms) % 1000 / 1000.0).astype(np.float32)
            self.assertTrue(np.array_equal(prob, want))
            self.assertTrue(np.array_equal(publish, np.minimum((s.t // 8_000_000 + 1) * 8_000_000, s.span_us)))
            self.assertTrue((publish > s.t).all())


if __name__ == "__main__":
    unittest.main()
