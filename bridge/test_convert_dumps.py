import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert_dumps import convert  # noqa: E402
from speed.data.dataset_card import iter_split, load_card  # noqa: E402
from speed.eval.results import load_result  # noqa: E402


class ConvertDumpTests(unittest.TestCase):
    def test_publish_times(self):
        t = np.array([10, 7960, 120, 4000])
        x, y = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
        tid = np.array([1, 0, 1, 0])
        loc = np.stack([x, y, t], 1).astype(np.int64)
        norm = np.stack([x / 352.0, y / 288.0, t / 8192.0, np.ones(4), tid > 0, tid], 1).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "data", "test"))
            os.makedirs(os.path.join(d, "dump"))
            np.savez(os.path.join(d, "data", "test", "test_000.npz"), ev=loc, ev_loc=loc, evs_norm=norm)
            zeros = np.zeros(4, np.int64)
            np.savez(os.path.join(d, "dump", "test_000.npz"), locs=np.stack([zeros, x, y, t], 1),
                     labels=(tid > 0).astype(np.float32), probabilities=np.full(4, 0.5, np.float32),
                     target_id=tid.astype(np.float64), prob_fused_d2=np.full(4, 0.7, np.float32),
                     prob_pub=np.full(4, 0.9, np.float32), age_pub=np.array([0, 3, 1, 2], np.float32))
            card = load_card("evuav")
            convert(card, os.path.join(d, "data"), "test", os.path.join(d, "dump"), ["net", "fused_d2", "pub"],
                    os.path.join(d, "out"))
            stream = next(iter_split(card, os.path.join(d, "data"), "test"))
            births = np.array([0, 159, 2, 80])
            expect = {"net": births, "fused_d2": np.minimum(births + 2, 159), "pub": births + [0, 3, 1, 2]}
            for readout, pw in expect.items():
                r = load_result(os.path.join(d, "out", readout), stream)
                np.testing.assert_array_equal(r["publish_us"], (pw + 1) * 50000)
            self.assertAlmostEqual(float(load_result(os.path.join(d, "out", "fused_d2"), stream)["prob"][0]), 0.7, 6)


if __name__ == "__main__":
    unittest.main()
