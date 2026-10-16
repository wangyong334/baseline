"""EV-Flying 4x4 合并转换的规则测试（合成数据，只依赖 numpy）。"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.stream_windows import load_npz_events  # noqa: E402
from tools import convert_evflying_x4 as C  # noqa: E402


def raw_events():
    """列 x, y, p, t_us, id, class；含边界坐标、同毫秒内的两个事件、两个目标。"""
    return np.array([[0, 0, 1, 105, -1, 0],
                     [3, 3, 0, 999, -1, 0],
                     [1279, 719, 1, 1000, 7, 2],
                     [640, 361, 0, 1999.6, 7, 2],
                     [5, 6, 1, 52000, 3, 1]], dtype=np.float64)


class ConvertTests(unittest.TestCase):
    def test_mapping(self):
        ev_loc, evs_norm, cls = C.convert_arrays(raw_events())
        self.assertEqual(ev_loc.tolist(), [[0, 0, 0], [0, 0, 0], [319, 179, 1], [160, 90, 1], [1, 1, 52]])
        self.assertEqual(evs_norm[:, 3].tolist(), [1, 0, 1, 0, 1])                  # p
        self.assertEqual(evs_norm[:, 4].tolist(), [0, 0, 1, 1, 1])                  # label
        self.assertEqual(evs_norm[:, 5].tolist(), [0, 0, 7, 7, 3])                  # target_id，背景 0
        self.assertEqual(cls.tolist(), [0, 0, 2, 2, 1])
        self.assertAlmostEqual(float(evs_norm[2, 0]), 319 / 320.0, places=6)
        self.assertAlmostEqual(float(evs_norm[2, 1]), 179 / 192.0, places=6)

    def test_rejects_bad_ids_and_inconsistent_class(self):
        bad = raw_events()
        bad[0, 4] = 0
        with self.assertRaises(ValueError):
            C.convert_arrays(bad)
        bad = raw_events()
        bad[0, 5] = 1                                                                # 背景却标了类别
        with self.assertRaises(ValueError):
            C.convert_arrays(bad)
        bad = raw_events()
        bad[2, 3] = 10                                                               # 时间倒退
        with self.assertRaises(ValueError):
            C.convert_arrays(bad)

    def test_round_trip_through_stream_reader(self):
        ev_loc, evs_norm, cls = C.convert_arrays(raw_events())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.npz")
            np.savez(path, ev_loc=ev_loc, evs_norm=evs_norm, cls=cls)
            seq = load_npz_events(path, C.VALID_H, C.VALID_W, C.WINDOW_MS, 2, 5)
            self.assertEqual(seq.n_events, 5)
            self.assertEqual(seq.x.tolist(), [0, 0, 319, 160, 1])
            self.assertEqual(int(seq.label.sum()), 3)
            self.assertEqual(C.sha256_array(ev_loc), C.sha256_array(ev_loc.copy()))

    def test_split_protocol(self):
        names = [k for items in C.SPLITS.values() for k in items]
        self.assertEqual(len(names), 27)
        self.assertEqual(len(set(names)), 27)
        self.assertEqual(len(C.SPLITS["train"]) + len(C.SPLITS["val"]), 21)
        self.assertEqual(sorted(i for _, i in C.SPLITS["test"]), [5, 9, 13, 23, 24, 26])
        self.assertEqual((C.CANVAS_W % 8, C.CANVAS_H % 8), (0, 0))
        self.assertTrue(C.CANVAS_H >= C.VALID_H and C.CANVAS_H - C.VALID_H < 8)


if __name__ == "__main__":
    unittest.main()
