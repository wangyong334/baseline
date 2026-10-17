"""EBSSA DAVIS 转换规则的测试（合成数据，只依赖 numpy；不需要 h5py）。"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.stream_windows import load_npz_events  # noqa: E402
from tools import convert_ebssa as C  # noqa: E402

T0 = 1.0e9                                                     # 绝对时间起点（µs），与 h5 一样不从 0 开始


def recording():
    """h5 原值（坐标从 1 起、p 为 -1/1、µs 绝对时间）。
    轨迹 id 0：静止，(100, 50)，t ∈ [T0+1000, T0+101000]；
    轨迹 id 4：快速，x 从 10 线性走到 110，y = 20，t ∈ [T0+1000, T0+51000]（100 px / 50 ms）。"""
    obj = {"id": np.array([0, 0, 4, 4], dtype=np.float64),
           "ts": np.array([T0 + 1000, T0 + 101000, T0 + 1000, T0 + 51000]),
           "x": np.array([100.0, 100.0, 10.0, 110.0]), "y": np.array([50.0, 50.0, 20.0, 20.0])}
    ev = [  # (x, y, p, ts)
        (100, 50, 1, T0 + 500),        # 标注区间之前 -> 丢弃
        (105, 55, 1, T0 + 1000),       # 框边界 |dx| = |dy| = 5 -> 目标（静止）
        (106, 50, -1, T0 + 2000),      # |dx| = 6 -> 背景
        (35, 24, 1, T0 + 13500),       # 快速轨迹此刻在 x = 35 -> 目标（快速）
        (35, 24, -1, T0 + 40000),      # 快速轨迹此刻在 x = 88 -> 背景
        (240, 180, 1, T0 + 60000),     # 快速轨迹已结束；离静止目标很远 -> 背景（右下角边界坐标）
        (101, 50, 1, T0 + 90000),      # 静止目标 -> 目标
        (100, 50, -1, T0 + 101000),    # 标注区间末端（含） -> 目标
        (100, 50, 1, T0 + 101001),     # 标注区间之后 -> 丢弃
    ]
    td = {k: np.array([e[i] for e in ev], dtype=np.float64) for i, k in enumerate(("x", "y", "p", "ts"))}
    return td, obj


class ConvertTests(unittest.TestCase):
    def test_mapping(self):
        td, obj = recording()
        ev_loc, evs_norm, cls, info = C.convert_recording(td, obj)
        self.assertEqual(ev_loc.tolist(), [[104, 54, 0], [105, 49, 1], [34, 23, 12], [34, 23, 39],
                                           [239, 179, 59], [100, 49, 89], [99, 49, 100]])
        self.assertEqual(evs_norm[:, 3].tolist(), [1, 0, 1, 0, 1, 1, 0])          # p: -1 -> 0
        self.assertEqual(evs_norm[:, 4].tolist(), [1, 0, 1, 0, 0, 1, 1])          # label
        self.assertEqual(evs_norm[:, 5].tolist(), [1, 0, 5, 0, 0, 1, 1])          # target_id = Obj.id + 1
        self.assertEqual(cls.tolist(), [1, 0, 3, 0, 0, 1, 1])                     # 1 静止 / 3 快速
        self.assertEqual((info["dropped_before"], info["dropped_after"], info["stratum"]), (1, 1, "F"))
        self.assertAlmostEqual(float(evs_norm[4, 0]), 239 / 256.0, places=6)
        self.assertAlmostEqual(float(evs_norm[4, 1]), 179 / 192.0, places=6)

    def test_overlapping_boxes_take_nearest_track(self):
        obj = {"id": np.array([2, 2, 7, 7], dtype=np.float64), "ts": np.array([T0, T0 + 20, T0, T0 + 20]),
               "x": np.array([50.0, 50.0, 56.0, 56.0]), "y": np.array([50.0, 50.0, 50.0, 50.0])}
        td = {"x": np.array([52.0, 54.0, 53.0]), "y": np.array([50.0, 50.0, 50.0]),
              "p": np.array([1.0, 1.0, 1.0]), "ts": np.array([T0, T0 + 10, T0 + 20])}
        _, evs_norm, _, _ = C.convert_recording(td, obj)
        self.assertEqual(evs_norm[:, 5].tolist(), [3, 8, 3])                       # 等距时取先出现的轨迹

    def test_motion_classes_and_strata(self):
        t = np.array([0.0, 1e6])
        self.assertEqual(C.motion_class(t, np.array([0.0, 1.9]), np.array([0.0, 0.0]))[0], 1)   # 0.095 px/50ms
        self.assertEqual(C.motion_class(t, np.array([0.0, 2.1]), np.array([0.0, 0.0]))[0], 2)   # 0.105
        self.assertEqual(C.motion_class(t, np.array([0.0, 19.9]), np.array([0.0, 0.0]))[0], 2)  # 0.995
        self.assertEqual(C.motion_class(t, np.array([0.0, 30.0]), np.array([0.0, 0.0]))[0], 3)  # 1.5
        self.assertEqual((C.stratum_of({1}), C.stratum_of({1, 2}), C.stratum_of({2}), C.stratum_of({2, 3})),
                         ("S", "D", "D", "F"))

    def test_rejects_bad_input(self):
        for field, index, value in (("p", 0, 3.0), ("x", 1, 0.0), ("y", 2, 181.0), ("ts", 3, T0)):
            td, obj = recording()
            td[field][index] = value
            with self.assertRaises(ValueError):
                C.convert_recording(td, obj)
        td, obj = recording()
        obj["ts"][1] = td["ts"][-1] + 5000                                          # 标注超出录像
        with self.assertRaises(ValueError):
            C.convert_recording(td, obj)

    def test_round_trip_through_stream_reader(self):
        ev_loc, evs_norm, cls, _ = C.convert_recording(*recording())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, C.file_name("atlasrb_1851]_davis_td"))
            self.assertTrue(path.endswith("ebssa_atlasrb_1851_davis_td.npz"))
            np.savez(path, ev_loc=ev_loc, evs_norm=evs_norm, cls=cls)
            seq = load_npz_events(path, C.SENSOR_H, C.SENSOR_W, C.WINDOW_MS, 0, 5)
            self.assertEqual((seq.n_events, seq.n_windows, int(seq.label.sum())), (7, 3, 4))
            self.assertEqual(seq.x.tolist(), ev_loc[:, 0].tolist())

    def test_balanced_split(self):
        rng = np.random.RandomState(1)
        rows = [{"key": "r%02d" % i, "stratum": "SDF"[i % 3], "duration": float(rng.randint(10, 100)),
                 "events": float(rng.randint(1000, 9000)), "targets": float(rng.randint(10, 90))} for i in range(30)]
        split, dev = C.balanced_split(rows, draws=3000)
        self.assertEqual(split, C.balanced_split(rows, draws=3000)[0])              # 固定种子可复现
        self.assertLess(dev, 0.1)
        for name, _ in C.SPLIT_FRACTIONS:
            self.assertEqual({r["stratum"] for r in rows if split[r["key"]] == name}, {"S", "D", "F"})

    def test_split_protocol(self):
        keys = [k for v in C.SPLITS.values() for k in v]
        self.assertEqual(len(keys), 34)
        self.assertEqual(len(set(keys)), 34)
        self.assertEqual([len(C.SPLITS[s]) for s, _ in C.SPLIT_FRACTIONS], [21, 7, 6])
        self.assertEqual(len({C.file_name(k) for k in keys}), 34)
        self.assertEqual((C.CANVAS_W % 8, C.CANVAS_H % 8), (0, 0))
        self.assertTrue(C.CANVAS_H >= C.SENSOR_H and C.CANVAS_H - C.SENSOR_H < 8 and C.CANVAS_W == C.SENSOR_W)


if __name__ == "__main__":
    unittest.main()
