"""tools/edds_convert.py 的测试（全部用合成数据，不需要真实的 EDDS）。

守住四件事：
    1 EVT 3.0 解码：手工构造的字序列（单个事件、向量事件、TIME_LOW、TIME_HIGH 回绕、首个 TIME_HIGH 之前的字）
      解码出预期的事件；任意分块大小与整块解码结果相同
    2 时间原点：first_time_high / label_time_shift 与定义一致
    3 端到端转换：热像素被去掉、目标事件一个不少、坐标整除合并倍数、按序列长度切段、读回（load_npz_events）一致
    4 分辨率上限：每格多数标签的最好情况 IoU / ACC 与手算一致
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from dataset.stream_windows import load_npz_events
from tools import edds_convert as ec


def th(v):
    return (0x8 << 12) | v


def tl(v):
    return (0x6 << 12) | v


def addr_y(v):
    return v


def addr_x(x, p):
    return (0x2 << 12) | (p << 11) | x


def vect_base(x, p):
    return (0x3 << 12) | (p << 11) | x


def vect12(mask):
    return (0x4 << 12) | mask


def vect8(mask):
    return (0x5 << 12) | mask


def write_raw(path, words):
    with open(path, "wb") as f:
        f.write(b"% evt 3.0\n% format EVT3;height=720;width=1280\n% end\n")
        f.write(np.asarray(words, dtype="<u2").tobytes())


def collect(path, chunk_words, shift=0):
    parts = [np.stack([x, y, t, p.astype(np.int64)], 1) for x, y, t, p in ec.iter_events(path, chunk_words, shift)
             if x.size]
    ev = np.concatenate(parts) if parts else np.zeros((0, 4), np.int64)
    return sorted(map(tuple, ev.tolist()))


def encode_events(events):
    """按时间顺序的 (x, y, t_us, p) 编成只含单个事件的 EVT3 字序列。"""
    words, last_high = [], None
    for x, y, t, p in sorted(events, key=lambda e: e[2]):
        high, low = t >> 12, t & 0xFFF
        if high != last_high:
            words.append(th(high & 0xFFF))
            last_high = high
        words += [tl(low), addr_y(y), addr_x(x, p)]
    return words


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def test_handbuilt_stream_and_chunk_invariance(self):
        words = [addr_y(1), addr_x(9, 1),                                   # 首个 TIME_HIGH 之前：丢弃
                 th(7), addr_y(10), addr_x(20, 1),                          # t = 7<<12
                 tl(5), addr_x(21, 0),                                      # t = 7<<12 + 5
                 vect_base(100, 1), vect12(0b000000000101), vect8(0b10000001),   # x 100,102 然后 112,119
                 th(4095), tl(1), addr_y(11), addr_x(30, 0),                # t = 4095<<12 + 1
                 th(0), addr_x(31, 1)]                                      # 回绕：t = 2^24（低位清零）
        path = os.path.join(self.dir, "a.raw")
        write_raw(path, words)
        t1, t2 = 7 << 12, (7 << 12) + 5
        want = sorted([(20, 10, t1, 1), (21, 10, t2, 0), (100, 10, t2, 1), (102, 10, t2, 1), (112, 10, t2, 1),
                       (119, 10, t2, 1), (30, 11, (4095 << 12) + 1, 0), (31, 11, 1 << 24, 1)])
        self.assertEqual(collect(path, 10 ** 6), want)
        for chunk in (1, 2, 3, 5, 7):
            self.assertEqual(collect(path, chunk), want, "chunk=%d" % chunk)
        self.assertEqual(ec.first_time_high(path), 7)
        self.assertEqual(ec.label_time_shift(path), 6 << 12)
        shifted = collect(path, 4, shift=ec.label_time_shift(path))
        self.assertEqual(shifted, sorted((x, y, t - (6 << 12), p) for x, y, t, p in want))

    def test_random_stream_chunk_invariance(self):
        rng = np.random.RandomState(0)
        events = [(int(rng.randint(0, 1280)), int(rng.randint(0, 720)), int(t), int(rng.randint(0, 2)))
                  for t in np.sort(rng.randint(20000, 3_000_000, 400))]
        path = os.path.join(self.dir, "b.raw")
        write_raw(path, encode_events(events))
        full = collect(path, 10 ** 6)
        self.assertEqual(full, sorted(events))
        for chunk in (1, 3, 17, 64):
            self.assertEqual(collect(path, chunk), full)


class ConvertTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def test_end_to_end(self):
        rng = np.random.RandomState(1)
        t0 = 5 << 12                                                        # 首个 TIME_HIGH = 5 -> 标签原点 4<<12
        shift = 4 << 12
        target = [(40 + i % 4, 20 + i % 2, t0 + 300 + 997 * i, i % 2) for i in range(150)]
        hot = [(500, 300, t0 + 7 + 390 * i, 0) for i in range(520)]         # 约 0.2 s 里 520 个：约 2600 次/秒
        background = [(int(rng.randint(0, 1280)), int(rng.randint(0, 720)), int(t0 + rng.randint(0, 200_000)),
                       int(rng.randint(0, 2))) for _ in range(300)]
        background = [e for e in background if not (e[0] == 500 and e[1] == 300)]
        events = target + hot + background
        write_raw(os.path.join(self.dir, "rec.raw"), encode_events(events))
        os.makedirs(os.path.join(self.dir, "rec"))
        with open(os.path.join(self.dir, "rec", "segmentation-000.csv"), "w") as f:
            f.write("x;y;t;p\n")
            for x, y, t, p in target:
                f.write("%d;%d;%d;%d\n" % (x, y, t - shift, p))
        out = os.path.join(self.dir, "out")
        rec = ec.convert_recording(self.dir, "rec", out, factor=4, hot_rate=1500.0, segment_ms=100, window_ms=50)
        self.assertEqual(rec["hot_pixels"][0][:2], [500, 300])
        self.assertEqual(len(rec["hot_pixels"]), 1)
        self.assertEqual(rec["segmentation_found_in_raw"], 1.0)
        self.assertEqual(rec["target_events_kept"], len(target))
        self.assertEqual(rec["target_events_on_hot_pixels"], 0)
        self.assertEqual(rec["events_kept"], len(target) + len(background))
        self.assertEqual(rec["binned_size"], [320, 180])
        total_events, total_target = 0, 0
        for s in rec["sequences"]:
            seq = load_npz_events(os.path.join(out, s["file"]), 260, 346, 50, 2, 5)
            total_events += seq.n_events
            total_target += int(seq.label.sum())
            self.assertTrue(np.array_equal(seq.target_id, seq.label.astype(np.float64)))
            self.assertLess(int(seq.x.max()), 320)
            self.assertLess(int(seq.y.max()), 180)
            self.assertLess(int(seq.t.max()), 100)
        self.assertEqual((total_events, total_target), (len(target) + len(background), len(target)))
        # 目标事件的合并坐标 = 原生坐标 // 4，时间 = (t - 原点) // 1000 再减去所在序列的起点
        first = load_npz_events(os.path.join(out, rec["sequences"][0]["file"]), 260, 346, 50, 2, 5)
        tx = sorted(zip(first.x[first.label > 0].tolist(), first.y[first.label > 0].tolist(),
                        first.t[first.label > 0].tolist()))
        want = sorted((x // 4, y // 4, (t - shift) // 1000) for x, y, t, _ in target if (t - shift) // 1000 < 100)
        self.assertEqual(tx, want)


class CeilingTests(unittest.TestCase):
    def test_majority_ceiling(self):
        t = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        x = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3])
        y = np.zeros(9, np.int64)
        lab = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0])
        c = ec.resolution_ceiling(t, x, y, lab, 50_000, 10)
        self.assertAlmostEqual(c["iou"], 2.0 / 4.0)
        self.assertAlmostEqual(c["acc"], 2.0 / 3.0)
        self.assertAlmostEqual(c["target_events_in_mixed_cells"], 1.0)


if __name__ == "__main__":
    unittest.main()
