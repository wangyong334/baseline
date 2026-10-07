import io
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speed.data.dataset_card import list_recordings, load_card  # noqa: E402
from speed.data.events import EventStream  # noqa: E402
from speed.data.readers import read_evflying_npy, read_evuav_npz  # noqa: E402
from speed.data.stats import object_window_rows, split_statistics  # noqa: E402


def evuav_blob(rng, n=5000, n_targets=3):
    x = rng.integers(0, 346, n)
    y = rng.integers(0, 260, n)
    t = rng.integers(0, 8000, n)
    t[:5] = [0, 50, 100, 7950, 7999]  # window edges
    p = rng.integers(0, 2, n)
    tid = np.where(rng.random(n) < 0.1, rng.integers(1, n_targets + 1, n), 0)
    label = (tid > 0).astype(np.float32)
    loc = np.stack([x, y, t], 1).astype(np.int64)
    norm = np.stack([x / 352.0, y / 288.0, t / 8192.0, p, label, tid], 1).astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, ev=loc, ev_loc=loc, evs_norm=norm)
    return buf.getvalue()


def stream_of(t, width=10, height=10, **kw):
    n = len(t)
    fields = dict(x=np.zeros(n, int), y=np.zeros(n, int), p=np.zeros(n, int), label=np.zeros(n, int),
                  target_id=np.zeros(n, int), cls=np.zeros(n, int))
    fields.update(kw)
    return EventStream("s", t=np.asarray(t), width=width, height=height, **fields)


class EventStreamTests(unittest.TestCase):
    def test_window_partition_is_stable_time_sort(self):
        rng = np.random.default_rng(0)
        t = rng.integers(0, 8000, 20000) * 1000
        s = stream_of(t)
        order, bounds = s.window_partition(50000, 160)
        np.testing.assert_array_equal(order, np.argsort(t, kind="stable"))
        for k in (0, 1, 80, 159):
            np.testing.assert_array_equal(np.sort(order[bounds[k]:bounds[k + 1]]),
                                          np.flatnonzero(t // 50000 == k))
        self.assertEqual(bounds[-1], t.size)

    def test_partition_any_window_and_origin(self):
        s = stream_of([5, 120, 250, 251, 999])
        order, bounds = s.window_partition(100, origin_us=100)
        self.assertEqual(s.n_windows(100, origin_us=100), 9)
        np.testing.assert_array_equal(order, [1, 2, 3, 4])  # the event before the origin is excluded
        np.testing.assert_array_equal(bounds[:4], [0, 1, 3, 3])
        self.assertEqual(bounds[-1], 4)

    def test_time_slice(self):
        s = stream_of([10, 20, 30, 40], x=np.array([1, 2, 3, 4]))
        sub, idx = s.time_slice(15, 35)
        np.testing.assert_array_equal(idx, [1, 2])
        np.testing.assert_array_equal(sub.t, [5, 15])
        np.testing.assert_array_equal(sub.x, [2, 3])
        self.assertEqual(sub.span_us, 20)

    def test_validate_rejects_inconsistent_labels(self):
        s = stream_of([1, 2], label=np.array([1, 0]), target_id=np.array([0, 0]), cls=np.array([1, 0]))
        with self.assertRaises(ValueError):
            s.validate()
        s = stream_of([1, 2], x=np.array([0, 10]))
        with self.assertRaises(ValueError):
            s.validate()


class ReaderTests(unittest.TestCase):
    def test_evuav_reader(self):
        card = load_card("evuav")
        blob = evuav_blob(np.random.default_rng(1))
        with np.load(io.BytesIO(blob)) as data:
            loc, norm = data["ev_loc"], data["evs_norm"]
        s = read_evuav_npz(io.BytesIO(blob), card, "x.npz")
        np.testing.assert_array_equal(s.x, loc[:, 0])
        np.testing.assert_array_equal(s.y, loc[:, 1])
        np.testing.assert_array_equal(s.t, loc[:, 2] * 1000)
        np.testing.assert_array_equal(s.target_id, norm[:, 5].astype(np.int64))
        np.testing.assert_array_equal(s.label, (norm[:, 5] > 0).astype(np.uint8))
        np.testing.assert_array_equal(s.cls, (norm[:, 5] > 0).astype(np.int16))
        self.assertEqual(s.span_us, 8000000)

    def test_evflying_reader(self):
        card = load_card("evflying")
        raw = np.array([[10, 20, 1, 105, -1, 0], [11, 21, 0, 106, 3, 2], [1279, 719, 1, 120000883, 1, 1]], float)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.npy")
            np.save(path, raw)
            s = read_evflying_npy(path, card)
        np.testing.assert_array_equal(s.target_id, [0, 3, 1])
        np.testing.assert_array_equal(s.label, [0, 1, 1])
        np.testing.assert_array_equal(s.cls, [0, 2, 1])
        np.testing.assert_array_equal(s.t, [105, 106, 120000883])
        self.assertEqual(s.span_us, 120000884)

    def test_evflying_reader_rejects_zero_id(self):
        card = load_card("evflying")
        raw = np.array([[1, 1, 1, 5, 0, 2]], float)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.npy")
            np.save(path, raw)
            with self.assertRaises(ValueError):
                read_evflying_npy(path, card)


class CardTests(unittest.TestCase):
    def test_evflying_splits(self):
        card = load_card("evflying")
        train, val, test = (set(card["splits"][s]["files"]) for s in ("train", "val", "test"))
        self.assertEqual(sorted(val), sorted(["Train/3/3.npy", "Train/10/10.npy", "Train/17/17.npy", "Train/21/21.npy"]))
        self.assertEqual(len(train), 17)
        self.assertEqual(len(test), 6)
        self.assertFalse(train & val or train & test or val & test)

    def test_directory_split(self):
        card = load_card("evuav")
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "test"))
            for name in ("b.npz", "a.npz", "note.txt"):
                open(os.path.join(d, "test", name), "w").close()
            names = [n for n, _ in list_recordings(card, d, "test")]
        self.assertEqual(names, ["test/a.npz", "test/b.npz"])


class StatsTests(unittest.TestCase):
    def test_known_target(self):
        # a 4x4 target moving +10 px per 50 ms window for 10 windows, 16 events per window, plus background
        xs, ys, ts, ids = [], [], [], []
        for k in range(10):
            for dx in range(4):
                for dy in range(4):
                    xs.append(10 + 10 * k + dx)
                    ys.append(20 + dy)
                    ts.append(k * 50000 + 1000)
                    ids.append(1)
        n_bg = 500
        rng = np.random.default_rng(2)
        xs += list(rng.integers(0, 200, n_bg))
        ys += list(rng.integers(0, 100, n_bg))
        ts += list(rng.integers(0, 1000000, n_bg))
        ids += [0] * n_bg
        ids = np.array(ids)
        s = EventStream("syn", xs, ys, ts, np.zeros(len(ts)), (ids > 0), ids, (ids > 0) * 2, 200, 100, span_us=1000000)
        s.validate()
        rows = object_window_rows(s, 50000)
        self.assertEqual(rows.shape[0], 10)
        stats = split_statistics([s], 50000)
        self.assertAlmostEqual(stats["target_size_px"]["p50"], 4.0)
        self.assertAlmostEqual(stats["target_speed_px_s"]["p50"], 200.0)
        self.assertAlmostEqual(stats["target_rel_speed_per_s"]["p50"], 50.0)
        self.assertAlmostEqual(stats["target_duration_s"]["p50"], 0.5)
        self.assertAlmostEqual(stats["target_events_per_s"]["p50"], 320.0)
        self.assertAlmostEqual(stats["bg_rate_per_px_s"]["p50"], n_bg / (200 * 100 * 1.0))
        self.assertEqual(stats["per_class"]["2"]["targets"], 1)
        self.assertEqual(stats["n_targets"], 1)


if __name__ == "__main__":
    unittest.main()
