import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speed.data.events import EventStream  # noqa: E402
from speed.eval.breakdown import (SIZE_EDGES, SPEED_EDGES, RecallBreakdown, bin_index, bin_labels,  # noqa: E402
                                  target_kinematics)

WIN = 50000


def reference_kinematics(stream, win_us=WIN):
    """Loop version used by the V4 problem analysis (M1/E), kept here as the specification."""
    ev_vx, ev_vy, ev_size = (np.full(stream.n_events, np.nan) for _ in range(3))
    tgt = np.flatnonzero(stream.label == 1)
    for ident in np.unique(stream.target_id[tgt]):
        m = tgt[stream.target_id[tgt] == ident]
        w = stream.t[m] // win_us
        uw = np.unique(w)
        cx = np.array([stream.x[m][w == k].mean() for k in uw])
        cy = np.array([stream.y[m][w == k].mean() for k in uw])
        vx, vy = np.full(uw.size, np.nan), np.full(uw.size, np.nan)
        for i in range(uw.size):
            lo = i - 1 if i > 0 and uw[i - 1] == uw[i] - 1 else i
            hi = i + 1 if i + 1 < uw.size and uw[i + 1] == uw[i] + 1 else i
            if hi > lo:
                dt_ms = (uw[hi] - uw[lo]) * win_us / 1000.0
                vx[i], vy[i] = (cx[hi] - cx[lo]) / dt_ms, (cy[hi] - cy[lo]) / dt_ms
        widths = []
        for i, k in enumerate(uw):
            sel = m[w == k]
            if sel.size < 5 or not np.isfinite(vx[i]):
                continue
            sp = np.hypot(vx[i], vy[i])
            if sp < 1e-6:
                widths.append(max(np.percentile(stream.x[sel], 95) - np.percentile(stream.x[sel], 5),
                                  np.percentile(stream.y[sel], 95) - np.percentile(stream.y[sel], 5)) + 1.0)
                continue
            proj = stream.x[sel] * (-vy[i] / sp) + stream.y[sel] * (vx[i] / sp)
            widths.append(np.percentile(proj, 95) - np.percentile(proj, 5) + 1.0)
        size = float(np.median(widths)) if widths else np.nan
        idx = np.searchsorted(uw, w)
        ev_vx[m], ev_vy[m], ev_size[m] = vx[idx], vy[idx], size
    return ev_vx, ev_vy, ev_size


def make_stream(parts, width=200, height=120, cls_of=None):
    """parts: list of (x, y, t_us, target_id) arrays."""
    x, y, t, tid = (np.concatenate([np.asarray(p[i]) for p in parts]) for i in range(4))
    order = np.random.RandomState(1).permutation(t.size)            # file order need not be time order
    x, y, t, tid = x[order], y[order], t[order], tid[order]
    cls = (tid > 0).astype(np.int16) if cls_of is None else np.array([cls_of.get(int(i), 0) for i in tid])
    return EventStream("s", x, y, t, np.zeros(t.size), tid > 0, tid, cls, width, height).validate()


def bar(target_id, vx_px_ms, vy_px_ms, t0_ms, t1_ms, x0=20.0, y0=40.0, width_px=3, per_ms=2):
    t = np.repeat(np.arange(t0_ms, t1_ms), per_ms * width_px) * 1000
    k = np.tile(np.arange(width_px), (t1_ms - t0_ms) * per_ms)
    tm = t / 1000.0
    x = np.round(x0 + vx_px_ms * (tm - t0_ms) + (k if vx_px_ms == 0 else 0)).astype(np.int64)
    y = np.round(y0 + vy_px_ms * (tm - t0_ms) + (k if vx_px_ms != 0 else 0)).astype(np.int64)
    return x, y, t, np.full(t.size, target_id)


class KinematicsTests(unittest.TestCase):
    def test_matches_reference_loop(self):
        rng = np.random.RandomState(0)
        parts = [bar(1, 0.1, 0.0, 0, 400), bar(2, 0.0, 0.0, 30, 260, x0=100), bar(3, -0.05, 0.08, 120, 500, x0=150),
                 bar(4, 0.3, 0.0, 600, 640, x0=10),                         # single window track: unknown velocity
                 (rng.randint(0, 200, 500), rng.randint(0, 120, 500), rng.randint(0, 700000, 500), np.zeros(500))]
        x, y, t, tid = (np.concatenate([np.asarray(p[i]) for p in parts]) for i in range(4))
        gap = ~((tid == 3) & (t >= 250000) & (t < 300000))                  # a missing window inside track 3
        s = make_stream([(x[gap], y[gap], t[gap], tid[gap])])
        got, want = target_kinematics(s, WIN), reference_kinematics(s, WIN)
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w, rtol=0, atol=1e-12, equal_nan=True)
        self.assertTrue(np.isnan(got[0][s.target_id == 4]).all())

    def test_known_velocity_and_size(self):
        s = make_stream([bar(1, 0.1, 0.0, 0, 300)])
        vx, vy, size = target_kinematics(s, WIN)
        inner = (s.t >= 50000) & (s.t < 250000)
        np.testing.assert_allclose(vx[inner], 0.1, atol=1e-9)
        np.testing.assert_allclose(vy[inner], 0.0, atol=1e-9)
        self.assertAlmostEqual(float(size[0]), 3.0)


class BreakdownTests(unittest.TestCase):
    def test_bins(self):
        b = bin_index([0.0, 0.99, 4.0, 63.9, 64.0, 1e6, np.nan], SPEED_EDGES)
        labels = bin_labels(SPEED_EDGES)
        self.assertEqual([labels[i] for i in b], ["0-1", "0-1", "4-8", "32-64", "64+", "64+", "unknown"])
        self.assertEqual(bin_labels(SIZE_EDGES)[-2:], ["24+", "unknown"])

    def test_recall_and_kept(self):
        # slow target: 0.03 px/ms = 1.5 px/step; fast target: 0.4 px/ms = 20 px/step; 50 background events
        rng = np.random.RandomState(2)
        bg = (rng.randint(0, 200, 50), rng.randint(0, 120, 50), rng.randint(0, 400000, 50), np.zeros(50))
        s = make_stream([bar(1, 0.03, 0.0, 0, 400), bar(2, 0.4, 0.0, 0, 400, y0=80), bg], cls_of={1: 1, 2: 2})
        slow, fast = s.target_id == 1, s.target_id == 2
        dec = {"a": s.label == 1, "b": slow.copy()}
        dec["b"][np.flatnonzero(s.label == 0)[:7]] = True                   # 7 false target events
        acc = RecallBreakdown(["a", "b"], reference="a")
        acc.update(s, dec)
        out = acc.result()
        rows = {r["bin"]: r for r in out["tables"]["speed"]["all"]}
        self.assertEqual(rows["1-2"]["recall"], {"a": 1.0, "b": 1.0})
        self.assertEqual(rows["16-32"]["recall"], {"a": 1.0, "b": 0.0})
        self.assertEqual(rows["16-32"]["kept_vs_reference"]["b"], 0.0)
        self.assertEqual(rows["1-2"]["events"] + rows["16-32"]["events"] + rows["unknown"]["events"],
                         int((s.label == 1).sum()))
        self.assertEqual(out["false_events"], {"a": 0, "b": 7})
        cls2 = {r["bin"]: r for r in out["tables"]["speed"]["class2"]}
        self.assertEqual(cls2["16-32"]["events"], rows["16-32"]["events"])
        self.assertIsNone(cls2["1-2"]["recall"]["a"])


if __name__ == "__main__":
    unittest.main()
