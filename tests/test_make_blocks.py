"""K5 基线 8 s 切块规则的测试（合成数据，只依赖 numpy）。"""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import make_blocks as M  # noqa: E402


def recording(times):
    t = np.asarray(times, dtype=np.int64)
    n = t.size
    x, y = np.arange(n) % 320, (np.arange(n) * 7) % 180
    label = (np.arange(n) % 3 == 0).astype(np.float32)
    ev_loc = np.stack([x, y, t], 1).astype(np.int64)
    evs_norm = np.stack([x / 320.0, y / 192.0, t / 8192.0, np.arange(n) % 2, label, label * 4], 1).astype(np.float32)
    return ev_loc, evs_norm


class BlockTests(unittest.TestCase):
    def test_partition_and_relative_time(self):
        ev_loc, evs_norm = recording([0, 7999, 8000, 8000, 15999, 24000, 120000])
        parts = M.split_blocks(ev_loc, evs_norm)
        self.assertEqual([b for b, _ in parts], [0, 1, 3, 15])                    # 空块（块 2）不写
        self.assertEqual([(s.start, s.stop) for _, s in parts], [(0, 2), (2, 5), (5, 6), (6, 7)])
        loc, norm = M.block_arrays(ev_loc, evs_norm, 1, parts[1][1], 320, 192)
        self.assertEqual(loc[:, 2].tolist(), [0, 0, 7999])                        # 块内时间
        self.assertAlmostEqual(float(norm[2, 2]), 7999 / 8192.0, places=6)
        self.assertEqual(norm[:, 3:6].tolist(), evs_norm[2:5, 3:6].tolist())       # p / label / target_id 照抄
        last = M.block_arrays(ev_loc, evs_norm, 15, parts[3][1], 320, 192)[0]
        self.assertEqual(last[:, 2].tolist(), [0])                                # 120000 ms 落在块 15 的开头

    def test_rejects_unsorted_time(self):
        ev_loc, evs_norm = recording([0, 10, 5])
        with self.assertRaises(ValueError):
            M.split_blocks(ev_loc, evs_norm)

    def test_tool_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
            for split in ("train", "val"):
                os.makedirs(os.path.join(src, split))
                ev_loc, evs_norm = recording(np.sort(np.random.RandomState(1).randint(0, 30000, 500)))
                np.savez(os.path.join(src, split, "rec_a.npz"), ev_loc=ev_loc, evs_norm=evs_norm,
                         cls=np.zeros(500, dtype=np.int8))
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_blocks.py"), "--src", src,
                                  "--dst", dst, "--k5-width", "320", "--k5-height", "192", "--splits", "train", "val"],
                                 cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", env=env)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("BLOCKS FINISHED", out.stdout)
            self.assertEqual(sorted(os.listdir(os.path.join(dst, "train"))),
                             ["rec_a_b000.npz", "rec_a_b001.npz", "rec_a_b002.npz", "rec_a_b003.npz"])
            with np.load(os.path.join(dst, "train", "rec_a_b001.npz")) as z:
                self.assertTrue(z["ev_loc"][:, 2].max() < 8000 and z["ev_loc"][:, 2].min() >= 0)
                self.assertEqual(set(z.files), {"ev_loc", "evs_norm", "cls"})


if __name__ == "__main__":
    unittest.main()
