"""tools/error_breakdown.py 的单元测试：手工构造一条序列，逐类核对漏检 / 误检的归类与"修好后 IoU"的算术。"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from tools import error_breakdown as eb


def write_dump(directory, name, x, y, t, labels, tid, **probs):
    locs = np.stack([np.zeros_like(x), x, y, t], 1).astype(np.float64)
    np.savez(os.path.join(directory, name), locs=locs, labels=np.asarray(labels, np.float32),
             target_id=np.asarray(tid, np.float64), **probs)


class ErrorBreakdownTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # 目标 1：t = 900 ms 起出现（不在序列开头），在第 18 / 30 窗各有一簇
        x, y, t, lab, tid, p_net, p_fix = [], [], [], [], [], [], []

        def add(xx, yy, tt, is_t, pn, pf):
            x.append(xx); y.append(yy); t.append(tt); lab.append(1 if is_t else 0); tid.append(1 if is_t else 0)
            p_net.append(pn); p_fix.append(pf)

        add(20, 20, 100, True, 0.1, 0.1)                      # start（t < 800）：但目标 1 也在这里出现 -> 首次时间 100
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):         # 第 30 窗（t=1510）的核心簇，age >= 250
            add(40 + dx, 40 + dy, 1510, True, 0.95, 0.95)
        add(46, 40, 1510, True, 0.2, 0.95)                      # 外圈：edge 漏检，被第二个读出修好
        add(41, 41, 1520, True, 0.3, 0.3)                       # core 漏检（离质心近）
        add(43, 40, 1512, False, 0.95, 0.1)                     # adjacent 误检（<= 3 px），被修好
        add(50, 40, 1512, False, 0.95, 0.95)                    # near 误检（<= 10 px）
        add(100, 100, 1512, False, 0.95, 0.95)                  # far 误检
        for k in range(5):                                      # 同一像素的 5 次 no_target 误检（重复像素）
            add(5, 5, 3000 + 50 * k, False, 0.95, 0.95)
        arr = lambda v: np.asarray(v, np.float64)  # noqa: E731
        write_dump(self.tmp, "seq.npz", arr(x), arr(y), arr(t), lab, tid,
                   probabilities=np.asarray(p_net, np.float32), prob_fix=np.asarray(p_fix, np.float32))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_classes_and_headroom(self):
        args = eb.parse_args(["--dump-dir", self.tmp, self.tmp + ":prob_fix", "--thresholds", "0.9"])
        seqs, names = eb.load_sequences(args.dump_dir)
        res = eb.breakdown(seqs, names, args.thresholds, args)["0.9"]
        net = res[names[0]]
        self.assertEqual(net["counts"], {"tp": 4, "fp": 8, "fn": 3})
        self.assertEqual({k: v["n"] for k, v in net["fn"].items()}, {"start": 1, "onset": 0, "edge": 1, "core": 1})
        self.assertEqual({k: v["n"] for k, v in net["fp"].items()}, {"adjacent": 1, "near": 1, "far": 1, "no_target": 5})
        self.assertEqual(net["fp_repeat_pixels"]["no_target"], 5)
        self.assertAlmostEqual(net["iou"], 4 / 15.0)
        self.assertAlmostEqual(net["fn"]["edge"]["iou_if_fixed"], 5 / 15.0)
        self.assertAlmostEqual(net["fp"]["no_target"]["iou_if_fixed"], 4 / 10.0)
        fix = res[names[1]]
        self.assertEqual(fix["vs_first"]["fixed"]["fn"]["edge"], 1)
        self.assertEqual(fix["vs_first"]["fixed"]["fp"]["adjacent"], 1)
        self.assertEqual(sum(fix["vs_first"]["new"]["fn"].values()) + sum(fix["vs_first"]["new"]["fp"].values()), 0)

    def test_onset_class(self):
        args = eb.parse_args(["--dump-dir", self.tmp, "--onset-ms", "2000"])
        seqs, names = eb.load_sequences(args.dump_dir)
        res = eb.breakdown(seqs, names, [0.9], args)["0.9"][names[0]]
        self.assertEqual(res["fn"]["onset"]["n"], 2)             # 年龄 < 2000 ms 的两个非开头漏检
        self.assertEqual(res["fn"]["start"]["n"], 1)


    def test_repeat_counts_distinct_windows(self):
        """同一像素、同一窗里的 6 个误检（一次突发）不算重复像素；分布在 5 个不同窗才算。"""
        n = 11
        x = np.array([7.0] * 6 + [9.0] * 5)
        t = np.array([2000.0] * 6 + [2000.0 + 50 * k for k in range(5)])
        write_dump(self.tmp, "burst.npz", x, np.full(n, 3.0), t, np.zeros(n), np.zeros(n),
                   probabilities=np.ones(n, np.float32), prob_fix=np.ones(n, np.float32))
        args = eb.parse_args(["--dump-dir", self.tmp, "--names", "burst"])
        seqs, names = eb.load_sequences(args.dump_dir, 0, args.names)
        res = eb.breakdown(seqs, names, [0.9], args)["0.9"][names[0]]
        self.assertEqual(res["fp"]["no_target"]["n"], 11)
        self.assertEqual(res["fp_repeat_pixels"]["no_target"], 5)

    def test_logit_sum_filter_and_per_sequence(self):
        n = 21
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=n).astype(np.float32), rng.normal(size=n).astype(np.float32)
        write_dump(self.tmp, "other.npz", np.arange(n, dtype=np.float64), np.zeros(n), np.full(n, 900.0),
                   np.zeros(n), np.zeros(n), probabilities=np.zeros(n, np.float32), prob_fix=np.zeros(n, np.float32),
                   la=a, lb=b)
        args = eb.parse_args(["--dump-dir", self.tmp + ":la+lb", "--names", "other", "--per-sequence"])
        seqs, names = eb.load_sequences(args.dump_dir, 0, args.names)
        self.assertEqual([s["name"] for s in seqs], ["other.npz"])
        want = 1.0 / (1.0 + np.exp(-(a.astype(np.float64) + b)))
        self.assertLess(float(np.abs(seqs[0]["probs"][names[0]] - want).max()), 1e-6)
        rows = eb.per_sequence(seqs, names, 0.5, args)
        self.assertEqual(rows[0]["readouts"][names[0]]["fp"], int((want >= 0.5).sum()))

    def test_select_eval_names(self):
        import train_stream_v2 as tv2
        self.assertEqual(tv2.select_eval_names(self.tmp, 0, ["seq"]), ["seq.npz"])
        self.assertEqual(tv2.select_eval_names(self.tmp, 1, None), ["seq.npz"])
        self.assertIsNone(tv2.select_eval_names(self.tmp, 0, None))
        with self.assertRaises(ValueError):
            tv2.select_eval_names(self.tmp, 0, ["missing"])
        with self.assertRaises(ValueError):
            tv2.select_eval_names(self.tmp, 2, ["seq"])


if __name__ == "__main__":
    unittest.main()
