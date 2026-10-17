"""长录像 / 可变长度支持的测试（CPU，合成数据）。

    1. FastRoc 与原 utils/eval.py 的 roc_update / cal_roc 逐项相同（含边界：t 为 50 的整数倍、序列不从 0 开始、
       同一像素虚警 >= 256 次的 uint8 回绕、目标号存在但标签为 0、概率恰为 0.9）
    2. n_windows <= 0 时读取器按录像自身长度取窗数；固定 160 时与原行为相同
    3. 长度不同的序列一起做完整评估（逐窗统计按窗号对齐累加）
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import stream_windows as sw  # noqa: E402
from utils import stream_metrics as sm  # noqa: E402


def have_eval_dependencies():
    try:
        import cv2  # noqa: F401
        import pandas  # noqa: F401
        import tqdm  # noqa: F401
        return True
    except ImportError:
        return False


def synthetic_roc_sequence(seed, start=0, length_ms=8000, n=4000):
    rng = np.random.RandomState(seed)
    t = np.sort(rng.randint(start, start + length_ms, n)).astype(np.int64)
    t[::17] = (t[::17] // 50) * 50                         # 一部分事件恰在帧边界上
    x, y = rng.randint(0, 346, n), rng.randint(0, 260, n)
    target = np.zeros(n)
    target[rng.rand(n) < 0.15] = rng.randint(1, 4, 1)[0]
    target[rng.rand(n) < 0.1] = 2
    label = (target != 0).astype(np.float32)
    label[rng.rand(n) < 0.02] = 0                          # 少量"有目标号但标签为 0"
    prob = rng.rand(n).astype(np.float32)
    prob[rng.rand(n) < 0.05] = np.float32(0.9)             # 恰为阈值
    hot = rng.rand(n) < 0.1                                # 一个热点像素上的大量虚警（触发 uint8 回绕）
    x[hot], y[hot] = 100, 50
    label[hot], target[hot], prob[hot] = 0, 0, 0.99
    return t, x, y, prob, target, label


@unittest.skipUnless(have_eval_dependencies(), "原评测依赖 cv2/pandas/tqdm")
class FastRocTests(unittest.TestCase):
    def test_matches_original_roc_update(self):
        from types import SimpleNamespace
        from utils.eval import evalute
        for thresh in (0.5, 0.9):
            original = evalute(SimpleNamespace(roc=True, pd_detT=50, correct_thresh=1e-4))
            fast = sm.FastRoc(50, 1e-4, 260, 346)
            for seed, start, n in ((0, 0, 4000), (1, 37, 4000), (2, 3, 6000)):
                t, x, y, prob, target, label = synthetic_roc_sequence(seed, start, n=n)
                if seed == 2:                                   # 热点像素恰好 256 个虚警
                    hot = np.flatnonzero((x == 100) & (y == 50))
                    keep = hot[:256]
                    drop = hot[256:]
                    x[drop], y[drop] = 5, 5
                    t[keep] = 1001                              # 同一帧
                zeros = np.zeros(t.size, dtype=np.float32)
                ev_locs = torch.from_numpy(np.stack([zeros, x, y, t], 1).astype(np.float32))
                original.roc_update(ev_locs[:, 3], torch.from_numpy(prob.copy()), target,
                                    torch.from_numpy(label), ev_locs, thresh=thresh)
                fast.update(t, prob, target, label, x, y, thresh)
            self.assertEqual((fast.obj_num, fast.correct_num, fast.false_num, fast.frame_num),
                             (original.obj_num, original.correct_num, original.false_num, original.frame_num))
            pd, fa = original.cal_roc()
            self.assertEqual(fast.result(), (pd, fa))


class ReaderTests(unittest.TestCase):
    def write(self, directory, t, n=None):
        n = t.size
        rng = np.random.RandomState(0)
        x, y = rng.randint(0, 320, n), rng.randint(0, 180, n)
        label = (rng.rand(n) < 0.1).astype(np.float64)
        loc = np.stack([x, y, t], 1).astype(np.int64)
        norm = np.column_stack([x / 320.0, y / 192.0, t / 8192.0, rng.randint(0, 2, n), label, label])
        path = os.path.join(directory, "seq.npz")
        np.savez(path, ev_loc=loc, evs_norm=norm.astype(np.float32))
        return path

    def test_auto_length(self):
        with tempfile.TemporaryDirectory() as directory:
            t = np.sort(np.random.RandomState(1).randint(0, 120001, 5000)).astype(np.int64)
            t[-1] = 120000
            path = self.write(directory, t)
            seq = sw.load_npz_events(path, 180, 320, 50, 0, 5)
            self.assertEqual(seq.n_windows, 2401)               # 120000 // 50 + 1
            sw.check_window_partition(seq, 50)
            self.assertEqual(sw.load_npz_events(path, 180, 320, 50, None, 5).n_windows, 2401)
            with self.assertRaises(ValueError):                 # 固定窗数时仍按原规则校验
                sw.load_npz_events(path, 180, 320, 50, 160, 5)
            short = self.write(directory, np.sort(np.random.RandomState(2).randint(0, 8000, 500)).astype(np.int64))
            fixed = sw.load_npz_events(short, 180, 320, 50, 160, 5)
            self.assertEqual(fixed.n_windows, 160)


@unittest.skipUnless(have_eval_dependencies(), "完整评估依赖 cv2/pandas/tqdm")
class VariableLengthEvaluationTests(unittest.TestCase):
    def test_full_evaluation_on_different_lengths(self):
        import train_stream_v1 as T
        from test_stream_layerwise import H, Q99, W, make_net
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            (root / "val").mkdir()
            rng = np.random.RandomState(3)
            for name, windows in (("a.npz", 12), ("b.npz", 20)):
                n = 50 * windows
                t = np.sort(rng.randint(0, 50 * windows, n)).astype(np.int64)
                x, y = rng.randint(0, W, n), rng.randint(0, H, n)
                label = (rng.rand(n) < 0.2).astype(np.float64)
                loc = np.stack([x, y, t], 1)
                np.savez(root / "val" / name, ev_loc=loc,
                         evs_norm=np.column_stack([x / W, y / H, t / 8192.0, rng.randint(0, 2, n), label, label]))
            cfg = dict(root=str(root), height=H, width_px=W, pad_height=H, pad_width=W, window_ms=50, n_windows=0,
                       time_bins=5, input_clip=3.0, threshold=0.5, pd_detT=50, correct_thresh=1e-4,
                       timing_warmup_windows=1, segments=[[0, 3], [8, 11]], rolling_window=4,
                       execution="step", input_device="cpu", state_mode="carry")
            net = make_net()
            result = T.evaluate_split(net, cfg, Q99, torch.device("cpu"), "val", "carry", full=True)
            self.assertEqual(len(result["per_window"]["tp"]), 20)
            self.assertTrue(np.isfinite(result["iou"]) and np.isfinite(result["pd"]) and np.isfinite(result["fa"]))


if __name__ == "__main__":
    unittest.main()
