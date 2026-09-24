"""滑窗基线（tools/sliding_baseline.py）调度部分的单元测试。

模型前向需要 spconv 与 CUDA，只能在服务器上跑；这里守住结论成立的前提——
    1. 每个事件恰好被预测一次（不漏、不重）
    2. 因果：输入里没有 t >= t_end 的事件，也没有早于 t_end - context 的事件
    3. 保留的正是最新 stride 内的事件
    4. context = stride = 整段时长时退化为原来的离线单次前向
"""
import unittest

import numpy as np

from tools.sliding_baseline import check_coverage, window_schedule


def random_times(seed=0, n=5000, total=8000):
    rng = np.random.RandomState(seed)
    return rng.randint(0, total, n)                     # 故意不排序


class WindowScheduleTests(unittest.TestCase):
    def test_every_event_predicted_exactly_once(self):
        t = random_times()
        for context, stride in ((50, 50), (250, 50), (1000, 50), (8000, 50), (1000, 250), (8000, 8000)):
            windows = window_schedule(t, context, stride)
            check_coverage(t.shape[0], windows)          # 不通过会抛 AssertionError

    def test_inputs_are_causal_and_within_context(self):
        t = random_times(seed=1)
        for context, stride in ((250, 50), (2000, 50), (1000, 250)):
            for t_end, ctx, keep in window_schedule(t, context, stride):
                self.assertLess(int(t[ctx].max()), t_end)
                self.assertGreaterEqual(int(t[ctx].min()), max(0, t_end - context))

    def test_kept_events_are_the_newest_stride(self):
        t = random_times(seed=2)
        for t_end, ctx, keep in window_schedule(t, 1000, 50):
            kept = t[ctx[keep]]
            self.assertTrue(np.all((kept >= t_end - 50) & (kept < t_end)))
            rest = np.delete(t[ctx], keep)
            self.assertTrue(np.all(rest < t_end - 50))    # 其余输入都是历史

    def test_context_equal_stride_uses_only_the_current_window(self):
        t = random_times(seed=3)
        for t_end, ctx, keep in window_schedule(t, 50, 50):
            self.assertEqual(len(keep), len(ctx))

    def test_full_length_is_the_offline_single_pass(self):
        """context = stride = 8000：一次前向、输入是全部事件，即原来的离线推理（用来核对与原 test.py 一致）。"""
        t = random_times(seed=4)
        windows = window_schedule(t, 8000, 8000)
        self.assertEqual(len(windows), 1)
        self.assertEqual(sorted(windows[0][1].tolist()), list(range(t.shape[0])))

    def test_empty_windows_are_skipped(self):
        t = np.array([10, 20, 400, 410, 7990])
        windows = window_schedule(t, 200, 50)
        self.assertEqual([w[0] for w in windows], [50, 450, 8000])
        check_coverage(t.shape[0], windows)

    def test_growing_history_with_full_context(self):
        """context = 8000、stride = 50：输入是"从序列开头到现在"的全部事件（因果全历史）。"""
        t = random_times(seed=5)
        for t_end, ctx, keep in window_schedule(t, 8000, 50):
            self.assertEqual(len(ctx), int(np.sum(t < t_end)))

    def test_rejects_context_shorter_than_stride(self):
        with self.assertRaises(ValueError):
            window_schedule(np.array([1, 2, 3]), 20, 50)
        with self.assertRaises(ValueError):
            window_schedule(np.array([-1, 2, 3]), 100, 50)

    def test_coverage_check_catches_duplicates(self):
        t = np.array([1, 2, 3])
        windows = window_schedule(t, 50, 50)
        with self.assertRaises(AssertionError):
            check_coverage(3, windows + windows)


if __name__ == "__main__":
    unittest.main()
