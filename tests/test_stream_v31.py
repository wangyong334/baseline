"""V3.1 对称出生判定（model/publish_readout 式 (5)）目前只在离线回放里：PublishRule.replay 的 birth_allow、
run_sequence 导出的 existence_birth、tools/publish_replay.py 的 --birth-lambda。全部在 CPU 上运行。

守住四件事：
    1 birth_allow 全为 True 时与 V3 原规则逐事件相同；给定的出生判定按式 (5) 生效，且只作用于年龄 0
    2 手算例子：出生即判被挡下的事件在下一年龄按收拢后的边界发布
    3 run_sequence：只有 export=True 时才导出 existence_birth（>= 0、长度 = 事件数），各读出逐位不变
    4 publish_replay：EventSet 读到 existence_birth；λ = None 时与原规则相同，给 λ 时等于手工调用 replay(birth_allow)；
      导出里没有该字段却给了 λ 时报错
"""
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from model.evidence_neuron import DriftCUSUM, velocity_grid
from model.publish_readout import DEADLINE, LOWER, UPPER, PublishRule

try:
    from tests import test_stream_v2_evidence as ev
except ImportError:
    import test_stream_v2_evidence as ev


class BirthRuleTests(unittest.TestCase):
    def test_all_allowed_equals_v3(self):
        rng = np.random.RandomState(0)
        n, D = 500, 4
        z = rng.randn(D + 1, n) * 3.0
        avail = rng.randint(0, D + 1, n)
        rule = PublishRule(0.3, 0.8, 1.5, D, "linear")
        ones = np.ones(n, bool)
        a = rule.replay(z, avail)
        b = rule.replay(z, avail, (ones, ones))
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y))

    def test_hand_example(self):
        rule = PublishRule(0.0, 1.0, 1.0, 3, "linear")          # 年龄 1 的边界收拢到 ±2/3
        z = np.array([[2.0, -2.0, 2.0, -2.0],                   # 年龄 0
                      [2.0, -2.0, 2.0, -2.0],                   # 年龄 1
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]])
        allow_up = np.array([False, True, True, True])
        allow_down = np.array([True, False, True, True])
        label, age, reason, _ = rule.replay(z, None, (allow_up, allow_down))
        self.assertEqual(label.tolist(), [True, False, True, False])
        self.assertEqual(age.tolist(), [1, 1, 0, 0])
        self.assertEqual(reason.tolist(), [UPPER, LOWER, UPPER, LOWER])

    def test_birth_gate_never_touches_forced_or_later_ages(self):
        rule = PublishRule(0.0, 1.0, 1.0, 0, "linear")          # D = 0：年龄 0 就是期限，强制发布不受出生判定影响
        z = np.array([[2.0, -2.0]])
        none = np.zeros(2, bool)
        label, age, reason, _ = rule.replay(z, None, (none, none))
        self.assertEqual((label.tolist(), age.tolist(), reason.tolist()), ([True, False], [0, 0], [DEADLINE, DEADLINE]))


class RunSequenceExportTests(unittest.TestCase):
    def test_existence_only_when_exporting(self):
        from train_stream_v2 import run_sequence
        seq = ev.synthetic_sequence(seed=5)
        frontend = ev.make_frontend()
        model = ev.make_model(frontend).eval()
        cusum = DriftCUSUM(velocity_grid([-1.0, 0.0, 1.0]), footprint=3, track_decay=0.8)
        with torch.no_grad():
            p0, x0, _ = run_sequence(model, frontend, cusum, seq, dict(ev.CFG), torch.device("cpu"), "carry")
            p1, x1, _ = run_sequence(model, frontend, cusum, seq, dict(ev.CFG), torch.device("cpu"), "carry", export=True)
        self.assertNotIn("existence_birth", x0)
        self.assertIn("existence_birth", x1)
        e = x1["existence_birth"]
        self.assertEqual(e.shape, (seq.n_events,))
        self.assertTrue(np.all(np.isfinite(e)) and np.all(e >= 0.0))
        self.assertGreater(float(e.max()), 0.0)
        for name in p0:
            self.assertTrue(np.array_equal(p0[name], p1[name]), name)
        for name in x0:
            self.assertTrue(np.array_equal(x0[name], x1[name]), name)


class ReplayToolTests(unittest.TestCase):
    def make_dump(self, with_existence):
        rng = np.random.RandomState(7)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        for s in range(2):
            n = 800
            t = np.sort(rng.randint(0, 50 * 12, n))
            tid = np.where(rng.rand(n) < 0.2, 1, 0).astype(np.float64)
            x = np.where(tid > 0, 100 + rng.randint(-3, 4, n), rng.randint(0, 346, n))
            y = np.where(tid > 0, 80 + rng.randint(-3, 4, n), rng.randint(0, 260, n))
            label = (tid > 0).astype(np.float32)
            logit = (2.0 * label - 1.0 + 1.5 * rng.randn(n)).astype(np.float32)
            prob = (1.0 / (1.0 + np.exp(-logit.astype(np.float64)))).astype(np.float32)
            fields = {"evidence_d%d" % d: (label * 1.5 - 0.3 + rng.randn(n)).astype(np.float32) for d in (1, 2)}
            if with_existence:
                fields["existence_birth"] = np.abs(3.0 * label + rng.randn(n)).astype(np.float32)
            np.savez(os.path.join(tmp, "s%d.npz" % s), locs=np.stack([np.zeros(n, np.int64), x, y, t], 1),
                     labels=label, probabilities=prob, target_id=tid, logit_net=logit, **fields)
        return tmp

    def args(self):
        return SimpleNamespace(deadline=2, window_ms=50, n_windows=12, pd_detT=50, correct_thresh=1e-4, max_seqs=0,
                               weight=1.0, evidence=["plain"])

    def test_birth_lambda_in_replay_tool(self):
        from tools import publish_replay as pr
        es = pr.EventSet(self.make_dump(True), self.args())
        self.assertIsNotNone(es.exist)
        self.assertEqual(es.exist.shape, es.label.shape)
        theta = 0.4
        rule = PublishRule(theta, 0.5, 1.0, 2, "linear", None, 1.0)
        z = pr.z_by_age(es, rule, "plain")
        v3 = pr.publish_policy(es, ("plain", 2, "linear", 0.5, 1.0, None, None), 1.0)(theta)
        want_v3 = rule.replay(z, es.avail)
        self.assertTrue(np.array_equal(v3[0], want_v3[0]) and np.array_equal(v3[1], want_v3[1]))
        lam = 1.5
        v31 = pr.publish_policy(es, ("plain", 2, "linear", 0.5, 1.0, None, lam), 1.0)(theta)
        present = es.exist >= lam
        want = rule.replay(z, es.avail, (present, ~present))
        self.assertTrue(np.array_equal(v31[0], want[0]) and np.array_equal(v31[1], want[1]))
        self.assertFalse(np.array_equal(v31[1], v3[1]))           # 出生判定确实改变了部分事件的发布年龄
        self.assertEqual(pr.spec_key(("plain", 2, "linear", 0.5, 1.0, None, None)), "plain_D2_linear_a0.5_b1_gNone")
        self.assertEqual(pr.spec_key(("plain", 2, "linear", 0.5, 1.0, None, lam)), "plain_D2_linear_a0.5_b1_gNone_e1.5")

    def test_missing_existence_field_is_an_error(self):
        from tools import publish_replay as pr
        es = pr.EventSet(self.make_dump(False), self.args())
        self.assertIsNone(es.exist)
        with self.assertRaises(SystemExit):
            pr.publish_policy(es, ("plain", 2, "linear", 0.5, 1.0, None, 1.0), 1.0)


if __name__ == "__main__":
    unittest.main()
