"""V3 阶段 1（等待安全的逐事件发布，model/publish_readout.py + utils/publish_eval.py）的测试。全部在 CPU 上运行。

守住六件事：
    1. 发布规则：上下界、线性/阶梯收拢、期限强制、归属门控逐位精确；任何发布都满足 标签 = 1[z_pub >= θ]；非法参数报错
    2. 离线回放：逐年龄决定、序列提前结束（eos）的处理
    3. 在线发布层与离线回放逐事件相同（标签、年龄、原因、分数），在线证据与 V2 的 TubeReadout 逐位相同
    4. 发布概率在 float32 阈值上与标签一致
    5. 按逐事件发布时刻算的首次检出延迟：同窗同时发布时与旧函数相同；后出生先发布时能提前
    6. run_sequence：打开时其余读出逐位不变，pub 与 V2 融合分数一致；关闭时不多任何字段；命令行进入配置
"""
import math
import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch

from model.evidence_neuron import DriftCUSUM, TubeReadout, velocity_grid
from model.publish_readout import DEADLINE, EOS, LOWER, UPPER, PublishRule, PublishUnits
from utils import publish_eval
from utils.stream_metrics import first_detection_latencies_by_event, first_detection_latencies_published

try:
    from tests import test_stream_v2_evidence as ev
except ImportError:
    import test_stream_v2_evidence as ev

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs",
                      "evisseg_stream_v2.yaml")


class RuleTests(unittest.TestCase):
    def test_margins_and_decisions(self):
        lin = PublishRule(2.0, 1.0, 2.0, 4, "linear")
        step = PublishRule(2.0, 1.0, 2.0, 4, "step")
        self.assertEqual(lin.margins(0), (1.0, 2.0))
        self.assertEqual(lin.margins(2), (0.5, 1.0))
        self.assertEqual(step.margins(3), (1.0, 2.0))
        z = np.array([3.1, 2.9, 0.1, -0.1, 2.0])
        up, down = lin.decide(z, 0)                       # 上界 3.0、下界 0.0
        self.assertEqual(up.tolist(), [True, False, False, False, False])
        self.assertEqual(down.tolist(), [False, False, False, True, False])
        up, down = lin.decide(z, 4)                       # 期限：按 θ 强制
        self.assertEqual(up.tolist(), [True, True, False, False, True])
        self.assertTrue(bool(np.all(up ^ down)))

    def test_gate_is_exact_in_numpy_and_torch(self):
        rule = PublishRule(0.0, 1.0, 1.0, 3, gate=0.5, weight=2.0)
        logit = np.array([1.0, 0.2, 0.2, 0.5, -3.0])
        F = np.array([0.7, 0.7, -0.4, 0.3, 0.0])
        want = np.array([1.0 + 1.4, 0.2 + 0.0, 0.2 - 0.8, 0.5 + 0.6, -3.0])
        self.assertTrue(np.array_equal(rule.score(logit, F), want))
        got = rule.score(torch.tensor(logit), torch.tensor(F))
        self.assertTrue(np.array_equal(got.numpy(), want))
        plain = PublishRule(0.0, 1.0, 1.0, 3, weight=2.0)
        self.assertTrue(np.array_equal(plain.score(logit, F), logit + 2.0 * F))

    def test_invalid_rules(self):
        for kw in (dict(lower=0.0), dict(upper=-0.1), dict(deadline=-1), dict(deadline=1.5), dict(collapse="exp")):
            args = dict(theta=0.0, upper=1.0, lower=1.0, deadline=2)
            args.update(kw)
            with self.assertRaises(ValueError):
                PublishRule(**args)

    def test_replay_cases(self):
        rule = PublishRule(1.0, 1.0, 1.0, 2, "step")      # 年龄 0/1：上界 2、下界 0；年龄 2：按 1 强制
        z = np.array([[2.5, -0.5, 1.0, 1.0, 1.5, 1.2],
                      [9.0, 9.0, 2.2, -0.2, 1.5, 0.3],
                      [9.0, 9.0, 9.0, 9.0, 1.1, 0.3]])
        label, age, reason, z_pub = rule.replay(z, available=np.array([2, 2, 2, 2, 2, 1]))
        self.assertEqual(label.tolist(), [True, False, True, False, True, False])
        self.assertEqual(age.tolist(), [0, 0, 1, 1, 2, 1])
        self.assertEqual(reason.tolist(), [UPPER, LOWER, UPPER, LOWER, DEADLINE, EOS])
        self.assertEqual(z_pub.tolist(), [2.5, -0.5, 2.2, -0.2, 1.1, 0.3])

    def test_label_always_matches_final_threshold(self):
        rng = np.random.RandomState(0)
        for collapse in ("linear", "step"):
            for gate in (None, 0.0):
                rule = PublishRule(0.7, 1.3, 0.9, 5, collapse, gate)
                z = np.cumsum(rng.randn(6, 5000), axis=0)
                label, age, reason, z_pub = rule.replay(z, available=rng.randint(0, 6, 5000))
                self.assertTrue(np.array_equal(label, z_pub >= 0.7))
                self.assertTrue(np.all((age >= 0) & (age <= 5)) and np.all(reason >= 0))


class OnlineTests(unittest.TestCase):
    """在线发布层 vs 离线回放 vs V2 的 TubeReadout，用真实的判决层证据递推。"""

    def setUp(self):
        torch.manual_seed(0)
        self.steps, self.h, self.w = 11, 9, 12
        self.counts = torch.poisson(torch.full((self.steps, 1, 1, self.h, self.w), 0.4, dtype=torch.float64))
        self.mu0 = torch.full_like(self.counts, 0.3)
        self.log_g = torch.randn(self.steps, 1, 1, self.h, self.w, dtype=torch.float64) * 1.5 - 1.0
        self.cusum = DriftCUSUM(velocity_grid([-1.0, 0.0, 1.0]), footprint=3, track_decay=0.6)
        g = torch.Generator().manual_seed(1)
        self.events = []
        for k in range(self.steps):
            n = int(torch.randint(0, 7, (1,), generator=g))
            y = torch.randint(0, self.h, (n,), generator=g)
            x = torch.randint(0, self.w, (n,), generator=g)
            logit = torch.randn(n, generator=g, dtype=torch.float64) * 2.5
            self.events.append((torch.zeros(n, dtype=torch.long), y, x, logit))

    def run_stream(self, rule, delays):
        """同一条证据流同时跑在线发布层与 TubeReadout。返回 (在线记录 {(k,i): (label,age,reason,z)}, 证据 {(k,d): F})。"""
        units, readout = PublishUnits(self.cusum, rule), TubeReadout(self.cusum, delays)
        state = self.cusum.init_state(1, self.h, self.w, "cpu", torch.float64)
        online, F = {}, {}

        def keep(records):
            for key, pos, label, z, published, reason, age in records:
                self.assertEqual(published - key, age if reason != EOS else published - key)
                for p, lab, zz in zip(pos.tolist(), label.tolist(), z.tolist()):
                    self.assertNotIn((key, p), online)
                    online[(key, p)] = (lab, age, reason, zz)

        for k in range(self.steps):
            state, _, _ = self.cusum.step(state, self.counts[k], self.mu0[k], self.log_g[k - 1] if k > 0 else None)
            b, y, x, logit = self.events[k]
            for key, d, scores, _ in readout.step(state, k, b, y, x, k):
                F[(key, d)] = scores
            keep(units.step(state, k, b, y, x, k, logit))
        for key, d, scores, _ in readout.flush():
            F[(key, d)] = scores
        keep(units.flush())
        return online, F

    def test_online_matches_replay_and_tube_readout(self):
        for collapse in ("linear", "step"):
            for gate in (None, 0.5):
                for D in (0, 1, 3):
                    rule = PublishRule(0.8, 0.7, 1.1, D, collapse, gate, weight=1.3)
                    online, F = self.run_stream(rule, list(range(1, max(D, 1) + 1)))
                    n_total = sum(int(e[1].shape[0]) for e in self.events)
                    self.assertEqual(len(online), n_total)
                    for k in range(self.steps):
                        logit = self.events[k][3]
                        n = int(logit.shape[0])
                        if n == 0:
                            continue
                        rows = [logit.numpy()]
                        for d in range(1, D + 1):
                            rows.append(rule.score(logit, F[(k, d)]).numpy())
                        available = np.full(n, min(D, self.steps - 1 - k))
                        label, age, reason, z_pub = rule.replay(np.stack(rows), available)
                        for i in range(n):
                            got = online[(k, i)]
                            self.assertEqual(got[:3], (bool(label[i]), int(age[i]), int(reason[i])), (collapse, gate, D))
                            self.assertEqual(got[3], float(z_pub[i]))

    def test_no_early_publishing_equals_v2_fused_score(self):
        D = 3
        rule = PublishRule(0.2, 1e9, 1e9, D, "step", weight=1.3)
        online, F = self.run_stream(rule, [D])
        for k in range(self.steps):
            logit = self.events[k][3]
            fused = logit + 1.3 * F[(k, D)]                  # V2 的融合分数（期限处或序列末尾截断处）
            for i in range(int(logit.shape[0])):
                lab, age, reason, z = online[(k, i)]
                self.assertEqual(reason, DEADLINE if k + D <= self.steps - 1 else EOS)
                self.assertEqual(z, float(fused[i]))          # 逐位相同
                self.assertEqual(lab, float(fused[i]) >= 0.2)


class ProbabilityAndLatencyTests(unittest.TestCase):
    def test_probability_matches_label_at_float32_threshold(self):
        rng = np.random.RandomState(1)
        theta = 1.7
        z = theta + np.concatenate([rng.randn(2000) * 3, np.array([0.0, 1e-9, -1e-9])])
        label = z >= theta
        for thr in (0.9, 0.5, 0.95):
            p = publish_eval.published_probability(z, label, theta, thr)
            self.assertEqual(p.dtype, np.float32)
            self.assertTrue(np.array_equal(torch.from_numpy(p) >= thr, torch.from_numpy(label)))
            self.assertTrue(np.array_equal(p >= np.float32(thr), label))

    def test_by_event_matches_published_when_windows_publish_together(self):
        rng = np.random.RandomState(2)
        n, windows = 400, 20
        t = np.sort(rng.randint(0, 50 * windows, n))
        target_id = rng.randint(0, 4, n).astype(np.float64)
        label = (target_id > 0).astype(np.float32)
        prob = rng.rand(n).astype(np.float32)
        publish_window = np.minimum(np.arange(windows) + 2, windows - 1)
        per_event = publish_window[t // 50]
        for ct in (1e-4, 0.3):
            a = first_detection_latencies_published(t, label, target_id, prob, 50, 0.5, ct, publish_window)
            b = first_detection_latencies_by_event(t, label, target_id, prob, 50, 0.5, ct, per_event)
            self.assertEqual(a, b)

    def test_later_event_published_earlier_detects_earlier(self):
        t = np.array([10, 60, 70])                         # 目标事件：第 0 窗一个、第 1 窗两个
        label = np.ones(3, np.float32)
        target_id = np.ones(3)
        prob = np.array([0.95, 0.95, 0.2], np.float32)
        publish = np.array([5, 1, 1])                       # 第 0 窗的事件等到第 5 窗才发布，第 1 窗的当窗发布
        r = first_detection_latencies_by_event(t, label, target_id, prob, 50, 0.9, 1e-4, publish)[0]
        self.assertEqual((r["detect_window"], r["publish_window"], r["latency_ms"]), (1, 1, 90.0))
        r = first_detection_latencies_by_event(t, label, target_id, prob, 50, 0.9, 0.6, publish)[0]
        self.assertEqual((r["detect_window"], r["publish_window"]), (0, 5))   # 第 1 窗只有 1/2 被判为目标，不足 60%


class ReplayToolTests(unittest.TestCase):
    """tools/publish_replay.py 的指标与原 utils/eval.py 逐位相同；回放的固定延迟 0 就是 net。"""

    def test_metrics_match_original_evaluator(self):
        import shutil
        import tempfile
        from types import SimpleNamespace
        from tools import publish_replay as pr
        from utils.eval import evalute
        rng = np.random.RandomState(3)
        tmp = tempfile.mkdtemp()
        try:
            ref = evalute(SimpleNamespace(roc=True, pd_detT=50, correct_thresh=1e-4))
            for s in range(3):
                n = 3000
                t = np.sort(rng.randint(0, 50 * 12, n))
                tid = np.where(rng.rand(n) < 0.15, rng.randint(1, 3, n), 0).astype(np.float64)
                x = np.where(tid > 0, 100 + rng.randint(-3, 4, n), rng.randint(0, 346, n))
                y = np.where(tid > 0, 80 + rng.randint(-3, 4, n), rng.randint(0, 260, n))
                label = (tid > 0).astype(np.float32)
                logit = (3.0 * label - 1.5 + 1.5 * rng.randn(n)).astype(np.float32)
                prob = (1.0 / (1.0 + np.exp(-logit.astype(np.float64)))).astype(np.float32)
                fields = {"evidence_d%d" % d: rng.randn(n).astype(np.float32) for d in (1, 2)}
                np.savez(os.path.join(tmp, "s%d.npz" % s), locs=np.stack([np.zeros(n, np.int64), x, y, t], 1),
                         labels=label, probabilities=prob, target_id=tid, logit_net=logit, **fields)
                locs = torch.from_numpy(np.stack([np.zeros(n), x, y, t], 1).astype(np.float32))
                ref.roc_update(locs[:, 3], torch.from_numpy(prob.copy()), tid, torch.from_numpy(label), locs, thresh=0.9)
                ref.matches[str(s)] = {"seg_pred": torch.from_numpy(prob), "seg_gt": torch.from_numpy(label)}
            args = SimpleNamespace(deadline=2, window_ms=50, n_windows=12, pd_detT=50, correct_thresh=1e-4, max_seqs=0,
                                   weight=1.0)
            es = pr.EventSet(tmp, args)
            pred = es.prob >= np.float32(0.9)
            m = es.metrics(pred, np.zeros(es.label.size), args)
            pd, fa = ref.cal_roc()
            iou, acc = ref.evaluate_iou_and_accuracy(thresh=0.9)
            self.assertEqual((m["pd"], m["fa"]), (pd, fa))
            self.assertAlmostEqual(m["iou"], iou, places=12)
            self.assertAlmostEqual(m["acc"], acc, places=12)
            fixed0 = pr.fixed_policy(es, 0, 1.0)(math.log(9.0))[0]
            self.assertTrue(np.array_equal(fixed0, es.logit >= math.log(9.0)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class RunSequenceTests(unittest.TestCase):
    def setUp(self):
        self.seq = ev.synthetic_sequence(seed=5)
        self.frontend = ev.make_frontend()
        self.model = ev.make_model(self.frontend).eval()
        self.cusum = DriftCUSUM(velocity_grid([-1.0, 0.0, 1.0]), footprint=3, track_decay=0.8)

    def run_seq(self, **kw):
        from train_stream_v2 import run_sequence
        cfg = dict(ev.CFG, **kw)
        with torch.no_grad():
            return run_sequence(self.model, self.frontend, self.cusum, self.seq, cfg, torch.device("cpu"), "carry")

    def test_off_adds_nothing_and_on_keeps_other_readouts(self):
        p0, x0, c0 = self.run_seq()
        p1, x1, c1 = self.run_seq(publish=True, publish_deadline=3, publish_upper=1.0, publish_lower=2.0,
                                  publish_collapse="linear", publish_gate=None)
        self.assertEqual(sorted(set(p1) - set(p0)), ["pub"])
        self.assertEqual(sorted(set(x1) - set(x0)), ["age_pub", "label_pub", "publish_pub", "reason_pub", "z_pub"])
        for name in p0:
            self.assertTrue(np.array_equal(p0[name], p1[name]), name)
        for name in x0:
            self.assertTrue(np.array_equal(x0[name], x1[name]), name)
        self.assertTrue(np.array_equal(c0, c1))
        thr = float(ev.CFG["threshold"])
        self.assertTrue(np.array_equal(p1["pub"] >= np.float32(thr), x1["label_pub"] > 0.5))
        self.assertTrue(np.all((x1["age_pub"] >= 0) & (x1["age_pub"] <= 3)))
        k = self.seq.t // 50
        self.assertTrue(np.array_equal(x1["publish_pub"], k + x1["age_pub"].astype(np.int64)))

    def test_no_early_publishing_matches_fused_readout(self):
        _, x, _ = self.run_seq(publish=True, publish_deadline=3, publish_upper=1e9, publish_lower=1e9,
                               publish_collapse="step", publish_gate=None)
        fused = x["logit_net"].astype(np.float64) + x["evidence_d3"].astype(np.float64)
        self.assertLess(float(np.abs(x["z_pub"] - fused).max()), 1e-5)
        k = self.seq.t // 50
        self.assertTrue(np.array_equal(x["publish_pub"], np.minimum(k + 3, ev.WINDOWS - 1)))


class CommandLineTests(unittest.TestCase):
    def parse(self, *extra):
        import train_stream_v2 as tv2
        argv = ["train_stream_v2.py", "--config", CONFIG, "--mode", "eval"] + list(extra)
        with mock.patch.object(sys, "argv", argv):
            return tv2.build_config(tv2.parse_args())

    def test_defaults_off_and_flags_reach_config(self):
        cfg = self.parse()
        self.assertFalse(publish_eval.enabled(cfg))
        self.assertEqual(publish_eval.readout_names(cfg), [])
        cfg = self.parse("--publish", "on", "--publish-deadline", "4", "--publish-theta", "1.5", "--publish-upper", "0.5",
                         "--publish-lower", "3", "--publish-collapse", "step", "--publish-gate", "0.2")
        rule = publish_eval.build_rule(cfg)
        self.assertEqual((rule.deadline, rule.theta, rule.upper, rule.lower, rule.collapse, rule.gate),
                         (4, 1.5, 0.5, 3.0, "step", 0.2))
        self.assertEqual(publish_eval.readout_names(cfg), ["pub"])
        self.assertAlmostEqual(publish_eval.theta_of(self.parse("--publish", "on")), math.log(9.0), places=12)
        self.assertIsNone(publish_eval.build_rule(self.parse("--publish", "on", "--publish-gate", "none")).gate)
        self.assertIsNone(publish_eval.describe(self.parse()))
        self.assertEqual(publish_eval.describe(cfg), {"deadline": 4, "theta": 1.5, "upper": 0.5, "lower": 3.0,
                                                      "collapse": "step", "gate": 0.2, "weight": 1.0})
        bare = {"publish": True, "threshold": 0.9}                 # 配置里没写的项：记录的是实际生效的缺省值
        self.assertEqual(publish_eval.describe(bare)["deadline"], 5)
        self.assertAlmostEqual(publish_eval.describe(bare)["theta"], math.log(9.0), places=12)

    def test_invalid_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            self.parse("--publish", "on", "--publish-lower", "0")
        with self.assertRaises(ValueError):
            self.parse("--publish", "on", "--publish-deadline", "-1")


if __name__ == "__main__":
    unittest.main()
