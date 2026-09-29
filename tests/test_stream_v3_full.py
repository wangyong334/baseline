"""V3（等待安全的逐事件发布）中发布层之外部分的测试：事件证据缓存（锚定证据链）与 V3 配置。全部在 CPU 上运行。

V3 只在推理期起作用（09-28 深度分析后撤掉了归属训练：它只是给已有标签重新加权，不带新信息，同虚警率下只会打平或变差），
所以训练与 V2 逐位相同。守住四件事：
    1. 锚定证据链：链连着时证据全额计入、断开后只计负证据；TubeReadout(anchor=True) 与暴力逐步计算一致；
       anchor=False 时 TubeReadout 与原实现一致（其余测试守着）；"目标后来路过"时 V2 证据加分而锚定证据不加
    2. 在线发布层打开锚定时与离线回放逐事件相同（回放用锚定延迟读出的 F(d)）
    3. 训练：打开 V3 的评估时开关，参数更新与 V2 逐位相同
    4. 配置：命令行进入配置；V3 配置文件与 V2 只差标了 [V3] 的三项评估时开关
"""
import math
import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch

from model.evidence_neuron import DriftCUSUM, TubeReadout, anchored_accumulate, velocity_grid
from model.publish_readout import PublishRule, PublishUnits

try:
    from tests import test_stream_v2_evidence as ev
except ImportError:
    import test_stream_v2_evidence as ev

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_V2 = os.path.join(ROOT, "configs", "evisseg_stream_v2.yaml")
CONFIG_V3 = os.path.join(ROOT, "configs", "evisseg_stream_v3.yaml")


class AnchoredChainTests(unittest.TestCase):
    def test_single_step_rule(self):
        run = torch.zeros(2, 3, dtype=torch.float64)
        alive = torch.tensor([[True, True, False], [True, False, False]])
        values = torch.tensor([[1.5, -0.5, 2.0], [0.7, 3.0, -1.0]], dtype=torch.float64)
        support = torch.tensor([[True, False, True], [False, True, True]])
        run, alive = anchored_accumulate(run, alive, values, support)
        self.assertEqual(run.tolist(), [[1.5, -0.5, 0.0], [0.7, 0.0, -1.0]])
        self.assertEqual(alive.tolist(), [[True, False, False], [False, False, False]])

    def fields(self, steps, h, w, seed=0):
        g = torch.Generator().manual_seed(seed)
        ell = torch.randn(steps, 1, 3, h, w, generator=g, dtype=torch.float64)
        support = (torch.rand(steps, 1, 1, h, w, generator=g, dtype=torch.float64) < 0.6).to(torch.float64)
        return ell, support

    def test_anchored_readout_matches_brute_force(self):
        steps, h, w = 8, 7, 9
        cusum = DriftCUSUM([(0.0, 0.0), (0.0, 1.0), (1.0, -1.0)], footprint=1)
        ell, support = self.fields(steps, h, w)
        y = torch.tensor([0, 3, 6, 2], dtype=torch.long)
        x = torch.tensor([0, 4, 8, 1], dtype=torch.long)
        b = torch.zeros(4, dtype=torch.long)
        readout = TubeReadout(cusum, [1, 3], anchor=True)
        got = {}
        for k in range(steps):
            for key, d, scores, _ in readout.step({"ell": ell[k]}, k, b, y, x, k, support[k]):
                got[(key, d)] = scores
        for key, d, scores, _ in readout.flush():
            got[(key, d)] = scores
        last = steps - 1
        for (k, d), scores in got.items():
            for i in range(4):
                runs = []
                for v, vel in enumerate(cusum.velocities):
                    run, alive, start = 0.0, True, ev.path_offset(vel, k)
                    for m in range(k + 1, min(k + d, last) + 1):
                        now = ev.path_offset(vel, m)
                        py, px = int(y[i]) + now[0] - start[0], int(x[i]) + now[1] - start[1]
                        inside = 0 <= py < h and 0 <= px < w
                        val = float(ell[m][0, v, py, px]) if inside else 0.0
                        sup = inside and float(support[m][0, 0, py, px]) > 0
                        run += val if alive else min(val, 0.0)
                        alive = alive and sup
                    runs.append(run)
                want = math.log(sum(math.exp(r) for r in runs) / len(runs))
                self.assertAlmostEqual(float(scores[i]), want, places=10)

    def test_target_passing_later_is_not_credited(self):
        """背景事件出生后原地没有事件（链断开），第 3 窗目标路过带来很强的正证据：V2 证据加分，锚定证据不加。"""
        h, w, steps = 5, 5, 5
        cusum = DriftCUSUM([(0.0, 0.0)], footprint=1)
        ell = torch.zeros(steps, 1, 1, h, w, dtype=torch.float64)
        support = torch.zeros(steps, 1, 1, h, w, dtype=torch.float64)
        ell[1, 0, 0, 2, 2], ell[2, 0, 0, 2, 2] = -0.1, -0.1          # 事件出生后原地空着：小的负证据
        ell[3, 0, 0, 2, 2], support[3, 0, 0, 2, 2] = 4.0, 1.0        # 第 3 窗目标路过
        y, x, b = torch.tensor([2]), torch.tensor([2]), torch.zeros(1, dtype=torch.long)
        results = {}
        for anchor in (False, True):
            readout = TubeReadout(cusum, [4], anchor=anchor)
            for k in range(steps):
                n = 1 if k == 0 else 0
                for key, d, scores, _ in readout.step({"ell": ell[k]}, k, b[:n], y[:n], x[:n], k,
                                                      support[k] if anchor else None):
                    results[anchor] = float(scores[0])
        self.assertAlmostEqual(results[False], 3.8, places=10)
        self.assertAlmostEqual(results[True], -0.2, places=10)

    def test_publish_units_with_anchor_match_replay(self):
        steps, h, w = 10, 8, 10
        cusum = DriftCUSUM(velocity_grid([-1.0, 0.0, 1.0]), footprint=3, track_decay=0.6)
        counts = torch.poisson(torch.full((steps, 1, 1, h, w), 0.5, dtype=torch.float64))
        mu0 = torch.full_like(counts, 0.3)
        log_g = torch.randn(steps, 1, 1, h, w, dtype=torch.float64) * 1.5 - 1.0
        g = torch.Generator().manual_seed(3)
        events = []
        for k in range(steps):
            n = int(torch.randint(0, 6, (1,), generator=g))
            events.append((torch.zeros(n, dtype=torch.long), torch.randint(0, h, (n,), generator=g),
                           torch.randint(0, w, (n,), generator=g), torch.randn(n, generator=g, dtype=torch.float64) * 2))
        for gate in (None, 0.3):
            rule = PublishRule(0.5, 0.8, 1.2, 3, "linear", gate, weight=1.2)
            units, readout = PublishUnits(cusum, rule, anchor=True), TubeReadout(cusum, [1, 2, 3], anchor=True)
            state = cusum.init_state(1, h, w, "cpu", torch.float64)
            online, Fd = {}, {}
            for k in range(steps):
                state, _, _ = cusum.step(state, counts[k], mu0[k], log_g[k - 1] if k > 0 else None)
                support = cusum.support_field(counts[k])
                b, y, x, logit = events[k]
                for key, d, scores, _ in readout.step(state, k, b, y, x, k, support):
                    Fd[(key, d)] = scores
                for key, pos, lab, z, _, reason, age in units.step(state, k, b, y, x, k, logit, support):
                    for p, l_, zz in zip(pos.tolist(), lab.tolist(), z.tolist()):
                        online[(key, p)] = (l_, age, reason, zz)
            for key, d, scores, _ in readout.flush():
                Fd[(key, d)] = scores
            for key, pos, lab, z, _, reason, age in units.flush():
                for p, l_, zz in zip(pos.tolist(), lab.tolist(), z.tolist()):
                    online[(key, p)] = (l_, age, reason, zz)
            for k in range(steps):
                logit = events[k][3]
                n = int(logit.shape[0])
                if not n:
                    continue
                rows = [logit.numpy()] + [rule.score(logit, Fd[(k, d)]).numpy() for d in (1, 2, 3)]
                label, age, reason, z_pub = rule.replay(np.stack(rows), np.full(n, min(3, steps - 1 - k)))
                for i in range(n):
                    self.assertEqual(online[(k, i)], (bool(label[i]), int(age[i]), int(reason[i]), float(z_pub[i])))

    def test_support_field(self):
        cusum = DriftCUSUM([(0.0, 0.0)], footprint=3)
        counts = torch.zeros(1, 1, 5, 6, dtype=torch.float64)
        counts[0, 0, 2, 2] = 2.0
        s = cusum.support_field(counts)
        self.assertEqual(int(s.sum()), 9)
        self.assertEqual(float(s[0, 0, 1, 1]), 1.0)
        self.assertEqual(float(s[0, 0, 0, 0]), 0.0)


class TrainingTests(unittest.TestCase):
    def run_once(self, **kw):
        from train_stream_v2 import train_sequence
        seq = ev.synthetic_sequence(seed=1)
        frontend = ev.make_frontend()
        model = ev.make_model(frontend)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        out = train_sequence(model, frontend, seq, optimizer, dict(ev.CFG, **kw), torch.device("cpu"),
                             np.random.RandomState(0))
        return out, [p.detach().clone() for p in model.parameters()]

    def test_v3_switches_do_not_touch_training(self):
        base, p0 = self.run_once()
        v3, p1 = self.run_once(publish=True, publish_anchor=True, publish_deadline=5)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(p0, p1)))
        self.assertEqual(base["loss_sum"], v3["loss_sum"])
        self.assertEqual(set(v3), {"loss_sum", "mark_sum", "intensity_sum", "events", "grad_norms", "missing_grads"})
        self.assertEqual(v3["missing_grads"], [])
        self.assertTrue(math.isfinite(v3["loss_sum"]))


class ConfigTests(unittest.TestCase):
    def parse(self, config, *extra):
        import train_stream_v2 as tv2
        argv = ["train_stream_v2.py", "--config", config, "--mode", "eval"] + list(extra)
        with mock.patch.object(sys, "argv", argv):
            return tv2.build_config(tv2.parse_args())

    def test_flags_and_v3_config(self):
        from utils import publish_eval
        cfg = self.parse(CONFIG_V2, "--publish", "on", "--publish-anchor", "on")
        self.assertTrue(publish_eval.describe(cfg)["anchor"])
        v2, v3 = self.parse(CONFIG_V2), self.parse(CONFIG_V3)
        diff = sorted(k for k in set(v2) | set(v3) if v2.get(k) != v3.get(k))
        self.assertEqual(diff, ["publish", "publish_anchor", "readout_delays"])
        self.assertTrue(v3["publish"] and v3["publish_anchor"])
        self.assertFalse(v2["publish"] or v2["publish_anchor"])
        self.assertFalse(any(k.startswith("loss_mark_boundary") for k in list(v2) + list(v3)))


class RunSequenceAnchorTests(unittest.TestCase):
    def test_anchor_fields_and_no_early_publish(self):
        from train_stream_v2 import run_sequence
        seq = ev.synthetic_sequence(seed=5)
        frontend = ev.make_frontend()
        model = ev.make_model(frontend).eval()
        cusum = DriftCUSUM(velocity_grid([-1.0, 0.0, 1.0]), footprint=3, track_decay=0.8)
        cfg = dict(ev.CFG, publish=True, publish_anchor=True, publish_deadline=3, publish_upper=1e9, publish_lower=1e9,
                   publish_collapse="step", publish_gate=None)
        with torch.no_grad():
            p0, x0, _ = run_sequence(model, frontend, cusum, seq, dict(ev.CFG), torch.device("cpu"), "carry")
            p1, x1, _ = run_sequence(model, frontend, cusum, seq, cfg, torch.device("cpu"), "carry", export=True)
            p2, x2, _ = run_sequence(model, frontend, cusum, seq, cfg, torch.device("cpu"), "carry")
        for name in p0:
            self.assertTrue(np.array_equal(p0[name], p1[name]), name)
        for d in (1, 2, 3):
            self.assertIn("evidence_anchor_d%d" % d, x1)
        fused = x1["logit_net"].astype(np.float64) + x1["evidence_anchor_d3"].astype(np.float64)
        self.assertLess(float(np.abs(x1["z_pub"] - fused).max()), 1e-5)
        # 锚定证据不会比 V2 证据更正：断开的链只少计正证据
        self.assertTrue(np.all(x1["evidence_anchor_d3"] <= x1["evidence_d3"] + 1e-5))
        # 不导出时不算锚定的逐延迟读出，发布结果逐位不变
        self.assertFalse(any(name.startswith("evidence_anchor_d") for name in x2))
        self.assertEqual(set(p1), set(p2))
        for name in p1:
            self.assertTrue(np.array_equal(p1[name], p2[name]), name)
        for name in ("z_pub", "label_pub", "age_pub", "reason_pub", "publish_pub"):
            self.assertTrue(np.array_equal(x1[name], x2[name]), name)


if __name__ == "__main__":
    unittest.main()
