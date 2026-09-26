"""V2-2 目标假设库与回溯分布修正读出的单元测试（V2 定稿设计 v1.0 第 7-9 节）。全部在 CPU 上运行，不需要数据集。

守住四件事：
    1. 几何：像素内积分的高斯对整数平移求和为 1；式 (6) 的分布归一、峰在中心；模糊只改变形状不改变质量；
       加权直线拟合在无噪声直线上精确，外推方差随距离增大
    2. 假设库：高置信的运动目标能生成、确认并被跟住（位置误差 < 1 像素、结构紧凑）；目标消失后结束；
       重复假设被合并；运动管道入口按速度生成候选；三种更新变体都能运行
    3. 读出的命题（设计页第 9 节）：P1 无新信息不修正（严格为 0）；P2 复制假设不改变修正；
       P3 有界与回退；P4 快照冻结；P5 锐化方向
    4. 机制：目标确认之前的事件（固定参考）在轨迹上的得到正修正，偏开的背景不加分；对照读出的取值范围
"""
import math
import unittest

import numpy as np
import torch

from model.attribution_readout import VARIANTS, AttributionReadout, rho_reference
from model.target_hypotheses import (F64, HypothesisBank, distribution, gaussian_template, line_fit,
                                     pixel_mass)

H, W = 48, 64


def activity_map(center, mass, spread=1.5):
    """SNN 活动估计的替身：以 center 为中心、总质量 mass 的高斯斑。"""
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2.0 * spread ** 2))
    return torch.from_numpy(g / g.sum() * mass)


def make_window(rng, center=None, n_target=10, p_target=0.95, n_bg=20, p_bg=0.05, extra=None, spread=1.0):
    """一窗的事件：目标斑（可无）+ 均匀背景 + 额外事件 [(y, x, p), ...]。返回 (ys, xs, probs, labels, g, mu0)。"""
    ys, xs, ps, labels = [], [], [], []
    if center is not None:
        pts = np.round(np.asarray(center) + rng.normal(0.0, spread, size=(n_target, 2))).astype(np.int64)
        for y, x in pts:
            ys.append(int(np.clip(y, 0, H - 1)))
            xs.append(int(np.clip(x, 0, W - 1)))
            ps.append(p_target)
            labels.append(1)
    for _ in range(n_bg):
        ys.append(int(rng.integers(0, H)))
        xs.append(int(rng.integers(0, W)))
        ps.append(p_bg)
        labels.append(0)
    for y, x, p in extra or []:
        ys.append(int(y))
        xs.append(int(x))
        ps.append(float(p))
        labels.append(0)
    g = activity_map(center, float(n_target)) if center is not None else torch.zeros(H, W, dtype=F64)
    mu0 = torch.full((H, W), 0.01, dtype=F64)
    return (torch.tensor(ys), torch.tensor(xs), torch.tensor(ps, dtype=F64), torch.tensor(labels), g, mu0)


def track(k, start=(20.0, 12.0), velocity=(0.6, 1.0)):
    return (start[0] + velocity[0] * k, start[1] + velocity[1] * k)


def run(bank, windows, readout=None, keys=None):
    """依次处理若干窗，返回读出结果 {(key, d): res}。windows 为 make_window 的输出列表。"""
    out = {}
    for k, (ys, xs, ps, _, g, mu0) in enumerate(windows):
        bank.step(k, ys, xs, ps, g, mu0)
        if readout is not None:
            for key, d, res, published in readout.step(k, k if keys is None else keys[k], ys, xs, mu0, g):
                out[(key, d)] = (res, published)
    return out


class Geometry(unittest.TestCase):
    def test_pixel_mass_sums_to_one(self):
        t = torch.arange(-30, 31, dtype=F64)
        for sigma in (0.5, 1.0, 3.0):
            for shift in (0.0, 0.3, -0.45):
                self.assertAlmostEqual(float(pixel_mass(t - shift, sigma).sum()), 1.0, places=9)

    def test_distribution_is_normalized_and_peaks_at_center(self):
        S = gaussian_template(7, 2.0)
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W))
        ys, xs = yy.reshape(-1), xx.reshape(-1)
        center = torch.tensor([20.3, 31.6], dtype=F64)
        for sigma in (0.5, 2.0):
            q = distribution(S, center, sigma, ys, xs)
            self.assertAlmostEqual(float(q.sum()), 1.0, places=6)
            i = int(torch.argmax(q))
            self.assertEqual((int(ys[i]), int(xs[i])), (20, 32))

    def test_sharpening_raises_center_and_lowers_tail(self):
        """P5：结构单峰、中心不变时，不确定度变小让模态附近的值上升、远处下降。"""
        S = gaussian_template(7, 1.5)
        center = torch.tensor([20.0, 30.0], dtype=F64)
        ys, xs = torch.tensor([20, 20]), torch.tensor([30, 37])
        broad, sharp = distribution(S, center, 3.0, ys, xs), distribution(S, center, 0.8, ys, xs)
        self.assertGreater(float(sharp[0]), float(broad[0]))
        self.assertLess(float(sharp[1]), float(broad[1]))

    def test_line_fit_exact_and_extrapolation_variance_grows(self):
        js = torch.tensor([3.0, 4.0, 5.0, 6.0], dtype=F64)
        zs = torch.stack([torch.tensor([10 + 0.5 * j, 5 - 1.2 * j], dtype=F64) for j in js.tolist()])
        ws = torch.ones(4, dtype=F64)
        c, var_mid, v = line_fit(js, zs, ws, 4.5)
        self.assertTrue(torch.allclose(c, torch.tensor([12.25, -0.4], dtype=F64)))
        self.assertTrue(torch.allclose(v, torch.tensor([0.5, -1.2], dtype=F64)))
        _, var_far, _ = line_fit(js, zs, ws, 0.0)
        self.assertGreater(var_far, var_mid)
        self.assertIsNone(line_fit(js[:1], zs[:1], ws[:1], 0.0))


class BankTracking(unittest.TestCase):
    def windows(self, n, seed=0, **kw):
        rng = np.random.default_rng(seed)
        return [make_window(rng, track(k), **kw) for k in range(n)]

    def test_confirms_and_tracks_moving_target(self):
        bank = HypothesisBank(H, W)
        run(bank, self.windows(14))
        conf = bank.confirmed_hypotheses()
        self.assertEqual(len(conf), 1)
        h = conf[0]
        center, sigma = bank.estimate(h, 13, 13)
        err = math.hypot(float(center[0]) - track(13)[0], float(center[1]) - track(13)[1])
        self.assertLess(err, 1.0)
        self.assertLess(sigma, 1.5)
        half = (h.S.shape[0] - 1) // 2
        u = torch.arange(-half, half + 1, dtype=F64)
        near = (u[:, None] ** 2 + u[None, :] ** 2) <= 16
        self.assertGreater(float(h.S[near].sum()), 0.8)          # 结构紧凑，没有被背景撑大
        self.assertEqual(bank.summary()["births_snn"], 1)

    def test_hypothesis_ends_after_target_disappears(self):
        bank = HypothesisBank(H, W)
        rng = np.random.default_rng(1)
        wins = [make_window(rng, track(k)) for k in range(8)] + [make_window(rng, None) for _ in range(5)]
        run(bank, wins)
        self.assertEqual(len(bank.alive), 0)
        self.assertGreaterEqual(bank.summary()["ended"], 1)

    def test_duplicates_are_merged(self):
        bank = HypothesisBank(H, W)
        rng = np.random.default_rng(2)
        wins = [make_window(rng, track(k)) for k in range(10)]
        run(bank, wins[:8])
        bank.clone_hypothesis(bank.confirmed_hypotheses()[0].hid)
        self.assertEqual(len(bank.alive), 2)
        for k in (8, 9):
            ys, xs, ps, _, g, mu0 = wins[k]
            bank.step(k, ys, xs, ps, g, mu0)
        self.assertEqual(len(bank.alive), 1)
        self.assertGreaterEqual(bank.summary()["merged"], 1)

    def test_tube_birth_uses_tube_velocity_and_respects_cover(self):
        velocities = [(vy, vx) for vy in (-1.0, 0.0, 1.0) for vx in (-1.0, 0.0, 1.0)]
        bank = HypothesisBank(H, W, velocities)
        tube = torch.zeros(len(velocities), H, W)
        tube[7, 30, 40] = 5.0                                    # 速度 (1, 0)
        tube[2, 30, 41] = 4.5                                    # 紧挨着的第二个峰：已被第一个候选覆盖
        ys, xs, ps, _, g, mu0 = make_window(np.random.default_rng(3), None)
        bank.step(0, ys, xs, ps, g, mu0, tube_C=tube[None])
        self.assertEqual(len(bank.alive), 1)
        h = bank.alive[0]
        self.assertEqual(h.origin, "tube")
        self.assertTrue(torch.equal(h.velocity, torch.tensor([1.0, 0.0], dtype=F64)))
        self.assertTrue(torch.equal(h.anchor, torch.tensor([30.0, 40.0], dtype=F64)))

    def test_update_variants_track_the_target(self):
        for update in ("generic", "confident"):
            bank = HypothesisBank(H, W, update=update)
            run(bank, self.windows(12, seed=4))
            self.assertGreaterEqual(len(bank.confirmed_hypotheses()), 1, update)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            HypothesisBank(H, W, update="bayes")
        with self.assertRaises(ValueError):
            HypothesisBank(H, W, no_such_param=1)
        with self.assertRaises(ValueError):
            AttributionReadout(HypothesisBank(H, W), [0])
        with self.assertRaises(ValueError):
            AttributionReadout(HypothesisBank(H, W), [1], variants=("oracle",))


class ReadoutPropositions(unittest.TestCase):
    def test_p1_no_new_information_gives_exactly_zero(self):
        """P1：第 9 窗之后没有新的有效测量（目标消失、背景远离），快照参照下修正严格为 0。"""
        rng = np.random.default_rng(5)
        wins = [make_window(rng, track(k)) for k in range(10)]
        wins += [make_window(rng, None, n_bg=0, extra=[(2, 2, 0.05), (45, 60, 0.05)]) for _ in range(3)]
        bank = HypothesisBank(H, W)
        readout = AttributionReadout(bank, [2])
        out = run(bank, wins, readout)
        res, published = out[(9, 2)]
        self.assertEqual(published, 11)
        case1 = res["case"] == 1
        self.assertTrue(bool(case1.any()))
        self.assertTrue(bool((res["attr"][case1] == 0).all()))

    def test_p2_duplicate_hypothesis_does_not_change_readout(self):
        rng = np.random.default_rng(6)
        wins = [make_window(rng, track(k)) for k in range(12)]
        bank = HypothesisBank(H, W)
        readout = AttributionReadout(bank, [2])
        run(bank, wins[:11])
        ys, xs, _, _, g, mu0 = wins[8]
        entry = readout._register(8, 8, ys, xs, mu0, g)
        before = readout._readout(entry, 10)
        bank.clone_hypothesis(bank.confirmed_hypotheses()[0].hid)
        after = readout._readout(entry, 10)
        for name in before:
            self.assertTrue(torch.equal(before[name], after[name]), name)

    def test_p3_bounds_and_fallback(self):
        rng = np.random.default_rng(7)
        wins = [make_window(rng, track(k)) for k in range(12)]
        empty = HypothesisBank(H, W)
        readout = AttributionReadout(empty, [1])
        ys, xs, _, _, g, mu0 = wins[0]
        res = readout._readout(readout._register(0, 0, ys, xs, mu0, g), 1)
        self.assertTrue(bool((res["attr"] == 0).all()) and bool((res["case"] == 0).all()))
        bank = HypothesisBank(H, W)
        readout = AttributionReadout(bank, [1, 2, 5], cap=1.5)
        out = run(bank, wins, readout)
        for res, _ in out.values():
            self.assertLessEqual(float(res["attr"].abs().max()), 1.5 + 1e-12)
            no = res["case"] == 0
            self.assertTrue(bool((res["attr"][no] == 0).all()))

    def test_p4_snapshot_is_frozen(self):
        rng = np.random.default_rng(8)
        wins = [make_window(rng, track(k)) for k in range(12)]
        bank = HypothesisBank(H, W)
        run(bank, wins[:7])
        frozen = {hid: tuple(t.clone() if torch.is_tensor(t) else t for t in snap)
                  for hid, snap in bank.snapshots[6].items()}
        self.assertTrue(frozen)
        for k in range(7, 12):
            ys, xs, ps, _, g, mu0 = wins[k]
            bank.step(k, ys, xs, ps, g, mu0)
        for hid, (center, sigma, S) in frozen.items():
            c2, s2, S2 = bank.snapshots[6][hid]
            self.assertTrue(torch.equal(center, c2) and sigma == s2 and torch.equal(S, S2))

    def test_pre_confirmation_events_on_track_gain_and_offset_background_does_not(self):
        """目标刚出现（前 3 窗初判不高，没有假设）；之后确认。第 1 窗的事件在 d = 5 时按固定参考回溯修正：
        轨迹上的事件得到明显的正修正，偏开 5 像素的背景不加分（V2-1 的位置级融合会给它加分）。"""
        rng = np.random.default_rng(9)
        wins = []
        for k in range(10):
            p = 0.5 if k < 3 else 0.95
            extra = None
            if k == 1:
                cy, cx = track(1)
                extra = [(round(cy - 5), round(cx + 3), 0.5), (round(cy + 5), round(cx - 3), 0.5)]
            wins.append(make_window(rng, track(k), p_target=p, extra=extra))
        bank = HypothesisBank(H, W)
        readout = AttributionReadout(bank, [5])
        out = run(bank, wins, readout)
        res, _ = out[(1, 5)]
        labels = wins[1][3]
        on_track = labels == 1
        offset = torch.zeros_like(on_track)
        offset[-2:] = True
        self.assertTrue(bool((res["case"][on_track] == 2).all()))
        self.assertGreater(float(res["attr"][on_track].mean()), 1.0)
        self.assertLessEqual(float(res["attr"][offset].max()), 1e-9)

    def test_variant_readouts_are_in_range(self):
        rng = np.random.default_rng(10)
        wins = [make_window(rng, track(k)) for k in range(12)]
        bank = HypothesisBank(H, W)
        readout = AttributionReadout(bank, [1, 2], cap=3.0, variants=VARIANTS)
        out = run(bank, wins, readout)
        self.assertTrue(out)
        for res, _ in out.values():
            self.assertTrue(set(VARIANTS) <= set(res))
            self.assertTrue(bool(((res["backfill"] == 0) | (res["backfill"] == 1)).all()))
            self.assertTrue(bool(((res["direct"] >= 0) & (res["direct"] <= 1)).all()))
            self.assertLessEqual(float(res["abs"].abs().max()), 3.0 + 1e-12)
            self.assertLessEqual(float(res["snnref"].abs().max()), 3.0 + 1e-12)

    def test_flush_truncates_pending_delays(self):
        rng = np.random.default_rng(11)
        wins = [make_window(rng, track(k)) for k in range(8)]
        bank = HypothesisBank(H, W)
        readout = AttributionReadout(bank, [1, 5])
        out = run(bank, wins, readout)
        flushed = readout.flush()
        keys = {(key, d) for key, d, _, _ in flushed}
        self.assertIn((7, 1), keys)                               # 最后一窗的 d=1 被截断在第 7 窗读出
        self.assertIn((3, 5), keys)
        self.assertNotIn((2, 5), keys)                            # 第 2 窗的 d=5 已在第 7 窗正常到期
        self.assertIn((2, 5), out)
        self.assertTrue(all(published == 7 for _, _, _, published in flushed))
        self.assertAlmostEqual(rho_reference(7), 1.0 / 225.0)


class EvalIntegration(unittest.TestCase):
    """U3：接入 train_stream_v2 的评估流程（命令行、读出名称、run_sequence）。"""

    def test_command_line_flags_reach_the_config(self):
        import os
        import sys
        from unittest import mock
        import train_stream_v2 as tv2
        config = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs",
                              "evisseg_stream_v2.yaml")
        argv = ["train_stream_v2.py", "--config", config, "--mode", "eval"]
        with mock.patch.object(sys, "argv", argv):
            cfg = tv2.build_config(tv2.parse_args())
        self.assertFalse(cfg["attr"])                              # 09-26 起默认关闭
        self.assertEqual(tv2.readout_names(cfg), ["net", "fused_d1", "fused_d2", "fused_d5"])
        with mock.patch.object(sys, "argv", argv + ["--attr", "on"]):
            self.assertEqual(tv2.readout_names(tv2.build_config(tv2.parse_args())),
                             ["net", "fused_d1", "fused_d2", "fused_d5", "attr_d1", "attr_d2", "attr_d5"])
        extra = ["--attr", "on", "--attr-delays", "2", "10", "--attr-variants", "backfill", "direct", "--attr-update", "generic",
                 "--attr-tube-birth", "off", "--hyp", "lag=3", "confirm_theta=5.5", "--no-alarms",
                 "--alarm-thetas", "8"]
        with mock.patch.object(sys, "argv", argv + extra):
            cfg = tv2.build_config(tv2.parse_args())
        self.assertEqual(cfg["attr_delays"], [2, 10])
        self.assertEqual(cfg["hyp_params"], {"lag": 3, "confirm_theta": 5.5})
        self.assertEqual(cfg["alarm_thetas"], [])                  # --no-alarms 优先
        self.assertFalse(cfg["attr_tube_birth"])
        names = tv2.readout_names(cfg)
        self.assertIn("attr_direct_d10", names)
        self.assertEqual([tv2.is_variant_readout(n) for n in ("attr_d2", "attr_direct_d2", "attr_backfill_d10")],
                         [False, True, True])
        self.assertEqual([tv2.readout_delay(n) for n in ("net", "fused_d5", "attr_direct_d10")], [0, 5, 10])
        self.assertEqual(tv2.publish_field("attr_direct_d10"), "publish_attr_d10")
        self.assertEqual(tv2.publish_field("fused_d2"), "publish_d2")
        bank, readout = tv2.build_attribution(cfg, tv2.build_cusum(cfg))
        self.assertEqual((bank.p["update"], bank.p["lag"], int(bank.velocities.shape[0])), ("generic", 3, 0))
        self.assertGreaterEqual(bank.p["keep_windows"], 3 + 10 + 2)
        self.assertEqual(readout.delays, [2, 10])
        with mock.patch.object(sys, "argv", argv + ["--hyp", "no_such=1"]):
            with self.assertRaises(ValueError):
                tv2.build_config(tv2.parse_args())
        with mock.patch.object(sys, "argv", argv + ["--attr", "off"]):
            self.assertEqual(tv2.readout_names(tv2.build_config(tv2.parse_args())),
                             ["net", "fused_d1", "fused_d2", "fused_d5"])

    def test_run_sequence_attr_leaves_v21_readouts_unchanged(self):
        try:
            from tests import test_stream_v2_evidence as ev
        except ImportError:
            import test_stream_v2_evidence as ev
        from model.evidence_neuron import DriftCUSUM, velocity_grid
        from train_stream_v2 import run_sequence
        seq = ev.synthetic_sequence(seed=2)
        frontend = ev.make_frontend()
        model = ev.make_model(frontend).eval()
        cusum = DriftCUSUM(velocity_grid([-1.0, 0.0, 1.0]), footprint=3, track_decay=0.8)
        cfg_off = dict(ev.CFG)
        cfg_on = dict(ev.CFG, attr=True, attr_delays=[1, 2], attr_variants=list(VARIANTS),
                      hyp_params={"birth_conf": 0.0, "min_effective": 1.0, "confirm_theta": 0.5})
        with torch.no_grad():
            p_off, x_off, c_off = run_sequence(model, frontend, cusum, seq, cfg_off, torch.device("cpu"), "carry")
            p_on, x_on, c_on = run_sequence(model, frontend, cusum, seq, cfg_on, torch.device("cpu"), "carry")
        for name in p_off:                                         # V2-1 的读出逐位不变
            self.assertTrue(np.array_equal(p_off[name], p_on[name]), name)
        for name in x_off:
            self.assertTrue(np.array_equal(x_off[name], x_on[name]), name)
        self.assertTrue(np.array_equal(c_off, c_on))
        n = seq.n_events
        bank = x_on.pop("attr_bank")
        self.assertGreater(bank["stats"]["births_snn"], 0)
        self.assertEqual(bank["windows"], ev.WINDOWS)
        for d in (1, 2):
            for name in ["attr_d%d" % d] + ["attr_%s_d%d" % (v, d) for v in VARIANTS]:
                self.assertEqual(p_on[name].shape[0], n)
                self.assertTrue(np.all((p_on[name] >= 0) & (p_on[name] <= 1)), name)
            delta, case = x_on["delta_attr_d%d" % d], x_on["case_attr_d%d" % d]
            self.assertTrue(np.all(delta[case == 0] == 0))
            self.assertTrue(set(np.unique(case).tolist()) <= {0.0, 1.0, 2.0})
            self.assertEqual(x_on["publish_attr_d%d" % d].tolist(), [min(k + d, ev.WINDOWS - 1)
                                                                     for k in range(ev.WINDOWS)])
            z = x_on["logit_net"].astype(np.float64) + delta.astype(np.float64)
            self.assertLess(float(np.abs(1.0 / (1.0 + np.exp(-z)) - p_on["attr_d%d" % d]).max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
