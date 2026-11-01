"""模块三 v5：目标神经元层（model/object_layer.py）的测试（CPU、合成数据）。

    1. 初始化时零等待输出与骨干 logit 逐位相同（读出末层零初始化）
    2. 片段切分不变：一次处理 12 窗与分 3 段（段间 detach）处理，零等待与终判输出逐位相同（弱目标，确有事件挂起）
    3. 目标神经元：一个清楚的目标只出生一个槽（依次出生，第二个候选被解释掉），电位越过 +A，位置跟得上；只有背景时没有槽存活
    4. 存在证据的公式：每个事件的输入 = log(1 + 后验几率 / 先验)，与泊松扩展目标似然比的直接计算一致
    5. 挂起与终判：目标未判定时事件挂起，神经元越过 +A 时终判，z_final = z_now + m x (u(目标此刻) - u(目标当时))
       （同一个读出按判决时刻的目标状态重读）；等满 defer_max 窗时强制终判
    6. 新录像开始（keep = 0）时只清空该流的目标槽与挂起事件
    7. 监督：匹配上的槽、存在 / 归属 / 位置损失有限且可反传，梯度到达关联、动力学、神经元、读出各部分；
       上一窗匹配上的槽在目标仍在时保持同一身份
    8. 配置：手设量只有槽数、宽度、EM 迭代次数、延迟预算、alpha；长度单位由骨干结构读出
    9. v2：窗内连续时间一窗量出速度（v1 量不出）；同样的事件，背景率越高存在证据越弱；没有背景率时报错；分段不变
   10. 消融对照：读出不看目标时输出与目标状态无关且不挂起；固定等待时已确认目标上的事件也等满 defer_max 窗
"""
import math
import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.object_layer import OBJECT_KEYS, ObjectLayer, backbone_stride  # noqa: E402
from test_track_head import FEAT, H, W, blob_windows, run_split_chunks, window_events  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_layer(seed=0, **kwargs):
    torch.manual_seed(seed)
    params = dict(slots=6, hidden=16, attn_hidden=16, em_iters=2, defer_max=10, defer_alpha=0.05, length_unit=8.0)
    params.update(kwargs)
    layer = ObjectLayer(FEAT, H, W, **params)
    layer.init_rate(7.0)                       # blob_windows: 7 events per target per window (the trainer's data init)
    return layer


def randomize_readout(layer, std=0.3):
    with torch.no_grad():
        for mod in (layer.read_out, layer.read_bg[2], layer.slot_query):
            for p in mod.parameters():
                p.normal_(0.0, std)


class IdentityAndChunkTests(unittest.TestCase):
    def test_zero_wait_equals_backbone_at_init(self):
        layer = make_layer()
        events, z0, feat, lab, tid = blob_windows(8, batch=2, stream=1, noise=12)
        z, _, aux = layer.forward_chunk(events, z0, feat, None, 8)
        self.assertTrue(torch.equal(z, z0))
        self.assertEqual(aux["slot_windows"], 16)

    def test_chunk_invariance(self):
        layer = make_layer(seed=1)
        randomize_readout(layer)
        events, z0, feat, lab, _ = blob_windows(12, batch=2, stream=0, noise=15, velocity=(1.5, 0.5), gaps=(5, 6))
        z0 = torch.where(lab > 0.5, torch.full_like(z0, 0.5), z0)    # weak target: its events wait for the neuron
        keys = torch.arange(z0.shape[0], dtype=torch.float32)
        z_a, _, d_a = run_split_chunks(layer, events, z0, feat, 2, [(0, 12)], keys=keys)
        z_b, _, d_b = run_split_chunks(layer, events, z0, feat, 2, [(0, 4), (4, 8), (8, 12)], keys=keys)
        self.assertTrue(torch.equal(z_a, z_b))
        self.assertGreater(len(d_a), 0)
        self.assertEqual(sorted(d_a), sorted(d_b))
        for k in d_a:
            self.assertEqual(d_a[k], d_b[k])


class NeuronTests(unittest.TestCase):
    def test_clear_target_confirmed_and_tracked(self):
        layer = make_layer()
        events, z0, feat, _, _ = blob_windows(10, start=(8.0, 12.0), velocity=(2.0, 0.0), noise=8)
        log = []
        _, state, _ = layer.forward_chunk(events, z0, feat, None, 10, log=log)
        act = state["active"][0]
        self.assertEqual(int(act.sum()), 1)                          # one object, one slot (no duplicate births)
        L = state["L"][0][act]
        self.assertTrue(bool((L >= layer.defer_A).any()))
        best = int(torch.argmax(torch.where(act, state["L"][0], torch.full_like(state["L"][0], -1e9))))
        pos = state["mu"][0, best] * layer.unit
        self.assertLess(float(torch.dist(pos, torch.tensor([8.0 + 2.0 * 9, 12.0]))), 2.0)
        self.assertEqual(len(log), 10)
        self.assertEqual(log[0]["pos"].shape, (1, layer.slots, 2))

    def test_background_only_leaves_no_slot(self):
        layer = make_layer()
        events, z0, feat, _, _ = blob_windows(6, noise=20, keep_out=range(6))
        _, state, _ = layer.forward_chunk(events, z0, feat, None, 6)
        self.assertFalse(bool(state["active"].any()))

    def test_evidence_term_matches_poisson_likelihood_ratio(self):
        # One slot k and background: log LR contribution of an event = log(1 + gamma p(e) / lambda_c(e));
        # with posterior beta = r gamma p / (lambda_c + r gamma p): log(1 + (beta / (1 - beta)) / r) is the same.
        torch.manual_seed(0)
        r, gamma = 0.3, 2.5
        p = torch.rand(50) * 0.2
        lam = torch.rand(50) * 0.1 + 0.01
        direct = torch.log1p(gamma * p / lam)
        beta = r * gamma * p / (lam + r * gamma * p)
        lb = torch.log(beta)
        via_posterior = F.softplus(lb - torch.log1p(-beta) - math.log(r))
        self.assertTrue(torch.allclose(direct, via_posterior, atol=1e-5))


class DeferTests(unittest.TestCase):
    def test_pending_then_released_by_formula(self):
        layer = make_layer()
        randomize_readout(layer)
        state = layer.init_state(1, torch.device("cpu"))
        state["active"][0, 0] = True
        state["gen"][0, 0] = 1
        state["mu"][0, 0] = torch.tensor([20.0, 16.0]) / layer.unit
        state["P"][0, 0] = torch.eye(2) * 0.05
        state["h"][0, 0] = 0.0
        state["L"][0, 0] = 0.0                                          # undecided object
        pts = [(20 + dx, 16 + dy) for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))]
        ev = window_events(pts, z=3.0)
        ev["key"] = torch.arange(len(pts), dtype=torch.float32)
        with torch.no_grad():
            layer.log_llr_scale.fill_(-20.0)                          # window 1: almost no evidence -> stays undecided
            z1, state, info = layer.forward_window(state, ev)
        self.assertIsNone(info["delayed"])
        dq = state["defer"]
        self.assertIsNotNone(dq)
        self.assertEqual(int(dq["b"].shape[0]), len(pts))
        self.assertTrue(bool((dq["L_then"].abs() < layer.defer_A).all()))
        self.assertTrue(torch.allclose(dq["znow"], z1))
        saved = {k: v.clone() for k, v in dq.items()}
        ev2 = window_events(pts, z=6.0)
        ev2["key"] = torch.arange(len(pts), len(pts) * 2, dtype=torch.float32)
        with torch.no_grad():
            layer.log_llr_scale.fill_(math.log(10.0))                 # window 2: strong evidence -> fires +A
            _, state2, info2 = layer.forward_window(state, ev2)
        zf, key, lag, status = info2["delayed"]
        self.assertTrue(bool((status == 0).all()))                    # confirmed
        self.assertTrue(bool((lag == 1).all()))
        L_new = state2["L"][0, 0]
        self.assertGreaterEqual(float(L_new), layer.defer_A)
        with torch.no_grad():
            qs = layer.read_slot(torch.cat([state2["h"][0, 0], torch.sigmoid(L_new)[None]]))
            u_now = layer.read_out(F.relu(qs[None, :] + saved["qe"])).squeeze(-1)
        expect = saved["znow"] + saved["m_then"] * (u_now - saved["u_then"])
        order = torch.argsort(key)
        self.assertTrue(torch.allclose(zf[order], expect[torch.argsort(saved["key"])], atol=1e-5))
        self.assertFalse(torch.allclose(zf, saved["znow"]))           # the object's decision changed the output

    def test_deadline_forces_release(self):
        layer = make_layer(defer_max=3)
        state = layer.init_state(1, torch.device("cpu"))
        state["active"][0, 0] = True
        state["gen"][0, 0] = 1
        state["mu"][0, 0] = torch.tensor([20.0, 16.0]) / layer.unit
        state["P"][0, 0] = torch.eye(2) * 0.05
        ev = window_events([(20, 16), (21, 16)], z=3.0)
        ev["key"] = torch.tensor([0.0, 1.0])
        statuses, lags = [], []
        with torch.no_grad():
            layer.log_llr_scale.fill_(-20.0)
            layer.pd_head.bias.fill_(-20.0)                            # P_D ~ 0: an empty window carries no evidence
            layer.surv_head.bias.fill_(20.0)                           # survival ~ 1: the membrane does not drift
            _, state, _ = layer.forward_window(state, ev)
            for _ in range(4):
                empty = window_events([], z=0.0)
                empty["key"] = torch.zeros(0)
                _, state, info = layer.forward_window(state, empty)
                if info["delayed"] is not None:
                    statuses.append(info["delayed"][3])
                    lags.append(info["delayed"][2])
        self.assertTrue(statuses)
        self.assertTrue(bool((torch.cat(statuses) == 4).all()))
        self.assertTrue(bool((torch.cat(lags) == 3).all()))

    def test_reset_clears_only_that_stream(self):
        layer = make_layer()
        events, z0, feat, _, _ = blob_windows(6, batch=2, stream=0, noise=6)
        e1, z1, f1, _, _ = blob_windows(6, batch=2, stream=1, noise=0, seed=3)
        merged = {k: torch.cat([events[k], e1[k]]) for k in events}
        z_all, f_all = torch.cat([z0, z1]), torch.cat([feat, f1])
        keep = torch.ones(6, 2)
        keep[4, 1] = 0.0
        _, state, _ = layer.forward_chunk(merged, z_all, f_all, None, 4, keep=keep[:4])
        self.assertTrue(bool(state["active"][1].any()))
        after = {k: v[merged["t"] >= 4] for k, v in merged.items()}
        after["t"] = after["t"] - 4
        sel = merged["t"] >= 4
        only_reset = {k: v[(after["t"] == 0) & (after["b"] == 0)] for k, v in after.items()}
        only_reset["t"] = torch.zeros_like(only_reset["t"])
        zr, fr = z_all[sel][(after["t"] == 0) & (after["b"] == 0)], f_all[sel][(after["t"] == 0) & (after["b"] == 0)]
        _, state2, _ = layer.forward_chunk(only_reset, zr, fr, state, 1, keep=keep[4:5])
        self.assertFalse(bool(state2["active"][1].any()))
        self.assertTrue(bool(state2["active"][0].any()))
        if state2["defer"] is not None:
            self.assertFalse(bool((state2["defer"]["b"] == 1).any()))


class SupervisionTests(unittest.TestCase):
    def test_losses_backprop_and_identity_persists(self):
        layer = make_layer(seed=2)
        randomize_readout(layer, std=0.1)
        events, z0, feat, lab, tid = blob_windows(8, batch=2, stream=1, noise=10, velocity=(1.0, 1.0))
        z0 = torch.where(lab > 0.5, torch.full_like(z0, 1.0), z0)    # evidence not saturated (float32)
        z, state, aux = layer.forward_chunk(events, z0, feat, None, 8, labels=lab, tid=tid, keys=lab)
        self.assertGreater(aux["exist_n"], 0)
        self.assertGreater(aux["mem_n"], 0)
        self.assertGreater(aux["pos_n"], 0)
        loss = F.binary_cross_entropy_with_logits(z, lab) + aux["exist_sum"] / aux["exist_n"] \
            + aux["mem_sum"] / aux["mem_n"] + 0.5 * aux["pos_sum"] / aux["pos_n"]
        if aux["delayed_z"] is not None:
            loss = loss + F.binary_cross_entropy_with_logits(aux["delayed_z"], aux["delayed_key"])
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for name in ("event_key.weight", "clutter_bias", "clutter_gain", "rate_head.bias", "pd_head.bias",
                     "extent_head.bias", "gru.weight_ih", "read_out.weight", "read_bg.2.weight", "birth_logit",
                     "log_llr_scale", "log_s0", "slot_query.weight"):
            grad = dict(layer.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(bool(torch.isfinite(grad).all()), name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)
        act = state["active"][1]
        self.assertTrue(bool((state["tid"][1][act] == 1.0).any()))    # the slot that follows the target keeps id 1

    def test_deterministic_mode_runs(self):
        torch.use_deterministic_algorithms(True)
        try:
            layer = make_layer(seed=3)
            events, z0, feat, lab, tid = blob_windows(5, noise=6)
            z, _, aux = layer.forward_chunk(events, z0, feat, None, 5, labels=lab, tid=tid, keys=lab)
            (z.sum() + aux["exist_sum"] + aux["mem_sum"]).backward()
        finally:
            torch.use_deterministic_algorithms(False)


class V2MechanismTests(unittest.TestCase):
    """v2：窗内连续时间（一窗量出速度）与杂波强度取模块二的背景率。"""

    def _fast_target_window(self, layer, v_true=2.0):
        state = layer.init_state(1, torch.device("cpu"))
        state["active"][0, 0] = True
        state["gen"][0, 0] = 1
        state["mu"][0, 0] = torch.tensor([24.0, 16.0]) / layer.unit
        state["P"][0, 0] = torch.tensor([[0.05, 0.0], [0.0, 4.0]])      # position known, velocity unknown
        state["L"][0, 0] = 5.0
        taus = torch.linspace(0.05, 0.95, 10)
        xs = 24.0 + v_true * layer.unit * (taus - 0.5)                 # moves v_true units per window inside the window
        ev = window_events([(int(round(float(x))), 16) for x in xs], z=4.0)
        ev["t_local"] = taus
        ev["key"] = torch.arange(10, dtype=torch.float32)
        with torch.no_grad():
            layer.extent_head.bias.fill_(-9.0)                          # tiny extent: the velocity measurement is sharp
            layer.surv_head.bias.fill_(20.0)
            _, new, _ = layer.forward_window(state, ev)
        return new

    def test_continuous_time_measures_velocity_in_one_window(self):
        moving = self._fast_target_window(make_layer(continuous_time=True))
        static = self._fast_target_window(make_layer(continuous_time=False))
        self.assertGreater(float(moving["v"][0, 0, 0]), 1.6)          # ~2 units/window from a single window
        self.assertLess(abs(float(static["v"][0, 0, 0])), 0.3)        # v1: only the position is measured

    def test_background_clutter_weakens_evidence(self):
        Ls = []
        for rate in (1e-3, 1e-1):                                        # quiet vs noisy environment
            layer = make_layer(clutter_source="background", ref_rate=(1e-4, 1e-4))
            state = layer.init_state(1, torch.device("cpu"))
            state["active"][0, 0] = True
            state["gen"][0, 0] = 1
            state["mu"][0, 0] = torch.tensor([20.0, 16.0]) / layer.unit
            state["P"][0, 0] = torch.eye(2) * 0.05
            pts = [(20 + dx, 16 + dy) for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))]
            ev = window_events(pts, z=0.5)
            ev["bg_event"] = torch.full((len(pts),), rate)
            ev["key"] = torch.arange(len(pts), dtype=torch.float32)
            with torch.no_grad():
                _, new, _ = layer.forward_window(state, ev)
            Ls.append(float(new["L"][0, 0]) if bool(new["active"][0, 0]) else -1e9)
        self.assertGreater(Ls[0], Ls[1])                                # same events, noisier background -> less evidence

    def test_background_mode_requires_rates(self):
        layer = make_layer(clutter_source="background")
        ev = window_events([(10, 10)], z=2.0)
        ev["key"] = torch.zeros(1)
        with self.assertRaises(ValueError):
            layer.forward_window(layer.init_state(1, torch.device("cpu")), ev)

    def test_v2_chunk_invariance(self):
        layer = make_layer(seed=4, clutter_source="background", continuous_time=True)
        randomize_readout(layer)
        events, z0, feat, lab, _ = blob_windows(12, batch=2, stream=0, noise=15, velocity=(2.5, 0.5))
        z0 = torch.where(lab > 0.5, torch.full_like(z0, 0.5), z0)
        events["bg_event"] = torch.rand(z0.shape[0]) * 0.01
        keys = torch.arange(z0.shape[0], dtype=torch.float32)
        z_a, _, d_a = run_split_chunks(layer, events, z0, feat, 2, [(0, 12)], keys=keys)
        z_b, _, d_b = run_split_chunks(layer, events, z0, feat, 2, [(0, 5), (5, 12)], keys=keys)
        self.assertTrue(torch.equal(z_a, z_b))
        self.assertEqual(d_a, d_b)


class ControlTests(unittest.TestCase):
    """消融对照：读出不看目标（object_context = False）；固定等待（decision_rule = fixed）。"""

    def _confirmed_state(self, layer, L=10.0):
        state = layer.init_state(1, torch.device("cpu"))
        state["active"][0, 0] = True
        state["gen"][0, 0] = 1
        state["mu"][0, 0] = torch.tensor([20.0, 16.0]) / layer.unit
        state["P"][0, 0] = torch.eye(2) * 0.05
        state["L"][0, 0] = L
        return state

    def test_no_object_context_ignores_slots(self):
        layer = make_layer(object_context=False)
        randomize_readout(layer)
        ev = window_events([(20, 16), (21, 16), (20, 17)], z=2.0)
        ev["key"] = torch.arange(3, dtype=torch.float32)
        with torch.no_grad():
            z_obj, st, _ = layer.forward_window(self._confirmed_state(layer), ev)
            z_none, _, _ = layer.forward_window(layer.init_state(1, torch.device("cpu")), ev)
            expect = ev["z0"] + layer.read_bg(layer._event_features(ev)).squeeze(-1)
        self.assertTrue(torch.allclose(z_obj, expect))
        self.assertTrue(torch.allclose(z_none, expect))
        self.assertIsNone(st["defer"])

    def test_fixed_rule_waits_full_budget(self):
        pts = [(20, 16), (21, 16), (20, 17)]
        for rule, waits in (("sprt", False), ("fixed", True)):
            layer = make_layer(decision_rule=rule, defer_max=3)
            state = self._confirmed_state(layer)
            ev = window_events(pts, z=4.0)
            ev["key"] = torch.arange(3, dtype=torch.float32)
            released = []
            with torch.no_grad():
                _, state, _ = layer.forward_window(state, ev)
                self.assertEqual(state["defer"] is not None and int(state["defer"]["b"].shape[0]) > 0, waits, rule)
                for _ in range(3):
                    nxt = window_events(pts, z=4.0)
                    nxt["key"] = torch.arange(100, 103, dtype=torch.float32)
                    _, state, info = layer.forward_window(state, nxt)
                    if info["delayed"] is not None:
                        released.append(info["delayed"])
            if waits:
                first = [r for r in released if bool((r[1] < 100).any())]
                self.assertTrue(first)
                self.assertTrue(bool((first[0][3] == 4).all()))        # forced at the deadline, not "confirmed"
                self.assertTrue(bool((first[0][2] == 3).all()))


class ConfigTests(unittest.TestCase):
    def test_only_capacity_requirement_keys(self):
        from utils.stream_common import load_flat_config
        import yaml
        with open(os.path.join(ROOT, "configs", "object_layer_v1.yaml"), encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        self.assertTrue(set(raw["HEAD"]) <= {"head_kind"} | set(OBJECT_KEYS), set(raw["HEAD"]))
        cfg = load_flat_config(os.path.join(ROOT, "configs", "object_layer_v1.yaml"))
        self.assertEqual(cfg["head_kind"], "object")
        self.assertFalse(cfg["calibrate_existence"])
        v4f = load_flat_config(os.path.join(ROOT, "configs", "track_head_v4f.yaml"))
        for key in ("base_checkpoint", "epochs", "lr", "lr_end", "streams", "tbptt_k", "grad_clip", "seed",
                    "threshold", "defer_max", "defer_alpha"):
            self.assertEqual(cfg[key], v4f[key], key)               # same protocol as v4f: only the module differs

    def test_length_unit_from_backbone_structure(self):
        net = nn.Sequential(nn.Conv2d(2, 4, 3, stride=2, padding=1),
                            nn.ConvTranspose2d(4, 4, 2, stride=2), nn.ConvTranspose2d(4, 4, 2, stride=2),
                            nn.ConvTranspose2d(4, 4, 2, stride=2))
        self.assertEqual(backbone_stride(net), 8.0)


if __name__ == "__main__":
    unittest.main()
