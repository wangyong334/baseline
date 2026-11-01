"""模块三 v6：目标 token 层（model/object_tokens.py）的测试（CPU、合成数据）。

    1. 初始化：零等待输出与骨干 logit 逐位相同；证据电流为 0，新 token 停在否定界上：没有 token 存活、没有事件挂起
    2. 片段切分不变：参数随机化（有 token 存活、有事件挂起）后，一次处理与分段处理的零等待、终判输出逐位相同
    3. 结构：发放阈值 = Wald 的 A（由 alpha 给出，不是参数）；漏电初值 1 - 1 / defer_max、静息电位初值 -A；
       没有数据集统计的初值（没有 init_rate）；配置的 HEAD 只含架构量与需求量，协议与 object_layer_v2 相同
    4. 提议：每条流只在本流事件上提议，最多 proposals 个、不超过空闲槽，按槽号放；已被老 token 认领的事件不再被提议
    5. 神经元与判决：证据为正时越过 +A 确认，为负时越过 -A 否定并释放槽，挂在它上面的事件随之终判（状态码 0 / 1）；
       终判 = 用同一个读出按 token 此刻的状态重读；等满 defer_max 窗时强制终判（状态码 4）
    6. 新录像开始（keep = 0）时只清空该流
    7. 监督：损失有限、可反传，梯度到达注意力、预测、校正、门、去重自注意力、提议嵌入、神经元（证据、漏电、静息电位）、读出；
       上一窗匹配上的 token 在目标仍在时保持同一身份；选择性分离：归属损失到达骨干输出，存在、位置损失不到达
    8. 能学：合成数据上训练后，目标的 token 被确认、位置跟得上，纯背景段没有 token 存活
    9. 确定性模式下前向、反向不报错；clutter_source = background 时没有背景率报错，背景率进入背景 logit
   10. 消融对照：读出不看 token 时输出与 token 状态无关且不挂起；固定等待时已确认 token 上的事件也等满 defer_max 窗
"""
import math
import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.object_tokens import TOKEN_KEYS, ObjectTokenLayer  # noqa: E402
from test_track_head import FEAT, H, W, blob_windows, run_split_chunks, window_events  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_layer(seed=0, **kwargs):
    torch.manual_seed(seed)
    params = dict(slots=6, proposals=3, hidden=16, attn_hidden=16, iters=2, defer_max=10, defer_alpha=0.05,
                  length_unit=8.0)
    params.update(kwargs)
    return ObjectTokenLayer(FEAT, H, W, **params)


def randomize(layer, std=0.3, evidence=None):
    """读出、内容相似度、门、去重输出随机化；evidence 给出时证据电流固定为该值（证据头末层权重为 0）。"""
    with torch.no_grad():
        for mod in (layer.read_out, layer.read_bg[2], layer.slot_query, layer.mix_out, layer.gate_head):
            for p in mod.parameters():
                p.normal_(0.0, std)
        if evidence is not None:
            layer.evid_head[2].weight.zero_()
            layer.evid_head[2].bias.fill_(float(evidence))


def single_token_state(layer, x=20.0, y=16.0, L=0.0):
    """一条流、槽 0 上一个已有 token（位置 (x, y) 像素，膜电位 L）。"""
    state = layer.init_state(1, torch.device("cpu"))
    state["active"][0, 0] = True
    state["gen"][0, 0] = 1
    state["r"][0, 0] = torch.tensor([x, y]) / layer.unit
    state["L"][0, 0] = L
    return state


BLOB = [(20 + dx, 16 + dy) for dx, dy in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))]


class IdentityAndChunkTests(unittest.TestCase):
    def test_zero_wait_equals_backbone_and_nothing_survives_at_init(self):
        layer = make_layer()
        events, z0, feat, _, _ = blob_windows(8, batch=2, stream=1, noise=12)
        z, state, aux = layer.forward_chunk(events, z0, feat, None, 8, keys=torch.arange(z0.shape[0]).float())
        self.assertTrue(torch.equal(z, z0))
        self.assertFalse(bool(state["active"].any()))               # zero evidence: proposals stay on the reject bound
        self.assertIsNone(state["defer"])
        self.assertIsNone(aux["delayed_z"])
        self.assertEqual(aux["slot_windows"], 16)

    def test_chunk_invariance(self):
        layer = make_layer(seed=1)
        randomize(layer, evidence=1.0)                               # tokens survive and stay undecided for a while
        events, z0, feat, lab, _ = blob_windows(12, batch=2, stream=0, noise=15, velocity=(1.5, 0.5), gaps=(5, 6))
        keys = torch.arange(z0.shape[0], dtype=torch.float32)
        z_a, _, d_a = run_split_chunks(layer, events, z0, feat, 2, [(0, 12)], keys=keys)
        z_b, _, d_b = run_split_chunks(layer, events, z0, feat, 2, [(0, 4), (4, 8), (8, 12)], keys=keys)
        self.assertTrue(torch.equal(z_a, z_b))
        self.assertGreater(len(d_a), 0)
        self.assertEqual(sorted(d_a), sorted(d_b))
        for k in d_a:
            self.assertEqual(d_a[k], d_b[k])


class StructureTests(unittest.TestCase):
    def test_extreme_width_has_finite_logits_loss_and_gradients(self):
        for width in (-1000.0, -100.0, 100.0, 1000.0):
            layer = make_layer()
            randomize(layer, std=0.1, evidence=1.5)
            with torch.no_grad():
                layer.width_head.bias.fill_(width)
            events, z0, feat, lab, tid = blob_windows(4, noise=10, velocity=(1., 0.5))
            z, state, aux = layer.forward_chunk(events, z0, feat, None, 4, labels=lab, tid=tid, keys=lab)
            loss = F.binary_cross_entropy_with_logits(z, lab)
            for key in ("exist", "pos", "mem"):
                loss = loss + aux[key + "_sum"] / max(aux[key + "_n"], 1)
            self.assertTrue(bool(torch.isfinite(loss)), width)
            loss.backward()
            for name, p in layer.named_parameters():
                if p.grad is not None:
                    self.assertTrue(bool(torch.isfinite(p.grad).all()), (width, name))
            for key in ("h", "r", "u", "L"):
                self.assertTrue(bool(torch.isfinite(state[key]).all()), (width, key))

    def test_precision_initial_value_and_subpixel_floor(self):
        layer = make_layer()
        h = torch.zeros(1, layer.slots, layer.hidden)
        self.assertTrue(torch.allclose(layer._precision(h), torch.ones(1, layer.slots)))
        with torch.no_grad():
            layer.width_head.bias.fill_(-1000.)
        self.assertLessEqual(float(layer._precision(h).max()), layer.unit ** 2)

    def test_large_displacement_stays_on_canvas_across_empty_windows(self):
        layer = make_layer()
        randomize(layer, evidence=1.5)
        with torch.no_grad():
            layer.step_head.bias.copy_(torch.tensor([1e20, -1e20]))
        state = single_token_state(layer, L=5.)
        for _ in range(20):
            _, state, _ = layer.forward_window(state, window_events([]))
            self.assertTrue(bool(torch.isfinite(state["r"]).all()))
            self.assertTrue(bool((state["r"] >= 0).all()))
            self.assertTrue(bool((state["r"] <= layer.xy_limit).all()))
            self.assertTrue(bool((state["u"].abs() <= layer.xy_limit).all()))

    def test_wald_bound_and_initialisation(self):
        layer = make_layer()
        self.assertFalse(hasattr(layer, "init_rate"))               # no data-statistic initial values
        self.assertAlmostEqual(layer.defer_A, math.log(0.95 / 0.05))
        names = dict(layer.named_parameters())
        self.assertNotIn("defer_A", names)                           # the bound is a requirement, not learned
        self.assertAlmostEqual(float(torch.sigmoid(layer.leak_head.bias)), 1.0 - 1.0 / 10, places=5)
        self.assertAlmostEqual(float(layer.rest), -layer.defer_A, places=5)
        self.assertEqual(layer.evidence_summary(), {"orphan_ref": float(layer.rest)})

    def test_config_keys_and_protocol(self):
        import yaml
        from utils.stream_common import load_flat_config
        with open(os.path.join(ROOT, "configs", "object_tokens_v1.yaml"), encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        self.assertTrue(set(raw["HEAD"]) <= {"head_kind"} | set(TOKEN_KEYS), set(raw["HEAD"]))
        cfg = load_flat_config(os.path.join(ROOT, "configs", "object_tokens_v1.yaml"))
        self.assertEqual(cfg["head_kind"], "tokens")
        self.assertFalse(cfg["calibrate_existence"])
        v5 = load_flat_config(os.path.join(ROOT, "configs", "object_layer_v2.yaml"))
        for key in ("base_checkpoint", "epochs", "lr", "lr_end", "streams", "tbptt_k", "grad_clip", "seed", "threshold",
                    "defer_max", "defer_alpha", "lambda_exist", "lambda_pos", "lambda_membership", "clutter_source"):
            self.assertEqual(cfg[key], v5[key], key)                 # same protocol as v5: only the module differs


class ProposalTests(unittest.TestCase):
    def test_per_stream_top_k_into_free_slots(self):
        layer = make_layer(slots=6, proposals=3)
        eb = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        onehot = (eb[:, None] == torch.arange(2)[None, :]).float()
        score = torch.arange(10, dtype=torch.float32)
        free = torch.ones(2, 6, dtype=torch.bool)
        free[0] = torch.tensor([False, True, False, False, True, False])          # stream 0: two free slots
        b, s, e = layer._propose(score, free, onehot)
        got = sorted(zip(b.tolist(), s.tolist(), e.tolist()))
        self.assertEqual(got, [(0, 1, 4), (0, 4, 3), (1, 0, 9), (1, 1, 8), (1, 2, 7)])

    def test_claimed_events_are_not_proposed(self):
        picked = []
        for claim in (False, True):
            layer = make_layer(slots=4, proposals=2)
            with torch.no_grad():
                layer.slot_bias.bias.fill_(20.0)                    # a token claims the events around it
            orig = layer._propose

            def spy(score, free, onehot, orig=orig):
                out = orig(score, free, onehot)
                picked.append(out[2].clone())
                return out
            layer._propose = spy
            ev = window_events(BLOB + [(40, 4), (3, 28), (44, 30)], z=6.0)
            ev["z0"][len(BLOB):] = -2.0                               # three background-like events far away
            ev["key"] = torch.arange(len(BLOB) + 3, dtype=torch.float32)
            state = single_token_state(layer, L=5.0) if claim else layer.init_state(1, torch.device("cpu"))
            with torch.no_grad():
                layer.forward_window(state, ev)
        self.assertTrue(bool((picked[0] < len(BLOB)).all()))          # nothing claims the blob: it is proposed
        self.assertTrue(bool((picked[1] >= len(BLOB)).all()))         # claimed by the token: proposals go elsewhere


class NeuronTests(unittest.TestCase):
    def _pending_window(self, layer):
        randomize(layer, evidence=0.0)
        with torch.no_grad():
            layer.slot_bias.bias.fill_(2.0)                          # the token wins the blob events over background
        state = single_token_state(layer)
        ev = window_events(BLOB, z=3.0)
        ev["key"] = torch.arange(len(BLOB), dtype=torch.float32)
        with torch.no_grad():
            _, state, info = layer.forward_window(state, ev)
        return state, info

    def test_confirm_releases_with_reread(self):
        layer = make_layer(slots=1)                                  # one slot, occupied: no proposals
        state, info = self._pending_window(layer)
        self.assertIsNone(info["delayed"])
        dq = state["defer"]
        self.assertIsNotNone(dq)
        self.assertEqual(int(dq["b"].shape[0]), len(BLOB))
        self.assertTrue(bool((dq["L_then"].abs() < layer.defer_A).all()))
        saved = {k: v.clone() for k, v in dq.items()}
        ev2 = window_events(BLOB, z=3.0)
        ev2["key"] = torch.arange(len(BLOB), 2 * len(BLOB), dtype=torch.float32)
        with torch.no_grad():
            layer.evid_head[2].bias.fill_(10.0)                      # strong evidence: fires +A
            _, state2, info2 = layer.forward_window(state, ev2)
        zf, key, lag, status = info2["delayed"]
        self.assertTrue(bool((status == 0).all()))
        self.assertTrue(bool((lag == 1).all()))
        self.assertGreaterEqual(float(state2["L"][0, 0]), layer.defer_A)
        with torch.no_grad():
            qs = layer.read_slot(torch.cat([state2["h"][0, 0], torch.sigmoid(state2["L"][0, 0])[None]]))
            u_now = layer.read_out(F.relu(qs[None, :] + saved["qe"])).squeeze(-1)
        expect = saved["znow"] + saved["m_then"] * (u_now - saved["u_then"])
        self.assertTrue(torch.allclose(zf[torch.argsort(key)], expect[torch.argsort(saved["key"])], atol=1e-5))
        self.assertFalse(torch.allclose(zf, saved["znow"]))

    def test_reject_frees_slot_and_releases(self):
        layer = make_layer(slots=1)
        state, _ = self._pending_window(layer)
        empty = window_events([], z=0.0)
        empty["key"] = torch.zeros(0)
        with torch.no_grad():
            layer.evid_head[2].bias.fill_(-10.0)                     # strong negative evidence: fires -A
            _, state2, info2 = layer.forward_window(state, empty)
        self.assertTrue(bool((info2["delayed"][3] == 1).all()))
        self.assertEqual(int(info2["delayed"][3].shape[0]), len(BLOB))
        self.assertFalse(bool(state2["active"][0, 0]))
        self.assertEqual(int(state2["defer"]["b"].shape[0]), 0)      # every pending event was released

    def test_deadline_forces_release(self):
        layer = make_layer(slots=1, defer_max=3)
        state, _ = self._pending_window(layer)
        statuses, lags = [], []
        with torch.no_grad():
            layer.leak_head.bias.fill_(30.0)                         # no leak: the undecided membrane stays put
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
        randomize(layer, evidence=2.0)
        events, z0, feat, _, _ = blob_windows(6, batch=2, stream=0, noise=6)
        e1, z1, f1, _, _ = blob_windows(6, batch=2, stream=1, noise=0, seed=3)
        merged = {k: torch.cat([events[k], e1[k]]) for k in events}
        z_all, f_all = torch.cat([z0, z1]), torch.cat([feat, f1])
        keep = torch.ones(6, 2)
        keep[4, 1] = 0.0
        sel4 = merged["t"] < 4
        part = {k: v[sel4] for k, v in merged.items()}
        _, state, _ = layer.forward_chunk(part, z_all[sel4], f_all[sel4], None, 4, keep=keep[:4],
                                          keys=torch.arange(int(sel4.sum())).float())
        self.assertTrue(bool(state["active"][1].any()))
        only0 = (merged["t"] == 4) & (merged["b"] == 0)
        nxt = {k: v[only0] for k, v in merged.items()}
        nxt["t"] = torch.zeros_like(nxt["t"])
        _, state2, _ = layer.forward_chunk(nxt, z_all[only0], f_all[only0], state, 1, keep=keep[4:5],
                                           keys=torch.arange(int(only0.sum())).float())
        self.assertFalse(bool(state2["active"][1].any()))
        self.assertTrue(bool(state2["active"][0].any()))
        if state2["defer"] is not None:
            self.assertFalse(bool((state2["defer"]["b"] == 1).any()))


class SupervisionTests(unittest.TestCase):
    def test_losses_backprop_and_identity_persists(self):
        layer = make_layer(seed=2)
        randomize(layer, std=0.1, evidence=1.5)
        events, z0, feat, lab, tid = blob_windows(8, batch=2, stream=1, noise=10, velocity=(1.0, 1.0))
        z, state, aux = layer.forward_chunk(events, z0, feat, None, 8, labels=lab, tid=tid, keys=lab)
        for key in ("exist_n", "mem_n", "pos_n"):
            self.assertGreater(aux[key], 0, key)
        loss = F.binary_cross_entropy_with_logits(z, lab) + aux["exist_sum"] / aux["exist_n"] \
            + aux["mem_sum"] / aux["mem_n"] + 0.5 * aux["pos_sum"] / aux["pos_n"]
        if aux["delayed_z"] is not None:
            loss = loss + F.binary_cross_entropy_with_logits(aux["delayed_z"], aux["delayed_key"])
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for name in ("event_norm.weight", "event_key.weight", "event_value.weight", "slot_query.weight",
                     "slot_bias.bias", "width_head.bias", "bg_head.2.weight", "pred.2.weight", "step_head.weight",
                     "gru.weight_ih", "ffn.2.weight", "gate_head.weight", "mix_out.weight", "mix_pos.2.weight",
                     "seed_embed.2.weight", "evid_head.2.weight", "leak_head.bias", "rest", "read_out.weight",
                     "read_bg.2.weight"):
            grad = dict(layer.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(bool(torch.isfinite(grad).all()), name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)
        act = state["active"][1]
        self.assertTrue(bool((state["tid"][1][act] == 1.0).any()))    # the token that follows the target keeps id 1

    def test_selective_separation(self):
        layer = make_layer(separation="selective")
        randomize(layer, std=0.1, evidence=1.5)
        events, z0, feat, lab, tid = blob_windows(6, noise=8)
        z0 = z0.clone().requires_grad_(True)
        feat = feat.clone().requires_grad_(True)
        _, _, aux = layer.forward_chunk(events, z0, feat, None, 6, labels=lab, tid=tid, keys=lab)
        g_mem = torch.autograd.grad(aux["mem_sum"], [z0, feat], retain_graph=True, allow_unused=True)
        g_exist = torch.autograd.grad(aux["exist_sum"], [z0, feat], retain_graph=True, allow_unused=True)
        g_pos = torch.autograd.grad(aux["pos_sum"], [z0, feat], retain_graph=True, allow_unused=True)
        self.assertTrue(any(g is not None and float(g.abs().sum()) > 0.0 for g in g_mem))
        self.assertTrue(all(g is None or float(g.abs().sum()) == 0.0 for g in g_exist))
        self.assertTrue(all(g is None or float(g.abs().sum()) == 0.0 for g in g_pos))

    def test_deterministic_mode_runs(self):
        torch.use_deterministic_algorithms(True)
        try:
            layer = make_layer(seed=3)
            randomize(layer, evidence=1.0)
            events, z0, feat, lab, tid = blob_windows(5, noise=6)
            z, _, aux = layer.forward_chunk(events, z0, feat, None, 5, labels=lab, tid=tid, keys=lab)
            (z.sum() + aux["exist_sum"] + aux["mem_sum"] + aux["pos_sum"]).backward()
        finally:
            torch.use_deterministic_algorithms(False)


class LearningTests(unittest.TestCase):
    def test_learns_to_confirm_targets_and_reject_background(self):
        layer = make_layer(seed=0)
        opt = torch.optim.Adam(layer.parameters(), lr=3e-3)
        for step in range(200):
            vel = ((step % 5) - 2) * 0.8, ((step % 3) - 1) * 0.6
            events, z0, feat, lab, tid = blob_windows(8, batch=2, stream=step % 2, noise=12, seed=step % 40,
                                                      start=(10.0 + (step % 7) * 3, 10.0 + (step % 4) * 3), velocity=vel)
            g = torch.Generator().manual_seed(step)
            z0 = z0 + torch.randn(z0.shape, generator=g) * 1.5
            z, _, aux = layer.forward_chunk(events, z0, feat, None, 8, labels=lab, tid=tid, keys=lab)
            loss = F.binary_cross_entropy_with_logits(z, lab)
            if aux["delayed_z"] is not None:
                loss = loss + F.binary_cross_entropy_with_logits(aux["delayed_z"], aux["delayed_key"])
            for key, w in (("exist", 1.0), ("mem", 1.0), ("pos", 0.5)):
                if aux[key + "_n"]:
                    loss = loss + w * aux[key + "_sum"] / aux[key + "_n"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            events, z0, feat, _, _ = blob_windows(10, start=(8.0, 12.0), velocity=(1.5, 0.5), noise=12, seed=123)
            _, state, _ = layer.forward_chunk(events, z0, feat, None, 10)
            act, L = state["active"][0], state["L"][0]
            confirmed = act & (L >= layer.defer_A)
            self.assertTrue(bool(confirmed.any()))
            pos = state["r"][0][confirmed] * layer.unit
            err = torch.norm(pos - torch.tensor([8.0 + 1.5 * 9, 12.0 + 0.5 * 9]), dim=1).min()
            self.assertLess(float(err), 2.0)
            events, z0, feat, _, _ = blob_windows(8, noise=20, keep_out=range(8), seed=7)
            _, state, _ = layer.forward_chunk(events, z0, feat, None, 8)
            self.assertFalse(bool(state["active"].any()))


class ClutterInputTests(unittest.TestCase):
    def test_background_mode_requires_rates(self):
        layer = make_layer(clutter_source="background", ref_rate=(1e-3, 1e-4))
        ev = window_events([(10, 10)], z=2.0)
        ev["key"] = torch.zeros(1)
        with self.assertRaises(ValueError):
            layer.forward_window(layer.init_state(1, torch.device("cpu")), ev)

    def test_background_rate_enters_background_logit(self):
        layer = make_layer(clutter_source="background", ref_rate=(1e-3, 1e-4))
        randomize(layer)
        with torch.no_grad():
            layer.bg_head[2].weight.normal_(0.0, 0.5)
        outs = []
        for rate in (1e-3, 1e-1):
            ev = window_events(BLOB, z=1.0)
            ev["bg_event"] = torch.full((len(BLOB),), rate)
            ev["key"] = torch.arange(len(BLOB), dtype=torch.float32)
            with torch.no_grad():
                z, _, _ = layer.forward_window(layer.init_state(1, torch.device("cpu")), ev)
            outs.append(z)
        self.assertFalse(torch.allclose(outs[0], outs[1]))


class ControlTests(unittest.TestCase):
    """消融对照：读出不看 token（object_context = False）；固定等待（decision_rule = fixed）。"""

    def test_no_object_context_ignores_tokens(self):
        layer = make_layer(slots=1, object_context=False)
        randomize(layer, evidence=0.0)
        ev = window_events(BLOB, z=2.0)
        ev["key"] = torch.arange(len(BLOB), dtype=torch.float32)
        with torch.no_grad():
            z_tok, st, _ = layer.forward_window(single_token_state(layer, L=5.0), ev)
            z_none, _, _ = layer.forward_window(layer.init_state(1, torch.device("cpu")), ev)
            expect = ev["z0"] + layer.read_bg(layer.event_norm(layer._raw_features(ev))).squeeze(-1)
        self.assertTrue(torch.allclose(z_tok, expect))
        self.assertTrue(torch.allclose(z_none, expect))
        self.assertIsNone(st["defer"])

    def test_fixed_rule_waits_full_budget(self):
        for rule, waits in (("sprt", False), ("fixed", True)):
            layer = make_layer(slots=1, decision_rule=rule, defer_max=3)
            randomize(layer, evidence=0.0)
            with torch.no_grad():
                layer.slot_bias.bias.fill_(2.0)
            state = single_token_state(layer, L=10.0)                # an already confirmed token
            ev = window_events(BLOB, z=4.0)
            ev["key"] = torch.arange(len(BLOB), dtype=torch.float32)
            released = []
            with torch.no_grad():
                _, state, _ = layer.forward_window(state, ev)
                self.assertEqual(state["defer"] is not None and int(state["defer"]["b"].shape[0]) > 0, waits, rule)
                for _ in range(3):
                    nxt = window_events(BLOB, z=4.0)
                    nxt["key"] = torch.arange(100, 100 + len(BLOB), dtype=torch.float32)
                    _, state, info = layer.forward_window(state, nxt)
                    if info["delayed"] is not None:
                        released.append(info["delayed"])
            if waits:
                first = [r for r in released if bool((r[1] < 100).any())]
                self.assertTrue(first)
                self.assertTrue(bool((first[0][3] == 4).all()))       # forced at the deadline, not "confirmed"
                self.assertTrue(bool((first[0][2] == 3).all()))


if __name__ == "__main__":
    unittest.main()
