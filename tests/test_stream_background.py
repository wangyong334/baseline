"""V6b 事件表示层（背景率通道，dataset/background.py，Stream-SNN-Innovations.md 13.15）与随之修改的部分的单元测试。

    1. 配置与常数：默认值、非法值、mu_ref 的导出规则、resolve_background 的核对
    2. 编码器：与 numpy 逐窗参考实现（式 B1~B4）一致；因果（第 k 窗只看 0..k-1 窗的计数）；门控；先验；热像素
    3. 数据来源：numpy 与 torch 两种来源逐位相同；分段与整段相同；前 12 通道与不开背景时相同；不连续取窗报错；fresh 从先验开始
    4. 多流拼批：新录像（含 batchify 切开的后半截）的背景从先验开始，与网络状态的清零同步
    5. 网络：输入 16 通道、读出另读 4 个背景率；逐窗与逐片段等价；run_sequence 两种执行一致
    6. 滞后读出头：出生窗特征接背景率；推理、因果、训练路径与推理路径一致（复用 test_stream_lag 的检查）
    7. 自适应发布的选择规则（保住等待增益的比例 rho）
    8. 配置：新配置只差该差的键；端到端（带背景的骨干训练 -> 评估 -> 读出头训练 -> 评估与导出 -> 发布回放）
"""
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import stream_windows as sw  # noqa: E402
from dataset.background import (BACKGROUND_CHANNELS, BACKGROUND_DEFAULTS, BackgroundEncoder,  # noqa: E402
                                background_settings, input_channel_count, make_background_track, reference_rate,
                                resolve_background)
from dataset.stream_source import NumpyWindowSource, TorchWindowSource  # noqa: E402
from model.evspsegnet_stream import EvSpSegNetStream, trailing_channels_at_events  # noqa: E402
from model.lag_readout import LagReadout  # noqa: E402
import test_stream_lag as lag_tests  # noqa: E402  只导入模块：基类的测试不在本文件里重复运行
from test_stream_layerwise import CFG, H, Q99, SMALL, W, WINDOWS, synthetic_sequence  # noqa: E402
from test_stream_tif import ACTIVE_TIF  # noqa: E402

MU_REF = 0.01
# 合成序列只有 12 窗，保护区取 3 窗（默认 10 窗会让几乎所有窗都只有先验，测不到递推）
BG_CFG = dict(CFG, height=H, width_px=W, window_ms=50, background=dict(BACKGROUND_DEFAULTS, guard_windows=3),
              background_mu_ref=MU_REF)


def reference_codes(counts, settings, mu_ref, window_ms, height, width):
    """式 (B1)~(B5) 的 numpy float64 逐窗参考实现（直接按定义求和，不用递推）。counts [T,2,Hp,Wp] -> [T,4,Hp,Wp]。

    第 k 窗：S = sum_{j <= k-1-G} beta^(k-1-G-j) C_j，O 同理对 1[C_j > 0]，W = sum_{j <= k-1-G} beta^(k-1-G-j)。
    """
    steps, _, hp, wp = counts.shape
    beta, n0 = math.exp(-window_ms / settings["tau_ms"]), settings["prior_windows"]
    guard = settings["guard_windows"]
    cell, r = settings["cell_px"], settings["region_cells"] // 2
    unit = math.log1p(1.0 / mu_ref)
    valid = np.zeros((hp, wp))
    valid[:height, :width] = 1.0
    gy, gx = hp // cell, wp // cell
    out = np.zeros((steps, 4, hp, wp))
    for k in range(steps):
        last = k - 1 - guard
        weights = [beta ** (last - j) for j in range(0, last + 1)]
        s = sum((wj * counts[j] for j, wj in enumerate(weights)), np.zeros((2, hp, wp)))
        o = sum((wj * (counts[j] > 0) for j, wj in enumerate(weights)), np.zeros((2, hp, wp)))
        w = float(sum(weights))
        pixel = (s + n0 * mu_ref) / (w + n0)
        region = np.zeros((2, hp, wp))
        for cy in range(gy):
            for cx in range(gx):
                ys = slice(max(cy - r, 0) * cell, min(cy + r + 1, gy) * cell)
                xs = slice(max(cx - r, 0) * cell, min(cx + r + 1, gx) * cell)
                mean = o[:, ys, xs].sum(axis=(1, 2)) / valid[ys, xs].sum()
                region[:, cy * cell:(cy + 1) * cell, cx * cell:(cx + 1) * cell] = mean[:, None, None]
        region = (region + n0 * mu_ref) / (w + n0)
        code = np.log1p(np.concatenate([pixel, region]) / mu_ref) / unit
        out[k] = code * (counts[k].sum(0) > 0)
    return out


def sequence_counts(seq, steps=None):
    """每窗整窗正 / 负极性计数 [T,2,H,W]（float64）。"""
    steps = seq.n_windows if steps is None else steps
    out = []
    for k in range(steps):
        idx = seq.window_index(k)
        c = sw.count_channels(seq.x[idx], seq.y[idx], seq.p[idx], seq.inner_bin[idx], 5, H, W)
        out.append(c[:2].astype(np.float64))
    return np.stack(out)


def make_bg_net(seed=5, threshold=0.05):
    """float64 小 TIF 网络，输入 16 通道、读出读 4 个背景率（与 test_stream_tif.make_tif_net 同样调低阈值、放大增益）。"""
    torch.manual_seed(seed)
    net = EvSpSegNetStream(in_channels=12 + BACKGROUND_CHANNELS, channels=SMALL, neuron="tif", v_threshold=threshold,
                           tif=dict(ACTIVE_TIF), readout_background=BACKGROUND_CHANNELS).double()
    for block in net.blocks():
        block.gain.set_gain(torch.full((block.out_ch,), 1.0 / threshold, dtype=torch.float64))
    return net


class SettingsTests(unittest.TestCase):
    def test_settings(self):
        self.assertIsNone(background_settings(None))
        self.assertIsNone(background_settings(False))
        self.assertEqual(background_settings(True), BACKGROUND_DEFAULTS)
        self.assertEqual(background_settings({"tau_ms": 2000})["tau_ms"], 2000.0)
        self.assertEqual(background_settings({"guard_windows": 0})["guard_windows"], 0)
        for bad in ({"unknown": 1}, {"tau_ms": 0}, {"prior_windows": -1}, {"cell_px": 0}, {"region_cells": 4},
                    {"region_cells": 2.5}, {"cell_px": True}, {"guard_windows": -1}, {"guard_windows": 1.5},
                    "yes", 3):
            with self.assertRaises(ValueError, msg=str(bad)):
                background_settings(bad)
        self.assertEqual(input_channel_count(dict(time_bins=5)), 12)
        self.assertEqual(input_channel_count(dict(time_bins=5, background={})), 16)

    def test_reference_rate_and_resolve(self):
        cfg = dict(height=10, width_px=20, background={})
        stats = {"train_neg": 4000, "audit": {"train": {"windows": 50}}}
        self.assertAlmostEqual(reference_rate(stats, cfg), 4000 / (2 * 50 * 10 * 20), places=15)
        with self.assertRaises(ValueError):
            reference_rate({"train_neg": 10}, cfg)
        mu = resolve_background(cfg, stats)
        self.assertEqual(cfg["background_mu_ref"], mu)
        self.assertEqual(resolve_background(cfg, stats), mu)                   # 已有且一致
        cfg["background_mu_ref"] = mu * 1.01
        with self.assertRaises(ValueError):
            resolve_background(cfg, stats)
        off = dict(height=10, width_px=20)
        self.assertIsNone(resolve_background(off, stats))
        self.assertNotIn("background_mu_ref", off)
        with self.assertRaises(ValueError):
            resolve_background(dict(off, background_mu_ref=0.1), stats)
        with self.assertRaises(ValueError):
            make_background_track(dict(BG_CFG, background_mu_ref=None), torch.device("cpu"))


class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.settings = background_settings({"tau_ms": 400.0, "guard_windows": 2})   # 短时间常数与保护区，让 10 窗内变化明显
        self.height, self.width = 13, 21                                     # 画布 16x24，下侧 3 行、右侧 3 列是补零区
        rng = np.random.RandomState(0)
        counts = (rng.rand(10, 2, H, W) < 0.15) * rng.randint(1, 4, (10, 2, H, W))
        counts[:, :, self.height:, :] = 0
        counts[:, :, :, self.width:] = 0
        counts[:, 0, 4, 5] = 6                                               # 热像素
        self.counts = counts.astype(np.float64)
        self.encoder = BackgroundEncoder(self.settings, MU_REF, 50, self.height, self.width, H, W, torch.device("cpu"))

    def encode(self, counts, settings=None):
        encoder = self.encoder if settings is None else BackgroundEncoder(
            settings, MU_REF, 50, self.height, self.width, H, W, torch.device("cpu"))
        state, out = encoder.fresh(), []
        for k in range(counts.shape[0]):
            code, state = encoder.step(torch.from_numpy(counts[k].astype(np.float32)), state)
            out.append(code.numpy().astype(np.float64))
        return np.stack(out)

    def test_matches_reference(self):
        for guard in (0, 2, 4):
            settings = dict(self.settings, guard_windows=guard)
            got = self.encode(self.counts, settings)
            want = reference_codes(self.counts, settings, MU_REF, 50, self.height, self.width)
            self.assertLess(float(np.abs(got - want).max()), 1e-5, guard)
            self.assertGreater(float(np.abs(want).max()), 0.1)

    def test_guard_excludes_passing_target_and_region_uses_occupancy(self):
        counts = np.zeros((8, 2, H, W))
        counts[:, 0, 1, 1] = 1                                               # 常亮的参照像素：每窗都有事件，B 每窗都可读
        counts[3:6, 0, 6, 9] = 4                                             # 目标在第 3~5 窗经过像素 (6, 9)
        counts[6, 0, 6, 9] = 1                                               # 第 6 窗在同一像素出现一个事件
        counts[6, 0, 7, 2] = 1                                               # 对照像素：之前从没有事件
        with_guard = self.encode(counts, dict(self.settings, guard_windows=3))
        no_guard = self.encode(counts, dict(self.settings, guard_windows=0))
        # 第 6 窗只用第 0~2 窗：目标不在里面，与从没有事件的对照像素相同
        self.assertAlmostEqual(with_guard[6, 0, 6, 9], with_guard[6, 0, 7, 2], places=6)
        self.assertGreater(no_guard[6, 0, 6, 9], 3 * no_guard[6, 0, 7, 2])   # 不设保护区：被目标自己抬高
        # 区域通道按占有率：一个每窗 50 个事件的热像素与一个每窗 1 个事件的像素，对区域的贡献相同
        hot = np.zeros((4, 2, H, W))
        hot[:, 0, 2, 2] = 50
        one = np.zeros((4, 2, H, W))
        one[:, 0, 2, 2] = 1
        a = self.encode(hot, dict(self.settings, guard_windows=0))
        b = self.encode(one, dict(self.settings, guard_windows=0))
        self.assertAlmostEqual(a[3, 2, 2, 2], b[3, 2, 2, 2], places=6)
        self.assertGreater(a[3, 0, 2, 2], b[3, 0, 2, 2])                     # 逐像素通道仍按计数

    def test_prior_gate_and_hot_pixel(self):
        got = self.encode(self.counts)
        active = self.counts.sum(1) > 0                                      # [T,H,W]
        for c in range(4):
            self.assertTrue(np.all(got[:, c][~active] == 0.0))               # 没有事件的像素严格为 0
        prior = math.log(2.0) / math.log1p(1.0 / MU_REF)                     # 第 0..G 窗只有先验：mu = mu_ref
        for k in range(self.settings["guard_windows"] + 1):
            self.assertTrue(np.allclose(got[k][:, active[k]], prior, atol=1e-6), k)
        self.assertGreater(got[-1, 0, 4, 5], got[-1, 2, 4, 5])               # 热像素：逐像素背景率高于区域

    def test_causality(self):
        base = self.encode(self.counts)
        guard = self.settings["guard_windows"]
        for k in (3, 5):
            later = self.counts.copy()
            later[k:] = np.where(later[k:] > 0, later[k:] + 5, 0)            # 第 k 窗起只改数值、不改有无事件
            got = self.encode(later)
            self.assertTrue(np.array_equal(base[:k + guard + 1], got[:k + guard + 1]), k)   # 保护区内看不到
            self.assertFalse(np.array_equal(base[k + guard + 1:], got[k + guard + 1:]), k)


class SourceTests(unittest.TestCase):
    def test_sources_match_and_sequential_rule(self):
        device = torch.device("cpu")
        seq = synthetic_sequence(seed=2)
        plain = NumpyWindowSource(seq, CFG, Q99, device).chunk(0, WINDOWS)[0]
        a = NumpyWindowSource(seq, BG_CFG, Q99, device).chunk(0, WINDOWS)
        b = TorchWindowSource(seq, BG_CFG, Q99, device).chunk(0, WINDOWS)
        torch_src = TorchWindowSource(seq, BG_CFG, Q99, device)
        c = [torch_src.chunk(s, e) for s, e in ((0, 3), (3, 8), (8, WINDOWS))]
        self.assertEqual(tuple(a[0].shape), (WINDOWS, 1, 16, H, W))
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(b[0], torch.cat([part[0] for part in c])))
        self.assertTrue(torch.equal(a[0][:, :, :12], plain))                  # 前 12 通道与不开背景时相同
        for key in ("t", "y", "x", "p", "t_local"):
            self.assertTrue(torch.equal(a[1][key], b[1][key]), key)
        want = reference_codes(sequence_counts(seq), background_settings(BG_CFG["background"]), MU_REF, 50, H, W)
        self.assertLess(float(np.abs(a[0][:, 0, 12:].double().numpy() - want).max()), 1e-5)
        step = NumpyWindowSource(seq, BG_CFG, Q99, device)                    # 逐窗取与整段取相同
        self.assertTrue(torch.equal(torch.stack([step.window(k)[0] for k in range(WINDOWS)]), a[0]))
        with self.assertRaises(RuntimeError):                                 # 不连续取窗
            TorchWindowSource(seq, BG_CFG, Q99, device).chunk(5, 8)
        with self.assertRaises(RuntimeError):
            NumpyWindowSource(seq, BG_CFG, Q99, device).window(4)
        fresh = TorchWindowSource(seq, BG_CFG, Q99, device).chunk(5, 9, fresh=True)[0][:, 0, 12:]
        want5 = reference_codes(sequence_counts(seq)[5:9], background_settings(BG_CFG["background"]), MU_REF, 50, H, W)
        self.assertLess(float(np.abs(fresh.double().numpy() - want5).max()), 1e-5)
        from dataset.ev_uav_stream import window_to_device
        with self.assertRaises(ValueError):
            window_to_device(seq, 0, Q99, BG_CFG, device)

    def test_assemble_streams_resets_background_with_network_state(self):
        import train_stream_v1 as T
        device = torch.device("cpu")
        seqs = [synthetic_sequence(seed=s) for s in (3, 4, 5)]
        # 流 0：录像 0 整段后接录像 2 的后半截（第 4 窗起，batchify 式切开，该处清零）；流 1：录像 1 的前 7 窗，延迟 2 窗
        plan = [(0, [(0, WINDOWS), (2, 4, WINDOWS - 4)]), (2, [(1, 0, 7)])]
        sources = T.StreamSources(seqs, BG_CFG, Q99, device, torch.float32)
        k, total = 5, T.stream_total_windows(plan)
        xs = []
        for c in range(int(math.ceil(total / float(k)))):
            pieces = T.stream_pieces(plan, c * k, (c + 1) * k)
            x, _, _, keep = T.assemble_streams(sources, pieces, k, BG_CFG, device, torch.float32)
            self.assertEqual(int(x.shape[2]), 16)
            xs.append(x)
        x = torch.cat(xs)
        settings = background_settings(BG_CFG["background"])
        counts = [sequence_counts(s) for s in seqs]
        expect0 = np.concatenate([reference_codes(counts[0], settings, MU_REF, 50, H, W),
                                  reference_codes(counts[2][4:], settings, MU_REF, 50, H, W)])
        expect1 = reference_codes(counts[1][:7], settings, MU_REF, 50, H, W)
        got0 = x[:2 * WINDOWS - 4, 0, 12:].double().numpy()
        got1 = x[2:9, 1, 12:].double().numpy()
        self.assertLess(float(np.abs(got0 - expect0).max()), 1e-5)
        self.assertLess(float(np.abs(got1 - expect1).max()), 1e-5)
        self.assertTrue(torch.all(x[:2, 1] == 0))                             # 流 1 开头的空窗


class NetworkTests(unittest.TestCase):
    def test_trailing_channels_and_validation(self):
        x4 = torch.arange(2 * 6 * 4 * 5, dtype=torch.float64).view(2, 6, 4, 5)
        events = {"b": torch.tensor([1, 0]), "y": torch.tensor([2, 3]), "x": torch.tensor([4, 0])}
        got = trailing_channels_at_events(x4, events, 2)
        self.assertTrue(torch.equal(got, torch.stack([x4[1, 4:, 2, 4], x4[0, 4:, 3, 0]])))
        x5 = torch.arange(3 * 2 * 6 * 4 * 5, dtype=torch.float64).view(3, 2, 6, 4, 5)
        got5 = trailing_channels_at_events(x5, dict(events, t=torch.tensor([2, 1])), 3)
        self.assertTrue(torch.equal(got5, torch.stack([x5[2, 1, 3:, 2, 4], x5[1, 0, 3:, 3, 0]])))
        for bad in (-1, True, 16, 1.5):
            with self.assertRaises(ValueError):
                EvSpSegNetStream(in_channels=16, channels=SMALL, neuron="tif", readout_background=bad)
        net = EvSpSegNetStream(channels=SMALL, neuron="tif")
        self.assertEqual(net.readout[0].in_features, SMALL[0] + 2)            # 默认不读背景，与 V5 相同

    def test_build_net_and_step_layer_equivalence(self):
        import train_stream_v1 as T
        cfg = dict(BG_CFG, channels=list(SMALL), neuron="tif", tif=dict(ACTIVE_TIF), norm="none", state_mode="carry",
                   readout_hidden=8, tau_init_ms=200.0, tau_min_ms=50.0, tau_max_ms=2000.0, v_threshold=1.0)
        built = T.build_net(cfg, torch.device("cpu"))
        self.assertEqual(built.enc1.conv.in_channels, 16)
        self.assertEqual(built.readout[0].in_features, SMALL[0] + 2 + 4)
        self.assertEqual(T.design_version(built), "v6-tif-background")
        net = make_bg_net().eval()
        seq = synthetic_sequence(seed=9)
        with torch.no_grad():
            step, conf_s = T.run_sequence(net, seq, dict(BG_CFG, execution="step", input_device="cpu"), Q99,
                                          torch.device("cpu"), "carry")
            layer, conf_l = T.run_sequence(net, seq, dict(BG_CFG, execution="layer", input_device="gpu",
                                                          eval_chunk=5), Q99, torch.device("cpu"), "carry")
        self.assertLess(float(np.abs(step - layer).max()), 1e-6)
        self.assertGreater(float(step.std()), 0.0)
        # 读出确实读到背景率：把背景通道置零，概率改变
        source = NumpyWindowSource(seq, BG_CFG, Q99, torch.device("cpu"), torch.float64)
        x, events, _, _ = source.chunk(0, WINDOWS)
        with torch.no_grad():
            logits, _, _ = net.forward_chunk(x, events, None)
            x0 = x.clone()
            x0[:, :, 12:] = 0
            logits0, _, _ = net.forward_chunk(x0, events, None)
        self.assertFalse(torch.equal(logits, logits0))


class BackgroundLagInferenceTests(lag_tests.InferenceTests):
    """test_stream_lag.InferenceTests 的全部检查，骨干换成带背景率通道的网络。"""

    def setUp(self):
        import train_stream_lag as L
        self.L = L
        self.cfg = dict(BG_CFG, input_device="cpu", execution="step", lags=lag_tests.LAGS, lag_eval_chunk=5)
        self.net = make_bg_net()
        self.net.eval()
        self.readouts = lag_tests.random_readouts(birth_extra=BACKGROUND_CHANNELS)

    def test_birth_dimension(self):
        r = LagReadout(SMALL, 3, "multiscale", birth_extra=4)
        self.assertEqual(r.birth_dim, SMALL[0] + 4)
        self.assertEqual(r.heads[0][0].in_features, r.now_dim + SMALL[0] + 4 + 3)
        with self.assertRaises(ValueError):
            LagReadout(SMALL, 3, "pixel", birth_extra=-1)


class PublishRuleTests(unittest.TestCase):
    def test_retain_rule(self):
        from tools import lag_publish as P
        rng = np.random.RandomState(1)
        n = 4000
        label = (rng.rand(n) < 0.3).astype(np.float32)
        noise = rng.randn(3, n)
        # 年龄越大分数越准：d = 2 的 IoU 最高
        prob = np.stack([1.0 / (1.0 + np.exp(-(3.0 * (2 * label - 1) * s + 1.5 * noise[d] + 2.2)))
                         for d, s in enumerate((0.6, 0.9, 1.2))]).astype(np.float32)
        seqs = [{"label": label, "avail": np.full(n, 2, dtype=np.int64)}]
        deadline, margin, sel = P.choose_rule(prob, seqs, 0.9, P.DEFAULT_MARGINS, 0.9)
        fixed = sel["fixed_iou"]
        self.assertEqual(deadline, int(np.argmax(fixed)))
        self.assertAlmostEqual(sel["target_iou"], fixed[0] + 0.9 * (fixed[deadline] - fixed[0]), places=12)
        ious = [row["iou"] for row in sel["margins"]]
        first = next(i for i, v in enumerate(ious) if v >= sel["target_iou"])
        self.assertEqual(sel["margin"], P.DEFAULT_MARGINS[first] if math.isfinite(P.DEFAULT_MARGINS[first]) else "inf")
        _, m0, _ = P.choose_rule(prob, seqs, 0.9, P.DEFAULT_MARGINS, 0.0)       # rho = 0：零等待
        self.assertEqual(m0, 0.0)
        flat = np.repeat(prob[:1], 3, axis=0)                                   # 等待没有增益：D* = 0
        d_flat, m_flat, _ = P.choose_rule(flat, seqs, 0.9, P.DEFAULT_MARGINS, 0.9)
        self.assertEqual((d_flat, m_flat), (0, 0.0))
        for bad in ((P.DEFAULT_MARGINS, 1.5), ((1.0, 0.5), 0.9), ((-1.0, 0.0), 0.9)):
            with self.assertRaises(ValueError):
                P.choose_rule(prob, seqs, 0.9, bad[0], bad[1])


class ConfigTests(unittest.TestCase):
    def test_background_configs(self):
        import train_stream_lag as L
        from utils.stream_common import load_flat_config
        for new, old in (("evisseg_stream_tif_bg", "evisseg_stream_tif_comm_peak"),
                         ("evisseg_stream_tif_bg_evflying", "evisseg_stream_tif_comm_peak_evflying")):
            a = load_flat_config("configs/%s.yaml" % new)
            b = load_flat_config("configs/%s.yaml" % old)
            self.assertEqual(set(a) - set(b), {"background"})
            self.assertEqual({k for k in b if a[k] != b[k]}, {"save_root"})
            self.assertEqual(background_settings(a["background"]), BACKGROUND_DEFAULTS)
            self.assertNotIn("background_mu_ref", a)                           # 由统计导出，不写进配置
        la = load_flat_config("configs/evisseg_stream_lag_bg.yaml")
        lb = load_flat_config("configs/evisseg_stream_lag_bg_evflying.yaml")
        l5 = load_flat_config("configs/evisseg_stream_lag.yaml")
        self.assertEqual({k for k in l5 if la[k] != l5[k]},
                         {"base_config", "backbone_checkpoint", "lag_variants", "lag_save_root"})
        self.assertEqual({k for k in la if la[k] != lb[k]},
                         {"base_config", "backbone_checkpoint", "lag_epochs", "lag_save_root"})
        for lag in (la, lb):
            cfg = L.build_config("configs/%s.yaml" % ("evisseg_stream_lag_bg" if lag is la
                                                      else "evisseg_stream_lag_bg_evflying"))
            base = load_flat_config(lag["base_config"])
            self.assertEqual(os.path.dirname(lag["backbone_checkpoint"]), base["save_root"])
            self.assertEqual(cfg["lag_variants"], ["multiscale"])
            self.assertIsNotNone(background_settings(base.get("background")))


class EndToEndTests(unittest.TestCase):
    def test_background_backbone_lag_publish(self):
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
            import yaml
        except ImportError as error:
            self.skipTest("完整入口需要原 V1 的 cv2/pandas/tqdm 依赖: %s" % error)
        import train_stream_lag as L
        import train_stream_v1 as T
        from tools import lag_publish as P
        from utils.stream_common import load_flat_config
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pos = neg = 0
            for split, seeds in (("train", (3, 4, 5)), ("val", (6, 7))):
                (root / split).mkdir()
                for seed in seeds:
                    seq = synthetic_sequence(seed=seed)
                    loc = np.stack((seq.x, seq.y, seq.t), axis=1)
                    norm = np.column_stack((loc, seq.p, seq.label, seq.target_id))
                    np.savez(root / split / ("seq_%d.npz" % seed), ev_loc=loc, evs_norm=norm)
                    if split == "train":
                        pos += int(seq.label.sum())
                        neg += seq.n_events - int(seq.label.sum())
            stats = dict(q99=Q99.tolist(), train_pos=pos, train_neg=neg, pos_weight=min(neg / pos, 30.0),
                         audit={"train": {"windows": 3 * WINDOWS}})
            (root / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
            run = root / "backbone"
            cfg = load_flat_config("configs/evisseg_stream_tif_bg.yaml")
            cfg.update(root=str(root), stats_path=str(root / "stats.json"), save_root=str(run),
                       channels=list(SMALL), height=H, width_px=W, pad_height=H, pad_width=W,
                       n_windows=WINDOWS, epochs=2, num_workers=0, tbptt_k=4, train_subset_size=1,
                       calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
                       segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=4,
                       execution="layer", input_device="gpu", streams=2)
            T.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
            mu = neg / (2.0 * 3 * WINDOWS * H * W)
            self.assertAlmostEqual(cfg["background_mu_ref"], mu, places=15)
            checkpoint = run / "best_val_iou_seed37.pt"
            ckpt = torch.load(str(checkpoint), map_location="cpu")
            self.assertEqual(ckpt["design_version"], "v6-tif-background")
            self.assertAlmostEqual(ckpt["config"]["background_mu_ref"], mu, places=15)
            v5_args = SimpleNamespace(checkpoint=str(checkpoint), device="cpu", mode="eval", split="val",
                                      overwrite=True, execution=None, input_device=None, readout_ablation="none",
                                      subthreshold_ablation="none", memory_ablation="none", memory_off_layers=None,
                                      reset_every=0, reset_offset=0, dump_dir=None)
            T.mode_eval(v5_args, cfg)
            reference = run / "eval_val_best_val_iou_seed37.json"
            carry = json.loads(reference.read_text(encoding="utf-8"))["carry"]
            self.assertEqual([s["name"] for s in carry["per_sequence"]], ["seq_6.npz", "seq_7.npz"])
            self.assertEqual(sum(s["tp"] for s in carry["per_sequence"]), sum(carry["per_window"]["tp"]))
            base_path, lag_path = root / "base.yaml", root / "lag.yaml"
            base_path.write_text(yaml.safe_dump({"ALL": cfg}), encoding="utf-8")
            lag = {"base_config": str(base_path), "backbone_checkpoint": str(checkpoint),
                   "lag_variants": ["multiscale"], "lags": 2, "lag_hidden": 8, "lag_epochs": 2,
                   "lag_lr": 0.01, "lag_lr_end": 0.001, "lag_grad_clip": 1.0, "lag_streams": 2, "lag_chunk": 4,
                   "lag_eval_chunk": 4, "lag_save_root": str(root / "lag")}
            lag_path.write_text(yaml.safe_dump({"LAG": lag}), encoding="utf-8")
            lcfg = L.build_config(str(lag_path))
            L.mode_smoke(SimpleNamespace(device="cpu"), lcfg)
            L.mode_train(SimpleNamespace(device="cpu"), lcfg)
            saved = torch.load(str(root / "lag" / "best_lag_multiscale.pt"), map_location="cpu")
            self.assertEqual(saved["birth_extra"], BACKGROUND_CHANNELS)
            L.mode_eval(SimpleNamespace(device="cpu", split="val", dump=True, reference=str(reference),
                                        overwrite=True), lcfg)
            result = json.loads((root / "lag" / "eval_lag_val.json").read_text(encoding="utf-8"))
            self.assertTrue(all(item["equal"] for item in result["d0_vs_v5"].values()), result["d0_vs_v5"])
            prob, seqs, meta = P.load_dump(str(root / "lag" / "dump_val_multiscale.npz"))
            deadline, margin, selection = P.choose_rule(prob, seqs, meta["threshold"], P.DEFAULT_MARGINS, 0.9)
            cfgp = {k: meta[k] for k in ("threshold", "pd_detT", "correct_thresh", "height", "width_px")}
            report = P.apply_rule(prob, seqs, cfgp, deadline, margin)
            self.assertGreaterEqual(report["iou"], selection["target_iou"] - 1e-12)


if __name__ == "__main__":
    unittest.main()
