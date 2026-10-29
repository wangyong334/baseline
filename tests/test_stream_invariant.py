"""模块二：不变输入接口（model/noise_invariant.py）的测试（CPU、合成数据，不需要数据集）。

    1. 查表与精确泊松矩一致
    2. 透明：背景处处不高于参照时，接口输出与输入逐位相同
    3. 不变性（核心性质）：两种明显不同的高背景率下，归一化后的背景电流均值、方差都回到参照工作点（蒙特卡洛）
    4. 背景跟踪器：因果（第 k 窗只用之前的计数）、按顺序请求与跳回重算结果相同
    5. 网络：开启接口时，逐窗 forward 与逐层 forward_chunk 数学等价；缺少 bg 时接口原样透传
    6. 数据源与配置：events 带 bg；E3a / E3b 配置与基础配置只差接口开关与输出目录
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import stream_windows as sw  # noqa: E402
from dataset.stream_source import make_window_source  # noqa: E402
from model.comod import REF_RATE, input_moments  # noqa: E402
from model.evspsegnet_stream import EvSpSegNetStream, SpikingConvBlock  # noqa: E402
from model.noise_invariant import BackgroundTracker, InvariantInterface, MomentTable  # noqa: E402
from test_stream_layerwise import CFG, H, Q99, SMALL, W, WINDOWS, synthetic_sequence  # noqa: E402
from test_stream_tif import ACTIVE_TIF  # noqa: E402

CLIP = 3.0


def make_block(seed=0):
    torch.manual_seed(seed)
    lif = dict(dt_ms=50.0, tau_init_ms=200.0, tau_min_ms=50.0, tau_max_ms=2000.0, v_threshold=1.0)
    block = SpikingConvBlock(12, 6, 1, "tif", "none", lif, {"levels": 7, "quantize": True, "k_init": 0.05})
    block.gain.set_gain(torch.full((6,), 2.0))
    iface = InvariantInterface(5, CLIP)
    iface.set_q99(Q99)
    block.invariant = iface
    return block, iface


def encode(counts_bins, q99, clip):
    """时间箱计数 [N,2,T,H,W] -> 12 通道输入（与 stream_windows 的通道布局一致）。"""
    whole = counts_bins.sum(2)                                         # [N,2,H,W]
    T = counts_bins.shape[2]
    chans = [whole[:, 0], whole[:, 1]] + [counts_bins[:, p, t] for t in range(T) for p in (0, 1)]
    x = torch.stack(chans, 1).float()
    q = torch.as_tensor(np.asarray(q99, dtype=np.float32)).view(1, -1, 1, 1)
    return torch.clamp(torch.log1p(x) / q, 0.0, clip)


class TableTests(unittest.TestCase):
    def test_lookup_matches_exact(self):
        table = MomentTable(Q99, 5, CLIP)
        rng = np.random.RandomState(0)
        b = torch.from_numpy(np.exp(rng.uniform(np.log(1e-4), np.log(3.0), size=(2, 2, 3, 4))))
        m0, v0, c0 = input_moments(b, Q99, 5, CLIP)
        m1, v1, c1 = table.lookup(b)
        self.assertLess(float((m0.float() - m1).abs().max()), 1e-4)
        self.assertLess(float((v0.float() - v1).abs().max()), 1e-4)
        self.assertLess(float((c0.float() - c1).abs().max()), 1e-4)


class InterfaceTests(unittest.TestCase):
    def test_transparent_at_or_below_reference(self):
        block, iface = make_block()
        x = torch.rand(3, 12, 9, 11)
        current = block.gain(block.conv(x))
        bg = torch.tensor(REF_RATE).view(1, 2, 1, 1).expand(3, 2, 9, 11) * torch.rand(3, 1, 9, 11)
        self.assertTrue(torch.equal(iface.normalize(current, bg, block), current))
        self.assertTrue(torch.equal(iface.normalize(current, None, block), current))

    def test_background_current_is_invariant_to_rate(self):
        block, iface = make_block(1)
        Hs, Ws, N, T = 7, 7, 3000, 5
        torch.manual_seed(2)
        stats = []
        for rate in (0.2, 1.0):                                        # 两种都远高于参照率、彼此差 5 倍
            counts = torch.poisson(torch.full((N, 2, T, Hs, Ws), rate / T))
            x = encode(counts, Q99, CLIP)
            with torch.no_grad():
                current = block.gain(block.conv(x))
                bg = torch.full((N, 2, Hs, Ws), rate)
                out = iface.normalize(current, bg, block)
            centre = out[:, :, Hs // 2, Ws // 2]                       # 内部像素：卷积不碰边界
            stats.append((centre.mean(0), centre.var(0), current[:, :, Hs // 2, Ws // 2].var(0)))
            b_ref = torch.tensor(REF_RATE, dtype=torch.float32).view(1, 2, 1, 1).expand(1, 2, Hs, Ws).contiguous()
            mu_ref, var_ref = iface.current_moments(b_ref, block)
        (m1, v1, raw1), (m2, v2, raw2) = stats
        self.assertTrue(bool((raw2 > 2.0 * raw1).all()))                # 原始电流的方差随背景明显增大
        ref_mu, ref_var = mu_ref[0, :, Hs // 2, Ws // 2], var_ref[0, :, Hs // 2, Ws // 2]
        tol_mu = 4.0 * torch.sqrt(ref_var / N) + 1e-3
        for m, v in ((m1, v1), (m2, v2)):                               # 归一化后两种背景都回到参照工作点
            self.assertTrue(bool(((m - ref_mu).abs() <= tol_mu).all()), (m, ref_mu))
            self.assertTrue(bool(((v / ref_var - 1.0).abs() < 0.2).all()), (v, ref_var))


class TrackerTests(unittest.TestCase):
    def cfg(self):
        return dict(CFG, pad_height=H, pad_width=W, invariant_input=True, height=H, width_px=W, window_ms=50)

    def test_causal_and_restart(self):
        seq = synthetic_sequence(3)
        a = BackgroundTracker(seq, self.cfg(), torch.device("cpu"))
        full = a.background(0, WINDOWS)
        self.assertEqual(tuple(full.shape), (WINDOWS, 2, H, W))
        self.assertEqual(float(full[0].abs().sum()), 0.0)              # 第 0 窗之前没有历史
        b = BackgroundTracker(seq, self.cfg(), torch.device("cpu"))
        parts = torch.cat([b.background(0, 4), b.background(4, 9), b.background(9, WINDOWS)])
        self.assertTrue(torch.allclose(full, parts))
        again = b.background(2, 6)                                     # 跳回：从头重算，结果相同
        self.assertTrue(torch.allclose(again, full[2:6]))
        # 因果：改动第 6 窗的事件，不影响第 0..6 窗的估计
        seq2 = synthetic_sequence(3)
        idx = seq2.window_index(6)
        seq2.x[idx] = (seq2.x[idx] + 5) % W
        c = BackgroundTracker(seq2, self.cfg(), torch.device("cpu")).background(0, WINDOWS)
        self.assertTrue(torch.allclose(c[:7], full[:7]))
        self.assertFalse(torch.allclose(c[7:], full[7:]))

    def test_sources_add_bg(self):
        seq = synthetic_sequence(4)
        for device_kind in ("cpu", "gpu"):
            cfg = dict(self.cfg(), input_device=device_kind, input_clip=CLIP)
            src = make_window_source(seq, cfg, Q99, torch.device("cpu"), torch.float64)
            _, ev, _, _ = src.chunk(0, 4)
            self.assertEqual(tuple(ev["bg"].shape), (4, 1, 2, H, W))
            _, ev1, _, _ = src.window(4)
            self.assertEqual(tuple(ev1["bg"].shape), (1, 2, H, W))
        plain = make_window_source(seq, dict(CFG, pad_height=H, pad_width=W, input_clip=CLIP), Q99,
                                   torch.device("cpu"))
        self.assertNotIn("bg", plain.window(0)[1])


class NetworkTests(unittest.TestCase):
    def test_step_equals_chunk_with_interface(self):
        torch.manual_seed(5)
        net = EvSpSegNetStream(channels=SMALL, neuron="tif", v_threshold=0.05, tif=dict(ACTIVE_TIF),
                               invariant=dict(time_bins=5, clip=CLIP)).double()
        for block in net.blocks():
            block.gain.set_gain(torch.full((block.out_ch,), 20.0, dtype=torch.float64))
        net.invariant.set_q99(Q99)
        net.eval()
        seq = synthetic_sequence(6, n=4000)                            # 较密：背景高于参照，接口真正起作用
        cfg = dict(CFG, pad_height=H, pad_width=W, invariant_input=True, height=H, width_px=W, window_ms=50,
                   input_clip=CLIP)
        with torch.no_grad():
            src = make_window_source(seq, cfg, Q99, torch.device("cpu"), torch.float64)
            states, step_logits = None, []
            for k in range(WINDOWS):
                x, ev, _, _ = src.window(k)
                logits, states, _ = net(x, ev, states)
                step_logits.append(logits)
            src2 = make_window_source(seq, cfg, Q99, torch.device("cpu"), torch.float64)
            x, ev, _, _ = src2.chunk(0, WINDOWS)
            chunk_logits, _, _ = net.forward_chunk(x, ev, None)
        self.assertLess(float((torch.cat(step_logits) - chunk_logits).abs().max()), 1e-6)
        # 接口确实改变了输出（不是恒等）
        with torch.no_grad():
            ev_no = {k: v for k, v in ev.items() if k != "bg"}
            plain_logits, _, _ = net.forward_chunk(x, ev_no, None)
        self.assertGreater(float((plain_logits - chunk_logits).abs().max()), 1e-6)

    def test_q99_saved_in_state_dict(self):
        net = EvSpSegNetStream(channels=SMALL, neuron="tif", tif=dict(ACTIVE_TIF), invariant=dict(time_bins=5, clip=CLIP))
        net.invariant.set_q99(Q99)
        restored = EvSpSegNetStream(channels=SMALL, neuron="tif", tif=dict(ACTIVE_TIF),
                                    invariant=dict(time_bins=5, clip=CLIP))
        restored.load_state_dict(net.state_dict())
        np.testing.assert_allclose(restored.invariant.q99.numpy(), Q99, rtol=1e-6)


class EndToEndTests(unittest.TestCase):
    def test_train_and_eval_with_interface(self):
        # 校准（带 bg）、多流训练（拼批带 bg）、checkpoint（含 q99 缓冲区）、正式评估（含测试时注入噪声）
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
        except ImportError as error:
            self.skipTest("完整训练入口需要 cv2/pandas/tqdm: %s" % error)
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        import train_stream_v1 as T
        from utils.stream_common import load_flat_config
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pos = neg = 0
            for split, seeds in (("train", (3, 4, 5)), ("val", (6,))):
                (root / split).mkdir()
                for seed in seeds:
                    seq = synthetic_sequence(seed=seed, n=3000)
                    loc = np.stack((seq.x, seq.y, seq.t), axis=1)
                    norm = np.column_stack((loc, seq.p, seq.label, seq.target_id))
                    np.savez(root / split / ("seq_%d.npz" % seed), ev_loc=loc, evs_norm=norm)
                    if split == "train":
                        pos += int(seq.label.sum())
                        neg += seq.n_events - int(seq.label.sum())
            stats = dict(q99=Q99.tolist(), train_pos=pos, train_neg=neg, pos_weight=min(neg / pos, 30.0))
            (root / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
            for config in ("evisseg_stream_tif_comm_peak_invariant", "evisseg_stream_tif_comm_peak_invariant_noiseaug"):
                run = root / config
                cfg = load_flat_config("configs/%s.yaml" % config)
                cfg.update(root=str(root), stats_path=str(root / "stats.json"), save_root=str(run),
                           channels=list(SMALL), height=H, width_px=W, pad_height=H, pad_width=W,
                           n_windows=12, epochs=2, num_workers=0, tbptt_k=4, train_subset_size=1,
                           calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
                           segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=4,
                           execution="layer", input_device="gpu", streams=2)
                T.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
                checkpoint = run / "best_val_iou_seed37.pt"
                ckpt = torch.load(str(checkpoint), map_location="cpu")
                np.testing.assert_allclose(ckpt["model"]["enc1.invariant.q99"].numpy(), Q99, rtol=1e-6)
                for load in (0.0, 2.0):
                    args = SimpleNamespace(checkpoint=str(checkpoint), device="cpu", mode="eval", split="val",
                                           overwrite=True, execution=None, input_device=None, readout_ablation="none",
                                           subthreshold_ablation="none", memory_ablation="none",
                                           memory_off_layers=None, reset_every=0, reset_offset=0, dump_dir=None,
                                           eval_noise_load=load, eval_noise_seed=0)
                    T.mode_eval(args, cfg)
                    name = "eval_val_best_val_iou_seed37%s.json" % ("" if load == 0 else "_noiseL2_seed0")
                    result = json.loads((run / name).read_text(encoding="utf-8"))
                    self.assertEqual(result["eval_noise_load"], load)
                    self.assertTrue(np.isfinite(result["carry"]["iou"]))


class ConfigTests(unittest.TestCase):
    def test_e3_configs(self):
        from utils.stream_common import load_flat_config
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = load_flat_config(os.path.join(root, "configs", "evisseg_stream_tif_comm_peak.yaml"))
        aug = load_flat_config(os.path.join(root, "configs", "evisseg_stream_tif_comm_peak_noiseaug.yaml"))
        for name, ref, out in (("evisseg_stream_tif_comm_peak_invariant.yaml", base, "log/stream_tif_commpeak_inv_seed37"),
                               ("evisseg_stream_tif_comm_peak_invariant_noiseaug.yaml", aug,
                                "log/stream_tif_commpeak_inv_noiseaug_seed37")):
            cfg = load_flat_config(os.path.join(root, "configs", name))
            self.assertEqual(set(cfg) - set(ref), {"invariant_input"}, name)
            self.assertTrue(cfg["invariant_input"])
            for key in ref:
                if key != "save_root":
                    self.assertEqual(cfg[key], ref[key], (name, key))
            self.assertEqual(cfg["save_root"], out)


class WideConfigTests(unittest.TestCase):
    def test_wide_configs(self):
        from utils.stream_common import load_flat_config
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pairs = (("evisseg_stream_tif_comm_peak_noiseaug_wide", "evisseg_stream_tif_comm_peak_noiseaug",
                  "log/stream_tif_commpeak_noiseaug_wide_seed37"),
                 ("evisseg_stream_tif_comm_peak_invariant_noiseaug_wide", "evisseg_stream_tif_comm_peak_invariant_noiseaug",
                  "log/stream_tif_commpeak_inv_noiseaug_wide_seed37"))
        for wide, base, out in pairs:
            w = load_flat_config(os.path.join(root, "configs", wide + ".yaml"))
            b = load_flat_config(os.path.join(root, "configs", base + ".yaml"))
            self.assertEqual(set(w), set(b), wide)
            for key in b:
                if key not in ("save_root", "noise_aug_rate_max"):
                    self.assertEqual(w[key], b[key], (wide, key))
            self.assertEqual((w["noise_aug_rate_max"], w["save_root"]), (0.12, out))

if __name__ == "__main__":
    unittest.main()
