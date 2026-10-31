"""细节 D1：partial-conv 补零（NET.border = partial）的测试。

全部在 CPU 上用 float64 小网络运行，不需要数据集：
    1. 比例图的数值：内部为 1，传感器边缘按有效输入数放大，传感器外为 0；各层尺寸与输出电流一致
    2. 默认 border = zero 与以往逐位相同；partial 不消耗随机数、比例图不进 state_dict，新旧权重互相可加载
    3. 卷积块的电流 = 普通补零的电流 x 比例图
    4. 逐窗 forward 与逐层 forward_chunk 等价（LIF / TIF），传感器外的 enc1 输出为 0
    5. 模块二的背景矩与比例图一致（均值 x r，方差 x r^2）
    6. 配置只差 border / seed / save_root；build_net 按配置构造；带 partial 的小网络端到端训练与评估
"""
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.stream_source import TorchWindowSource  # noqa: E402
from model.comod import REF_RATE  # noqa: E402
from model.evspsegnet_stream import LAYER_NAMES, EvSpSegNetStream, border_ratio_maps  # noqa: E402
from test_stream_layerwise import CFG, CHUNKS, H, Q99, SMALL, TOL, W, rel_error, synthetic_sequence  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENSOR = (H - 3, W - 3)          # 13 x 21 的传感器放在 16 x 24 的画布左上角：下、右两侧在传感器外


def make(border="zero", neuron="tif", seed=5, invariant=None):
    torch.manual_seed(seed)
    kwargs = dict(sensor_hw=SENSOR, canvas_hw=(H, W)) if border == "partial" else {}
    return EvSpSegNetStream(channels=SMALL, neuron=neuron, border=border, invariant=invariant, **kwargs)


class BorderTests(unittest.TestCase):
    def test_ratio_values(self):
        maps = border_ratio_maps(SENSOR, (H, W))
        shapes = {"enc1": (H, W), "enc2": (H // 2, W // 2), "enc3": (H // 4, W // 4), "enc4": (H // 8, W // 8),
                  "dec3": (H // 4, W // 4), "dec2": (H // 2, W // 2), "dec1": (H, W)}
        for name in LAYER_NAMES:
            self.assertEqual(tuple(maps[name].shape), (1, 1) + shapes[name], name)
        hs, ws = SENSOR
        r = maps["enc1"][0, 0]
        self.assertEqual(float(r[5, 5]), 1.0)                              # 内部不变
        self.assertEqual(float(r[hs - 1, 5]), 1.5)                         # 传感器下边缘：9 个输入里 6 个有效
        self.assertEqual(float(r[hs - 1, ws - 1]), 2.25)                   # 传感器右下角：4 个有效
        self.assertEqual(float(r[0, 0]), 2.25)                             # 画布左上角（画布外的补零同样处理）
        self.assertTrue(bool((r[hs:] == 0).all()) and bool((r[:, ws:] == 0).all()))   # 传感器外为 0
        r2 = maps["enc2"][0, 0]                                             # stride 2：输出格 i 读输入的 2i-1..2i+1 行
        self.assertEqual(float(r2[3, 3]), 1.0)
        self.assertEqual(float(r2[6, 5]), 1.5)                             # 第 11、12 行有效，第 13 行在传感器外
        self.assertEqual(float(r2[7, 5]), 0.0)                             # 第 13～15 行都在传感器外
        self.assertEqual(float(r2[0, 0]), 2.25)
        with self.assertRaises(ValueError):
            border_ratio_maps((H + 1, W), (H, W))                          # 传感器比画布大
        with self.assertRaises(ValueError):
            make(border="reflect")

    def test_default_unchanged_and_state_dict_compatible(self):
        zero, part = make("zero"), make("partial")
        self.assertEqual(zero.border, "zero")
        self.assertTrue(all(block.border_ratio is None for block in zero.blocks()))
        self.assertTrue(all(block.border_ratio is not None for block in part.blocks()))
        sz, sp = zero.state_dict(), part.state_dict()
        self.assertEqual(list(sz), list(sp))                               # 比例图不进 state_dict
        for key in sz:
            self.assertTrue(torch.equal(sz[key], sp[key]), key)            # partial 不消耗随机数：初始化逐位相同
        zero.load_state_dict(part.state_dict())
        part.load_state_dict(zero.state_dict())
        torch.manual_seed(5)
        legacy = EvSpSegNetStream(channels=SMALL, neuron="tif")            # 不传 border 的旧调用
        for key, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, sz[key]), key)

    def test_block_current_is_zero_current_times_ratio(self):
        zero, part = make("zero").double(), make("partial").double()
        part.load_state_dict(zero.state_dict())
        g = torch.Generator().manual_seed(1)
        for name in ("enc1", "enc2", "dec1"):
            bz, bp = getattr(zero, name), getattr(part, name)
            x = torch.rand(2, bz.conv.in_channels, H, W, generator=g, dtype=torch.float64)
            _, pre_z = bz._currents(x)
            _, pre_p = bp._currents(x)
            self.assertLess(rel_error(pre_p, pre_z * bp.border_ratio), TOL, name)
            self.assertFalse(torch.equal(pre_p, pre_z), name)               # 边缘确实变了

    def test_forward_chunk_matches_step_forward(self):
        seq = synthetic_sequence()
        source = TorchWindowSource(seq, CFG, Q99, torch.device("cpu"), torch.float64)
        hs, ws = SENSOR
        for neuron in ("lif", "tif"):
            net_a = make("partial", neuron).double()
            for block in net_a.blocks():                                   # 调低阈值让小网络各层都有活动
                if hasattr(block.neuron, "v_threshold"):
                    block.neuron.v_threshold = 0.05
            net_b = copy.deepcopy(net_a)
            states, logits_a, active = None, [], 0
            for start, end in CHUNKS:
                for k in range(start, end):
                    x, events, _, _ = source.window(k)
                    logits, states, info = net_a(x, events, states, collect=True)
                    logits_a.append(logits)
                    s1 = info["spikes"][0]
                    self.assertEqual(float(s1[..., hs:, :].abs().sum() + s1[..., :, ws:].abs().sum()), 0.0)
                    active += int((s1 != 0).sum())
            self.assertGreater(active, 0)
            states, logits_b = None, []
            for start, end in CHUNKS:
                x, events, _, _ = source.chunk(start, end)
                logits, states, _ = net_b.forward_chunk(x, events, states)
                logits_b.append(logits)
            self.assertLess(rel_error(torch.cat(logits_a), torch.cat(logits_b)), TOL, neuron)

    def test_invariant_moments_follow_ratio(self):
        inv = dict(time_bins=5, clip=3.0, ref_rate=tuple(REF_RATE))
        zero, part = make("zero", invariant=inv), make("partial", invariant=inv)   # 查表是 float32，与训练一致
        part.load_state_dict(zero.state_dict())
        for net in (zero, part):
            net.invariant.set_q99(Q99)
        g = torch.Generator().manual_seed(2)
        b = torch.rand(1, 2, H, W, generator=g) * 0.02
        with torch.no_grad():
            mu_z, var_z = zero.invariant.current_moments(b, zero.enc1)
            mu_p, var_p = part.invariant.current_moments(b, part.enc1)
        r = part.enc1.border_ratio
        self.assertLess(rel_error(mu_p, mu_z * r), 1e-6)
        self.assertLess(rel_error(var_p, var_z * r * r), 1e-6)
        self.assertGreater(float(mu_z.abs().max()), 0.0)

    def test_config_and_build_net(self):
        import train_stream_v1 as T
        from utils.stream_common import load_flat_config
        base = load_flat_config(os.path.join(ROOT, "configs", "evisseg_stream_tif_comm_peak_invariant.yaml"))
        cfg = load_flat_config(os.path.join(ROOT, "configs", "evisseg_stream_tif_comm_peak_invariant_pconv.yaml"))
        self.assertEqual(set(cfg), set(base) | {"border"})
        self.assertEqual({k for k in base if cfg[k] != base[k]}, {"seed", "save_root"})
        self.assertEqual((cfg["border"], cfg["seed"], cfg["save_root"]),
                         ("partial", 38, "log/stream_tif_commpeak_inv_pconv_seed38"))
        net = T.build_net(cfg, torch.device("cpu"))
        r = net.enc1.border_ratio[0, 0]
        self.assertEqual(net.border, "partial")
        self.assertEqual(tuple(r.shape), (cfg["pad_height"], cfg["pad_width"]))
        self.assertEqual((float(r[cfg["height"] - 1, 10]), float(r[cfg["height"], 10])), (1.5, 0.0))
        self.assertTrue(T.design_version(net).endswith("-pconv"))
        plain = T.build_net(base, torch.device("cpu"))
        self.assertIsNone(plain.enc1.border_ratio)
        self.assertFalse(T.design_version(plain).endswith("-pconv"))

    def test_end_to_end_train_and_eval(self):
        import train_stream_v1 as T
        from utils.stream_common import load_flat_config
        from test_stream_layerwise import SMALL as SMALL_CH
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pos = neg = 0
            for split, seeds in (("train", (3, 4)), ("val", (6,)), ("test", (7,))):
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
            cfg = load_flat_config(os.path.join(ROOT, "configs", "evisseg_stream_tif_comm_peak_invariant_pconv.yaml"))
            cfg.update(root=str(root), stats_path=str(root / "stats.json"), save_root=str(root / "pconv"),
                       channels=list(SMALL_CH), height=H, width_px=W, pad_height=H, pad_width=W,
                       n_windows=12, epochs=2, num_workers=0, tbptt_k=4, train_subset_size=1,
                       calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
                       segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=4,
                       execution="layer", input_device="gpu", streams=2)
            T.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
            ckpt = root / "pconv" / ("best_val_iou_seed%d.pt" % int(cfg["seed"]))
            self.assertTrue(ckpt.exists())
            saved = torch.load(str(ckpt), map_location="cpu")
            self.assertEqual(saved["config"]["border"], "partial")
            self.assertTrue(saved["design_version"].endswith("-pconv"))
            args = SimpleNamespace(checkpoint=str(ckpt), device="cpu", mode="eval", split="test", overwrite=True,
                                   execution=None, input_device=None, readout_ablation="none",
                                   subthreshold_ablation="none", memory_ablation="none", memory_off_layers=None,
                                   reset_every=0, reset_offset=0, dump_dir=None, eval_noise_load=0.0,
                                   eval_noise_seed=0)
            T.mode_eval(args, cfg)
            result = json.loads((root / "pconv" / ("eval_test_%s.json" % ckpt.stem)).read_text(encoding="utf-8"))
            for key in ("iou", "acc", "pd", "fa"):
                self.assertTrue(np.isfinite(result["carry"][key]), key)


if __name__ == "__main__":
    unittest.main()
