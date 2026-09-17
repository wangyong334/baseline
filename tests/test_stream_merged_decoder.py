"""电流合并解码器（merged_decoder）的回归测试：CPU 即可运行，兼容 torch 1.9。

误差一律用 (a - b).abs().max() 比较，不用 torch.testing.assert_close（torch 1.9 默认比较步长）。
"""
import unittest

import numpy as np
import torch

from dataset import stream_windows as sw
from model.evspsegnet_stream import (DECODER_STAGES, LAYER_NAMES, EvSpSegNetStream, calibrate_gains,
                                     count_parameters, estimate_operations, merge_decoder_state_dict)
from tools.convert_to_merged_decoder import (FLOAT64_TOLERANCE, convert_checkpoint, synthetic_sequences,
                                             verify_conversion)
from train_stream_v1 import build_net, train_sequence
from utils.stream_common import load_flat_config

SMALL = (2, 4, 4, 4)


def max_error(a, b):
    """两个张量的最大绝对误差。"""
    return float((a - b).abs().max())


def make_pair(dtype=torch.float64, neuron="lif", norm="none", threshold=0.05, seed=5):
    """构造原版网络与由它换算出的电流合并网络；调低阈值让各层发放，核对才不是在沉默网络上做的。"""
    torch.manual_seed(seed)
    original = EvSpSegNetStream(channels=SMALL, neuron=neuron, norm=norm).to(dtype).eval()
    merged = EvSpSegNetStream(channels=SMALL, neuron=neuron, norm=norm, merged_decoder=True).to(dtype).eval()
    merged.load_state_dict(merge_decoder_state_dict(original.state_dict()))
    for net in (original, merged):
        for block in net.blocks():
            if hasattr(block.neuron, "v_threshold"):
                block.neuron.v_threshold = threshold
    return original, merged


def windows(count, dtype, size=16, seed=11):
    """固定种子的小尺寸窗口：约 30% 像素有计数，每窗 6 个事件。"""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        mask = (torch.rand(1, 12, size, size, generator=g) < 0.3).to(dtype)
        x = mask * torch.rand(1, 12, size, size, generator=g).to(dtype) * 2
        ev = {"b": torch.zeros(6, dtype=torch.long),
              "y": torch.randint(0, size, (6,), generator=g), "x": torch.randint(0, size, (6,), generator=g),
              "p": torch.randint(0, 2, (6,), generator=g).to(dtype), "t_local": torch.rand(6, generator=g).to(dtype)}
        out.append((x, ev))
    return out


class MergedDecoderTests(unittest.TestCase):
    def test_conversion_is_exact_in_float64(self):
        for neuron, norm in (("lif", "none"), ("lif", "groupnorm_noshift"), ("relu", "none")):
            for mode in ("carry", "reset_each_window"):
                with self.subTest(neuron=neuron, norm=norm, mode=mode):
                    original, merged = make_pair(neuron=neuron, norm=norm)
                    state_a = state_b = None
                    active = [0] * len(LAYER_NAMES)
                    for x, ev in windows(8, torch.float64):
                        with torch.no_grad():
                            logits_a, state_a, info_a = original(x, ev, state_a, state_mode=mode, collect=True)
                            logits_b, state_b, info_b = merged(x, ev, state_b, state_mode=mode, collect=True)
                        self.assertLess(max_error(logits_a, logits_b), 1e-10)
                        for a, b in zip(state_a, state_b):
                            if a is not None:
                                self.assertLess(max_error(a, b), 1e-10)
                        for i, (a, b) in enumerate(zip(info_a["spikes"], info_b["spikes"])):
                            self.assertEqual(int(((a > 0) != (b > 0)).sum()), 0)
                            active[i] += int((a > 0).sum())
                    # 解码三层必须真的有活动，否则"零翻转"没有说服力
                    self.assertTrue(all(n > 0 for n in active[4:]), active)

    def test_float32_currents_match_at_real_channel_sizes(self):
        torch.manual_seed(3)
        original = EvSpSegNetStream().eval()
        merged = EvSpSegNetStream(merged_decoder=True).eval()
        merged.load_state_dict(merge_decoder_state_dict(original.state_dict()))
        g = torch.Generator().manual_seed(4)
        for (up, dec), (deep_ch, skip_ch, hw) in zip(DECODER_STAGES, ((48, 48, 11), (48, 24, 22), (24, 12, 44))):
            deep = (torch.rand(1, deep_ch, hw, hw, generator=g) < 0.2).float()
            skip = (torch.rand(1, skip_ch, 2 * hw, 2 * hw, generator=g) < 0.2).float()
            with torch.no_grad():
                a = getattr(original, dec).conv(torch.cat([getattr(original, up)(deep), skip], 1))
                b = getattr(merged, up)(deep) + getattr(merged, dec).conv(skip)
            self.assertEqual(tuple(a.shape), tuple(b.shape))
            self.assertLess(max_error(a, b), 1e-5, dec)

    def test_every_synaptic_input_is_binary_and_readout_unchanged(self):
        _, net = make_pair(dtype=torch.float32)
        modules = {"enc2": net.enc2.conv, "enc3": net.enc3.conv, "enc4": net.enc4.conv,
                   "up3": net.up3, "dec3": net.dec3.conv, "up2": net.up2, "dec2": net.dec2.conv,
                   "up1": net.up1, "dec1": net.dec1.conv}
        seen, handles = {}, []
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(
                lambda m, args, name=name: seen.update({name: args[0].detach()})))
        handles.append(net.readout.register_forward_pre_hook(lambda m, args: seen.update({"readout": args[0].detach()})))
        x, ev = windows(1, torch.float32)[0]
        with torch.no_grad():
            _, states, info = net(x, ev, None, collect=True)
        for handle in handles:
            handle.remove()
        self.assertEqual(len(states), 7)
        self.assertEqual(len(info["spikes"]), 7)
        for name in modules:
            self.assertTrue(bool(((seen[name] == 0) | (seen[name] == 1)).all()), name)
        for name in ("up3", "up2", "up1"):
            self.assertGreater(float(seen[name].sum()), 0.0, name)
        expected = info["u_pre"][-1][ev["b"], :, ev["y"], ev["x"]]
        self.assertLess(max_error(seen["readout"][:, :SMALL[0]], expected), 1e-6)

    def test_structure_parameters_and_state_count(self):
        net = EvSpSegNetStream(merged_decoder=True).eval()
        shapes = {name: tuple(p.shape) for name, p in net.named_parameters()}
        self.assertEqual(shapes["up3.weight"], (48, 48, 4, 4))
        self.assertEqual(shapes["dec3.conv.weight"], (48, 48, 3, 3))
        self.assertEqual(shapes["up2.weight"], (48, 24, 4, 4))
        self.assertEqual(shapes["dec2.conv.weight"], (24, 24, 3, 3))
        self.assertEqual(shapes["up1.weight"], (24, 12, 4, 4))
        self.assertEqual(shapes["dec1.conv.weight"], (12, 12, 3, 3))
        self.assertEqual(count_parameters(EvSpSegNetStream())["total"], 105345)
        self.assertEqual(count_parameters(net)["total"], 123057)
        self.assertEqual(len(net.blocks()), 7)
        from_config = build_net(load_flat_config("configs/evisseg_stream_merged_decoder.yaml"), torch.device("cpu"))
        self.assertTrue(from_config.merged_decoder)
        self.assertEqual(count_parameters(from_config)["total"], 123057)
        empty = {"b": torch.zeros(0, dtype=torch.long), "y": torch.zeros(0, dtype=torch.long),
                 "x": torch.zeros(0, dtype=torch.long), "p": torch.zeros(0), "t_local": torch.zeros(0)}
        with torch.no_grad():
            logits, states, _ = net(torch.zeros(1, 12, 264, 352), empty, None)
        self.assertEqual(tuple(logits.shape), (0,))
        self.assertEqual(tuple(states[-1].shape), (1, 12, 264, 352))
        self.assertEqual(sum(s.numel() for s in states), 3972672)

    def test_operation_accounting_counts_decoder_as_spike_driven(self):
        net = EvSpSegNetStream(channels=SMALL, merged_decoder=True)
        rates = dict.fromkeys(LAYER_NAMES, 0.0)
        rates.update({"enc3": 0.5, "enc4": 0.25})
        ops = estimate_operations(net, 16, 16, rates, 4)
        layers = {r["layer"]: r for r in ops["per_layer"]}
        for name in ("up3", "dec3", "up2", "dec2", "up1", "dec1"):
            self.assertEqual(layers[name]["mac"], 0.0, name)
        self.assertEqual(layers["dec3"]["dense"], 4 * 4 * 9 * 16)        # 跳连 4->4，3x3，4x4 分辨率
        self.assertEqual(layers["dec3"]["sop"], 0.5 * 4 * 4 * 9 * 16)
        self.assertEqual(layers["up3"]["dense"], 4 * 4 * 16 * 4)         # ConvT4x4 按 2x2 输入位置计
        self.assertEqual(layers["up3"]["sop"], 0.25 * 4 * 4 * 16 * 4)
        self.assertEqual(ops["mac"], layers["enc1"]["mac"] + layers["readout"]["mac"])
        dense = estimate_operations(net, 16, 16, dict.fromkeys(LAYER_NAMES, 1.0), 4)
        self.assertEqual(dense["mac"] + dense["sop"], dense["dense_ops"])
        with torch.no_grad():
            _, states, _ = net(*windows(1, torch.float32)[0], None)
        self.assertEqual(ops["state_elements"], sum(s.numel() for s in states))
        relu = estimate_operations(EvSpSegNetStream(channels=SMALL, neuron="relu", merged_decoder=True),
                                   16, 16, rates, 4)
        self.assertEqual(relu["sop"], 0.0)

    def test_invalid_configurations_and_conversions_are_rejected(self):
        with self.assertRaises(ValueError):
            EvSpSegNetStream(channels=SMALL, merged_decoder=1)
        # 已删除的 10 层版本：权重形状与原版相同但多了 up*_spike，必须拒绝，不能静默换算
        legacy = EvSpSegNetStream(channels=SMALL).state_dict()
        legacy["up3_spike.gain.log_gain"] = torch.zeros(SMALL[2])
        with self.assertRaises(ValueError):
            merge_decoder_state_dict(legacy)
        with self.assertRaises(ValueError):
            merge_decoder_state_dict(EvSpSegNetStream(channels=SMALL, merged_decoder=True).state_dict())
        with self.assertRaises(RuntimeError):
            EvSpSegNetStream(channels=SMALL, merged_decoder=True).load_state_dict(
                EvSpSegNetStream(channels=SMALL).state_dict())
        cfg = load_flat_config("configs/evisseg_stream_v1.yaml")
        with self.assertRaises(ValueError):
            build_net(dict(cfg, spiking_decoder=True), torch.device("cpu"))
        with self.assertRaises(ValueError):
            EvSpSegNetStream(channels=SMALL)(*windows(1, torch.float32)[0], [None] * 10)

    def test_original_forward_is_unchanged(self):
        torch.manual_seed(21)
        net = EvSpSegNetStream(channels=SMALL).eval()
        for block in net.blocks():
            block.neuron.v_threshold = 0.05
        x, ev = windows(1, torch.float32)[0]
        with torch.no_grad():
            actual, states, _ = net(x, ev, None)
            s1, _, _, _ = net.enc1(x, None)
            s2, _, _, _ = net.enc2(s1, None)
            s3, _, _, _ = net.enc3(s2, None)
            s4, _, _, _ = net.enc4(s3, None)
            s5, _, _, _ = net.dec3(torch.cat([net.up3(s4), s3], 1), None)
            s6, _, _, _ = net.dec2(torch.cat([net.up2(s5), s2], 1), None)
            _, _, u, _ = net.dec1(torch.cat([net.up1(s6), s1], 1), None)
            feat = u[ev["b"], :, ev["y"], ev["x"]]
            expected = net.readout(torch.cat([feat, torch.stack([ev["p"], ev["t_local"]], 1)], 1)).squeeze(1)
        self.assertEqual(max_error(actual, expected), 0.0)                # 逐位一致
        self.assertEqual(len(states), 7)
        # 原 V1 配置没有 merged_decoder 键，必须仍然构造原版结构
        v1 = build_net(load_flat_config("configs/evisseg_stream_v1.yaml"), torch.device("cpu"))
        self.assertFalse(v1.merged_decoder)
        self.assertEqual(count_parameters(v1)["total"], 105345)

    def test_calibration_covers_seven_layers_in_order(self):
        torch.manual_seed(9)
        net = EvSpSegNetStream(channels=SMALL, merged_decoder=True)
        data = windows(2, torch.float32)
        report = calibrate_gains(net, lambda: [data], 0.99, 32, 1, 0.05, 20.0, 3)
        self.assertEqual([r["layer"] for r in report], list(LAYER_NAMES))
        self.assertTrue(bool(net.gain_calibrated))

    def test_training_from_scratch_reaches_every_parameter(self):
        torch.manual_seed(13)
        net = EvSpSegNetStream(channels=SMALL, merged_decoder=True)
        with torch.no_grad():
            for name, param in net.named_parameters():
                if name.endswith("weight"):
                    param.fill_(0.1)
        for block in net.blocks():
            block.neuron.v_threshold = 0.01
        t = np.array([0, 1, 50, 51, 150, 151], dtype=np.int64)
        x = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
        p = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
        labels = p.astype(np.float32)
        order, bounds = sw.split_windows(t, 50, 4)
        bins, local = sw.time_bin_and_local(t, 50, 5)
        seq = sw.StreamSequence("synthetic", x, x.copy(), t, p, labels, labels.astype(np.float64),
                                bins, local, order, bounds)
        cfg = dict(tbptt_k=2, optimizer_step="per_sequence", grad_clip=1.0, time_bins=5,
                   pad_height=16, pad_width=16, input_clip=3.0)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        before = net.up3.weight.detach().clone()
        result = train_sequence(net, seq, optimizer, cfg, np.ones(12, dtype=np.float32), 2.0,
                                torch.device("cpu"), np.random.RandomState(0))
        self.assertEqual(result["missing_grads"], [])
        self.assertTrue(np.isfinite(result["loss_sum"]))
        self.assertFalse(torch.equal(before, net.up3.weight.detach()))

    def test_convert_checkpoint_keeps_metadata_and_is_exact(self):
        cfg = load_flat_config("configs/evisseg_stream_v1.yaml")
        cfg["channels"] = list(SMALL)
        torch.manual_seed(17)
        original = build_net(cfg, torch.device("cpu"))
        original.gain_calibrated.fill_(True)
        ckpt = {"model": original.state_dict(), "optimizer": {"state": {}}, "epoch": 12, "best_val_iou": 0.87,
                "config": cfg, "stats": {"q99": [1.0] * 12}, "design_version": "v1-3"}
        converted = convert_checkpoint(ckpt, source="best.pt")
        self.assertNotIn("optimizer", converted)
        self.assertTrue(converted["config"]["merged_decoder"])
        self.assertNotIn("merged_decoder", ckpt["config"])               # 不修改输入
        for key in ("epoch", "best_val_iou", "stats", "design_version"):
            self.assertEqual(converted[key], ckpt[key])
        merged = build_net(converted["config"], torch.device("cpu"))
        merged.load_state_dict(converted["model"])                        # 严格加载
        self.assertTrue(bool(merged.gain_calibrated))
        factory = synthetic_sequences(12, 16, 16, 1, 6, torch.device("cpu"), density=0.3, events=8)
        # verify_conversion 按配置重新构网；通过配置调低阈值，让随机权重的小网络也有足够发放
        converted_cfg = dict(converted["config"], v_threshold=0.05)
        original_low = build_net(dict(cfg, v_threshold=0.05), torch.device("cpu"))
        original_low.load_state_dict(ckpt["model"])
        (math_report, deploy_report), passed = verify_conversion(
            original_low, converted_cfg, ckpt["model"], converted["model"], factory, torch.device("cpu"))
        self.assertTrue(passed)
        self.assertEqual(math_report["windows"], 6)
        self.assertLessEqual(math_report["logit_max_error"], FLOAT64_TOLERANCE)
        self.assertLessEqual(math_report["state_max_error"], FLOAT64_TOLERANCE)
        self.assertEqual(math_report["spike_flips"], 0)
        self.assertGreater(math_report["active_neurons"], 0)
        self.assertEqual(deploy_report["windows"], 6)                    # 部署核对只报告，不设门槛
        with self.assertRaises(ValueError):
            convert_checkpoint(converted)
        with self.assertRaises(ValueError):
            convert_checkpoint(dict(ckpt, config=dict(cfg, spiking_decoder=True)))


if __name__ == "__main__":
    unittest.main()
