"""双仓机制与全 U-Net/TBPTT/监控/保存恢复的回归测试；CPU 合成数据即可。"""
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

from dataset.stream_source import TorchWindowSource
from model.evspsegnet_stream import EvSpSegNetStream, LAYER_NAMES, count_parameters, estimate_operations
from model.lif2d_stream import StreamingLIF2d, StreamingDualMemoryLIF2d, detach_states, state_tensors
from test_stream_layerwise import CFG, Q99, SMALL, H, W, synthetic_sequence
from tools import check_stream_execution, stream_energy
from utils.stream_common import load_flat_config
import train_stream_v1 as T

CPU = torch.device("cpu")
PLACEMENTS = (("enc4",), ("enc3", "enc4"), ("enc2", "enc3", "enc4"))
PLACEMENTS += (("enc2",), ("enc3",), ("dec3",), ("dec2",), ("dec1",), ("enc3", "dec3"))
PLACEMENTS += (("enc2", "enc3", "enc4", "dec3", "dec2"),
               ("enc2", "enc3", "enc4", "dec3", "dec2", "dec1"))


def cell(**overrides):
    options = dict(channels=3, dt_ms=50.0, tau_init_ms=200.0, tau_min_ms=50.0,
                   tau_max_ms=2000.0, v_threshold=1.0)
    options.update(overrides)
    return StreamingDualMemoryLIF2d(**options).double()


def net(layers=("enc4",), merged=False, nonzero=False):
    torch.manual_seed(37)
    model = EvSpSegNetStream(channels=SMALL, v_threshold=0.05, merged_decoder=merged,
                            dual_memory={"layers": list(layers)} if layers else None).double()
    if nonzero:
        with torch.no_grad():
            for name in model.dual_memory_layers:
                getattr(model, name).neuron.kappa_raw.fill_(0.7)
    return model


class DualCellTests(unittest.TestCase):
    def test_bounds_decay_no_self_write_and_no_memory_only_spikes(self):
        n = cell()
        with torch.no_grad():
            n.kappa_raw.fill_(20.0)
            current = torch.full((1, 3, 2, 2), 100.0, dtype=torch.float64)
            state = None
            for _ in range(160):
                _, state, _ = n(current, state)
                self.assertLessEqual(float(state[1].abs().max()), 1.0)
            old_m = state[1].clone()
            # 相同 I/M，不同 U：记忆候选必须相同，不允许 U 回写 M。
            zero = torch.zeros_like(current)
            _, a, _ = n(zero, (zero, old_m))
            _, b, _ = n(zero, (torch.full_like(zero, 100.0), old_m))
            self.assertTrue(torch.equal(a[1], b[1]))
            # 零胞体 + 有界记忆 + 零突触输入：新增路径不能自己触发放电。
            state = (zero, old_m)
            alpha = n.memory_alpha().view(1, -1, 1, 1)
            for step in range(1, 161):
                spikes, state, pre = n(zero, state)
                self.assertEqual(float(spikes.sum()), 0.0)
                self.assertLess(float(pre.abs().max()), n.kappa_max * n.v_threshold)
                self.assertTrue(torch.allclose(state[1], old_m * alpha ** step, atol=1e-13, rtol=1e-12))

    def test_reads_old_memory_not_current_write(self):
        n = cell()
        with torch.no_grad():
            n.kappa_raw.fill_(0.8)
        current = torch.full((1, 3, 1, 1), 0.4, dtype=torch.float64)
        _, state, pre = n(current, None)
        self.assertTrue(torch.equal(pre, current))
        self.assertGreater(float(state[1].detach().abs().sum()), 0)

    def test_memory_gradient_through_time_and_tbptt_detach(self):
        n = cell()
        with torch.no_grad():
            n.kappa_raw.fill_(0.7)
        first = torch.full((1, 3, 1, 1), 0.2, dtype=torch.float64, requires_grad=True)
        _, state, _ = n(first, None)
        # 排除 U 历史，只保留 M 的历史，检查未来损失能训练先前写入。
        _, _, pre = n(torch.zeros_like(first), (torch.zeros_like(first), state[1]))
        pre.sum().backward()
        self.assertGreater(float(first.grad.abs().sum()), 0)
        self.assertGreater(float(n.memory_a.grad.abs().sum()), 0)
        self.assertGreater(float(n.kappa_raw.grad.abs().sum()), 0)
        detached = detach_states([None, state])
        self.assertIsInstance(detached[1], tuple)
        for before, after in zip(state_tensors([state]), state_tensors(detached)):
            self.assertTrue(torch.equal(before, after))
            self.assertFalse(after.requires_grad)

    def test_zero_kappa_has_trainable_readout_gradient(self):
        n = cell()
        current = torch.full((1, 3, 1, 1), 0.3, dtype=torch.float64)
        _, state, _ = n(current, None)
        _, _, pre = n(current, state)
        pre.sum().backward()
        self.assertTrue(torch.equal(n.kappa(), torch.zeros(3, dtype=torch.float64)))
        self.assertGreater(float(n.kappa_raw.grad.abs().sum()), 0)
        self.assertIsNotNone(n.memory_a.grad)

    def test_validation(self):
        for options in ({"kappa_max": 1.0}, {"kappa_max": 0.0}, {"kappa_max": float("nan")},
                        {"memory_tau_init_ms": 20.0}, {"memory_tau_min_ms": -1},
                        {"dt_ms": 0}, {"v_threshold": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                cell(**options)


class DualNetworkTests(unittest.TestCase):
    def test_zero_initialized_or_disabled_memory_exactly_matches_v1(self):
        source = TorchWindowSource(synthetic_sequence(), CFG, Q99, CPU, torch.float64)
        for layers in PLACEMENTS:
            for merged in (False, True):
                baseline, candidate = net((), merged), net(layers, merged)
                for name, parameter in baseline.state_dict().items():
                    self.assertTrue(torch.equal(parameter, candidate.state_dict()[name]), name)
                for disable in (False, True):
                    if disable:
                        with torch.no_grad():
                            for name in layers:
                                getattr(candidate, name).neuron.kappa_raw.fill_(0.9)
                        candidate.set_memory_read_enabled(False)
                    left = right = None
                    with torch.no_grad():
                        for k in range(8):
                            x, events, _, _ = source.window(k)
                            a, left, ia = baseline(x, events, left, collect=True)
                            b, right, ib = candidate(x, events, right, collect=True)
                            self.assertTrue(torch.equal(a, b), (layers, merged, disable, k))
                            for i, name in enumerate(LAYER_NAMES):
                                soma = right[i][0] if name in layers else right[i]
                                self.assertTrue(torch.equal(left[i], soma), name)
                                self.assertTrue(torch.equal(ia["spikes"][i], ib["spikes"][i]), name)

    def test_step_chunk_states_gradients_and_training_updates(self):
        seq = synthetic_sequence(seed=3)
        cfg = dict(CFG, lr=1e-3, seed=37, n_windows=12)
        for layers in PLACEMENTS:
            for merged in (False, True):
                model = net(layers, merged, nonzero=True)
                report = check_stream_execution.compare_chunks(model, seq, cfg, Q99, CPU,
                                                               torch.float64, 1, 3, 3.0)
                for key in ("logit_rel_error", "state_rel_error", "u_pre_rel_error_max", "grad_rel_error"):
                    self.assertLess(report[key], 1e-8, (layers, merged, key, report[key]))
                self.assertEqual(report["spike_flips_total"], 0)
                self.assertGreater(report["spikes"]["enc3"], 0)
            for optimizer_step in ("per_sequence", "per_chunk"):
                trained = []
                for execution, input_device in (("step", "cpu"), ("layer", "gpu")):
                    model = net(layers, nonzero=True)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    out = T.train_sequence(model, seq, optimizer,
                                           dict(cfg, execution=execution, input_device=input_device,
                                                optimizer_step=optimizer_step),
                                           Q99, 3.0, CPU, np.random.RandomState(7))
                    self.assertEqual(out["missing_grads"], [])
                    trained.append(model)
                for (name, a), (_, b) in zip(trained[0].named_parameters(), trained[1].named_parameters()):
                    self.assertLess(float((a.detach() - b.detach()).abs().max()), 1e-8,
                                    (layers, optimizer_step, name))

    def test_reset_clears_both_states_and_monitor_matches_chunks(self):
        seq = synthetic_sequence(seed=4)
        for layers in PLACEMENTS:
            model = net(layers, nonzero=True)
            for mode in ("carry", "reset_each_window"):
                results = []
                for execution in ("step", "layer"):
                    monitor = T.LayerMonitor(model.v_threshold)
                    with torch.no_grad():
                        probs, confusion = T.run_sequence(model, seq, dict(CFG, execution=execution),
                                                         Q99, CPU, mode, monitor)
                    results.append((probs, confusion, monitor.summary()))
                self.assertLess(float(np.abs(results[0][0] - results[1][0]).max()), 1e-6)
                self.assertTrue(np.array_equal(results[0][1], results[1][1]))
                for name, row in results[0][2].items():
                    for key, value in row.items():
                        self.assertAlmostEqual(value, results[1][2][name][key], delta=1e-10)
            source = TorchWindowSource(seq, CFG, Q99, CPU, torch.float64)
            x, events, _, _ = source.window(0)
            with torch.no_grad():
                _, states, _ = model(x, events, None)
                a, cleared, _ = model(x, events, states, state_mode="reset_each_window")
                b, fresh, _ = model(x, events, None)
            self.assertTrue(torch.equal(a, b))
            for p, q in zip(state_tensors(cleared), state_tensors(fresh)):
                self.assertTrue(torch.equal(p, q))

    def test_checkpoint_roundtrip_and_extra_state_accounting(self):
        for layers in PLACEMENTS:
            model = net(layers, nonzero=True)
            config = load_flat_config("configs/evisseg_stream_dual_enc4.yaml")
            config.update(channels=list(SMALL), v_threshold=0.05, dual_memory={"layers": list(layers)})
            restored = T.build_net(config, CPU).double()
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored.load_state_dict(torch.load(buffer, map_location="cpu"))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)
            x = torch.zeros(1, 12, H, W, dtype=torch.float64)
            e = {"b": torch.zeros(0, dtype=torch.long), "y": torch.zeros(0, dtype=torch.long),
                 "x": torch.zeros(0, dtype=torch.long), "p": torch.zeros(0), "t_local": torch.zeros(0)}
            with torch.no_grad():
                _, states, _ = model(x, e, None)
            ops = estimate_operations(model, H, W, dict.fromkeys(LAYER_NAMES, 0.1), 5)
            self.assertEqual(ops["state_elements"], sum(s.numel() for s in state_tensors(states)))
            extra_params = count_parameters(model)["total"] - count_parameters(net(()))["total"]
            self.assertEqual(ops["dual_memory"]["parameters"], extra_params)
            self.assertIn("dual_memory_unpriced_operations", stream_energy.account(ops, 0.01, 160))
            stats = T.tau_statistics(model)
            for name in layers:
                self.assertGreater(stats[name]["kappa_abs_mean"], 0)

    def test_train_resume_and_eval_entrypoints(self):
        # 实际走文件加载、校准、两轮训练、checkpoint、续训与两种正式 eval 入口。
        # 只用临时目录中的小尺寸合成数据，不依赖 EV-UAV 或 GPU。
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
        except ImportError as error:
            self.skipTest("完整评估入口需要原 V1 的 cv2/pandas/tqdm 依赖: %s" % error)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            seq = synthetic_sequence(seed=8)
            loc = np.stack((seq.x, seq.y, seq.t), axis=1)
            norm = np.column_stack((loc, seq.p, seq.label, seq.target_id))
            for split in ("train", "val", "test"):
                (root / split).mkdir()
                np.savez(root / split / seq.name, ev_loc=loc, evs_norm=norm)
            pos = int(seq.label.sum())
            neg = seq.n_events - pos
            stats = dict(q99=Q99.tolist(), train_pos=pos, train_neg=neg, pos_weight=neg / pos)
            (root / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
            cfg = load_flat_config("configs/evisseg_stream_dual_enc234.yaml")
            cfg.update(dual_memory={"layers": ["enc2", "dec3", "dec2"]},
                       root=str(root), stats_path=str(root / "stats.json"), save_root=str(root / "run"),
                       channels=list(SMALL), height=H, width_px=W, pad_height=H, pad_width=W,
                       n_windows=12, epochs=2, num_workers=0, tbptt_k=4, train_subset_size=1,
                       calib_sequences=1, calib_windows=8, calib_samples_per_channel=32,
                       calib_min_positive=4, segments=[[0, 3], [4, 7], [8, 11]],
                       timing_warmup_windows=1, eval_chunk=4)
            args = SimpleNamespace(device="cpu", mode="train", resume=False)
            T.mode_train(args, cfg)
            checkpoint = root / "run" / "best_val_iou_seed37.pt"
            self.assertTrue(checkpoint.exists())
            records_before = (root / "run" / "metrics.jsonl").read_text(encoding="utf-8")
            args.resume = True
            T.mode_train(args, cfg)  # 已完成的 run 正确恢复，不能重复追加 epoch。
            self.assertEqual(records_before, (root / "run" / "metrics.jsonl").read_text(encoding="utf-8"))
            args = SimpleNamespace(device="cpu", checkpoint=str(checkpoint), execution=None,
                                   input_device=None, readout_ablation="none", memory_ablation="none",
                                   overwrite=False, split="test", dump_dir=None)
            for ablation, off_layers in (("none", None), ("off", None), ("none", ["dec3"])):
                args.memory_ablation = ablation
                args.memory_off_layers = off_layers
                T.mode_eval(args, cfg)
                suffix = "" if ablation == "none" else "_memory_off"
                if off_layers:
                    suffix = "_memory_off_dec3"
                result = json.loads((root / "run" / ("eval_test_best_val_iou_seed37%s.json" % suffix))
                                    .read_text(encoding="utf-8"))
                self.assertEqual(result["memory_ablation"], "selected" if off_layers else ablation)
                for mode in ("carry", "reset_each_window"):
                    for layer in cfg["dual_memory"]["layers"]:
                        self.assertEqual(result[mode]["tau"][layer]["memory_read_enabled"],
                                         ablation == "none" and layer not in (off_layers or []))
                    self.assertIn("dual_memory", result[mode]["operations"])
                    self.assertTrue(math.isfinite(result[mode]["iou"]))

    def test_config_scope(self):
        for suffix, layers in zip(("enc4", "enc34", "enc234"), PLACEMENTS):
            cfg = load_flat_config("configs/evisseg_stream_dual_%s.yaml" % suffix)
            self.assertEqual(T.build_net(cfg, CPU).dual_memory_layers, layers)
        base = load_flat_config("configs/evisseg_stream_v1.yaml")
        for layer in ("enc2", "enc3", "dec3", "dec2"):
            cfg = load_flat_config("configs/evisseg_stream_dual_%s.yaml" % layer)
            self.assertEqual(T.build_net(cfg, CPU).dual_memory_layers, (layer,))
            self.assertEqual({k:v for k,v in cfg.items() if k not in ("dual_memory", "save_root")},
                             {k:v for k,v in base.items() if k != "save_root"})
        for invalid in ({"layers": ["dec4"]}, {"layers": []}, {"layers": "enc3"},
                        {"layers": ["enc3", "enc3"]}, {"layers": ["enc3"], "typo": 1}):
            with self.assertRaises(ValueError):
                EvSpSegNetStream(dual_memory=invalid)
        with self.assertRaises(ValueError):
            EvSpSegNetStream(neuron="relu", dual_memory={"layers": ["enc3"]})

    def test_selective_memory_read_control(self):
        model = net(("enc3", "dec3"), nonzero=True)
        model.set_memory_read_enabled(False, ["enc3"])
        self.assertFalse(model.enc3.neuron.memory_read_enabled)
        self.assertTrue(model.dec3.neuron.memory_read_enabled)
        for layers in (["enc4"], ["enc3", "enc3"], [], "enc3"):
            with self.assertRaises(ValueError):
                model.set_memory_read_enabled(False, layers)
        model.set_memory_read_enabled(True)
        self.assertTrue(model.enc3.neuron.memory_read_enabled)
        self.assertTrue(model.dec3.neuron.memory_read_enabled)


if __name__ == "__main__":
    unittest.main()
