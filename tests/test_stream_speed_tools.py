"""计时、等价核对、能耗工具的冒烟与口径测试（CPU、合成数据，不需要数据集、GPU 或 spconv）。

服务器上这些工具要用真实数据跑几十分钟，这里先确认它们在小数据上能跑通、计数口径正确。
"""
import unittest

import numpy as np
import torch

from dataset.stream_source import NumpyWindowSource
from model.evspsegnet_stream import LAYER_NAMES, EvSpSegNetStream, estimate_operations
from tests.test_stream_layerwise import CFG, Q99, SMALL, WINDOWS, H, W, make_net, synthetic_sequence
from tools import baseline_energy, bench_stream_speed, check_stream_execution, stream_energy

TOOL_CFG = dict(CFG, lr=1e-3, seed=37, tbptt_k=4, n_windows=WINDOWS)
CPU = torch.device("cpu")


class StreamEnergyTests(unittest.TestCase):
    def test_account_matches_existing_formula_and_scales_enc1(self):
        net = EvSpSegNetStream(channels=SMALL, merged_decoder=True)
        rates = dict.fromkeys(LAYER_NAMES, 0.1)
        ops = estimate_operations(net, 16, 16, rates, 7)
        dense = stream_energy.account(ops, 1.0, 160)
        # 与 CLAUDE.md 汇总脚本的公式一致：E = (mac*4.6e-12 + sop*0.9e-12) J
        expected = (ops["mac"] * 4.6e-12 + ops["sop"] * 0.9e-12) * 1e3
        self.assertAlmostEqual(dense["dense"]["energy_mj_per_window"], expected, delta=1e-12)
        self.assertAlmostEqual(dense["event"]["energy_mj_per_window"], expected, delta=1e-12)   # 占比 1 时两口径相同
        self.assertAlmostEqual(dense["dense"]["energy_mj_per_8s"], 160 * expected, delta=1e-9)
        sparse = stream_energy.account(ops, 0.01, 160)
        enc1 = [r for r in ops["per_layer"] if r["layer"] == "enc1"][0]
        self.assertAlmostEqual(sparse["event"]["mac_per_window"], ops["mac"] - enc1["dense"] * 0.99, delta=1e-6)
        self.assertEqual(sparse["event"]["sop_per_window"], ops["sop"])
        self.assertGreater(sparse["membrane_leak_upper_bound_mj_per_8s"], 0.0)

    def test_nonzero_fraction_matches_network_input(self):
        seq = synthetic_sequence()
        fraction, windows = stream_energy.input_nonzero_fraction([seq], CFG)
        source = NumpyWindowSource(seq, CFG, Q99, CPU)
        nonzero = sum(int((source.window(k)[0] != 0).sum()) for k in range(WINDOWS))
        self.assertEqual(windows, WINDOWS)
        self.assertAlmostEqual(fraction, nonzero / float(WINDOWS * 12 * H * W), delta=1e-12)


class BaselineEnergyHelperTests(unittest.TestCase):
    def test_counting_helpers(self):
        self.assertEqual(baseline_energy.attention_operations(1, 10, 24), 10 * (4 * 24 * 24 + 2 * 24))
        self.assertEqual(baseline_energy.linear_operations(5, 12, 1), 60)
        records = {"conv1.0.pwconv.0": {"kind": "sparse_conv", "ops": 100, "calls": 1},
                   "conv1.0.se.fc.0": {"kind": "linear", "ops": 7, "calls": 1},
                   "pa2.multihead_attn": {"kind": "attention", "ops": 30, "calls": 1}}
        summary = baseline_energy.summarize_records(records)
        self.assertEqual(summary["mac"], 137)
        self.assertEqual(summary["by_group"], {"conv1": 107, "pa2": 30})
        self.assertEqual(summary["by_kind"], {"sparse_conv": 100, "linear": 7, "attention": 30})


class CheckToolTests(unittest.TestCase):
    def test_checks_pass_on_synthetic_data(self):
        seq = synthetic_sequence(seed=4)
        net = make_net(merged=True, seed=2).float()
        inputs = check_stream_execution.check_inputs(seq, TOOL_CFG, Q99, CPU, 4)
        self.assertTrue(inputs["passed"], inputs)
        self.assertGreater(inputs["nonzero_input_fraction"], 0.0)
        logic = check_stream_execution.compare_chunks(net, seq, TOOL_CFG, Q99, CPU, torch.float64, 2, 4, 3.0)
        self.assertEqual(logic["windows"], [2, 10])
        self.assertLessEqual(logic["logit_rel_error"], check_stream_execution.FLOAT64_TOLERANCE)
        self.assertLessEqual(logic["u_pre_rel_error_max"], check_stream_execution.FLOAT64_TOLERANCE)
        self.assertLessEqual(logic["grad_rel_error"], check_stream_execution.FLOAT64_TOLERANCE)
        self.assertEqual(logic["spike_flips_total"], 0)
        self.assertGreater(min(logic["spikes"].values()), 0)
        train = check_stream_execution.compare_train_sequence(net, seq, TOOL_CFG, Q99, CPU, torch.float64, 8, 3.0)
        self.assertLessEqual(train["param_max_abs_error"], check_stream_execution.PARAM_TOLERANCE)
        numeric = check_stream_execution.compare_chunks(net, seq, TOOL_CFG, Q99, CPU, torch.float32, 2, 4, 3.0)
        self.assertIn("grad_rel_error", numeric)
        rows = check_stream_execution.compare_sequences(net.eval(), [seq, synthetic_sequence(seed=5)], TOOL_CFG,
                                                        Q99, CPU, 2, "carry")
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(row["reference_iou"], row["fast_iou"], delta=1e-6)


class BenchToolTests(unittest.TestCase):
    def test_bench_functions_run(self):
        seqs = [synthetic_sequence(seed=6)]
        clock = bench_stream_speed.Clock(CPU)
        net = make_net(merged=True, seed=3).float()
        inputs = bench_stream_speed.bench_inputs(seqs, TOOL_CFG, Q99, CPU, clock)
        self.assertGreater(inputs["numpy_ms_per_window"], 0.0)
        training = bench_stream_speed.bench_training(net, seqs, TOOL_CFG, Q99, 3.0, CPU, 1, clock)
        self.assertEqual(set(training), {"step+cpu", "step+gpu", "layer+cpu", "layer+gpu"})
        losses = [r["loss_sum_first"] for r in training.values()]
        self.assertLess(max(losses) - min(losses), 1e-3 * max(losses))
        x, events, labels = bench_stream_speed.synthetic_chunk(4, 16, 16, 30, CPU)
        network = bench_stream_speed.bench_network(net, x, events, labels, 3.0, 1, clock)
        self.assertEqual(set(network), {"step", "layer"})
        for with_monitor in (True, False):
            evaluation = bench_stream_speed.bench_evaluation(net, seqs, TOOL_CFG, Q99, CPU, 1, clock, with_monitor)
            self.assertGreater(evaluation["layer+gpu"]["ms_per_window"], 0.0)
        counts = {"train": 99, "val": 24, "subset": 24, "windows": 160}
        p = bench_stream_speed.project_epoch(10.0, 6.0, 5.0, 0.5, counts, 5)
        self.assertAlmostEqual(p["train_s"], 99 * 160 * 0.01)
        self.assertAlmostEqual(p["val_s"], 24 * (0.5 + 160 * 0.006))
        self.assertAlmostEqual(p["subset_s"], 24 * (0.5 + 160 * 0.005) / 5)
        self.assertAlmostEqual(p["epoch_s_low"], p["train_s"] + p["val_s"] + p["subset_s"])
        self.assertAlmostEqual(p["epoch_s_high"], p["epoch_s_low"] + 99 * 0.5)


if __name__ == "__main__":
    unittest.main()
