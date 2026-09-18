"""计时、等价核对、能耗工具的冒烟与口径测试（CPU、合成数据，不需要数据集、GPU 或 spconv）。

服务器上这些工具要用真实数据跑几十分钟，这里先确认它们在小数据上能跑通、计数口径正确。
"""
import os
import sys
import unittest

import numpy as np
import torch

# 直接从测试目录导入共用的合成数据，避免环境里第三方安装的同名 tests 包遮挡
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.stream_source import NumpyWindowSource  # noqa: E402
from model.evspsegnet_stream import LAYER_NAMES, EvSpSegNetStream, estimate_operations  # noqa: E402
from test_stream_layerwise import CFG, Q99, SMALL, WINDOWS, H, W, make_net, synthetic_sequence  # noqa: E402
from tools import baseline_energy, bench_stream_speed, check_stream_execution, stream_energy  # noqa: E402

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



class GradedNeuronTests(unittest.TestCase):
    """有状态但不发放的对照：膜电位轨迹必须与 LIF 一致，只有输出不同。"""

    def test_membrane_matches_lif_and_output_is_graded(self):
        from model.lif2d_stream import StreamingGraded2d, StreamingLIF2d
        torch.manual_seed(0)
        args = dict(dt_ms=50.0, tau_init_ms=200.0, tau_min_ms=50.0, tau_max_ms=2000.0, v_threshold=1.0)
        lif, graded = StreamingLIF2d(3, **args).double(), StreamingGraded2d(3, **args).double()
        graded.a.data.copy_(lif.a.data)
        current = torch.rand(6, 2, 3, 4, 5, dtype=torch.float64) * 3
        spikes, state_a, u_a = lif.forward_steps(current, None)
        values, state_b, u_b = graded.forward_steps(current, None)
        self.assertLess(float((u_a - u_b).abs().max()), 1e-12)          # 膜电位轨迹一致
        self.assertLess(float((state_a - state_b).abs().max()), 1e-12)  # 复位后的状态也一致
        self.assertTrue(torch.equal((values > 0).double(), spikes))     # 发放位置一致
        self.assertLess(float((values - spikes * u_a).abs().max()), 1e-12)   # 发放时传出膜电位本身
        self.assertGreater(float(values.max()), 1.0)                    # 输出是实数，不是 0/1
        self.assertGreaterEqual(float(values[values > 0].min()), 1.0)   # 幅值不小于阈值，与脉冲同量级

    def test_network_flags_and_operations(self):
        for kind, spiking, stateful in (("lif", True, True), ("graded", False, True), ("relu", False, False)):
            net = EvSpSegNetStream(channels=SMALL, neuron=kind, merged_decoder=True)
            self.assertEqual((net.spiking, net.stateful), (spiking, stateful), kind)
            ops = estimate_operations(net, 16, 16, dict.fromkeys(LAYER_NAMES, 0.2), 5)
            if spiking:
                self.assertGreater(ops["sop"], 0.0)
            else:
                self.assertEqual(ops["sop"], 0.0, kind)
                self.assertAlmostEqual(ops["mac"], ops["dense_ops"], delta=1e-6)
            self.assertEqual(ops["state_elements"] > 0, stateful, kind)


class ThresholdSweepTests(unittest.TestCase):
    def test_event_metrics_and_interpolation(self):
        from tools import sweep_threshold as S
        labels = np.array([1, 1, 0, 0], dtype=np.float32)
        probs = np.array([0.95, 0.4, 0.92, 0.1], dtype=np.float32)
        m = S.event_metrics(labels, probs, 0.9)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 1))
        self.assertAlmostEqual(m["iou"], 1 / 3.0)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["precision"], 0.5)
        rows = [{"threshold": 0.9, "fa": 1e-6, "iou": 0.80, "recall": 0.7, "precision": 0.9, "pd": 0.85},
                {"threshold": 0.5, "fa": 1e-4, "iou": 0.70, "recall": 0.9, "precision": 0.6, "pd": 0.95}]
        point = S.interpolate_at_fa(rows, 1e-5)                 # log 空间正中间
        self.assertAlmostEqual(point["threshold"], 0.7, places=6)
        self.assertAlmostEqual(point["pd"], 0.90, places=6)
        self.assertIsNone(S.interpolate_at_fa(rows, 1e-9))      # 超出范围不外推

    def test_sweep_on_synthetic_dump(self):
        from tools import sweep_threshold as S
        import tempfile
        rng = np.random.RandomState(0)
        with tempfile.TemporaryDirectory() as directory:
            for i in range(2):
                n = 500
                labels = (rng.rand(n) < 0.2).astype(np.float32)
                probs = np.clip(labels * 0.6 + rng.rand(n) * 0.4, 0, 1).astype(np.float32)
                locs = np.stack([np.zeros(n), rng.randint(0, 346, n), rng.randint(0, 260, n),
                                 rng.randint(0, 8000, n)], 1).astype(np.int64)
                np.savez(os.path.join(directory, "seq%d.npz" % i), locs=locs, labels=labels,
                         probabilities=probs, target_id=labels.astype(np.float64))
            sequences = S.load_dump(directory)
            self.assertEqual(len(sequences), 2)
            rows = S.sweep(sequences, [0.5, 0.9], with_pd=False, pd_detT=50, correct_thresh=1e-4)
            self.assertEqual([r["threshold"] for r in rows], [0.5, 0.9])
            self.assertGreater(rows[0]["recall"], rows[1]["recall"])     # 阈值越高召回越低
            self.assertLessEqual(rows[0]["precision"], rows[1]["precision"])

class SummarizeRunsTests(unittest.TestCase):
    """汇总工具：分组、跨种子求均值、按训练时的状态模式取主指标、两种能耗口径。"""

    def write_eval(self, root, run, seed, split, iou, mode="carry", density=None):
        import json
        directory = os.path.join(root, "%s_seed%d" % (run, seed))
        os.makedirs(directory, exist_ok=True)
        net = EvSpSegNetStream(channels=SMALL, merged_decoder=True)
        ops = estimate_operations(net, 16, 16, dict.fromkeys(LAYER_NAMES, 0.1), 5, density)
        other = "reset_each_window" if mode == "carry" else "carry"
        payload = {"epoch": 7, "parameters": {"total": 123}, "trained_state_mode": mode, "neuron": "lif",
                   mode: {"iou": iou, "acc": 0.9, "pd": 0.94, "fa": 1e-5, "operations": ops,
                          "layers": {n: {"firing_rate": 0.1} for n in LAYER_NAMES},
                          "tau": {"enc1": {"tau_mean": 200.0}},
                          "segment_iou": {"0-15": iou - 0.05}, "latency": {"latency_median_ms": 55.0,
                                                                           "detection_rate": 1.0}},
                   other: {"iou": iou - 0.2}}
        if density is not None:
            payload[mode]["input_nonzero_fraction"] = density
        with open(os.path.join(directory, "eval_%s_best_val_iou_seed%d.json" % (split, seed)), "w",
                  encoding="utf-8") as stream:
            json.dump(payload, stream)

    def test_grouping_modes_and_energy(self):
        from tools import summarize_runs as S
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            self.write_eval(root, "runA", 37, "test", 0.80, density=0.001)
            self.write_eval(root, "runA", 38, "test", 0.84, density=0.001)
            self.write_eval(root, "runB", 37, "test", 0.50, mode="reset_each_window")
            self.write_eval(root, "runA", 37, "val", 0.88, density=0.001)
            runs = S.collect(root, "test", "")
            self.assertEqual(sorted(runs), ["runA", "runB"])
            self.assertEqual(len(runs["runA"]), 2)
            a = S.summarize(runs["runA"], 0.5, 160)
            self.assertAlmostEqual(a["iou"], 0.82)
            self.assertAlmostEqual(a["iou_std"], 0.0283, places=3)
            self.assertAlmostEqual(a["other_mode_iou"], 0.62)     # 另一种状态模式
            self.assertTrue(a["measured_density"])                # 用的是结果里实测的占比
            self.assertLess(a["energy_event_mj"], a["energy_dense_mj"])
            b = S.summarize(runs["runB"], 0.5, 160)
            self.assertEqual(b["state_mode"], "reset_each_window")
            self.assertAlmostEqual(b["iou"], 0.50)                # reset 训练的模型不按 carry 取主指标
            self.assertFalse(b["measured_density"])
            self.assertEqual(len(S.collect(root, "val", "")), 1)
            self.assertEqual(sorted(S.collect(root, "test", "runB")), ["runB"])

    def test_energy_matches_manual_formula(self):
        from tools import summarize_runs as S
        net = EvSpSegNetStream(channels=SMALL, merged_decoder=True)
        ops = estimate_operations(net, 16, 16, dict.fromkeys(LAYER_NAMES, 0.1), 5)
        dense, event = S.energy_mj_per_8s(ops, 0.01, 160)
        expected = (ops["mac"] * 4.6e-12 + ops["sop"] * 0.9e-12) * 1e3 * 160
        self.assertAlmostEqual(dense, expected, delta=1e-9)
        enc1 = [r for r in ops["per_layer"] if r["layer"] == "enc1"][0]
        mac_event = ops["mac"] - enc1["mac"] + enc1["dense"] * 0.01
        self.assertAlmostEqual(event, (mac_event * 4.6e-12 + ops["sop"] * 0.9e-12) * 1e3 * 160, delta=1e-9)


class InputDensityTests(unittest.TestCase):
    """评估时记录的非零输入占比：两种执行方式都要等于直接数出来的值。"""

    def test_counts_match_direct_count_in_both_executions(self):
        import train_stream_v1 as T
        seq = synthetic_sequence(seed=8)
        net = make_net(merged=True, seed=4).float().eval()
        source = NumpyWindowSource(seq, CFG, Q99, CPU)
        expected_nonzero = sum(int((source.window(k)[0] != 0).sum()) for k in range(WINDOWS))
        expected_total = WINDOWS * 12 * H * W
        for execution, input_device in (("step", "cpu"), ("layer", "gpu")):
            cfg = dict(TOOL_CFG, execution=execution, input_device=input_device)
            stats = {"nonzero": 0, "elements": 0}
            with torch.no_grad():
                T.run_sequence(net, seq, cfg, Q99, CPU, "carry", input_stats=stats)
            self.assertEqual(int(stats["nonzero"]), expected_nonzero, execution)
            self.assertEqual(stats["elements"], expected_total, execution)
            self.assertAlmostEqual(T.nonzero_fraction(stats), expected_nonzero / float(expected_total), places=12)

    def test_operations_carry_event_driven_accounting(self):
        net = EvSpSegNetStream(channels=SMALL, merged_decoder=True)
        rates = dict.fromkeys(LAYER_NAMES, 0.1)
        plain = estimate_operations(net, 16, 16, rates, 5)
        self.assertNotIn("mac_event_driven", plain)          # 不给占比时不额外报告
        with_density = estimate_operations(net, 16, 16, rates, 5, 0.01)
        enc1 = [r for r in with_density["per_layer"] if r["layer"] == "enc1"][0]
        self.assertAlmostEqual(with_density["mac_event_driven"],
                               plain["mac"] - enc1["mac"] + enc1["dense"] * 0.01, delta=1e-6)
        self.assertEqual(with_density["input_density"], 0.01)


if __name__ == "__main__":
    unittest.main()
