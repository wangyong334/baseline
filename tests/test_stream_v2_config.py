"""V2 入口脚本（train_stream_v2.py）配置检查的单元测试。

09-23 服务器上 `--neuron-u-floor` 报 unrecognized arguments：服务器上的 train_stream_v2.py 还是旧版本。
第一组测试直接走命令行解析，只要测试文件是新的、入口脚本是旧的，这组测试就会失败——
所以"启动训练前先跑测试并核对测试数"能拦住这类同步问题。其余几组守住三件事：
    1. 没有意义的配置组合在构造模块之前就报错（ReLU / reset 模式加膜电位界、损失权重全为 0）
    2. 续训时配置与 checkpoint 不一致会被发现（膜电位上下界不在 state_dict 里，加载权重时查不出来）
    3. eval 模式下给了"需要重新训练"的参数且与 checkpoint 不同时报错，而不是静默忽略、把结果贴错标签
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import torch

import train_stream_v2 as tv2

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs",
                      "evisseg_stream_v2.yaml")


def parse(*extra):
    """按命令行解析 + build_config，返回 (args, cfg)。"""
    argv = ["train_stream_v2.py", "--config", CONFIG, "--mode", "train"] + list(extra)
    with mock.patch.object(sys, "argv", argv):
        args = tv2.parse_args()
    return args, tv2.build_config(args)


class CommandLineTests(unittest.TestCase):
    def test_bound_flags_are_recognised(self):
        """回归 09-23：--neuron-u-floor / --neuron-u-ceil 必须被识别并写进配置（旧入口脚本在这里就会失败）。"""
        _, cfg = parse("--neuron-u-floor", "-1", "--neuron-u-ceil", "4")
        self.assertEqual(cfg["neuron_u_floor"], -1.0)
        self.assertEqual(cfg["neuron_u_ceil"], 4.0)

    def test_defaults_leave_bounds_unset(self):
        _, cfg = parse()
        self.assertIsNone(cfg["neuron_u_floor"])
        self.assertIsNone(cfg["neuron_u_ceil"])

    def test_bounds_reach_every_neuron(self):
        """命令行 -> 配置 -> build_model -> 七个神经元，整条链路都要通。"""
        _, cfg = parse("--neuron-u-floor", "-2")
        model = tv2.build_model(cfg, tv2.build_frontend(cfg))
        for block in model.blocks():
            self.assertAlmostEqual(block.neuron.u_floor, -2.0 * float(cfg["v_threshold"]), places=12)
            self.assertIsNone(block.neuron.u_ceil)


class ValidateConfigTests(unittest.TestCase):
    def test_relu_with_bounds_is_rejected(self):
        with self.assertRaises(ValueError):
            parse("--neuron", "relu", "--neuron-u-floor", "-1")

    def test_reset_mode_with_bounds_is_rejected(self):
        with self.assertRaises(ValueError):
            parse("--state-mode", "reset_each_window", "--neuron-u-ceil", "4")

    def test_graded_with_bounds_is_allowed(self):
        """graded 与 LIF 的膜电位轨迹相同，加界有意义。"""
        _, cfg = parse("--neuron", "graded", "--neuron-u-floor", "-1")
        self.assertEqual(cfg["neuron_u_floor"], -1.0)

    def test_loss_weights(self):
        """权重不能为负、不能同时为 0；单头消融（intensity 权重 0）是合法的。"""
        with self.assertRaises(ValueError):
            parse("--loss-mark-weight", "0", "--loss-intensity-weight", "0")
        with self.assertRaises(ValueError):
            parse("--loss-intensity-weight", "-1")
        _, cfg = parse("--loss-intensity-weight", "0")
        self.assertEqual(cfg["loss_intensity_weight"], 0.0)


class ConfigDriftTests(unittest.TestCase):
    def setUp(self):
        _, self.cfg = parse()

    def test_identical_config_has_no_drift(self):
        self.assertEqual(tv2.config_drift(dict(self.cfg), self.cfg, tv2.TRAINING_KEYS), {})

    def test_forgotten_floor_is_detected(self):
        """原训练带了下界、续训命令漏了：必须被发现（load_state_dict 发现不了）。"""
        drift = tv2.config_drift(dict(self.cfg, neuron_u_floor=-1.0), self.cfg, tv2.TRAINING_KEYS)
        self.assertEqual(drift, {"neuron_u_floor": (-1.0, None)})

    def test_old_checkpoint_without_new_keys(self):
        """后来才加的键在旧 checkpoint 里不存在：按加入之前的行为比较，不误报。"""
        saved = {k: v for k, v in self.cfg.items() if k not in tv2.CONFIG_DEFAULTS}
        self.assertEqual(tv2.config_drift(saved, self.cfg, tv2.TRAINING_KEYS), {})

    def test_types_are_normalised(self):
        """元组与列表、整数与浮点视为相同。"""
        saved = dict(self.cfg, channels=tuple(self.cfg["channels"]), v_threshold=int(self.cfg["v_threshold"]))
        self.assertEqual(tv2.config_drift(saved, self.cfg, tv2.TRAINING_KEYS), {})

    def test_resume_checks_before_writing_anything(self):
        """load_resume_checkpoint：配置不一致时报错并指出是哪一项，且目录里不多出任何文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(self.cfg, save_root=tmp)
            payload = {"config": dict(cfg, neuron_u_floor=-1.0), "model": {}, "optimizer": {}, "epoch": 3,
                       "best_val_iou": 0.5}
            torch.save(payload, os.path.join(tmp, "last.pt"))
            with self.assertRaises(RuntimeError) as ctx:
                tv2.load_resume_checkpoint(cfg, torch.device("cpu"))
            self.assertIn("neuron_u_floor", str(ctx.exception))
            self.assertEqual(os.listdir(tmp), ["last.pt"])
            ckpt = tv2.load_resume_checkpoint(dict(cfg, neuron_u_floor=-1.0), torch.device("cpu"))
            self.assertEqual(ckpt["epoch"], 3)

    def test_resume_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                tv2.load_resume_checkpoint(dict(self.cfg, save_root=tmp), torch.device("cpu"))


class EvalConflictTests(unittest.TestCase):
    def test_retrain_flag_differing_from_checkpoint_is_reported(self):
        args, cfg = parse("--neuron-u-floor", "-1")
        conflicts = tv2.eval_conflicts(args, cfg, dict(cfg, neuron_u_floor=None))
        self.assertEqual(len(conflicts), 1)
        self.assertIn("--neuron-u-floor", conflicts[0])

    def test_matching_or_absent_flags_are_fine(self):
        args, cfg = parse("--neuron-u-floor", "-1")
        self.assertEqual(tv2.eval_conflicts(args, cfg, dict(cfg)), [])
        args, cfg = parse()                                    # 命令行没给就不检查，一律取 checkpoint 的值
        self.assertEqual(tv2.eval_conflicts(args, cfg, dict(cfg, neuron_u_floor=-1.0)), [])

    def test_decision_layer_switches_are_not_conflicts(self):
        """判决层参数本来就是评估时开关（同一 checkpoint 换 CUSUM 设置），不在检查范围内。"""
        args, cfg = parse("--cusum-velocities", "0", "--cusum-aggregate", "sum")
        self.assertEqual(tv2.eval_conflicts(args, cfg, dict(cfg, cusum_axis_velocities=[-1, 0, 1])), [])


if __name__ == "__main__":
    unittest.main()
