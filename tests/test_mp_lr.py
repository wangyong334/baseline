"""Dependency-free policy tests. These do not validate CUDA training."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "train_mp.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"), feature_version=(3, 8))
FUNCTIONS = [node for node in TREE.body if isinstance(node, ast.FunctionDef)
             and node.name in ("linear_epoch_lr", "lr_policy")]
NAMESPACE = {"math": math}
exec(compile(ast.Module(body=FUNCTIONS, type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)
linear_epoch_lr = NAMESPACE["linear_epoch_lr"]
lr_policy = NAMESPACE["lr_policy"]


class LearningRateTests(unittest.TestCase):
    def test_endpoints_and_no_undershoot(self):
        self.assertEqual(linear_epoch_lr(0, 50, 0.001, 0.0001), 0.001)
        self.assertEqual(linear_epoch_lr(49, 50, 0.001, 0.0001), 0.0001)
        self.assertEqual(linear_epoch_lr(50, 50, 0.001, 0.0001), 0.0001)

    def test_fifty_epoch_sequence(self):
        rates = [linear_epoch_lr(e, 50, 0.001, 0.0001) for e in range(50)]
        for a, b in zip(rates, rates[1:]):
            self.assertGreater(a, b)
            self.assertAlmostEqual(a - b, 0.0009 / 49, places=15)

    def test_legacy_default(self):
        policy = lr_policy(SimpleNamespace(lr=0.001, epochs=50))
        self.assertEqual(policy, {"name": "step", "start_lr": 0.001,
                                  "step_size": 10, "gamma": 0.1})

    def test_explicit_linear(self):
        policy = lr_policy(SimpleNamespace(lr_schedule="linear", lr=0.001,
                                          lr_end=0.0001, epochs=50))
        self.assertEqual(policy["name"], "linear")
        self.assertEqual(policy["end_lr"], 0.0001)

    def test_invalid_config(self):
        for fields in ({"lr_schedule": "unknown"},
                       {"lr_schedule": "linear"},
                       {"lr_schedule": "linear", "lr_end": 0},
                       {"lr_schedule": "linear", "lr_end": float("nan")},
                       {"lr_schedule": "linear", "lr_end": float("inf")},
                       {"lr_schedule": "linear", "lr_end": 0.01},
                       {"lr_schedule": "linear", "lr_end": 0.0001, "epochs": 1}):
            values = {"lr": 0.001, "epochs": 50}
            values.update(fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                lr_policy(SimpleNamespace(**values))


if __name__ == "__main__":
    unittest.main()
