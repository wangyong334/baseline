"""CPU-only helper tests without importing CUDA dataset/model modules."""
import ast
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch required for checkpoint tests")
class TestRecovery(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "train_snn_v0.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"atomic_save", "cpu_tree", "rng_state", "restore_rng", "validation_preflight"}
        extracted = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        self.scope = {"torch": torch, "np": np, "random": random, "os": os}
        exec(compile(extracted, str(source), "exec"), self.scope)

    def test_atomic_save_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recovery.pt'
            self.scope['atomic_save']({'epoch': 7}, path)
            with mock.patch.object(torch, 'save', side_effect=OSError('simulated disk failure')):
                with self.assertRaises(OSError):
                    self.scope['atomic_save']({'epoch': 8}, path)
            self.assertEqual(torch.load(str(path))['epoch'], 7)

    def test_adam_roundtrip_next_update(self):
        torch.manual_seed(37)
        net = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(net.parameters(), lr=.001)
        x = torch.tensor([[1., 2.]])
        def step(model, optim):
            optim.zero_grad()
            model(x).square().sum().backward()
            optim.step()
        step(net, optimizer)
        state = self.scope['cpu_tree']({'model': net.state_dict(), 'optimizer': optimizer.state_dict()})
        step(net, optimizer)
        restored = torch.nn.Linear(2, 1)
        restored.load_state_dict(state['model'])
        other = torch.optim.Adam(restored.parameters(), lr=.001)
        other.load_state_dict(state['optimizer'])
        step(restored, other)
        for a, b in zip(net.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_preflight_restores_rng_and_bn_on_failure(self):
        net = torch.nn.BatchNorm1d(2).train()
        original = {k: v.clone() for k, v in net.state_dict().items()}
        torch_rng = torch.get_rng_state().clone()
        def evaluate(model, mode):
            model.train()
            model(torch.rand(4, 2))
            raise RuntimeError('simulated validation failure')
        self.scope.update(cpu_state=lambda m: {k: v.clone() for k,v in m.state_dict().items()},
                          evaluate=evaluate, emit=lambda x: None, clear_unused=lambda: None)
        with mock.patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), mock.patch.object(torch.cuda, 'set_rng_state_all'):
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                self.scope['validation_preflight'](net)
        self.assertTrue(net.training)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        for k, value in net.state_dict().items():
            self.assertTrue(torch.equal(value, original[k]))


if __name__ == '__main__':
    unittest.main()
