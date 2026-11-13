"""Contracts for bounded paired evidence, independent ablations and gradients."""
import os
import sys
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.context_evidence import ContextEvidence


def make(variant="full", device="cpu", dtype=torch.float64):
    return ContextEvidence(6, 4, 0.75, variant).to(device=device, dtype=dtype)


def observations(n=23, device="cpu", dtype=torch.float64, requires_grad=False):
    return tuple(torch.randn(*shape, device=device, dtype=dtype, requires_grad=requires_grad)
                 for shape in ((n,), (n,), (n, 6), (n, 6)))


def activate(model):
    with torch.no_grad():
        if model.has_context:
            model.reliability[-1].weight.normal_(0, .2)
        if model.has_revision:
            model.output.weight.normal_(0, .4)


class ContextEvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_initialization_and_real_ablation_bypasses(self):
        z1, z2, h1, h2 = observations()
        for variant in ContextEvidence.VARIANTS:
            model = make(variant)
            result, info = model(z1, z2, h1, h2, return_info=True)
            expected = (z1 + z2) / 2 if model.has_context else z2
            self.assertTrue(torch.allclose(result, expected, atol=1e-14, rtol=1e-14))
            self.assertTrue(torch.equal(info["delta"], torch.zeros_like(z2)))
            self.assertEqual(hasattr(model, "reliability"), model.has_context)
            self.assertEqual(hasattr(model, "difference"), model.has_revision)
            if variant == "second":
                self.assertEqual(sum(p.numel() for p in model.parameters()), 0)
                self.assertEqual(model.macs_per_pair(), 0)
                self.assertIs(result, z2)

    def test_consensus_symmetric_convex_and_duplicate_idempotent(self):
        model = make("no_revision")
        activate(model)
        z1, z2, h1, h2 = observations()
        result, info = model(z1, z2, h1, h2, return_info=True)
        swapped = model(z2, z1, h2, h1)
        self.assertTrue(torch.allclose(result, swapped, atol=1e-14, rtol=1e-14))
        self.assertTrue(bool((result >= torch.minimum(z1, z2) - 1e-14).all()))
        self.assertTrue(bool((result <= torch.maximum(z1, z2) + 1e-14).all()))
        self.assertTrue(bool(((info["alpha_first"] >= 0) & (info["alpha_first"] <= 1)).all()))
        duplicate = model(z1, z1, h1, h2)
        self.assertTrue(torch.equal(duplicate, z1))

    def test_equal_hidden_difference_is_exactly_zero_even_with_different_scores(self):
        for variant in ("full", "no_context"):
            model = make(variant)
            activate(model)
            z1, z2, h1, _ = observations()
            _, info = model(z1, z2, h1, h1.clone(), return_info=True)
            self.assertTrue(torch.equal(info["delta"], torch.zeros_like(z1)))
            same = model(z1, z1, h1, h1.clone())
            self.assertTrue(torch.equal(same, z1))

    def test_revision_is_bounded_and_can_have_both_signs(self):
        model = make("no_context")
        with torch.no_grad():
            model.condition.weight.zero_()
            model.condition.bias.fill_(10)
            model.difference.weight.zero_()
            model.difference.weight[0, 0] = 1
            model.output.weight.zero_()
            model.output.weight[0, 0] = 100
        z1, z2, h1, h2 = observations()
        h1.zero_()
        h2.zero_()
        h2[:, 0] = torch.linspace(-2, 2, len(z1))
        result, info = model(z1, z2, h1, h2, return_info=True)
        self.assertLessEqual(float(info["delta"].detach().abs().max()), model.kappa)
        self.assertLess(float(info["delta"].detach().min()), 0.)
        self.assertGreater(float(info["delta"].detach().max()), 0.)
        self.assertTrue(torch.allclose(result - z2, info["delta"], atol=1e-14, rtol=1e-14))

    def test_gradients_reach_both_views_and_revision_after_initial_update(self):
        model = make()
        z1, z2, h1, h2 = observations(requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=.2)
        target = torch.linspace(-1, 1, len(z1), dtype=z1.dtype)
        initial = model.output.weight.clone()
        loss = (model(z1, z2, h1, h2) - target).square().mean()
        loss.backward()
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.)
        optimizer.step()
        self.assertFalse(torch.equal(model.output.weight, initial))
        optimizer.zero_grad()
        for value in (z1, z2, h1, h2):
            value.grad.zero_()
        (model(z1, z2, h1, h2) - target).square().mean().backward()
        for value in (z1, z2, h1, h2):
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
            self.assertGreater(float(value.grad.abs().sum()), 0.)
        self.assertGreater(float(model.difference.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.condition.weight.grad.abs().sum()), 0.)

    def test_batch_chunking_empty_and_singleton_agree(self):
        for variant in ContextEvidence.VARIANTS:
            model = make(variant)
            activate(model)
            values = observations()
            batch = model(*values)
            scalar = torch.cat([model(*(v[i:i+1] for v in values)) for i in range(len(batch))])
            self.assertTrue(torch.allclose(batch, scalar, atol=1e-13, rtol=1e-13))
            empty, info = model(*(v[:0] for v in values), return_info=True)
            self.assertEqual(tuple(empty.shape), (0,))
            self.assertTrue(all(tuple(v.shape) == (0,) for v in info.values()))

    def test_shared_train_normalization_and_checkpoint_round_trip(self):
        model = make()
        activate(model)
        mean = torch.linspace(-1, 1, 6, dtype=torch.float64)
        std = torch.linspace(.5, 2, 6, dtype=torch.float64)
        model.set_normalization(mean, std)
        values = observations()
        result = model(*values)
        restored = make()
        restored.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(result, restored(*values)))
        same = model(values[0], values[0], values[2], values[2])
        self.assertTrue(torch.equal(same, values[0]))
        for invalid in (torch.zeros(6), torch.full((6,), float("nan"))):
            with self.assertRaises(ValueError):
                model.set_normalization(mean, invalid)
        with self.assertRaises(ValueError):
            model.set_normalization(torch.zeros(5), torch.ones(5))

    def test_cost_counts_only_instantiated_operations(self):
        model = ContextEvidence()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 833)
        self.assertEqual(model.macs_per_pair(), 1088)
        self.assertEqual(model.estimated_macs(100), 108800)
        self.assertEqual(ContextEvidence(variant="no_context").macs_per_pair(), 544)
        self.assertEqual(ContextEvidence(variant="no_revision").macs_per_pair(), 544)
        with self.assertRaises(ValueError):
            model.estimated_macs(-1)

    def test_input_contract_and_configuration_errors(self):
        for kwargs in ({"feature_dim": 0}, {"hidden": 1.5}, {"kappa": 0},
                       {"kappa": float("inf")}, {"variant": "bad"}):
            with self.assertRaises(ValueError):
                ContextEvidence(**kwargs)
        model = make()
        z1, z2, h1, h2 = observations()
        for values in ((z1[:, None], z2, h1, h2), (z1, z2[:-1], h1, h2),
                       (z1, z2, h1[:, :5], h2), (z1, z2.float(), h1, h2)):
            with self.assertRaises(ValueError):
                model(*values)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_deterministic_backward_and_cpu_agreement(self):
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            cpu = make(dtype=torch.float32)
            activate(cpu)
            gpu = make(device="cuda", dtype=torch.float32)
            gpu.load_state_dict(cpu.state_dict())
            values = observations(dtype=torch.float32)
            cuda_values = tuple(v.cuda().requires_grad_() for v in values)
            expected = cpu(*values)
            actual = gpu(*cuda_values)
            self.assertTrue(torch.allclose(expected, actual.cpu(), atol=2e-6, rtol=2e-6))
            actual.square().sum().backward()
            self.assertTrue(all(bool(torch.isfinite(v.grad).all()) for v in cuda_values))
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == "__main__":
    unittest.main()
