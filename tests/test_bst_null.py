"""Zero-valued null evidence is an abstention path, not a calibration claim."""
import unittest

import torch

from model.bst_window import WindowEvidence


def run(block, weights, values, context, dw=None, mask=None, query=None):
    shape = weights.shape[:2]
    if dw is None:
        dw = torch.zeros(shape, dtype=context.dtype)
    if mask is None:
        mask = torch.ones(shape, dtype=torch.bool)
    if query is None:
        query = torch.zeros(context.shape[0], context.shape[-1], dtype=context.dtype)
    grouping = block.summarize_visible(context, mask, dw)
    return block(weights, values, context, query, grouping), grouping


def close(left, right, atol=1e-7, rtol=1e-6):
    torch.testing.assert_allclose(left, right, atol=atol, rtol=rtol)


class NullEvidenceTests(unittest.TestCase):
    def test_zero_affinity_is_zero_evidence_even_with_discount_bias(self):
        block = WindowEvidence(3, 2, 1, 2, null_evidence=True)
        with torch.no_grad():
            block.discount[-1].bias.fill_(12.)
        weights = torch.zeros(1, 4, 2, requires_grad=True)
        values = torch.tensor([[[1., -1.], [3., 2.], [4., 5.], [8., 7.]]])
        context = torch.ones(1, 4, 3)
        result, grouping = run(block, weights, values, context, dw=torch.tensor([[-1., 0., 0., 1.]]))
        for name in ("head_correction", "summary", "mass_h", "neff", "effective_weight",
                     "window_support", "effective_window_mass", "context_support"):
            close(result[name], torch.zeros_like(result[name]), atol=0, rtol=0)
        close(result["null_mass"][grouping["active"]], torch.ones(3, 2), atol=0, rtol=0)
        close(result["null_mass"][~grouping["active"]], torch.zeros(1, 2), atol=0, rtol=0)
        result["head_correction"].sum().backward()
        self.assertTrue(bool(torch.isfinite(weights.grad).all()))
        self.assertTrue(bool(torch.isfinite(block.raw_null.grad).all()))

    def test_uniformly_weak_affinities_vanish_without_becoming_uniform_evidence(self):
        block = WindowEvidence(2, 1, 0, 0, gate=False, bounded=False, null_evidence=True).double()
        base = torch.tensor([[[.25], [.75]]], dtype=torch.float64)
        values = torch.tensor([[[2.], [4.]]], dtype=torch.float64)
        context = torch.tensor([[[1., 2.], [3., 4.]]], dtype=torch.float64)
        results = [run(block, base*scale, values, context)[0] for scale in (1., 1e-3, 1e-8, 0.)]
        for key in ("head_correction", "summary", "mass_h", "context_support"):
            magnitudes = [float(result[key].abs().max()) for result in results]
            self.assertTrue(all(a > b for a, b in zip(magnitudes, magnitudes[1:])), (key, magnitudes))
            self.assertLess(magnitudes[2], 3e-8)
        # No term should secretly renormalize the accepted mass back to one.
        close(results[0]["window_support"], torch.tensor([[[1./3.]]], dtype=torch.float64))
        close(results[0]["head_correction"], torch.tensor([[7./6.]], dtype=torch.float64))
        close(results[0]["summary"], torch.tensor([[5./6., 7./6.]], dtype=torch.float64))

    def test_complete_window_replication_preserves_all_applied_outputs(self):
        torch.manual_seed(37)
        block = WindowEvidence(3, 2, 1, 2, null_evidence=True)
        weights = torch.rand(1, 5, 2)
        values = torch.randn(1, 5, 2)
        context = torch.randn(1, 5, 3)
        dw = torch.tensor([[-1., 0., 0., 1., 2.]])
        copies = torch.tensor([0, 1, 2, 1, 2, 1, 2, 3, 4])
        left, groups_left = run(block, weights, values, context, dw)
        right, groups_right = run(block, weights[:, copies], values[:, copies], context[:, copies], dw[:, copies])
        for key in ("head_correction", "summary", "mass_h", "neff", "window_discount",
                    "window_value", "units", "window_support", "null_mass", "effective_window_mass",
                    "context_support"):
            close(left[key], right[key])
        close(groups_left["balanced"], groups_right["balanced"])

    def test_null_mass_is_not_multiplied_twice_and_context_is_subprobability_mixture(self):
        block = WindowEvidence(2, 1, 0, 1, gate=False, bounded=False, null_evidence=True)
        weights = torch.ones(1, 2, 1)
        values = torch.tensor([[[2.], [4.]]])
        context = torch.tensor([[[2., 0.], [0., 4.]]])
        result, _ = run(block, weights, values, context, dw=torch.tensor([[0., 1.]]))
        # nu=1 and one source/window: each window has support 1/2.
        close(result["head_correction"], torch.tensor([[3.]]))
        close(result["summary"], torch.tensor([[.5, 1.]]))
        close(result["mass_h"], torch.tensor([[1.]]))
        close(result["context_support"], torch.tensor([.5]))
        close(result["head_correction"], (result["effective_weight"]*values).sum(1))

    def test_empty_masks_and_empty_token_arrays_are_finite_and_zero(self):
        block = WindowEvidence(2, 2, 1, 1, null_evidence=True)
        for size in (0, 3):
            weights = torch.ones(2, size, 2)
            context = torch.zeros(2, size, 2)
            result, _ = run(block, weights, weights, context, mask=torch.zeros(2, size, dtype=torch.bool))
            for name in ("head_correction", "summary", "mass_h", "neff", "window_support", "null_mass"):
                self.assertTrue(bool(torch.isfinite(result[name]).all()))
                close(result[name], torch.zeros_like(result[name]), atol=0, rtol=0)

    def test_bounded_values_and_null_never_increase_window_contribution_cap(self):
        torch.manual_seed(2)
        block = WindowEvidence(3, 2, 0, 2, null_evidence=True)
        weights = torch.rand(2, 9, 2)
        values = block.bound_values(30*torch.randn(2, 9, 2))
        context = torch.randn(2, 9, 3)
        dw = torch.tensor([[0., 0., 0., 1., 1., 1., 2., 2., 2.]]).expand(2, -1)
        result, _ = run(block, weights, values, context, dw)
        scale = torch.nn.functional.softplus(block.raw_scale)
        self.assertTrue(bool((result["window_value"].abs() <= result["window_support"]*scale[None, None]+1e-6).all()))
        self.assertTrue(bool((result["head_correction"].abs() <= result["mass_h"]*scale[None]+1e-6).all()))
        self.assertTrue(bool((result["context_support"] >= 0).all()))
        self.assertTrue(bool((result["context_support"] <= 1).all()))

    def test_extreme_sigmoid_and_null_parameters_have_finite_gradients(self):
        for null_raw in (-100., 0., 100.):
            block = WindowEvidence(2, 1, 0, 0, null_evidence=True)
            with torch.no_grad():
                block.raw_null.fill_(null_raw)
            logits = torch.tensor([[[-100.], [-89.], [-1.], [12.]]], requires_grad=True)
            values = torch.tensor([[[1.], [-2.], [3.], [-4.]]], requires_grad=True)
            context = torch.ones(1, 4, 2, requires_grad=True)
            result, _ = run(block, logits.sigmoid(), values, context)
            objective = result["head_correction"].sum() + result["summary"].sum() + result["mass_h"].sum()
            objective.backward()
            for gradient in (logits.grad, values.grad, context.grad, block.raw_null.grad):
                self.assertTrue(bool(torch.isfinite(gradient).all()), null_raw)

    def test_opt_out_preserves_old_state_dict_and_exact_outputs(self):
        torch.manual_seed(9)
        old = WindowEvidence(2, 2, 0, 1)
        explicit = WindowEvidence(2, 2, 0, 1, null_evidence=False)
        explicit.load_state_dict(old.state_dict(), strict=True)
        self.assertNotIn("raw_null", old.state_dict())
        weights = torch.tensor([[[1e-30, .3], [.5, .8], [.7, .9]]])
        values = torch.randn(1, 3, 2)
        context = torch.randn(1, 3, 2)
        dw = torch.tensor([[0., 0., 1.]])
        a, _ = run(old, weights, values, context, dw)
        b, _ = run(explicit, weights, values, context, dw)
        self.assertEqual(set(a), set(b))
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)
        enabled = WindowEvidence(2, 2, 0, 1, null_evidence=True)
        self.assertEqual(sum(p.numel() for p in enabled.parameters())-sum(p.numel() for p in old.parameters()), 2)


if __name__ == "__main__":
    unittest.main()
