"""Contracts for the train-fitted covariance context head, not accuracy claims."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.covariance_revision import CovarianceRevision


def observations(n=512):
    y = (torch.arange(n) % 2).double()
    common = torch.randn(n, 6, dtype=torch.float64)
    h2 = common + .2 * torch.randn_like(common)
    h1 = .8 * common + .3 * torch.randn_like(common)
    h2[:, 0] += .4 * y
    h1[:, 2] += .7 * y
    z1, z2 = h1[:, 0].clone(), h2[:, 0].clone()
    return (z1, z2, h1, h2), y


def make(variant="full"):
    return CovarianceRevision(feature_dim=6, hidden=4, channels=3, kappa=.8,
                              variant=variant, shrinkage=.05).double()


class CovarianceRevisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.values, self.labels = observations()

    def fit(self, variant="full"):
        model = make(variant)
        report = model.fit_statistics(*self.values, self.labels)
        return model, report

    def test_training_statistics_define_spd_whitening_and_schur_identity(self):
        model, report = self.fit()
        covariance, transform = model.regularized_covariance, model.joint_transform
        self.assertGreater(report["minimum_eigenvalue"], 0.)
        self.assertLess(report["whitening_error_float64"], 1e-9)
        self.assertLess(report["fisher_decomposition_error"], 1e-9)
        self.assertGreaterEqual(report["fisher_conditional_increment"], 0.)
        self.assertTrue(torch.allclose(transform.mm(covariance).mm(transform.t()),
                                       torch.eye(6, dtype=torch.float64), atol=1e-9))
        # The residual is decorrelated from the second measurement under the
        # fitted regularized within-class covariance, not independent of it.
        residual = torch.cat((-model.cross_regression, torch.eye(3, dtype=torch.float64)), 1)
        self.assertLess(float(residual.mm(covariance)[:, :3].abs().max()), 1e-9)
        self.assertTrue(torch.allclose(model.projection.t().mm(model.projection),
                                       torch.eye(3, dtype=torch.float64), atol=1e-9))

    def test_zero_initialization_and_cache_path_preserve_second_score(self):
        for variant in CovarianceRevision.VARIANTS:
            with self.subTest(variant=variant):
                model, _ = self.fit(variant)
                actual = model(*self.values)
                self.assertTrue(torch.equal(actual, self.values[1]))
                if variant != "second":
                    q = model.encode_pair(*self.values)
                    self.assertTrue(torch.equal(actual, model.forward_encoded(q, self.values[1])))

    def test_unfitted_learned_head_fails_in_training_and_eval(self):
        model = make()
        for mode in (True, False):
            model.train(mode)
            with self.assertRaisesRegex(RuntimeError, "training-only"):
                model(*self.values)
        self.assertTrue(torch.equal(make("second")(*self.values), self.values[1]))

    def test_single_view_control_does_not_depend_on_first_observation(self):
        model, _ = self.fit("no_context")
        changed = (self.values[0] * 19., self.values[1], self.values[2] - 100., self.values[3])
        with torch.no_grad():
            model.output.weight.normal_()
        self.assertTrue(torch.equal(model(*self.values), model(*changed)))
        other = make("no_context")
        other.fit_statistics(*changed, self.labels)
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value, dict(other.named_buffers())[name]), name)

    def test_covariance_ablation_has_same_trainable_capacity(self):
        model, _ = self.fit()
        ablated, report = self.fit("no_revision")
        self.assertEqual(sum(x.numel() for x in model.parameters()),
                         sum(x.numel() for x in ablated.parameters()))
        self.assertTrue(torch.equal(model.projection, ablated.projection))
        self.assertTrue(torch.equal(model.regularized_covariance, ablated.regularized_covariance))
        self.assertFalse(torch.allclose(model.encode_pair(*self.values), ablated.encode_pair(*self.values)))
        self.assertEqual(report["coordinate_transform"], "diagonal_standardization")
        z1, z2, h1, h2 = self.values
        project = ablated.projection / ablated.raw_scale[:, None]
        first = (torch.cat((h1, z1[:, None]), 1) - ablated.raw_mean).mm(project)
        second = (torch.cat((h2, z2[:, None]), 1) - ablated.raw_mean).mm(project)
        scale = torch.diag(ablated.regularized_covariance).sqrt()
        expected = (torch.cat((second, first), 1) - ablated.joint_mean) / scale
        self.assertTrue(torch.equal(ablated.encode_pair(*self.values), expected))
        normalized_cov = ablated.regularized_covariance / (scale[:, None] * scale[None, :])
        self.assertTrue(torch.allclose(torch.diag(normalized_cov), torch.ones_like(scale)))
        self.assertGreater(float((normalized_cov - torch.diag(torch.diag(normalized_cov))).abs().max()), .1)

    def test_gradients_reach_both_views_and_head_after_initial_update(self):
        model, _ = self.fit()
        trainable = tuple(v.clone().requires_grad_(True) for v in self.values)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        (model(*trainable) - self.labels * 3.).square().mean().backward()
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.)
        optimizer.step()
        optimizer.zero_grad()
        for value in trainable:
            value.grad = None
        (model(*trainable) - self.labels * 3.).square().mean().backward()
        for value in trainable:
            self.assertIsNotNone(value.grad)
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
            self.assertGreater(float(value.grad.abs().sum()), 0.)
        self.assertGreater(float(model.input.weight.grad.abs().sum()), 0.)

    def test_forward_never_refits_buffers_and_update_is_bounded(self):
        model, _ = self.fit()
        before = {name: value.clone() for name, value in model.named_buffers()}
        with torch.no_grad():
            model.output.weight.fill_(50.)
        for training in (True, False):
            model.train(training)
            _, info = model(*self.values, return_info=True)
            self.assertLessEqual(float(info["delta"].abs().max()), model.kappa)
            for name, value in model.named_buffers():
                self.assertTrue(torch.equal(before[name], value), name)

    def test_serialized_model_restores_fitted_state_and_exact_outputs(self):
        model, _ = self.fit()
        with torch.no_grad():
            model.output.weight.normal_()
        loaded = make()
        loaded.load_state_dict(model.state_dict())
        self.assertTrue(loaded._is_fitted)
        self.assertTrue(torch.equal(model(*self.values), loaded(*self.values)))

    def test_bad_statistics_are_rejected_before_installation(self):
        model = make()
        with self.assertRaisesRegex(ValueError, "two training"):
            model.fit_statistics(*self.values, torch.zeros_like(self.labels))
        broken = list(self.values)
        broken[2] = broken[2].clone()
        broken[2][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            model.fit_statistics(*broken, self.labels)
        self.assertFalse(model._is_fitted)

    def test_variance_penalty_constrains_both_linear_layers(self):
        model, _ = self.fit()
        with torch.no_grad():
            model.output.weight.fill_(.2)
        penalty = model.variance_regularizer()
        penalty.backward()
        self.assertGreater(float(model.input.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.)
        self.assertEqual(make("second").macs_per_pair(), 0)
        self.assertGreater(model.macs_per_pair(), 0)


if __name__ == "__main__":
    unittest.main()
