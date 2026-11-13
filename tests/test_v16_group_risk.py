"""Recording-balanced adaptation objectives and independent FN constraints."""
import math
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from tools.v16.train_covariance import (
    fit_cached_head, record_class_means, record_constrained_loss,
    record_loss_weights, record_stationary_multipliers, update_record_multipliers,
)


class RecordingRiskTests(unittest.TestCase):
    def setUp(self):
        self.boundary = math.log(9.)
        self.logits = torch.tensor([2., 1., -1., -2., .3, -.4], dtype=torch.float64) + self.boundary
        self.labels = torch.tensor([1., 1., 0., 0., 1., 0.], dtype=torch.float64)
        self.ids = torch.tensor([0, 0, 0, 0, 1, 1])
        self.multipliers = torch.tensor([.4, .7], dtype=torch.float64)
        self.reference = torch.tensor([.1, .2], dtype=torch.float64)

    def weights(self, logits=None, labels=None, ids=None):
        logits = self.logits if logits is None else logits
        labels = self.labels if labels is None else labels
        ids = self.ids if ids is None else ids
        (_, positive), (_, negative) = record_class_means(logits, labels, ids, 2)
        return record_loss_weights(positive, negative, len(logits))

    def loss(self, logits, labels, ids, weights):
        return record_constrained_loss(logits, labels, ids, self.multipliers,
                                       weights[0], weights[1], self.reference)

    def test_objective_equals_class_means_with_each_record_counted_once(self):
        actual = self.loss(self.logits, self.labels, self.ids, self.weights())
        fp, fn = [], []
        for j in range(2):
            margin = self.logits[self.ids == j] - self.boundary
            positive = self.labels[self.ids == j] > .5
            fp.append(F.softplus(margin[~positive]).mean())
            fn.append(F.softplus(-margin[positive]).mean())
        expected = torch.stack(fp).mean() + (self.multipliers * (torch.stack(fn)-self.reference)).mean()
        self.assertAlmostEqual(float(actual), float(expected), places=12)

    def test_unequal_minibatches_recover_full_objective_and_gradient(self):
        logits = self.logits.clone().requires_grad_(True)
        weights = self.weights()
        whole = self.loss(logits, self.labels, self.ids, weights)
        whole_gradient, = torch.autograd.grad(whole, logits)
        total = logits.sum() * 0.
        for take in (slice(0, 1), slice(1, 5), slice(5, 6)):
            size = len(logits[take])
            total = total + size / float(len(logits)) * self.loss(
                logits[take], self.labels[take], self.ids[take], weights)
        gradient, = torch.autograd.grad(total, logits)
        self.assertAlmostEqual(float(whole), float(total), places=12)
        self.assertTrue(torch.allclose(whole_gradient, gradient, atol=1e-12, rtol=0.))

    def test_duplicating_events_in_one_record_does_not_change_record_weight(self):
        take = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4, 5])
        z, y, ids = self.logits[take], self.labels[take], self.ids[take]
        old = self.loss(self.logits, self.labels, self.ids, self.weights())
        new = self.loss(z, y, ids, self.weights(z, y, ids))
        self.assertAlmostEqual(float(old), float(new), places=12)

    def test_one_record_improvement_cannot_cancel_another_dual_violation(self):
        initial = torch.tensor([1., 1.])
        reference = torch.tensor([.4, .4])
        risks = torch.tensor([.6, .2])
        result = update_record_multipliers(initial, risks, reference, .5)
        self.assertGreater(float(result[0]), float(initial[0]))
        self.assertLess(float(result[1]), float(initial[1]))
        self.assertAlmostEqual(float((risks-reference).mean()), 0., places=7)

    def test_training_derived_initial_duals_balance_independent_logit_shifts(self):
        dual = record_stationary_multipliers(self.logits, self.labels, self.ids, 2)
        shifts = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        weights = self.weights()
        loss = record_constrained_loss(self.logits + shifts[self.ids], self.labels, self.ids,
                                       dual, weights[0], weights[1], self.reference)
        derivative, = torch.autograd.grad(loss, shifts)
        self.assertTrue(torch.allclose(derivative, torch.zeros_like(derivative), atol=1e-12, rtol=0.))

    def test_records_without_one_class_are_finite_and_excluded_from_that_average(self):
        labels = torch.tensor([1., 1., 0., 0., 0.], dtype=torch.float64)
        ids = torch.tensor([0, 0, 1, 1, 1])
        logits = torch.tensor([3., 2., -1., 0., 1.], dtype=torch.float64)
        (fn, p), (_, n) = record_class_means(F.softplus(self.boundary-logits), labels, ids, 2)
        positive_weights, negative_weights = record_loss_weights(p, n, len(logits))
        dual = record_stationary_multipliers(logits, labels, ids, 2)
        loss = record_constrained_loss(logits, labels, ids, dual, positive_weights, negative_weights, fn)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(positive_weights[1]), 0.)
        self.assertEqual(float(negative_weights[0]), 0.)
        self.assertEqual(float(dual[1]), 0.)

    def test_invalid_record_ids_and_nonfinite_dual_risks_fail(self):
        with self.assertRaises(ValueError):
            record_class_means(self.logits, self.labels, self.ids + 1, 2)
        with self.assertRaises(ValueError):
            update_record_multipliers(torch.ones(2), torch.tensor([0., float('nan')]), torch.zeros(2), .5)


class CachedFitTests(unittest.TestCase):
    def test_record_fit_exports_per_record_risks_and_list_duals_and_reuses_reference(self):
        class SmallHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 1)
                torch.nn.init.zeros_(self.layer.weight)
                torch.nn.init.zeros_(self.layer.bias)

            def forward_encoded(self, q, z2):
                return z2 + self.layer(q).squeeze(1)

            def variance_regularizer(self):
                return self.layer.weight.square().mean()

        rows = []
        for j in range(2):
            rows.append({'record': 'record%d' % j, 'paired': np.ones(4, dtype=bool),
                         'q': np.array([[1., 0.], [2., 0.], [0., 1.], [0., 2.]], np.float32),
                         'z2': np.array([3., 2., 1., 0.], np.float32),
                         'label': np.array([1, 1, 0, 0], np.uint8)})
        baseline = {'iou': .5, 'acc': .5, 'pd': .5, 'fa': .1, 'tp': 2, 'fp': 0, 'fn': 2}
        validation = {'metrics': dict(baseline), 'readouts': {'second': dict(baseline)}}
        cfg = {'threshold': .9, 'constraint': 'record_fn_reference', 'lr': .01, 'lr_end': .001,
               'head_batch_size': 3, 'epochs': 2, 'seed': 37, 'covariance_regularizer': .001,
               'grad_clip': 1., 'recall_dual_lr': .5}
        for constraint in ('record_fn_reference', 'smooth_fn_reference'):
            with self.subTest(constraint=constraint):
                cfg['constraint'] = constraint
                callbacks = []
                with mock.patch('tools.v16.train_covariance.cached_evaluation', return_value=validation) as evaluate:
                    history = fit_cached_head(SmallHead(), rows, [], None, cfg, torch.device('cpu'),
                                              callback=lambda *args: callbacks.append(args))
                self.assertEqual(len(history), 2)
                self.assertEqual(len(callbacks), 3)
                if constraint == 'record_fn_reference':
                    self.assertTrue(all(isinstance(args[-1], list) and len(args[-1]) == 2 for args in callbacks))
                    for record in history:
                        self.assertEqual(record['constraint'], constraint)
                        self.assertEqual([r['record'] for r in record['record_risks']], ['record0', 'record1'])
                        self.assertTrue(all(math.isfinite(r['constraint_gap']) for r in record['record_risks']))
                else:
                    self.assertTrue(all(isinstance(args[-1], float) for args in callbacks))
                    self.assertTrue(all('record_risks' not in record for record in history))
                self.assertNotIn('reference', evaluate.call_args_list[0].kwargs)
                self.assertTrue(all(call.kwargs['reference'] is validation for call in evaluate.call_args_list[1:]))


if __name__ == '__main__':
    unittest.main()
