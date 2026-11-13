"""CPU contracts for the single anchored V16 context correction.

These tests check observable guarantees and gradient routes, not classification
performance. Dataset validation is a separate experiment.
"""
import os
import sys
import unittest
import math
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.nn.functional as F

from model.anchored_revision import AnchoredRevision
from tools.v16.train import (boundary_retention_loss, classification_loss,
                             detach_pending, validate_config)
from utils.stream_common import load_flat_config


def make(variant="full"):
    return AnchoredRevision(feature_dim=6, hidden=4, kappa=.75,
                            variant=variant).to(dtype=torch.float64, device="cpu")


def observations(n=23, requires_grad=False):
    return tuple(torch.randn(*shape, dtype=torch.float64, device="cpu",
                             requires_grad=requires_grad)
                 for shape in ((n,), (n,), (n, 6), (n, 6)))


def activate(model):
    if model.has_revision:
        with torch.no_grad():
            model.output.weight.normal_(0., .4)


class AnchoredRevisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_zero_initialization_preserves_second_view_for_all_variants(self):
        values = observations()
        for variant in AnchoredRevision.VARIANTS:
            with self.subTest(variant=variant):
                model = make(variant)
                result, info = model(*values, return_info=True)
                self.assertTrue(torch.equal(result, values[1]))
                self.assertTrue(torch.equal(info["delta"], torch.zeros_like(values[1])))
                self.assertTrue(torch.equal(info["consensus"], values[1]))
                self.assertFalse(hasattr(model, "reliability"))

    def test_identical_views_cannot_accumulate_evidence_after_learning(self):
        model = make()
        activate(model)
        z, _, h, _ = observations()
        actual, info = model(z, z.clone(), h, h.clone(), return_info=True)
        self.assertTrue(torch.equal(actual, z))
        self.assertTrue(torch.equal(info["delta"], torch.zeros_like(z)))

    def test_correction_is_bounded_and_can_increase_or_decrease_a_score(self):
        model = make()
        with torch.no_grad():
            model.condition.weight.zero_()
            model.condition.bias.fill_(10.)
            model.difference.weight.zero_()
            model.difference.weight[0, 0] = 1.
            model.output.weight.zero_()
            model.output.weight[0, 0] = 100.
        z1, z2, h1, h2 = observations()
        h1.zero_()
        h2.zero_()
        h1[:, 0] = torch.linspace(-2., 2., len(z1), dtype=h1.dtype)
        actual, info = model(z1, z2, h1, h2, return_info=True)
        delta = info["delta"].detach()
        self.assertLess(float(delta.min()), 0.)
        self.assertGreater(float(delta.max()), 0.)
        self.assertLessEqual(float(delta.abs().max()), model.kappa)
        self.assertTrue(torch.allclose(actual - z2, info["delta"], atol=1e-14, rtol=1e-14))

    def test_first_optimizer_update_opens_gradients_to_both_observations(self):
        model = make()
        values = observations(requires_grad=True)
        target = torch.linspace(-1., 1., len(values[0]), dtype=values[0].dtype)
        optimizer = torch.optim.SGD(model.parameters(), lr=.2)
        initial = model.output.weight.detach().clone()
        (model(*values) - target).square().mean().backward()
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.)
        optimizer.step()
        self.assertFalse(torch.equal(initial, model.output.weight))
        optimizer.zero_grad()
        for value in values:
            value.grad = None
        (model(*values) - target).square().mean().backward()
        for i, value in enumerate(values):
            with self.subTest(input=i):
                self.assertIsNotNone(value.grad)
                self.assertTrue(bool(torch.isfinite(value.grad).all()))
                self.assertGreater(float(value.grad.abs().sum()), 0.)
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_single_observation_control_cannot_read_first_view(self):
        model = make("no_context")
        activate(model)
        z1, z2, h1, h2 = observations(requires_grad=True)
        actual = model(z1, z2, h1, h2)
        changed = model(z1.detach() * 1000. + 17., z2,
                        h1.detach() * -1000. - 31., h2)
        self.assertTrue(torch.equal(actual, changed))
        actual.square().sum().backward()
        self.assertIsNone(z1.grad)
        self.assertIsNone(h1.grad)
        self.assertGreater(float(z2.grad.abs().sum()), 0.)
        self.assertGreater(float(h2.grad.abs().sum()), 0.)

    def test_batch_chunks_duplicate_rows_singletons_and_empty_are_equivalent(self):
        values = observations()
        for variant in AnchoredRevision.VARIANTS:
            with self.subTest(variant=variant):
                model = make(variant)
                activate(model)
                batch = model(*values)
                chunks = torch.cat([model(*(v[a:b] for v in values))
                                    for a, b in ((0, 1), (1, 7), (7, 17), (17, 23))])
                duplicate = model(*(torch.cat((v, v), dim=0) for v in values))
                self.assertTrue(torch.allclose(batch, chunks, atol=1e-13, rtol=1e-13))
                self.assertTrue(torch.allclose(duplicate, torch.cat((batch, batch)),
                                               atol=1e-13, rtol=1e-13))
                empty, info = model(*(v[:0] for v in values), return_info=True)
                self.assertEqual(tuple(empty.shape), (0,))
                self.assertTrue(all(tuple(value.shape) == (0,) for value in info.values()))

    def test_hidden_scale_is_bounded_without_fitting_dataset_statistics(self):
        h = torch.randn(17, 6, dtype=torch.float64) * 1e6
        normalized = AnchoredRevision.bounded_hidden(h)
        self.assertTrue(bool(torch.isfinite(normalized).all()))
        self.assertTrue(bool((normalized.square().mean(dim=-1) <= 1.).all()))
        self.assertTrue(torch.equal(AnchoredRevision.bounded_hidden(torch.zeros_like(h)),
                                    torch.zeros_like(h)))

    def test_cost_and_capacity_match_single_view_control(self):
        for variant in ("full", "no_context"):
            model = AnchoredRevision(variant=variant)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 544)
            self.assertEqual(model.macs_per_pair(), 536)
            self.assertEqual(model.estimated_macs(100), 53600)
        for variant in ("no_revision", "second"):
            model = AnchoredRevision(variant=variant)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 0)
            self.assertEqual(model.macs_per_pair(), 0)
        with self.assertRaises(ValueError):
            make().estimated_macs(-1)

    def test_checkpoint_round_trip_preserves_learned_output(self):
        model = make()
        activate(model)
        restored = make()
        restored.load_state_dict(model.state_dict(), strict=True)
        values = observations()
        self.assertTrue(torch.equal(model(*values), restored(*values)))

    def test_invalid_configuration_and_input_shapes_are_rejected(self):
        for kwargs in ({"feature_dim": 0}, {"hidden": 1.5}, {"kappa": 0.},
                       {"kappa": float("inf")}, {"variant": "unknown"}):
            with self.assertRaises(ValueError):
                AnchoredRevision(**kwargs)
        model = make()
        z1, z2, h1, h2 = observations()
        for values in ((z1[:, None], z2, h1, h2), (z1, z2[:-1], h1, h2),
                       (z1, z2, h1[:, :-1], h2), (z1, z2.float(), h1, h2)):
            with self.assertRaises(ValueError):
                model(*values)


class BoundaryRetentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.boundary = math.log(9.)

    def test_background_scores_are_free_to_fall_even_when_reference_is_high(self):
        z = torch.tensor([-.5, self.boundary, 8.], dtype=torch.float64,
                         requires_grad=True)
        labels = torch.zeros_like(z)
        reference = torch.tensor([9., 20., 100.], dtype=z.dtype, requires_grad=True)
        loss = boundary_retention_loss(z, reference, labels)
        base = F.binary_cross_entropy_with_logits(z - self.boundary, labels,
                                                  reduction="none")
        self.assertTrue(torch.equal(loss, base))
        loss.sum().backward()
        self.assertTrue(bool((z.grad > 0.).all()))
        self.assertIsNone(reference.grad)
        updated = z.detach() - .1 * z.grad
        self.assertTrue(bool((updated < z.detach()).all()))
        self.assertTrue(bool((boundary_retention_loss(updated, reference, labels) < loss).all()))

    def test_only_correct_reference_positives_receive_detached_retention(self):
        z = torch.full((4,), self.boundary - .5, dtype=torch.float64,
                       requires_grad=True)
        labels = z.new_tensor([0., 1., 1., 1.])
        reference = z.new_tensor([self.boundary + .5, self.boundary - .01,
                                  self.boundary, self.boundary + .5], requires_grad=True)
        loss = boundary_retention_loss(z, reference, labels)
        base = F.binary_cross_entropy_with_logits(z - self.boundary, labels,
                                                  reduction="none")
        extra = loss - base
        self.assertTrue(torch.equal(extra[:2], torch.zeros_like(extra[:2])))
        self.assertTrue(torch.allclose(extra[2:], z.new_tensor([1.25, 5.])))
        loss.sum().backward()
        self.assertIsNone(reference.grad)
        base_gradient = torch.sigmoid(z.detach() - self.boundary) - labels
        self.assertTrue(torch.allclose(z.grad[:2], base_gradient[:2]))
        self.assertTrue(bool((z.grad[2:] < base_gradient[2:]).all()))

    def test_retention_caps_reference_confidence_and_never_blocks_improvement(self):
        z = torch.tensor([self.boundary, self.boundary + 1., self.boundary + 2.],
                         dtype=torch.float64, requires_grad=True)
        labels = torch.ones_like(z)
        reference = torch.full_like(z, 1000., requires_grad=True)
        loss = boundary_retention_loss(z, reference, labels, margin=1.)
        base = F.binary_cross_entropy_with_logits(z - self.boundary, labels,
                                                  reduction="none")
        self.assertTrue(torch.allclose(loss - base, z.new_tensor([5., 0., 0.])))
        loss.sum().backward()
        self.assertTrue(bool((z.grad < 0.).all()))
        self.assertIsNone(reference.grad)

    def test_extreme_scores_have_finite_cpu_loss_and_gradients_in_training_dtypes(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                z = torch.tensor([-1000., 1000., -1000., 1000.], dtype=dtype,
                                 requires_grad=True)
                reference = torch.tensor([1000., -1000., 1000., 1000.], dtype=dtype,
                                         requires_grad=True)
                labels = z.new_tensor([0., 0., 1., 1.])
                loss = boundary_retention_loss(z, reference, labels)
                self.assertEqual(loss.dtype, dtype)
                self.assertEqual(loss.device.type, "cpu")
                self.assertTrue(bool(torch.isfinite(loss).all()))
                loss.sum().backward()
                self.assertTrue(bool(torch.isfinite(z.grad).all()))
                self.assertIsNone(reference.grad)

    def test_fixed_point_loss_places_zero_margin_at_existing_point_nine_threshold(self):
        z = torch.full((2,), self.boundary, dtype=torch.float64, requires_grad=True)
        labels = z.new_tensor([0., 1.])
        reference = torch.full_like(z, self.boundary - 1.)
        before = z.detach().clone()
        loss = boundary_retention_loss(z, reference, labels)
        self.assertTrue(torch.allclose(loss, torch.full_like(z, math.log(2.))))
        loss.sum().backward()
        self.assertTrue(torch.allclose(z.grad, z.new_tensor([.5, -.5])))
        self.assertTrue(torch.equal(z.detach(), before))
        self.assertTrue(torch.allclose(torch.sigmoid(z.detach()), torch.full_like(z, .9)))


class AnchoredTrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.cfg = {"fusion_loss": "boundary_retention", "threshold": .9,
                    "retention_weight": 5., "retention_margin": 1.}
        # Three windows, two shared events and two one-view boundary events.
        self.ids = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
        self.seq = SimpleNamespace(bounds=np.array([0, 2, 4, 6]), n_windows=3,
                                   coverage_count=np.array([1, 2, 2, 1]),
                                   first_position=np.array([0, 1, 3, 5]),
                                   last_position=np.array([0, 2, 4, 5]),
                                   last_window=np.array([0, 1, 2, 2]))
        self.labels = torch.tensor([0., 1., 1., 0., 0., 1.], dtype=torch.float64)
        self.logits = torch.tensor([-.2, 2.7, 2.5, 3., 3.2, 1.],
                                   dtype=torch.float64, requires_grad=True)
        self.hidden = torch.randn(6, 6, dtype=torch.float64, requires_grad=True)
        self.head = make()
        activate(self.head)

    def classify(self, start, stop, pending):
        lo, hi = int(self.seq.bounds[start]), int(self.seq.bounds[stop])
        segment = (0, start, stop, 0, self.seq, self.ids[lo:hi])
        return classification_loss(None, self.head, None, self.logits[lo:hi],
                                   self.labels[lo:hi], {}, [segment], [], pending,
                                   2., self.hidden[lo:hi], self.cfg)

    def test_each_original_event_has_unit_supervision_mass_and_correct_loss(self):
        pending = {}
        loss, mass, committed, cross = self.classify(0, 3, pending)
        self.assertEqual(mass, 4.)
        self.assertEqual(committed, 4)
        self.assertEqual(cross, 0)
        self.assertEqual(pending, {})
        raw = F.binary_cross_entropy_with_logits(self.logits, self.labels,
                                                pos_weight=self.logits.new_tensor([2.]),
                                                reduction="none")
        expected = (raw * self.logits.new_tensor([1., .25, .25, .25, .25, 1.])).sum()
        final = self.head(self.logits[[1, 3]], self.logits[[2, 4]],
                          self.hidden[[1, 3]], self.hidden[[2, 4]])
        expected += .5 * boundary_retention_loss(final, self.logits[[2, 4]],
                                                 self.labels[[2, 4]]).sum()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-13, rtol=1e-13))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(self.logits.grad).all()))
        self.assertTrue(bool(torch.isfinite(self.hidden.grad).all()))
        for row in (1, 2, 3, 4):
            self.assertGreater(float(self.hidden.grad[row].abs().sum()), 0.)
        self.assertGreater(float(self.head.output.weight.grad.abs().sum()), 0.)

    def test_tbptt_detached_cache_preserves_loss_and_counts_without_old_graph(self):
        whole, whole_mass, whole_count, _ = self.classify(0, 3, {})
        pending = {}
        first, mass1, count1, cross1 = self.classify(0, 1, pending)
        pending = detach_pending(pending)
        self.assertFalse(pending[0]["logits"].requires_grad)
        self.assertFalse(pending[0]["hidden"].requires_grad)
        second, mass2, count2, cross2 = self.classify(1, 3, pending)
        self.assertTrue(torch.allclose(first + second, whole, atol=1e-13, rtol=1e-13))
        self.assertEqual(mass1 + mass2, whole_mass)
        self.assertEqual(count1 + count2, whole_count)
        self.assertEqual(cross1, 0)
        self.assertEqual(cross2, 1)
        self.assertEqual(pending, {})
        second.backward()
        self.assertTrue(torch.equal(self.hidden.grad[:2], torch.zeros_like(self.hidden.grad[:2])))
        self.assertGreater(float(self.hidden.grad[2:].abs().sum()), 0.)

    def test_r3_formal_config_keeps_v5_state_clock_budget_and_fixed_threshold(self):
        path = Path(__file__).resolve().parents[1] / "configs/v16/r3/full.yaml"
        cfg = load_flat_config(path)
        validate_config(cfg)
        self.assertEqual(cfg["revision_architecture"], "anchored")
        self.assertEqual(cfg["revision_variant"], "full")
        self.assertEqual(cfg["threshold"], .9)
        self.assertEqual(cfg["state_mode"], "carry")
        self.assertEqual(cfg["trainer"], "streams")
        self.assertGreater(cfg["comm_budget_messages"], 0)
        self.assertEqual(cfg["window_ms"], 2 * cfg["stride_ms"])
        self.assertTrue(cfg["require_initialization"])
        self.assertTrue(cfg["validate_initial"])
        self.assertEqual(cfg["eval_execution"], "step")
        with self.assertRaisesRegex(ValueError, "threshold"):
            validate_config(dict(cfg, threshold=.8))


if __name__ == "__main__":
    unittest.main()
