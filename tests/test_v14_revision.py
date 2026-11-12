"""V14 conditional revision: parameter budget, identifiable update and gradients."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.event_score_revision import ConcatScoreFusion, EventScoreRevision


class RevisionTests(unittest.TestCase):
    def test_parameter_budget_and_mean_initialization(self):
        head = EventScoreRevision()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 34)
        self.assertEqual(sum(p.numel() for p in ConcatScoreFusion().parameters()), 33)
        r1 = torch.linspace(-10, 10, 31)
        r2 = torch.linspace(3, -6, 31)
        fused, info = head(r1, r2, return_details=True)
        self.assertTrue(torch.allclose(fused, (r1 + r2) * 0.5, atol=1e-6))
        self.assertTrue(torch.allclose(info["a"], torch.full_like(r1, 0.5)))
        self.assertTrue(torch.equal(info["c"], torch.zeros_like(r1)))

    def test_rng_state_is_not_advanced(self):
        torch.manual_seed(37)
        state = torch.get_rng_state().clone()
        EventScoreRevision()
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        ConcatScoreFusion()
        self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_loss_reaches_both_scores_and_trainable_head(self):
        head = EventScoreRevision()
        r1 = torch.tensor([-3.0, -1.0, 2.0], requires_grad=True)
        r2 = torch.tensor([1.0, -2.0, 4.0], requires_grad=True)
        fused = head(r1, r2)
        fused.sum().backward()
        self.assertTrue(torch.allclose(r1.grad, torch.full_like(r1, 0.5)))
        self.assertTrue(torch.allclose(r2.grad, torch.full_like(r2, 0.5)))
        self.assertTrue(torch.any(head.net[-1].weight.grad != 0))
        self.assertTrue(torch.any(head.net[-1].bias.grad != 0))

    def test_gain_uses_only_first_and_is_positive(self):
        head = EventScoreRevision()
        with torch.no_grad():
            head.net[-1].weight.fill_(0.1)
        r1 = torch.tensor([-2.0, 0.0, 1.0])
        r2 = torch.tensor([-1.0, 1.0, 4.0])
        a, ai = head(r1, r2, return_details=True)
        b, bi = head(r1, r2 + 2.0, return_details=True)
        self.assertTrue(torch.equal(ai["a"], bi["a"]))
        self.assertTrue(torch.equal(ai["c"], bi["c"]))
        self.assertTrue(torch.all(ai["a"] > 0))
        self.assertTrue(torch.all(b > a))
        self.assertTrue(torch.allclose(b - a, 2 * ai["a"], atol=1e-6))

    def test_modes_and_calibration_control(self):
        head = EventScoreRevision()
        r1, r2 = torch.tensor([-2.0, 2.0]), torch.tensor([1.0, 3.0])
        for mode, expected in (("first", r1), ("second", r2), ("mean", (r1 + r2) / 2),
                               ("calibration_only", r1)):
            self.assertTrue(torch.allclose(head(r1, r2, mode=mode), expected))
        with torch.no_grad():
            head.net[-1].bias[1] = 0.25
        self.assertTrue(torch.allclose(head(r1, r2, mode="calibration_only"), r1 + 0.25))
        self.assertTrue(torch.allclose(head(r1, r1), r1 + 0.25))
        with self.assertRaises(ValueError):
            EventScoreRevision(mode="multiply_probabilities")
        with self.assertRaises(ValueError):
            head(r1, r2, mode="unknown")

    def test_shapes_and_empty_batches(self):
        for shape in ((), (0,), (4,), (4, 1), (2, 3)):
            r1, r2 = torch.zeros(shape), torch.ones(shape)
            self.assertEqual(EventScoreRevision()(r1, r2).shape, r1.shape)
            self.assertEqual(ConcatScoreFusion()(r1, r2).shape, r1.shape)
        for head in (EventScoreRevision(), ConcatScoreFusion()):
            with self.assertRaises(ValueError):
                head(torch.zeros(2), torch.zeros(2, 1))


if __name__ == "__main__":
    unittest.main()
