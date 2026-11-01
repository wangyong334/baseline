"""Regression tests for non-finite object-head updates."""
import math
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.train_numerics import clip_grad_norm_checked_


class GradientChecks(unittest.TestCase):
    def test_frozen_trainer_survives_240_updates_with_carried_state(self):
        import train_track_head as TH
        from test_object_tokens import make_layer
        from test_track_head import blob_windows
        head = make_layer()
        batches = [blob_windows(4, batch=2, stream=i % 2, noise=40, seed=i, velocity=(0.5, 0.))
                   for i in range(4)]
        calls = [0]

        def assemble(*args):
            i = calls[0]
            calls[0] += 1
            events, z0, feat, labels, tid = batches[i % len(batches)]
            keep = torch.ones(4, 2)
            if i % 40 == 0:
                keep[0] = 0
            return events, z0, feat, labels, keep, tid

        cfg = dict(tbptt_k=4, streams=2, lambda_exist=1., lambda_pos=0.5, grad_clip=1.)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        with patch.object(TH, "assemble_cached", side_effect=assemble):
            result = TH.train_epoch(None, head, [960, 960], opt, cfg, 30., torch.device("cpu"),
                                    np.random.RandomState(0), max_chunks=240)
        self.assertEqual(result["updates"], 240)
        for key in ("event_loss", "exist_loss", "pos_loss", "grad_norm_median"):
            self.assertTrue(math.isfinite(result[key]), key)
        for name, p in head.named_parameters():
            self.assertTrue(bool(torch.isfinite(p).all()), name)

    def test_finite_large_gradients_are_clipped_not_zeroed(self):
        p = torch.nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor([1e30, -1e30])
        norm = clip_grad_norm_checked_([("weight", p)], 1.0)
        self.assertTrue(math.isfinite(norm))
        self.assertAlmostEqual(norm / 1e30, math.sqrt(2), places=5)
        self.assertTrue(torch.allclose(p.grad, torch.tensor([1., -1.]) / math.sqrt(2)))

    def test_matches_standard_clipping_for_normal_gradients(self):
        a, b = torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(3))
        a.grad = torch.tensor([3., 4., 0.])
        b.grad = a.grad.clone()
        got = clip_grad_norm_checked_([("a", a)], 2.0)
        expected = float(torch.nn.utils.clip_grad_norm_([b], 2.0))
        self.assertAlmostEqual(got, expected)
        self.assertTrue(torch.allclose(a.grad, b.grad))

    def test_invalid_gradient_does_not_modify_any_gradient(self):
        for value in (float("inf"), float("nan")):
            a, b = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
            a.grad, b.grad = torch.tensor([100.]), torch.tensor([value])
            with self.assertRaisesRegex(RuntimeError, "chunk 227.*width_head"):
                clip_grad_norm_checked_([("good", a), ("width_head", b)], 1.0, "chunk 227")
            self.assertEqual(float(a.grad), 100.)
            self.assertFalse(bool(torch.isfinite(b.grad).all()))

    def test_empty_gradients(self):
        self.assertEqual(clip_grad_norm_checked_([], 1.), 0.)

    def test_frozen_trainer_stops_before_adam_update(self):
        import train_track_head as TH
        from test_object_tokens import make_layer
        from test_track_head import blob_windows
        head = make_layer()
        events, z0, feat, labels, tid = blob_windows(2, noise=4)
        keep = torch.ones(2, 1)
        cfg = dict(tbptt_k=2, streams=1, lambda_exist=1., lambda_pos=0.5, grad_clip=1.)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        before = {n: p.detach().clone() for n, p in head.named_parameters()}
        hook = head.read_bg[2].bias.register_hook(lambda g: torch.full_like(g, float("nan")))
        try:
            with patch.object(TH, "assemble_cached", return_value=(events, z0, feat, labels, keep, tid)):
                with patch.object(opt, "step", wraps=opt.step) as step:
                    with self.assertRaisesRegex(RuntimeError, "head chunk 0.*read_bg.2.bias"):
                        TH.train_epoch(None, head, [2], opt, cfg, 1., torch.device("cpu"), np.random.RandomState(0))
                    step.assert_not_called()
        finally:
            hook.remove()
        self.assertFalse(opt.state)
        for name, p in head.named_parameters():
            self.assertTrue(torch.equal(p, before[name]), name)


if __name__ == "__main__":
    unittest.main()
