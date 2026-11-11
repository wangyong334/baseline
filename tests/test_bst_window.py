"""Structural properties of window-balanced evidence, not accuracy guarantees."""
import copy
import unittest

import torch
from torch.nn import functional as F

from model.bst import BSTConfig, SharedSequentialModel
from model.bst_window import WindowEvidence


def example(**kwargs):
    torch.manual_seed(37)
    options = dict(normalization="window", past_windows=1, max_wait_steps=3,
                   source_dim=12, state_dim=8, relation_dim=4, heads=2)
    options.update(kwargs)
    net = SharedSequentialModel(BSTConfig(**options)).eval()
    with torch.no_grad():
        net.value_weight.normal_(0, .3)
        net.value_bias.normal_(0, .1)
    source = torch.tensor([[-3., 0.], [2., 1.], [-1., 1.], [4., 0.], [-6., 1.]])
    ids = torch.tensor([[0, 1, 2, 3, 4, -1]])
    rel = torch.tensor([[[1., 0., -50., -1.], [2., 1., 0., 0.],
                         [3., 0., 5., 0.], [1., 2., 50., 1.],
                         [2., 0., 100., 2.], [0., 0., 0., 0.]]])
    prior, polarity = torch.tensor([-.5]), torch.tensor([1.])
    return net, source, ids, rel, prior, polarity


def close(a, b, atol=1e-6, rtol=1e-5):
    torch.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


class WindowEvidenceContracts(unittest.TestCase):
    def test_tiny_positive_affinities_preserve_group_replication_and_gradients(self):
        block = WindowEvidence(2, 1, past_windows=0, max_wait=1, gate=False, bounded=False)
        weights = torch.tensor([[[1e-12], [2e-12]]], requires_grad=True)
        values = torch.tensor([[[1.], [3.]]])
        source_context = torch.tensor([[[.3, -.2], [-.1, .4]]])
        def run(w, v, c, present=True):
            dw = torch.zeros(w.shape[:2])
            mask = torch.full(w.shape[:2], present, dtype=torch.bool)
            groups = block.summarize_visible(c, mask, dw)
            return block(w, v, c, torch.zeros(1, 2), groups)
        one = run(weights, values, source_context)
        replicated = run(weights.repeat(1, 3, 1), values.repeat(1, 3, 1),
                         source_context.repeat(1, 3, 1))
        close(one["head_correction"], replicated["head_correction"])
        close(one["head_correction"], torch.tensor([[7./3.]]))
        close(one["summary"], replicated["summary"])
        one["head_correction"].sum().backward()
        self.assertTrue(bool(torch.isfinite(weights.grad).all()))
        underflow = run(torch.zeros_like(weights), values, source_context)
        # Underflowed but present affinities get the same positive numerical
        # floor. A wholly masked group remains empty, never uniform evidence.
        close(underflow["head_correction"], torch.tensor([[2.]]))
        empty = run(torch.zeros_like(weights), values, torch.zeros_like(source_context), present=False)
        close(empty["head_correction"], torch.zeros_like(empty["head_correction"]), atol=0, rtol=0)
        self.assertTrue(bool(torch.isfinite(empty["summary"]).all()))

    def test_underflowed_sigmoid_affinities_have_finite_gradients(self):
        block = WindowEvidence(2, 1, past_windows=0, max_wait=1, gate=False, bounded=False)
        logits = torch.tensor([[[-89.], [-88.]]], requires_grad=True)
        weights = logits.sigmoid()
        values = torch.tensor([[[1.], [3.]]])
        context = torch.tensor([[[.3, -.2], [-.1, .4]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        grouping = block.summarize_visible(context, mask, torch.zeros(1, 2))
        result = block(weights, values, context, torch.zeros(1, 2), grouping)
        close(result["head_correction"], torch.tensor([[2.]]))
        result["head_correction"].sum().backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_whole_window_replication_preserves_entire_prefix_trajectory(self):
        net, src, ids, rel, prior, pol = example()
        # Repeat all selected tokens of window 0 together, without changing
        # the BIR score/geometry of any source or rerunning the raw backbone.
        copies = torch.tensor([0, 1, 2, 1, 2, 1, 2, 3, 4, 5])
        bank = net.encode_sources(src)
        states = [net.make_state(prior, pol), net.make_state(prior, pol)]
        for age in range(4):
            left, states[0], _ = net.reaudit(bank, ids, rel, states[0], prior, pol, age)
            right, states[1], _ = net.reaudit(bank, ids[:, copies], rel[:, copies],
                                              states[1], prior, pol, age)
            close(left, right)
            for key in states[0]:
                if key != "count":
                    close(states[0][key], states[1][key])
            close(net.value_features(states[0]), net.value_features(states[1]))
            self.assertGreater(float(states[1]["count"].sum()), float(states[0]["count"].sum()))

    def test_group_operator_distinguishes_density_from_repeated_time_units(self):
        # Deliberately freeze per-token potentials and disable discounts: this
        # tests the aggregation law, not full-network confidence monotonicity.
        block = WindowEvidence(3, 1, past_windows=0, max_wait=2, gate=False, bounded=False)
        context = torch.zeros(1, 4, 3)
        mask = torch.ones(1, 4, dtype=torch.bool)
        weights, values = torch.ones(1, 4, 1), torch.full((1, 4, 1), .7)
        def correction(dw):
            group = block.summarize_visible(context, mask, torch.tensor([dw], dtype=torch.float32))
            return block(weights, values, context, torch.zeros(1, 3), group)["head_correction"]
        close(correction([0, 0, 0, 0]), torch.tensor([[.7]]))
        close(correction([0, 0, 1, 1]), torch.tensor([[1.4]]))
        close(correction([0, 1, 2, 2]), torch.tensor([[2.1]]))

    def test_each_window_has_bounded_potential_and_tokens_reconstruct_correction(self):
        net, src, ids, rel, prior, pol = example()
        with torch.no_grad():
            net.value_weight.fill_(1000.)
            net.value_bias.fill_(1000.)
        score, _, diag = net(src, ids, rel, prior, pol, 2, return_tokens=True)
        scales = F.softplus(net.window_evidence.raw_scale)
        self.assertTrue(bool((diag["value"].abs() <= scales[None, None] + 1e-6).all()))
        self.assertTrue(bool((diag["window_value"].abs() <= scales[None, None] + 1e-6).all()))
        bound = diag["units"] * scales.sum()
        self.assertTrue(bool((diag["correction"].abs() <= bound + 1e-6).all()))
        close((diag["effective_weight"] * diag["value"]).sum((1, 2)), score-prior)
        self.assertTrue(bool((diag["window_discount"] >= 0).all()))
        self.assertTrue(bool((diag["window_discount"] <= 1).all()))

    def test_future_tokens_cannot_change_visible_beliefs_or_gradients(self):
        net, src, ids, rel, prior, pol = example()
        inp = src.clone().requires_grad_()
        loc = rel.clone().requires_grad_()
        before, state, _ = net(inp, ids, loc, prior, pol, 0)
        before.sum().backward()
        close(inp.grad[3:], torch.zeros_like(inp.grad[3:]), atol=0, rtol=0)
        close(loc.grad[:, 3:5], torch.zeros_like(loc.grad[:, 3:5]), atol=0, rtol=0)
        self.assertGreater(float(inp.grad[:3].abs().sum()), 0.)
        other = src.clone()
        other[3:] = torch.tensor([[10000., -100.], [-9000., 800.]])
        moved = rel.clone()
        moved[:, 3:5, :3] = 10000.
        after, after_state, _ = net(other, ids, moved, prior, pol, 0)
        close(before, after, atol=0, rtol=0)
        close(state["context"], after_state["context"], atol=0, rtol=0)

    def test_disabling_reweight_freezes_old_affinity_and_window_discount(self):
        net, src, ids, rel, prior, pol = example(reweight=False)
        bank = net.encode_sources(src)
        _, state, before = net.reaudit(bank, ids, rel, None, prior, pol, 0, return_tokens=True)
        _, _, after = net.reaudit(bank, ids, rel, state, prior, pol, 1, return_tokens=True)
        close(before["gate"][:, :3], after["gate"][:, :3], atol=0, rtol=0)
        close(before["window_discount"][:, :2], after["window_discount"][:, :2], atol=0, rtol=0)

    def test_disabling_revalue_freezes_old_signed_values(self):
        net, src, ids, rel, prior, pol = example(revalue=False)
        bank = net.encode_sources(src)
        _, state, before = net.reaudit(bank, ids, rel, None, prior, pol, 0, return_tokens=True)
        _, _, after = net.reaudit(bank, ids, rel, state, prior, pol, 1, return_tokens=True)
        close(before["value"][:, :3], after["value"][:, :3], atol=0, rtol=0)

    def test_new_source_content_can_reinterpret_old_affinity_value_and_discount(self):
        net, src, ids, rel, prior, pol = example()
        old_state = net.make_state(prior, pol)
        other = src.clone()
        other[3] = torch.tensor([-15., 1.])
        _, _, before = net(src, ids, rel, prior, pol, 1, state=old_state, return_tokens=True)
        _, _, after = net(other, ids, rel, prior, pol, 1, state=old_state, return_tokens=True)
        for key in ("gate", "value"):
            self.assertGreater(float((before[key][:, :3]-after[key][:, :3]).abs().max()), 1e-6)
        self.assertGreater(float((before["window_discount"][:, :2]-after["window_discount"][:, :2]).abs().max()), 1e-6)

    def test_empty_later_window_and_same_prefix_leave_belief_exactly_unchanged(self):
        net, src, ids, rel, prior, pol = example()
        bank = net.encode_sources(src)
        score, state, _ = net.reaudit(bank, ids, rel, None, prior, pol, 2)
        for age in (2, 3):
            later, new_state, diag = net.reaudit(bank, ids, rel, state, prior, pol, age)
            close(score, later, atol=0, rtol=0)
            for key in ("context", "anchor", "correction", "units", "mass", "neff"):
                close(state[key], new_state[key], atol=0, rtol=0)
            self.assertFalse(bool(diag["updated"].any()))
            self.assertEqual(float(new_state["age"][0]), age)

    def test_cached_pairs_preserve_prefix_outputs_and_parameter_gradients(self):
        net, src, ids, rel, prior, pol = example()
        other = copy.deepcopy(net)
        outputs, derivatives = [], []
        for model, cached in ((net, False), (other, True)):
            inp = src.clone().requires_grad_()
            bank = model.encode_sources(inp)
            pairs = model.prepare_pairs(bank, ids, rel) if cached else None
            state, scores = None, []
            for age in range(3):
                score, state, _ = model.reaudit(bank, ids, rel, state, prior, pol, age, pair_cache=pairs)
                scores.append(score)
            whole = torch.stack(scores)
            derivatives.append(torch.autograd.grad(whole.square().sum(), list(model.parameters()) + [inp]))
            outputs.append(whole)
        close(outputs[0], outputs[1], atol=0, rtol=0)
        for a, b in zip(*derivatives):
            close(a, b, atol=2e-6, rtol=1e-4)

    def test_empty_history_padding_and_invalid_query_are_neutral(self):
        net, _, ids, rel, prior, pol = example()
        empty = torch.empty(0, 2)
        missing = torch.full_like(ids, -1)
        rel[:] = float("nan")
        for index, geometry in ((missing, rel), (missing[:, :0], rel[:, :0])):
            initial = net.make_state(prior, pol)
            score, state, diag = net(empty, index, geometry, prior, pol, 0, state=initial)
            close(score, prior, atol=0, rtol=0)
            close(state["context"], initial["context"], atol=0, rtol=0)
            self.assertEqual(float(state["units"][0]), 0.)
            self.assertFalse(bool(diag["updated"].any()))
        net, source, ids, rel, prior, pol = example()
        initial = net.make_state(prior, pol)
        _, state, diag = net(source, ids, rel, prior, pol, 2, state=initial,
                             valid=torch.tensor([False]))
        for key in initial:
            close(state[key], initial[key], atol=0, rtol=0)
        self.assertFalse(bool(diag["updated"].any()))

    def test_invalid_window_offsets_are_rejected_before_grouping(self):
        net, src, ids, rel, prior, pol = example()
        for offset in (-2., .5, 4., float("nan")):
            bad = rel.clone()
            bad[:, 0, 3] = offset
            with self.assertRaises(ValueError):
                net(src, ids, bad, prior, pol, 0)


if __name__ == "__main__":
    unittest.main()
