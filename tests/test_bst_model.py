"""Causal and re-audit contracts for shared evidence / event-specific beliefs."""
import math
import copy
import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.bst import (BSTConfig, BirthValueHead, ContinuationValueHead,
                       SharedSequentialModel, concat_states, detach_state,
                       posterior_decision, stop_risk, take_state)


def example(nonzero=True):
    torch.manual_seed(37)
    model = SharedSequentialModel(BSTConfig(max_wait_steps=3))
    if nonzero:
        with torch.no_grad():
            model.value_weight.normal_(0, 0.3)
            model.value_bias.normal_(0, 0.1)
    source = torch.tensor([[-3., 0.], [2., 1.], [-1., 1.], [4., 0.], [-6., 1.]])
    indices = torch.tensor([[0, 1, 2, 3, -1], [1, 2, 0, 4, -1]])
    relative = torch.zeros(2, 5, 4)
    relative[..., 0] = torch.tensor([[1., 2., 3., 4., 0.], [2., 1., 4., 2., 0.]])
    relative[..., 1] = 0.5
    relative[..., 2] = torch.tensor([-50., 0., 50., 100., 0.])
    relative[..., 3] = torch.tensor([-1., 0., 1., 2., 0.])
    prior = torch.tensor([-0.5, 0.5])
    p = torch.tensor([0., 1.])
    return model.eval(), source, indices, relative, prior, p


class SharedModelTests(unittest.TestCase):
    def test_initial_state_fields_have_independent_storage(self):
        model, _, _, _, prior, p = example()
        state = model.make_state(prior, p)
        original = prior.clone()
        anchor = state["anchor"].detach().clone()
        with torch.no_grad():
            state["score"].add_(3)
            state["correction"].add_(4)
            state["context"].add_(5)
        torch.testing.assert_allclose(prior, original, rtol=0, atol=0)
        torch.testing.assert_allclose(state["prior"], original, rtol=0, atol=0)
        torch.testing.assert_allclose(state["previous_score"], original, rtol=0, atol=0)
        torch.testing.assert_allclose(state["anchor"], anchor, rtol=0, atol=0)
        for key in ("mass", "neff", "age"):
            torch.testing.assert_allclose(state[key], torch.zeros_like(prior), rtol=0, atol=0)

    def test_initialization_is_exact_prior(self):
        model, source, idx, rel, prior, p = example(False)
        state = model.make_state(prior, p)
        bank = model.encode_sources(source)
        for age in range(4):
            score, state, _ = model.reaudit(bank, idx, rel, state, prior, p, age)
            torch.testing.assert_allclose(score, prior, rtol=0, atol=0)

    def test_future_sources_and_relations_cannot_change_prefix(self):
        model, source, idx, rel, prior, p = example()
        # IDs 3 and 4 occur only in future windows at age zero.
        before, _, _ = model(source, idx, rel, prior, p, age=0)
        other = source.clone()
        other[3:] = torch.tensor([[800., -100.], [-900., 300.]])
        later_rel = rel.clone()
        later_rel[:, 2:, :3] = 1e5
        after, _, _ = model(other, idx, later_rel, prior, p, age=0)
        torch.testing.assert_allclose(before, after, rtol=0, atol=0)
        source = source.requires_grad_()
        out, _, _ = model(source, idx, rel, prior, p, age=0)
        out.sum().backward()
        torch.testing.assert_allclose(source.grad[3:], torch.zeros_like(source.grad[3:]), rtol=0, atol=0)
        self.assertGreater(float(source.grad[:3].abs().sum()), 0.0)

    def test_masked_padding_cannot_inject_nan(self):
        model, source, idx, rel, prior, p = example()
        rel[:, -1] = float("nan")
        score, state, diag = model(source, idx, rel, prior, p, age=0, return_tokens=True)
        self.assertTrue(bool(torch.isfinite(score).all()))
        self.assertTrue(bool(torch.isfinite(state["context"]).all()))
        self.assertTrue(bool((diag["gate"][:, -1] == 0).all()))

    def test_shared_cache_equals_duplicated_source_rows(self):
        model, source, idx, rel, prior, p = example()
        score, state, _ = model(source, idx, rel, prior, p, age=2)
        duplicate_source = source.repeat(2, 1)
        duplicate_idx = idx.clone()
        duplicate_idx[1] = torch.where(idx[1] >= 0, idx[1] + source.shape[0], idx[1])
        other, other_state, _ = model(duplicate_source, duplicate_idx, rel, prior, p, age=2)
        torch.testing.assert_allclose(score, other)
        torch.testing.assert_allclose(state["context"], other_state["context"])

    def test_pair_cache_preserves_prefix_outputs_and_gradients(self):
        model, source, idx, rel, prior, p = example()
        cached_model = copy.deepcopy(model)
        gradients, results = [], []
        for net, use_cache in ((model, False), (cached_model, True)):
            inp = source.clone().requires_grad_()
            bank = net.encode_sources(inp)
            cache = net.prepare_pairs(bank, idx, rel) if use_cache else None
            state = net.make_state(prior, p)
            out = []
            for age in range(3):
                score, state, _ = net.reaudit(bank, idx, rel, state, prior, p, age, pair_cache=cache)
                out.append(score)
            outputs = torch.stack(out)
            loss = outputs.square().sum()
            params = list(net.parameters()) + [inp]
            gradients.append(torch.autograd.grad(loss, params))
            results.append(outputs.detach())
        torch.testing.assert_allclose(results[0], results[1], rtol=0, atol=0)
        for left, right in zip(*gradients):
            torch.testing.assert_allclose(left, right, rtol=1e-4, atol=2e-6)

    def test_neighbor_order_does_not_matter(self):
        model, source, idx, rel, prior, p = example()
        score, state, _ = model(source, idx, rel, prior, p, age=2)
        perm = torch.tensor([3, 1, 4, 0, 2])
        other, other_state, _ = model(source, idx[:, perm], rel[:, perm], prior, p, age=2)
        torch.testing.assert_allclose(score, other)
        torch.testing.assert_allclose(state["context"], other_state["context"])

    def test_no_cross_query_or_batch_normalization(self):
        model, source, idx, rel, prior, p = example()
        both, _, _ = model(source, idx, rel, prior, p, age=2)
        single, _, _ = model(source, idx[:1], rel[:1], prior[:1], p[:1], age=2)
        torch.testing.assert_allclose(both[:1], single)
        perturbed_prior = prior.clone()
        perturbed_prior[1] = 10000
        perturbed, _, _ = model(source, idx, rel, perturbed_prior, p, age=2)
        torch.testing.assert_allclose(both[:1], perturbed[:1], rtol=0, atol=0)

    def test_repeated_evidence_does_not_accumulate_or_drift(self):
        model, source, idx, rel, prior, p = example()
        bank = model.encode_sources(source)
        score, state, _ = model.reaudit(bank, idx, rel, None, prior, p, age=0)
        repeat, repeated_state, diag = model.reaudit(bank, idx, rel, state, prior, p, age=0)
        torch.testing.assert_allclose(score, repeat, rtol=0, atol=0)
        torch.testing.assert_allclose(state["context"], repeated_state["context"], rtol=0, atol=0)
        self.assertFalse(bool(diag["updated"].any()))
        # There is no new evidence between ages 2 and 3; only policy age changes.
        score2, state2, _ = model.reaudit(bank, idx, rel, state, prior, p, age=2)
        score3, state3, _ = model.reaudit(bank, idx, rel, state2, prior, p, age=3)
        torch.testing.assert_allclose(score2, score3, rtol=0, atol=0)
        torch.testing.assert_allclose(state3["score"] - state3["previous_score"],
                                      torch.zeros_like(score3), rtol=0, atol=0)
        self.assertTrue(bool((state3["age"] == 3).all()))

    def test_new_evidence_reinterprets_old_values_immediately(self):
        model, source, idx, rel, prior, p = example()
        state = model.make_state(prior, p)
        bank = model.encode_sources(source)
        _, _, old_diag = model.reaudit(bank, idx, rel, state, prior, p, age=0, return_tokens=True)
        _, _, new_diag = model.reaudit(bank, idx, rel, state, prior, p, age=1, return_tokens=True)
        # Hold the starting context fixed, change only the newly visible prefix.
        self.assertGreater(float((old_diag["value"][:, :2] - new_diag["value"][:, :2]).abs().max()), 1e-5)
        self.assertGreater(float((old_diag["gate"][:, :2] - new_diag["gate"][:, :2]).abs().max()), 1e-5)

    def test_freezing_revalue_ablates_old_value_changes(self):
        model, source, idx, rel, prior, p = example()
        model.config.revalue = False
        state = model.make_state(prior, p)
        bank = model.encode_sources(source)
        _, _, before = model.reaudit(bank, idx, rel, state, prior, p, age=0, return_tokens=True)
        _, _, after = model.reaudit(bank, idx, rel, state, prior, p, age=1, return_tokens=True)
        torch.testing.assert_allclose(before["value"][:, :2], after["value"][:, :2], rtol=0, atol=0)

    def test_new_evidence_content_revises_old_values_at_fixed_age(self):
        model, source, idx, rel, prior, p = example()
        state = model.make_state(prior, p)
        other = source.clone()
        other[2] = torch.tensor([12., 0.])
        _, _, before = model.reaudit(model.encode_sources(source), idx, rel, state, prior, p,
                                     age=1, return_tokens=True)
        _, _, after = model.reaudit(model.encode_sources(other), idx, rel, state, prior, p,
                                    age=1, return_tokens=True)
        # Query zero's old IDs 0 and 1 are fixed, including all geometry, mask,
        # age and previous state. Only newly visible source ID 2 changed content.
        self.assertGreater(float((before["value"][0, :2] - after["value"][0, :2]).abs().max()), 1e-5)
        self.assertGreater(float((before["gate"][0, :2] - after["gate"][0, :2]).abs().max()), 1e-5)
        model.config.revalue = False
        _, _, fixed_before = model.reaudit(model.encode_sources(source), idx, rel, state, prior, p,
                                           age=1, return_tokens=True)
        _, _, fixed_after = model.reaudit(model.encode_sources(other), idx, rel, state, prior, p,
                                          age=1, return_tokens=True)
        torch.testing.assert_allclose(fixed_before["value"][0, :2], fixed_after["value"][0, :2],
                                      rtol=0, atol=0)

    def test_softmax_is_over_visible_evidence_even_for_large_negative_logits(self):
        model, source, idx, rel, prior, p = example()
        model.config.normalization = "softmax"
        with torch.no_grad():
            model.query_key.weight.zero_()
            model.query_key.bias.fill_(1000)
            model.source_key.weight.zero_()
            model.source_key.bias.fill_(-1000)
            model.relation_key.weight.zero_()
            model.relation_bias.weight.zero_()
        _, _, diag = model(source, idx, rel, prior, p, age=0, return_tokens=True)
        torch.testing.assert_allclose(diag["gate"].sum(1), torch.ones(2, model.config.heads))
        torch.testing.assert_allclose(diag["gate"][:, 2:], torch.zeros_like(diag["gate"][:, 2:]),
                                      rtol=0, atol=0)

    def test_empty_source_bank_and_empty_neighborhood(self):
        model, _, idx, rel, prior, p = example()
        empty = torch.empty(0, 2)
        indices = torch.full_like(idx, -1)
        for norm in ("mean", "sum", "softmax"):
            model.config.normalization = norm
            score, state, diag = model(empty, indices, rel, prior, p, age=0)
            torch.testing.assert_allclose(score, prior, rtol=0, atol=0)
            self.assertTrue(bool(torch.isfinite(state["context"]).all()))
            self.assertFalse(bool(diag["updated"].any()))
            zero_score, _, _ = model(empty, indices[:, :0], rel[:, :0], prior, p, age=0)
            torch.testing.assert_allclose(zero_score, prior, rtol=0, atol=0)

    def test_invalid_query_is_unchanged(self):
        model, source, idx, rel, prior, p = example()
        state = model.make_state(prior, p)
        score, new_state, diag = model(source, idx, rel, prior, p, age=1, state=state,
                                      valid=torch.tensor([True, False]))
        for key in state:
            torch.testing.assert_allclose(state[key][1:], new_state[key][1:], rtol=0, atol=0)
        self.assertFalse(bool(diag["updated"][1]))
        torch.testing.assert_allclose(score[1:], prior[1:], rtol=0, atol=0)

    def test_state_queue_helpers_and_vow_features(self):
        model, source, idx, rel, prior, p = example()
        _, state, _ = model(source, idx, rel, prior, p, age=1)
        a, b = take_state(state, torch.tensor([0])), take_state(state, torch.tensor([1]))
        rebuilt = concat_states([a, b])
        for key in state:
            torch.testing.assert_allclose(state[key], rebuilt[key], rtol=0, atol=0)
        frozen = detach_state(state)
        self.assertTrue(all(not value.requires_grad for value in frozen.values()))
        features = model.value_features(state)
        self.assertEqual(features.shape, (2, model.value_feature_dim))
        self.assertTrue(bool(torch.isfinite(features).all()))
        self.assertTrue(bool((ContinuationValueHead(model.value_feature_dim)(features) > 0).all()))
        cheap = torch.stack([torch.asinh(prior), p], -1)
        self.assertTrue(bool((BirthValueHead()(cheap) > 0).all()))


class RiskTests(unittest.TestCase):
    def test_cost_threshold_is_not_assumed_half(self):
        logits = torch.tensor([0., math.log(9.) - 0.01, math.log(9.) + 0.01])
        action, risk = posterior_decision(logits, fp_cost=9., fn_cost=1.)
        self.assertEqual(action.tolist(), [False, False, True])
        p = torch.sigmoid(logits)
        torch.testing.assert_allclose(risk, torch.minimum(p, 9 * (1 - p)))
        torch.testing.assert_allclose(stop_risk(logits, 9., 1.), risk)

    def test_extreme_logits_are_finite(self):
        logits = torch.tensor([-1e6, 1e6])
        action, risk = posterior_decision(logits, 0.05, 0.95)
        self.assertEqual(action.tolist(), [False, True])
        self.assertTrue(bool(torch.isfinite(risk).all()))
        self.assertTrue(bool((risk == 0).all()))

    def test_invalid_costs_are_rejected(self):
        for fp, fn in ((0, 1), (1, -1), (float("nan"), 1)):
            with self.assertRaises(ValueError):
                stop_risk(torch.zeros(1), fp, fn)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
