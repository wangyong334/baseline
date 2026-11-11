"""Birth-referenced innovation preserves causal re-audit and its gradients."""
import copy
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from model.bst import BSTConfig, SharedSequentialModel, concat_states, take_state
from tools.bst.data import NeighborIndex, Recording
from tools.bst.explain import explain_query
from tools.bst.pipeline import rollout, weighted_event_loss
from tools.bst.stream import run_recording
from tools.bst.training_geometry import GeometryBatches


def example(**options):
    torch.manual_seed(37)
    cfg = dict(normalization="window", past_windows=1, max_wait_steps=3,
               source_dim=12, state_dim=8, relation_dim=4, heads=2,
               null_evidence=True, innovation=True)
    cfg.update(options)
    net = SharedSequentialModel(BSTConfig(**cfg)).eval()
    with torch.no_grad():
        net.value_weight.normal_(0, .3)
        net.value_bias.normal_(0, .1)
    source = torch.tensor([[-3., 0.], [2., 1.], [-1., 1.], [4., 0.], [-6., 1.]])
    ids = torch.tensor([[0, 1, 2, 3, 4, -1]])
    relative = torch.tensor([[[1., 0., -50., -1.], [2., 1., 0., 0.],
                              [3., 0., 5., 0.], [1., 2., 50., 1.],
                              [2., 0., 100., 2.], [0., 0., 0., 0.]]])
    return net, source, ids, relative, torch.tensor([-.5]), torch.tensor([1.])


def close(a, b, atol=1e-6, rtol=1e-5):
    torch.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


class BirthInnovationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_age_zero_is_exact_prior_and_parameter_source_gradients_cancel(self):
        net, source, ids, relative, prior, polarity = example()
        source.requires_grad_()
        prior.requires_grad_()
        score, state, _ = net(source, ids, relative, prior, polarity, 0)
        close(score, prior, atol=0, rtol=0)
        close(state["correction"], torch.zeros_like(prior), atol=0, rtol=0)
        self.assertGreater(float(state["birth_potential"].abs().max()), 1e-5)
        score.sum().backward()
        close(prior.grad, torch.ones_like(prior), atol=0, rtol=0)
        close(source.grad, torch.zeros_like(source), atol=0, rtol=0)
        for name, parameter in net.named_parameters():
            if parameter.grad is not None:
                close(parameter.grad, torch.zeros_like(parameter.grad), atol=0, rtol=0)

    def test_late_first_call_with_nonempty_birth_is_rejected(self):
        net, source, ids, relative, prior, polarity = example()
        for state in (None, net.make_state(prior, polarity)):
            for age in (1, torch.tensor([1.])):
                with self.assertRaisesRegex(ValueError, "age-zero birth reference"):
                    net(source, ids, relative, prior, polarity, age, state=state)

    def test_mixed_ages_validate_each_valid_row_and_preserve_uninitialized_padding(self):
        net, source, ids, relative, prior, polarity = example()
        ids, relative = ids.expand(2, -1), relative.expand(2, -1, -1)
        prior, polarity = prior.expand(2), polarity.expand(2)
        ages = torch.tensor([0., 1.])
        with self.assertRaisesRegex(ValueError, "age-zero birth reference"):
            net(source, ids, relative, prior, polarity, ages)
        _, state, _ = net(source, ids, relative, prior, polarity, ages,
                          valid=torch.tensor([True, False]))
        self.assertEqual(state["birth_initialized"].tolist(), [True, False])
        with self.assertRaisesRegex(ValueError, "age-zero birth reference"):
            net(source, ids, relative, prior, polarity, torch.ones(2), state=state)

    def test_mixed_initialized_empty_birth_and_new_birth_match_individual_rollouts(self):
        net, source, ids, relative, prior, polarity = example()
        encoded = net.encode_sources(source)
        _, initialized, _ = net.reaudit(encoded, ids, relative, None, prior, polarity, 0)
        empty_ids = ids.clone()
        empty_ids[:, :3] = -1  # only positive-window tokens remain
        empty_relative = relative.clone()
        empty_relative[:, :3] = 0
        states = [initialized, net.make_state(prior, polarity), net.make_state(prior, polarity)]
        batch_state = concat_states(states)
        batch_ids = torch.cat([ids, empty_ids, ids])
        batch_relative = torch.cat([relative, empty_relative, relative])
        scores, after, _ = net.reaudit(encoded, batch_ids, batch_relative, batch_state,
                                      prior.expand(3), polarity.expand(3), torch.tensor([1., 1., 0.]))
        _, empty_birth, _ = net.reaudit(encoded, empty_ids, empty_relative, None, prior, polarity, 0)
        expected_inputs = [(ids, relative, initialized, 1), (empty_ids, empty_relative, empty_birth, 1),
                           (ids, relative, None, 0)]
        for row, (own_ids, own_rel, own_state, age) in enumerate(expected_inputs):
            expected, expected_state, _ = net.reaudit(encoded, own_ids, own_rel, own_state, prior, polarity, age)
            close(scores[row:row+1], expected)
            actual = take_state(after, torch.tensor([row]))
            for name, value in expected_state.items():
                if value.dtype == torch.bool:
                    self.assertTrue(torch.equal(actual[name], value), name)
                else:
                    close(actual[name], value)
        self.assertEqual(after["birth_initialized"].tolist(), [True, True, True])

    def test_stream_registers_skipped_empty_birth_and_matches_checked_offline(self):
        net, _, _, _, _, _ = example()
        rec = Recording("empty_birth", [0, 0, 0], [0, 0, 0], [0, 50, 100],
                        [1, -1, 1], [-.5, .2, .8], [0, 0, 1], 50, n_windows=4)
        evidence = dict(past_windows=1, max_wait=3, k_near=2, k_evidence=0, radius=0)
        observed_late_initialized = []
        original = net.reaudit
        def checked(*args, **kwargs):
            state, age = args[3], args[6]
            age = net._ages(age, args[4])
            if bool((age > 0).any()):
                observed_late_initialized.append(bool(state["birth_initialized"][age > 0].all()))
            kwargs["validate_birth"] = True
            return original(*args, **kwargs)
        with mock.patch.object(net, "reaudit", side_effect=checked):
            streamed = run_recording(net, rec, evidence, birth="all", return_traces=True)
            with torch.no_grad():
                replay = rollout(net, rec, NeighborIndex(rec, **evidence), np.arange(len(rec)))
        self.assertTrue(observed_late_initialized and all(observed_late_initialized))
        final = replay["logits"].gather(1, replay["max_age"][:, None]).squeeze(1).numpy()
        np.testing.assert_allclose(streamed["logits"], final, atol=2e-6, rtol=1e-5)
        self.assertFalse(bool(streamed["traces"][0]["updated"].any()))

    def test_final_architecture_trim_and_geometry_preserve_full_training_step(self):
        cfg = json.loads((Path(__file__).resolve().parents[1]/"configs/bst_final.json").read_text())
        self.assertTrue(cfg["model"]["innovation"] and cfg["model"]["null_evidence"])
        torch.manual_seed(237)
        initial = SharedSequentialModel(BSTConfig(**cfg["model"]))
        # Nonzero potentials exercise source, query, window, null and birth
        # reference gradients; zero residual initialization would hide errors.
        with torch.no_grad():
            initial.value_weight.normal_(0, .3)
            initial.value_bias.normal_(0, .1)
        initial_weights = copy.deepcopy(initial.state_dict())
        rec = Recording("r3_training", [0, 1, 2, 1, 2, 0, 1], [0]*7,
                        [0, 50, 60, 150, 160, 200, 250], [1, -1, 1, -1, 1, -1, 1],
                        [-.5, .2, .8, -1.3, 2.1, -.7, 1.2], [0, 1, 0, 1, 0, 0, 1],
                        50, n_windows=6)
        # Empty birth at window 0; an empty intervening window; mixed remaining
        # horizons including the terminal age-zero-only query at the right edge.
        ids = np.array([0, 1, 3, 5, 6], dtype=np.int64)
        results = []
        for prepared, trim in ((False, False), (False, True), (True, False), (True, True)):
            block = SharedSequentialModel(BSTConfig(**cfg["model"])).train()
            block.load_state_dict(initial_weights)
            index = NeighborIndex(rec, **cfg["evidence"])
            batch = next(iter(GeometryBatches(index, ids, len(ids), cache_mb=64)))[2] if prepared else None
            inputs = {}
            encode, make_state = block.encode_sources, block.make_state
            def capture_sources(source):
                inputs["source"] = source.requires_grad_()
                return encode(source)
            def capture_prior(prior, polarity):
                inputs["prior"] = prior.requires_grad_()
                inputs["polarity"] = polarity.requires_grad_()
                return make_state(prior, polarity)
            optimizer = torch.optim.AdamW(block.parameters(), lr=.001, weight_decay=.0001)
            with mock.patch.object(block, "encode_sources", side_effect=capture_sources), \
                 mock.patch.object(block, "make_state", side_effect=capture_prior):
                output = rollout(block, rec, index, ids, prepared_batch=batch, trim_prefix=trim)
            loss = weighted_event_loss(output["logits"], output["valid"],
                                       torch.as_tensor(rec.label[ids]), torch.tensor([1., 2., 1., 3., 1.])) / len(ids)
            loss.backward()
            gradients = {name: None if p.grad is None else p.grad.detach().clone()
                         for name, p in block.named_parameters()}
            input_gradients = {name: value.grad.detach().clone() for name, value in inputs.items()}
            torch.nn.utils.clip_grad_norm_(block.parameters(), 1.)
            optimizer.step()
            results.append({"logits": output["logits"].detach(), "valid": output["valid"],
                            "gradients": gradients, "input_gradients": input_gradients,
                            "parameters": copy.deepcopy(block.state_dict())})
        reference = results[0]
        self.assertEqual(reference["valid"].sum(1).tolist(), [6, 5, 3, 2, 1])
        for actual in results[1:]:
            self.assertTrue(torch.equal(actual["valid"], reference["valid"]))
            close(actual["logits"], reference["logits"], atol=2e-6, rtol=1e-5)
            for field in ("gradients", "input_gradients", "parameters"):
                for name, expected in reference[field].items():
                    if expected is None:
                        self.assertIsNone(actual[field][name], name)
                    else:
                        close(actual[field][name], expected, atol=2e-6, rtol=1e-4)

    def test_birth_reference_is_differentiable_and_receives_negative_unit_gradient(self):
        net, source, ids, relative, prior, polarity = example()
        source.requires_grad_()
        encoded = net.encode_sources(source)
        _, state, _ = net.reaudit(encoded, ids, relative, None, prior, polarity, 0)
        reference = state["birth_potential"]
        self.assertTrue(reference.requires_grad)
        reference_gradient = torch.autograd.grad(reference.sum(), source, retain_graph=True)[0]
        self.assertGreater(float(reference_gradient[:3].abs().sum()), 1e-6)
        score, _, _ = net.reaudit(encoded, ids, relative, state, prior, polarity, 1)
        close(torch.autograd.grad(score.sum(), reference, retain_graph=True)[0], -torch.ones_like(reference),
              atol=0, rtol=0)
        self.assertTrue(bool(torch.isfinite(torch.autograd.grad(score.sum(), source)[0]).all()))

    def test_only_arrived_future_evidence_can_change_birth_belief(self):
        net, source, ids, relative, prior, polarity = example()
        encoded = net.encode_sources(source)
        birth, state, _ = net.reaudit(encoded, ids, relative, None, prior, polarity, 0)
        # Future slots are present in storage but invisible at birth.
        changed = source.clone()
        changed[3:] = torch.tensor([[1000., -1.], [-1000., 1.]])
        birth_changed, altered, _ = net(changed, ids, relative, prior, polarity, 0)
        close(birth, birth_changed, atol=0, rtol=0)
        close(state["birth_potential"], altered["birth_potential"], atol=0, rtol=0)
        later, next_state, _ = net.reaudit(encoded, ids, relative, state, prior, polarity, 1)
        self.assertGreater(float((later-birth).abs().max()), 1e-6)
        close(state["birth_potential"], next_state["birth_potential"], atol=0, rtol=0)

    def test_new_content_still_reweights_revalues_and_discounts_old_sources(self):
        net, source, ids, relative, prior, polarity = example()
        _, birth_state, _ = net(source, ids, relative, prior, polarity, 0)
        changed = source.clone()
        changed[3] = torch.tensor([-15., 1.])
        _, _, a = net(source, ids, relative, prior, polarity, 1, state=birth_state, return_tokens=True)
        _, _, b = net(changed, ids, relative, prior, polarity, 1, state=birth_state, return_tokens=True)
        for key in ("gate", "value"):
            self.assertGreater(float((a[key][:, :3]-b[key][:, :3]).abs().max()), 1e-6, key)
        self.assertGreater(float((a["window_discount"][:, :2]-b["window_discount"][:, :2]).abs().max()), 1e-6)
        close(a["birth_potential"], b["birth_potential"], atol=0, rtol=0)

    def test_empty_birth_does_not_rebase_on_first_future_observation(self):
        net, source, ids, relative, prior, polarity = example()
        # The only selected source arrives one window after the query.
        ids, relative = ids[:, 3:4], relative[:, 3:4]
        encoded = net.encode_sources(source)
        first, state, _ = net.reaudit(encoded, ids, relative, None, prior, polarity, 0)
        close(first, prior, atol=0, rtol=0)
        close(state["birth_potential"], torch.zeros_like(prior), atol=0, rtol=0)
        later, new_state, diag = net.reaudit(encoded, ids, relative, state, prior, polarity, 1)
        self.assertTrue(bool(diag["updated"].all()))
        close(new_state["birth_potential"], torch.zeros_like(prior), atol=0, rtol=0)
        close(later-prior, diag["head_correction"].sum(-1))
        self.assertGreater(float((later-prior).abs().max()), 1e-6)

    def test_same_prefix_and_later_empty_window_are_exact_belief_noops(self):
        net, source, ids, relative, prior, polarity = example()
        encoded = net.encode_sources(source)
        _, state, _ = net.reaudit(encoded, ids, relative, None, prior, polarity, 0)
        for age in (1, 2):
            score, state, _ = net.reaudit(encoded, ids, relative, state, prior, polarity, age)
        for age in (2, 3):
            repeated, next_state, diag = net.reaudit(encoded, ids, relative, state, prior, polarity, age)
            self.assertFalse(bool(diag["updated"].any()))
            close(score, repeated, atol=0, rtol=0)
            for name in ("context", "anchor", "correction", "birth_potential", "mass", "neff", "units"):
                close(state[name], next_state[name], atol=0, rtol=0)

    def test_force_update_does_not_replace_an_existing_birth_reference(self):
        net, source, ids, relative, prior, polarity = example()
        encoded = net.encode_sources(source)
        _, state, _ = net.reaudit(encoded, ids, relative, None, prior, polarity, 0)
        _, forced, _ = net.reaudit(encoded, ids, relative, state, prior, polarity, 0, force_update=True)
        close(forced["birth_potential"], state["birth_potential"], atol=0, rtol=0)

    def test_cached_pair_rollout_preserves_outputs_and_all_parameter_gradients(self):
        net, source, ids, relative, prior, polarity = example()
        reference = copy.deepcopy(net)
        results, gradients = [], []
        for block, cached in ((net, False), (reference, True)):
            inp = source.clone().requires_grad_()
            encoded = block.encode_sources(inp)
            pairs = block.prepare_pairs(encoded, ids, relative) if cached else None
            state, scores = None, []
            for age in range(3):
                score, state, _ = block.reaudit(encoded, ids, relative, state, prior, polarity, age, pair_cache=pairs)
                scores.append(score)
            scores = torch.stack(scores)
            results.append(scores)
            gradients.append(torch.autograd.grad(scores.square().sum(), list(block.parameters())+[inp]))
        close(results[0], results[1], atol=0, rtol=0)
        for a, b in zip(*gradients):
            close(a, b, atol=2e-6, rtol=1e-4)

    def test_online_cached_and_offline_causal_paths_agree(self):
        net, _, _, _, _, _ = example()
        timestamps = np.repeat(np.arange(5)*50, 3)+np.tile([1, 10, 25], 5)
        rec = Recording("innovation", np.tile([0, 1, 2], 5), np.zeros(15), timestamps,
                        np.tile([1, -1, 1], 5), np.tile([-.5, .2, .8], 5), np.tile([0, 0, 1], 5), 50)
        evidence = dict(past_windows=1, max_wait=3, k_near=2, k_evidence=0, radius=0)
        cached = run_recording(net, rec, evidence, birth="all", return_traces=True)
        placeholder = Recording(rec.name, rec.x, rec.y, rec.t, rec.p, np.zeros(len(rec)), rec.label, 50)
        def arrival(window, ids):
            return {"prior": rec.z0[ids], "source_features": rec.source_features[ids]}
        online = run_recording(net, placeholder, evidence, birth="all", return_traces=True,
                               arrival_provider=arrival)
        np.testing.assert_array_equal(cached["logits"], online["logits"])
        np.testing.assert_array_equal(cached["lags"], online["lags"])
        with torch.no_grad():
            replay = rollout(net, rec, NeighborIndex(rec, **evidence), np.arange(len(rec)))
        np.testing.assert_array_equal(replay["logits"][:, 0].numpy(), rec.z0)
        final = replay["logits"].gather(1, replay["max_age"][:, None]).squeeze(1).numpy()
        np.testing.assert_allclose(final, online["logits"], atol=2e-6, rtol=1e-5)
        for trace in online["traces"]:
            self.assertLessEqual(trace["max_encoded_window"], trace["window"])

    def test_explanation_reconstructs_innovation_including_birth_and_gap(self):
        net, _, _, _, _, _ = example()
        rec = Recording("explain_innovation", [0, 1, 2, 3], [0, 0, 0, 0], [0, 1, 50, 150],
                        [1, -1, 1, -1], [.2, -.5, 1.2, 10], [1, 0, 1, 0], 50, n_windows=4)
        evidence = dict(past_windows=1, max_wait=3, k_near=2, k_evidence=0, radius=0)
        report = explain_query(net, rec, evidence, 0)
        self.assertLess(report["max_reconstruction_error"], 1e-6)
        steps = report["steps"]
        self.assertEqual(steps[0]["logit"], steps[0]["prior_logit"])
        self.assertTrue(all(token["total_contribution"] == 0 for token in steps[0]["tokens"]))
        for step in steps:
            total = sum(token["total_contribution"] for token in step["tokens"])
            self.assertAlmostEqual(total, step["logit"]-step["prior_logit"], places=6)
            windows = sum(sum(window["contribution"]) for window in step["window_potentials"])
            self.assertAlmostEqual(total, windows, places=6)
        self.assertFalse(steps[2]["score_updated"])
        self.assertEqual(len(steps[1]["tokens"]), len(steps[2]["tokens"]))
        for previous, carried in zip(steps[1]["tokens"], steps[2]["tokens"]):
            self.assertEqual(previous["neighbor_id"], carried["neighbor_id"])
            self.assertEqual(previous["total_contribution"], carried["total_contribution"])
        self.assertEqual(steps[2]["attribution_carried_from_age"], 1)
        self.assertEqual(steps[1]["contribution_sum"], steps[2]["contribution_sum"])

    def test_null_zero_support_returns_residual_context_to_initial_anchor(self):
        net, source, ids, relative, prior, polarity = example()
        state = net.make_state(prior, polarity)
        state["context"] = state["context"] + 5.
        with torch.no_grad():
            net.context_update[0].bias.fill_(3.)
            net.context_update[-2].bias.fill_(3.)
        original = net.window_evidence.forward
        def reject_all(weights, *args, **kwargs):
            return original(torch.zeros_like(weights), *args, **kwargs)
        with mock.patch.object(net.window_evidence, "forward", side_effect=reject_all):
            score, after, diag = net(source, ids, relative, prior, polarity, 0, state=state)
        self.assertTrue(bool(diag["updated"].all()))
        close(after["context"], state["anchor"], atol=0, rtol=0)
        close(after["mass"], torch.zeros_like(prior), atol=0, rtol=0)
        close(score, prior, atol=0, rtol=0)

    def test_later_null_abstention_removes_current_potential_not_frozen_reference(self):
        net, source, ids, relative, prior, polarity = example()
        _, birth_state, _ = net(source, ids, relative, prior, polarity, 0)
        self.assertGreater(float(birth_state["birth_potential"].abs().max()), 1e-5)
        original = net.window_evidence.forward
        def reject_all(weights, *args, **kwargs):
            return original(torch.zeros_like(weights), *args, **kwargs)
        with mock.patch.object(net.window_evidence, "forward", side_effect=reject_all):
            score, after, diag = net(source, ids, relative, prior, polarity, 1, state=birth_state)
        close(diag["head_correction"], torch.zeros_like(diag["head_correction"]), atol=0, rtol=0)
        close(after["context"], birth_state["anchor"], atol=0, rtol=0)
        close(after["birth_potential"], birth_state["birth_potential"], atol=0, rtol=0)
        close(score, prior-birth_state["birth_potential"], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
