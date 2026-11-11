"""Stopped-query execution, shared caches and chronological availability."""
import unittest

import numpy as np
import torch

from model.bst import BSTConfig, SharedSequentialModel, ContinuationValueHead
from tools.bst.data import Recording, NeighborIndex
from tools.bst.stream import CausalSourceCache, run_recording


def recording(future_shift=0.0):
    # Three events/window, with two nearby queries sharing the same neighbours.
    t = np.repeat(np.arange(5) * 50, 3) + np.tile([1, 10, 25], 5)
    z0 = np.tile(np.array([-0.5, 0.2, 0.8], np.float32), 5)
    z0[t >= 150] += future_shift
    return Recording("stream", np.tile([0, 1, 2], 5), np.zeros(15), t,
                     np.tile([1, -1, 1], 5), z0, np.tile([0, 0, 1], 5), 50)


def model():
    torch.manual_seed(37)
    net = SharedSequentialModel(BSTConfig(source_dim=8, state_dim=6, relation_dim=3,
                                         heads=2, max_wait_steps=2))
    with torch.no_grad():
        net.value_weight.normal_(0, 0.15)
    return net


EVIDENCE = dict(past_windows=1, max_wait=2, k_near=2, k_evidence=0, radius=0)


class TestStoppedStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_commit_really_removes_future_query_work(self):
        rec, net = recording(), model()
        full = run_recording(net, rec, EVIDENCE, birth="all", return_traces=True)
        immediate = run_recording(net, rec, EVIDENCE, birth="all",
                                  policy=lambda f, r, a: torch.ones_like(r, dtype=torch.bool),
                                  return_traces=True)
        self.assertGreater(full["stats"]["queries_updated"], immediate["stats"]["queries_updated"])
        self.assertGreater(full["stats"]["pair_visits"], immediate["stats"]["pair_visits"])
        np.testing.assert_array_equal(immediate["lags"], 0)
        for trace in immediate["traces"]:
            self.assertTrue(np.all(rec.window[trace["ids"]] == trace["window"]))
        self.assertEqual(full["stats"]["remaining_active"], 0)

    def test_future_perturbations_cannot_change_prefix(self):
        net = model()
        original = run_recording(net, recording(), EVIDENCE, birth="all", return_traces=True)
        changed = run_recording(net, recording(20), EVIDENCE, birth="all", return_traces=True)
        for first, second in zip(original["traces"][:3], changed["traces"][:3]):
            np.testing.assert_array_equal(first["ids"], second["ids"])
            np.testing.assert_allclose(first["logits"], second["logits"], atol=1e-7)
        for trace in original["traces"]:
            self.assertLessEqual(trace["max_encoded_window"], trace["window"])

    def test_deadline_and_final_window_flush(self):
        rec = recording()
        result = run_recording(model(), rec, EVIDENCE, birth="all")
        np.testing.assert_array_equal(result["lags"], np.minimum(2, rec.n_windows - 1 - rec.window))
        self.assertEqual(result["stats"]["deadline_commits"], len(rec))
        self.assertTrue(np.isfinite(result["logits"]).all())

    def test_source_encoding_reused_and_committed_observations_retained(self):
        rec = recording()
        result = run_recording(model(), rec, EVIDENCE, birth="all",
                               policy=lambda f, r, a: torch.ones_like(r, dtype=torch.bool))
        self.assertLessEqual(result["stats"]["source_encoded"], len(rec))
        self.assertGreater(result["stats"]["pair_visits"], result["stats"]["source_encoded"])
        # New queries still read past observations whose own labels were committed.
        self.assertGreater(result["stats"]["pair_visits"], 2 * len(rec))

    def test_no_new_evidence_does_not_run_sdu_again(self):
        rec = Recording("gap", [0, 1], [0, 0], [0, 1], [1, -1], [0.1, -0.2], [1, 0],
                        50, n_windows=4)
        result = run_recording(model(), rec, EVIDENCE, birth="all", return_traces=True)
        self.assertEqual(result["stats"]["queries_updated"], 2)
        self.assertEqual(result["stats"]["decision_checks"], 6)
        self.assertFalse(result["traces"][1]["updated"].any())
        np.testing.assert_array_equal(result["lags"], 2)

    def test_birth_gate_precedes_neighbor_and_source_work(self):
        calls = []
        def gate(ids, rec, window):
            calls.append((window, ids.copy()))
            return np.zeros(len(ids), bool)
        result = run_recording(model(), recording(), EVIDENCE, birth="learned", birth_policy=gate)
        self.assertEqual(len(calls), 5)
        self.assertEqual(result["stats"]["queries_updated"], 0)
        self.assertEqual(result["stats"]["source_encoded"], 0)
        self.assertEqual(result["stats"]["trees_built"], 0)
        self.assertEqual(result["stats"]["birth_commits"], 15)

    def test_batch_size_and_record_reset_do_not_change_answers(self):
        net, rec = model(), recording()
        a = run_recording(net, rec, EVIDENCE, birth="all", batch_size=1)
        b = run_recording(net, rec, EVIDENCE, birth="all", batch_size=50)
        c = run_recording(net, rec, EVIDENCE, birth="all", batch_size=1)
        np.testing.assert_allclose(a["logits"], b["logits"], atol=2e-7)
        np.testing.assert_array_equal(a["logits"], c["logits"])
        self.assertEqual(a["stats"]["source_encoded"], c["stats"]["source_encoded"])
        self.assertTrue(net.training)  # Caller mode restored.

    def test_cache_rejects_future_reads_and_evicts(self):
        net, rec = model(), recording()
        with torch.no_grad():
            cache = CausalSourceCache(net, rec, "cpu", 2)
            with self.assertRaisesRegex(ValueError, "future"):
                cache.ensure([9], 1)
            cache.ensure([0, 1, 3], 1)
            cache.ensure([0, 1, 3], 1)
            self.assertEqual(cache.encoded_total, 3)
            cache.evict_before(1)
            np.testing.assert_array_equal(cache.source_ids, [3])
            with self.assertRaisesRegex(RuntimeError, "evicted"):
                cache.local_indices([[0]])

    def test_new_only_index_matches_full_prefix(self):
        rec = recording()
        index = NeighborIndex(rec, **EVIDENCE)
        prefix = index.query([3, 4], age=0)
        for age in [1, 2]:
            new = index.query([3, 4], age=age, new_only=True)
            prefix["idx"][new["mask"]] = new["idx"][new["mask"]]
            prefix["relative"][new["mask"]] = new["relative"][new["mask"]]
            complete = index.query([3, 4], age=age)
            np.testing.assert_array_equal(prefix["idx"], complete["idx"])
            np.testing.assert_array_equal(prefix["relative"], complete["relative"])

    def test_cost_threshold_consistency_and_value_head(self):
        net = model()
        head = ContinuationValueHead(net.value_feature_dim, width=4)
        result = run_recording(net, recording(), EVIDENCE, birth="all", policy=head, wait_cost=1)
        np.testing.assert_array_equal(result["lags"], 0)
        self.assertTrue(head.training)
        with self.assertRaisesRegex(ValueError, "ratio"):
            run_recording(net, recording(), EVIDENCE, fp_cost=1, fn_cost=1, threshold=.9)

    def test_empty_recording(self):
        rec = Recording("empty", [], [], [], [], [], [], 50, n_windows=3)
        result = run_recording(model(), rec, EVIDENCE, birth="all")
        self.assertEqual(result["logits"].shape, (0,))
        self.assertEqual(result["stats"]["remaining_active"], 0)

    def test_online_arrival_provider_matches_cached_and_does_not_mutate_input(self):
        net, truth = model(), recording()
        cached = run_recording(net, truth, EVIDENCE, birth="all", return_traces=True)
        placeholder = Recording(truth.name, truth.x, truth.y, truth.t, truth.p,
                                np.zeros(len(truth)), truth.label, 50)
        calls = []
        def provider(window, ids):
            calls.append(window)
            self.assertTrue(np.all(truth.window[ids] == window))
            return {"prior": truth.z0[ids], "source_features": truth.source_features[ids]}
        live = run_recording(net, placeholder, EVIDENCE, birth="all", return_traces=True,
                             arrival_provider=provider)
        self.assertEqual(calls, list(range(truth.n_windows)))
        np.testing.assert_array_equal(cached["logits"], live["logits"])
        np.testing.assert_array_equal(cached["lags"], live["lags"])
        np.testing.assert_array_equal(placeholder.z0, 0)
        np.testing.assert_array_equal(placeholder.source_features[:, 0], 0)
        for a, b in zip(cached["traces"], live["traces"]):
            np.testing.assert_array_equal(a["logits"], b["logits"])
            self.assertLessEqual(b["max_encoded_window"], b["window"])

    def test_provider_runs_before_birth_filter_and_in_empty_windows(self):
        rec = Recording("gaps", [0, 1], [0, 0], [0, 101], [1, -1], [0, 0], [1, 0],
                        50, n_windows=4)
        order = []
        def provider(window, ids):
            order.append(("provider", window))
            return {"prior": np.full(len(ids), 2 + window, np.float32)}
        def gate(ids, live, window):
            order.append(("birth", window))
            np.testing.assert_array_equal(live.z0[ids], 2 + window)
            return np.zeros(len(ids), bool)
        result = run_recording(model(), rec, EVIDENCE, birth="learned", birth_policy=gate,
                               arrival_provider=provider)
        self.assertEqual(order, [(kind, w) for w in range(4) for kind in ("provider", "birth")])
        np.testing.assert_array_equal(result["logits"], [2, 4])
        np.testing.assert_array_equal(rec.z0, 0)


if __name__ == "__main__":
    unittest.main()
