"""Train-time prefix trajectories must equal real stopped-query execution.

Use a nonzero evidence reader, gaps, unsorted event rows and short tail
horizons. Comparing only zero-initialized logits would miss state/VoW drift.
"""
import unittest

import numpy as np
import torch

from model.bst import BSTConfig, SharedSequentialModel
from tools.bst.data import NeighborIndex, Recording
from tools.bst.pipeline import rollout
from tools.bst.stream import run_recording


EVIDENCE = dict(past_windows=2, max_wait=3, k_near=2, k_evidence=0, radius=0)


def recording():
    # Windows 1 and 3 are empty; births in windows 4/5/6 have ages 2/1/0.
    t = np.array([1, 19, 101, 123, 202, 227, 252, 301], np.int64)
    order = np.array([4, 0, 7, 2, 6, 1, 5, 3])
    return Recording(
        "prefix_contract", np.array([0, 1, 0, 2, 1, 3, 0, 2])[order],
        np.array([0, 1, 2, 0, 1, 0, 2, 1])[order], t[order],
        np.array([1, -1, 1, 1, -1, 1, -1, 1])[order],
        np.array([-0.9, 0.3, 1.1, -0.4, 0.7, -1.3, 1.7, -0.1], np.float32)[order],
        np.array([0, 0, 1, 0, 1, 0, 1, 0])[order], 50, n_windows=7)


def nonzero_model():
    torch.manual_seed(137)
    model = SharedSequentialModel(BSTConfig(
        source_dim=8, state_dim=6, relation_dim=3, heads=2, max_wait_steps=3))
    with torch.no_grad():
        model.value_weight.normal_(0, 0.35)
        model.value_bias.fill_(0.07)
    return model.eval()


class TestPipelineEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_close(self, a, b):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)

    def offline(self, model, rec, ids=None):
        if ids is None:
            ids = np.arange(len(rec), dtype=np.int64)
        with torch.no_grad():
            out = rollout(model, rec, NeighborIndex(rec, **EVIDENCE), ids,
                          collect_features=True)
        return {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in out.items()}

    def test_every_online_score_and_policy_feature_matches_training_prefix(self):
        model, rec = nonzero_model(), recording()
        offline = self.offline(model, rec)
        online = run_recording(model, rec, EVIDENCE, birth="all", batch_size=3,
                               return_traces=True)
        self.assertEqual(set(offline["max_age"].tolist()), {0, 1, 2, 3})
        self.assertGreater(float(np.max(np.abs(offline["logits"][:, 0] - rec.z0))), 1e-4)
        seen = set()
        for trace in online["traces"]:
            for row, event_id in enumerate(trace["ids"]):
                age = trace["window"] - rec.window[event_id]
                self.assertTrue(offline["valid"][event_id, age])
                self.assert_close(trace["logits"][row], offline["logits"][event_id, age])
                self.assert_close(trace["features"][row], offline["features"][event_id, age])
                seen.add((int(event_id), int(age)))
            if trace["window"] in (1, 3):
                self.assertFalse(trace["updated"].any())
                # No new evidence: the score delta fed to VoW is zero, while
                # elapsed/remaining time still changes.
                np.testing.assert_array_equal(trace["features"][:, model.config.state_dim + 4], 0)
        expected = {(i, d) for i in range(len(rec)) for d in range(int(offline["max_age"][i]) + 1)}
        self.assertEqual(seen, expected)
        self.assert_close(online["logits"], offline["logits"][np.arange(len(rec)), offline["max_age"]])
        np.testing.assert_array_equal(online["lags"], offline["max_age"])

    def test_shared_sources_and_minibatch_partition_leave_trajectories_unchanged(self):
        model, rec = nonzero_model(), recording()
        shared = self.offline(model, rec)
        # A query encoded alone has no other query's observations in its source
        # bank. Shared-bank row membership must never change a causal result.
        for event_id in range(len(rec)):
            single = self.offline(model, rec, np.array([event_id]))
            valid = shared["valid"][event_id]
            self.assert_close(single["logits"][0, valid], shared["logits"][event_id, valid])
            self.assert_close(single["features"][0, valid], shared["features"][event_id, valid])
        one = run_recording(model, rec, EVIDENCE, birth="all", batch_size=1, return_traces=True)
        many = run_recording(model, rec, EVIDENCE, birth="all", batch_size=32, return_traces=True)
        for a, b in zip(one["traces"], many["traces"]):
            np.testing.assert_array_equal(a["ids"], b["ids"])
            self.assert_close(a["logits"], b["logits"])
            self.assert_close(a["features"], b["features"])
        self.assertLessEqual(many["stats"]["source_encoded"], len(rec))
        self.assertGreater(many["stats"]["pair_visits"], many["stats"]["source_encoded"])

    def test_committing_one_query_preserves_other_queries_and_its_source_observation(self):
        model, rec = nonzero_model(), recording()
        full = run_recording(model, rec, EVIDENCE, birth="all", return_traces=True)
        victim = int(np.flatnonzero(rec.z0 == np.float32(-0.9))[0])
        prior_column = model.config.state_dim + 1
        victim_prior_feature = torch.asinh(torch.tensor(float(rec.z0[victim])))

        def stop_one(features, risk, age):
            # Identify the event by its unique immutable prior, never its label.
            return torch.isclose(features[:, prior_column], victim_prior_feature.to(features.device))

        stopped = run_recording(model, rec, EVIDENCE, birth="all", policy=stop_one,
                                batch_size=2, return_traces=True)
        self.assertEqual(stopped["lags"][victim], 0)
        self.assertGreater(full["stats"]["queries_updated"], stopped["stats"]["queries_updated"])
        for a, b in zip(full["traces"], stopped["traces"]):
            original = {int(i): row for row, i in enumerate(a["ids"])}
            for row, event_id in enumerate(b["ids"]):
                old_row = original[int(event_id)]
                self.assert_close(a["logits"][old_row], b["logits"][row])
                self.assert_close(a["features"][old_row], b["features"][row])
            if b["window"] > rec.window[victim]:
                self.assertNotIn(victim, b["ids"])
        kept = np.arange(len(rec)) != victim
        self.assert_close(full["logits"][kept], stopped["logits"][kept])
        # The committed event is still a valid source for the later birth at
        # window 2; removing its query must not delete the observation.
        later = int(np.flatnonzero(rec.t == 101)[0])
        neighbors = NeighborIndex(rec, **EVIDENCE).query([later], age=0)["idx"]
        self.assertIn(victim, neighbors)


if __name__ == "__main__":
    unittest.main()
