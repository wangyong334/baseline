"""Explanations reconstruct applied evidence, including unchanged prefixes."""
import unittest

import numpy as np
import torch

from model.bst import BSTConfig, SharedSequentialModel
from tools.bst.data import Recording
from tools.bst.explain import explain_query


class TestEvidenceExplanation(unittest.TestCase):
    def test_actual_token_contributions_reconstruct_all_normalizations(self):
        torch.set_num_threads(1)
        rec = Recording("explain", [0, 1, 2, 3], [0, 0, 0, 0], [0, 1, 50, 150],
                        [1, -1, 1, -1], [.2, -.5, 1.2, 10], [1, 0, 1, 0], 50,
                        n_windows=4)
        for normalization in ("mean", "sum", "softmax"):
            torch.manual_seed(17)
            model = SharedSequentialModel(BSTConfig(source_dim=8, state_dim=6, heads=2,
                                                   relation_dim=3, max_wait_steps=2,
                                                   normalization=normalization))
            with torch.no_grad():
                model.value_weight.normal_(0, .2)
            report = explain_query(model, rec, dict(past_windows=1, max_wait=2, k_near=2,
                                                    k_evidence=0, radius=0), 0)
            self.assertLess(report["max_reconstruction_error"], 1e-6)
            steps = report["steps"]
            self.assertEqual(len(steps), 3)
            for step in steps:
                total = sum(token["total_contribution"] for token in step["tokens"])
                self.assertAlmostEqual(total, step["logit"] - step["prior_logit"], places=6)
                self.assertTrue(all(token["time_ms"] < (step["window"] + 1) * 50 for token in step["tokens"]))
            self.assertFalse(steps[2]["score_updated"])
            self.assertEqual(steps[2]["attribution_carried_from_age"], 1)
            self.assertEqual(steps[1]["logit"], steps[2]["logit"])
            self.assertEqual(steps[2]["old_evidence_change"]["mean_abs_value_change"], 0)
            self.assertGreater(steps[1]["old_evidence_change"]["mean_abs_value_change"], 0)


if __name__ == "__main__":
    unittest.main()
