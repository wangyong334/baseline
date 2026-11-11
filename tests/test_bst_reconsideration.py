"""Mechanism traces are same-weight interventions, not oracle-selected examples."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from model.bst import BSTConfig, SharedSequentialModel
from tools.bst.audit_reconsideration import MODES, audit_query, checked_inputs, pack_tokens
from tools.bst.data import Recording


class ReconsiderationAudit(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.record = Recording("example", [0, 1, 2, 3], [0]*4, [0, 1, 50, 150],
                                [1, -1, 1, -1], [.2, -.5, 1.2, 10], [1, 0, 1, 0],
                                50, n_windows=4)
        self.config = {"threshold": .9,
                       "evidence": dict(past_windows=1, max_wait=2, k_near=2, k_evidence=0, radius=0)}

    def model(self, normalization="mean", learned=True):
        model = SharedSequentialModel(BSTConfig(source_dim=8, state_dim=6, heads=2,
                                                relation_dim=3, max_wait_steps=2, normalization=normalization))
        if learned:
            with torch.no_grad():
                model.value_weight.normal_(0, .2)
        return model

    def test_all_interventions_preserve_weights_causality_and_exact_no_op(self):
        model = self.model()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        traces = audit_query(model, self.record, self.config, 0)
        self.assertEqual(set(traces), {m[0] for m in MODES})
        self.assertTrue(model.config.reweight and model.config.revalue)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
        for trace in traces.values():
            self.assertTrue(all(x["exact_belief_no_op"] and not x["updated_again"] for x in trace["same_prefix_checks"]))
            self.assertLess(trace["max_reconstruction_error"], 1e-6)
            for step in trace["steps"]:
                self.assertTrue(all(t["time_ms"] < (step["window"]+1)*50 for t in step["tokens"]))
            self.assertFalse(trace["steps"][-1]["score_updated"])
        self.assertGreater(traces["full"]["old_token_summary"]["mean_absolute_change"]["value"], 0.)
        self.assertEqual(traces["no_revalue"]["old_token_summary"]["mean_absolute_change"]["value"], 0.)
        self.assertEqual(traces["no_reweight"]["old_token_summary"]["mean_absolute_change"]["gate"], 0.)

    def test_no_learned_effect_is_reported_without_failure(self):
        traces = audit_query(self.model(learned=False), self.record, self.config, 0)
        for trace in traces.values():
            self.assertEqual(trace["old_token_summary"]["max_absolute_change"]["total_contribution"], 0.)
            self.assertEqual(trace["comparison_to_full"]["max_absolute_prefix_logit_difference"], 0.)

    def test_npz_rows_reconstruct_scores_for_every_normalization(self):
        for normalization in ("mean", "sum", "softmax"):
            traces = audit_query(self.model(normalization), self.record, self.config, 0)
            arrays = pack_tokens([{"traces": traces}], 2)
            self.assertEqual(arrays["value"].shape[1], 2)
            for mi, (name, _, _) in enumerate(MODES):
                for step in traces[name]["steps"]:
                    mask = (arrays["mode_index"] == mi) & (arrays["age"] == step["age"])
                    total = arrays["contribution_per_head"][mask].sum()
                    self.assertAlmostEqual(float(total), step["logit"]-step["prior_logit"], places=6)

    def test_probe_permission_never_allows_known_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root/"data"
            prior = root/"priors"
            (raw/"test").mkdir(parents=True)
            prior.mkdir()
            (raw/"test"/"test.npz").touch()
            (prior/"test.npz").touch()
            metadata = {"split": "test", "backbone_sha256": "actual", "data_root": str(raw),
                        "expected_recordings": ["test.npz"], "completed_recordings": ["test.npz"], "complete": True}
            (prior/"prior_provenance.json").write_text(json.dumps(metadata), encoding="utf-8")
            ck = {"manifest": {"prior_provenance": {"backbone_sha256": "different"}}}
            with self.assertRaises(ValueError):
                checked_inputs(prior, ck, allow_probe=True)
            ck["manifest"]["prior_provenance"]["backbone_sha256"] = None
            with self.assertRaises(ValueError):
                checked_inputs(prior, ck)
            _, _, names, provenance = checked_inputs(prior, ck, allow_probe=True)
            self.assertEqual(names, ["test.npz"])
            self.assertTrue(provenance["probe"])
            self.assertFalse(provenance["identity_verified"])


if __name__ == "__main__":
    unittest.main()
