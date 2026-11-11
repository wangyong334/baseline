"""Execution optimizations preserve neighbor identities and posterior traces."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.bst import BSTConfig, SharedSequentialModel
from tools.bst.data import Recording, NeighborIndex
from tools.bst.stream import CausalPairCache, run_recording


def fixture():
    rng = np.random.RandomState(41)
    t = np.repeat([0, 50, 150, 200, 300, 350, 400, 450], 9) + np.tile(np.arange(9), 8)
    n = len(t)
    order = rng.permutation(n)
    # Duplicate coordinates exercise distance ties and self-exclusion.
    return Recording("ties", rng.randint(0, 4, n)[order], rng.randint(0, 3, n)[order],
                     t[order], rng.choice([-1, 1], n), rng.randn(n).astype(np.float32),
                     rng.randint(0, 2, n), 50, n_windows=12)


class RecomputePairs(CausalPairCache):
    """Independent oracle: reconstruct every old pair on each read, no reuse."""
    def read(self, rows, end):
        saved = super().read(rows, end)
        global_ids = saved["indices"].cpu().numpy().copy()
        global_ids[~saved["present"].cpu().numpy()] = -1
        local = torch.as_tensor(self.source.local_indices(global_ids), device=self.source.device)
        result = self.model.prepare_pairs(self.source.encoded, local, saved["relative"])
        result["indices"], result["relative"] = saved["indices"], saved["relative"]
        return result


class ExecutionCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_compact_search_exactly_matches_full_slots(self):
        rec = fixture()
        for radius in [0, .1, 2.5]:
            for near, evidence in [(3, 3), (0, 3), (3, 0)]:
                index = NeighborIndex(rec, past_windows=2, max_wait=3, k_near=near,
                                      k_evidence=evidence, filter_logit=0, radius=radius)
                for window in range(rec.n_windows):
                    q = np.flatnonzero((rec.window <= window) & (rec.window >= window-3))
                    compact = index.query_arriving(q, window)
                    full = index.query(q, current_window=window, new_only=True)
                    slots = compact["slot"][:, None] + np.arange(index.K)
                    rows = np.arange(len(q))[:, None]
                    np.testing.assert_array_equal(compact["idx"], full["idx"][rows, slots])
                    np.testing.assert_array_equal(compact["relative"], full["relative"][rows, slots])

    def test_pair_reuse_matches_full_recompute_with_real_stopping(self):
        rec = fixture()
        evidence = dict(past_windows=2, max_wait=3, k_near=3, k_evidence=2,
                        radius=0, filter_logit=0)
        for normalization in ["mean", "sum", "softmax"]:
            torch.manual_seed(37)
            model = SharedSequentialModel(BSTConfig(source_dim=8, state_dim=6,
                relation_dim=3, heads=2, max_wait_steps=3, normalization=normalization))
            with torch.no_grad():
                model.value_weight.normal_(0, .2)
            # Different birth cohorts stop at different ages, exercising slot reuse.
            policy = lambda f, risk, age: (age >= 1) & (f[:, 2] > 0)
            kwargs = dict(birth="all", policy=policy, batch_size=7, return_traces=True)
            actual = run_recording(model, rec, evidence, **kwargs)
            with patch("tools.bst.stream.CausalPairCache", RecomputePairs):
                oracle = run_recording(model, rec, evidence, **kwargs)
            self.assertGreater(actual["stats"]["policy_commits"], 0)
            self.assertGreater(actual["stats"]["deadline_commits"], 0)
            self.assertLess(actual["stats"]["pairs_prepared"], actual["stats"]["pair_visits"])
            for key in ["lags", "reasons"]:
                np.testing.assert_array_equal(actual[key], oracle[key])
            for a, b in zip(actual["traces"], oracle["traces"]):
                for key in ["ids", "committed", "updated"]:
                    np.testing.assert_array_equal(a[key], b[key])
                for key in ["logits", "features"]:
                    np.testing.assert_allclose(a[key], b[key], atol=2e-6, rtol=2e-6)
                self.assertLessEqual(a["max_encoded_window"], a["window"])

    def test_batched_policy_validation_still_rejects_invalid_values(self):
        rec = fixture()
        model = SharedSequentialModel(BSTConfig(max_wait_steps=3))
        class BadPolicy(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value
            def forward(self, features):
                return features.new_full((len(features),), self.value)
        for value in [float("nan"), float("inf"), -1.]:
            with self.assertRaisesRegex(ValueError, "finite nonnegative"):
                run_recording(model, rec, dict(past_windows=2, max_wait=3, k_near=2),
                              birth="all", batch_size=2, policy=BadPolicy(value))

    def test_fused_projections_refresh_after_weight_and_dtype_changes(self):
        torch.manual_seed(41)
        model = SharedSequentialModel().eval()
        x = torch.randn(33, 2)
        with torch.no_grad():
            first = model.encode_sources(x)
            self.assertTrue(model._inference_projection_cache)
            model.source_key.weight.add_(.03)
            changed = model.encode_sources(x)
            self.assertFalse(torch.equal(first["key"], changed["key"]))
            model.fused_inference = False
            explicit = model.encode_sources(x)
            for key in changed:
                torch.testing.assert_allclose(changed[key], explicit[key], atol=2e-6, rtol=2e-6)
            model.double()
            explicit = model.encode_sources(x.double())
            model.fused_inference = True
            joined = model.encode_sources(x.double())
            for key in joined:
                torch.testing.assert_allclose(joined[key], explicit[key], atol=1e-12, rtol=1e-12)
        model.train()
        self.assertFalse(model._inference_projection_cache)
        model.encode_sources(x.double())["key"].sum().backward()
        self.assertTrue(torch.isfinite(model.source_key.weight.grad).all())
        self.assertFalse(model._inference_projection_cache)

    def test_fusion_preserves_all_ablation_modes(self):
        for reweight, revalue in [(False, False), (False, True), (True, False), (True, True)]:
            torch.manual_seed(4)
            net = SharedSequentialModel(BSTConfig(max_wait_steps=3, reweight=reweight, revalue=revalue))
            with torch.no_grad():
                net.value_weight.normal_(0, .3)
            kwargs = dict(birth="all", batch_size=8, return_traces=True)
            evidence = dict(past_windows=2, max_wait=3, k_near=3)
            net.fused_inference = False
            old = run_recording(net, fixture(), evidence, **kwargs)
            net.fused_inference = True
            new = run_recording(net, fixture(), evidence, **kwargs)
            for a, b in zip(old["traces"], new["traces"]):
                np.testing.assert_allclose(a["logits"], b["logits"], atol=2e-6, rtol=2e-6)
                np.testing.assert_allclose(a["features"], b["features"], atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
