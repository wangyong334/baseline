"""The window profiler's pair-recompute oracle preserves stopped outputs."""
import unittest

import numpy as np
import torch

from model.bst import BSTConfig, SharedSequentialModel
from tools.bst.data import Recording
from tools.bst import stream
from tools.bst.profile_window_architecture import recompute_pairs, compare_outputs


class WindowProfileTests(unittest.TestCase):
    def test_recompute_matches_cached_and_restores_executor(self):
        torch.set_num_threads(1)
        torch.manual_seed(39)
        rng = np.random.RandomState(12)
        n = 36
        rec = Recording("tiny", rng.randint(0, 4, n), rng.randint(0, 4, n),
                        np.repeat([0, 50, 100, 200, 250, 300], 6), rng.choice([-1, 1], n),
                        rng.normal(size=n).astype(np.float32), rng.randint(0, 2, n), 50, n_windows=8)
        model = SharedSequentialModel(BSTConfig(source_dim=8, state_dim=6, relation_dim=3,
            heads=2, max_wait_steps=2, past_windows=2, normalization="window"))
        with torch.no_grad():
            model.value_weight.normal_(0, .15)
        evidence = dict(past_windows=2, max_wait=2, k_near=2, k_evidence=2, radius=0, filter_logit=0)
        kwargs = dict(birth="all", batch_size=5,
                      policy=lambda features, risk, age: (age >= 1) & (features[:, 2] > 0))
        cls = stream.CausalPairCache
        a = stream.run_recording(model, rec, evidence, **kwargs)
        with recompute_pairs(True):
            b = stream.run_recording(model, rec, evidence, **kwargs)
        self.assertIs(stream.CausalPairCache, cls)
        a["prior"], b["prior"] = rec.z0, rec.z0.copy()
        self.assertTrue(compare_outputs(a, b, rec.log_w)["equivalent"])
        self.assertGreater(a["stats"]["policy_commits"], 0)
        with self.assertRaisesRegex(RuntimeError, "test"):
            with recompute_pairs(True):
                raise RuntimeError("test")
        self.assertIs(stream.CausalPairCache, cls)


if __name__ == "__main__":
    unittest.main()
