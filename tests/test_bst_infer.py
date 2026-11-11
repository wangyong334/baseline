"""Original event identity at the online backbone/decision boundary."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from tools.bst.data import Recording
from tools.bst.infer import BackboneArrivalProvider, recording_from_sequence, verify_backbone_provenance


class FakeBackbone(torch.nn.Module):
    def forward(self, x, events, states, state_mode="carry"):
        self.last_state = states
        return events["raw_logit"], 1 if states is None else states + 1, None


class FakeSource:
    def window(self, window):
        ids = [np.array([4, 1]), np.array([3]), np.empty(0, np.int64)][window]
        return None, {"raw_logit": torch.as_tensor(ids + 100, dtype=torch.float32)}, None, ids


class TestOnlineBackboneAdapter(unittest.TestCase):
    def test_reorders_backbone_scores_to_original_query_rows(self):
        rec = Recording("small", [0, 1, 2], [0, 0, 0], [1, 55, 0], [1, -1, 1],
                        [0, 0, 0], [1, 0, 1], 50, n_windows=3)
        backbone = FakeBackbone()
        provider = BackboneArrivalProvider(backbone, FakeSource(), rec, [1, 3, 4], 5, 3.0)
        first = provider(0, np.array([0, 2]))
        np.testing.assert_array_equal(first["prior"], [98, 101])
        second = provider(1, np.array([1]))
        np.testing.assert_array_equal(second["prior"], [100])
        provider(2, np.empty(0, np.int64))
        self.assertEqual(provider.windows, 3)
        self.assertEqual(provider.events, 3)
        self.assertEqual(provider.states, 3)
        np.testing.assert_array_equal(provider.arrival_scores, [98, 100, 101])

    def test_rejects_repeated_windows_and_wrong_original_ids(self):
        rec = Recording("small", [0, 1, 2], [0, 0, 0], [1, 55, 0], [1, -1, 1],
                        [0, 0, 0], [1, 0, 1], 50)
        provider = BackboneArrivalProvider(FakeBackbone(), FakeSource(), rec, [1, 3, 4], 5, 0)
        with self.assertRaisesRegex(ValueError, "consecutively"):
            provider(1, np.array([1]))
        with self.assertRaisesRegex(ValueError, "indices"):
            provider(0, np.array([1, 2]))

    def test_probe_truncation_keeps_original_mapping(self):
        seq = SimpleNamespace(name="raw", x=np.arange(4), y=np.zeros(4), t=np.array([120, 0, 55, 10]),
                              p=np.ones(4), label=np.array([1, 0, 1, 0]), target_id=np.arange(4), n_windows=4)
        rec, original_ids = recording_from_sequence(seq, 50, 3.0, max_windows=2)
        np.testing.assert_array_equal(original_ids, [1, 2, 3])
        np.testing.assert_array_equal(rec.t, [0, 55, 10])
        self.assertEqual(rec.n_windows, 2)

    def test_backbone_identity_is_checked_even_with_matching_score_scale(self):
        ck = {"manifest": {"prior_provenance": {"verified": True, "backbone_sha256": "abc"}}}
        self.assertTrue(verify_backbone_provenance(ck, "abc"))
        with self.assertRaisesRegex(ValueError, "different BIR"):
            verify_backbone_provenance(ck, "def")
        with self.assertWarnsRegex(RuntimeWarning, "Legacy"):
            self.assertFalse(verify_backbone_provenance({}, "abc"))


if __name__ == "__main__":
    unittest.main()
