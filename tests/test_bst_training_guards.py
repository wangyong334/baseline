"""Final-run provenance, VoW return validity and nonfinite-fit failures."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from model.bst import ContinuationValueHead
from tools.bst.fit_policy import regress
from tools.bst.policy import policy_returns
from tools.bst.train import require_complete_priors, validate_training_config


class FinalTrainingGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg = json.loads((Path(__file__).parents[1] / "configs/bst_shared.json").read_text())

    def test_both_final_seeds_are_valid(self):
        for seed in [37, 38]:
            cfg = copy.deepcopy(self.cfg)
            cfg["train"]["seed"] = seed
            validate_training_config(cfg)

    def test_invalid_empty_or_nonfinite_fits_rejected(self):
        for group, key, value in [("train", "epochs", 0), ("train", "seed", -1),
                                  ("train", "background_budget", -1), ("train", "lr", float("nan")),
                                  ("policy", "iterations", 0), ("policy", "wait_cost", -1),
                                  ("policy", "init_cost", float("inf"))]:
            cfg = copy.deepcopy(self.cfg)
            cfg[group][key] = value
            with self.assertRaises(ValueError):
                validate_training_config(cfg)

    def test_final_provenance_requires_complete_identity_and_split(self):
        good = {"verified": True, "complete": True, "backbone_sha256": "backbone"}
        require_complete_priors(good, {"probe": False})
        for key in good:
            bad = dict(good)
            bad[key] = False
            with self.assertRaises(ValueError):
                require_complete_priors(bad, {"probe": False})
        with self.assertRaises(ValueError):
            require_complete_priors(good, {"probe": True})

    def test_returns_reject_holes_nonfinite_cost_and_invalid_labels(self):
        z = np.zeros((1, 3), np.float32)
        valid = np.ones_like(z, bool)
        for mask, y, cost in [(np.array([[1, 0, 1]], bool), [1], .1),
                              (valid, [.3], .1), (valid, [1], float("nan")),
                              (np.zeros_like(valid), [1], .1)]:
            with self.assertRaises(ValueError):
                policy_returns(z, np.asarray(y), mask, valid, 0, 1, 1, cost)

    def test_regression_rejects_nonfinite_inputs(self):
        head = ContinuationValueHead(2, 3)
        pc = dict(self.cfg["policy"], fit_epochs=1, batch_size=4)
        for features, labels, weights in [(np.full((2, 2), np.nan), np.ones(2), np.ones(2)),
                                           (np.zeros((2, 2)), -np.ones(2), np.ones(2)),
                                           (np.zeros((2, 2)), np.ones(2), np.zeros(2))]:
            with self.assertRaises(ValueError):
                regress(head, features, labels, weights, pc, "cpu", np.random.default_rng(1))

    def test_large_finite_return_has_finite_initialization_and_fit(self):
        torch.manual_seed(2)
        head = ContinuationValueHead(2, 3)
        pc = dict(self.cfg["policy"], fit_epochs=1, batch_size=4)
        mse = regress(head, np.zeros((4, 2)), np.full(4, 1000.), np.ones(4), pc,
                      "cpu", np.random.default_rng(1))
        self.assertTrue(np.isfinite(mse))
        self.assertTrue(torch.isfinite(head.net[-1].bias).all())


if __name__ == "__main__":
    unittest.main()
