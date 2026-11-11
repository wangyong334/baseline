"""Training benchmark isolation, gradient equality and inclusive cache timing."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import model.bst as model_code
from tools.bst import pipeline
from tools.bst.data import NeighborIndex, Recording
from tools.bst.profile_training import compare_steps, load_reference, run_sequence


class TrainingProfileTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)
        rng = np.random.default_rng(17)
        n = 36
        self.rec = Recording("synthetic", rng.integers(0, 8, n), rng.integers(0, 8, n),
            np.arange(n)*8, np.where(np.arange(n)%2, 1., -1.), rng.normal(size=n),
            (np.arange(n)%3 == 0).astype(float), 50, n_windows=6)
        self.config = {"model": {"source_dim": 8, "state_dim": 6, "heads": 2,
                       "relation_dim": 3, "max_wait_steps": 2, "past_windows": 2},
            "evidence": {"past_windows": 2, "max_wait": 2, "k_near": 2, "k_evidence": 1,
                         "filter_logit": -1., "radius": 0},
            "train": {"batch_size": 6, "lr": .001, "weight_decay": .0001, "grad_clip": 1.}}
        initial = model_code.SharedSequentialModel(model_code.BSTConfig(**self.config["model"]))
        with torch.no_grad():
            initial.value_weight.normal_(0, .2)
            initial.value_bias.fill_(.1)
        self.initial = copy.deepcopy(initial.state_dict())
        self.ids = rng.permutation(n)[:17]
        self.weights = np.where(self.rec.label[self.ids] > .5, 1., 2.)

    def run_path(self, geometry):
        model = model_code.SharedSequentialModel(model_code.BSTConfig(**self.config["model"]))
        model.load_state_dict(self.initial)
        index = NeighborIndex(self.rec, **self.config["evidence"])
        return run_sequence(model, self.rec, index, pipeline, self.ids, self.weights,
                            self.config, "cpu", geometry=geometry, capture=True)

    def test_full_optimizer_sequences_match_and_cache_build_is_included(self):
        original, a = self.run_path(False)
        cached, b = self.run_path(True)
        result = compare_steps(b, a)
        self.assertTrue(result["equivalent"], result)
        self.assertEqual(original["batches"], 3)
        self.assertEqual(cached["queries"], 17)
        self.assertEqual(a[-1]["logits"].shape[0], 5)
        self.assertEqual(cached["query_calls"], 1)
        self.assertGreater(cached["geometry_stats"]["query_build_seconds"], 0)
        self.assertGreater(cached["geometry_materialize_seconds"], 0)
        self.assertGreaterEqual(cached["wall_seconds"], cached["query_seconds"])
        self.assertGreaterEqual(cached["fetch_seconds"], cached["query_seconds"])
        self.assertGreater(cached["geometry_stats"]["peak_id_cache_bytes"], 0)

    def test_comparison_rejects_optimizer_or_gradient_semantic_changes(self):
        _, reference = self.run_path(False)
        modified = copy.deepcopy(reference)
        modified[0]["gradients"][0] += .01
        self.assertFalse(compare_steps(modified, reference)["equivalent"])
        modified = copy.deepcopy(reference)
        modified[-1]["parameters"][0] += .01
        self.assertFalse(compare_steps(modified, reference)["equivalent"])
        modified = copy.deepcopy(reference)
        modified[0]["valid"][0, 0] = ~modified[0]["valid"][0, 0]
        self.assertFalse(compare_steps(modified, reference)["equivalent"])

    def test_reference_loader_rebinds_archived_dependencies_not_current_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"model").mkdir(); (root/"tools/bst").mkdir(parents=True)
            (root/"model/bst.py").write_text("class BSTConfig: pass\nclass SharedSequentialModel: pass\n")
            (root/"tools/bst/data.py").write_text(
                "class NeighborIndex: pass\ndef compact_neighbors(x): return 'archived'\n"
                "def load_recording(*a): return 'archived'\ndef sample_queries(*a): return 'archived'\n")
            (root/"tools/bst/pipeline.py").write_text(
                "from model.bst import BSTConfig, SharedSequentialModel\n"
                "from tools.bst.data import NeighborIndex, compact_neighbors, load_recording, sample_queries\n")
            reference = load_reference(root)
            self.assertIs(reference.pipeline.NeighborIndex, reference.data.NeighborIndex)
            self.assertIs(reference.pipeline.BSTConfig, reference.model.BSTConfig)
            self.assertIs(reference.pipeline.SharedSequentialModel, reference.model.SharedSequentialModel)
            self.assertEqual(reference.pipeline.compact_neighbors(None), "archived")
            self.assertEqual(reference.pipeline.load_recording(None), "archived")
            self.assertEqual(reference.pipeline.sample_queries(None), "archived")
            self.assertEqual(len(reference.source_sha256), 3)


if __name__ == "__main__":
    unittest.main()
