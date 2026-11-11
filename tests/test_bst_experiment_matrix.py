"""Backbone transfer plans cannot silently share priors, seeds or test winners."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.bst.experiment_matrix import build_plan, inspect_backbone, variants, write_plan


class ExperimentMatrixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root/"data"
        for split in ("train", "val", "test"):
            (self.data/split).mkdir(parents=True)
            (self.data/split/"clip.npz").write_bytes(b"raw membership placeholder")
        self.config = {"recording_n_windows": 160,
            "model": {"source_features": 2, "window_ms": 50, "normalization": "window", "window_gate": True,
                      "bounded_values": True, "reweight": True, "revalue": True}, "train": {"seed": 37}}
        self.spec = {"backbones": [{"id": "bir", "architecture_family": "EvSpSegNetStream",
            "adapter": "evspsegnet_stream", "neuron": "tif", "checkpoints": {"37": "bir37", "38": "bir38"}},
            {"id": "lif", "architecture_family": "EvSpSegNetStream", "adapter": "evspsegnet_stream",
             "neuron": "lif", "checkpoints": {"37": "lif37", "38": "lif38"}}]}

    def tearDown(self):
        self.temp.cleanup()

    def inspect(self, path, seed, spec, cfg):
        if path is None:
            raise ValueError("checkpoint unavailable")
        return {"checkpoint": path, "sha256": (path+"_")*13, "seed": seed,
                "architecture_family": "EvSpSegNetStream", "neuron": spec["neuron"]}

    def plan(self, spec=None, inspector=None):
        return build_plan(spec or self.spec, self.config, self.root/"plan", self.data,
                          inspector=inspector or self.inspect)

    def test_retrained_cases_isolate_cache_identity_and_do_not_claim_distinct_architectures(self):
        plan = self.plan()
        self.assertTrue(plan["all_backbones_ready"])
        self.assertEqual(len(plan["experiments"]), 28)
        self.assertEqual(len(plan["prior_export_commands"]), 12)
        self.assertFalse(plan["distinct_architecture_transfer_ready"])
        self.assertEqual(plan["ready_architecture_families"], ["EvSpSegNetStream"])
        self.assertEqual(len({case["priors"] for case in plan["experiments"]}), 4)
        self.assertEqual(len({case["out"] for case in plan["experiments"]}), 28)
        for case in plan["experiments"]:
            self.assertEqual(case["config"]["train"]["seed"], case["seed"])
            self.assertTrue(case["retrained_sdu"] and case["refitted_vow"])
            self.assertFalse(case["test_selects_variant"])
            self.assertIn("train.py", case["commands"][0])
            self.assertIn("--require-complete-priors", case["commands"][0])
            self.assertIn("fit_policy.py", case["commands"][1])
            arrival = [line for line in case["commands"] if "evaluate.py" in line and "--commit arrival" in line]
            self.assertEqual(len(arrival), 2)
            for split in ("val", "test"):
                matching = [line for line in arrival if "--split " + split in line]
                self.assertEqual(len(matching), 1)
                self.assertIn("--birth band", matching[0])
                self.assertIn(split+"_band_arrival", matching[0])
        self.assertEqual(self.config["train"]["seed"], 37)

    def test_missing_seed_and_reused_checkpoint_never_generate_runnable_cases(self):
        spec = copy.deepcopy(self.spec)
        spec["backbones"][0]["checkpoints"]["38"] = None
        plan = self.plan(spec)
        blocked = [case for case in plan["experiments"] if case["backbone"] == "bir" and case["seed"] == 38]
        self.assertFalse(plan["all_backbones_ready"])
        self.assertTrue(all(not case["ready"] and not case["commands"] for case in blocked))
        def repeated(path, seed, backbone, cfg):
            result = self.inspect(path, seed, backbone, cfg)
            result["sha256"] = "same_weights"
            return result
        plan = self.plan(inspector=repeated)
        self.assertEqual(sum(entry["ready"] for entry in plan["backbones"]), 1)
        self.assertTrue(any("reused" in error for entry in plan["backbones"] for error in entry["errors"]))

    def test_ablation_patches_are_explicit_and_full_config_is_untouched(self):
        before = copy.deepcopy(self.config)
        options = variants(self.config)
        self.assertEqual(len(options), 7)
        self.assertEqual(self.config, before)
        self.assertFalse(options["no_window_gate"]["model"]["window_gate"])
        self.assertFalse(options["unbounded_values"]["model"]["bounded_values"])
        self.assertFalse(options["no_reaudit"]["model"]["reweight"])
        self.assertFalse(options["no_reaudit"]["model"]["revalue"])
        self.assertEqual(options["legacy_flat_mean"]["model"]["normalization"], "mean")
        self.assertFalse(options["legacy_flat_mean"]["model"]["bounded_values"])

    def test_publishing_plan_writes_only_plan_configs_and_preserves_existing(self):
        plan = self.plan()
        write_plan(plan)
        root = self.root/"plan"
        self.assertEqual(len(list((root/"configs").glob("*.json"))), 28)
        self.assertFalse((root/"runs").exists())
        self.assertFalse((root/"priors").exists())
        saved = json.loads((root/"matrix.json").read_text())
        self.assertEqual(saved["mode"], "plan_only_no_training_or_queue_submission")
        with self.assertRaises(FileExistsError):
            write_plan(plan)

    def test_actual_neuron_variants_share_event_output_contract_but_seed_is_checked(self):
        import numpy as np
        import torch
        import train_stream_v1 as T
        from tools.bst.data import Recording
        from tools.bst.infer import BackboneArrivalProvider
        torch.set_num_threads(1)
        cfg = {"seed": 37, "time_bins": 1, "channels": [2, 4, 4, 4], "norm": "none",
               "state_mode": "carry", "readout_hidden": 4, "window_ms": 50,
               "tau_init_ms": 200, "tau_min_ms": 50, "tau_max_ms": 2000, "v_threshold": 1,
               "n_windows": 160, "height": 8, "width_px": 8, "pad_height": 8, "pad_width": 8}
        class Source:
            def window(self, index):
                events = {"b": torch.zeros(1, dtype=torch.long), "x": torch.tensor([1]),
                          "y": torch.tensor([2]), "p": torch.tensor([1.]), "t_local": torch.tensor([.2])}
                return torch.ones(1, 4, 8, 8), events, None, np.array([index])
        rec = Recording("contract", [1, 1], [2, 2], [10, 60], [1, 1], [0, 0], [0, 0], 50, n_windows=2)
        for neuron in ("lif", "graded", "relu", "tif"):
            cfg["neuron"] = neuron
            net = T.build_net(cfg, "cpu").eval()
            net.gain_calibrated.fill_(True)
            path = self.root/(neuron+".pt")
            torch.save({"config": cfg, "stats": {"pos_weight": 2., "q99": np.ones(4)},
                        "model": net.state_dict()}, str(path))
            spec = {"adapter": "evspsegnet_stream", "architecture_family": "EvSpSegNetStream", "neuron": neuron}
            details = inspect_backbone(path, 37, spec, self.config)
            self.assertEqual(details["neuron"], neuron)
            provider = BackboneArrivalProvider(net, Source(), rec, [0, 1], 2, math.log(2.))
            for window in range(2):
                value = provider(window, np.array([window]))
                self.assertEqual(value["source_features"].shape, (1, 2))
                self.assertTrue(np.isfinite(value["prior"]).all())
            with self.assertRaisesRegex(ValueError, "seed"):
                inspect_backbone(path, 38, spec, self.config)


if __name__ == "__main__":
    unittest.main()
