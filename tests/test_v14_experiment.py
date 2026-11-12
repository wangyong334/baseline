"""Protocol checks and a real two-epoch B4/K16 BIR train/eval smoke test."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("v14_runner", ROOT / "tools/v14/run_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ProtocolTests(unittest.TestCase):
    def test_only_state_and_output_differ(self):
        pair = runner.validate_pair()
        a, b = (runner.flatten(pair[m]) for m in ("carry", "reset"))
        self.assertEqual({k for k in a if a[k] != b[k]}, {"state_mode", "save_root"})
        self.assertEqual((a["trainer"], a["streams"], a["tbptt_k"], a["accumulate"],
                          a["execution"], a["input_device"], a["stream_layout"]),
                         ("streams", 4, 16, 1, "layer", "gpu", "greedy"))
        self.assertFalse(any(k.startswith("comm_") for k in a))

    def test_reject_mismatched_seed_or_state_checkpoint(self):
        expected = runner.validate_pair()["reset"]
        actual = runner.flatten(expected)
        runner.validate_checkpoint_config(actual, expected)
        for key, value in (("state_mode", "carry"), ("seed", 38), ("comm_budget_messages", 100)):
            changed = dict(actual, **{key: value})
            with self.assertRaises(RuntimeError):
                runner.validate_checkpoint_config(changed, expected)

    def test_extract_trained_mode_not_default_carry(self):
        result = {"trained_state_mode": "reset_each_window", "seed": 37,
                  "carry": {"iou": 0.01},
                  "reset_each_window": {"iou": 0.8, "acc": 0.9, "pd": 0.7, "fa": 1e-5}}
        self.assertEqual(runner.trained_metrics(result, "reset_each_window", 37)["iou"], 0.8)
        with self.assertRaises(RuntimeError):
            runner.trained_metrics(result, "carry", 37)

    def test_paths_reject_release_traversal(self):
        with self.assertRaises(ValueError):
            runner.runtime_config("carry", 37, "../old", ROOT, ROOT)


class IndependentTrainingTests(unittest.TestCase):
    def test_both_arms_train_b4k16_then_evaluate_their_own_mode(self):
        import numpy as np
        import torch
        import yaml
        import train_stream_v1 as trainer
        sys.path.insert(0, str(ROOT / "tests"))
        from test_stream_layerwise import H, W, SMALL, Q99, synthetic_sequence

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(temporary)
            data = directory / "data"
            pos = neg = 0
            for split, seeds in (("train", (3, 4, 5, 6)), ("val", (7,)), ("test", (8,))):
                (data / split).mkdir(parents=True)
                for seed in seeds:
                    seq = synthetic_sequence(seed=seed)
                    loc = np.stack((seq.x, seq.y, seq.t), axis=1)
                    norm = np.column_stack((loc, seq.p, seq.label, seq.target_id))
                    np.savez(data / split / ("seq_%d.npz" % seed), ev_loc=loc, evs_norm=norm)
                    if split == "train":
                        pos += int(seq.label.sum())
                        neg += seq.n_events - int(seq.label.sum())
            stats = dict(q99=Q99.tolist(), train_pos=pos, train_neg=neg, pos_weight=min(neg / pos, 30.0))
            runner.write_json(directory / "stats.json", stats)
            weights = []
            states = []
            for mode, raw in runner.validate_pair().items():
                output = directory / mode
                cfg = runner.flatten(copy.deepcopy(raw))
                cfg.update(root=str(data), stats_path=str(directory / "stats.json"), save_root=str(output),
                           channels=list(SMALL), height=H, width_px=W, pad_height=H, pad_width=W,
                           n_windows=20, epochs=2, num_workers=0, train_subset_size=1,
                           calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
                           segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=16)
                self.assertEqual((cfg["streams"], cfg["tbptt_k"]), (4, 16))
                trainer.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
                records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), 2)
                for record in records:
                    self.assertEqual((record["trainer"], record["streams"], record["tbptt_k"]),
                                     ("streams", 4, 16))
                    self.assertGreater(record["updates"], 0)
                    self.assertIsNone(record["comm"])
                path = directory / (mode + "_runtime.yaml")
                path.write_text(yaml.safe_dump({"TEST": cfg}), encoding="utf-8")
                checkpoint = output / "best_val_iou_seed37.pt"
                weights.append(runner.sha256(checkpoint))
                try:
                    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
                except TypeError:
                    payload = torch.load(str(checkpoint), map_location="cpu")
                states.append(payload["model"])
                evaluated = runner.evaluate_trained_mode(path, checkpoint, "cpu")
                for split, evaluation in evaluated.items():
                    self.assertEqual(evaluation["trained_state_mode"], runner.MODES[mode])
                    self.assertNotIn("carry" if mode == "reset" else "reset_each_window", evaluation)
                    runner.trained_metrics(evaluation, runner.MODES[mode], 37)
                args = SimpleNamespace(checkpoint=str(checkpoint), device="cpu", mode="eval", split="val",
                                       overwrite=False, execution=None, input_device=None, readout_ablation="none",
                                       subthreshold_ablation="none", memory_ablation="none", memory_off_layers=None,
                                       reset_every=0, reset_offset=0, dump_dir=None)
                trainer.mode_eval(args, cfg)
                original = runner.read_json(output / "eval_val_best_val_iou_seed37.json")[runner.MODES[mode]]
                got = evaluated["val"][runner.MODES[mode]]
                for key in ("iou", "acc", "pd", "fa"):
                    self.assertEqual(got[key], original[key], (mode, key))
            self.assertNotEqual(weights[0], weights[1])
            self.assertTrue(any(not torch.equal(states[0][key], states[1][key]) for key in states[0]))


if __name__ == "__main__":
    unittest.main()
