"""全局端到端训练（train_end2end.py）的测试（CPU、合成数据）。

    1. 端到端：训练一个小骨干 -> 骨干与目标神经元层一起训练 2 轮 -> test 评估；四个指标有限，零等待 / 终判 / 骨干读出都有，
       延迟决策的统计、证据系数随结果存盘；骨干权重确实被更新（梯度到达骨干）
    2. 多流拼批：每个事件的 target_id 与录像对得上；每个事件的背景率取自模块二的背景图（所在窗、流、极性、像素）
    3. 配置：full_model_v2 与 object_layer_v2 的目标层设置相同（只差训练方式）；object_layer_v2 与 v1 只差两项 v2 机制
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ConfigTests(unittest.TestCase):
    def test_v2_configs(self):
        from utils.stream_common import load_flat_config
        full = load_flat_config(os.path.join(ROOT, "configs", "full_model_v2.yaml"))
        frozen = load_flat_config(os.path.join(ROOT, "configs", "object_layer_v2.yaml"))
        v1 = load_flat_config(os.path.join(ROOT, "configs", "object_layer_v1.yaml"))
        from model.object_layer import OBJECT_KEYS
        for key in OBJECT_KEYS + ("head_kind",):
            self.assertEqual(full.get(key), frozen.get(key), key)
        self.assertEqual(full["clutter_source"], "background")
        self.assertTrue(full["continuous_time"])
        diff = {k for k in set(frozen) | set(v1) if frozen.get(k) != v1.get(k)}
        self.assertEqual(diff, {"clutter_source", "continuous_time", "save_root"})


class BackgroundLookupTests(unittest.TestCase):
    def test_event_background_indexing(self):
        import train_track_head as TH
        bg = torch.zeros(3, 2, 2, 4, 5)
        bg[1, 0, 0, 2, 3] = 0.5                                       # window 1, stream 0, ON, y 2, x 3
        bg[2, 1, 1, 0, 4] = 0.25                                      # window 2, stream 1, OFF, y 0, x 4
        events = {"t": torch.tensor([1, 2]), "b": torch.tensor([0, 1]), "y": torch.tensor([2, 0]),
                  "x": torch.tensor([3, 4]), "p": torch.tensor([1.0, 0.0]), "bg": bg}
        self.assertEqual(TH.event_background(events).tolist(), [0.5, 0.25])


class EndToEndTests(unittest.TestCase):
    def test_train_and_eval(self):
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
        except ImportError as error:
            self.skipTest("完整训练入口需要 cv2/pandas/tqdm: %s" % error)
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        import yaml
        import train_end2end as E2E
        import train_stream_v1 as T
        from test_stream_layerwise import H as SH, Q99, SMALL, W as SW, synthetic_sequence
        from utils.stream_common import load_flat_config
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pos = neg = 0
            for split, seeds in (("train", (3, 4, 5)), ("val", (6,)), ("test", (7,))):
                (root / split).mkdir()
                for seed in seeds:
                    seq = synthetic_sequence(seed=seed, n=3000)
                    loc = np.stack((seq.x, seq.y, seq.t), axis=1)
                    norm = np.column_stack((loc, seq.p, seq.label, seq.target_id))
                    np.savez(root / split / ("seq_%d.npz" % seed), ev_loc=loc, evs_norm=norm)
                    if split == "train":
                        pos += int(seq.label.sum())
                        neg += seq.n_events - int(seq.label.sum())
            stats = dict(q99=Q99.tolist(), train_pos=pos, train_neg=neg, pos_weight=min(neg / pos, 30.0))
            (root / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
            base_dir = root / "base"
            cfg = load_flat_config(os.path.join(ROOT, "configs", "evisseg_stream_tif_comm_peak_invariant.yaml"))
            cfg.update(root=str(root), stats_path=str(root / "stats.json"), save_root=str(base_dir),
                       channels=list(SMALL), height=SH, width_px=SW, pad_height=SH, pad_width=SW,
                       n_windows=12, epochs=2, num_workers=0, tbptt_k=4, train_subset_size=1,
                       calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
                       segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=4,
                       execution="layer", input_device="gpu", streams=2)
            T.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
            base_ckpt = str(base_dir / "best_val_iou_seed37.pt")
            e2e = load_flat_config(os.path.join(ROOT, "configs", "full_model_v2.yaml"))
            e2e.update(root=str(root), base_checkpoint=base_ckpt, save_root=str(root / "e2e"), epochs=2, eval_chunk=4,
                       slots=4, segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1)
            cfg_path = root / "full_model_v2.yaml"
            cfg_path.write_text(yaml.safe_dump({"ALL": e2e}), encoding="utf-8")
            E2E.main(["--config", str(cfg_path), "--mode", "train", "--device", "cpu"])
            ckpt_path = root / "e2e" / "best_val_iou_seed37.pt"
            self.assertTrue(ckpt_path.exists())
            E2E.main(["--config", str(cfg_path), "--mode", "eval", "--device", "cpu", "--split", "test",
                      "--checkpoint", str(ckpt_path), "--dump-dir", str(root / "dump"),
                      "--tracks-dir", str(root / "tracks")])
            result = json.loads((root / "e2e" / "eval_test_best_val_iou_seed37.json").read_text(encoding="utf-8"))
            for kind in ("base", "track_now", "track"):
                for key in ("iou", "acc", "pd", "fa"):
                    self.assertTrue(np.isfinite(result[kind][key]), (kind, key))
            self.assertEqual(result["clutter_source"], "background")
            self.assertTrue(result["continuous_time"])
            self.assertEqual(set(result["head_stats"]["status"]), {"confirmed", "rejected", "reacquired", "lost", "forced"})
            self.assertTrue(result["evidence"]["llr_scale"] > 0.0)
            before = torch.load(base_ckpt, map_location="cpu")["model"]
            after = torch.load(str(ckpt_path), map_location="cpu")["model"]
            changed = [k for k in before if before[k].dtype.is_floating_point and not torch.equal(before[k], after[k])]
            self.assertTrue(changed)                                    # gradients reached the backbone
            with np.load(str(root / "dump" / "seq_7.npz")) as dump:
                self.assertEqual(dump["probabilities"].shape, dump["base_probabilities"].shape)
                self.assertIn("decision_wait", dump.files)
            lines = (root / "e2e" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            rec = json.loads(lines[-1])
            self.assertIsNotNone(rec["train"]["pix_loss"])
            self.assertIsNotNone(rec["train"]["now_loss"])


if __name__ == "__main__":
    unittest.main()
