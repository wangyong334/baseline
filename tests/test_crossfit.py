"""交叉拟合（RCL v4.1，堆叠泛化）的测试（CPU、合成数据）。

    1. 分折：每条录像恰好一折；连续 block 条为一块、块轮流分折；参数不合法时报错
    2. train_file_names：cv_folds > 0 时去掉第 cv_fold 折；各折留出的录像合起来恰好是全部训练录像；不开时不变
    3. 全流程：两个折骨干 + 完整骨干（train_stream_v1 mode_train）-> tools/export_oof_outputs.py -> 判决层用样本外输出训练一轮。
       每条训练录像的样本外输出来自没见过它的折骨干，与该骨干直接推理的输出逐位相同；事件数不一致、缺录像时报错
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_stream_v1 as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FoldTests(unittest.TestCase):
    def test_partition(self):
        names = ["r%02d.npz" % i for i in range(11)]
        fold = T.crossfit_folds(list(reversed(names)), 3, 2)
        self.assertEqual(set(fold), set(names))
        self.assertEqual([fold[n] for n in names], [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2])
        for bad in ((1, 1), (3, 0)):
            with self.assertRaises(ValueError):
                T.crossfit_folds(names, *bad)

    def test_train_file_names(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "train").mkdir()
            for i in range(10):
                (Path(directory) / "train" / ("r%02d.npz" % i)).write_bytes(b"")
            every = T.train_file_names({"root": directory})
            self.assertEqual(len(every), 10)
            held = []
            for k in range(3):
                keep = T.train_file_names({"root": directory, "cv_folds": 3, "cv_fold": k, "cv_block": 2})
                held += sorted(set(every) - set(keep))
            self.assertEqual(sorted(held), every)                     # each recording held out exactly once


class PipelineTests(unittest.TestCase):
    def test_crossfit_pipeline(self):
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
            import yaml
        except ImportError as error:
            self.skipTest("完整训练入口需要 cv2/pandas/tqdm/yaml: %s" % error)
        import train_track_head as TH
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import export_oof_outputs as EXP
        from test_stream_layerwise import H as SH, Q99, SMALL, W as SW, synthetic_sequence
        from utils.stream_common import load_flat_config
        cpu = torch.device("cpu")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pos = neg = 0
            for split, seeds in (("train", (3, 4, 5, 6)), ("val", (8,)), ("test", (9,))):
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
            cfg = load_flat_config(os.path.join(ROOT, "configs", "evisseg_stream_tif_comm_peak.yaml"))
            cfg.update(root=str(root), stats_path=str(root / "stats.json"),
                       channels=list(SMALL), height=SH, width_px=SW, pad_height=SH, pad_width=SW,
                       n_windows=12, epochs=2, num_workers=0, tbptt_k=4, train_subset_size=1,
                       calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
                       segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=4,
                       execution="layer", input_device="gpu", streams=2)
            run = SimpleNamespace(device="cpu", mode="train", resume=False)
            folds = []
            for k in range(2):
                fcfg = dict(cfg, save_root=str(root / ("f%d" % k)), cv_folds=2, cv_fold=k, cv_block=1)
                T.mode_train(run, fcfg)
                folds.append((fcfg, str(root / ("f%d" % k) / "best_val_iou_seed37.pt")))
            T.mode_train(run, dict(cfg, save_root=str(root / "full")))
            head = load_flat_config(os.path.join(ROOT, "configs", "changepoint_v41.yaml"))
            head.update(root=str(root), base_checkpoint=str(root / "full" / "best_val_iou_seed37.pt"),
                        save_root=str(root / "head"), epochs=2, eval_chunk=4, dynamics_prefit_steps=5,
                        segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1,
                        train_outputs=str(root / "oof.pt"))
            head_path = root / "head.yaml"
            head_path.write_text(yaml.safe_dump({"ALL": head}), encoding="utf-8")
            EXP.main(["--config", str(head_path), "--checkpoints", folds[0][1], folds[1][1],
                      "--out", str(root / "oof.pt"), "--device", "cpu"])
            blob = torch.load(str(root / "oof.pt"))
            train_names = T.train_file_names({"root": str(root)})
            self.assertEqual(sorted(blob["outputs"]), train_names)
            for entry, (fcfg, ckpt) in zip(blob["manifest"]["folds"], folds):
                self.assertTrue(entry["held_out"])
                self.assertFalse(set(entry["held_out"]) & set(T.train_file_names(fcfg)))   # never trained on them
                net, tap, base, q99, _, _ = TH.load_base(dict(head, base_checkpoint=ckpt), cpu)
                name = entry["held_out"][0]
                ds = TH.EvUAVStream(base["root"], "train", base, names=[name])
                _, z0, feat = TH.BaseOutputs(net, tap, base, q99, cpu, ds).get(0)
                self.assertTrue(torch.equal(blob["outputs"][name][0], z0.float()))
                self.assertTrue(torch.equal(blob["outputs"][name][1], feat.float()))
            TH.main(["--config", str(head_path), "--mode", "train", "--device", "cpu"])
            manifest = json.loads((root / "head" / "train_outputs_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["n_recordings"], len(train_names))
            self.assertTrue((root / "head" / "best_val_iou_seed37.pt").exists())
            # inconsistent or incomplete cross-fitted outputs are refused
            net, tap, base, q99, _, _ = TH.load_base(head, cpu)
            outputs = TH.BaseOutputs(net, tap, base, q99, cpu, TH.split_dataset(base, "train"))
            bad = {"outputs": dict(blob["outputs"]), "manifest": {}}
            z0, feat = bad["outputs"][train_names[0]]
            bad["outputs"][train_names[0]] = (z0[:-1], feat[:-1])
            torch.save(bad, str(root / "bad.pt"))
            with self.assertRaises(RuntimeError):
                TH.load_crossfit_outputs(outputs, str(root / "bad.pt"))
            del bad["outputs"][train_names[0]]
            torch.save(bad, str(root / "bad.pt"))
            with self.assertRaises(RuntimeError):
                TH.load_crossfit_outputs(outputs, str(root / "bad.pt"))


if __name__ == "__main__":
    unittest.main()
