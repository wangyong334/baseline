"""V5 @ 25 ms（configs/evisseg_stream_tif_comm_peak_w25.yaml，v6-3）的单元测试（CPU）。

    1. 配置：与 50 ms 定型版只差按窗计的量，换算成物理量后相同（窗长 x 窗数、片段时长、每秒消息预算、分段时刻）
    2. 端到端：合成数据按 25 ms 切窗，完整训练（增益校准、多流 TBPTT、消息约束、验证）+ 正式评估都能跑通，
       评估的计时按实际窗长折算 8 s
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_stream_layerwise import H, Q99, SMALL, W, WINDOWS, synthetic_sequence  # noqa: E402
from utils.stream_common import load_flat_config  # noqa: E402

CLOCK_KEYS = {"window_ms", "n_windows", "tbptt_k", "eval_chunk", "calib_windows", "comm_budget_messages",
              "rolling_window", "segments", "timing_warmup_windows", "save_root", "stats_path"}


class W25ConfigTests(unittest.TestCase):
    def test_same_physics_as_50ms(self):
        a = load_flat_config("configs/evisseg_stream_tif_comm_peak.yaml")
        b = load_flat_config("configs/evisseg_stream_tif_comm_peak_w25.yaml")
        self.assertEqual(set(a), set(b))
        self.assertEqual({k for k in a if a[k] != b[k]}, CLOCK_KEYS)
        r = a["window_ms"] // b["window_ms"]
        self.assertEqual((a["window_ms"], b["window_ms"], r), (50, 25, 2))
        self.assertEqual(a["window_ms"] * a["n_windows"], b["window_ms"] * b["n_windows"])           # 8 s
        for key in ("tbptt_k", "eval_chunk", "calib_windows", "rolling_window", "timing_warmup_windows"):
            self.assertEqual(a[key] * r, b[key], key)                                                # 时长不变
        self.assertEqual(a["comm_budget_messages"] / r, b["comm_budget_messages"])                    # 每秒预算不变
        self.assertEqual([[x * r, y * r + r - 1] for x, y in a["segments"]], b["segments"])           # 同一时段
        self.assertEqual(b["stats_path"], "configs/stream_v1_train_stats_w25.json")
        self.assertEqual(b["save_root"], "log/stream_tif_commpeak_msg_w25_seed37")
        for key in ("tau_init_ms", "tau_min_ms", "tau_max_ms", "pd_detT", "threshold", "time_bins", "epochs",
                    "streams", "accumulate", "lr", "lr_end", "tif"):
            self.assertEqual(a[key], b[key], key)


class W25EndToEndTests(unittest.TestCase):
    def test_train_and_eval_at_25ms(self):
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
        except ImportError as error:
            self.skipTest("完整入口需要原 V1 的 cv2/pandas/tqdm 依赖: %s" % error)
        import train_stream_v1 as T
        from dataset.ev_uav_stream import EvUAVStream
        windows = 2 * WINDOWS                                    # 合成序列 600 ms：50 ms 时 12 窗，25 ms 时 24 窗
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pos = neg = 0
            for split, seeds in (("train", (3, 4, 5)), ("val", (6, 7))):
                (root / split).mkdir()
                for seed in seeds:
                    seq = synthetic_sequence(seed=seed)
                    loc = np.stack((seq.x, seq.y, seq.t), axis=1)
                    norm = np.column_stack((loc, seq.p, seq.label, seq.target_id))
                    np.savez(root / split / ("seq_%d.npz" % seed), ev_loc=loc, evs_norm=norm)
                    if split == "train":
                        pos += int(seq.label.sum())
                        neg += seq.n_events - int(seq.label.sum())
            stats = dict(q99=Q99.tolist(), train_pos=pos, train_neg=neg, pos_weight=min(neg / pos, 30.0))
            (root / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
            run = root / "w25"
            cfg = load_flat_config("configs/evisseg_stream_tif_comm_peak_w25.yaml")
            cfg.update(root=str(root), stats_path=str(root / "stats.json"), save_root=str(run),
                       channels=list(SMALL), height=H, width_px=W, pad_height=H, pad_width=W,
                       n_windows=windows, epochs=2, num_workers=0, tbptt_k=8, train_subset_size=1,
                       calib_sequences=1, calib_windows=16, calib_samples_per_channel=32, calib_min_positive=4,
                       segments=[[0, 7], [8, 15], [16, 23]], timing_warmup_windows=2, eval_chunk=8,
                       execution="layer", input_device="gpu", streams=2)
            seq = EvUAVStream(str(root), "val", cfg)[0]
            self.assertEqual(seq.n_windows, windows)
            self.assertTrue(np.all(seq.t[seq.window_index(5)] // 25 == 5))
            T.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
            checkpoint = run / "best_val_iou_seed37.pt"
            self.assertTrue(checkpoint.exists())
            args = SimpleNamespace(checkpoint=str(checkpoint), device="cpu", mode="eval", split="val", overwrite=True,
                                   execution=None, input_device=None, readout_ablation="none",
                                   subthreshold_ablation="none", memory_ablation="none", memory_off_layers=None,
                                   reset_every=0, reset_offset=0, dump_dir=None)
            T.mode_eval(args, cfg)
            result = json.loads((run / "eval_val_best_val_iou_seed37.json").read_text(encoding="utf-8"))
            carry = result["carry"]
            self.assertTrue(np.isfinite(carry["iou"]))
            timing = carry["timing"]
            self.assertAlmostEqual(timing["ms_per_8s"], timing["total_ms_mean"] * 8000.0 / 25.0, places=6)


if __name__ == "__main__":
    unittest.main()
