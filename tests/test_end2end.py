"""全局端到端训练（train_end2end.py）的测试（CPU、合成数据）。

    1. 端到端：训练一个小骨干 -> 骨干与目标神经元层一起训练 2 轮 -> test 评估；四个指标有限，零等待 / 终判 / 骨干读出都有，
       延迟决策的统计、证据系数随结果存盘；骨干权重确实被更新（梯度到达骨干）
    2. 多流拼批：每个事件的 target_id 与录像对得上；每个事件的背景率取自模块二的背景图（所在窗、流、极性、像素）
    3. 配置：full_model_v2 与 object_layer_v2 的目标层设置相同（只差训练方式）；object_layer_v2 与 v1 只差两项 v2 机制
    4. SPRINT 整体模型：从头（按骨干配置新建、增益校准）+ 学出来的杂波场 + 目标层一起训练 2 轮并评估；杂波场的损失、
       衰减系数、权重随结果存盘；选择性分离：归属损失到达骨干输出、存在损失不到达；v5（手写滤波程序）与 v6（目标 token 层）各跑一遍
    5. 配置：sprint_tokens 与 sprint_full 只差模块三的设置与输出目录；sprint_tokens 与冻结版 object_tokens_v1 的目标层设置相同
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
        for name, key, value in (("noctx", "object_context", False), ("fixed", "decision_rule", "fixed")):
            ctrl = load_flat_config(os.path.join(ROOT, "configs", "object_layer_v2_%s.yaml" % name))
            diff = {k for k in set(frozen) | set(ctrl) if frozen.get(k) != ctrl.get(k)}
            self.assertEqual(diff, {key, "save_root"}, name)            # controls differ from v2 in one switch only
            self.assertEqual(ctrl[key], value)


    def test_tokens_configs(self):
        from utils.stream_common import load_flat_config
        from model.object_tokens import TOKEN_KEYS
        tokens = load_flat_config(os.path.join(ROOT, "configs", "sprint_tokens.yaml"))
        frozen = load_flat_config(os.path.join(ROOT, "configs", "object_tokens_v1.yaml"))
        v5 = load_flat_config(os.path.join(ROOT, "configs", "sprint_full.yaml"))
        for key in TOKEN_KEYS + ("head_kind",):
            self.assertEqual(tokens.get(key), frozen.get(key), key)
        self.assertEqual(tokens["head_kind"], "tokens")
        from model.object_layer import OBJECT_KEYS
        head_keys = set(TOKEN_KEYS) | set(OBJECT_KEYS) | {"head_kind"}
        diff = {k for k in set(tokens) | set(v5) if tokens.get(k) != v5.get(k)}
        self.assertTrue(diff <= head_keys | {"save_root"}, diff - head_keys)   # only module 3 differs


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


class ScratchSprintTests(unittest.TestCase):
    """SPRINT 整体模型：从头端到端（骨干按配置新建 + 增益校准 + 学出来的杂波场 + 目标神经元层，选择性分离）。"""

    def test_scratch_with_learned_field(self):
        for name in ("sprint_full", "sprint_tokens"):
            with self.subTest(config=name):
                self._scratch_run(name)

    def _scratch_run(self, name):
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
        except ImportError as error:
            self.skipTest("完整训练入口需要 cv2/pandas/tqdm: %s" % error)
        import contextlib
        import io as _io
        import json
        import tempfile
        from pathlib import Path
        import yaml
        import train_end2end as E2E
        from test_stream_layerwise import H as SH, Q99, SMALL, W as SW, synthetic_sequence
        from utils.stream_common import load_flat_config
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(_io.StringIO()):
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
            bb = load_flat_config(os.path.join(ROOT, "configs", "evisseg_stream_tif_comm_peak_invariant.yaml"))
            bb.update(root=str(root), stats_path=str(root / "stats.json"), channels=list(SMALL), height=SH, width_px=SW,
                      pad_height=SH, pad_width=SW, n_windows=12, num_workers=0, tbptt_k=4, train_subset_size=1,
                      calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
                      segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=4, input_device="gpu",
                      streams=2)
            bb_path = root / "backbone.yaml"
            bb_path.write_text(yaml.safe_dump({"ALL": bb}), encoding="utf-8")
            full = load_flat_config(os.path.join(ROOT, "configs", name + ".yaml"))
            full.update(root=str(root), backbone_config=str(bb_path), save_root=str(root / "sprint"), epochs=2,
                        eval_chunk=4, slots=4, segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1)
            cfg_path = root / (name + ".yaml")
            cfg_path.write_text(yaml.safe_dump({"ALL": full}), encoding="utf-8")
            E2E.main(["--config", str(cfg_path), "--mode", "train", "--device", "cpu"])
            ckpt_path = root / "sprint" / "best_val_iou_seed37.pt"
            self.assertTrue((root / "sprint" / "calibration.json").exists())   # gains calibrated before training
            E2E.main(["--config", str(cfg_path), "--mode", "eval", "--device", "cpu", "--split", "test",
                      "--checkpoint", str(ckpt_path), "--dump-dir", str(root / "dump")])
            result = json.loads((root / "sprint" / "eval_test_best_val_iou_seed37.json").read_text(encoding="utf-8"))
            for kind in ("base", "track_now", "track"):
                for key in ("iou", "acc", "pd", "fa"):
                    self.assertTrue(np.isfinite(result[kind][key]), (kind, key))
            self.assertEqual(result["clutter_estimator"], "learned")
            self.assertEqual(len(result["clutter_decays"]), 3)
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            self.assertIsNotNone(ckpt["field"])
            self.assertEqual(ckpt["base_config"]["clutter_estimator"], "learned")
            rec = json.loads((root / "sprint" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            self.assertIsNotNone(rec["train"]["clutter_loss"])
            self.assertIsNotNone(rec["train"]["mem_loss"])

    def test_selective_separation_gradients(self):
        """选择性分离：归属损失的梯度到达骨干的输出（z0、特征），存在损失的不到达。"""
        from model.object_layer import ObjectLayer
        from test_track_head import FEAT, H as TH_H, W as TH_W, blob_windows
        torch.manual_seed(0)
        layer = ObjectLayer(FEAT, TH_H, TH_W, slots=6, hidden=16, attn_hidden=16, separation="selective")
        layer.init_rate(7.0)
        with torch.no_grad():
            layer.slot_query.weight.normal_(0.0, 0.3)
        events, z0, feat, lab, tid = blob_windows(6, noise=8)
        z0 = torch.where(lab > 0.5, torch.full_like(z0, 1.0), z0).requires_grad_(True)
        feat = feat.clone().requires_grad_(True)
        _, _, aux = layer.forward_chunk(events, z0, feat, None, 6, labels=lab, tid=tid, keys=lab)
        g_mem = torch.autograd.grad(aux["mem_sum"], [z0, feat], retain_graph=True, allow_unused=True)
        g_exist = torch.autograd.grad(aux["exist_sum"], [z0, feat], retain_graph=True, allow_unused=True)
        self.assertTrue(any(g is not None and float(g.abs().sum()) > 0.0 for g in g_mem))
        self.assertTrue(all(g is None or float(g.abs().sum()) == 0.0 for g in g_exist))

if __name__ == "__main__":
    unittest.main()
