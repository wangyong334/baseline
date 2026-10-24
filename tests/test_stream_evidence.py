"""v6-2：V2/V3 判决层接 V5（model/evidence_readout.py、train_stream_evidence.py）的单元测试（CPU）。

    1. 照搬：判决层、发布层、损失三个文件与 v3 分支逐字节相同（sha256）
    2. 背景强度 mu0 与 v3 前端的公式相同；两个头的初值；目标计数
    3. 判决层接线：net = sigmoid(mark)；没有目标强度时 fused = net；不提前发布且不锚定时 pub 标签 = fused_d{D} 的标签；
       逐窗与逐片段两种执行一致
    4. 训练路径：梯度只到两个头；损失与 v3 的 mark_loss / intensity_loss 相同
    5. 配置；端到端（V5 小骨干训练 -> 冒烟 / 训练 / 评估）
"""
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.evidence_readout import BackgroundRate, EvidenceHeads, EvidenceRun, pixel_counts  # noqa: E402
from test_stream_layerwise import CFG, H, Q99, SMALL, W, WINDOWS, synthetic_sequence  # noqa: E402
from test_stream_tif import make_tif_net  # noqa: E402
from utils.evidence_loss import intensity_loss, mark_loss  # noqa: E402

V3_SHA256 = {"model/evidence_neuron.py": "45abb0fd7bc314d213c596aa8ef5bc1dfba09f5df7375bfaa78f2b4913783e6b",
             "model/publish_readout.py": "3b0bb081953730bf87478581f850746830e21c907ec00e399202b22a21d9ab1c",
             "utils/evidence_loss.py": "7e289dcd58fed3850e65017f33a811491954a3fd6fb1e3b06627f40d1e147726"}
ROOT = Path(__file__).resolve().parents[1]


def decision_config(**overrides):
    cfg = dict(CFG, input_device="cpu", execution="step", window_ms=50, height=H, width_px=W, seed=37,
               head_hidden=8, mark_prior=0.03, intensity_prior=2e-4, log_g_max=8.0, loss_mark_weight=1.0,
               loss_intensity_weight=1.0, loss_intensity_smooth=3, bg_fast_ms=500, bg_slow_ms=5000, bg_prior=0.01,
               bg_prior_windows=2, bg_floor=0.001, bg_smooth_radius=7,
               cusum_axis_velocities=[-1, 0, 1], cusum_footprint=3, cusum_compensator="poisson", cusum_nb_kappa=None,
               cusum_aggregate="lme", cusum_track_tau_ms=250, cusum_memory_gain=1.0, cusum_gate_eps=0.03,
               readout_delays=[1, 2, 3], fusion_weight=1.0, publish_deadline=3, publish_upper=1.0,
               publish_lower=2.0, publish_collapse="linear", publish_gate=None, publish_anchor=True,
               evidence_chunk=5, evidence_streams=2, evidence_grad_clip=1.0, evidence_eval_chunk=5, threshold=0.9)
    cfg.update(overrides)
    return cfg


def random_heads(seed=3, c1=None):
    torch.manual_seed(seed)
    heads = EvidenceHeads(SMALL[0] if c1 is None else c1, hidden=8)
    for layer in (heads.head[0], heads.head[2]):
        torch.nn.init.normal_(layer.weight, std=0.5)
    return heads.eval()


class CopiedModuleTests(unittest.TestCase):
    def test_files_are_byte_identical_to_v3(self):
        for path, digest in V3_SHA256.items():
            data = (ROOT / path).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest, path)


class BackgroundAndHeadTests(unittest.TestCase):
    def test_background_matches_v3_frontend_formula(self):
        rng = np.random.RandomState(0)
        bg = BackgroundRate(50.0)
        state = bg.init_state(1, 20, 24, torch.device("cpu"), torch.float64)
        S_fast = np.zeros((20, 24))
        S_slow = np.zeros((20, 24))
        W_fast = W_slow = 0.0
        bf, bs = math.exp(-50.0 / 500.0), math.exp(-50.0 / 5000.0)
        for _ in range(6):
            mu0 = bg.background(state)[0, 0].numpy()
            fast = torch.from_numpy((S_fast + 2 * 0.01) / (W_fast + 2)).view(1, 1, 20, 24)
            fast = F.avg_pool2d(fast, 15, stride=1, padding=7, count_include_pad=False)[0, 0].numpy()
            slow = (S_slow + 2 * 0.01) / (W_slow + 2)
            np.testing.assert_allclose(mu0, np.maximum(np.maximum(fast, slow), 0.001), rtol=1e-12)
            total = rng.poisson(0.3, size=(20, 24)).astype(np.float64)
            state = bg.update(state, torch.from_numpy(total).view(1, 1, 20, 24))
            S_fast, S_slow = bf * S_fast + total, bs * S_slow + total
            W_fast, W_slow = bf * W_fast + 1.0, bs * W_slow + 1.0

    def test_head_initial_values_and_bound(self):
        torch.manual_seed(0)
        heads = EvidenceHeads(6, hidden=8, mark_prior=0.03, intensity_prior=2e-4, log_g_max=8.0)
        self.assertAlmostEqual(float(heads.head[2].bias[0]), math.log(0.03 / 0.97), places=6)
        self.assertAlmostEqual(float(heads.head[2].bias[1]), math.log(2e-4), places=6)
        mark, log_g = heads(torch.randn(2, 6, 5, 7) * 100)
        self.assertEqual(tuple(mark.shape), (2, 1, 5, 7))
        self.assertTrue(bool((log_g < 8.0).all()))

    def test_pixel_counts(self):
        y = torch.tensor([0, 0, 2, 2, 2])
        x = torch.tensor([1, 1, 3, 3, 0])
        b = torch.tensor([0, 0, 0, 1, 1])
        c = pixel_counts(y, x, b, 2, 4, 5, torch.float64)
        self.assertEqual(float(c[0, 0, 0, 1]), 2.0)
        self.assertEqual(float(c[1, 0, 2, 3]), 1.0)
        w = pixel_counts(y, x, b, 2, 4, 5, torch.float64, weight=torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0]))
        self.assertEqual(float(w.sum()), 3.0)


def run_synthetic(cfg, log_g_value=None, seed=0, n_windows=12, per_window=30):
    """在随机事件上直接驱动 EvidenceRun；log_g_value 给定时强度场取常数，否则取随机场。"""
    rng = np.random.RandomState(seed)
    dev = torch.device("cpu")
    run = EvidenceRun(cfg, 16, 20, dev, torch.float64)
    n_total = n_windows * per_window
    logits_all = torch.from_numpy(rng.normal(0, 3, n_total))
    order = 0
    for k in range(n_windows):
        y = torch.from_numpy(rng.randint(0, 16, per_window)).long()
        x = torch.from_numpy(rng.randint(0, 20, per_window)).long()
        if log_g_value is None:
            log_g = torch.from_numpy(rng.normal(-1.0, 1.5, (1, 1, 16, 20)))
        else:
            log_g = torch.full((1, 1, 16, 20), float(log_g_value), dtype=torch.float64)
        idx = np.arange(order, order + per_window)
        run.step(k, log_g, logits_all[idx], y, x, idx)
        order += per_window
    probs, extra = run.outputs(n_total)
    return probs, extra, logits_all.numpy()


class DecisionWiringTests(unittest.TestCase):
    def test_net_is_sigmoid_of_mark(self):
        probs, _, logits = run_synthetic(decision_config())
        np.testing.assert_allclose(probs["net"], (1 / (1 + np.exp(-logits))).astype(np.float32), rtol=1e-6)

    def test_no_target_intensity_gives_fused_equal_net(self):
        probs, _, _ = run_synthetic(decision_config(cusum_gate_eps=0.03), log_g_value=-30.0)
        for d in (1, 2, 3):
            np.testing.assert_allclose(probs["fused_d%d" % d], probs["net"], atol=1e-6)

    def test_pub_without_early_publishing_equals_fused_deadline(self):
        cfg = decision_config(publish_upper=1e9, publish_lower=1e9, publish_anchor=False)
        probs, extra, _ = run_synthetic(cfg)
        thr = np.float32(0.9)
        last = 12 - 1 - np.repeat(np.arange(12), 30)
        full = last >= 3
        self.assertTrue(np.array_equal((probs["pub"] >= thr)[full], (probs["fused_d3"] >= thr)[full]))
        self.assertTrue(np.all(extra["age_pub"][full] == 3))
        self.assertFalse(np.array_equal(probs["fused_d3"] >= thr, probs["net"] >= thr))

    def test_early_publishing_happens_and_ages_are_bounded(self):
        probs, extra, _ = run_synthetic(decision_config())
        self.assertTrue(np.all((extra["age_pub"] >= 0) & (extra["age_pub"] <= 3)))
        self.assertGreater(int(np.count_nonzero(extra["age_pub"] == 0)), 0)
        self.assertGreater(int(np.count_nonzero(extra["age_pub"] > 0)), 0)


class TrainingPathTests(unittest.TestCase):
    def test_losses_match_v3_and_gradients_only_reach_heads(self):
        import train_stream_v1 as T
        import train_stream_evidence as E
        seqs = [synthetic_sequence(seed=s) for s in (3, 4, 5)]
        cfg = decision_config()
        plan = [(1, [(0, WINDOWS), (2, WINDOWS)]), (3, [(1, WINDOWS)])]
        dev = torch.device("cpu")
        net = make_tif_net(merged=False)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        heads = random_heads().double()
        sources = T.StreamSources(seqs, cfg, Q99, dev, torch.float64)
        pieces = T.stream_pieces(plan, 5, 10)
        x, events, labels, keep = T.assemble_streams(sources, pieces, 5, cfg, dev, torch.float64)
        ones = torch.ones((5, 2), dtype=torch.float64)
        with torch.no_grad():
            _, _, info = net.forward_chunk(x, events, None, keep=keep, collect=True)
        u7 = info["u_pre"][6]
        bce, nll = E.head_losses(heads, u7, events, labels, ones, 3)
        mark, log_g = heads(u7.reshape((10,) + tuple(u7.shape[2:])))
        flat = events["t"] * 2 + events["b"]
        ref_bce = mark_loss(mark[flat, 0, events["y"], events["x"]], labels)
        target = pixel_counts(events["y"], events["x"], flat, 10, H, W, torch.float64, weight=labels)
        ref_nll = intensity_loss(log_g.view(5, 2, 1, H, W), target.view(5, 2, 1, H, W), 3)
        self.assertAlmostEqual(float(bce), float(ref_bce), places=8)
        self.assertAlmostEqual(float(nll), float(ref_nll), places=8)
        heads.train()
        opt = torch.optim.SGD(heads.parameters(), lr=0.0)
        out = E.train_epoch(net, heads, seqs, None, opt, cfg, cfg, Q99, dev, np.random.RandomState(0), plan=plan)
        self.assertGreater(out["updates"], 0)
        self.assertTrue(all(p.grad is not None for p in heads.parameters()))
        self.assertTrue(all(p.grad is None for p in net.parameters()))

    def test_step_and_layer_inference_agree(self):
        import train_stream_evidence as E
        net = make_tif_net(merged=False)
        net.eval()
        heads = random_heads().double()
        cfg = decision_config()
        seq = synthetic_sequence(seed=9)
        with torch.no_grad():
            a = E.evaluate_sequence(net, heads, seq, cfg, cfg, Q99, torch.device("cpu"), "step")
            b = E.evaluate_sequence(net, heads, seq, cfg, dict(cfg, input_device="gpu"), Q99, torch.device("cpu"),
                                    "layer")
        self.assertEqual(set(a["probs"]), set(E.readout_names(cfg)))
        for name in a["probs"]:
            self.assertLess(float(np.abs(a["probs"][name] - b["probs"][name]).max()), 1e-6, name)
        np.testing.assert_allclose(a["v5"], b["v5"], atol=1e-6)
        self.assertTrue(np.array_equal(a["extra"]["age_pub"], b["extra"]["age_pub"]))


class ConfigTests(unittest.TestCase):
    def test_config_matches_v3_decision_constants(self):
        import train_stream_evidence as E
        from utils.stream_common import load_flat_config
        cfg = E.build_config("configs/evisseg_stream_evidence.yaml")
        base = load_flat_config(cfg["base_config"])
        self.assertEqual(os.path.dirname(cfg["backbone_checkpoint"]), base["save_root"])
        expected = {"cusum_axis_velocities": [-4, -2, -1, 0, 1, 2, 4], "cusum_footprint": 3,
                    "cusum_compensator": "poisson", "cusum_aggregate": "lme", "cusum_track_tau_ms": 250,
                    "cusum_memory_gain": 1.0, "cusum_gate_eps": 0.03, "fusion_weight": 1.0, "publish_deadline": 5,
                    "publish_upper": 1.0, "publish_lower": 2.0, "publish_collapse": "linear", "publish_gate": None,
                    "publish_anchor": True, "bg_fast_ms": 500, "bg_slow_ms": 5000, "bg_prior": 0.01,
                    "bg_prior_windows": 2, "bg_floor": 0.001, "bg_smooth_radius": 7, "head_hidden": 16,
                    "mark_prior": 0.03, "intensity_prior": 0.0002, "log_g_max": 8.0, "loss_intensity_smooth": 3}
        for key, value in expected.items():
            self.assertEqual(cfg[key], value, key)


class EndToEndTests(unittest.TestCase):
    def test_backbone_then_evidence(self):
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
            import yaml
        except ImportError as error:
            self.skipTest("完整入口需要原 V1 的 cv2/pandas/tqdm 依赖: %s" % error)
        import train_stream_v1 as T
        import train_stream_evidence as E
        from utils.stream_common import load_flat_config
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pos = neg = 0
            for split, seeds in (("train", (3, 4, 5)), ("val", (6, 7)), ("test", (8,))):
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
            run = root / "backbone"
            cfg = load_flat_config("configs/evisseg_stream_tif_comm_peak.yaml")
            cfg.update(root=str(root), stats_path=str(root / "stats.json"), save_root=str(run),
                       channels=list(SMALL), height=H, width_px=W, pad_height=H, pad_width=W,
                       n_windows=WINDOWS, epochs=2, num_workers=0, tbptt_k=4, train_subset_size=1,
                       calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
                       segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=4,
                       execution="layer", input_device="gpu", streams=2)
            T.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
            checkpoint = run / "best_val_iou_seed37.pt"
            v5_args = SimpleNamespace(checkpoint=str(checkpoint), device="cpu", mode="eval", split="val",
                                      overwrite=True, execution=None, input_device=None, readout_ablation="none",
                                      subthreshold_ablation="none", memory_ablation="none", memory_off_layers=None,
                                      reset_every=0, reset_offset=0, dump_dir=None)
            T.mode_eval(v5_args, cfg)
            reference = run / "eval_val_best_val_iou_seed37.json"
            base_path, ev_path = root / "base.yaml", root / "evidence.yaml"
            base_path.write_text(yaml.safe_dump({"ALL": cfg}), encoding="utf-8")
            ev = load_flat_config("configs/evisseg_stream_evidence.yaml")
            ev.update(base_config=str(base_path), backbone_checkpoint=str(checkpoint), head_hidden=8,
                      cusum_axis_velocities=[-1, 0, 1], readout_delays=[1, 2, 3], publish_deadline=3,
                      evidence_epochs=2, evidence_lr=0.01, evidence_lr_end=0.001, evidence_streams=2,
                      evidence_chunk=4, evidence_eval_chunk=4, evidence_save_root=str(root / "evidence"))
            ev_path.write_text(yaml.safe_dump({"EVIDENCE": ev}), encoding="utf-8")
            ecfg = E.build_config(str(ev_path))
            E.mode_smoke(SimpleNamespace(device="cpu"), ecfg)
            E.mode_train(SimpleNamespace(device="cpu"), ecfg)
            self.assertTrue((root / "evidence" / "best_heads.pt").exists())
            E.mode_eval(SimpleNamespace(device="cpu", split="val", reference=str(reference), overwrite=True), ecfg)
            result = json.loads((root / "evidence" / "eval_evidence_val.json").read_text(encoding="utf-8"))
            self.assertTrue(all(item["equal"] for item in result["v5_vs_reference"].values()), result["v5_vs_reference"])
            self.assertEqual(set(result["readouts"]), {"net", "fused_d1", "fused_d2", "fused_d3", "pub"})
            self.assertEqual(len(result["publish"]["age_hist_target"]), 4)


if __name__ == "__main__":
    unittest.main()
