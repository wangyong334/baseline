"""模块三：目标级航迹头（model/track_head.py、train_track_head.py）的测试（CPU、合成数据）。

    1. 初始化时最终 logit 与骨干 logit 逐位相同（修正项末层零初始化）
    2. 片段切分不变：一次处理 12 窗与分 3 段（段间 detach）处理，输出逐位相同
    3. 匀速运动的目标：只出生一条航迹，位置、速度跟得上；短暂消失 2 窗时航迹外推并重新接上（同一个槽）
    4. 新录像开始（keep = 0）时只清空该流的航迹
    5. 近期轨迹距离：落在航迹 2～20 窗前位置上的事件，该项接近 0；远离的事件为 1
    6. 监督：标注目标与航迹的匹配、存在 / 位置损失可反传，梯度只到航迹头
    7. 确定性模式下前向、反向不报错
    8. 端到端：训练一个小骨干 -> 航迹头训练 1 轮 -> 评估（四个指标、误差类别、关口）
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.track_head import TrackHead, greedy_match, window_targets  # noqa: E402

H, W, FEAT = 32, 48, 4


def make_head(seed=0, **kwargs):
    torch.manual_seed(seed)
    return TrackHead(FEAT, H, W, cell=4, slots=6, hidden=16, history=21, **kwargs)


def blob_windows(n_windows, start=(8.0, 12.0), velocity=(2.0, 0.0), gaps=(), batch=1, stream=0, noise=10, seed=0,
                 keep_out=()):
    """每窗一个 7 个事件的目标（z0 = +6），外加随机背景事件（z0 = -6）。返回片段事件与 z0、feat、标签、target_id。"""
    rng = np.random.RandomState(seed)
    offs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
    cols = {key: [] for key in ("t", "b", "x", "y", "p", "t_local")}
    z0, lab, tid = [], [], []
    for t in range(n_windows):
        for b in range(batch):
            if b == stream and t not in gaps and t not in keep_out:
                cx, cy = start[0] + velocity[0] * t, start[1] + velocity[1] * t
                for dx, dy in offs:
                    cols["t"].append(t); cols["b"].append(b)
                    cols["x"].append(int(round(cx + dx))); cols["y"].append(int(round(cy + dy)))
                    cols["p"].append(float(rng.randint(2))); cols["t_local"].append(float(rng.rand()))
                    z0.append(6.0); lab.append(1.0); tid.append(1.0)
            for _ in range(noise):
                cols["t"].append(t); cols["b"].append(b)
                cols["x"].append(int(rng.randint(W))); cols["y"].append(int(rng.randint(H)))
                cols["p"].append(float(rng.randint(2))); cols["t_local"].append(float(rng.rand()))
                z0.append(-6.0); lab.append(0.0); tid.append(0.0)
    events = {key: torch.tensor(v, dtype=torch.long if key in ("t", "b", "x", "y") else torch.float32)
              for key, v in cols.items()}
    z0 = torch.tensor(z0, dtype=torch.float32)
    feat = torch.from_numpy(rng.randn(z0.shape[0], FEAT).astype(np.float32)) * 0.1
    return events, z0, feat, torch.tensor(lab), torch.tensor(tid)


class IdentityAndChunkTests(unittest.TestCase):
    def test_zero_init_identity(self):
        head = make_head()
        events, z0, feat, _, _ = blob_windows(8, noise=30)
        with torch.no_grad():
            z, _, _ = head.forward_chunk(events, z0, feat, None, 8)
        self.assertTrue(torch.equal(z, z0))

    def test_chunk_split_invariance(self):
        head = make_head()
        with torch.no_grad():
            for p in head.corr.parameters():                       # 让修正项非零，检验状态延续
                p.normal_(0.0, 0.3)
        events, z0, feat, _, _ = blob_windows(12, batch=2, noise=20)
        with torch.no_grad():
            z_full, s_full, _ = head.forward_chunk(events, z0, feat, head.init_state(2, z0.device), 12)
            parts, state = [], head.init_state(2, z0.device)
            for a, b in ((0, 3), (3, 8), (8, 12)):
                sel = (events["t"] >= a) & (events["t"] < b)
                ev = {key: v[sel] for key, v in events.items()}
                ev["t"] = ev["t"] - a
                z_part, state, _ = head.forward_chunk(ev, z0[sel], feat[sel], state, b - a)
                state = head.detach_state(state)
                parts.append((sel, z_part))
        z_split = torch.zeros_like(z_full)
        for sel, z_part in parts:
            z_split[sel] = z_part
        self.assertTrue(torch.allclose(z_full, z_split, atol=0, rtol=0))
        for key in s_full:
            self.assertTrue(torch.equal(s_full[key], state[key]), key)


class TrackingTests(unittest.TestCase):
    def run_blob(self, n, **kwargs):
        head = make_head()
        events, z0, feat, _, _ = blob_windows(n, **kwargs)
        log = []
        with torch.no_grad():
            _, state, _ = head.forward_chunk(events, z0, feat, head.init_state(1, z0.device), n, log=log)
        return head, state, log

    def test_follows_constant_velocity_target(self):
        _, state, log = self.run_blob(12, noise=15)
        active = state["active"][0]
        self.assertEqual(int(active.sum()), 1)                     # 噪声事件证据太弱，不出生
        slot = int(torch.nonzero(active)[0])
        pos, vel = state["pos"][0, slot], state["vel"][0, slot]
        self.assertLess(float((pos - torch.tensor([8.0 + 2.0 * 11, 12.0])).abs().max()), 1.5)
        self.assertLess(abs(float(vel[0]) - 2.0), 0.7)
        self.assertLess(abs(float(vel[1])), 0.7)

    def test_coasts_through_short_gap(self):
        _, state, log = self.run_blob(12, noise=15, gaps=(6, 7))
        slots = [set(np.flatnonzero(w["active"][0]).tolist()) for w in log]
        self.assertEqual(slots[5], slots[6])                       # 消失的两窗里航迹还在（外推）
        self.assertEqual(slots[5], slots[11])                      # 目标回来后接上同一个槽
        self.assertEqual(int(state["miss"][0, sorted(slots[11])[0]]), 0)

    def test_reset_clears_only_that_stream(self):
        head = make_head()
        events, z0, feat, _, _ = blob_windows(10, batch=2, stream=0, noise=10)
        ev1, z1, f1, _, _ = blob_windows(10, batch=2, stream=1, noise=0, keep_out=range(6, 10), seed=1)
        events = {key: torch.cat([events[key], ev1[key]]) for key in events}
        z0, feat = torch.cat([z0, z1]), torch.cat([feat, f1])
        keep = torch.ones(10, 2)
        keep[6, 1] = 0
        with torch.no_grad():
            _, state, _ = head.forward_chunk(events, z0, feat, head.init_state(2, z0.device), 10, keep=keep)
        self.assertEqual(int(state["active"][0].sum()), 1)
        self.assertEqual(int(state["active"][1].sum()), 0)         # 清空后流 1 没有新证据


class ContextTests(unittest.TestCase):
    def test_recent_path_distance(self):
        head = make_head()
        st = head.init_state(1, torch.device("cpu"))
        st["active"][0, 0] = True
        st["ex"][0, 0] = 0.9
        st["pos"][0, 0] = torch.tensor([40.0, 20.0])
        st["P"][0, 0] = torch.eye(2)
        st["hist"][0, 0, 5] = torch.tensor([10.0, 20.0])          # 5 窗之前目标在 (10, 20)
        st["hist_ok"][0, 0, 5] = True
        ev = {"b": torch.zeros(2, dtype=torch.long), "x": torch.tensor([10, 46]), "y": torch.tensor([20, 0]),
              "z0": torch.zeros(2)}
        ctx = head._context(ev, st)
        self.assertLess(float(ctx[0, 5]), 0.05)                    # 落在近期轨迹上
        self.assertEqual(float(ctx[1, 5]), 1.0)                    # 远离近期轨迹
        self.assertGreater(float(ctx[0, 4]), 0.9)                  # 离当前位置远（30 像素）


class SupervisionTests(unittest.TestCase):
    def test_window_targets_and_greedy(self):
        eb = torch.tensor([0, 0, 0, 1, 1])
        ex = torch.tensor([10, 12, 30, 5, 7])
        ey = torch.tensor([10, 10, 30, 5, 5])
        label = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
        tid = torch.tensor([1.0, 1.0, 0.0, 2.0, 2.0])
        gb, gc, gr = window_targets(eb, ex, ey, label, tid, 8.0)
        self.assertEqual(gb.tolist(), [0, 1])
        self.assertTrue(torch.allclose(gc, torch.tensor([[11.0, 10.0], [6.0, 5.0]])))
        self.assertTrue(torch.all(gr >= 8.0))
        dist = np.array([[1.0, 5.0], [2.0, np.inf]])
        self.assertEqual(greedy_match(dist, np.array([0, 0])), [(0, 0)])   # 同一流内一个槽只配一次
        self.assertEqual(sorted(greedy_match(dist, np.array([0, 1]))), [(0, 0), (1, 0)])

    def test_losses_backward_to_head_only(self):
        head = make_head()
        events, z0, feat, lab, tid = blob_windows(10, noise=20)
        z0 = z0.clone().requires_grad_(False)
        z, _, aux = head.forward_chunk(events, z0, feat, None, 10, labels=lab, tid=tid)
        self.assertGreater(aux["exist_n"], 0)
        self.assertGreater(aux["pos_n"], 0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, lab) + aux["exist_sum"] / aux["exist_n"] \
            + aux["pos_sum"] / aux["pos_n"]
        loss.backward()
        grads = {name: p.grad for name, p in head.named_parameters()}
        for name in ("exist.weight", "gru.weight_ih", "corr.2.weight", "r_net.2.bias", "log_q"):
            self.assertIsNotNone(grads[name], name)
            self.assertTrue(torch.isfinite(grads[name]).all(), name)

    def test_deterministic_mode(self):
        torch.use_deterministic_algorithms(True)
        try:
            head = make_head()
            events, z0, feat, lab, tid = blob_windows(6, batch=2, noise=10)
            z, _, aux = head.forward_chunk(events, z0, feat, None, 6, labels=lab, tid=tid)
            (z.sum() + aux["exist_sum"]).backward()
        finally:
            torch.use_deterministic_algorithms(False)


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
        import train_stream_v1 as T
        import train_track_head as TH
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
            cfg = load_flat_config("configs/evisseg_stream_tif_comm_peak_invariant.yaml")
            cfg.update(root=str(root), stats_path=str(root / "stats.json"), save_root=str(base_dir),
                       channels=list(SMALL), height=SH, width_px=SW, pad_height=SH, pad_width=SW,
                       n_windows=12, epochs=2, num_workers=0, tbptt_k=4, train_subset_size=1,
                       calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
                       segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=4,
                       execution="layer", input_device="gpu", streams=2)
            T.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
            head_cfg = load_flat_config("configs/track_head_e3a.yaml")
            head_cfg.update(root=str(root), base_checkpoint=str(base_dir / "best_val_iou_seed37.pt"),
                            save_root=str(root / "head"), epochs=2, streams=2, tbptt_k=4, eval_chunk=4,
                            slots=4, segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1)
            cfg_path = root / "head.yaml"
            sections = {"DATA": head_cfg}
            import yaml
            cfg_path.write_text(yaml.safe_dump(sections), encoding="utf-8")
            TH.main(["--config", str(cfg_path), "--mode", "train", "--device", "cpu"])
            ckpt = root / "head" / "best_val_iou_seed37.pt"
            self.assertTrue(ckpt.exists())
            TH.main(["--config", str(cfg_path), "--mode", "eval", "--device", "cpu", "--split", "test",
                     "--checkpoint", str(ckpt), "--dump-dir", str(root / "dump"), "--tracks-dir", str(root / "tracks")])
            result = json.loads((root / "head" / "eval_test_best_val_iou_seed37.json").read_text(encoding="utf-8"))
            for kind in ("base", "track"):
                for key in ("iou", "acc", "pd", "fa"):
                    self.assertTrue(np.isfinite(result[kind][key]), (kind, key))
            self.assertIn("pass", result["gate"])
            with np.load(str(root / "dump" / "seq_7.npz")) as dump:
                self.assertEqual(dump["probabilities"].shape, dump["base_probabilities"].shape)
            with np.load(str(root / "tracks" / "seq_7.npz")) as tracks:
                self.assertEqual(tracks["pos"].shape[1:], (4, 2))
            # base 一列是骨干的原输出：与正式评估（逐窗执行）的 IoU 一致
            args = SimpleNamespace(checkpoint=str(base_dir / "best_val_iou_seed37.pt"), device="cpu", mode="eval",
                                   split="test", overwrite=True, execution=None, input_device=None,
                                   readout_ablation="none", subthreshold_ablation="none", memory_ablation="none",
                                   memory_off_layers=None, reset_every=0, reset_offset=0, dump_dir=None,
                                   eval_noise_load=0.0, eval_noise_seed=0)
            T.mode_eval(args, cfg)
            official = json.loads((base_dir / "eval_test_best_val_iou_seed37.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(result["base"]["iou"], official["carry"]["iou"], places=3)


if __name__ == "__main__":
    unittest.main()
