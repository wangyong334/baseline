"""V6 第一版（滞后读出头 + 自适应发布，Stream-SNN-Innovations.md 13.12）的单元测试（CPU、float64 小网络）。

    1. 读出头：零初始化时 logit_d = z0；特征图按 (窗, 流) 展平；enc4 邻域均值
    2. 待判缓存：每个 (事件, 滞后) 恰好处理一次，录像结束后不再读
    3. 推理：逐窗与逐片段两种执行一致；d = 0 与 V5 run_sequence 逐位相同；因果性（删掉未来事件不改变已到期的分数）
    4. 训练路径与推理路径对同一事件给出相同的 logit（多流排布、错开起点）；梯度只到读出头
    5. 判决与指标：m = 0 即零等待、m = inf 即固定截止；IoU/ACC 与 utils/eval.py 相同；目标旁误检的计数
    6. 配置：两份 LAG 配置只差该差的键；端到端（V5 小骨干训练 -> 读出头训练 -> 评估与导出 -> 发布回放）
"""
import contextlib
import copy
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import stream_windows as sw  # noqa: E402
from model.lag_readout import LagBuffer, LagReadout, gather_map, lag_maps  # noqa: E402
from test_stream_layerwise import CFG, H, Q99, SMALL, W, WINDOWS, synthetic_sequence  # noqa: E402
from test_stream_tif import make_tif_net  # noqa: E402
from utils.lag_metrics import adaptive_publish, iou_acc_counts  # noqa: E402

LAGS = 3
NET_CFG = dict(CFG, input_device="cpu", execution="step", lags=LAGS, lag_eval_chunk=5)


def random_readouts(seed=11, dtype=torch.float64, birth_extra=0):
    """两个 variant 的读出头，最后一层设成非零（否则 logit_d 恒等于 z0，测试测不出东西）。"""
    torch.manual_seed(seed)
    out = {}
    for v in ("pixel", "multiscale"):
        r = LagReadout(SMALL, LAGS, v, hidden=8, birth_extra=birth_extra).to(dtype)
        for head in r.heads:
            torch.nn.init.normal_(head[2].weight, std=0.5)
            torch.nn.init.normal_(head[2].bias, std=0.5)
        out[v] = r.eval()
    return out


def subset_sequence(seq, keep):
    """保留部分事件，重新切窗（窗数不变）。"""
    x, y, t, p, label, tid = (a[keep] for a in (seq.x, seq.y, seq.t, seq.p, seq.label, seq.target_id))
    order, bounds = sw.split_windows(t, 50, WINDOWS)
    bins, local = sw.time_bin_and_local(t, 50, 5)
    return sw.StreamSequence(seq.name, x, y, t, p, label, tid, bins, local, order, bounds)


class ReadoutTests(unittest.TestCase):
    def test_zero_init_returns_z0(self):
        g = torch.Generator().manual_seed(0)
        for v in ("pixel", "multiscale"):
            r = LagReadout(SMALL, 4, v).double()
            n = 7
            now = torch.randn(n, r.now_dim, generator=g, dtype=torch.float64)
            birth = torch.randn(n, SMALL[0], generator=g, dtype=torch.float64)
            z0 = torch.randn(n, generator=g, dtype=torch.float64)
            p, tl = torch.rand(n, generator=g, dtype=torch.float64), torch.rand(n, generator=g, dtype=torch.float64)
            for d in range(1, 5):
                self.assertTrue(torch.equal(r(d, now, birth, z0, p, tl), z0), (v, d))
            with self.assertRaises(ValueError):
                r(0, now, birth, z0, p, tl)
        with self.assertRaises(ValueError):
            LagReadout(SMALL, 2, "nearest")

    def test_maps_flatten_and_neighbourhood_mean(self):
        g = torch.Generator().manual_seed(1)
        steps, batch = 3, 2
        shapes = [(SMALL[0], H, W), (SMALL[1], H // 2, W // 2), (SMALL[2], H // 4, W // 4), (SMALL[3], H // 8, W // 8),
                  (SMALL[2], H // 4, W // 4), (SMALL[1], H // 2, W // 2), (SMALL[0], H, W)]
        u_pre = [torch.randn((steps, batch) + s, generator=g, dtype=torch.float64) for s in shapes]
        maps = lag_maps(u_pre, ["pixel", "multiscale"])
        self.assertEqual(set(maps), {"dec1", "dec2", "dec3", "enc4", "enc4_mean"})
        t, b, y, x = 2, 1, 13, 21
        flat = torch.tensor([t * batch + b])
        self.assertTrue(torch.equal(gather_map(maps["dec1"], flat, torch.tensor([y]), torch.tensor([x]), 1)[0],
                                    u_pre[6][t, b, :, y, x]))
        self.assertTrue(torch.equal(gather_map(maps["dec2"], flat, torch.tensor([y]), torch.tensor([x]), 2)[0],
                                    u_pre[5][t, b, :, y // 2, x // 2]))
        enc4 = u_pre[3][t, b]
        for (cy, cx) in ((0, 0), (1, 1), (1, 2)):
            ys, xs = slice(max(cy - 1, 0), cy + 2), slice(max(cx - 1, 0), cx + 2)
            manual = enc4[:, ys, xs].mean(dim=(1, 2))
            got = maps["enc4_mean"][t * batch + b, :, cy, cx]
            self.assertLess(float((manual - got).abs().max()), 1e-12)
        # 逐窗 forward 给出的 [B,C,h,w] 也能用
        single = lag_maps([u[0] for u in u_pre], ["pixel"])
        self.assertEqual(tuple(single["dec1"].shape), (batch, SMALL[0], H, W))


class BufferTests(unittest.TestCase):
    def test_each_pair_processed_once(self):
        rng = np.random.RandomState(3)
        lags, n = 4, 400
        g = rng.randint(0, 60, n)
        end = g + 1 + rng.randint(0, 8, n)                       # 录像在出生后 0..7 窗结束
        for sizes in ((5,) * 20, (4, 7, 9, 5, 6, 8, 10, 11, 5, 4, 12), (60, 40)):
            buffer, seen = LagBuffer(lags), np.zeros((n, lags + 1), dtype=np.int64)
            start, added = 0, np.zeros(n, dtype=bool)
            for size in sizes:
                stop = start + size
                new = (g >= start) & (g < stop) & ~added
                ids = np.flatnonzero(new)
                if ids.size:
                    zeros = torch.zeros(ids.size)
                    buffer.add(g=torch.from_numpy(g[ids]), end=torch.from_numpy(end[ids]),
                               b=torch.zeros(ids.size, dtype=torch.long), y=torch.zeros(ids.size, dtype=torch.long),
                               x=torch.zeros(ids.size, dtype=torch.long), p=zeros, t_local=zeros, z0=zeros,
                               birth=torch.zeros(ids.size, 1), label=zeros, key=torch.from_numpy(ids))
                    added |= new
                for d in range(1, lags + 1):
                    mask = buffer.due(d, start, stop)
                    if mask is not None:
                        rows = buffer.select(mask)
                        np.add.at(seen[:, d], rows["key"].numpy(), 1)
                buffer.prune(stop)
                start = stop
            expected = np.stack([(g + d < end).astype(np.int64) for d in range(lags + 1)], 1)
            expected[:, 0] = 0
            self.assertTrue(np.array_equal(seen, expected), sizes)
            self.assertEqual(len(buffer), 0)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        import train_stream_lag as L
        self.L = L
        self.cfg = NET_CFG
        self.net = make_tif_net(merged=False)
        self.net.eval()
        self.readouts = random_readouts()

    def run_eval(self, seq, execution):
        with torch.no_grad():
            return self.L.evaluate_sequence(self.net, self.readouts, seq, self.cfg, self.cfg, Q99,
                                            torch.device("cpu"), execution)

    def test_step_layer_equal_and_d0_is_v5(self):
        import train_stream_v1 as T
        seq = synthetic_sequence(seed=4)
        step, win, avail = self.run_eval(seq, "step")
        layer, _, _ = self.run_eval(seq, "layer")
        for v in self.readouts:
            self.assertLess(float(np.abs(step[v] - layer[v]).max()), 1e-6, v)
            self.assertFalse(np.allclose(step[v][1], step[v][0]), v)          # 读出头确实改变了分数
        with torch.no_grad():
            probs, _ = T.run_sequence(self.net, seq, self.cfg, Q99, torch.device("cpu"), "carry")
        self.assertTrue(np.array_equal(step["pixel"][0], probs))             # d = 0 与 V5 逐位相同
        self.assertTrue(np.array_equal(step["multiscale"][0], probs))
        self.assertTrue(np.array_equal(avail, np.minimum(LAGS, WINDOWS - 1 - win)))
        for v in self.readouts:                                               # 读不到的滞后用最后可读年龄填充
            late = avail < LAGS
            self.assertTrue(np.array_equal(step[v][LAGS, late], step[v][avail[late], np.flatnonzero(late)]))

    def test_causality(self):
        seq = synthetic_sequence(seed=5)
        cut = 8
        keep = seq.t < 50 * cut                                              # 删掉第 8 窗起的全部事件
        short = subset_sequence(seq, keep)
        full, win, _ = self.run_eval(seq, "step")
        part, win_s, _ = self.run_eval(short, "step")
        self.assertTrue(np.array_equal(win[keep], win_s))
        for v in self.readouts:
            for d in range(LAGS + 1):
                due = win_s + d < cut                                         # 判决窗在第 8 窗之前
                self.assertTrue(np.array_equal(full[v][d][keep][due], part[v][d][due]), (v, d))

    def test_train_path_matches_eval_path(self):
        import train_stream_lag as L
        import train_stream_v1 as T
        seqs = [synthetic_sequence(seed=6), synthetic_sequence(seed=7), synthetic_sequence(seed=8)]
        plan = [(1, [(0, WINDOWS), (2, WINDOWS)]), (3, [(1, WINDOWS)])]   # 两条流，错开起点，第一条流连着两段录像
        for p in self.net.parameters():
            p.requires_grad_(False)
        readouts = copy.deepcopy(self.readouts)
        params = [p for r in readouts.values() for p in r.parameters()]
        optimizer = torch.optim.SGD(params, lr=0.0)
        cfg = dict(self.cfg, lag_chunk=5, lag_streams=2, lag_grad_clip=1.0)
        collected = []
        original = L.lag_logits

        def spy(*args, **kwargs):
            groups = original(*args, **kwargs)
            collected.extend((v, d, {k: r[k].detach().clone() for k in ("g", "b", "y", "x", "p", "t_local")},
                              logit.detach().clone()) for v, d, r, logit in groups)
            return groups

        L.lag_logits = spy
        try:
            out = L.train_epoch(self.net, readouts, seqs, None, optimizer, cfg, cfg, Q99, 2.0,
                                torch.device("cpu"), np.random.RandomState(0), plan=plan)
        finally:
            L.lag_logits = original
        self.assertGreater(out["updates"], 0)
        for p in params:
            self.assertIsNotNone(p.grad)
        self.assertTrue(all(p.grad is None for p in self.net.parameters()))
        # 推理路径：每条序列单独从零状态开始（流上新录像开始时状态清零，等价）
        starts = {0: (0, 1), 2: (0, 1 + WINDOWS), 1: (1, 3)}                 # 录像 -> (流, 流时间起点)
        expected = {}
        for index, (b, s0) in starts.items():
            probs, win, _ = self.run_eval(seqs[index], "step")
            for d in range(1, LAGS + 1):
                ok = win + d < WINDOWS
                for v in self.readouts:
                    for i in np.flatnonzero(ok):
                        key = (v, d, b, s0 + int(win[i]), int(seqs[index].y[i]), int(seqs[index].x[i]),
                               float(seqs[index].p[i]), float(seqs[index].t_local[i]))
                        expected.setdefault(key, []).append(float(probs[v][d][i]))
        got = {}
        for v, d, rows, logit in collected:
            prob = torch.sigmoid(logit).float().numpy()
            for j in range(prob.shape[0]):
                key = (v, d, int(rows["b"][j]), int(rows["g"][j]), int(rows["y"][j]), int(rows["x"][j]),
                       float(rows["p"][j]), float(np.float32(rows["t_local"][j])))
                got.setdefault(key, []).append(float(prob[j]))
        self.assertEqual(set(got), set(expected))
        for key in expected:
            a, b = sorted(expected[key]), sorted(got[key])
            self.assertEqual(len(a), len(b), key)
            self.assertLess(max(abs(u - w) for u, w in zip(a, b)), 1e-6, key)


class PublishMetricTests(unittest.TestCase):
    def test_adaptive_publish_extremes(self):
        rng = np.random.RandomState(2)
        lags, n, thr = 4, 3000, 0.9
        prob = rng.rand(lags + 1, n).astype(np.float32)
        prob[:, :40] = np.float32(0.9)                                       # 恰好等于阈值
        avail = rng.randint(0, lags + 1, n)
        for deadline in (1, 3, 4):
            pub, age, reason = adaptive_publish(prob, avail, deadline, 0.0, thr)
            self.assertTrue(np.all(age == 0) and np.array_equal(pub, prob[0]))
            pub, age, reason = adaptive_publish(prob, avail, deadline, math.inf, thr)
            last = np.minimum(avail, deadline)
            self.assertTrue(np.array_equal(age, last))
            self.assertTrue(np.array_equal(pub, prob[last, np.arange(n)]))
            for m in (0.3, 1.0, 2.5):
                pub, age, reason = adaptive_publish(prob, avail, deadline, m, thr)
                self.assertTrue(np.all(age >= 0) and np.all(age <= last))
                self.assertTrue(np.array_equal(pub, prob[age, np.arange(n)]))
                self.assertTrue(np.all(pub[reason == 0] >= np.float32(thr)))
                self.assertTrue(np.all(pub[reason == 1] < np.float32(thr)))
                self.assertTrue(np.all(age[reason == 2] == deadline))
                self.assertTrue(np.all(age[reason == 3] == avail[reason == 3]))

    def test_iou_acc_match_utils_eval(self):
        try:
            from utils.eval import evalute
        except ImportError as error:
            self.skipTest("utils/eval.py 需要原 V1 依赖: %s" % error)
        rng = np.random.RandomState(4)
        evaluator = evalute(SimpleNamespace(roc=False, pd_detT=50, correct_thresh=1e-4))
        probs, labels = [], []
        for i in range(3):
            p = rng.rand(500).astype(np.float32)
            p[:20] = np.float32(0.9)
            lab = (rng.rand(500) < 0.3).astype(np.float32)
            evaluator.matches[str(i)] = {"seg_pred": torch.from_numpy(p), "seg_gt": torch.from_numpy(lab)}
            probs.append(p)
            labels.append(lab)
        iou, acc = evaluator.evaluate_iou_and_accuracy(thresh=0.9)
        mine = iou_acc_counts(np.concatenate(probs), np.concatenate(labels), 0.9)
        self.assertEqual((iou, acc), mine[:2])

    def test_leakage_counts(self):
        from utils.lag_metrics import leakage_counts
        # 2 个窗、6x8 画面：目标事件在 (w0, 2, 2)；误检 (w0, 4, 5) 在 3 像素内，(w0, 0, 7) 不在，(w1, 2, 2) 不在同一窗
        s = {"win": np.array([0, 0, 0, 1, 1]), "y": np.array([2, 4, 0, 2, 5]), "x": np.array([2, 5, 7, 2, 0]),
             "label": np.array([1, 0, 0, 0, 0], dtype=np.float32)}
        prob = np.array([0.95, 0.95, 0.95, 0.95, 0.1], dtype=np.float32)
        self.assertEqual(leakage_counts(s, prob, 0.9, 6, 8), (3, 1))
        none = {k: s[k] for k in ("win", "y", "x")}                         # 新字典（不带上一次的缓存）
        none["label"] = np.zeros(5, dtype=np.float32)
        self.assertEqual(leakage_counts(none, prob, 0.9, 6, 8), (4, 0))


class ConfigTests(unittest.TestCase):
    def test_lag_configs(self):
        import train_stream_lag as L
        from utils.stream_common import load_flat_config
        a = L.build_config("configs/evisseg_stream_lag.yaml")
        b = L.build_config("configs/evisseg_stream_lag_evflying.yaml")
        la = load_flat_config("configs/evisseg_stream_lag.yaml")
        lb = load_flat_config("configs/evisseg_stream_lag_evflying.yaml")
        self.assertEqual(set(la), set(lb))
        differ = {k for k in la if la[k] != lb[k]}
        self.assertEqual(differ, {"base_config", "backbone_checkpoint", "lag_epochs", "lag_save_root"})
        for cfg, lag in ((a, la), (b, lb)):
            base = load_flat_config(lag["base_config"])
            self.assertEqual(os.path.dirname(lag["backbone_checkpoint"]), base["save_root"])   # 骨干 = 该配置训出的权重
            self.assertEqual(cfg["lag_variants"], ["pixel", "multiscale"])
            self.assertGreaterEqual(cfg["lag_chunk"], cfg["lags"])
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.yaml"
            bad.write_text("LAG:\n  base_config: configs/evisseg_stream_tif_comm_peak.yaml\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                L.build_config(str(bad))


class EndToEndTests(unittest.TestCase):
    def test_backbone_lag_train_eval_publish(self):
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
            import yaml
        except ImportError as error:
            self.skipTest("完整入口需要原 V1 的 cv2/pandas/tqdm 依赖: %s" % error)
        import train_stream_lag as L
        import train_stream_v1 as T
        from tools import lag_publish as P
        from utils.stream_common import load_flat_config
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
            base_path, lag_path = root / "base.yaml", root / "lag.yaml"
            base_path.write_text(yaml.safe_dump({"ALL": cfg}), encoding="utf-8")
            lag = {"base_config": str(base_path), "backbone_checkpoint": str(checkpoint),
                   "lag_variants": ["pixel", "multiscale"], "lags": 2, "lag_hidden": 8, "lag_epochs": 2,
                   "lag_lr": 0.01, "lag_lr_end": 0.001, "lag_grad_clip": 1.0, "lag_streams": 2, "lag_chunk": 4,
                   "lag_eval_chunk": 4, "lag_save_root": str(root / "lag")}
            lag_path.write_text(yaml.safe_dump({"LAG": lag}), encoding="utf-8")
            lcfg = L.build_config(str(lag_path))
            L.mode_smoke(SimpleNamespace(device="cpu"), lcfg)
            L.mode_train(SimpleNamespace(device="cpu"), lcfg)
            for v in ("pixel", "multiscale"):
                self.assertTrue((root / "lag" / ("best_lag_%s.pt" % v)).exists())
            L.mode_eval(SimpleNamespace(device="cpu", split="val", dump=True, reference=str(reference),
                                        overwrite=True), lcfg)
            result = json.loads((root / "lag" / "eval_lag_val.json").read_text(encoding="utf-8"))
            self.assertTrue(all(item["equal"] for item in result["d0_vs_v5"].values()), result["d0_vs_v5"])
            self.assertEqual(set(result["variants"]), {"pixel", "multiscale"})
            self.assertEqual(set(result["variants"]["pixel"]), {"1", "2"})
            dump = root / "lag" / "dump_val_multiscale.npz"
            prob, seqs, meta = P.load_dump(str(dump))
            self.assertEqual(prob.shape[0], 3)
            deadline, margin, selection = P.choose_rule(prob, seqs, meta["threshold"], P.DEFAULT_MARGINS, 0.9)
            cfgp = {k: meta[k] for k in ("threshold", "pd_detT", "correct_thresh", "height", "width_px")}
            report = P.apply_rule(prob, seqs, cfgp, deadline, margin)
            self.assertGreaterEqual(report["iou"], selection["target_iou"] - 1e-12)
            zero = P.apply_rule(prob, seqs, cfgp, deadline, 0.0)                 # m = 0 即 d = 0（V5）
            self.assertEqual((zero["iou"], zero["acc"], zero["pd"], zero["fa"]),
                             tuple(result["d0"][k] for k in ("iou", "acc", "pd", "fa")))


if __name__ == "__main__":
    unittest.main()
