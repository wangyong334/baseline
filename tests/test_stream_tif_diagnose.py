"""tools/diagnose_tif.py 的单元测试（CPU，不需要数据集）。

    1. 膨胀、近邻计数与暴力计算一致
    2. 手工序列：FP 三类（near / trail / far，含回看窗数）、目标速度分层、FN 计数
    3. 层内钩子：挂上与不挂输出逐位相同、与 run_sequence 相同；各区域占比之和为 1
    4. 入口：两个 checkpoint（TIF 量化 / 不量化）在合成数据上跑完，事件级计数与 run_sequence 相同，
       逐序列消息率与神经元统计一致，JSON 合法
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import stream_windows as sw  # noqa: E402
from model.evspsegnet_stream import LAYER_NAMES  # noqa: E402
from test_stream_layerwise import CFG, H, Q99, SMALL, W, WINDOWS, synthetic_sequence  # noqa: E402
from test_stream_tif import make_tif_net  # noqa: E402
from tools import diagnose_tif as D  # noqa: E402
from train_stream_v1 import build_net, run_sequence  # noqa: E402
from utils.stream_common import load_flat_config  # noqa: E402


def chebyshev_near(points, queries, radius):
    """暴力：每个查询点是否距某个点的切比雪夫距离 <= radius。"""
    if not len(points):
        return np.zeros(len(queries), dtype=bool)
    p = np.asarray(points)
    return np.array([bool((np.maximum(abs(p[:, 0] - qx), abs(p[:, 1] - qy)) <= radius).any()) for qx, qy in queries])


class GeometryTests(unittest.TestCase):
    def test_dilate_matches_brute_force(self):
        rng = np.random.RandomState(0)
        mask = rng.rand(9, 11) < 0.08
        ys, xs = np.nonzero(mask)
        grid = [(x, y) for y in range(9) for x in range(11)]
        for radius in range(4):
            expected = chebyshev_near(list(zip(xs, ys)), grid, radius).reshape(9, 11)
            self.assertTrue(np.array_equal(D.dilate(mask, radius), expected), radius)

    def test_count_near_matches_brute_force(self):
        rng = np.random.RandomState(1)
        tx, ty = rng.randint(5, 15, 6), rng.randint(5, 15, 6)
        fx, fy = rng.randint(0, 25, 80), rng.randint(0, 25, 80)
        for radius in (0, 1, 3):
            expected = int(chebyshev_near(list(zip(tx, ty)), list(zip(fx, fy)), radius).sum())
            self.assertEqual(D.count_near(fx, fy, tx, ty, radius), expected)
        self.assertEqual(D.count_near(fx[:0], fy[:0], tx, ty, 3), 0)


def handcrafted_sequence():
    """6 窗；目标 1 每窗向右 2 像素（两像素宽，y=8），另有三个 FP 与一个 FN。

    返回 (seq, probs)：
        窗 2 的一个正事件预测为负（FN）
        窗 3 的 FP 在 x=11：距本窗目标（x=8,9）2 像素 -> near
        窗 4 的 FP 在 x=2：距本窗目标（x=10,11）8 像素；窗 0 目标在 x=2,3 -> 回看 >= 4 窗时为 trail，回看 1 窗时为 far
        窗 5 的 FP 在 (22, 1)：远离一切 -> far
    """
    x, y, t, label, prob = [], [], [], [], []
    for k in range(6):
        for dx in (0, 1):
            x.append(2 + 2 * k + dx), y.append(8), t.append(50 * k + 10 + dx), label.append(1.0)
            prob.append(0.1 if (k == 2 and dx == 0) else 0.95)
        x.append(20), y.append(14), t.append(50 * k + 30), label.append(0.0), prob.append(0.2)   # 判对的负事件
    for k, fx, fy in ((3, 11, 8), (4, 2, 8), (5, 22, 1)):
        x.append(fx), y.append(fy), t.append(50 * k + 40), label.append(0.0), prob.append(0.95)
    x, y, t = (np.asarray(v, dtype=np.int64) for v in (x, y, t))
    label = np.asarray(label, dtype=np.float32)
    order, bounds = sw.split_windows(t, 50, 6)
    bins, local = sw.time_bin_and_local(t, 50, 5)
    seq = sw.StreamSequence("hand.npz", x, y, t, np.ones(t.size, dtype=np.int8), label, label.astype(np.float64),
                            bins, local, order, bounds)
    return seq, np.asarray(prob, dtype=np.float32)


class EventLevelTests(unittest.TestCase):
    def test_fp_kinds_speed_and_counts(self):
        seq, probs = handcrafted_sequence()
        r = D.analyze_sequence(seq, probs, 0.9, 16, 24, near_px=3, trail_windows=10)
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (11, 3, 1))
        self.assertEqual((r["fp_near"], r["fp_trail"], r["fp_far"]), (1, 1, 1))
        self.assertEqual(r["neg_per_window"], 9 / 6.0)
        self.assertEqual(r["pos_without_target"], 0)
        speeds = [rec["speed"] for rec in r["targets"]]
        self.assertIsNone(speeds[0])
        for k, s in enumerate(speeds[1:], 1):
            # 窗 2 的两个正事件都在（FN 也是正事件），质心照常前移 2 像素
            self.assertAlmostEqual(s, 2.0, msg=str(k))
        self.assertEqual([rec["fp_near"] for rec in r["targets"]], [0, 0, 0, 1, 0, 0])
        table = {row["bin"]: row for row in D.speed_table(r["targets"])}
        self.assertEqual((table["new"]["target_windows"], table["2-4"]["target_windows"]), (1, 5))
        self.assertEqual((table["2-4"]["tp"], table["2-4"]["fn"], table["2-4"]["fp_near"]), (9, 1, 1))
        self.assertAlmostEqual(table["2-4"]["recall"], 0.9)
        self.assertAlmostEqual(table["2-4"]["local_iou"], 9 / 11.0)
        self.assertIsNone(table["0-0.5"]["recall"])
        # 回看 1 窗：窗 4 的 FP 只与窗 3 的目标比较（x=8,9，距离 6），变成 far
        r1 = D.analyze_sequence(seq, probs, 0.9, 16, 24, near_px=3, trail_windows=1)
        self.assertEqual((r1["fp_near"], r1["fp_trail"], r1["fp_far"]), (1, 0, 2))

    def test_paired_and_spearman(self):
        a = [{"name": "s%d" % i, "neg_per_window": float(i), "pos_per_window": 1.0, "iou": 0.5, "tp": 1,
              "fp": 10 + i, "fn": 2, "fp_near": 1, "fp_trail": 0, "fp_far": 9 + i} for i in range(4)]
        b = [dict(s, fp=10, fp_far=9, iou=0.6) for s in a]
        p = D.paired("A", a, "B", b)
        self.assertEqual(p["total"]["d_fp"], 6)
        self.assertEqual(p["total"]["d_fp_far"], 6)
        self.assertEqual(p["sequences"][0]["name"], "s3")
        self.assertAlmostEqual(p["spearman_vs_neg_per_window"]["d_fp"], 1.0)
        self.assertIsNone(p["spearman_vs_neg_per_window"]["d_fn"])          # 方差为 0
        self.assertAlmostEqual(p["top3_share_of_positive_d_fp"], 1.0)


class ProbeTests(unittest.TestCase):
    def test_probe_is_read_only_and_matches_neuron_stats(self):
        seq = synthetic_sequence(seed=2)
        cfg = dict(CFG, input_device="cpu", execution="step")
        device = torch.device("cpu")
        net = make_tif_net().eval()
        with torch.no_grad():
            _, confusion = run_sequence(net, seq, cfg, Q99, device, "carry")
            plain = D.run_sequence_probe(net, seq, cfg, Q99, device, D.TargetRegions(4))
            regions = D.TargetRegions(4)
            probe = D.LayerProbe(net, regions)
            probed = D.run_sequence_probe(net, seq, cfg, Q99, device, regions)
            probe.remove()
        self.assertFalse(net.enc1.neuron._forward_hooks)
        self.assertTrue(np.array_equal(plain, probed))
        record = D.analyze_sequence(seq, probed, cfg["threshold"], H, W, 1, 4)
        self.assertEqual([record["tp"], record["fp"], record["fn"]], confusion.sum(axis=0)[:3].tolist())
        summary = probe.summary()
        self.assertEqual(set(summary), set(LAYER_NAMES))
        for name in LAYER_NAMES:
            s = summary[name]
            self.assertAlmostEqual(sum(s["memory_share"].values()), 1.0, places=9)
            self.assertAlmostEqual(sum(s["area_share"].values()), 1.0, places=9)
            self.assertGreater(s["area_share"]["now"], 0.0)
            self.assertGreater(s["area_share"]["trail"], 0.0)


class ErrorDiagnosisTests(unittest.TestCase):
    def test_auc_and_best_threshold_match_brute_force(self):
        rng = np.random.RandomState(0)
        scores = np.round(rng.rand(200), 2).astype(np.float32)          # 有并列
        labels = (rng.rand(200) < 0.3).astype(np.float32)
        pos, neg = scores[labels == 1], scores[labels == 0]
        brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
        self.assertAlmostEqual(D.rank_auc(scores, labels), brute, places=10)
        best, thr = D.best_threshold_iou(scores, labels)
        ious = []
        for t in np.unique(scores):
            pred = scores >= t
            tp = int((pred & (labels == 1)).sum())
            ious.append((tp / float(int(pred.sum()) + int(((~pred) & (labels == 1)).sum())), float(t)))
        self.assertAlmostEqual(best, max(ious)[0], places=10)
        self.assertAlmostEqual(thr, max(ious, key=lambda v: (v[0], -v[1]))[1], places=6)
        self.assertIsNone(D.rank_auc(scores, np.zeros(200)))
        self.assertEqual(D.best_threshold_iou(scores, np.zeros(200)), (None, None))

    def test_local_density_matches_brute_force(self):
        rng = np.random.RandomState(1)
        xs, ys = rng.randint(0, 9, 60), rng.randint(0, 7, 60)
        got = D.local_density(xs, ys, 7, 9)
        brute = [int(((abs(xs - x) <= 2) & (abs(ys - y) <= 2)).sum()) for x, y in zip(xs, ys)]
        self.assertEqual(got.tolist(), brute)

    def test_sequence_record_totals_and_layer_saturation(self):
        seq = synthetic_sequence(seed=2)
        cfg = dict(CFG, input_device="cpu", execution="step")
        device = torch.device("cpu")
        net = make_tif_net().eval()
        regions = D.TargetRegions(4)
        probe = D.LayerProbe(net, regions)
        with torch.no_grad():
            probe.start_sequence()
            probs = D.run_sequence_probe(net, seq, cfg, Q99, device, regions)
        record = D.analyze_sequence(seq, probs, cfg["threshold"], H, W, 1, 4)
        layers = probe.sequence_summary()
        probe.remove()
        density = np.asarray(record["density"])
        self.assertEqual(density[:, :3].sum(axis=0).tolist(), [record["tp"], record["fp"], record["fn"]])
        self.assertEqual(int(density[:, 3].sum()), int((seq.label != 1).sum()))
        windows = np.asarray(record["per_window"])
        self.assertEqual(windows.shape, (seq.n_windows, 4))
        self.assertEqual(windows[:, 1:].sum(axis=0).tolist(), [record["tp"], record["fp"], record["fn"]])
        self.assertLessEqual(record["iou"], record["best_iou"] + 1e-12)
        for name in LAYER_NAMES:
            s = layers[name]
            self.assertEqual(len(s["bg_series"]), seq.n_windows)
            for region in D.REGIONS:
                frac = s["saturated_fraction"][region]
                self.assertTrue(frac is None or 0.0 <= frac <= 1.0)
            for region in D.REGIONS:            # 小画面的粗层上背景面积可能为 0（None）
                level = s["mean_level"][region]
                self.assertTrue(level is None or 0.0 <= level <= getattr(net, name).neuron.levels + 1e-6)
        self.assertEqual(set(probe.summary()), set(LAYER_NAMES))


class EntryTests(unittest.TestCase):
    def test_main_two_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "val").mkdir()
            for seed in (6, 7):
                seq = synthetic_sequence(seed=seed)
                loc = np.stack((seq.x, seq.y, seq.t), axis=1)
                norm = np.column_stack((loc, seq.p, seq.label, seq.target_id))
                np.savez(root / "val" / ("seq_%d.npz" % seed), ev_loc=loc, evs_norm=norm)
            base = load_flat_config("configs/evisseg_stream_tif_notransport.yaml")
            base.update(root=str(root), channels=list(SMALL), height=H, width_px=W, pad_height=H, pad_width=W,
                        n_windows=WINDOWS, threshold=0.5)
            paths = {}
            for name, quantize in (("tifq", True), ("tif", False)):
                cfg = dict(base, tif=dict(base["tif"], quantize=quantize, k_init=1e-3))
                torch.manual_seed(3)
                net = build_net(cfg, torch.device("cpu"))
                for block in net.blocks():
                    block.gain.set_gain(torch.full((block.out_ch,), 4.0))
                net.gain_calibrated.fill_(True)
                paths[name] = root / ("%s.pt" % name)
                torch.save({"config": cfg, "model": net.state_dict(), "stats": {"q99": Q99.tolist()}, "epoch": 1},
                           paths[name])
            yaml_path = root / "eval.yaml"
            yaml_path.write_text(json.dumps({"DATA": {"root": str(root)}, "EVAL": {
                key: base[key] for key in D.EVAL_KEYS}}), encoding="utf-8")       # JSON 是合法的 YAML
            out = root / "diag.json"
            argv = ["--config", str(yaml_path), "--split", "val", "--device", "cpu", "--near-px", "1",
                    "--trail-windows", "3", "--out", str(out),
                    "--run", "tifq=%s" % paths["tifq"], "--run", "tif=%s" % paths["tif"]]
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                report = D.main(argv)
                with self.assertRaises(RuntimeError):
                    D.main(argv)                                                  # 不覆盖已有结果
            saved = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(saved["paired"]["a"], "tifq")
            self.assertEqual(len(saved["runs"]["tifq"]["sequences"]), 2)
            self.assertEqual(set(saved["runs"]["tifq"]["layers"]), set(LAYER_NAMES))
            self.assertEqual(set(saved["runs"]["tif"]["message_vs_noise"]), set(LAYER_NAMES) | {"mean"})
            for name in ("tifq", "tif"):
                net, train_cfg, q99, _ = D.load_run(str(paths[name]), dict(base, threshold=0.5), torch.device("cpu"))
                total = np.zeros(3, dtype=np.int64)
                for i, seed in enumerate((6, 7)):
                    seq = sw.load_npz_events(str(root / "val" / ("seq_%d.npz" % seed)), H, W, 50, WINDOWS, 5)
                    with torch.no_grad():
                        net.start_tif_stats()
                        _, confusion = run_sequence(net, seq, train_cfg, q99, torch.device("cpu"), "carry")
                        stats = net.stop_tif_stats()
                    total += confusion.sum(axis=0)[:3]
                    rates = report["runs"][name]["sequences"][i]["message_abs_rate"]
                    for layer in LAYER_NAMES:
                        self.assertAlmostEqual(rates[layer], stats[layer]["message_abs_rate"], places=12)
                t = report["runs"][name]["totals"]
                self.assertEqual([t["tp"], t["fp"], t["fn"]], total.tolist(), name)


if __name__ == "__main__":
    unittest.main()
