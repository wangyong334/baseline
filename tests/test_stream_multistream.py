"""标准有状态 TBPTT（trainer=streams）的测试（CPU、float64 小网络、合成序列，不需要数据集）。

    1. keep 掩码：某条流在第 t 窗清零状态，等价于该流从 None 重新开始；其他流不受影响（LIF 与双仓）
    2. 多流批次的 forward_chunk 与逐条流单独运行逐位相同（含录像在片段中途结束、下一段从头开始）
    3. 录像分配与片段拆分：每条录像的每个窗恰好出现一次，新录像起点恰好清零一次，各流长度接近
    4. B=1、n=1 时，多流训练与原 per_chunk 逐段更新的参数轨迹逐位相同
    5. 每 n 个片段更新一次
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_stream_v1 as T  # noqa: E402
from model.lif2d_stream import StreamingDualMemoryLIF2d, StreamingLIF2d  # noqa: E402
from test_stream_layerwise import CFG, Q99, WINDOWS, make_net, synthetic_sequence  # noqa: E402

CPU = torch.device("cpu")
DTYPE = torch.float64
POS_WEIGHT = 4.0


def max_error(a, b):
    return float((a - b).abs().max()) if a.numel() else 0.0


class KeepMaskTests(unittest.TestCase):
    def check_neuron(self, neuron, start):
        current = (torch.rand(6, 2, 3, 4, 5, dtype=DTYPE) * 2).requires_grad_()
        keep = torch.ones(6, 2, dtype=DTYPE)
        keep[3, 1] = 0                                     # 第 1 条流在第 3 窗开始新录像
        spikes, final, u_pre = neuron.forward_steps(current, start, keep=keep)[:3]
        # 参照：第 0 条流连续；第 1 条流前 3 窗接着 start，后 3 窗从 None 开始
        s0, f0, u0 = neuron.forward_steps(current[:, :1], select(start, slice(0, 1)))[:3]
        s1a, _, u1a = neuron.forward_steps(current[:3, 1:], select(start, slice(1, 2)))[:3]
        s1b, f1b, u1b = neuron.forward_steps(current[3:, 1:], None)[:3]
        self.assertLess(max_error(u_pre[:, :1], u0), 1e-12)
        self.assertLess(max_error(u_pre[:, 1:], torch.cat([u1a, u1b])), 1e-12)
        self.assertLess(max_error(spikes[:, 1:], torch.cat([s1a, s1b])), 1e-12)
        for got, want in zip(tensors(select(final, slice(1, 2))), tensors(f1b)):
            self.assertLess(max_error(got, want), 1e-12)
        for got, want in zip(tensors(select(final, slice(0, 1))), tensors(f0)):
            self.assertLess(max_error(got, want), 1e-12)
        # 梯度在清零处截断：第 1 条流后 3 窗的输出与它前 3 窗的输入无关
        grad = torch.autograd.grad(u_pre[3:, 1].sum(), current)[0]
        self.assertEqual(float(grad[:3, 1].abs().sum()), 0.0)
        self.assertGreater(float(grad[3:, 1].abs().sum()), 0.0)

    def test_lif(self):
        torch.manual_seed(0)
        neuron = StreamingLIF2d(3, 50.0, 200.0, 50.0, 2000.0, 1.0).double()
        self.check_neuron(neuron, torch.rand(2, 3, 4, 5, dtype=DTYPE))

    def test_dual_memory(self):
        torch.manual_seed(1)
        neuron = StreamingDualMemoryLIF2d(3, 50.0, 200.0, 50.0, 2000.0, 1.0).double()
        neuron.kappa_raw.data.uniform_(-1, 1)              # 让旧 M 真的参与 U 的递推
        start = (torch.rand(2, 3, 4, 5, dtype=DTYPE), torch.rand(2, 3, 4, 5, dtype=DTYPE) * 0.5)
        self.check_neuron(neuron, start)


def select(state, index):
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(s[index] for s in state)
    return state[index]


def tensors(state):
    return list(state) if isinstance(state, tuple) else [state]


class MultiStreamChunkTests(unittest.TestCase):
    def test_batch_equals_separate_streams(self):
        dataset = [synthetic_sequence(seed) for seed in (0, 1, 2)]
        sources = T.StreamSources(dataset, CFG, Q99, CPU, DTYPE)
        # 流 0：录像 0 的第 0..5 窗；流 1：录像 1 的第 0..2 窗，接着录像 2 从第 0 窗开始（片段中途清零）
        pieces = [[(0, 0, 6, 0, True)], [(1, 0, 3, 0, True), (2, 0, 3, 3, True)]]
        x, events, labels, keep = T.assemble_streams(sources, pieces, 6, CFG, CPU, DTYPE)
        self.assertEqual(tuple(x.shape), (6, 2, 12, CFG["pad_height"], CFG["pad_width"]))
        self.assertEqual(keep[:, 1].tolist(), [0, 1, 1, 0, 1, 1])
        net = make_net()
        logits, states, _ = net.forward_chunk(x, events, None, keep=keep)
        reference_logits, finals = [], []
        for index, lo, hi in ((0, 0, 6), (1, 0, 3), (2, 0, 3)):
            xs, ev, lab, _ = sources.get(index).chunk(lo, hi)
            out, final, _ = net.forward_chunk(xs, ev, None)
            reference_logits.append(out)
            finals.append(final)
        self.assertLess(max_error(logits, torch.cat(reference_logits)), 1e-10)
        self.assertTrue(torch.equal(labels, torch.cat([sources.get(i).chunk(0, h)[2] for i, h in ((0, 6), (1, 3), (2, 3))])))
        for layer in range(7):
            self.assertLess(max_error(states[layer][0:1], finals[0][layer]), 1e-10)
            self.assertLess(max_error(states[layer][1:2], finals[2][layer]), 1e-10)


class PlanTests(unittest.TestCase):
    def test_every_window_once_and_balanced(self):
        lengths, k = [12, 7, 30, 5, 18, 9], 4
        plan = T.plan_streams(lengths, 2, k, np.random.RandomState(0))
        self.assertEqual(plan, T.plan_streams(lengths, 2, k, np.random.RandomState(0)))   # 同种子可复现
        self.assertEqual(sorted(i for _, items in plan for i, _ in items), list(range(len(lengths))))
        self.assertTrue(all(0 <= delay < k for delay, _ in plan))
        totals = [sum(n for _, n in items) for _, items in plan]
        self.assertLessEqual(max(totals) - min(totals), max(lengths))
        seen, starts = {}, {}
        total = T.stream_total_windows(plan)
        for c in range(-(-total // k)):
            for stream in T.stream_pieces(plan, c * k, (c + 1) * k):
                previous_end = 0
                for index, lo, hi, offset, new_start in stream:
                    self.assertTrue(0 <= offset < k and offset + (hi - lo) <= k)
                    self.assertGreaterEqual(offset, previous_end)
                    previous_end = offset + (hi - lo)
                    for w in range(lo, hi):
                        seen[(index, w)] = seen.get((index, w), 0) + 1
                    if new_start:
                        self.assertEqual(lo, 0)
                        starts[index] = starts.get(index, 0) + 1
        self.assertEqual(seen, {(i, w): 1 for i, n in enumerate(lengths) for w in range(n)})
        self.assertEqual(starts, {i: 1 for i in range(len(lengths))})


class TrainStreamsTests(unittest.TestCase):
    def cfg(self, **kw):
        base = dict(CFG, execution="layer", tbptt_k=4, streams=1, accumulate=1)
        base.update(kw)
        return base

    def test_single_stream_equals_per_chunk_training(self):
        self.assertEqual(WINDOWS % 4, 0)                    # 录像长度是 k 的整数倍：片段不跨录像
        dataset = [synthetic_sequence(seed) for seed in (0, 1)]
        lengths = [seq.n_windows for seq in dataset]
        net_a, net_b = make_net(), make_net()
        opt_a = torch.optim.SGD(net_a.parameters(), lr=0.05)
        opt_b = torch.optim.SGD(net_b.parameters(), lr=0.05)
        out = T.train_streams(net_a, dataset, lengths, opt_a, self.cfg(), Q99, POS_WEIGHT, CPU, None,
                              plan=[(0, [(0, lengths[0]), (1, lengths[1])])])
        steps = 0
        for seq in dataset:
            steps += T.train_sequence(net_b, seq, opt_b, self.cfg(optimizer_step="per_chunk"), Q99, POS_WEIGHT,
                                      CPU, None, first_len=4)["steps"]
        self.assertEqual(out["steps"], steps)
        self.assertEqual(out["chunks"], sum(lengths) // 4)
        for (name, pa), (_, pb) in zip(net_a.named_parameters(), net_b.named_parameters()):
            self.assertLess(max_error(pa.detach(), pb.detach()), 1e-12, name)

    def test_single_stream_block_equals_per_sequence(self):
        # B=1、n = 录像窗数 / k：一次更新恰好是一整条录像，按整条录像事件数归一 -> 与原 per_sequence 逐位相同
        dataset = [synthetic_sequence(seed) for seed in (0, 1)]
        lengths = [seq.n_windows for seq in dataset]
        net_a, net_b = make_net(), make_net()
        opt_a = torch.optim.SGD(net_a.parameters(), lr=0.05)
        opt_b = torch.optim.SGD(net_b.parameters(), lr=0.05)
        out = T.train_streams(net_a, dataset, lengths, opt_a, self.cfg(accumulate=WINDOWS // 4), Q99, POS_WEIGHT,
                              CPU, None, plan=[(0, [(0, lengths[0]), (1, lengths[1])])])
        for seq in dataset:
            T.train_sequence(net_b, seq, opt_b, self.cfg(optimizer_step="per_sequence"), Q99, POS_WEIGHT,
                             CPU, None, first_len=4)
        self.assertEqual(out["steps"], len(dataset))
        for (name, pa), (_, pb) in zip(net_a.named_parameters(), net_b.named_parameters()):
            self.assertLess(max_error(pa.detach(), pb.detach()), 1e-12, name)

    def test_mid_chunk_boundary_gradient_equals_separate_parts(self):
        # 一个片段里录像 0 在第 2 窗结束、录像 1 从第 2 窗开始：整片段的梯度 = 两段分开跑（后一段从 None 开始）的梯度之和
        dataset = [synthetic_sequence(seed) for seed in (0, 1)]
        sources = T.StreamSources(dataset, CFG, Q99, CPU, DTYPE)
        weight = torch.tensor([POS_WEIGHT], dtype=DTYPE)
        net = make_net()
        with torch.no_grad():
            xs, ev, _, _ = sources.get(0).chunk(0, 10)
            _, start, _ = net.forward_chunk(xs, ev, None)          # 录像 0 前 10 窗之后的状态

        def grads(loss):
            net.zero_grad()
            loss.backward()
            return [None if p.grad is None else p.grad.clone() for p in net.parameters()]

        pieces = [[(0, 10, 12, 0, False), (1, 0, 2, 2, True)]]
        x, events, labels, keep = T.assemble_streams(sources, pieces, 4, CFG, CPU, DTYPE)
        logits, _, _ = net.forward_chunk(x, events, start, keep=keep)
        joint = grads(torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=weight, reduction="sum"))
        xa, eva, la, _ = sources.get(0).chunk(10, 12)
        xb, evb, lb, _ = sources.get(1).chunk(0, 2)
        out_a, _, _ = net.forward_chunk(xa, eva, start)
        out_b, _, _ = net.forward_chunk(xb, evb, None)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(out_a, la, pos_weight=weight, reduction="sum")
                + torch.nn.functional.binary_cross_entropy_with_logits(out_b, lb, pos_weight=weight, reduction="sum"))
        separate = grads(loss)
        for got, want in zip(joint, separate):
            self.assertEqual(got is None, want is None)
            if got is not None:
                self.assertLess(max_error(got, want), 1e-10)

    def test_accumulate_and_multiple_streams(self):
        dataset = [synthetic_sequence(seed) for seed in (0, 1, 2)]
        lengths = [seq.n_windows for seq in dataset]
        net = make_net()
        before = [p.detach().clone() for p in net.parameters()]
        opt = torch.optim.SGD(net.parameters(), lr=0.05)
        plan = [(0, [(0, 12)]), (1, [(1, 12), (2, 12)])]           # 两条流，第二条延迟 1 窗
        out = T.train_streams(net, dataset, lengths, opt, self.cfg(streams=2, accumulate=2), Q99, POS_WEIGHT,
                              CPU, None, plan=plan)
        self.assertEqual(out["chunks"], 7)                          # ceil((1 + 24) / 4)
        self.assertEqual(out["steps"], 4)                           # 每 2 个片段一次，最后不足 2 个也更新
        self.assertEqual(out["events"], sum(seq.n_events for seq in dataset))
        self.assertTrue(any(not torch.equal(a, b.detach()) for a, b in zip(before, net.parameters())))
        self.assertEqual(out["missing_grads"], [])

    def test_mode_train_end_to_end(self):
        # 文件读取、校准、一轮多流训练、验证、checkpoint、metrics 记录；临时目录中的小尺寸合成数据，不需要 GPU。
        try:
            import cv2  # noqa: F401
            import pandas  # noqa: F401
            import tqdm  # noqa: F401
        except ImportError as error:
            self.skipTest("完整训练入口需要原 V1 的 cv2/pandas/tqdm 依赖: %s" % error)
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from test_stream_layerwise import H, SMALL, W
        from utils.stream_common import load_flat_config
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pos = neg = 0
            for split, seeds in (("train", (3, 4, 5)), ("val", (6,))):
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
            cfg = load_flat_config("configs/evisseg_stream_v1.yaml")
            cfg.update(root=str(root), stats_path=str(root / "stats.json"), save_root=str(root / "run"),
                       channels=list(SMALL), height=H, width_px=W, pad_height=H, pad_width=W,
                       n_windows=12, epochs=2, num_workers=0, tbptt_k=4, train_subset_size=1,
                       calib_sequences=1, calib_windows=8, calib_samples_per_channel=32, calib_min_positive=4,
                       segments=[[0, 3], [4, 7], [8, 11]], timing_warmup_windows=1, eval_chunk=4,
                       execution="layer", input_device="gpu", trainer="streams", streams=2, accumulate=1)
            T.mode_train(SimpleNamespace(device="cpu", mode="train", resume=False), cfg)
            record = json.loads((root / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual((record["trainer"], record["streams"], record["accumulate"]), ("streams", 2, 1))
            self.assertGreater(record["updates"], 0)
            self.assertIn(record["updates"], (6, 7))   # 3 条 12 窗录像分到 2 条流：较长一条 24 窗 + 延迟 0..3 窗 -> 6 或 7 个片段
            self.assertTrue((root / "run" / "best_val_iou_seed37.pt").exists())

    def test_requires_layer_execution(self):
        dataset = [synthetic_sequence(0)]
        with self.assertRaises(ValueError):
            T.train_streams(make_net(), dataset, [12], torch.optim.SGD(make_net().parameters(), lr=0.1),
                            dict(CFG, execution="step", streams=1, accumulate=1), Q99, POS_WEIGHT, CPU,
                            np.random.RandomState(0))


if __name__ == "__main__":
    unittest.main()
