"""V5 numerical regression retained for the V15 release; legacy variants removed."""
import os


import sys


import unittest


import numpy as np


import torch


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import train_stream_v1 as T
from test_stream_layerwise import CFG, Q99, synthetic_sequence
from test_stream_tif import make_tif_net, tif_neuron


CPU = torch.device("cpu")


DTYPE = torch.float64


POS_WEIGHT = 4.0


def max_error(a, b):
    return float((a - b).abs().max()) if a.numel() else 0.0


class KeepMaskTests(unittest.TestCase):
    def test_tif_keep_boundary_stops_history_and_gradient(self):
        torch.manual_seed(0)
        neuron = tif_neuron()
        self.check_neuron(neuron, torch.rand(2, 3, 4, 5, dtype=DTYPE))


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
        # V5 layer-parallel BIR uses checkpoint recomputation on torch 1.9.
        # Its backward API deliberately does not support autograd.grad().
        u_pre[3:, 1].sum().backward()
        grad = current.grad
        self.assertEqual(float(grad[:3, 1].abs().sum()), 0.0)
        self.assertGreater(float(grad[3:, 1].abs().sum()), 0.0)


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
        net = make_tif_net()
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

    def test_unknown_layout_rejected(self):
        with self.assertRaises(ValueError):
            T.plan_streams([4, 4], 2, 4, np.random.RandomState(0), "round_robin")


if __name__ == "__main__":
    unittest.main()
