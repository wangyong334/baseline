"""V20 past-information ablation (11-14): segment resets (memory_reset_every) and steps per window (window_steps).

Checked on the small float64 network of tests/test_stream_layerwise.py (CPU, no dataset):
    1. defaults reproduce the original paths exactly; memory_reset_every = 1 equals state_mode reset_each_window;
    2. step execution, layer execution (training-time validation) and window_forward (prior export) agree;
    3. segment resets equal a hand-written reference (state cleared before windows L, 2L, ...);
    4. window_steps = T equals running the same input T times and reading out after the last step;
    5. the training batch (assemble_streams + expand_window_steps) clears the state where the inference paths do,
       and one stream trained on it gives the same forward as the inference path.
"""
import unittest

import numpy as np
import torch

import train_stream_v1 as T
from dataset import stream_windows as sw
from dataset.stream_source import make_window_source
from tests.test_stream_layerwise import CFG, Q99, WINDOWS, make_net, synthetic_sequence

DEV = torch.device("cpu")


def run(net, seq, cfg, mode="carry", **kw):
    with torch.no_grad():
        return T.run_sequence(net, seq, cfg, Q99, DEV, mode, **kw)


class MemoryOptionTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(T.memory_options({}), (0, 1))
        self.assertEqual(T.memory_options({"memory_reset_every": 4, "window_steps": 3}), (4, 3))
        for bad in ({"memory_reset_every": -1}, {"window_steps": 0},
                    {"window_steps": 2, "state_mode": "reset_each_window"}):
            with self.assertRaises(ValueError):
                T.memory_options(bad)
        self.assertFalse(T.resets_before(0, 1))
        self.assertEqual([k for k in range(10) if T.resets_before(k, 4)], [4, 8])
        self.assertEqual([k for k in range(10) if T.resets_before(k, 0, 5, 2)], [2, 7])

    def test_defaults_and_reset1(self):
        seq = synthetic_sequence(seed=5)
        net = make_net(seed=17).eval()
        for execution in ("step", "layer"):
            base = dict(CFG, execution=execution, input_device="cpu")
            plain = run(net, seq, base)
            explicit = run(net, seq, dict(base, memory_reset_every=0, window_steps=1))
            self.assertTrue(np.array_equal(plain[0], explicit[0]) and np.array_equal(plain[1], explicit[1]))
            reset = run(net, seq, base, "reset_each_window")
            every1 = run(net, seq, dict(base, memory_reset_every=1))
            self.assertLess(float(np.abs(every1[0] - reset[0]).max()), 1e-9, execution)
            self.assertTrue(np.array_equal(every1[1], reset[1]))

    def test_segments_match_reference_in_both_executions(self):
        seq = synthetic_sequence(seed=6)
        net = make_net(seed=19).eval()
        cfg = dict(CFG, memory_reset_every=4)
        step = run(net, seq, dict(cfg, execution="step", input_device="cpu"))
        layer = run(net, seq, dict(cfg, execution="layer", input_device="cpu"))
        source = make_window_source(seq, dict(CFG, execution="step"), Q99, DEV, torch.float64)
        states, parts, probs = None, [], []
        with torch.no_grad():
            for k in range(seq.n_windows):
                if k in (4, 8):
                    states = None
                x, events, _, idx = source.window(k)
                logits, states, _ = net(x, events, states, state_mode="carry")
                parts.append(idx)
                probs.append(torch.sigmoid(logits).float().numpy())
        reference = sw.refill_by_index(seq.n_events, parts, probs)
        self.assertLess(float(np.abs(step[0] - reference).max()), 1e-6)
        self.assertLess(float(np.abs(layer[0] - reference).max()), 1e-6)
        self.assertTrue(np.array_equal(step[1], layer[1]))
        carry = run(net, seq, dict(CFG, execution="step"))
        self.assertGreater(float(np.abs(step[0] - carry[0]).max()), 0.0)

    def test_window_steps_match_reference(self):
        seq = synthetic_sequence(seed=7)
        for neuron in ("lif", "tif"):
            net = make_net(seed=23, neuron=neuron).eval()
            cfg = dict(CFG, memory_reset_every=1, window_steps=3)
            step = run(net, seq, dict(cfg, execution="step", input_device="cpu"))
            layer = run(net, seq, dict(cfg, execution="layer", input_device="cpu"))
            source = make_window_source(seq, dict(CFG, execution="step"), Q99, DEV, torch.float64)
            parts, probs = [], []
            with torch.no_grad():
                for k in range(seq.n_windows):
                    x, events, _, idx = source.window(k)
                    states = None
                    for _ in range(3):                      # same input three times, read out after the last step
                        logits, states, _ = net(x, events, states, state_mode="carry")
                    parts.append(idx)
                    probs.append(torch.sigmoid(logits).float().numpy())
            reference = sw.refill_by_index(seq.n_events, parts, probs)
            self.assertLess(float(np.abs(step[0] - reference).max()), 1e-6, neuron)
            self.assertLess(float(np.abs(layer[0] - reference).max()), 1e-6, neuron)
            if neuron == "lif":                             # (the small BIR net is nearly silent at these gains)
                one = run(net, seq, dict(CFG, memory_reset_every=1, execution="step"))
                self.assertGreater(float(np.abs(step[0] - one[0]).max()), 0.0, neuron)

    def test_window_forward_is_the_step_path(self):
        seq = synthetic_sequence(seed=8)
        net = make_net(seed=29).eval()
        for extra in ({}, {"memory_reset_every": 4}, {"memory_reset_every": 1, "window_steps": 2}):
            cfg = dict(CFG, execution="step", input_device="cpu", **extra)
            probs, _ = run(net, seq, cfg)
            source = make_window_source(seq, cfg, Q99, DEV, torch.float64)
            states, parts, out = None, [], []
            with torch.no_grad():
                for k in range(seq.n_windows):
                    x, events, _, idx = source.window(k)
                    logits, states, _ = T.window_forward(net, x, events, states, cfg, k, "carry")
                    parts.append(idx)
                    out.append(torch.sigmoid(logits).float().numpy())
            self.assertTrue(np.array_equal(sw.refill_by_index(seq.n_events, parts, out), probs), extra)


class TrainingBatchTests(unittest.TestCase):
    def batch(self, cfg, steps=12):
        seq = synthetic_sequence(seed=9)

        class Data(object):
            def __getitem__(self, i):
                return seq

        sources = T.StreamSources(Data(), dict(CFG, input_device="gpu", **cfg), Q99, DEV, torch.float64)
        pieces = [[(0, 0, WINDOWS, 0, True)]]               # one stream: the whole recording from window 0
        x, events, labels, keep = T.assemble_streams(sources, pieces, steps, dict(CFG, **cfg), DEV, torch.float64)
        valid = T.stream_valid(pieces, steps, DEV, torch.float64)
        return seq, x, events, labels, keep, valid

    def test_keep_marks_segment_starts(self):
        _, _, _, _, keep, _ = self.batch({"memory_reset_every": 4})
        self.assertEqual([t for t in range(WINDOWS) if float(keep[t, 0]) == 0.0], [0, 4, 8])
        _, _, _, _, keep, _ = self.batch({})
        self.assertEqual([t for t in range(WINDOWS) if float(keep[t, 0]) == 0.0], [0])
        # a piece that starts mid-recording keeps the recording's own window numbers
        seq = synthetic_sequence(seed=9)

        class Data(object):
            def __getitem__(self, i):
                return seq

        cfg = dict(CFG, input_device="gpu", memory_reset_every=4)
        sources = T.StreamSources(Data(), cfg, Q99, DEV, torch.float64)
        x, _, _, keep = T.assemble_streams(sources, [[(0, 2, 10, 0, False)]], 8, cfg, DEV, torch.float64)
        self.assertEqual([t for t in range(8) if float(keep[t, 0]) == 0.0], [2, 6])   # windows 4 and 8

    def test_training_forward_equals_inference(self):
        for extra in ({"memory_reset_every": 4}, {"memory_reset_every": 1, "window_steps": 3}):
            seq, x, events, labels, keep, valid = self.batch(extra)
            x2, ev2, keep2, valid2 = T.expand_window_steps(x, events, keep, valid, T.memory_options(extra)[1])
            steps = T.memory_options(extra)[1]
            self.assertEqual(tuple(x2.shape)[0], WINDOWS * steps)
            self.assertEqual(float(valid2.sum()), float(valid.sum()) * steps)
            net = make_net(seed=31).eval()
            with torch.no_grad():
                logits, _, _ = net.forward_chunk(x2, ev2, None, keep=keep2)
                probs = torch.sigmoid(logits).float().numpy()
                ref, _ = T.run_sequence(net, seq, dict(CFG, execution="step", input_device="cpu", **extra), Q99, DEV,
                                        "carry")
            # training batch events follow the window order of the source; compare as a multiset per window
            source = make_window_source(seq, dict(CFG, input_device="gpu"), Q99, DEV, torch.float64)
            _, _, _, idx = source.chunk(0, WINDOWS)
            self.assertLess(float(np.abs(probs - ref[idx]).max()), 1e-6, extra)


if __name__ == "__main__":
    unittest.main()
