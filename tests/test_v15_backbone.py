"""V15 feature taps retain V5 scoring and expose a history-free input current.

Small deterministic CPU tests; compatible with Python 3.8 / PyTorch 1.9.
"""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

from model.evspsegnet_stream import EvSpSegNetStream
from model.lif2d_stream import state_tensors


H, W = 16, 24


def make_net(seed=31, **kwargs):
    torch.manual_seed(seed)
    net = EvSpSegNetStream(
        channels=(2, 4, 4, 4), neuron="tif", norm="none",
        state_mode="carry", readout_hidden=5, v_threshold=0.05,
        tif={"k_init": 0.001, "quantize": True}, **kwargs).double()
    for block in net.blocks():
        block.gain.set_gain(torch.full((block.out_ch,), 20.0, dtype=torch.float64))
    return net


def events(batch=2, steps=None, empty=False):
    n = 0 if empty else 9
    idx = torch.arange(n, dtype=torch.long)
    data = {
        "b": idx.remainder(batch), "y": (idx * 3).remainder(H),
        "x": (idx * 7).remainder(W),
        "p": idx.remainder(2).double(),
        "t_local": (idx.double() + 0.5) / max(n, 1),
    }
    if steps is not None:
        data["t"] = (idx * 2 + 1).remainder(steps)
    return data


class V15BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_tensor_close(self, actual, expected, atol=1e-10):
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        if actual.numel():
            self.assertLessEqual(float((actual - expected).abs().max()), atol)

    def assert_states_close(self, actual, expected):
        aa, bb = list(state_tensors(actual)), list(state_tensors(expected))
        self.assertEqual(len(aa), len(bb))
        for a, b in zip(aa, bb):
            self.assert_tensor_close(a, b)

    def test_default_and_explicit_plain_input_keep_initialization_and_gradients(self):
        default = make_net()
        default_rng = torch.get_rng_state().clone()
        explicit = make_net(invariant=None)
        self.assertTrue(torch.equal(default_rng, torch.get_rng_state()))
        self.assertEqual(list(default.state_dict()), list(explicit.state_dict()))
        for name, value in default.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[name]), name)
        self.assertIsNone(default.invariant)
        x = torch.rand(2, 12, H, W, dtype=torch.float64, requires_grad=True)
        xx = x.detach().clone().requires_grad_(True)
        pred, state, info = default(x, events(), None, return_features=True)
        other, other_state, other_info = explicit(xx, events(), None, return_features=True)
        self.assertTrue(torch.equal(pred, other))
        self.assertTrue(torch.equal(info["observation"], other_info["observation"]))
        self.assert_states_close(state, other_state)
        pred.square().sum().backward()
        other.square().sum().backward()
        self.assertTrue(torch.equal(x.grad, xx.grad))
        for (name, p), (_, q) in zip(default.named_parameters(), explicit.named_parameters()):
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:
                self.assertTrue(torch.equal(p.grad, q.grad), name)

    def test_legacy_invariant_is_rejected_without_optional_module_import(self):
        path = Path(__file__).resolve().parents[1] / "model" / "evspsegnet_stream.py"
        spec = importlib.util.spec_from_file_location("v15_plain_backbone_import_test", str(path))
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"model.noise_invariant": None}):
            spec.loader.exec_module(module)
        for invalid in ({}, {"time_bins": 6}, False):
            before = torch.get_rng_state().clone()
            with self.assertRaisesRegex(ValueError, "invariant is not supported"):
                module.EvSpSegNetStream(invariant=invalid)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_feature_opt_in_and_deferred_readout_preserve_v5_exactly(self):
        net = make_net()
        x = torch.rand(2, 12, H, W, dtype=torch.float64)
        ev = events()
        old, old_states, old_info = net(x, ev, None)
        rich, rich_states, info = net(x, ev, None, return_features=True)
        self.assertIsNone(old_info)
        self.assertTrue(torch.equal(old, rich))
        self.assert_states_close(old_states, rich_states)
        self.assertEqual(tuple(info["features"].shape), (2, 2, H, W))
        self.assertEqual(tuple(info["observation"].shape), (2, 2, H, W))
        calls = []
        handle = net.readout.register_forward_hook(lambda *args: calls.append(1))
        try:
            skipped, skipped_states, skipped_info = net(
                x, ev, None, return_features=True, skip_readout=True)
            self.assertIsNone(skipped)
            self.assertEqual(calls, [])
            self.assertTrue(torch.equal(net.read_features(skipped_info["features"], ev), old))
            self.assertEqual(calls, [1])
        finally:
            handle.remove()
        self.assert_states_close(old_states, skipped_states)

    def test_observation_tap_is_history_free_but_deep_feature_is_not(self):
        net = make_net()
        x = torch.zeros(2, 12, H, W, dtype=torch.float64)
        _, states, _ = net(x, events(), None)
        zero_states = [torch.zeros_like(s) for s in states]
        memory_states = [torch.full_like(s, 0.02, requires_grad=True) for s in states]
        before = [s.detach().clone() for s in memory_states]
        _, _, cold = net(x, events(), zero_states, return_features=True)
        _, _, warm = net(x, events(), memory_states, return_features=True)
        self.assertTrue(torch.equal(cold["observation"], warm["observation"]))
        self.assertGreater(float((cold["features"] - warm["features"]).abs().max()), 0.0)
        grads = torch.autograd.grad(warm["observation"].sum(), memory_states,
                                    allow_unused=True)
        self.assertTrue(all(g is None for g in grads))
        for old, current in zip(before, memory_states):
            self.assertTrue(torch.equal(old, current.detach()))

    def test_observation_is_enc1_driving_current_without_a_second_convolution(self):
        net = make_net()
        x = torch.rand(2, 12, H, W, dtype=torch.float64)
        expected, _ = net.enc1._currents(x)
        calls = []
        handle = net.enc1.conv.register_forward_hook(lambda *args: calls.append(1))
        try:
            _, _, info = net(x, events(), None, return_features=True, skip_readout=True)
        finally:
            handle.remove()
        self.assertEqual(calls, [1])
        self.assertTrue(torch.equal(info["observation"], expected))

    def test_read_features_uses_original_event_metadata_and_time_batch_indices(self):
        net = make_net()
        feature = torch.randn(3, 2, 2, H, W, dtype=torch.float64)
        ev = events(steps=3)
        gathered = feature[ev["t"], ev["b"], :, ev["y"], ev["x"]]
        self.assertTrue(torch.equal(net.read_features(feature, ev), net._readout(gathered, ev)))
        single = feature[0]
        one_ev = events()
        gathered = single[one_ev["b"], :, one_ev["y"], one_ev["x"]]
        self.assertTrue(torch.equal(net.read_features(single, one_ev), net._readout(gathered, one_ev)))
        self.assertEqual(net.read_features(single, events(empty=True)).numel(), 0)

    def test_chunk_taps_match_step_with_recording_boundary_reset(self):
        net = make_net()
        x = torch.rand(4, 2, 12, H, W, dtype=torch.float64)
        ev = events(steps=4)
        keep = torch.ones(4, 2, dtype=torch.float64)
        keep[2, 1] = 0.0
        step_features, step_observation, states = [], [], None
        for t in range(4):
            if states is not None:
                states = [s * keep[t].view(2, 1, 1, 1) for s in states]
            _, states, info = net(x[t], events(), states,
                                  return_features=True, skip_readout=True)
            step_features.append(info["features"])
            step_observation.append(info["observation"])
        logits, chunk_states, chunk = net.forward_chunk(
            x, ev, None, keep=keep, return_features=True, skip_readout=True)
        self.assertIsNone(logits)
        self.assert_tensor_close(chunk["features"], torch.stack(step_features))
        self.assert_tensor_close(chunk["observation"], torch.stack(step_observation))
        self.assert_states_close(chunk_states, states)
        direct, _, _ = net.forward_chunk(x, ev, None, keep=keep)
        self.assertTrue(torch.equal(net.read_features(chunk["features"], ev), direct))

    def test_deferred_readout_keeps_backbone_and_head_gradients(self):
        legacy = make_net()
        deferred = copy.deepcopy(legacy)
        x = torch.rand(2, 12, H, W, dtype=torch.float64)
        ev = events()
        target = torch.linspace(-0.5, 0.8, ev["b"].numel(), dtype=torch.float64)
        old, _, _ = legacy(x, ev, None)
        _, _, info = deferred(x, ev, None, return_features=True, skip_readout=True)
        new = deferred.read_features(info["features"], ev)
        ((old - target) ** 2).mean().backward()
        ((new - target) ** 2).mean().backward()
        useful = set()
        for (name, a), (_, b) in zip(legacy.named_parameters(), deferred.named_parameters()):
            if a.grad is None:
                self.assertIsNone(b.grad, name)
            else:
                self.assertIsNotNone(b.grad, name)
                self.assert_tensor_close(a.grad, b.grad)
                if float(b.grad.abs().sum()) > 0:
                    useful.add(name.split(".")[0])
        self.assertIn("enc1", useful)
        self.assertIn("readout", useful)


if __name__ == "__main__":
    unittest.main()
