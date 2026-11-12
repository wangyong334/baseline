"""Behavioral tests for bounded local V15 observation revision.

Check evidence availability separately from observed silence, gradients through
the actual future input, and local rather than global attention allocation.
"""
import copy
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from model.observation_revision import ObservationRevision


def make_module(seed=47, **kwargs):
    torch.manual_seed(seed)
    cfg = dict(channels=2, width=4, heads=2, patch=2, radius=1,
               pair_batch=2, source="observation")
    cfg.update(kwargs)
    return ObservationRevision(**cfg).double()


def activate_output(module):
    with torch.no_grad():
        module.output.weight.copy_(
            torch.linspace(-0.3, 0.4, module.output.weight.numel(), dtype=torch.float64)
            .reshape_as(module.output.weight))


class V15RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_close(self, actual, expected, atol=1e-12):
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        if actual.numel():
            self.assertLessEqual(float((actual - expected).abs().max()), atol)

    def sample(self, module, batch=2, height=12, width=16):
        old = torch.randn(batch, 2, height, width, dtype=torch.float64)
        old_raw = torch.randn_like(old)
        new_raw = torch.randn_like(old)
        return old, module.encode_observation(old_raw), module.encode_observation(new_raw)

    def test_zero_initialization_is_exact_v5_and_starts_learning(self):
        module = make_module()
        old, old_e, new_e = self.sample(module)
        previous = old.clone()
        out, info = module.revise(old, old_e, new_e, return_info=True)
        self.assertTrue(torch.equal(out, old))
        self.assertTrue(torch.equal(old, previous))
        self.assertTrue(torch.equal(info["correction_rms"], torch.zeros(2, dtype=torch.float64)))
        (out * torch.randn_like(out)).sum().backward()
        self.assertIsNotNone(module.output.weight.grad)
        self.assertGreater(float(module.output.weight.grad.abs().sum()), 0.0)

    def test_unavailable_future_is_exact_fallback_per_stream(self):
        module = make_module()
        activate_output(module)
        old, old_e, new_e = self.sample(module, batch=3)
        available = torch.tensor([True, False, True])
        full = module.revise(old, old_e, new_e)
        out, info = module.revise(old, old_e, new_e, available, return_info=True)
        self.assertTrue(torch.equal(out[1], old[1]))
        self.assertEqual(float(info["correction_rms"][1]), 0.0)
        self.assert_close(out[available], full[available])
        self.assertGreater(float((out[available] - old[available]).abs().sum()), 0.0)
        all_missing = module.revise(old, old_e, new_e, torch.zeros(3, dtype=torch.bool))
        self.assertTrue(torch.equal(all_missing, old))

    def test_observed_silence_is_not_the_same_as_missing_future(self):
        module = make_module(width=2, heads=1)
        with torch.no_grad():
            module.value.weight.zero_()
            module.value.bias.fill_(1.0)
            module.output.weight.copy_(torch.eye(2, dtype=torch.float64).reshape(2, 2, 1, 1))
        old = torch.zeros(1, 2, 8, 8, dtype=torch.float64)
        silent_e = module.encode_observation(torch.zeros_like(old))
        observed = module.revise(old, silent_e, silent_e, torch.tensor([True]))
        missing = module.revise(old, silent_e, silent_e, torch.tensor([False]))
        self.assertTrue(torch.equal(missing, old))
        self.assertGreater(float(observed.abs().sum()), 0.0)

    def test_gradient_reaches_old_features_new_input_and_observation_encoder(self):
        module = make_module()
        activate_output(module)
        old = torch.randn(2, 2, 12, 16, dtype=torch.float64, requires_grad=True)
        old_raw = torch.randn_like(old, requires_grad=True)
        new_raw = torch.randn_like(old, requires_grad=True)
        out = module.revise(old, module.encode_observation(old_raw),
                            module.encode_observation(new_raw))
        (out * torch.randn_like(out)).sum().backward()
        for name, tensor in (("old_features", old), ("old_observation", old_raw),
                             ("future_observation", new_raw)):
            self.assertIsNotNone(tensor.grad, name)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()), name)
            self.assertGreater(float(tensor.grad.abs().sum()), 0.0, name)
        for name in ("encoder.0.weight", "query.weight", "key.weight", "value.weight"):
            grad = dict(module.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(bool(torch.isfinite(grad).all()), name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)

    def test_unavailable_pair_has_no_future_gradient(self):
        module = make_module()
        activate_output(module)
        old = torch.randn(2, 2, 8, 8, dtype=torch.float64, requires_grad=True)
        old_raw = torch.randn_like(old, requires_grad=True)
        new_raw = torch.randn_like(old, requires_grad=True)
        result = module.revise(old, module.encode_observation(old_raw),
                               module.encode_observation(new_raw), torch.tensor([True, False]))
        (result * torch.randn_like(result)).sum().backward()
        self.assertTrue(torch.equal(new_raw.grad[1], torch.zeros_like(new_raw.grad[1])))
        self.assertGreater(float(new_raw.grad[0].abs().sum()), 0.0)

    def test_padding_cannot_compete_with_valid_neighbors_and_null(self):
        module = make_module(width=2, heads=1)
        with torch.no_grad():
            for layer in (module.query, module.key):
                layer.weight.zero_()
                layer.bias.zero_()
            module.value.weight.zero_()
            module.value.bias.fill_(1.0)
            module.output.weight.copy_(torch.eye(2, dtype=torch.float64).reshape(2, 2, 1, 1))
            module.gate.weight.zero_()
            module.gate.bias.zero_()
        old = torch.zeros(1, 2, 6, 8, dtype=torch.float64)
        compact = torch.ones(1, 2, 3, 4, dtype=torch.float64)
        out, info = module.revise(old, compact, compact, return_info=True)
        count = F.conv2d(torch.ones(1, 1, 3, 4, dtype=torch.float64),
                         torch.ones(1, 1, 3, 3, dtype=torch.float64), padding=1)
        self.assert_close(info["null_probability"], 1.0 / (count + 1.0))
        expected = F.interpolate(count / (count + 1.0), size=(6, 8),
                                 mode="bilinear", align_corners=False).expand_as(old) * 0.5
        self.assert_close(out, expected)
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_tiny_canvas_larger_than_radius_remains_finite(self):
        module = make_module(radius=3)
        activate_output(module)
        old, old_e, new_e = self.sample(module, batch=1, height=2, width=2)
        out, info = module.revise(old, old_e, new_e, return_info=True)
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool(torch.isfinite(info["null_probability"]).all()))
        self.assertGreaterEqual(float(info["null_probability"].min()), 0.0)
        self.assertLessEqual(float(info["null_probability"].max()), 1.0)

    def test_future_perturbation_has_local_receptive_field(self):
        module = make_module()
        activate_output(module)
        old, old_e, new_e = self.sample(module, batch=1, height=24, width=32)
        changed = new_e.detach().clone()
        changed[:, :, 6, 8] += 3.0
        before = module.revise(old, old_e, new_e)
        after = module.revise(old, old_e, changed)
        diff = (after - before).abs()
        self.assertGreater(float(diff.sum()), 0.0)
        # Radius-one compact attention plus bilinear upsampling cannot affect
        # these distant full-resolution rows/columns.
        self.assertEqual(float(diff[:, :, :7, :].max()), 0.0)
        self.assertEqual(float(diff[:, :, 18:, :].max()), 0.0)
        self.assertEqual(float(diff[:, :, :, :11].max()), 0.0)
        self.assertEqual(float(diff[:, :, :, 22:].max()), 0.0)

    def test_pair_batching_preserves_values_and_parameter_gradients(self):
        one = make_module(pair_batch=1)
        activate_output(one)
        many = copy.deepcopy(one)
        many.pair_batch = 20
        old, old_e, new_e = self.sample(one, batch=5)
        # Test revision batching independently of the observation encoder graph.
        old_e, new_e = old_e.detach(), new_e.detach()
        available = torch.tensor([True, False, True, True, False])
        a, ia = one.revise(old, old_e, new_e, available, return_info=True)
        b, ib = many.revise(old, old_e, new_e, available, return_info=True)
        self.assert_close(a, b)
        for key in ia:
            self.assert_close(ia[key], ib[key])
        weights = torch.randn_like(a)
        (a * weights).sum().backward()
        (b * weights).sum().backward()
        for (name, pa), (_, pb) in zip(one.named_parameters(), many.named_parameters()):
            if pa.grad is None:
                self.assertIsNone(pb.grad, name)
            else:
                self.assert_close(pa.grad, pb.grad, atol=1e-10)

    def test_attention_neighbors_do_not_grow_with_canvas_area(self):
        module = make_module()
        recorded = []
        original = torch.softmax

        def capture(tensor, *args, **kwargs):
            recorded.append(tuple(tensor.shape))
            return original(tensor, *args, **kwargs)

        with mock.patch("model.observation_revision.torch.softmax", side_effect=capture):
            for height, width in ((8, 12), (16, 24)):
                old, old_e, new_e = self.sample(module, batch=1, height=height, width=width)
                module.revise(old, old_e, new_e)
        self.assertEqual(recorded, [(1, 2, 10, 24), (1, 2, 10, 96)])
        self.assertEqual(module.dense_macs(16, 24), 4 * module.dense_macs(8, 12))
        for value in module._mask_cache.values():
            self.assertFalse(value.requires_grad)

    def test_causal_control_is_exact_and_shape_errors_are_explicit(self):
        module = make_module(source="causal")
        activate_output(module)
        old, old_e, new_e = self.sample(module)
        out, info = module.revise(old, old_e, new_e, return_info=True)
        self.assertTrue(torch.equal(out, old))
        self.assertTrue(torch.equal(info["correction_rms"], torch.zeros(2, dtype=torch.float64)))
        with self.assertRaises(ValueError):
            module.encode_observation(torch.zeros(1, 2, 7, 8, dtype=torch.float64))
        with self.assertRaises(ValueError):
            module.revise(old, old_e[:, :, :-1], new_e)
        with self.assertRaises(ValueError):
            module.revise(old, old_e, new_e, torch.tensor([True]))
        with self.assertRaises(ValueError):
            make_module(width=3, heads=2)

    def test_sensor_padding_cannot_enter_encoder_or_partial_patches(self):
        module = make_module(sensor_hw=(7, 9))
        raw = torch.randn(1, 2, 12, 12, dtype=torch.float64, requires_grad=True)
        changed = raw.detach().clone()
        changed[:, :, 7:, :] = 1e6
        changed[:, :, :, 9:] = -1e6
        encoded = module.encode_observation(raw)
        self.assertTrue(torch.equal(encoded, module.encode_observation(changed)))
        self.assertEqual(float(encoded[:, :, 4:, :].abs().sum()), 0.0)
        self.assertEqual(float(encoded[:, :, :, 5:].abs().sum()), 0.0)
        (encoded * torch.randn_like(encoded)).sum().backward()
        self.assertEqual(float(raw.grad[:, :, 7:, :].abs().sum()), 0.0)
        self.assertEqual(float(raw.grad[:, :, :, 9:].abs().sum()), 0.0)
        # The physical last pixel lives in a partial patch and is retained.
        self.assertGreater(float(raw.grad[:, :, 6, 8].abs().sum()), 0.0)

    def test_sensor_padding_cannot_modify_valid_predictions_or_receive_revision(self):
        module = make_module(sensor_hw=(7, 9))
        activate_output(module)
        old = torch.randn(1, 2, 12, 12, dtype=torch.float64)
        old_raw, new_raw = torch.randn_like(old), torch.randn_like(old)
        old_e, new_e = module.encode_observation(old_raw), module.encode_observation(new_raw)
        out = module.revise(old, old_e, new_e)
        altered_old, altered_old_e, altered_new_e = old.clone(), old_e.clone(), new_e.clone()
        altered_old[:, :, 7:, :] += 1000
        altered_old[:, :, :, 9:] += 1000
        for compact in (altered_old_e, altered_new_e):
            compact[:, :, 4:, :] += 1000
            compact[:, :, :, 5:] += 1000
        altered = module.revise(altered_old, altered_old_e, altered_new_e)
        self.assertTrue(torch.equal(out[:, :, :7, :9], altered[:, :, :7, :9]))
        self.assertTrue(torch.equal(out[:, :, 7:, :], old[:, :, 7:, :]))
        self.assertTrue(torch.equal(out[:, :, :, 9:], old[:, :, :, 9:]))
        self.assertGreater(float((out[:, :, :7, :9] - old[:, :, :7, :9]).abs().sum()), 0.0)

    def test_sensor_invalid_tokens_are_null_only_and_diagnostics_ignore_them(self):
        module = make_module(width=2, heads=1, sensor_hw=(5, 7))
        with torch.no_grad():
            for layer in (module.query, module.key):
                layer.weight.zero_()
                layer.bias.zero_()
            module.value.weight.zero_()
            module.value.bias.fill_(1.0)
            module.output.weight.copy_(torch.eye(2, dtype=torch.float64).reshape(2, 2, 1, 1))
            module.gate.weight.zero_()
            module.gate.bias.zero_()
        old = torch.zeros(1, 2, 8, 12, dtype=torch.float64)
        compact = module.encode_observation(torch.zeros_like(old))
        _, info = module.revise(old, compact, compact, return_info=True)
        valid = torch.zeros(1, 1, 4, 6, dtype=torch.float64)
        valid[:, :, :3, :4] = 1.0
        count = F.conv2d(valid, torch.ones(1, 1, 3, 3, dtype=torch.float64), padding=1)
        expected = torch.where(valid.bool(), 1.0 / (count + 1.0), torch.ones_like(count))
        self.assert_close(info["null_probability"], expected)
        self.assert_close(info["null_mass_mean"], (expected * valid).sum((1, 2, 3)) / valid.sum())
        self.assert_close(info["gate_mean"], torch.tensor([0.5], dtype=torch.float64))

    def test_full_sensor_geometry_preserves_output_and_all_gradients(self):
        legacy = make_module()
        activate_output(legacy)
        candidate = make_module(sensor_hw=(12, 16))
        candidate.load_state_dict(legacy.state_dict())
        inputs_a = [torch.randn(2, 2, 12, 16, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        inputs_b = [value.detach().clone().requires_grad_() for value in inputs_a]
        outputs = []
        for module, inputs in ((legacy, inputs_a), (candidate, inputs_b)):
            outputs.append(module.revise(inputs[0], module.encode_observation(inputs[1]),
                                          module.encode_observation(inputs[2])))
        self.assertTrue(torch.equal(outputs[0], outputs[1]))
        weights = torch.randn_like(outputs[0])
        for output in outputs:
            (output * weights).sum().backward()
        for a, b in zip(inputs_a, inputs_b):
            self.assertTrue(torch.equal(a.grad, b.grad))
        for (name, a), (_, b) in zip(legacy.named_parameters(), candidate.named_parameters()):
            self.assertTrue(torch.equal(a.grad, b.grad), name)
        with self.assertRaises(ValueError):
            make_module(sensor_hw=(0, 7))
        with self.assertRaises(ValueError):
            make_module(sensor_hw=(13, 16)).encode_observation(inputs_a[1])


if __name__ == "__main__":
    unittest.main()
