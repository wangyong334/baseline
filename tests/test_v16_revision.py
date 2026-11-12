"""Mechanism, geometry, gradients and deterministic runtime checks for V16."""
import os
import sys
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.nn import functional as F

from model.context_revision import ContextRevision
from model.deterministic_resize import bilinear_resize


def small_model(variant="full", dtype=torch.float64):
    return ContextRevision(channels=6, dim=6, hidden=8, heads=2, radius=1,
                           downsample=2, height=9, width=13, variant=variant).to(dtype)


def pair(requires_grad=False, device="cpu", dtype=torch.float64):
    return tuple(torch.randn(2, 6, 12, 16, dtype=dtype, device=device,
                             requires_grad=requires_grad) for _ in range(2))


def activate_output(model):
    with torch.no_grad():
        model.revision[-1].weight.normal_(0, 0.04)
        model.revision[-1].bias.normal_(0, 0.02)


class ContextRevisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_initial_full_and_no_context_are_exact_second(self):
        first, second = pair()
        for variant in ("full", "no_context"):
            model = small_model(variant)
            result, info = model(first, second, return_info=True)
            self.assertTrue(torch.equal(result, second))
            self.assertTrue(torch.equal(info["correction_rms"], torch.zeros(2, dtype=second.dtype)))

    def test_no_revision_initializes_to_mean_on_sensor(self):
        model = small_model("no_revision")
        first, second = pair()
        result = model(first, second)
        self.assertTrue(torch.allclose(result[:, :, :9, :13], ((first + second) / 2)[:, :, :9, :13]))
        self.assertTrue(torch.equal(result[:, :, 9:], second[:, :, 9:]))
        self.assertTrue(torch.equal(result[:, :, :, 13:], second[:, :, :, 13:]))

    def test_constant_valid_values_have_zero_contrast_including_boundary(self):
        model = small_model()
        constants = torch.tensor([0.125, -2.0, 0.75, 3.0, -0.5, 1.5], dtype=torch.float64)
        features = constants[None, :, None, None].expand(2, -1, 12, 16).clone()
        features[:, :, 9:] = 1e8
        features[:, :, :, 13:] = -1e8
        encoded = model.encode(features)
        self.assertLess(float(encoded["r"].detach().abs().max()), 1e-12)
        self.assertTrue(torch.equal(encoded["u"][:, :, :5, :7],
                                    constants[None, :, None, None].expand(2, -1, 5, 7)))
        self.assertTrue(torch.equal(encoded["null_mass"][:, :, 5:],
                                    torch.ones_like(encoded["null_mass"][:, :, 5:])))

    def test_context_not_ordinary_average_and_null_can_reject(self):
        model = small_model()
        features, _ = pair()
        encoded = model.encode(features)
        self.assertGreater(float(encoded["r"].detach().abs().sum()), 1e-6)
        with torch.no_grad():
            model.null_bias.fill_(80)
        rejected = model.encode(features)
        self.assertLess(float(rejected["r"].detach().abs().max()), 1e-20)
        self.assertTrue(torch.isfinite(rejected["r"]).all())

    def test_no_context_is_actual_masked_uniform_mean(self):
        model = small_model("no_context")
        first, _ = pair()
        encoded = model.encode(first)
        u = encoded["u"]
        # Last valid coarse query (4, 6) has four valid neighboring patches.
        reference = u[:, :, 3:5, 5:7].mean((2, 3))
        self.assertTrue(torch.allclose(encoded["r"][:, :, 4, 6], reference))
        self.assertFalse(hasattr(model, "query"))
        self.assertFalse(any("null" in name for name, _ in model.named_parameters()))

    def test_padding_does_not_enter_encoding_or_valid_revision(self):
        for variant in ("full", "no_context", "no_revision"):
            model = small_model(variant)
            activate_output(model)
            first, second = pair()
            changed_first, changed_second = first.clone(), second.clone()
            changed_first[:, :, 9:] = 9e5
            changed_first[:, :, :, 13:] = -8e5
            changed_second[:, :, 9:] = -7e5
            changed_second[:, :, :, 13:] = 6e5
            before, after = model.encode(first), model.encode(changed_first)
            self.assertTrue(torch.equal(before["u"], after["u"]))
            self.assertTrue(torch.equal(before["r"], after["r"]))
            original = model(first, second)
            changed = model(changed_first, changed_second)
            self.assertTrue(torch.equal(original[:, :, :9, :13], changed[:, :, :9, :13]))

    def test_padding_only_has_no_encoder_gradient(self):
        first, _ = pair(requires_grad=True)
        encoded = small_model().encode(first)
        (encoded["u"].square().sum() + encoded["r"].square().sum()).backward()
        self.assertEqual(float(first.grad[:, :, 9:].abs().sum()), 0)
        self.assertEqual(float(first.grad[:, :, :, 13:].abs().sum()), 0)
        self.assertGreater(float(first.grad[:, :, :9, :13].abs().sum()), 0)

    def test_cached_encoding_and_pair_batching_preserve_result(self):
        model = small_model()
        activate_output(model)
        first, second = pair()
        direct = model(first, second)
        cached = model.fuse(first, second, model.encode(first), model.encode(second))
        split = torch.cat([model(first[j:j + 1], second[j:j + 1]) for j in range(2)])
        self.assertTrue(torch.equal(direct, cached))
        self.assertTrue(torch.allclose(direct, split, atol=1e-12, rtol=1e-12))

    def test_zero_init_learning_then_all_modules_and_both_views_receive_gradient(self):
        for variant in ("full", "no_context", "no_revision"):
            model = small_model(variant)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.04)
            first, second = pair(requires_grad=True)
            target = torch.randn_like(first)
            result = model(first, second)
            ((result - target).square().mean()).backward()
            self.assertGreater(float(model.revision[-1].weight.grad.abs().sum()), 0)
            if variant != "no_context":
                self.assertEqual(float(model.query.weight.grad.abs().sum()), 0)
            if variant != "no_revision":
                self.assertEqual(float(first.grad.abs().sum()), 0)
            optimizer.step()
            optimizer.zero_grad()
            first.grad = None
            second.grad = None
            result = model(first, second)
            ((result - target).square().mean()).backward()
            self.assertGreater(float(first.grad.abs().sum()), 0)
            self.assertGreater(float(second.grad.abs().sum()), 0)
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, (variant, name))
                self.assertTrue(torch.isfinite(parameter.grad).all(), (variant, name))
                self.assertGreater(float(parameter.grad.abs().sum()), 0, (variant, name))

    def test_correction_can_change_agreeing_features(self):
        # No hard U1==U2 idempotence: agreement can still support a signed
        # residual correction, including reducing an agreeing false alarm.
        model = small_model()
        first, _ = pair()
        with torch.no_grad():
            model.revision[-1].bias[model.channels:].fill_(-0.25)
        result = model(first, first)
        self.assertTrue(torch.allclose(result[:, :, :9, :13], first[:, :, :9, :13] - 0.25))

    def test_high_resolution_detail_bypasses_pooling(self):
        model = small_model()
        first, second = pair()
        with torch.no_grad():
            model.revision[-1].bias[:model.channels].fill_(math_atanh(0.3))
        result = model(first, second)
        expected = second + 0.3 * (first - second)
        self.assertTrue(torch.allclose(result[:, :, :9, :13], expected[:, :, :9, :13]))

    def test_variants_are_structurally_independent(self):
        full, no_context, no_revision = [small_model(v) for v in ("full", "no_context", "no_revision")]
        self.assertEqual(full.revision[0].in_channels, 36)
        self.assertEqual(no_context.revision[0].in_channels, 36)
        self.assertEqual(no_revision.revision[0].in_channels, 24)
        self.assertTrue(hasattr(no_revision, "query"))
        self.assertEqual(no_revision.revision[-1].out_channels, 6)
        self.assertEqual(full.revision[-1].out_channels, 12)

    def test_default_shapes_macs_and_state_dict_roundtrip(self):
        model = ContextRevision()
        with torch.no_grad():
            encoded = model.encode(torch.zeros(1, 12, 264, 352))
        self.assertEqual(encoded["u"].shape, (1, 12, 66, 88))
        self.assertEqual(encoded["r"].shape, (1, 12, 66, 88))
        self.assertEqual(model.dense_macs(264, 352), 5808 * (1044 + 2304))
        self.assertEqual(ContextRevision(variant="no_context").dense_macs(264, 352), 5808 * 2304)
        clone = ContextRevision()
        clone.load_state_dict(model.state_dict(), strict=True)
        self.assertFalse(any("cache" in key for key in model.state_dict()))

    def test_projection_for_different_descriptor_width(self):
        model = ContextRevision(channels=6, dim=4, heads=2, hidden=8,
                                downsample=2, height=9, width=13).double()
        first, second = pair()
        self.assertEqual(model.encode(first)["u"].shape, (2, 4, 6, 8))
        self.assertTrue(torch.equal(model(first, second), second))

    def test_validation(self):
        for args in ({"dim": 7}, {"variant": "other"}, {"radius": -1},
                     {"downsample": 0}, {"height": 2.5}):
            with self.assertRaises(ValueError):
                ContextRevision(**args)
        model = small_model()
        first, second = pair()
        with self.assertRaises(ValueError):
            model.encode(first[:, :, :11])
        with self.assertRaises(ValueError):
            model(first, second[:1])
        with self.assertRaises(ValueError):
            model(first, second, {"u": torch.zeros(2, 6, 6, 8)})
        with self.assertRaises(ValueError):
            model.encode(torch.randn(1, 6, 8, 16, dtype=torch.float64))


def math_atanh(value):
    import math
    return math.atanh(value)


class DeterministicResizeTests(unittest.TestCase):
    def test_adjoint_matches_native_for_integer_scales(self):
        torch.manual_seed(7)
        for shape in ((1, 2, 1, 1), (1, 2, 2, 3), (1, 1, 3, 2)):
            for sy, sx in ((1, 1), (2, 2), (3, 5), (4, 4)):
                x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
                native_x = x.detach().clone().requires_grad_(True)
                size = (shape[-2] * sy, shape[-1] * sx)
                custom = bilinear_resize(x, size)
                native = F.interpolate(native_x, size=size, mode="bilinear", align_corners=False)
                weight = torch.randn_like(custom)
                (custom * weight).sum().backward()
                (native * weight).sum().backward()
                self.assertTrue(torch.equal(custom, native))
                self.assertTrue(torch.allclose(x.grad, native_x.grad, atol=1e-12, rtol=1e-12),
                                (shape, sy, sx, float((x.grad - native_x.grad).abs().max())))

    def test_custom_adjoint_gradcheck(self):
        x = torch.randn(1, 1, 2, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda value: bilinear_resize(value, (8, 12)), (x,)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_strict_cuda_full_pair_backward_is_repeatable(self):
        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(37)
            model = small_model(dtype=torch.float32).cuda()
            activate_output(model)
            first, second = pair(requires_grad=True, device="cuda", dtype=torch.float32)
            outputs, gradients = [], []
            for _ in range(2):
                model.zero_grad()
                first.grad = None
                second.grad = None
                result = model(first, second)
                result.square().mean().backward()
                torch.cuda.synchronize()
                outputs.append(result.detach().clone())
                gradients.append([first.grad.clone(), second.grad.clone()]
                                 + [p.grad.clone() for p in model.parameters()])
            self.assertTrue(torch.equal(outputs[0], outputs[1]))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(*gradients)))
        finally:
            torch.use_deterministic_algorithms(prior)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
