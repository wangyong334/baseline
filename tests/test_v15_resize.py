"""Check the adjoint against native CPU bilinear and the strict CUDA path."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from model.observation_revision import bilinear_resize, ObservationRevision


class V15ResizeTests(unittest.TestCase):
    def test_forward_and_adjoint_match_native_including_boundaries(self):
        torch.manual_seed(19)
        for height, width in ((1, 1), (1, 5), (4, 1), (3, 5), (7, 9)):
            for sy, sx in ((1, 1), (2, 2), (3, 3), (4, 4), (2, 3)):
                with self.subTest(shape=(height, width), scale=(sy, sx)):
                    original = torch.randn(2, 3, height, width, dtype=torch.float64, requires_grad=True)
                    reference = original.detach().clone().requires_grad_()
                    size = (height * sy, width * sx)
                    actual = bilinear_resize(original, size)
                    expected = F.interpolate(reference, size=size, mode="bilinear", align_corners=False)
                    self.assertTrue(torch.equal(actual, expected))
                    weight = torch.randn_like(expected)
                    (actual * weight).sum().backward()
                    (expected * weight).sum().backward()
                    self.assertLess(float((original.grad - reference.grad).abs().max()), 2e-13)

    def test_numerical_gradient(self):
        value = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x: bilinear_resize(x, (8, 12)), (value,)))

    def test_no_resize_with_native_backward_is_created(self):
        value = torch.randn(1, 2, 3, 5, requires_grad=True)
        result = bilinear_resize(value, (12, 20))
        self.assertIn("BilinearResizeBackward", type(result.grad_fn).__name__)
        with torch.no_grad():
            self.assertTrue(torch.equal(bilinear_resize(value, (12, 20)), result))
        with self.assertRaisesRegex(ValueError, "integer enlargement"):
            bilinear_resize(value, (7, 11))

    def test_complete_revision_matches_native_outputs_and_parameter_gradients(self):
        import copy
        torch.manual_seed(31)
        actual = ObservationRevision(channels=2, width=4, heads=2, patch=4,
                                     radius=1, sensor_hw=(11, 17)).double()
        with torch.no_grad():
            actual.output.weight.normal_(0, 0.2)
        expected = copy.deepcopy(actual)
        old = torch.randn(1, 2, 12, 20, dtype=torch.float64)
        old_input, new_input = torch.randn_like(old), torch.randn_like(old)
        weight = torch.randn_like(old)
        a = actual(old, actual.encode_observation(old_input), actual.encode_observation(new_input))
        (a * weight).sum().backward()

        def native(value, size):
            return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

        with mock.patch("model.observation_revision.bilinear_resize", side_effect=native):
            b = expected(old, expected.encode_observation(old_input), expected.encode_observation(new_input))
            (b * weight).sum().backward()
        self.assertTrue(torch.equal(a, b))
        for (name, pa), (_, pb) in zip(actual.named_parameters(), expected.named_parameters()):
            self.assertIsNotNone(pa.grad, name)
            self.assertLess(float((pa.grad - pb.grad).abs().max()), 2e-12, name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_strict_deterministic_backward_and_repeatability(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(37)
            base = torch.randn(2, 3, 7, 9, device="cuda")
            weight = torch.randn(2, 3, 28, 36, device="cuda")
            gradients = []
            for _ in range(2):
                value = base.detach().clone().requires_grad_()
                out = bilinear_resize(value, (28, 36))
                self.assertTrue(torch.equal(out, F.interpolate(value.detach(), size=(28, 36),
                                                              mode="bilinear", align_corners=False)))
                (out * weight).sum().backward()
                gradients.append(value.grad)
            self.assertTrue(torch.equal(*gradients))
            reference = base.cpu().requires_grad_()
            (F.interpolate(reference, size=(28, 36), mode="bilinear", align_corners=False)
             * weight.cpu()).sum().backward()
            self.assertLess(float((gradients[0].cpu() - reference.grad).abs().max()), 5e-6)
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__ == "__main__":
    unittest.main()
