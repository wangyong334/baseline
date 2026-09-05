"""Dependency-free routing contract tests; NOT a CUDA/numerical verification."""
import contextlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class Device:
    def __init__(self, value):
        self.value = str(value)

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return self.value == str(other)


class Tensor:
    dtype = "float32"

    def __init__(self, value, device="cuda:0"):
        self.value, self.device = value, Device(device)

    def to(self, device):
        return Tensor(self.value, device)


class Sparse:
    def __init__(self, features, indices, spatial_shape, batch_size):
        self.features, self.indices = features, indices
        self.spatial_shape, self.batch_size = spatial_shape, batch_size
        self.indice_dict = {}

    def replace_feature(self, features):
        result = Sparse(features, self.indices, self.spatial_shape, self.batch_size)
        result.indice_dict = self.indice_dict.copy()
        return result


class Layer:
    def __init__(self, name):
        self.name = name

    def to(self, device):
        self.device = device
        return self

    def __call__(self, x):
        if isinstance(x, Tensor):
            assert x.device == self.device
            return x
        assert x.features.device == self.device, self.name
        assert x.indices.device == self.device, self.name
        result = x.replace_feature(x.features)
        if self.name in ("conv2", "conv3", "conv4"):
            key = "spconv" + self.name[-1]
            result.indice_dict[key] = (x.indices, self.device)
            result.indices = Tensor(self.name, self.device)
        return result


class Base:
    def __init__(self, cfg):
        for name in ("conv_input", "semantic_linear", "conv5"):
            setattr(self, name, Layer(name))
        for stage in range(1, 5):
            for prefix in ("conv", "conv_up_t", "conv_up_m", "pa", "inv_conv"):
                name = prefix + str(stage)
                setattr(self, name, Layer(name))

    def UR_block_forward(self, lateral, bottom, conv_t, conv_m, inverse):
        assert lateral.indices.value == bottom.indices.value
        assert lateral.features.device == bottom.features.device
        result = conv_m(conv_t(lateral))
        if inverse.name.startswith("inv_conv"):
            key = "spconv" + inverse.name[-1]
            indices, owner = result.indice_dict[key]
            assert owner == inverse.device  # Never consume another GPU's cache.
            result.indices = indices
        else:
            result = inverse(result)
        return result


class RoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch = types.ModuleType("torch")
        torch.device = Device
        torch.float32 = "float32"
        torch.equal = lambda a, b: a.value == b.value and a.device == b.device
        torch.is_autocast_enabled = lambda: False
        torch.cuda = types.SimpleNamespace(
            device_count=lambda: 2, device=lambda d: contextlib.nullcontext())
        spconv = types.ModuleType("spconv")
        spconv.pytorch = types.ModuleType("spconv.pytorch")
        spconv.pytorch.SparseConvTensor = Sparse
        base = types.ModuleType("model.evspsegnet")
        base.evspsegnet = Base
        path = Path(__file__).resolve().parents[1] / "model" / "evspsegnet_mp.py"
        spec = importlib.util.spec_from_file_location("mp_under_test", path)
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": torch, "spconv": spconv,
                                     "spconv.pytorch": spconv.pytorch,
                                     "model.evspsegnet": base}):
            spec.loader.exec_module(cls.module)

    def test_all_splits_restore_input_order_and_use_local_inverse_cache(self):
        for split in (1, 2, 3):
            with self.subTest(split=split):
                net = self.module.evspsegnet_mp(None, split)
                x = Sparse(Tensor("features"), Tensor("input-order"), [8, 8, 8], 1)
                preds, voxel = net.forward(x)
                self.assertEqual(preds.device, Device("cuda:0"))
                self.assertEqual(voxel.indices.value, "input-order")

    def test_transfer_drops_cache_but_preserves_feature_payload(self):
        x = Sparse(Tensor("payload"), Tensor("order"), [8, 8, 8], 1)
        x.indice_dict["old_gpu_cache"] = object()
        moved = self.module.transfer_sparse(x, "cuda:1")
        self.assertEqual(moved.indice_dict, {})
        self.assertEqual(moved.features.value, "payload")
        self.assertEqual(moved.indices.device, Device("cuda:1"))
        self.assertIn("old_gpu_cache", x.indice_dict)
        self.assertIs(self.module.transfer_sparse(x, "cuda:0"), x)

    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):
            self.module.evspsegnet_mp(None, 0)

    def test_amp_and_wrong_input_device_rejected(self):
        net = self.module.evspsegnet_mp(None, 2)
        x = Sparse(Tensor("features"), Tensor("input-order"), [8, 8, 8], 1)
        with patch.object(self.module.torch, "is_autocast_enabled", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "FP32"):
                net.forward(x)
        moved = self.module.transfer_sparse(x, "cuda:1")
        with self.assertRaisesRegex(RuntimeError, "logical cuda:0"):
            net.forward(moved)


if __name__ == "__main__":
    unittest.main()
