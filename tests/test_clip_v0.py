"""CPU regression tests; no CUDA/spconv import needed for activation installer."""
import ast
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required')
class TestClip(unittest.TestCase):
    def test_forward_and_gradient(self):
        x = torch.tensor([-.5, .25, .75, 1.5], requires_grad=True)
        y = torch.nn.Hardtanh(0., 1.)(x)
        self.assertTrue(torch.equal(y, torch.tensor([0., .25, .75, 1.])))
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([0., 1., 1., 0.])))

    def test_sites_and_weights_unchanged(self):
        path = Path(__file__).resolve().parents[1] / 'model' / 'evspsegnet_clip_v0.py'
        tree = ast.parse(path.read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        namespace = {'nn': torch.nn}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), 'exec'), namespace)
        net = torch.nn.Module()
        for stage in range(1, 5):
            setattr(net, 'conv%d' % stage, torch.nn.Sequential(
                torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Identity(), torch.nn.ReLU()),
                torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Identity(), torch.nn.ReLU())))
        before = {k: v.clone() for k, v in net.state_dict().items()}
        rng = torch.get_rng_state().clone()
        sites = namespace['install_clipped_activations'](net, [3, 4])
        self.assertEqual(sites, ['conv3.1.2', 'conv4.1.2'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, v in net.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))
        self.assertIsInstance(net.conv3[1][2], torch.nn.Hardtanh)
        self.assertIsInstance(net.conv3[0][2], torch.nn.ReLU)
        self.assertIsInstance(net.conv2[1][2], torch.nn.ReLU)
        with self.assertRaises(ValueError):
            namespace['install_clipped_activations'](net, [3, 3])


if __name__ == '__main__':
    unittest.main()
