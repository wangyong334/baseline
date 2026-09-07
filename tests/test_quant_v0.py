import unittest
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from model.quant_rate import QuantRate5, install_quantized_activations


@unittest.skipIf(torch is None, 'PyTorch required')
class TestQuant(unittest.TestCase):
    def test_levels_and_ties(self):
        x = torch.tensor([-1., 0., .124, .125, .374, .375, .625, .875, 1., 2.])
        expected = torch.tensor([0., 0., 0., .25, .25, .5, .75, 1., 1., 1.])
        self.assertTrue(torch.equal(QuantRate5()(x), expected))

    def test_gradient_matches_clip_mask(self):
        x = torch.tensor([-1., 0., .125, .4, .9, 1., 2.], requires_grad=True)
        weights = torch.arange(1., 8.)
        (QuantRate5()(x) * weights).sum().backward()
        y = x.detach().clone().requires_grad_()
        (torch.nn.Hardtanh(0., 1.)(y) * weights).sum().backward()
        self.assertTrue(torch.equal(x.grad, y.grad))

    def test_memoryless_no_mutation(self):
        layer = QuantRate5()
        x = torch.tensor([.1, .4, .8])
        original = x.clone()
        first = layer(x)
        layer(torch.ones(4, 3) * 8.)
        self.assertTrue(torch.equal(first, layer(x)))
        self.assertTrue(torch.equal(original, x))
        self.assertEqual(len(layer.state_dict()), 0)

    def test_sites_preserve_weights_rng(self):
        net = torch.nn.Module()
        for stage in range(1, 5):
            setattr(net, 'conv%d' % stage, torch.nn.Sequential(*[
                torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Identity(), torch.nn.ReLU())
                for _ in range(2)]))
        before = {k: v.clone() for k, v in net.state_dict().items()}
        rng = torch.get_rng_state().clone()
        self.assertEqual(install_quantized_activations(net, [3, 4]), ['conv3.1.2', 'conv4.1.2'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, v in net.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))
        self.assertIsInstance(net.conv3[1][2], QuantRate5)
        self.assertIsInstance(net.conv3[0][2], torch.nn.ReLU)
        self.assertIsInstance(net.conv2[1][2], torch.nn.ReLU)
        with self.assertRaises(ValueError):
            install_quantized_activations(net, [3, 3])


if __name__ == '__main__':
    unittest.main()
