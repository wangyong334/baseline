"""CPU tests: python -m unittest discover -s tests -p test_lif_rate.py -v"""
import unittest

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from model.lif_rate import LIFRate


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TestLIFRate(unittest.TestCase):
    def test_reference_recurrence(self):
        x = torch.tensor([[0.0, 0.3, 0.75, 2.0]])
        layer = LIFRate(4, 0.9, 1.0)
        expected = []
        for current in x[0].tolist():
            voltage, spikes = 0.0, 0
            for _ in range(4):
                voltage = 0.9 * voltage + current
                spike = int(voltage >= 1.0)
                spikes += spike
                voltage -= spike
            expected.append(spikes / 4.0)
        self.assertTrue(torch.equal(layer(x), torch.tensor([expected])))

    def test_reset_and_new_shape(self):
        layer = LIFRate()
        x = torch.full((3, 2), 0.4)
        first = layer(x)
        layer(torch.full((7, 2), 3.0))
        self.assertTrue(torch.equal(first, layer(x)))
        self.assertEqual(layer.state_dict(), {})

    def test_surrogate_gradient(self):
        x = torch.full((2, 3), 0.4, requires_grad=True)
        LIFRate()(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad != 0).any())

    def test_statistics_and_bounds(self):
        layer = LIFRate(4, 0.9, 1.0)
        layer.collect = True
        out = layer(torch.tensor([[-2.0, 0.0, 2.0]]))
        self.assertTrue(torch.equal(out, torch.tensor([[0.0, 0.0, 1.0]])))
        self.assertAlmostEqual(layer.stats["firing_rate"], 1 / 3, places=6)
        self.assertEqual(layer.stats["spike_count"], 4)

    def test_independent_graphs(self):
        layer = LIFRate()
        for _ in range(2):
            x = torch.full((2, 3), 0.5, requires_grad=True)
            layer(x).sum().backward()
            self.assertIsNotNone(x.grad)

    def test_invalid_parameters(self):
        for args in ((0, .9, 1.), (1.5, .9, 1.), (4, 1.1, 1.), (4, .9, 0.)):
            with self.assertRaises(ValueError):
                LIFRate(*args)


if __name__ == "__main__":
    unittest.main()
