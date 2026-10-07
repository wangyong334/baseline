import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speed.eval.energy import AC_PJ, MAC_PJ, energy_mj, part, pixel_head, system_energy  # noqa: E402


class EnergyTests(unittest.TestCase):
    def test_energy_units(self):
        ops = part(mac=1e9, ac=1e9, transcendental=1e6)
        self.assertAlmostEqual(energy_mj(ops, 10.0), (1e9 * MAC_PJ + 1e9 * AC_PJ + 1e7 * MAC_PJ) * 1e-9)

    def test_pixel_head(self):
        h = pixel_head(12, 16, 2, positions=100, transcendental_per_position=1)
        self.assertEqual(h["mac"], 100 * (12 * 16 + 16 * 2))
        self.assertEqual(h["ac"], 1600)
        self.assertEqual(h["transcendental"], 100)

    def test_system_totals(self):
        parts = {"a": part(mac=1e6), "b": part(ac=2e6)}
        e = system_energy(parts, window_us=50000, clip_us=8000000, transcendental_macs=(10.0,))
        per_window = (1e6 * MAC_PJ + 2e6 * AC_PJ) * 1e-9
        self.assertAlmostEqual(e["total"]["x10"]["mj_per_window"], per_window)
        self.assertAlmostEqual(e["total"]["x10"]["mj_per_s"], per_window * 20)
        self.assertAlmostEqual(e["total"]["x10"]["mj_per_clip"], per_window * 160)


if __name__ == "__main__":
    unittest.main()
