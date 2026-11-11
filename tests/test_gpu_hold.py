"""Elastic GPU placeholder (tools/final/gpu_hold.py, 11-10) decision rules; no GPU needed.

    1. Free-memory target: idle margin without a job, the start margin during warm-up, the job margin afterwards
    2. Chunk plan: release whenever free memory is below the target (never more chunks than held), allocate only while
       more than one chunk above it, at most max_step chunks per poll, nothing inside the dead band
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "final"))
import gpu_hold as GH  # noqa: E402


class TargetTests(unittest.TestCase):
    def test_margins(self):
        self.assertEqual(GH.target_free_mib(False, 100.0, 200.0, 1, 4, 8), 1024)
        self.assertEqual(GH.target_free_mib(True, 100.0, 200.0, 1, 4, 8), 8 * 1024)
        self.assertEqual(GH.target_free_mib(True, 300.0, 200.0, 1, 4, 8), 4 * 1024)


class PlanTests(unittest.TestCase):
    def test_release_when_below_target(self):
        self.assertEqual(GH.plan(1000, 4096, held=40, chunk_mib=256, max_step=16), -13)
        self.assertEqual(GH.plan(1000, 4096, held=5, chunk_mib=256, max_step=16), -5)
        self.assertEqual(GH.plan(1000, 4096, held=0, chunk_mib=256, max_step=16), 0)

    def test_allocate_when_above_target(self):
        self.assertEqual(GH.plan(4096 + 1024, 4096, held=0, chunk_mib=256, max_step=16), 4)
        self.assertEqual(GH.plan(20000, 1024, held=0, chunk_mib=256, max_step=16), 16)

    def test_dead_band(self):
        self.assertEqual(GH.plan(4096, 4096, held=10, chunk_mib=256, max_step=16), 0)
        self.assertEqual(GH.plan(4096 + 200, 4096, held=10, chunk_mib=256, max_step=16), 0)


if __name__ == "__main__":
    unittest.main()
