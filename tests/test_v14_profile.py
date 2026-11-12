"""Density selection and optional board-power accounting, without CUDA/NVIDIA."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("test_v14_profile_module", str(ROOT / "tools/v14/profile_overlap.py"))
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class DensitySelectionTests(unittest.TestCase):
    def test_selects_low_middle_high_density_without_using_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, count in enumerate((90, 10, 60, 30, 100)):
                path = Path(temporary) / ("record_%d.npz" % index)
                # No labels or model outputs exist: selection must use input only.
                np.savez_compressed(str(path), ev_loc=np.zeros((count, 3), dtype=np.int32))
                paths.append(path)
            dataset = mock.Mock(paths=paths)
            self.assertEqual(profile.select_density_records(dataset, 3), [1, 2, 4])
            self.assertEqual(len(profile.select_density_records(dataset, 20)), 5)


class BoardPowerTests(unittest.TestCase):
    def test_cpu_does_not_query_nvidia(self):
        power = profile.BoardPower(torch.device("cpu"))
        with mock.patch.object(profile, "process_gpus") as query:
            power.start()
        query.assert_not_called()
        result = power.finish(0., 1.)
        self.assertIsNone(result["board_joules"])
        self.assertIsNone(result["gpu_uuid"])

    def test_uses_actual_process_uuid_instead_of_logical_cuda_index(self):
        power = profile.BoardPower(torch.device("cuda:0"))
        with mock.patch.object(profile.shutil, "which", return_value="nvidia-smi"), \
             mock.patch.object(profile, "process_gpus", return_value=[(os.getpid(), "GPU-physical-one"), (4321, "GPU-other")]), \
             mock.patch.object(profile, "smi", return_value="65.0") as query, \
             mock.patch.object(profile.threading, "Thread"):
            power.start()
        self.assertEqual(power.uuid, "GPU-physical-one")
        self.assertEqual(query.call_args[0][0][:2], ["-i", "GPU-physical-one"])
        self.assertEqual(power.other_pids, [])

    def test_ambiguous_multi_gpu_process_does_not_guess_power_device(self):
        power = profile.BoardPower(torch.device("cuda:0"))
        with mock.patch.object(profile.shutil, "which", return_value="nvidia-smi"), \
             mock.patch.object(profile, "process_gpus", return_value=[(os.getpid(), "GPU-a"), (os.getpid(), "GPU-b")]), \
             mock.patch.object(profile, "smi") as query:
            power.start()
        query.assert_not_called()
        self.assertIsNone(power.uuid)
        self.assertTrue(power.errors)

    def test_integrates_seconds_and_watts_at_interpolated_boundaries(self):
        power = profile.BoardPower(torch.device("cpu"))
        power.samples = [(0., 10.), (1., 30.), (3., 30.)]
        result = power.finish(.5, 2.)
        self.assertAlmostEqual(result["board_joules"], 42.5)
        self.assertAlmostEqual(result["mean_board_watts"], 42.5 / 1.5)

    def test_unbracketed_measurements_do_not_fabricate_energy(self):
        power = profile.BoardPower(torch.device("cpu"))
        power.samples = [(1., 10.), (2., 30.)]
        self.assertIsNone(power.finish(0., 2.)["board_joules"])
        power.samples = []
        self.assertIsNone(power.finish(0., 2.)["board_joules"])

    def test_shared_board_records_other_processes_seen_at_start_or_finish(self):
        power = profile.BoardPower(torch.device("cuda:0"))
        power.uuid = "GPU-a"
        power.other_pids = [12]
        power.samples = [(0., 50.), (2., 50.)]
        with mock.patch.object(power, "sample"), \
             mock.patch.object(profile, "process_gpus", return_value=[(os.getpid(), "GPU-a"), (13, "GPU-a"), (14, "GPU-b")]):
            result = power.finish(.5, 1.5)
        self.assertEqual(result["other_compute_pids_observed"], [12, 13])
        self.assertEqual(result["board_joules"], 50.)
        self.assertIn("whole NVIDIA GPU board", result["scope"])


if __name__ == "__main__":
    unittest.main()
