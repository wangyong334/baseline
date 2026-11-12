"""V15 profiling protocol tests without a GPU or NVIDIA driver."""
import contextlib
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from tools.v15 import profile as P


class FakePower:
    def __init__(self, device):
        self.device = device

    def start(self):
        pass

    def finish(self, start, end):
        return {"board_joules": None, "scope": "mock CPU"}


class FakeHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))

    def dense_macs(self, height, width):
        return height * width * 2


class V15ProfileTests(unittest.TestCase):
    def test_cpu_never_queries_nvidia(self):
        power = P.BoardPower(torch.device("cpu"))
        with mock.patch.object(P, "process_gpus") as query, mock.patch.object(P, "smi") as smi:
            power.start()
            result = power.finish(1.0, 2.0)
        query.assert_not_called()
        smi.assert_not_called()
        self.assertIsNone(result["board_joules"])

    def test_gpu_mapping_uses_process_uuid_not_visible_device_index(self):
        power = P.BoardPower(torch.device("cuda:0"))
        with mock.patch.object(P.shutil, "which", return_value="nvidia-smi"), \
                mock.patch.object(P.os, "getpid", return_value=123), \
                mock.patch.object(P, "process_gpus", return_value=[(999, "GPU-physical-0"),
                                                                  (123, "GPU-physical-1")]), \
                mock.patch.object(P, "smi", return_value="100") as query, \
                mock.patch.object(P.time, "perf_counter", return_value=0.0), \
                mock.patch.object(P.threading, "Thread"):
            power.start()
        self.assertEqual(power.uuid, "GPU-physical-1")
        self.assertEqual(power.other_pids, [])
        self.assertEqual(query.call_args[0][0][:2], ["-i", "GPU-physical-1"])

    def test_multiple_process_gpus_are_not_resolved_by_guess(self):
        power = P.BoardPower(torch.device("cuda:0"))
        with mock.patch.object(P.shutil, "which", return_value="nvidia-smi"), \
                mock.patch.object(P.os, "getpid", return_value=123), \
                mock.patch.object(P, "process_gpus", return_value=[(123, "GPU-A"), (123, "GPU-B")]), \
                mock.patch.object(P, "smi") as query:
            power.start()
        self.assertIsNone(power.uuid)
        self.assertTrue(power.errors)
        query.assert_not_called()

    def test_energy_interpolates_boundaries_and_reports_shared_board(self):
        power = P.BoardPower(torch.device("cuda:0"))
        power.uuid = "GPU-A"
        power.samples = [(0.0, 100.0), (2.0, 200.0)]
        power.other_pids = [321]
        with mock.patch.object(P, "smi", return_value="300"), \
                mock.patch.object(P.time, "perf_counter", return_value=4.0), \
                mock.patch.object(P.os, "getpid", return_value=123), \
                mock.patch.object(P, "process_gpus", return_value=[(123, "GPU-A"), (456, "GPU-A")]):
            result = power.finish(1.0, 3.0)
        self.assertAlmostEqual(result["board_joules"], 400.0)
        self.assertAlmostEqual(result["mean_board_watts"], 200.0)
        self.assertEqual(result["other_compute_pids_observed"], [321, 456])
        self.assertIn("not attributable to model if shared", result["scope"])

    def test_insufficient_power_coverage_does_not_fabricate_energy(self):
        for samples in ([], [(0.0, 100.0)], [(0.0, 100.0), (2.0, 100.0)]):
            with self.subTest(samples=samples):
                power = P.BoardPower(torch.device("cpu"))
                power.samples = samples
                self.assertIsNone(power.finish(1.0, 3.0)["board_joules"])

    def test_nonfinite_power_is_not_recorded(self):
        power = P.BoardPower(torch.device("cuda:0"))
        power.uuid = "GPU-A"
        for value in ("nan", "inf", "-1", "[N/A]"):
            with mock.patch.object(P, "smi", return_value=value):
                power.sample()
        self.assertEqual(power.samples, [])

    def test_density_selection_includes_low_median_high_not_first_three(self):
        with tempfile.TemporaryDirectory() as directory:
            names = ["record_%d.npz" % i for i in range(5)]
            for name, n in zip(names, (3, 100, 7, 50, 20)):
                np.savez(Path(directory) / name, ev_loc=np.zeros((n, 3), dtype=np.int32))
            dataset = SimpleNamespace(directory=directory, names=names)
            self.assertEqual(P.density_records(dataset, 3), [0, 4, 1])

    def run_mock_profile(self, causal=False, nondeterministic=False, readouts=4, device="cpu"):
        cfg = {"channels": [2, 4, 4, 4], "window_ms": 50, "state_mode": "carry",
               "neuron": "tif", "seed": 37, "deterministic": True,
               "pad_height": 16, "pad_width": 24, "height": 15, "width_px": 23,
               "readout_hidden": 5, "_training_scope": {"limit_records": None, "max_chunks": None},
               "revision_source": "causal" if causal else "observation"}
        seq = SimpleNamespace(name="one.npz", n_events=4, n_windows=2,
                              t=np.asarray([0, 49, 50, 99], dtype=np.int64))

        class Dataset:
            names = ["one.npz"]

            def __getitem__(self, index):
                return seq

        net = torch.nn.Linear(1, 1)
        head = None if causal else FakeHead()
        payload = {"design": P.DESIGN, "config": cfg, "model": net.state_dict(),
                   "revision": None if head is None else head.state_dict(),
                   "stats": {"q99": np.ones(12)}}
        call_count = [0]

        def inference(*args, **kwargs):
            call_count[0] += 1
            prediction = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            if nondeterministic and call_count[0] >= 3:
                prediction[0] += 0.01
            release = np.asarray([0, 0, 1, 1] if causal else [1, 1, 1, 1])
            return {"final": prediction}, {
                "all_step_ms": [10.0, 20.0], "network_ms": [5.0, 10.0],
                "source_prepare_seconds": 0.3}, {
                "release_window": release, "output_time_ms": (release + 1) * 50,
                "event_readouts": readouts}

        def baseline(*args, **kwargs):
            return {"final": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)}, {
                "all_step_ms": [10.0, 20.0], "network_ms": [5.0, 10.0],
                "source_prepare_seconds": 0.3}, {
                "release_window": np.asarray([0, 0, 1, 1]),
                "output_time_ms": np.asarray([50, 50, 100, 100]), "event_readouts": 4}

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            argv = ["profile.py", "--config", "config.yaml", "--checkpoint", "model.pt",
                    "--data-root", "dataset", "--out", directory, "--device", device,
                    "--records", "1", "--repeats", "2"]
            stack.enter_context(mock.patch.object(P.sys, "argv", argv))
            loader = stack.enter_context(mock.patch.object(P, "load_checkpoint", return_value=payload))
            stack.enter_context(mock.patch.object(P, "load_flat_config", return_value=cfg))
            stack.enter_context(mock.patch.object(P, "validate_config"))
            stack.enter_context(mock.patch.object(P, "seed_everything"))
            stack.enter_context(mock.patch.object(P, "make_models", return_value=(net, head)))
            stack.enter_context(mock.patch.object(P, "EvUAVStream", return_value=Dataset()))
            stack.enter_context(mock.patch.object(P, "density_records", return_value=[0]))
            infer = stack.enter_context(mock.patch.object(P, "infer_record", side_effect=inference))
            stack.enter_context(mock.patch.object(P, "infer_v5", side_effect=baseline))
            stack.enter_context(mock.patch.object(P, "BoardPower", FakePower))
            stack.enter_context(mock.patch.object(P, "sync"))
            stack.enter_context(mock.patch.object(P.torch.cuda, "reset_peak_memory_stats"))
            stack.enter_context(mock.patch.object(P.torch.cuda, "max_memory_allocated", return_value=1024))
            stack.enter_context(mock.patch.object(P.torch.cuda, "get_device_name", return_value="Mock GPU"))
            stack.enter_context(mock.patch.object(P, "sha256", return_value="checkpoint-digest"))
            clock = itertools.count()
            stack.enter_context(mock.patch.object(P.time, "perf_counter", side_effect=lambda: next(clock) * 0.01))
            writer = stack.enter_context(mock.patch.object(P, "write_json"))
            P.main()
            result = writer.call_args[0][1]
            self.assertEqual(torch.device(loader.call_args[0][1]).type, "cpu")
            for call in infer.call_args_list:
                self.assertEqual(call[1]["modes"], ("final",))
                self.assertTrue(call[1]["step"])
                self.assertFalse(call[1]["diagnostics"])
            return result

    def test_profile_main_repeats_and_models_delay_without_auxiliary_readout(self):
        # A CUDA inference request must still load the full optimizer-bearing
        # checkpoint on CPU; only model weights should subsequently move.
        result = self.run_mock_profile(device="cuda:0")
        final = result["variants"]["v15_final"]
        self.assertEqual(final["events"], 8)
        self.assertEqual(len(final["records"]), 2)
        self.assertEqual(final["acquisition_delay_ms"]["count"], 8)
        self.assertAlmostEqual(final["acquisition_delay_ms"]["mean"], 50.5)
        self.assertAlmostEqual(final["single_device_replay_delay_ms"]["mean"], 70.5)
        self.assertTrue(all(row["repeat_max_abs_error"] == 0 for row in final["records"]))
        self.assertIsNone(final["board_energy"]["board_joules"])
        self.assertIsNone(final["board_energy"]["model_only_joules"])
        self.assertIn("disk loading", result["throughput_scope"]["events_per_second"])
        self.assertIn("including disk", result["throughput_scope"]["workload_events_per_second"])
        self.assertEqual(final["event_readout_dense_macs"], 8 * ((2 + 2) * 5 + 5))
        self.assertEqual(result["sensor_hw"], [15, 23])
        self.assertEqual(result["padded_canvas_hw"], [16, 24])
        self.assertFalse(result["checkpoint_probe_only"])

    def test_profile_main_supports_causal_without_a_revision_module(self):
        result = self.run_mock_profile(causal=True)
        self.assertEqual(result["revision_parameters"], 0)
        self.assertEqual(result["revision_dense_macs_per_pair_plus_new_encoding"], 0)
        final = result["variants"]["v15_final"]
        self.assertAlmostEqual(final["acquisition_delay_ms"]["mean"], 25.5)

    def test_nondeterministic_repeats_and_extra_event_readouts_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "nondeterministic"):
            self.run_mock_profile(nondeterministic=True)
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            self.run_mock_profile(readouts=8)

    def test_profile_configuration_rejects_geometry_revision_and_design_mismatch(self):
        cfg = {"height": 260, "width_px": 346, "revision_radius": 2,
               "revision_source": "observation", "root": "original", "_private": 2}
        checkpoint = {"design": P.DESIGN, "config": cfg}
        copied = dict(cfg, root="relocated")
        self.assertEqual(P.validate_profile_checkpoint(checkpoint, copied), cfg)
        for key, value in (("height", 264), ("width_px", 352), ("revision_radius", 3),
                           ("revision_source", "deep")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "mismatch: " + key):
                P.validate_profile_checkpoint(checkpoint, dict(cfg, **{key: value}))
        with self.assertRaisesRegex(ValueError, "current V15"):
            P.validate_profile_checkpoint(dict(checkpoint, design="legacy-model"), cfg)

    def test_unknown_probe_scope_is_not_a_formal_training_checkpoint(self):
        self.assertFalse(P.training_is_complete_scope({}))
        self.assertFalse(P.training_is_complete_scope({"_training_scope": {"limit_records": None}}))
        self.assertFalse(P.training_is_complete_scope({"_training_scope": {"limit_records": None, "max_chunks": 3}}))
        self.assertTrue(P.training_is_complete_scope({"_training_scope": {"limit_records": None, "max_chunks": None}}))

    def test_zero_or_negative_energy_duration_is_not_integrated(self):
        power = P.BoardPower(torch.device("cpu"))
        power.samples = [(0., 100.), (3., 100.)]
        for start, end in ((1., 1.), (2., 1.)):
            with self.subTest(start=start, end=end):
                self.assertIsNone(power.finish(start, end)["board_joules"])


if __name__ == "__main__":
    unittest.main()
