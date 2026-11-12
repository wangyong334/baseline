"""Actual revision backward preflight and isolation of caller runtime state."""
import contextlib
import io
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import torch

from tools.v15.runtime_check import check_revision_backward, _determinism


def settings():
    warn_getter = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
    return {"deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": bool(warn_getter()) if warn_getter is not None else False,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32}


def restore(values):
    _determinism(values["deterministic"], values["warn_only"])
    torch.backends.cudnn.deterministic = values["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = values["cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32 = values["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = values["matmul_tf32"]


class RuntimeCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        previous, rng = settings(), torch.random.get_rng_state().clone()
        self.addCleanup(restore, previous)
        self.addCleanup(torch.random.set_rng_state, rng)

    def test_real_cpu_backward_two_updates_finite_and_all_paths_active(self):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            report = check_revision_backward("cpu")
        self.assertIn("V15_RUNTIME_BACKWARD_OK", stream.getvalue())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["optimizer_steps"], 2)
        self.assertTrue(report["deterministic_algorithms"])
        self.assertFalse(report["tf32"])
        self.assertEqual(report["canvas_hw"], [16, 20])
        self.assertEqual(report["sensor_hw"], [13, 15])
        self.assertTrue(report["nonzero_output_initialization"])
        self.assertTrue(all(math.isfinite(value) for value in report["losses"]))
        for step in report["gradient_l1"]:
            for name in ("encoder.0.weight", "query.weight", "key.weight", "value.weight",
                         "output.weight", "gate.weight", "future_observation"):
                self.assertGreater(step[name], 0.0, name)

    def test_restores_rng_and_runtime_flags_for_enabled_and_disabled_callers(self):
        for enabled in (False, True):
            _determinism(enabled)
            torch.backends.cudnn.deterministic = not enabled
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            previous = settings()
            rng = torch.random.get_rng_state().clone()
            with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
                check_revision_backward(torch.device("cpu"))
                self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(settings(), previous)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_failure_is_early_and_restores_flags_and_rng(self):
        _determinism(False)
        before = settings()
        rng = torch.random.get_rng_state().clone()
        def failure(*args, **kwargs):
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            raise RuntimeError("deterministic CUDA backward unavailable")
        with mock.patch("tools.v15.runtime_check.ObservationRevision.revise", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "before calibration/training"):
                check_revision_backward("cpu")
        self.assertEqual(settings(), before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_does_not_change_an_existing_model(self):
        from model.observation_revision import ObservationRevision
        actual = ObservationRevision(channels=2, width=4, heads=2, patch=4)
        before = {key: value.clone() for key, value in actual.state_dict().items()}
        with contextlib.redirect_stdout(io.StringIO()):
            check_revision_backward("cpu")
        for key, value in actual.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_probe_itself_is_repeatable(self):
        with contextlib.redirect_stdout(io.StringIO()):
            first = check_revision_backward("cpu")
            second = check_revision_backward("cpu")
        self.assertEqual(first["losses"], second["losses"])
        self.assertEqual(first["gradient_l1"], second["gradient_l1"])

    def test_unavailable_cuda_never_silently_runs_cpu(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                check_revision_backward("cuda:0")

    def test_direct_cli_does_not_import_local_profile_as_stdlib(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-B", str(root / "tools/v15/runtime_check.py"),
                                 "--device", "cpu"], cwd=str(root), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("V15_RUNTIME_BACKWARD_OK", result.stdout)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime not present")
    def test_cuda_real_backward_and_all_rng_states_preserved(self):
        states = [state.clone() for state in torch.cuda.get_rng_state_all()]
        cpu_state = torch.random.get_rng_state().clone()
        before = settings()
        with contextlib.redirect_stdout(io.StringIO()):
            result = check_revision_backward("cuda:0")
        self.assertEqual(result["device"], "cuda:0")
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(len(result["gradient_l1"]), 2)
        self.assertEqual(settings(), before)
        self.assertTrue(torch.equal(cpu_state, torch.random.get_rng_state()))
        for expected, actual in zip(states, torch.cuda.get_rng_state_all()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
