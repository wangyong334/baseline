"""Aggregate runtime/energy denominators and driver-counter fallback semantics."""
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.bst import BSTConfig, SharedSequentialModel
from tools.bst.energy_meter import EnergyInterval, NVMLBoard, integrate_power
from tools.bst.final_metrics import POLICIES, calibration_stats, commit_policy, pool_calibration, summarize
from tools.bst.data import Recording
from tools.bst.stream import run_recording


def recording(events, duration, times, energies):
    return {"events": events, "duration_seconds": duration,
            "repeats": [{"seconds": t, "energy": {"board_joules": e}}
                        for t, e in zip(times, energies)]}


class FinalMetricAggregation(unittest.TestCase):
    def test_age_zero_control_refines_once_and_matches_fixed_first_prefix(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)
        n = 6
        rec = Recording("age0", np.zeros(n), np.zeros(n), np.asarray([0, 10, 50, 60, 100, 110]),
                        np.ones(n), np.zeros(n, np.float32), np.zeros(n), 50, n_windows=5)
        model = SharedSequentialModel(BSTConfig(source_dim=8, state_dim=6, relation_dim=3,
            heads=2, max_wait_steps=2, past_windows=2, normalization="window"))
        with torch.no_grad():
            model.value_bias.fill_(.4)
        evidence = dict(past_windows=2, max_wait=2, k_near=2, k_evidence=1, radius=0, filter_logit=0)
        arrival = run_recording(model, rec, evidence, birth="band", batch_size=3,
                                policy=commit_policy("band_arrival", None))
        fixed = run_recording(model, rec, evidence, birth="band", batch_size=3,
                              policy=commit_policy("band_fixed", None), return_traces=True)
        self.assertIn("band_arrival", POLICIES)
        self.assertEqual(arrival["stats"]["births"], n)
        self.assertEqual(arrival["stats"]["queries_updated"], n)
        self.assertEqual(arrival["stats"]["remaining_active"], 0)
        np.testing.assert_array_equal(arrival["lags"], np.zeros(n))
        self.assertTrue(np.all(arrival["logits"] > rec.z0))  # SDU ran; this is not BIR-only.
        self.assertGreater(fixed["stats"]["queries_updated"], n)
        for trace in fixed["traces"]:
            born = rec.window[trace["ids"]] == trace["window"]
            np.testing.assert_allclose(arrival["logits"][trace["ids"][born]], trace["logits"][born], atol=2e-6)

    def test_unmeasured_results_cannot_report_zero_energy(self):
        with self.assertRaises(ValueError):
            summarize([])
        with self.assertRaises(ValueError):
            summarize([recording(100, 8, [], [])])

    def test_totals_use_recording_medians_not_average_of_rates(self):
        rows = [recording(100, 20, [1, 2, 100], [2, 5, 90]),
                recording(900, 80, [4, 8, 20], [10, 30, 100])]
        result = summarize(rows)
        self.assertEqual(result["events"], 1000)
        self.assertEqual(result["recorded_seconds"], 100)
        self.assertEqual(result["sum_recording_median_compute_seconds"], 10)
        self.assertEqual(result["events_per_second"], 100)
        self.assertEqual(result["real_time_factor"], .1)
        self.assertEqual(result["sum_recording_median_board_joules"], 35)
        self.assertEqual(result["board_joules_per_million_events"], 35000)

    def test_missing_repeat_energy_never_becomes_partial_total_or_zero(self):
        rows = [recording(100, 20, [1, 2], [5, None]),
                recording(900, 80, [4, 8], [10, 30])]
        result = summarize(rows)
        self.assertIsNone(result["sum_recording_median_board_joules"])
        self.assertIsNone(result["board_joules_per_million_events"])
        self.assertAlmostEqual(result["events_per_second"], 1000/7.5)

    def test_no_event_or_duration_denominator_is_not_infinite(self):
        result = summarize([recording(0, 0, [1], [2])])
        self.assertEqual(result["events_per_second"], 0)
        self.assertIsNone(result["real_time_factor"])
        self.assertIsNone(result["board_joules_per_million_events"])

    def test_probability_calibration_pooling_is_event_weighted(self):
        probabilities = np.asarray([.1, .9, .8, .2])
        logits = np.log(probabilities/(1-probabilities))
        labels = np.asarray([0, 1, 1, 0])
        whole = pool_calibration([calibration_stats(logits, labels)])
        partitioned = pool_calibration([calibration_stats(logits[:1], labels[:1]),
                                        calibration_stats(logits[1:], labels[1:])])
        self.assertEqual(whole, partitioned)
        self.assertAlmostEqual(whole["brier"], .025)
        self.assertAlmostEqual(whole["nll"], (-math.log(.9)-math.log(.8))/2)
        self.assertAlmostEqual(whole["ece_15_equal_width_bins"], .15)


class FakeThread:
    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def join(self):
        pass


class FakeBoard:
    def __init__(self, counters, powers):
        self.counters, self.powers = iter(counters), iter(powers)
        self.error = None

    def joules(self):
        return next(self.counters)

    def watts(self):
        return next(self.powers)


class FinalEnergyMeasurement(unittest.TestCase):
    def measure(self, counters, powers=(100., 100.)):
        meter = EnergyInterval(FakeBoard(counters, powers))
        with patch("tools.bst.energy_meter.threading.Thread", FakeThread), \
             patch("tools.bst.energy_meter.time.perf_counter", side_effect=[10., 10., 12., 12.]):
            meter.begin()
            return meter.end()

    def test_trapezoid_integrates_watts_times_seconds_as_joules(self):
        samples = [{"seconds": 0., "watts": 20.}, {"seconds": 2., "watts": 40.},
                   {"seconds": 3., "watts": 80.}]
        self.assertEqual(integrate_power(samples), 120.)
        self.assertIsNone(integrate_power(samples[:1]))
        self.assertIsNone(integrate_power([samples[0], {"seconds": 2., "watts": None}]))

    def test_nvml_milliunits_are_converted_once(self):
        board = object.__new__(NVMLBoard)
        board.scalar = lambda name, kind: 1500 if "Energy" in name else 85000
        self.assertEqual(board.joules(), 1.5)
        self.assertEqual(board.watts(), 85.)
        board.scalar = lambda name, kind: None
        self.assertIsNone(board.joules())
        self.assertIsNone(board.watts())

    def test_advancing_counter_has_priority_over_power_estimate(self):
        measured = self.measure([1000., 1010.])
        self.assertEqual(measured["board_joules"], 10.)
        self.assertEqual(measured["power_integral_joules"], 200.)
        self.assertEqual(measured["method"], "NVML_total_energy_counter")
        self.assertEqual(measured["measurement_seconds"], 2.)

    def test_missing_reset_or_stale_counter_uses_power_integral(self):
        for counters in ([None, None], [1000., None], [1000., 999.], [1000., 1000.]):
            measured = self.measure(counters)
            self.assertEqual(measured["board_joules"], 200.)
            self.assertIsNone(measured["counter_joules"])
            self.assertEqual(measured["method"], "NVML_power_trapezoidal_estimate")

    def test_unavailable_power_and_counter_stay_unavailable(self):
        measured = self.measure([None, None], (100., None))
        self.assertIsNone(measured["board_joules"])
        self.assertEqual(measured["method"], "unavailable")


if __name__ == "__main__":
    unittest.main()
