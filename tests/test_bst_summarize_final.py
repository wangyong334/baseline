"""Final-summary handling of partial results, rate denominators and missing energy."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.bst.summarize_final import LEGACY_POLICIES, POLICIES, markdown, summarize


class FinalSummaryTests(unittest.TestCase):
    def complete_reports(self, declared_by_seed, omit=()):
        with tempfile.TemporaryDirectory() as directory:
            inputs = []
            for seed, declared in declared_by_seed.items():
                path = Path(directory)/("seed%s.json" % seed)
                report = {"config": {"train": {"seed": seed}}, "probe": False, "complete": True,
                          "complete_prior_provenance": True, "args": {"split": "test", "policies": list(declared)},
                          "policies": {name: {"performance": {"sum_recording_median_board_joules": 1.}}
                                       for name in declared if (seed, name) not in omit}}
                path.write_text(json.dumps(report), encoding="utf-8")
                inputs.append((path, seed))
            return summarize(inputs)

    def test_old_declared_four_policy_release_is_complete_without_age_zero(self):
        result = self.complete_reports({37: LEGACY_POLICIES, 38: LEGACY_POLICIES})
        self.assertTrue(result["final_ready"])
        self.assertTrue(result["measurements_complete"])
        self.assertEqual(len(result["rows"]), 8)
        self.assertNotIn("band_arrival", result["expected_policies"])

    def test_new_five_policy_release_requires_declared_age_zero(self):
        result = self.complete_reports({37: POLICIES, 38: POLICIES})
        self.assertTrue(result["final_ready"])
        self.assertEqual(len(result["rows"]), 10)
        missing = self.complete_reports({37: POLICIES, 38: POLICIES}, omit=((38, "band_arrival"),))
        self.assertFalse(missing["measurements_complete"])
        rows = [row for row in missing["rows"] if row["status"] == "missing"]
        self.assertEqual([(row["seed"], row["policy"]) for row in rows], [(38, "band_arrival")])

    def test_mixed_protocols_are_separate_without_inventing_missing_old_policy(self):
        result = self.complete_reports({37: LEGACY_POLICIES, 38: POLICIES})
        self.assertTrue(result["measurements_complete"])
        self.assertFalse(result["policy_sets_match"])
        self.assertFalse(result["final_ready"])
        self.assertEqual(len(result["rows"]), 9)
        self.assertTrue(all(row["status"] == "complete" for row in result["rows"]))

    def test_explicit_single_policy_probe_does_not_require_unrequested_reruns(self):
        result = self.complete_reports({37: ("band_arrival",)})
        self.assertEqual(result["expected_policies"], ["band_arrival"])
        self.assertEqual([(row["seed"], row["policy"]) for row in result["rows"]],
                         [(37, "band_arrival"), (38, "band_arrival")])
        self.assertEqual(result["missing_seeds"], [38])

    def test_partial_seed_preserves_totals_missing_energy_and_competition(self):
        # Totals represent unequal recordings (100 and 900 events, 2 and 8 s).
        # A rate average would be wrong; the summary must preserve the given totals.
        report = {"config": {"train": {"seed": 37}}, "probe": True, "complete": False,
                  "complete_prior_provenance": True, "policies": {"bir": {
                      "accuracy_and_evidence_wait": {"iou": .6, "acc": .75, "pd": .9, "fa": 1e-5},
                      "performance": {"events": 1000, "recorded_seconds": 100,
                                      "sum_recording_median_compute_seconds": 10,
                                      "real_time_factor": .1, "sum_recording_median_board_joules": 35,
                                      "board_joules_per_million_events": 35000},
                      "recordings": [{"repeats": [{"energy": {"no_other_compute_at_boundaries": False}}]}]},
                      "band_fixed": {"accuracy_and_evidence_wait": {}, "performance": {
                          "sum_recording_median_board_joules": None, "board_joules_per_million_events": None}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = summarize([(path, 37), (Path(directory) / "missing.json", 38)])
        self.assertEqual(result["missing_seeds"], [38])
        self.assertFalse(result["final_ready"])
        self.assertFalse(result["measurements_complete"])
        self.assertFalse(result["energy_complete"])
        self.assertEqual(len(result["rows"]), 8)
        row, missing_energy = result["rows"][:2]
        self.assertEqual(row["status"], "partial")
        self.assertTrue(row["probe"])
        self.assertEqual(row["sum_recording_median_compute_seconds"], 10)
        self.assertEqual(row["real_time_factor"], .1)
        self.assertEqual(row["board_joules_per_million_events"], 35000)
        self.assertEqual(row["sum_recording_median_board_joules"], 35)  # competition flags never erase measured joules
        self.assertTrue(row["energy_exclusivity_not_confirmed"])
        self.assertTrue(missing_energy["energy_missing"])
        self.assertIsNone(missing_energy["board_joules_per_million_events"])
        self.assertTrue(missing_energy["energy_exclusivity_not_confirmed"])
        self.assertEqual(result["rows"][4]["status"], "missing")
        text = markdown(result)
        self.assertIn("ACC=TP/(TP+FN)", text)
        self.assertIn("| 37 | bir | partial | true | false | 60.0000 | 75.0000 | 90.0000 |", text)
        self.assertIn("Missing seeds: [38]", text)
        self.assertIn("NA", text)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
