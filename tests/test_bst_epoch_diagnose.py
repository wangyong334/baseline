"""Validation diagnostics preserve event pairing and separate error changes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.bst.diagnose_epoch import analyze, main, Metrics, paired_record, transitions


class EpochDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.arrival, self.fixed = self.root/"arrival", self.root/"fixed"
        self.arrival.mkdir(); self.fixed.mkdir()
        self.labels = np.array([1, 1, 0, 0, 1, 0], np.float32)
        self.prior = np.array([-4, 4, 4, -4, 4, -4], np.float32)
        self.values = {
            "arrival": dict(logits=np.array([4, -4, -4, 4, 4, -4], np.float32),
                lag=np.zeros(6), reasons=np.array(["policy"]*4+["birth"]*2)),
            "fixed": dict(logits=np.array([4, 4, -4, -4, 4, -4], np.float32),
                lag=np.array([2]*4+[0]*2), reasons=np.array(["deadline"]*4+["birth"]*2))}
        for values in self.values.values():
            values.update(z0=self.prior.copy(), labels=self.labels.copy(), log_w=0.)
        for stage, path in (("arrival", self.arrival), ("fixed", self.fixed)):
            np.savez(str(path/"val_001.npz"), **self.values[stage])
            report = {"split": "val", "subset": "selection", "birth": "band", "commit": stage,
                "checkpoint_sha256": "same-checkpoint", "prior_provenance": {"complete": True},
                "config": {"threshold": .9}, "runtime": [{"recording": "val_001.npz"}], "probe": False,
                "baseline": {"pd": .8, "fa": .001}, "model": {"pd": .9, "fa": .002}}
            (path/"summary.json").write_text(json.dumps(report))

    def tearDown(self):
        self.temp.cleanup()

    def change_summary(self, path, **updates):
        p = path/"summary.json"
        value = json.loads(p.read_text()); value.update(updates)
        p.write_text(json.dumps(value))

    def test_error_changes_are_separate_from_net_accuracy_and_birth_groups(self):
        report = analyze(self.arrival, self.fixed)
        self.assertEqual(report["groups"]["all"]["bir"]["tp"], 2)
        self.assertEqual(report["groups"]["all"]["fixed"]["iou"], 1.)
        first = report["changes"]["all"]["bir_to_arrival"]
        self.assertEqual([first[k] for k in ("rescued_fn", "introduced_fn", "removed_fp", "introduced_fp")], [1]*4)
        self.assertEqual(first["net_corrected"], 0)
        self.assertEqual(report["changes"]["all"]["arrival_to_fixed"]["net_corrected"], 2)
        self.assertEqual(report["groups"]["birth_direct"]["fixed"]["events"], 2)
        self.assertEqual(report["groups"]["birth_direct"]["fixed"]["mean_logit_correction"], 0.)
        self.assertEqual(report["official_metrics_from_evaluation"]["bir"]["pd"], .8)

    def test_calibration_is_unshifted_and_combines_counts_not_recording_means(self):
        m = Metrics()
        m.add(np.zeros(3), np.array([1, 0, 0]), np.zeros(3, bool), np.zeros(3), np.zeros(3))
        m.add(np.zeros(1), np.array([1]), np.ones(1, bool), np.zeros(1), np.zeros(1))
        row = m.result()
        self.assertEqual(row["events"], 4)
        self.assertEqual(row["brier"], .25)
        self.assertAlmostEqual(row["nll"], np.log(2.))
        self.assertEqual(row["ece_15"], 0.)
        self.assertEqual(sum(row["calibration_bins"]["count"]), 4)
        empty = Metrics().result()
        self.assertIsNone(empty["iou"])
        self.assertIsNone(empty["brier"])

    def test_rejects_test_fit_partial_sets_and_checkpoint_mismatch(self):
        for updates in ({"split": "test"}, {"subset": "fit"}, {"checkpoint_sha256": "different"},
                        {"runtime": [{"recording": "val_003.npz"}]}):
            original = (self.fixed/"summary.json").read_text()
            self.change_summary(self.fixed, **updates)
            with self.assertRaises(ValueError):
                analyze(self.arrival, self.fixed)
            (self.fixed/"summary.json").write_text(original)

    def test_rejects_misaligned_rows_nonzero_arrival_lag_and_changed_direct_outputs(self):
        for key, replacement in (("z0", self.prior[::-1]), ("labels", 1-self.labels),
                                  ("log_w", 1.), ("logits", np.zeros(6))):
            modified = copy.deepcopy(self.values["fixed"]); modified[key] = replacement
            with self.assertRaises(ValueError):
                paired_record(self.values["arrival"], modified)
        arrival = copy.deepcopy(self.values["arrival"]); arrival["lag"][0] = 1
        with self.assertRaisesRegex(ValueError, "zero extra wait"):
            paired_record(arrival, self.values["fixed"])

    def test_missing_reasons_never_invents_admission_groups_and_output_preserved(self):
        arrival, fixed = (copy.deepcopy(self.values[s]) for s in ("arrival", "fixed"))
        arrival.pop("reasons"); fixed.pop("reasons")
        groups = paired_record(arrival, fixed)[-1]
        self.assertNotIn("birth_admitted", groups)
        out = self.root/"report"
        main(["--arrival", str(self.arrival), "--fixed", str(self.fixed), "--out", str(out)])
        self.assertTrue((out/"diagnosis.json").is_file())
        with self.assertRaises(FileExistsError):
            main(["--arrival", str(self.arrival), "--fixed", str(self.fixed), "--out", str(out)])


if __name__ == "__main__":
    unittest.main()
