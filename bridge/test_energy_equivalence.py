"""SPEED full-system accounting vs the legacy operation counts of a frozen V2-1 / V3 evaluation (same measured
statistics in, same counts out). The only intended difference: SPEED's base verifier no longer runs the CUSUM membrane
accumulation (it fed only the dropped alarm output), so that legacy part is excluded."""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "SPEED"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert_checkpoint import convert_config  # noqa: E402
from speed.core.accounting import energy_parts  # noqa: E402
from speed.core.build import build_system  # noqa: E402

EVAL_JSON = os.path.join(ROOT, "Res", "v3f_results", "log", "v21_floor4_seed37", "eval_test_best_val_iou_seed37_v3f.json")


def close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


@unittest.skipUnless(os.path.isfile(EVAL_JSON), "frozen evaluation JSON not available")
class EnergyEquivalence(unittest.TestCase):
    def test_parts_match_legacy(self):
        with open(EVAL_JSON, encoding="utf-8") as handle:
            data = json.load(handle)
        r = data["carry"]
        system = build_system(convert_config(data["config"]))
        act = dict(zip(r["decision_activity"]["eps"], r["decision_activity"]["footprint_active_fraction"]))
        stats = {"events_per_step": r["events_per_window"], "input_density": r["input_nonzero_fraction"],
                 "firing_rates": {k: v["firing_rate"] for k, v in r["layers"].items()},
                 "verifier_active_fraction": act[0.03], "publish_unit_steps_per_step": 0.0}
        parts = energy_parts(system, 264, 352, stats)
        fe = r["operations_frontend"]
        self.assertTrue(close(parts["representation"]["mac"], fe["mac"]))
        self.assertTrue(close(parts["representation"]["ac"], fe["elementwise"]))
        self.assertTrue(close(parts["representation"]["transcendental"], fe["transcendental"]))
        bb = r["operations_backbone"]
        self.assertTrue(close(parts["backbone"]["mac"], bb["mac_event_driven"]))
        self.assertTrue(close(parts["backbone"]["ac"], bb["sop"]))
        legacy = r["operations_decision_sparse"]["0.03"]["per_part"]
        keep = [p for p in legacy if not p["part"].startswith("膜电位累加") and not p["part"].startswith("延迟读出")]
        reads = [p for p in legacy if p["part"].startswith("延迟读出")]
        for key, mine in (("mac", "mac"), ("elementwise", "ac"), ("transcendental", "transcendental")):
            self.assertTrue(close(parts["verifier"][mine], sum(p[key] for p in keep)), key)
            self.assertTrue(close(parts["readout_fixed_delay"][mine], sum(p[key] for p in reads)), key)


if __name__ == "__main__":
    unittest.main()
