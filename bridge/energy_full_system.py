"""Full-system energy of frozen V2-1 / V3 runs: the three parts already reported by tools/energy_v2.py
(backbone, front-end, decision layer) reproduced through SPEED's accounting, plus the two per-pixel output heads that
the legacy accounting never counted.

Head variants: dense (as implemented: both heads on every pixel of the padded canvas), mark-at-events (mark only where
events occur, the intensity head still dense because the decision layer needs it everywhere), and all-at-events
(lower bound, not achievable by the current design).

    python bridge/energy_full_system.py --eval-json Res/.../eval_test_*.json
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "SPEED"))

from speed.eval.energy import part, pixel_head, system_energy  # noqa: E402
from tools.energy_v2 import analyse  # noqa: E402

CANVAS = 264 * 352
HEAD = dict(c_in=12, hidden=16)


def legacy_parts(result, event_driven=True, decision_eps="0.03"):
    b = result["operations_backbone"]
    parts = {"backbone": part(mac=b["mac_event_driven"] if event_driven else b["mac"], ac=b["sop"]),
             "frontend": part(result["operations_frontend"]["mac"], result["operations_frontend"]["elementwise"],
                              result["operations_frontend"]["transcendental"])}
    d = result.get("operations_decision_sparse", {}).get(decision_eps) or result["operations_decision"]
    parts["decision"] = part(d["mac"], d["elementwise"], d["transcendental"])
    return parts


def head_parts(events_per_window):

    dense = pixel_head(n_out=2, positions=CANVAS, transcendental_per_position=1, **HEAD)
    g_dense = pixel_head(n_out=1, positions=CANVAS, transcendental_per_position=1, **HEAD)
    mark_events = part(mac=events_per_window * HEAD["hidden"], note="second layer only, hidden shared with g")
    at_events = pixel_head(n_out=2, positions=events_per_window, transcendental_per_position=1, **HEAD)
    return {"dense": {"heads": dense},
            "mark_at_events": {"heads_g_dense": g_dense, "heads_mark_extra": mark_events},
            "all_at_events": {"heads": at_events}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", nargs="+", required=True)
    parser.add_argument("--state-mode", default="carry")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = {}
    for path in args.eval_json:
        with open(path, "r", encoding="utf-8") as handle:
            result = json.load(handle)[args.state_mode]
        legacy = analyse(result, 160, 10.0)
        base = legacy_parts(result, event_driven=True, decision_eps="__dense__")
        check = system_energy(base, 50000, 8000000, (10.0,))["total"]["x10"]["mj_per_clip"]
        assert abs(check - legacy["total_event_driven_mj_per_8s"]) < 1e-9, (check, legacy)
        variants = head_parts(float(result["events_per_window"]))
        entry = {"legacy_event_driven_mj_per_8s": legacy["total_event_driven_mj_per_8s"], "variants": {}}
        for decision in ("dense", "0.03"):
            parts = legacy_parts(result, True, decision if decision != "dense" else "__dense__")
            for name, heads in variants.items():
                e = system_energy(dict(parts, **heads), 50000, 8000000)
                entry["variants"]["decision_%s/heads_%s" % (decision, name)] = {
                    "parts_mj_per_8s": {k: v["x10"]["mj_per_clip"] for k, v in e["parts"].items()},
                    "total_mj_per_8s": {k: v["mj_per_clip"] for k, v in e["total"].items()}}
        report[path] = entry
        print("\n#", path)
        print("legacy (backbone event-driven + front-end + dense decision): %.2f mJ/8s"
              % entry["legacy_event_driven_mj_per_8s"])
        for key, v in entry["variants"].items():
            parts = "  ".join("%s %.2f" % (k, x) for k, x in v["parts_mj_per_8s"].items())
            print("%-38s total %.2f mJ/8s (exp/log x1 %.2f, x20 %.2f) | %s" % (
                key, v["total_mj_per_8s"]["x10"], v["total_mj_per_8s"]["x1"], v["total_mj_per_8s"]["x20"], parts))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
