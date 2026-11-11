"""Stdlib-only, per-seed final-metrics table; never select or average policies."""
import argparse
import json
import math
from pathlib import Path

SEEDS = (37, 38)
POLICIES = ("bir", "band_arrival", "band_fixed", "band_learned", "learned_learned")
LEGACY_POLICIES = ("bir", "band_fixed", "band_learned", "learned_learned")
PERFORMANCE = ("events", "recorded_seconds", "sum_recording_median_compute_seconds",
               "real_time_factor", "sum_recording_median_board_joules",
               "board_joules_per_million_events")


def finite(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def declared_policies(report):
    """Honor a report's predeclared set; pre-declaration-era files used four."""
    declared = report.get("args", {}).get("policies")
    source = "args.policies"
    if declared is None:
        declared, source = list(LEGACY_POLICIES), "legacy_four_policy_default"
    if (not isinstance(declared, list) or not declared or
            any(not isinstance(name, str) or name not in POLICIES for name in declared) or
            len(set(declared)) != len(declared)):
        raise ValueError("args.policies must declare a nonempty unique known policy list")
    if set(report.get("policies", {})) - set(declared):
        raise ValueError("reported policies differ from the predeclared args.policies")
    return tuple(declared), source


def summarize(inputs):
    """inputs: [(metrics_path, expected_seed_or_None)]; absent files remain visible."""
    reports, sources, errors, declared_sets = {}, [], [], {}
    for path, expected in inputs:
        path = Path(path)
        try:
            report = json.loads(path.read_text(encoding="utf-8-sig"))
            seed = report["config"]["train"]["seed"]
            if seed not in SEEDS or isinstance(seed, bool) or (expected is not None and seed != expected):
                raise ValueError("unexpected/mismatched seed %r (expected %r)" % (seed, expected))
            declared, declaration_source = declared_policies(report)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({"path": str(path), "expected_seed": expected, "error": str(error)})
            continue
        if seed in reports:
            raise ValueError("duplicate seed %s: provide exactly one metrics file per seed; no test-based selection" % seed)
        reports[seed] = report
        declared_sets[seed] = declared
        sources.append({"seed": seed, "path": str(path), "probe": report.get("probe"),
                        "expected_policies": list(declared), "policy_declaration_source": declaration_source,
                        "complete": report.get("complete"),
                        "complete_prior_provenance": report.get("complete_prior_provenance"),
                        "split": report.get("args", {}).get("split"),
                        "hardware": report.get("hardware"),
                        "checkpoint_sha256": report.get("checkpoint_sha256"),
                        "timing_scope": report.get("timing_scope"),
                        "energy_scope": report.get("energy_scope")})
    rows = []
    # Missing seed files inherit the known protocol union, not today's defaults.
    # Thus an intact old four-policy release never acquires a missing age-zero row.
    fallback = tuple(name for name in POLICIES if any(name in names for names in declared_sets.values())) or POLICIES
    expected_by_seed = {seed: declared_sets.get(seed, fallback) for seed in SEEDS}
    for seed in SEEDS:
        report = reports.get(seed, {})
        for name in expected_by_seed[seed]:
            policy = report.get("policies", {}).get(name)
            row = {"seed": seed, "policy": name, "available": policy is not None,
                   "probe": report.get("probe"), "complete": report.get("complete", False),
                   "complete_prior_provenance": report.get("complete_prior_provenance")}
            policy = policy or {}
            accuracy = policy.get("accuracy_and_evidence_wait", {})
            row.update({key: finite(accuracy.get(key)) for key in ("iou", "acc", "pd", "fa")})
            row.update({key: finite(policy.get("performance", {}).get(key)) for key in PERFORMANCE})
            row["energy_missing"] = row["sum_recording_median_board_joules"] is None
            checks = [repeat.get("energy", {}).get("no_other_compute_at_boundaries")
                      for recording in policy.get("recordings", []) for repeat in recording.get("repeats", [])]
            row["energy_exclusivity_not_confirmed"] = not checks or any(check is not True for check in checks)
            row["energy_boundaries_are_not_interval_exclusivity_proof"] = True
            row["status"] = "missing" if not row["available"] else (
                "partial" if row["complete"] is not True else "probe" if row["probe"] is not False else "complete")
            rows.append(row)
    coverage = all(row["available"] and row["complete"] is True for row in rows) and not errors
    same_protocol = len({frozenset(names) for names in declared_sets.values()}) <= 1
    return {"expected_seeds": list(SEEDS), "expected_policies": list(fallback),
            "expected_policies_by_seed": {str(seed): list(names) for seed, names in expected_by_seed.items()},
            "policy_sets_match": same_protocol,
            "missing_seeds": [seed for seed in SEEDS if seed not in reports],
            "measurements_complete": coverage,
            "final_ready": coverage and same_protocol and all(row["probe"] is False and row["complete_prior_provenance"] is True for row in rows)
                and all(source["split"] == "test" for source in sources),
            "energy_complete": all(not row["energy_missing"] for row in rows),
            "selection": "No cross-seed averaging or test-based policy selection; copy declared aggregate metrics.",
            "metric_definitions": {
                "iou": "TP/(TP+FP+FN), pooled event counts; JSON fraction, table percent",
                "acc": "Project ACC=TP/(TP+FN), target-event recall, NOT overall classification accuracy; JSON fraction, table percent",
                "pd": "Existing FastRoc detected/total target-frame groups (50 ms); JSON fraction, table percent",
                "fa": "Existing FastRoc false-positive 8-connected components/(frame_count*height*width); raw value, NOT event FPR",
                "sum_recording_median_compute_seconds": "Sum of each recording's median repeated compute time; not job wall time or sum of all repeats",
                "real_time_factor": "sum_recording_median_compute_seconds / recorded_seconds; smaller is faster",
                "board_joules_per_million_events": "sum_recording_median_board_joules / events * 1e6; no averaging of per-recording rates",
                "energy": "Gross GPU-board joules, no idle subtraction; no CPU/system/neuromorphic or process-exclusive attribution. Boundary checks never prove exclusivity throughout the interval."},
            "sources": sources, "errors": errors, "rows": rows}


def markdown(summary):
    def fmt(value, percent=False):
        return "NA" if value is None else ("%.4f" % (100 * value) if percent else "%.6g" % value)
    def flag(value):
        return "?" if value is None else str(value).lower()
    lines = ["# Final metrics by seed", "",
             "Final ready: **%s**; measurements complete: **%s**; energy complete: **%s**." %
             tuple(str(summary[key]).lower() for key in ("final_ready", "measurements_complete", "energy_complete")),
             "Missing seeds: %s. No averaging or test-based best-policy selection." % (summary["missing_seeds"] or "none"), "",
             "Expected policies follow each report's `args.policies` (legacy fallback: four policies); policy sets match: **%s**." % str(summary["policy_sets_match"]).lower(), "",
             "| Seed | Policy | Status | Probe | Complete | IoU % | ACC % | Pd % | Fa raw | RTF | Compute s | Board J | J/million events | Energy missing | Exclusivity unconfirmed |",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in summary["rows"]:
        values = [str(row["seed"]), row["policy"], row["status"], flag(row["probe"]), flag(row["complete"])]
        values += [fmt(row[key], True) for key in ("iou", "acc", "pd")]
        values += [fmt(row[key]) for key in ("fa", "real_time_factor", "sum_recording_median_compute_seconds",
                                             "sum_recording_median_board_joules", "board_joules_per_million_events")]
        values += [flag(row[key]) for key in ("energy_missing", "energy_exclusivity_not_confirmed")]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "`complete=true` means that file's requested evaluation finished; `probe=true` still prevents a final-experiment claim.",
              "`final_ready` only describes complete non-probe test files with matching declared policy sets, not model quality or publication readiness.",
              "`partial` values describe only recordings processed so far. Missing values stay NA, never zero.", ""]
    lines += ["- **%s**: %s" % item for item in summary["metric_definitions"].items()]
    for source in summary["sources"]:
        lines += ["", "Seed %s source: `%s`; split=%s; complete prior provenance=%s." %
                  (source["seed"], source["path"], source["split"], flag(source["complete_prior_provenance"])),
                  "Declared policies: %s; declaration source=%s." % (", ".join(source["expected_policies"]), source["policy_declaration_source"]),
                  "Timing scope: %s" % (source["timing_scope"] or "unspecified"),
                  "Energy scope: %s" % (source["energy_scope"] or "unspecified")]
    if summary["errors"]:
        lines += ["", "Input errors (summary is incomplete):"]
        lines += ["- `%s`: %s" % (error["path"], error["error"]) for error in summary["errors"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--release", type=Path, help="release.json; locate jobs[].out/final_metrics/metrics.json")
    inputs.add_argument("--metrics", nargs="+", type=Path, help="downloaded or server metrics.json files")
    parser.add_argument("--out", type=Path, help="write final_summary.md and final_summary.json")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown", help="terminal output format")
    args = parser.parse_args(argv)
    try:
        if args.release:
            spec = json.loads(args.release.read_text(encoding="utf-8-sig"))
            root = Path(spec.get("root", Path(__file__).resolve().parents[2]))
            files = [(Path(job["out"]) if Path(job["out"]).is_absolute() else root / job["out"], job["seed"])
                     for job in spec["jobs"]]
            files = [(path / "final_metrics" / "metrics.json", seed) for path, seed in files]
        else:
            files = [(path, None) for path in args.metrics]
        summary = summarize(files)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    md = markdown(summary)
    encoded = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "final_summary.md").write_text(md, encoding="utf-8")
        (args.out / "final_summary.json").write_text(encoded, encoding="utf-8")
    print(md if args.format == "markdown" else encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
