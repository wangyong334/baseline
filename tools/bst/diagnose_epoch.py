"""Paired full-event validation diagnosis: BIR -> age-zero SDU -> later SDU.

Consumes existing evaluate.py exports; never fits, samples events, or selects a
threshold. Only the held-out selection half of VAL is accepted. Error changes
are algebraic prediction transitions, not causal effects of a learned module.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.final_analysis import report_prob


STAGES = ("bir", "arrival", "fixed")
PAIRS = (("bir", "arrival"), ("arrival", "fixed"), ("bir", "fixed"))
# Diagnostic bins only; not learned/model constants or selected from accuracy.
MARGINS = (-np.inf, -2., -1., 0., 1., 2., np.inf)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_export(path):
    with np.load(str(path), allow_pickle=False) as source:
        required = ("logits", "z0", "labels", "lag", "log_w")
        values = {key: np.asarray(source[key]) for key in required}
        for key in ("reasons", "original_ids"):
            if key in source:
                values[key] = np.asarray(source[key])
    n = len(values["labels"])
    for key in ("logits", "z0", "labels", "lag"):
        if values[key].shape != (n,) or not np.isfinite(values[key]).all():
            raise ValueError("invalid event vector %s in %s" % (key, path))
    if not np.isin(values["labels"], (0., 1.)).all():
        raise ValueError("labels must be binary")
    if (values["lag"] < 0).any() or (values["lag"] != np.floor(values["lag"])).any():
        raise ValueError("invalid event lags")
    if values["log_w"].size != 1 or not np.isfinite(values["log_w"]).all():
        raise ValueError("log_w must be one finite scalar")
    values["log_w"] = float(values["log_w"].reshape(-1)[0])
    for key in ("reasons", "original_ids"):
        if key in values and values[key].shape != (n,):
            raise ValueError("invalid %s shape" % key)
    return values


class Metrics:
    def __init__(self):
        self.count = self.tp = self.fp = self.fn = self.tn = 0
        self.brier = self.nll = self.delta = self.abs_delta = self.lag = 0.
        self.bin_count = np.zeros(15, np.int64)
        self.bin_probability = np.zeros(15, np.float64)
        self.bin_label = np.zeros(15, np.float64)

    def add(self, z, y, prediction, prior, lag):
        z, y = np.asarray(z, np.float64), np.asarray(y, bool)
        prediction = np.asarray(prediction, bool)
        self.count += len(z)
        self.tp += int((prediction & y).sum())
        self.fp += int((prediction & ~y).sum())
        self.fn += int((~prediction & y).sum())
        self.tn += int((~prediction & ~y).sum())
        prob = 1. / (1. + np.exp(-np.clip(z, -60., 60.)))
        self.brier += float(np.square(prob - y).sum())
        self.nll += float((np.logaddexp(0., z) - z * y).sum())
        self.delta += float((z - prior).sum())
        self.abs_delta += float(np.abs(z - prior).sum())
        self.lag += float(np.asarray(lag).sum())
        bins = np.minimum((prob * 15).astype(np.int64), 14)
        self.bin_count += np.bincount(bins, minlength=15)
        self.bin_probability += np.bincount(bins, weights=prob, minlength=15)
        self.bin_label += np.bincount(bins, weights=y, minlength=15)

    def result(self):
        def ratio(n, d):
            return float(n) / d if d else None
        return {"events": self.count, "tp": self.tp, "fp": self.fp,
                "fn": self.fn, "tn": self.tn,
                "iou": ratio(self.tp, self.tp + self.fp + self.fn),
                "target_recall_project_ACC": ratio(self.tp, self.tp + self.fn),
                "overall_classification_accuracy": ratio(self.tp + self.tn, self.count),
                "brier": ratio(self.brier, self.count), "nll": ratio(self.nll, self.count),
                "ece_15": ratio(np.abs(self.bin_probability - self.bin_label).sum(), self.count),
                "mean_logit_correction": ratio(self.delta, self.count),
                "mean_absolute_logit_correction": ratio(self.abs_delta, self.count),
                "mean_extra_wait_windows": ratio(self.lag, self.count),
                "calibration_bins": {"count": self.bin_count.tolist(),
                    "probability_sum": self.bin_probability.tolist(), "label_sum": self.bin_label.tolist()}}


def transitions(before, after, labels):
    a, b, y = (np.asarray(v, bool) for v in (before, after, labels))
    result = {"rescued_fn": int((~a & b & y).sum()), "introduced_fn": int((a & ~b & y).sum()),
              "removed_fp": int((a & ~b & ~y).sum()), "introduced_fp": int((~a & b & ~y).sum())}
    result["corrected"] = result["rescued_fn"] + result["removed_fp"]
    result["harmed"] = result["introduced_fn"] + result["introduced_fp"]
    result["net_corrected"] = result["corrected"] - result["harmed"]
    return result


def paired_record(arrival, fixed, threshold=.9):
    for key in ("z0", "labels"):
        if not np.array_equal(arrival[key], fixed[key]):
            raise ValueError("paired exports differ in event order/%s" % key)
    if arrival["log_w"] != fixed["log_w"]:
        raise ValueError("paired score scales differ")
    if (arrival["lag"] != 0).any():
        raise ValueError("arrival control must have zero extra wait")
    if ("original_ids" in arrival) != ("original_ids" in fixed):
        raise ValueError("only one export supplies original event IDs")
    if "original_ids" in arrival and not np.array_equal(arrival["original_ids"], fixed["original_ids"]):
        raise ValueError("paired original event IDs differ")
    z0, labels = arrival["z0"], arrival["labels"].astype(bool)
    logits = {"bir": z0, "arrival": arrival["logits"], "fixed": fixed["logits"]}
    predictions = {stage: report_prob(z, arrival["log_w"]) >= np.float32(threshold)
                   for stage, z in logits.items()}
    lags = {"bir": np.zeros(len(z0)), "arrival": arrival["lag"], "fixed": fixed["lag"]}
    groups = {"all": np.ones(len(z0), bool), "target": labels, "background": ~labels,
              "prior_fn": ~predictions["bir"] & labels, "prior_fp": predictions["bir"] & ~labels,
              "prior_tp": predictions["bir"] & labels, "prior_tn": ~predictions["bir"] & ~labels}
    margin = z0.astype(np.float64) + arrival["log_w"] - np.log(threshold / (1. - threshold))
    for index, (left, right) in enumerate(zip(MARGINS[:-1], MARGINS[1:])):
        groups["prior_margin_bin_%d" % index] = (margin >= left) & (margin < right)
    if "reasons" in arrival and "reasons" in fixed:
        for values in (arrival, fixed):
            if not np.isin(values["reasons"], ("birth", "deadline", "policy")).all():
                raise ValueError("unknown commit reason")
        a, b = arrival["reasons"] != "birth", fixed["reasons"] != "birth"
        if not np.array_equal(a, b):
            raise ValueError("paired policies admitted different queries")
        groups["birth_admitted"], groups["birth_direct"] = a, ~a
        for stage in ("arrival", "fixed"):
            if not np.array_equal(logits[stage][~a], z0[~a]):
                raise ValueError("direct-at-birth output changed")
            if (lags[stage][~a] != 0).any():
                raise ValueError("direct-at-birth event has a nonzero lag")
    return z0, labels, logits, predictions, lags, groups


def analyze(arrival_dir, fixed_dir):
    directories = {"arrival": Path(arrival_dir), "fixed": Path(fixed_dir)}
    summaries = {k: json.loads((v/"summary.json").read_text(encoding="utf-8")) for k,v in directories.items()}
    for stage, summary in summaries.items():
        if summary.get("split") != "val" or summary.get("subset") != "selection":
            raise ValueError("diagnosis accepts only VAL selection exports; never test or policy-fit data")
        if summary.get("birth") != "band" or summary.get("commit") != stage:
            raise ValueError("expected paired band-arrival / band-fixed exports")
    left, right = summaries["arrival"], summaries["fixed"]
    if not left.get("checkpoint_sha256") or left["checkpoint_sha256"] != right.get("checkpoint_sha256"):
        raise ValueError("paired exports must use the same checkpoint")
    if left["config"] != right["config"]:
        raise ValueError("paired configurations differ")
    if left.get("prior_provenance") != right.get("prior_provenance"):
        raise ValueError("paired prior provenance differs")
    threshold = float(left["config"]["threshold"])
    if threshold != .9:
        raise ValueError("official report threshold must be 0.9")
    names = {stage: {p.name for p in path.glob("*.npz")} for stage,path in directories.items()}
    if not names["arrival"] or names["arrival"] != names["fixed"]:
        raise ValueError("paired recording sets differ or are empty")
    for stage, summary in summaries.items():
        if {row["recording"] for row in summary["runtime"]} != names[stage]:
            raise ValueError("summary and export recording sets differ")
    pooled, changes, recordings, source_hashes = {}, {}, [], {}
    expected_log_w = None
    for name in sorted(names["arrival"]):
        values = {stage: load_export(path/name) for stage,path in directories.items()}
        if expected_log_w is None:
            expected_log_w = values["arrival"]["log_w"]
        elif not np.isclose(values["arrival"]["log_w"], expected_log_w, atol=1e-6, rtol=0):
            raise ValueError("recordings use mixed backbone score scales")
        z0, labels, logits, predictions, lags, groups = paired_record(values["arrival"], values["fixed"], threshold)
        rows, effects = {}, {}
        for group, mask in groups.items():
            rows[group], effects[group] = {}, {}
            for stage in STAGES:
                metric = Metrics()
                selected = (logits[stage][mask], labels[mask], predictions[stage][mask], z0[mask], lags[stage][mask])
                metric.add(*selected)
                pooled.setdefault(group, {}).setdefault(stage, Metrics()).add(*selected)
                rows[group][stage] = metric.result()
            for before, after in PAIRS:
                key = before+"_to_"+after
                effects[group][key] = transitions(predictions[before][mask], predictions[after][mask], labels[mask])
                total = changes.setdefault(group, {}).setdefault(key, {k: 0 for k in effects[group][key]})
                for field, value in effects[group][key].items():
                    total[field] += value
        recordings.append({"recording": name, "groups": rows, "changes": effects})
        source_hashes[name] = {stage: digest(path/name) for stage,path in directories.items()}
    for stage, source in (("bir", left.get("baseline", {})), ("arrival", left.get("model", {})),
                          ("fixed", right.get("model", {}))):
        actual = pooled["all"][stage].result()
        for key in ("tp", "fp", "fn"):
            if key in source and actual[key] != source[key]:
                raise ValueError("%s counts differ from evaluator summary: %s" % (stage, key))
    return {"complete": True, "split": "val", "subset": "selection", "recordings": recordings,
            "checkpoint_sha256": left["checkpoint_sha256"], "recording_count": len(recordings),
            "groups": {g: {s:m.result() for s,m in row.items()} for g,row in pooled.items()},
            "changes": changes, "source_sha256": source_hashes,
            "official_metrics_from_evaluation": {"bir": left.get("baseline"),
                "arrival": left.get("model"), "fixed": right.get("model")},
            "probes": {s: v.get("probe") for s,v in summaries.items()},
            "scope": "Full exported events of held-out selection recordings; not final test performance or a converged-model claim.",
            "calibration_scope": "sigmoid(unweighted logit), distinct from cost-shifted report-scale classification",
            "margin_bins": ["[-inf,-2)", "[-2,-1)", "[-1,0)", "[0,1)", "[1,2)", "[2,inf)"],
            "group_scope": "Diagnostic bins fixed in code, not tuned model constants; birth groups require matching reason arrays.",
            "alignment_scope": "Same checkpoint/config/recording sets and identical labels/priors; original IDs additionally checked when supplied.",
            "no_test_selection": True}


def markdown(report):
    lines = ["# First-epoch validation diagnosis", "", report["scope"], "",
             "| Stage | Events | TP | FP | FN | IoU | Brier | NLL |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for stage in STAGES:
        row = report["groups"]["all"][stage]
        values = [stage] + [str(row[k]) for k in ("events", "tp", "fp", "fn")]
        values += ["NA" if row[k] is None else "%.6f" % row[k] for k in ("iou", "brier", "nll")]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "| Transition | Rescued FN | Introduced FN | Removed FP | Introduced FP | Net corrected |",
              "|---|---:|---:|---:|---:|---:|"]
    for pair, row in report["changes"]["all"].items():
        lines.append("| " + " | ".join([pair] + [str(row[k]) for k in
            ("rescued_fn", "introduced_fn", "removed_fp", "introduced_fp", "net_corrected")]) + " |")
    lines += ["", "The JSON contains per-recording, admission, class and prior-margin groups.",
              "Calibration uses unweighted probabilities. Official Pd/Fa are copied from evaluator summaries, not recomputed without geometry."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("arrival", "fixed", "out"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    target = Path(args.out)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("choose an empty diagnostic directory")
    report = analyze(args.arrival, args.fixed)
    target.mkdir(parents=True, exist_ok=True)
    (target/"diagnosis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (target/"REPORT.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report), flush=True)


if __name__ == "__main__":
    main()
