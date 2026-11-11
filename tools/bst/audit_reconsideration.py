"""Trace old-evidence reconsideration on sampled real test queries.

Same checkpoint weights, four inference interventions; NOT retrained ablations,
accuracy estimates, causal attribution, or a speed benchmark. Complete causal
prefixes are replayed even if a deployed birth/stop policy would have exited.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bst.explain import explain_query
from tools.bst.pipeline import checkpoint_scope, file_digest, get_record, load_model, save_json


MODES = (("full", True, True), ("no_reweight", False, True),
         ("no_revalue", True, False), ("neither", False, False))
BELIEF_FIELDS = ("score", "correction", "context", "anchor", "prior", "polarity", "mass", "neff", "count")


def old_token_changes(trace):
    """Compare the same old IDs only when new neighbors actually arrived."""
    sums = {key: 0. for key in ("gate", "value", "normalized_weight", "total_contribution")}
    counts = dict.fromkeys(sums, 0)
    maximum = dict.fromkeys(sums, 0.)
    observations = transitions = sign_flips = 0
    previous = {}
    for step in trace["steps"]:
        current = {t["neighbor_id"]: t for t in step["tokens"]}
        common = sorted(set(previous) & set(current))
        if step["new_neighbors"] > 0 and common:
            transitions += 1
            observations += len(common)
            for key in common:
                old, new = previous[key], current[key]
                for field in sums:
                    delta = np.abs(np.asarray(new[field])-np.asarray(old[field]))
                    sums[field] += float(delta.sum())
                    counts[field] += int(delta.size)
                    maximum[field] = max(maximum[field], float(delta.max()))
                sign_flips += int(old["total_contribution"] * new["total_contribution"] < 0)
        previous = current
    return {"new_arrival_transitions_with_old_tokens": transitions,
            "old_token_observations": observations, "contribution_sign_flips": sign_flips,
            "mean_absolute_change": {key: sums[key]/max(counts[key], 1) for key in sums},
            "max_absolute_change": maximum}


def audit_query(model, recording, config, event_id, device="cpu"):
    """Reuse explain_query's actual applied contributions and causal cache."""
    old_flags = model.config.reweight, model.config.revalue
    if old_flags != (True, True):
        raise ValueError("the audited checkpoint must train both reweight and revalue mechanisms")
    traces = {}
    original = model.reaudit
    try:
        for name, reweight, revalue in MODES:
            model.config.reweight, model.config.revalue = reweight, revalue
            checks = []

            def checked(encoded, indices, relative, state, prior, polarity, age, **kwargs):
                result = original(encoded, indices, relative, state, prior, polarity, age, **kwargs)
                score, updated, _ = result
                # Candidate diagnostic gates can change because context was
                # updated. They are not applied again: check committed belief.
                frozen_score, frozen, diag = original(encoded, indices, relative, updated,
                                                       prior, polarity, age, **kwargs)
                differences = {key: float((updated[key]-frozen[key]).abs().max()) for key in BELIEF_FIELDS}
                exact = torch.equal(score, frozen_score) and all(torch.equal(updated[k], frozen[k]) for k in BELIEF_FIELDS)
                checks.append({"age": int(age), "exact_belief_no_op": bool(exact),
                               "updated_again": bool(diag["updated"].any()),
                               "max_absolute_belief_change": max(differences.values())})
                if not exact or bool(diag["updated"].any()):
                    raise RuntimeError("replaying an unchanged prefix changed the applied belief")
                return result

            model.reaudit = checked
            trace = explain_query(model, recording, config["evidence"], event_id,
                                  threshold=config["threshold"], device=device)
            trace["same_prefix_checks"] = checks
            trace["old_token_summary"] = old_token_changes(trace)
            trace["intervention"] = {"reweight": reweight, "revalue": revalue, "retrained": False}
            traces[name] = trace
        baseline = np.asarray([s["logit"] for s in traces["full"]["steps"]])
        tau = traces["full"]["decision_logit"]
        for trace in traces.values():
            scores = np.asarray([s["logit"] for s in trace["steps"]])
            trace["comparison_to_full"] = {"mean_absolute_prefix_logit_difference": float(np.abs(scores-baseline).mean()),
                "max_absolute_prefix_logit_difference": float(np.abs(scores-baseline).max()),
                "deadline_logit_difference": float(scores[-1]-baseline[-1]),
                "prefix_decision_disagreements": int(((scores >= tau) != (baseline >= tau)).sum())}
    finally:
        model.reaudit = original
        model.config.reweight, model.config.revalue = old_flags
    return traces


def checked_inputs(priors, checkpoint, data_root=None, allow_probe=False):
    folder = Path(priors)
    if (folder/"z0_test").is_dir():
        folder = folder/"z0_test"
    manifest_file = folder/"prior_provenance.json"
    if not manifest_file.is_file():
        raise ValueError("test priors require a prior_provenance.json identity manifest")
    metadata = json.loads(manifest_file.read_text(encoding="utf-8"))
    if metadata.get("split") != "test":
        raise ValueError("this diagnostic expects test priors")
    expected_hash = checkpoint.get("manifest", {}).get("prior_provenance", {}).get("backbone_sha256")
    actual_hash = metadata.get("backbone_sha256")
    if expected_hash and actual_hash and expected_hash != actual_hash:
        raise ValueError("checkpoint and test-prior backbone identities do not match")
    identity_verified = bool(expected_hash and actual_hash and expected_hash == actual_hash)
    if not identity_verified and not allow_probe:
        raise ValueError("checkpoint and test-prior backbone identities are not verified")
    root = Path(data_root or metadata.get("data_root", ""))
    if not (root/"test").is_dir():
        raise ValueError("raw test recordings are needed; supply --data-root")
    available = {p.name for p in (root/"test").glob("*.npz")}
    expected = set(metadata.get("expected_recordings", []))
    completed = set(metadata.get("completed_recordings", []))
    exported = {p.name for p in folder.glob("*.npz")}
    if not completed or not completed.issubset(available) or not completed.issubset(exported):
        raise ValueError("prior manifest lists no recordings or missing/unknown recordings")
    complete = bool(metadata.get("complete")) and expected == completed == available == exported
    train_prior = checkpoint.get("manifest", {}).get("prior_provenance", {})
    scope = checkpoint_scope(checkpoint, str(root))
    probe = not complete or not identity_verified or scope["probe"] or not train_prior.get("verified") or not train_prior.get("complete")
    if probe and not allow_probe:
        raise ValueError("complete test priors and full verified training are required; local smoke may use --allow-probe")
    return folder, root, sorted(completed), {"probe": probe, "complete_test_priors": complete,
        "training_scope": scope, "backbone_sha256": actual_hash, "identity_verified": identity_verified,
        "prior_manifest_sha256": file_digest(manifest_file)}


def pack_tokens(reports, heads):
    scalar_names = ("query_index", "mode_index", "age", "neighbor_id", "seen_in_previous_prefix", "total_contribution")
    vector_names = ("gate", "value", "normalized_weight", "contribution_per_head")
    arrays = {name: [] for name in scalar_names+vector_names}
    for qi, query in enumerate(reports):
        for mi, (mode, _, _) in enumerate(MODES):
            for step in query["traces"][mode]["steps"]:
                for token in step["tokens"]:
                    for key, value in (("query_index", qi), ("mode_index", mi), ("age", step["age"]),
                                       ("neighbor_id", token["neighbor_id"]),
                                       ("seen_in_previous_prefix", token["seen_in_previous_prefix"]),
                                       ("total_contribution", token["total_contribution"])):
                        arrays[key].append(value)
                    for key in vector_names:
                        arrays[key].append(token[key])
    return {key: np.asarray(value, dtype=np.float32 if key in vector_names or key == "total_contribution" else np.int64)
            .reshape((-1, heads) if key in vector_names else (-1,)) for key, value in arrays.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--priors", required=True, help="z0_test directory, or its parent")
    parser.add_argument("--data-root", help="raw root; defaults to prior export manifest data_root")
    parser.add_argument("--out", required=True, help="new directory for audit.json and tokens.npz")
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--seed", type=int, default=37, help="diagnostic sampling only")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-probe", action="store_true", help="mark incomplete local smoke inputs as probe")
    args = parser.parse_args(argv)
    if args.max_queries < 1 or args.threads < 1 or not 0 <= args.seed < 2**32:
        parser.error("positive sample/thread limits and a uint32 sampling seed are required")
    destination = Path(args.out)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("choose a new audit output directory")
    torch.set_num_threads(args.threads)
    model, checkpoint = load_model(args.checkpoint, args.device)
    config = checkpoint["config"]
    folder, raw_root, names, provenance = checked_inputs(args.priors, checkpoint, args.data_root, args.allow_probe)
    rng = np.random.default_rng(args.seed)
    order = [names[i] for i in rng.permutation(len(names))]
    reports = []
    # Balanced across recordings; sample before inspecting model effects or labels.
    quota, remainder = divmod(args.max_queries, len(order))
    for ri, name in enumerate(order):
        count = quota + int(ri < remainder)
        if count == 0:
            continue
        record, _ = get_record((str(raw_root/"test"/name), str(folder/name)), config)
        if not np.isclose(record.log_w, checkpoint["log_w"], atol=1e-6, rtol=0):
            raise ValueError("test prior score scale differs from training")
        with np.load(str(folder/name), allow_pickle=False) as source:
            if "complete" not in source or not bool(source["complete"].item()):
                raise ValueError("individual test recording is not a complete arrival export")
        eligible = np.flatnonzero(record.window < record.n_windows-1)
        ids = rng.choice(eligible, min(count, len(eligible)), replace=False)
        for event_id in ids:
            traces = audit_query(model, record, config, int(event_id), args.device)
            reports.append({"recording": name, "event_id": int(event_id),
                            "label_for_diagnostic_only": int(record.label[event_id]), "traces": traces})
            print("REAUDIT %d/%d %s event=%d old_token_observations=%d" %
                  (len(reports), args.max_queries, name, event_id,
                   traces["full"]["old_token_summary"]["old_token_observations"]), flush=True)
    if not reports:
        raise ValueError("no eligible queries with any remaining recording horizon")
    destination.mkdir(parents=True, exist_ok=True)
    arrays = pack_tokens(reports, model.config.heads)
    np.savez_compressed(str(destination/"tokens.npz"), **arrays)
    # Large token-level values live in NPZ; JSON retains prefix traces/summaries.
    for query in reports:
        for trace in query["traces"].values():
            for step in trace["steps"]:
                step["token_rows_in_npz"] = len(step.pop("tokens"))
    report = {"complete": True, "args": vars(args), "checkpoint_sha256": file_digest(args.checkpoint),
              "provenance": provenance, "queries_requested": args.max_queries, "queries_audited": len(reports),
              "sampling": "seeded recording-balanced uniform events with remaining horizon; no labels/outcomes used",
              "evidence_budget_unchanged": config["evidence"], "mode_indices": [m[0] for m in MODES],
              "token_arrays": "tokens.npz: query_index/mode_index/age/neighbor_id identify each row",
              "interpretation": "same-weight inference interventions, NOT retrained ablations or a population accuracy estimate",
              "normalized_weight_caveat": "disabling reweight fixes context-dependent affinity, not its changing normalization denominator",
              "no_op_scope": "same prefix repeated once; applied beliefs stay exact; previous_score/age bookkeeping is not an attribution",
              "limits": "birth is forced; no VoW early stop; later prefixes are diagnostic continuations; effects may be zero; no causal or calibration claim",
              "queries": reports}
    save_json(str(destination/"audit.json"), report)
    print("BST RECONSIDERATION AUDIT FINISHED: " + str(destination), flush=True)


if __name__ == "__main__":
    main()
