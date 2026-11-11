"""Inspect a single event's actual sequential evidence contributions.

Gate/value/contribution are algebraic quantities in this fitted neural model.
They are not causal effects, calibrated source membership probabilities, or
verified individual log-likelihood ratios. A full prefix correction replaces
the old correction; old evidence is not added a second time.
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from model.bst import posterior_decision
from tools.bst.data import NeighborIndex
from tools.bst.evaluate import load_policy
from tools.bst.pipeline import file_digest, get_record, load_model, save_json, threshold_costs
from tools.bst.stream import CausalSourceCache


def _probability(logit):
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60))))


def _old_changes(old, new):
    common = sorted(set(old).intersection(new))
    if not common:
        return {"count": 0, "mean_abs_gate_change": 0.0, "mean_abs_value_change": 0.0,
                "mean_abs_contribution_change": 0.0, "contribution_sign_flips": 0}
    gates, values, contributions, signs = [], [], [], 0
    for key in common:
        a, b = old[key], new[key]
        gates.extend(np.abs(np.asarray(b["gate"]) - np.asarray(a["gate"])).tolist())
        values.extend(np.abs(np.asarray(b["value"]) - np.asarray(a["value"])).tolist())
        contributions.append(abs(b["total_contribution"] - a["total_contribution"]))
        signs += int(b["total_contribution"] * a["total_contribution"] < 0)
    return {"count": len(common), "mean_abs_gate_change": float(np.mean(gates)),
            "mean_abs_value_change": float(np.mean(values)),
            "mean_abs_contribution_change": float(np.mean(contributions)),
            "contribution_sign_flips": signs}


def explain_query(model, recording, evidence_config, event_id, threshold=0.9,
                  policy=None, wait_cost=0.0, device="cpu"):
    """Return a diagnostic trace; continue counterfactually after policy commit.

    Every prefix uses only acquired evidence. Continuing after a hypothetical
    early commit is explicitly diagnostic, never reported as stopped execution.
    """
    event_id = int(event_id)
    if not 0 <= event_id < len(recording):
        raise IndexError("event_id outside recording")
    if not np.isfinite(wait_cost) or wait_cost < 0:
        raise ValueError("wait_cost must be finite and nonnegative")
    r, device = recording, torch.device(device)
    index = NeighborIndex(r, **evidence_config)
    if index.max_wait != model.config.max_wait_steps:
        raise ValueError("evidence and model horizons differ")
    tau, fp_cost, fn_cost = threshold_costs(threshold, r.log_w)
    initial = float(r.z0[event_id])
    prior = torch.as_tensor([initial], dtype=torch.float32, device=device)
    polarity = torch.as_tensor([r.p[event_id]], dtype=torch.float32, device=device)
    maximum = min(index.max_wait, r.n_windows - 1 - int(r.window[event_id]))
    was_training = model.training
    policy_training = policy.training if isinstance(policy, torch.nn.Module) else None
    model.to(device).eval()
    if isinstance(policy, torch.nn.Module):
        policy.to(device).eval()
    steps, applied, applied_age, applied_windows = [], {}, None, None
    first_policy_commit = None
    try:
        with torch.no_grad():
            state = model.make_state(prior, polarity)
            cache = CausalSourceCache(model, r, device, batch_size=1024)
            for age in range(maximum + 1):
                window = int(r.window[event_id]) + age
                batch = index.query([event_id], age=age)
                indices = batch["idx"]
                cache.ensure(indices[indices >= 0], window)
                last_slot = (index.past_windows + age + 1) * index.K
                indices = indices[:, :last_slot]
                relative = batch["relative"][:, :last_slot]
                local = torch.as_tensor(cache.local_indices(indices), dtype=torch.long, device=device)
                previous = copy.deepcopy(applied)
                score, state, diag = model.reaudit(cache.encoded, local,
                                                   torch.as_tensor(relative, device=device),
                                                   state, prior, polarity, age, return_tokens=True)
                updated = bool(diag["updated"][0])
                if updated:
                    gate = diag["gate"][0].cpu().numpy()
                    value = diag["value"][0].cpu().numpy()
                    valid = diag["mask"][0].cpu().numpy()
                    mass = gate.sum(axis=0)
                    divisor = np.ones_like(mass) if model.config.normalization == "sum" else np.maximum(mass, 1e-8)
                    normalized = diag["effective_weight"][0].cpu().numpy() if "effective_weight" in diag else gate / divisor[None, :]
                    contribution = normalized * value
                    if "window_discount" in diag:
                        discount = diag["window_discount"][0].cpu().numpy()
                        window_value = diag["window_value"][0].cpu().numpy()
                        active = diag["window_active"][0].cpu().numpy()
                        applied_windows = [{"dw": int(g-model.config.past_windows),
                            "discount": discount[g].tolist(), "value": window_value[g].tolist(),
                            "contribution": (discount[g]*window_value[g]).tolist()}
                            for g in np.flatnonzero(active)]
                    applied = {}
                    for slot in np.flatnonzero(valid):
                        source = int(indices[0, slot])
                        applied[source] = {"neighbor_id": source, "x": int(r.x[source]), "y": int(r.y[source]),
                                           "time_ms": int(r.t[source]), "polarity": float(r.p[source]),
                                           "source_prior_logit": float(r.z0[source]),
                                           "dw": int(relative[0, slot, 3]),
                                           "relative": relative[0, slot].tolist(),
                                           "gate": gate[slot].tolist(), "value": value[slot].tolist(),
                                           "normalized_weight": normalized[slot].tolist(),
                                           "contribution_per_head": contribution[slot].tolist(),
                                           "total_contribution": float(contribution[slot].sum())}
                    applied_age = age
                # No new evidence means the core freezes its belief. Its
                # candidate diagnostic gates are NOT used for attribution.
                current_score = float(score[0])
                correction = float(state["correction"][0])
                contribution_sum = float(sum(x["total_contribution"] for x in applied.values()))
                residual = current_score - initial
                error = max(abs(contribution_sum - correction), abs(contribution_sum - residual))
                if error > 2e-5 * (1 + abs(correction)):
                    raise RuntimeError("token contributions do not reconstruct the actual score correction")
                changes = _old_changes(previous, applied)
                action, risk = posterior_decision(score, fp_cost, fn_cost)
                deadline = age == maximum
                future_loss, q_wait = None, None
                if policy is not None and not deadline:
                    features = model.value_features(state, age=age)
                    predicted = policy(features)
                    if predicted.shape != (1,) or not torch.isfinite(predicted).all() or (predicted < 0).any():
                        raise ValueError("policy must predict one finite nonnegative continuation loss")
                    future_loss = float(predicted[0])
                    q_wait = float(wait_cost) + future_loss
                commit = deadline or (q_wait is not None and float(risk[0]) <= q_wait)
                active_policy_prefix = first_policy_commit is None
                if commit and first_policy_commit is None:
                    first_policy_commit = age
                chosen = ("commit_target" if bool(action[0]) else "commit_background") if commit else "wait"
                token_rows = []
                for key in sorted(applied):
                    item = copy.deepcopy(applied[key])
                    item["seen_in_previous_prefix"] = key in previous
                    if key in previous:
                        item["gate_change"] = (np.asarray(item["gate"]) - np.asarray(previous[key]["gate"])).tolist()
                        item["value_change"] = (np.asarray(item["value"]) - np.asarray(previous[key]["value"])).tolist()
                        item["contribution_change"] = item["total_contribution"] - previous[key]["total_contribution"]
                    token_rows.append(item)
                steps.append({"age": age, "window": window, "extra_wait_ms": age * r.window_ms,
                              "prior_logit": initial, "logit": current_score,
                              "posterior_probability_estimate": _probability(current_score),
                              "report_probability": _probability(current_score + r.log_w),
                              "correction": correction, "contribution_sum": contribution_sum,
                              "reconstruction_error": error, "score_updated": updated,
                              "attribution_carried_from_age": applied_age,
                              "visible_neighbors": int(diag["count"][0]),
                              "new_neighbors": len(set(applied) - set(previous)),
                              "old_evidence_change": changes, "tokens": token_rows,
                              "window_potentials": copy.deepcopy(applied_windows),
                              "R_stop": float(risk[0]), "predicted_future_loss": future_loss,
                              "wait_cost": float(wait_cost), "Q_wait": q_wait,
                              "policy_action": chosen, "deadline": deadline,
                              "within_actual_policy_prefix": active_policy_prefix})
    finally:
        model.train(was_training)
        if isinstance(policy, torch.nn.Module):
            policy.train(policy_training)
    return {"recording": r.name, "event_id": event_id,
            "event": {"x": int(r.x[event_id]), "y": int(r.y[event_id]), "time_ms": int(r.t[event_id]),
                      "polarity": float(r.p[event_id]), "arrival_window": int(r.window[event_id])},
            "prior_logit": initial, "decision_logit": tau, "report_threshold": threshold,
            "log_w": r.log_w, "normalization": model.config.normalization,
            "maximum_age": maximum, "first_policy_commit_age": first_policy_commit,
            "source_encoded": cache.encoded_total, "steps": steps,
            "interpretation": "Algebraic contributions in a fitted model; not causal effects, calibrated "
                "association probabilities, or verified token log-likelihood ratios.",
            "update_rule": "logit = prior + replacement full-prefix correction; never accumulate old corrections",
            "after_commit": "Later prefixes are counterfactual diagnostic continuations, not online compute",
            "birth": "The requested query is explicitly admitted for this diagnostic; birth gate is not applied",
            "max_reconstruction_error": max(x["reconstruction_error"] for x in steps)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="one raw recording npz")
    ap.add_argument("--prior", required=True, help="matching BIR arrival trajectory npz")
    ap.add_argument("--event-id", type=int, required=True, help="original recording row index")
    ap.add_argument("--policy", help="optional compatible VoW checkpoint")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    if os.path.exists(args.out):
        raise FileExistsError("choose a new explanation output path: " + args.out)
    model, checkpoint = load_model(args.checkpoint, args.device)
    cfg = checkpoint["config"]
    rec, _ = get_record((args.data, args.prior), cfg)
    continuation = None
    pc = cfg["policy"]
    if args.policy:
        continuation, _, metadata = load_policy(args.policy, args.checkpoint, model, args.device)
        if not np.isclose(rec.log_w, metadata["log_w"], atol=1e-6, rtol=0):
            raise ValueError("prior score scale differs from policy training")
        pc = metadata["config"]["policy"]
    report = explain_query(model, rec, cfg["evidence"], args.event_id, cfg["threshold"],
                           continuation, pc["wait_cost"], args.device)
    report.update(checkpoint_sha256=file_digest(args.checkpoint),
                  prior_path=os.path.abspath(args.prior), policy_path=args.policy,
                  analysis="single-query diagnostic, not a benchmark or causal attribution")
    save_json(args.out, report)
    print("BST EXPLANATION %s event %d: %d prefixes, max reconstruction error %.3g -> %s" %
          (rec.name, args.event_id, len(report["steps"]), report["max_reconstruction_error"], args.out), flush=True)


if __name__ == "__main__":
    main()
