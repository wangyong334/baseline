"""Fit VoW on disjoint validation recordings using frozen-policy iteration."""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from model.bst import BirthValueHead, ContinuationValueHead, stop_risk
from tools.bst.pipeline import (choose_queries, file_digest, get_record, load_model, paths, prior_provenance, rollout, save_json,
                               threshold_costs, validation_partition)
from tools.bst.policy import BirthContext, policy_returns
from tools.bst.train import require_complete_priors, validate_training_config
from tools.bst.pipeline import checkpoint_scope


def regress(head, x, y, weights, cfg, device, rng):
    if not len(y):
        raise ValueError("no nonterminal policy fitting examples")
    x, y, weights = np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(weights, np.float32)
    if (x.ndim != 2 or y.ndim != 1 or weights.shape != y.shape or len(x) != len(y)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(weights).all()
            or (weights <= 0).any() or (y < 0).any()):
        raise ValueError("policy regression requires finite features, nonnegative costs and positive weights")
    mean_weight = max(float(weights.mean()), 1e-12)
    optimizer = torch.optim.Adam(head.parameters(), lr=cfg["lr"])
    # A useful output scale at initialization, avoiding softplus(0) dominating
    # very small event error rates during the entire short fitting stage.
    if not getattr(head, "_fitted", False):
        mean = float(np.sum(y * weights, dtype=np.float64) / max(float(weights.sum()), 1e-12))
        with torch.no_grad():
            mean = max(mean, 1e-6)
            # Stable inverse softplus, including large finite declared costs.
            head.net[-1].bias.fill_(mean + np.log(-np.expm1(-mean)))
        head._fitted = True
    for _ in range(cfg["fit_epochs"]):
        permutation = rng.permutation(len(y))
        for start in range(0, len(y), cfg["batch_size"]):
            ids = permutation[start:start + cfg["batch_size"]]
            xb = torch.as_tensor(x[ids], device=device)
            yb = torch.as_tensor(y[ids], device=device)
            wb = torch.as_tensor(weights[ids], device=device)
            optimizer.zero_grad()
            loss = (wb * (head(xb) - yb).square()).sum() / (cfg["batch_size"] * mean_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite VoW regression loss")
            loss.backward()
            finite_gradients = [torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None]
            if finite_gradients and not torch.stack(finite_gradients).all():
                raise FloatingPointError("nonfinite VoW gradient; refusing optimizer update")
            optimizer.step()
    error_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(y), cfg["batch_size"]):
            stop = start + cfg["batch_size"]
            xb = torch.as_tensor(x[start:stop], device=device)
            pred = head(xb).cpu().numpy()
            if not np.isfinite(pred).all():
                raise FloatingPointError("nonfinite fitted VoW prediction")
            error_sum += float(np.sum(weights[start:stop] * (pred - y[start:stop]) ** 2, dtype=np.float64))
    mse = error_sum / max(float(weights.sum()), 1e-12)
    return mse


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--z0-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--background-budget", type=int)
    ap.add_argument("--fit-epochs", type=int)
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--require-complete-priors", action="store_true", help="require final SDU and complete matching BIR priors")
    args = ap.parse_args(argv)
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    model, ck = load_model(args.checkpoint, args.device)
    model.eval()
    cfg = copy.deepcopy(ck["config"])
    pc = cfg["policy"]
    for arg in ("background_budget", "fit_epochs", "iterations"):
        if getattr(args, arg) is not None:
            pc[arg] = getattr(args, arg)
    validate_training_config(cfg)
    if os.path.exists(args.out):
        raise FileExistsError("preserving existing policy checkpoint: " + args.out)
    seed = cfg["train"]["seed"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    fit_names = ck["manifest"]["policy_fit"]
    if not fit_names or set(fit_names) & set(ck["manifest"]["selection"]):
        raise ValueError("VoW fitting and SDU selection recordings must be nonempty and disjoint")
    fit_pairs = paths(args.data_root, args.z0_root, "val", names=fit_names)
    provenance = prior_provenance(args.z0_root, ("val",))
    expected_hash = ck.get("manifest", {}).get("prior_provenance", {}).get("backbone_sha256")
    if expected_hash and provenance.get("backbone_sha256") and expected_hash != provenance["backbone_sha256"]:
        raise ValueError("policy priors came from a different BIR checkpoint")
    if args.require_complete_priors:
        require_complete_priors(provenance, checkpoint_scope(ck, args.data_root))
        if not expected_hash or expected_hash != provenance["backbone_sha256"]:
            raise ValueError("final policy requires verified matching SDU/BIR identity")
    context_config = pc.get("birth_context", {"cell": 4, "history": cfg["evidence"]["past_windows"]})
    chunks, log_weights = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for pair in fit_pairs:
            rec, index = get_record(pair, cfg)
            if ck.get("log_w") is not None and not np.isclose(rec.log_w, ck["log_w"], atol=1e-6, rtol=0):
                raise ValueError("policy prior score scale differs from SDU training")
            qs = choose_queries(rec, pc["background_budget"], rng)
            context = BirthContext(rec, retain_all=True, **context_config)
            log_weights.append(rec.log_w)
            for start in range(0, len(qs.ids), cfg["train"]["batch_size"]):
                ids = qs.ids[start:start + cfg["train"]["batch_size"]]
                result = rollout(model, rec, index, ids, args.device, collect_features=True)
                chunks.append({"features": result["features"].cpu().numpy(),
                               "logits": result["logits"].cpu().numpy(),
                               "valid": result["valid"].cpu().numpy(),
                               "prior": rec.z0[ids], "labels": rec.label[ids],
                               "weight": qs.weights[start:start + len(ids)].astype(np.float32),
                               "birth": context.features(ids)})
            print("policy traces %s: %d queries" % (rec.name, len(qs.ids)), flush=True)
    if not np.allclose(log_weights, log_weights[0], atol=1e-6, rtol=0):
        raise ValueError("one policy cannot silently mix backbone score scales")
    data = {key: np.concatenate([ch[key] for ch in chunks]) for key in chunks[0]}
    tau, fp, fn = threshold_costs(cfg["threshold"], log_weights[0])
    head = ContinuationValueHead(model.value_feature_dim, pc["hidden"]).to(args.device)
    birth_head = BirthValueHead(BirthContext.feature_dim, pc["hidden"]).to(args.device)
    p = 1 / (1 + np.exp(-np.clip(data["logits"], -60, 60)))
    risk = np.where(data["logits"] >= tau, fp * (1-p), fn * p)
    iterations = []
    for iteration in range(pc["iterations"]):
        # Freeze all current-policy predictions before constructing targets.
        if iteration == 0:
            waits = data["valid"].copy()  # initial reference policy waits to deadline
        else:
            with torch.no_grad():
                flat = data["features"].reshape(-1, model.value_feature_dim)
                pred = np.concatenate([head(torch.as_tensor(flat[a:a + pc["batch_size"]], device=args.device)).cpu().numpy()
                                       for a in range(0, len(flat), pc["batch_size"])])
            waits = risk > pc["wait_cost"] + pred.reshape(risk.shape)
        target, returns, eligible = policy_returns(data["logits"], data["labels"], data["valid"], waits,
                                                    tau, fp, fn, pc["wait_cost"])
        # Each event has total fitting weight w_i, regardless of remaining horizon.
        weights = data["weight"][:, None] / np.maximum(eligible.sum(1, keepdims=True), 1)
        weights = np.broadcast_to(weights, eligible.shape)
        mse = regress(head, data["features"][eligible], target[eligible], weights[eligible], pc, args.device, rng)
        iterations.append({"iteration": iteration + 1, "weighted_mse": mse,
                           "frozen_policy_wait_fraction": float(waits[eligible].mean())})
        print(iterations[-1], flush=True)
    # Birth learns realized risk of the FINAL frozen post-SDU policy, including
    # its subsequent waiting costs. Initial evidence lookup has no temporal wait.
    with torch.no_grad():
        flat = data["features"].reshape(-1, model.value_feature_dim)
        pred = np.concatenate([head(torch.as_tensor(flat[a:a + pc["batch_size"]], device=args.device)).cpu().numpy()
                               for a in range(0, len(flat), pc["batch_size"])])
    waits = risk > pc["wait_cost"] + pred.reshape(risk.shape)
    _, returns, _ = policy_returns(data["logits"], data["labels"], data["valid"], waits,
                                  tau, fp, fn, pc["wait_cost"])
    birth_mse = regress(birth_head, data["birth"], returns[:, 0], data["weight"], pc, args.device, rng)
    end = data["valid"].sum(1) - 1
    first_correct = (data["prior"] >= tau) == data["labels"]
    last_correct = (data["logits"][np.arange(len(end)), end] >= tau) == data["labels"]
    meta = {"config": cfg, "model_checkpoint": os.path.abspath(args.checkpoint),
            "model_sha256": file_digest(args.checkpoint),
            "prior_provenance": provenance,
            "fit_recordings": fit_names, "log_w": float(log_weights[0]), "iterations": iterations,
            "birth_mse": birth_mse, "birth_context": context_config,
            "sample_counts": {"queries": len(end), "rescue": int((~first_correct & last_correct).sum()),
                              "harm": int((first_correct & ~last_correct).sum())},
            "seconds": time.perf_counter() - t0, "risk_guarantee": False,
            "target": "frozen_policy_realized_return", "test_used": False}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"continuation": head.state_dict(), "birth": birth_head.state_dict(),
                "feature_dim": model.value_feature_dim, "birth_feature_dim": BirthContext.feature_dim,
                "meta": meta}, args.out)
    save_json(os.path.splitext(args.out)[0] + ".json", meta)
    print("BST POLICY FINISHED: " + args.out, flush=True)


if __name__ == "__main__":
    main()
