"""Train the shared SDU on frozen BIR scores; select only on held-out recordings."""
import argparse
import math
import os
import sys
import time
import uuid

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from model.bst import BSTConfig, SharedSequentialModel
from tools.bst.pipeline import (checkpoint_scope, choose_queries, evaluate_sampled, get_record, load_config, prior_provenance,
                                paths, rollout, save_json, validation_partition, weighted_event_loss)
from tools.bst.training_geometry import GeometryBatches


def validate_training_config(cfg):
    """Reject silent empty fits and invalid cost/optimizer settings before work."""
    for group, positive, nonnegative in (
            ("train", ("epochs", "batch_size"), ("background_budget", "validation_background_budget")),
            ("policy", ("iterations", "fit_epochs", "batch_size", "hidden"), ("background_budget",))):
        for name in positive + nonnegative:
            value = cfg[group][name]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
                raise ValueError("%s.%s must be an integer" % (group, name))
            if value < (1 if name in positive else 0):
                raise ValueError("invalid %s.%s" % (group, name))
    seed = cfg["train"]["seed"]
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("train.seed must be an integer in [0, 2**32)")
    for group, name, positive in (("train", "lr", True), ("train", "grad_clip", True),
                                  ("train", "weight_decay", False), ("policy", "lr", True),
                                  ("policy", "wait_cost", False), ("policy", "init_cost", False)):
        value = float(cfg[group][name])
        if not math.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError("invalid %s.%s" % (group, name))
    if not 0 < float(cfg["policy"]["band_alpha"]) < .5:
        raise ValueError("policy.band_alpha must lie in (0, 0.5)")
    budget = float(cfg["train"].get("geometry_cache_mb", 64))
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("train.geometry_cache_mb must be finite and nonnegative")


def save_checkpoint(checkpoint, path):
    """Readers see a complete old or new checkpoint, never a partial write."""
    temporary = str(path) + "." + uuid.uuid4().hex[:8] + ".tmp"
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require_complete_priors(provenance, scope):
    if not provenance.get("verified") or not provenance.get("complete") or not provenance.get("backbone_sha256"):
        raise ValueError("final training requires complete, verified BIR prior manifests")
    if scope["probe"]:
        raise ValueError("final training requires all raw train/validation recordings")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/bst_shared.json")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--z0-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int, help="override SDU/VoW seed; backbone identity is checked separately")
    ap.add_argument("--require-complete-priors", action="store_true", help="reject incomplete/legacy caches for final runs")
    ap.add_argument("--background-budget", type=int)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--geometry-cache-mb", type=float, help="bounded CPU neighbor-ID cache; 0 disables")
    args = ap.parse_args(argv)
    if args.threads < 1 or args.limit_train < 0 or args.limit_val < 0:
        raise ValueError("threads must be positive and recording limits nonnegative")
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.background_budget is not None:
        cfg["train"]["background_budget"] = args.background_budget
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.geometry_cache_mb is not None:
        cfg["train"]["geometry_cache_mb"] = args.geometry_cache_mb
    validate_training_config(cfg)
    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    train = paths(args.data_root, args.z0_root, "train", limit=args.limit_train)
    fit, select = validation_partition(paths(args.data_root, args.z0_root, "val", limit=args.limit_val))
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, "model.pt")
    if os.path.exists(target):
        raise FileExistsError("choose a new output directory; preserving %s" % target)
    manifest = {"train": [os.path.basename(p[0]) for p in train],
                "policy_fit": [os.path.basename(p[0]) for p in fit],
                "selection": [os.path.basename(p[0]) for p in select],
                "data_root": os.path.abspath(args.data_root), "z0_root": os.path.abspath(args.z0_root),
                "config": cfg, "stage": "frozen_BIR_shared_SDU", "test_used": False}
    manifest["prior_provenance"] = prior_provenance(args.z0_root)
    manifest["scope"] = checkpoint_scope(manifest, args.data_root)
    if args.require_complete_priors:
        require_complete_priors(manifest["prior_provenance"], manifest["scope"])
    save_json(os.path.join(args.out, "manifest.json"), manifest)
    model = SharedSequentialModel(BSTConfig(**cfg["model"])).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                 weight_decay=cfg["train"]["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["train"]["epochs"])
    selection_metric = cfg["train"].get("selection_metric", "weighted_prefix_bce")
    if selection_metric not in ("weighted_prefix_bce", "weighted_band_iou", "weighted_deadline_iou"):
        raise ValueError("unknown selection metric")
    minimize = selection_metric == "weighted_prefix_bce"
    best = float("inf") if minimize else -float("inf")
    expected_log_w = None
    history = []
    for epoch in range(cfg["train"]["epochs"]):
        start_time = time.perf_counter()
        model.train()
        train_loss, population, query_count, encoded_count = 0.0, 0, 0, 0
        order = rng.permutation(len(train))
        recording_stats = []
        for ordinal, ri in enumerate(order, 1):
            record_started = time.perf_counter()
            rec, index = get_record(train[ri], cfg)
            load_seconds = time.perf_counter() - record_started
            if expected_log_w is None:
                expected_log_w = rec.log_w
            elif not np.isclose(rec.log_w, expected_log_w, atol=1e-6, rtol=0):
                raise ValueError("mixed backbone score scales in training")
            qs = choose_queries(rec, cfg["train"]["background_budget"], rng)
            permutation = rng.permutation(len(qs.ids))
            population += len(rec)
            geometry = GeometryBatches(index, qs.ids[permutation], cfg["train"]["batch_size"],
                                       cfg["train"].get("geometry_cache_mb", 64))
            next_progress = time.perf_counter() + 30.0
            update_seconds = 0.0
            for start, stop, prepared in geometry:
                update_started = time.perf_counter()
                rows = permutation[start:stop]
                ids = qs.ids[rows]
                if not len(ids):
                    continue
                optimizer.zero_grad()
                result = rollout(model, rec, index, ids, args.device, prepared_batch=prepared)
                labels = torch.as_tensor(rec.label[ids], dtype=torch.float32, device=args.device)
                weights = torch.as_tensor(qs.weights[rows], dtype=torch.float32, device=args.device)
                loss_sum = weighted_event_loss(result["logits"], result["valid"], labels, weights)
                # Fixed denominator across recordings and partial minibatches:
                # summing all sampled minibatch objectives is proportional to
                # the global Horvitz-Thompson event loss, not an equal-record loss.
                loss = loss_sum / float(cfg["train"]["batch_size"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss in %s" % rec.name)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("nonfinite gradient in %s; refusing optimizer update" % rec.name)
                optimizer.step()
                train_loss += float(loss_sum.detach())
                query_count += len(ids)
                encoded_count += result["source_count"]
                update_seconds += time.perf_counter() - update_started
                if time.perf_counter() >= next_progress:
                    progress = {"epoch": epoch + 1, "stage": "train", "recording": rec.name,
                        "recordings_completed": ordinal - 1, "recordings_total": len(train),
                        "queries_completed_in_recording": stop, "queries_total_in_recording": len(qs.ids),
                        "elapsed_epoch_seconds": time.perf_counter() - start_time}
                    save_json(os.path.join(args.out, "progress.json"), progress)
                    print("TRAIN PROGRESS " + str(progress), flush=True)
                    next_progress = time.perf_counter() + 30.0
            print("epoch %d record %s queries %d / %d" % (epoch + 1, rec.name, len(qs.ids), len(rec)), flush=True)
            stats = {"recording": rec.name, "ordinal": ordinal, "queries": len(qs.ids),
                     "load_seconds": load_seconds, "update_seconds": update_seconds,
                     "recording_seconds": time.perf_counter() - record_started, "geometry": dict(geometry.stats)}
            recording_stats.append(stats)
            progress = {"epoch": epoch + 1, "stage": "train", "recording": rec.name,
                        "recordings_completed": ordinal, "recordings_total": len(train),
                        "elapsed_epoch_seconds": time.perf_counter() - start_time, "last_recording": stats}
            save_json(os.path.join(args.out, "progress.json"), progress)
            print("RECORD TIMING " + str(stats), flush=True)
        schedule.step()
        save_json(os.path.join(args.out, "progress.json"), {"epoch": epoch + 1, "stage": "validation",
                  "recordings_completed": len(train), "recordings_total": len(train)})
        validation_started = time.perf_counter()
        validation = evaluate_sampled(model, select, cfg, args.device,
                                      cfg["train"]["validation_background_budget"], seed)
        candidate = validation[selection_metric]
        if not math.isfinite(candidate):
            raise FloatingPointError("nonfinite validation selection metric; refusing checkpoint save")
        row = {"epoch": epoch + 1, "train_prefix_bce": train_loss / max(population, 1),
               "queries": query_count, "encoded_source_rows": encoded_count,
               "validation": validation, "seconds": time.perf_counter() - start_time,
               "validation_seconds": time.perf_counter() - validation_started, "recordings": recording_stats}
        history.append(row)
        checkpoint = {"model": model.state_dict(), "config": cfg, "epoch": epoch + 1,
                      "manifest": manifest, "validation": validation, "format": "bst_shared_v1",
                      "log_w": expected_log_w, "selection_metric": selection_metric}
        save_checkpoint(checkpoint, os.path.join(args.out, "last.pt"))
        improved = (candidate < best) if minimize else (candidate > best)
        if improved:
            best = candidate
            save_checkpoint(checkpoint, target)
        save_json(os.path.join(args.out, "training.json"), history)
        print({key: value for key, value in row.items() if key != "recordings"}, flush=True)
        save_json(os.path.join(args.out, "progress.json"), {"epoch": epoch + 1, "stage": "epoch_complete",
                  "checkpoint": "last.pt", "validation": validation, "seconds": row["seconds"]})
    print("BST TRAIN FINISHED: " + target, flush=True)


if __name__ == "__main__":
    main()
