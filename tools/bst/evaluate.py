"""Evaluate real stopped-query execution with the existing four official metrics."""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from model.bst import BirthValueHead, ContinuationValueHead
from tools.bst.pipeline import (checkpoint_scope, file_digest, get_record, load_model, paths, prior_provenance, save_json, torch_load)
from tools.bst.policy import make_birth_callback
from tools.bst.stream import run_recording
from tools.final_analysis import THRESHOLD, PolicyStats, references, report_prob
from utils.recording_groups import dense_names


def validate_official_threshold(config):
    if not np.isclose(config["threshold"], THRESHOLD, atol=1e-12, rtol=0):
        raise ValueError("official metrics use a fixed report threshold of 0.9; "
                         "changing only the model decision threshold would misreport results")


def load_policy(path, checkpoint, model, device):
    policy = torch_load(path, device)
    if policy["meta"].get("model_sha256") != file_digest(checkpoint):
        raise ValueError("VoW checkpoint belongs to a different SDU; fit it again")
    pc = policy["meta"]["config"]["policy"]
    head = ContinuationValueHead(model.value_feature_dim, pc["hidden"]).to(device)
    head.load_state_dict(policy["continuation"])
    birth = BirthValueHead(policy["birth_feature_dim"], pc["hidden"]).to(device)
    birth.load_state_dict(policy["birth"])
    return head.eval(), birth.eval(), policy["meta"]


def metric_record(rec):
    return {"name": rec.name, "x": rec.x, "y": rec.y, "t": rec.t,
            "tid": rec.target_id if rec.target_id is not None else rec.label,
            "pos": rec.label > .5, "traj": rec.z0[:, None], "log_w": rec.log_w}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--policy")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--z0-root", required=True)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--subset", choices=("selection", "fit", "all"), default="selection")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--birth", choices=("band", "all", "learned"))
    ap.add_argument("--commit", choices=("fixed", "learned", "arrival"), default="learned")
    ap.add_argument("--limit", type=int, default=0, help="recording limit for a PROBE, not a full benchmark")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1024, help="inference query batch; independent of training batch")
    args = ap.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("inference batch_size must be positive")
    torch.set_num_threads(args.threads)
    model, ck = load_model(args.checkpoint, args.device)
    model.eval()
    scope = checkpoint_scope(ck, args.data_root)
    cfg = ck["config"]
    validate_official_threshold(cfg)
    pc = cfg["policy"]
    birth_mode = args.birth or pc["birth"]
    continuation, birth_head, meta = None, None, None
    if args.policy:
        continuation, birth_head, meta = load_policy(args.policy, args.checkpoint, model, args.device)
        pc = meta["config"]["policy"]
    if (args.commit == "learned" or birth_mode == "learned") and meta is None:
        raise ValueError("learned policies require --policy; use --commit fixed for the baseline")
    names = None
    if args.split == "val" and args.subset != "all":
        names = ck["manifest"]["selection" if args.subset == "selection" else "policy_fit"]
    pairs = paths(args.data_root, args.z0_root, args.split, names=names, limit=args.limit)
    provenance = prior_provenance(args.z0_root, (args.split,))
    expected_hash = ck.get("manifest", {}).get("prior_provenance", {}).get("backbone_sha256")
    actual_hash = provenance.get("backbone_sha256")
    if expected_hash and actual_hash and expected_hash != actual_hash:
        raise ValueError("evaluation priors came from a different BIR checkpoint")
    if os.path.exists(args.out) and os.listdir(args.out):
        raise FileExistsError("choose an empty evaluation directory: " + args.out)
    os.makedirs(args.out, exist_ok=True)
    grouped = set(dense_names(args.data_root, (args.split,), 260, 346)[args.split])
    baseline = PolicyStats("BIR", cfg["model"]["window_ms"])
    result = PolicyStats("BST", cfg["model"]["window_ms"])
    runtime = []
    for pair in pairs:
        rec, _ = get_record(pair, cfg)
        if ck.get("log_w") is not None and not np.isclose(rec.log_w, ck["log_w"], atol=1e-6, rtol=0):
            raise ValueError("backbone score scale differs from SDU training")
        if meta is not None and not np.isclose(rec.log_w, meta["log_w"], atol=1e-6, rtol=0):
            raise ValueError("backbone score scale differs from policy fit")
        birth_callback = None
        if birth_mode == "learned":
            birth_callback = make_birth_callback(birth_head, model, rec, cfg["threshold"], rec.log_w,
                                                 pc["init_cost"], meta["birth_context"], args.device)
        policy = continuation if args.commit == "learned" else None
        if args.commit == "arrival":
            policy = lambda features, risk, age: torch.ones_like(risk, dtype=torch.bool)
        out = run_recording(model, rec, cfg["evidence"], policy=policy, birth=birth_mode,
                            threshold=cfg["threshold"], band_alpha=pc["band_alpha"],
                            band_kind=pc.get("band_kind", "sym"),
                            batch_size=args.batch_size, device=args.device,
                            wait_cost=pc["wait_cost"], birth_policy=birth_callback)
        mr = metric_record(rec)
        ref = references(mr, 0, rec.window_ms)
        prob = report_prob(out["logits"], rec.log_w)
        baseline.add(mr, report_prob(rec.z0, rec.log_w), np.zeros(len(rec), np.int64), ref,
                     dense=rec.name in grouped)
        result.add(mr, prob, out["lags"], ref, dense=rec.name in grouped)
        np.savez(os.path.join(args.out, rec.name), logits=out["logits"], z0=rec.z0,
                 posterior_probability=1 / (1 + np.exp(-np.clip(out["logits"], -60, 60))),
                 report_probability=prob, lag=out["lags"], labels=rec.label,
                 reasons=out["reasons"], log_w=rec.log_w, window_ms=rec.window_ms)
        runtime.append({"recording": rec.name, **out["stats"]})
        print("evaluated %s %s" % (rec.name, out["stats"]), flush=True)
    report = {"baseline": baseline.result(), "model": result.result(), "runtime": runtime,
              "split": args.split, "subset": args.subset if args.split == "val" else "all",
              "birth": birth_mode, "commit": args.commit, "config": cfg,
              "band_kind": pc.get("band_kind", "sym"), "prior_provenance": provenance,
              "probe": bool(args.limit) or scope["probe"] or not provenance.get("complete", False)
                  or not ck.get("manifest", {}).get("prior_provenance", {}).get("complete", False),
              "checkpoint_scope": scope,
              "evaluation_limit": args.limit, "checkpoint_sha256": file_digest(args.checkpoint),
              "inference_batch_size": args.batch_size,
              "timing_scope": "SDU/VoW/index/cache on precomputed BIR priors; excludes BIR and file load",
              "guaranteed_risk_bound": False}
    save_json(os.path.join(args.out, "summary.json"), report)
    print("BST EVALUATE FINISHED %s" % report["model"], flush=True)


if __name__ == "__main__":
    main()
