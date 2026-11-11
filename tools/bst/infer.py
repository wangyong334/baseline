"""End-to-end chronological BIR -> shared SDU -> VoW inference.

Unlike evaluate.py this entry point does not require precomputed z0 files.
Disk input is loaded before timing, but event representation, backbone forward,
shared evidence encoding and real stopped-query decisions execute per window.
The existing dense BIR convolution kernels are retained; reduced communication
must not be reported as skipped GPU convolution work.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import train_stream_v1 as T
from dataset.ev_uav_stream import EvUAVStream
from dataset.stream_source import make_window_source
from tools.bst.data import Recording
from tools.bst.evaluate import load_policy, metric_record, validate_official_threshold
from tools.bst.pipeline import checkpoint_scope, file_digest, load_model, save_json, torch_load
from tools.bst.policy import make_birth_callback
from tools.bst.stream import run_recording
from tools.final_analysis import PolicyStats, references, report_prob
from utils.recording_groups import dense_names


class BackboneArrivalProvider:
    """Execute each BIR window once and return scores in requested event order."""

    def __init__(self, backbone, source, recording, original_ids, original_count, log_w):
        self.backbone, self.source = backbone, source
        self.rec, self.log_w = recording, float(log_w)
        self.original_to_local = np.full(int(original_count), -1, dtype=np.int64)
        original_ids = np.asarray(original_ids, np.int64)
        if len(original_ids) != len(recording) or len(np.unique(original_ids)) != len(original_ids):
            raise ValueError("original_ids must map each recording event exactly once")
        if ((original_ids < 0) | (original_ids >= int(original_count))).any():
            raise ValueError("original id outside raw recording")
        self.original_to_local[original_ids] = np.arange(len(recording))
        self.arrival_scores = np.full(len(recording), np.nan, dtype=np.float32)
        self.states = None
        self.next_window = 0
        self.windows = 0
        self.events = 0
        self.seconds = 0.0

    def __call__(self, window, ids):
        if int(window) != self.next_window:
            raise ValueError("BIR windows must be executed consecutively, including empty windows")
        started = time.perf_counter()
        x, events, _, original = self.source.window(int(window))
        original = np.asarray(original, np.int64)
        local = self.original_to_local[original]
        ids = np.asarray(ids, np.int64)
        if (local < 0).any() or not np.array_equal(np.sort(local), np.sort(ids)):
            raise ValueError("backbone window rows do not match original recording indices")
        with torch.no_grad():
            logits, self.states, _ = self.backbone(x, events, self.states, state_mode="carry")
            values = logits.detach().float().cpu().numpy() - self.log_w
        if values.shape != local.shape or not np.isfinite(values).all():
            raise ValueError("backbone returned invalid arrival scores")
        self.arrival_scores[local] = values
        prior = self.arrival_scores[ids].copy()
        self.seconds += time.perf_counter() - started
        self.next_window += 1
        self.windows += 1
        self.events += len(ids)
        return {"prior": prior,
                "source_features": np.stack([prior, self.rec.p[ids]], axis=1).astype(np.float32)}


def recording_from_sequence(seq, window_ms, log_w, max_windows=0):
    count = seq.n_windows if not max_windows else min(seq.n_windows, int(max_windows))
    if count < 1:
        raise ValueError("at least one window is required")
    original_ids = np.flatnonzero(seq.t < count * int(window_ms))
    r = Recording(seq.name, seq.x[original_ids], seq.y[original_ids], seq.t[original_ids],
                  seq.p[original_ids], np.zeros(len(original_ids), np.float32),
                  seq.label[original_ids], int(window_ms), float(log_w), count,
                  seq.target_id[original_ids])
    return r, original_ids


def verify_backbone_provenance(checkpoint, actual_hash):
    provenance = checkpoint.get("manifest", {}).get("prior_provenance") or {}
    expected = provenance.get("backbone_sha256")
    if expected is not None and expected != actual_hash:
        raise ValueError("SDU was trained with priors from a different BIR checkpoint")
    verified = expected is not None and bool(provenance.get("verified", True))
    if not verified:
        warnings.warn("Legacy SDU/prior cache has no verified BIR identity; matching log_w alone "
                      "does not verify backbone provenance.", RuntimeWarning, stacklevel=2)
    return verified


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", required=True, help="frozen BIR checkpoint")
    ap.add_argument("--checkpoint", required=True, help="shared SDU checkpoint")
    ap.add_argument("--policy")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), required=True)
    ap.add_argument("--subset", choices=("selection", "fit", "all"), default="selection")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--birth", choices=("band", "all", "learned"))
    ap.add_argument("--commit", choices=("fixed", "learned", "arrival"), default="fixed")
    ap.add_argument("--limit", type=int, default=0, help="recording limit: probe only")
    ap.add_argument("--max-windows", type=int, default=0, help="truncate each recording: probe only")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1024, help="inference query batch; independent of training batch")
    args = ap.parse_args(argv)
    if args.limit < 0 or args.max_windows < 0 or args.threads < 1 or args.batch_size < 1:
        raise ValueError("limits must be nonnegative and threads positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, ck = load_model(args.checkpoint, device)
    model.eval()
    scope = checkpoint_scope(ck, args.data_root)
    cfg = ck["config"]
    validate_official_threshold(cfg)
    if model.config.source_features != 2:
        raise ValueError("live BIR adapter currently supplies [z0, polarity] only")
    backbone_hash = file_digest(args.backbone)
    provenance_verified = verify_backbone_provenance(ck, backbone_hash)
    bb = torch_load(args.backbone, device)
    bcfg = dict(bb["config"])
    bcfg["root"] = args.data_root
    bcfg["execution"], bcfg["input_device"] = "step", "cpu"
    if int(bcfg["window_ms"]) != int(cfg["model"]["window_ms"]):
        raise ValueError("BIR and SDU windows must have the same duration")
    declared_horizon = cfg.get("recording_n_windows")
    backbone_horizon = int(bcfg.get("n_windows", 0) or 0)
    if declared_horizon is not None and backbone_horizon > 0 and int(declared_horizon) != backbone_horizon:
        raise ValueError("BIR recording duration disagrees with SDU training duration")
    if declared_horizon is None:
        warnings.warn("Legacy SDU config lacks recording_n_windows; training may have inferred "
                      "a different final deadline from the last event.", RuntimeWarning)
    log_w = float(np.log(float(bb["stats"]["pos_weight"])))
    if ck.get("log_w") is not None and not np.isclose(log_w, ck["log_w"], atol=1e-6, rtol=0):
        raise ValueError("BIR prior score scale differs from SDU training")
    q99 = np.asarray(bb["stats"]["q99"], dtype=np.float32)
    backbone = T.build_net(bcfg, device)
    backbone.load_state_dict(bb["model"])
    if not bool(backbone.gain_calibrated):
        raise ValueError("BIR checkpoint has no calibrated layer gains")
    backbone.eval()
    continuation, birth_head, meta = None, None, None
    pc = cfg["policy"]
    if args.policy:
        continuation, birth_head, meta = load_policy(args.policy, args.checkpoint, model, device)
        pc = meta["config"]["policy"]
        if not np.isclose(log_w, meta["log_w"], atol=1e-6, rtol=0):
            raise ValueError("BIR prior score scale differs from the policy fit")
    birth_mode = args.birth or pc["birth"]
    if (args.commit == "learned" or birth_mode == "learned") and meta is None:
        raise ValueError("learned policies require --policy")
    selected = None
    if args.split == "val" and args.subset != "all":
        selected = ck["manifest"]["selection" if args.subset == "selection" else "policy_fit"]
    dataset = EvUAVStream(args.data_root, args.split, bcfg, names=selected)
    names = dataset.names[:args.limit] if args.limit else dataset.names
    if os.path.exists(args.out) and os.listdir(args.out):
        raise FileExistsError("choose an empty inference directory: " + args.out)
    os.makedirs(args.out, exist_ok=True)
    grouped = set(dense_names(args.data_root, (args.split,), int(bcfg["height"]),
                              int(bcfg["width_px"]))[args.split])
    baseline, result = PolicyStats("BIR", bcfg["window_ms"]), PolicyStats("BST", bcfg["window_ms"])
    runtime = []
    for name in names:
        seq = EvUAVStream(args.data_root, args.split, bcfg, names=[name])[0]
        if declared_horizon is not None and seq.n_windows != int(declared_horizon):
            raise ValueError("raw recording horizon differs from declared SDU training horizon")
        rec, original_ids = recording_from_sequence(seq, bcfg["window_ms"], log_w, args.max_windows)
        source = make_window_source(seq, bcfg, q99, device, next(backbone.parameters()).dtype)
        provider = BackboneArrivalProvider(backbone, source, rec, original_ids, len(seq.t), log_w)
        birth_callback = None
        if birth_mode == "learned":
            # run_recording uses a private mutable score array. Bind context to
            # that live recording on first call, not the zero-prior placeholder.
            callback_holder = []
            def birth_callback(ids, live_recording, window):
                if not callback_holder:
                    callback_holder.append(make_birth_callback(
                        birth_head, model, live_recording, cfg["threshold"], log_w,
                        pc["init_cost"], meta["birth_context"], device))
                return callback_holder[0](ids, live_recording, window)
        policy = continuation if args.commit == "learned" else None
        if args.commit == "arrival":
            policy = lambda features, risk, age: torch.ones_like(risk, dtype=torch.bool)
        out = run_recording(model, rec, cfg["evidence"], policy=policy, birth=birth_mode,
                            threshold=cfg["threshold"], band_alpha=pc["band_alpha"],
                            batch_size=args.batch_size, device=device,
                            wait_cost=pc["wait_cost"], birth_policy=birth_callback,
                            arrival_provider=provider, band_kind=pc.get("band_kind", "sym"))
        if not np.isfinite(provider.arrival_scores).all():
            raise RuntimeError("some events were never processed by BIR")
        scored_rec = Recording(rec.name, rec.x, rec.y, rec.t, rec.p, provider.arrival_scores,
                               rec.label, rec.window_ms, rec.log_w, rec.n_windows, rec.target_id)
        mr = metric_record(scored_rec)
        ref = references(mr, 0, rec.window_ms)
        probability = report_prob(out["logits"], log_w)
        baseline.add(mr, report_prob(provider.arrival_scores, log_w), np.zeros(len(rec), np.int64),
                     ref, dense=rec.name in grouped)
        result.add(mr, probability, out["lags"], ref, dense=rec.name in grouped)
        np.savez(os.path.join(args.out, rec.name), logits=out["logits"], z0=provider.arrival_scores,
                 posterior_probability=1 / (1 + np.exp(-np.clip(out["logits"], -60, 60))),
                 report_probability=probability, lag=out["lags"], labels=rec.label,
                 reasons=out["reasons"], original_ids=original_ids, log_w=log_w,
                 window_ms=rec.window_ms)
        item = {"recording": rec.name, **out["stats"], "backbone_windows": provider.windows,
                "backbone_events": provider.events, "backbone_seconds": provider.seconds}
        runtime.append(item)
        print("online %s %s" % (name, item), flush=True)
    report = {"baseline": baseline.result(), "model": result.result(), "runtime": runtime,
              "split": args.split, "subset": args.subset if args.split == "val" else "all",
              "birth": birth_mode, "band_kind": pc.get("band_kind", "sym"),
              "commit": args.commit, "config": cfg,
              "probe": bool(args.limit or args.max_windows or scope["probe"] or not provenance_verified
                            or not ck.get("manifest", {}).get("prior_provenance", {}).get("complete", False)),
              "checkpoint_scope": scope, "max_windows": args.max_windows,
              "backbone_sha256": backbone_hash, "provenance_verified": provenance_verified,
              "provenance_note": "matched training prior identity" if provenance_verified else
                  "legacy prior metadata missing; BIR identity cannot be verified",
              "recording_horizon_verified": declared_horizon is not None,
              "checkpoint_sha256": file_digest(args.checkpoint),
              "timing_scope": "event representation + BIR + SDU/VoW/index/cache, excluding disk load/metrics/write",
              "backbone_execution": "dense PyTorch convolutions; communication reduction is not kernel skipping",
              "inference_batch_size": args.batch_size,
              "guaranteed_risk_bound": False}
    save_json(os.path.join(args.out, "summary.json"), report)
    print("BST ONLINE INFERENCE FINISHED %s" % report["model"], flush=True)


if __name__ == "__main__":
    main()
