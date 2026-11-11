"""Paired raw-recording probe for window grouping and pair-cache correctness.

The mean-reference checkpoint has DIFFERENT learned weights/architecture; its
comparison measures observed pipeline overhead, not equivalent-output speedup.
The window recompute oracle retains the ordinary cache and additionally rebuilds
all old pairs on read. It is a correctness oracle, not a lean no-cache baseline.
"""
import argparse
from contextlib import contextmanager
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bst import infer as I, stream
from tools.bst.pipeline import load_model, torch_load, file_digest, checkpoint_scope, save_json
from tools.bst.profile_inference import sync, window_timer, describe_windows


class RecomputePairs(stream.CausalPairCache):
    """Rebuild fixed source/relative projections on every query-cache read."""
    def read(self, rows, end):
        saved = super().read(rows, end)
        ids = saved["indices"].cpu().numpy().copy()
        ids[~saved["present"].cpu().numpy()] = -1
        local = torch.as_tensor(self.source.local_indices(ids), device=self.source.device)
        rebuilt = self.model.prepare_pairs(self.source.encoded, local, saved["relative"])
        rebuilt["indices"], rebuilt["relative"] = saved["indices"], saved["relative"]
        return rebuilt


@contextmanager
def recompute_pairs(enabled):
    previous = stream.CausalPairCache
    if enabled:
        stream.CausalPairCache = RecomputePairs
    try:
        yield
    finally:
        stream.CausalPairCache = previous


def compare_outputs(actual, reference, log_w, threshold=.9):
    delta = np.abs(actual["logits"]-reference["logits"])
    result = {"events": len(delta), "max_absolute_logit_difference": float(delta.max()) if len(delta) else 0.,
              "logits_allclose": bool(np.allclose(actual["logits"], reference["logits"], atol=2e-6, rtol=2e-6))}
    for key in ("lags", "reasons", "prior"):
        result[key+"_exact"] = bool(np.array_equal(actual[key], reference[key]))
    a = I.report_prob(actual["logits"], log_w) >= np.float32(threshold)
    b = I.report_prob(reference["logits"], log_w) >= np.float32(threshold)
    result["classification_disagreements"] = int((a != b).sum())
    result["equivalent"] = result["logits_allclose"] and all(result[k+"_exact"] for k in ("lags", "reasons", "prior")) and not result["classification_disagreements"]
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("backbone", "checkpoint", "data-root", "name", "out"):
        ap.add_argument("--"+name, required=True)
    ap.add_argument("--reference-checkpoint")
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args(argv)
    if min(args.repeats, args.threads, args.batch_size) < 1:
        ap.error("repeats, threads and batch-size must be positive")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("preserving existing profile: " + str(out))
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    window_model, checkpoint = load_model(args.checkpoint, device)
    if window_model.config.normalization != "window":
        raise ValueError("--checkpoint must use window aggregation")
    cfg = checkpoint["config"]
    I.validate_official_threshold(cfg)
    models = {"window_cached": window_model.eval(), "window_recomputed": window_model}
    checkpoints = {"window": checkpoint}
    backbone_hash = file_digest(args.backbone)
    verified = {"window": I.verify_backbone_provenance(checkpoint, backbone_hash)}
    if args.reference_checkpoint:
        reference, ck = load_model(args.reference_checkpoint, device)
        if reference.config.normalization != "mean":
            raise ValueError("reference checkpoint must be the legacy mean architecture")
        if ck["config"]["evidence"] != cfg["evidence"]:
            raise ValueError("reference and window evidence budgets must match")
        for key in ("window_ms", "max_wait_steps", "source_features"):
            if getattr(reference.config, key) != getattr(window_model.config, key):
                raise ValueError("reference/window model mismatch: " + key)
        verified["mean"] = I.verify_backbone_provenance(ck, backbone_hash)
        models = dict(mean_reference=reference.eval(), **models)
        checkpoints["mean"] = ck
    bb = torch_load(args.backbone, device)
    bcfg = dict(bb["config"], root=args.data_root, execution="step", input_device="cpu")
    if bcfg["window_ms"] != window_model.config.window_ms or window_model.config.source_features != 2:
        raise ValueError("live BIR adapter/window configuration mismatch")
    log_w = math.log(float(bb["stats"]["pos_weight"]))
    if any(not np.isclose(ck["log_w"], log_w, atol=1e-6, rtol=0) for ck in checkpoints.values()):
        raise ValueError("checkpoint and backbone prior scales differ")
    backbone = I.T.build_net(bcfg, device)
    backbone.load_state_dict(bb["model"])
    backbone.eval()
    if not bool(backbone.gain_calibrated):
        raise ValueError("BIR gain calibration missing")
    q99 = np.asarray(bb["stats"]["q99"], np.float32)
    disk_start = time.perf_counter()
    dataset = I.EvUAVStream(args.data_root, args.split, bcfg, names=[args.name])
    seq = dataset[0]
    rec, original = I.recording_from_sequence(seq, bcfg["window_ms"], log_w)
    disk_seconds = time.perf_counter()-disk_start
    if rec.n_windows != cfg["recording_n_windows"]:
        raise ValueError("recording duration differs from training configuration")
    pc = cfg["policy"]

    def execute(kind):
        source = I.make_window_source(seq, bcfg, q99, device, next(backbone.parameters()).dtype)
        provider = I.BackboneArrivalProvider(backbone, source, rec, original, len(seq.t), log_w)
        windows = []
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with recompute_pairs(kind == "window_recomputed"), window_timer(device, windows):
            result = stream.run_recording(models[kind], rec, cfg["evidence"], birth="band", policy=None,
                threshold=cfg["threshold"], band_alpha=pc["band_alpha"], band_kind=pc.get("band_kind", "sym"),
                batch_size=args.batch_size, device=device, wait_cost=pc["wait_cost"], arrival_provider=provider)
        sync(device)
        seconds = time.perf_counter()-start
        if len(windows) != rec.n_windows or not np.isfinite(provider.arrival_scores).all():
            raise RuntimeError("incomplete recording execution")
        result["prior"] = provider.arrival_scores
        timing = {"seconds": seconds, "windows": windows, "window_summary": describe_windows(windows, rec.window_ms),
                  "work": result["stats"], "backbone_seconds": provider.seconds,
                  "oracle_valid_pairs_recomputed": int(result["stats"]["pair_visits"]) if kind == "window_recomputed" else 0,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        return result, timing

    order = list(models)
    for kind in order:
        execute(kind)
        print("WARMUP " + kind, flush=True)
    records, first, comparisons = {kind: [] for kind in order}, {}, []
    execution_order = []
    for repeat in range(args.repeats):
        current = order[repeat % len(order):]+order[:repeat % len(order)]
        execution_order.append(current)
        for kind in current:
            result, timing = execute(kind)
            if kind in first:
                comparison = compare_outputs(result, first[kind], log_w, cfg["threshold"])
                comparisons.append({"kind": kind, "repeat": repeat, "comparison": "repeat_vs_first", **comparison})
            else:
                first[kind] = {k: result[k].copy() for k in ("logits", "lags", "reasons", "prior")}
            records[kind].append(timing)
            print("PROFILE %s repeat=%d %.3fs" % (kind, repeat, timing["seconds"]), flush=True)
    comparisons.append({"comparison": "window_cached_vs_recomputed",
                        **compare_outputs(first["window_cached"], first["window_recomputed"], log_w, cfg["threshold"])})
    summaries = {kind: {"median_seconds": float(np.median([r["seconds"] for r in values])),
                        "median_window_p95_ms": float(np.median([r["window_summary"]["p95_ms"] for r in values])),
                        "max_peak_allocated_bytes": max((r["peak_allocated_bytes"] or 0) for r in values) or None}
                 for kind, values in records.items()}
    scopes = {key: checkpoint_scope(ck, args.data_root) for key, ck in checkpoints.items()}
    report = {"complete": True, "args": vars(args), "torch": torch.__version__, "cuda": torch.version.cuda,
              "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "recording": rec.name, "events": len(rec), "recorded_seconds": rec.n_windows*rec.window_ms/1000,
              "disk_load_seconds_separate": disk_seconds, "config": cfg, "scope": scopes, "provenance_verified": verified,
              "backbone_sha256": backbone_hash, "window_checkpoint_sha256": file_digest(args.checkpoint),
              "reference_checkpoint_sha256": file_digest(args.reference_checkpoint) if args.reference_checkpoint else None,
              "timing_scope": "synchronized serial windows: representation+BIR+neighbor/cache+SDU+CPU output; excludes load/init/write",
              "memory_scope": "PyTorch CUDA peak allocated bytes with all comparison model weights resident; includes pipeline tensors, excludes driver/runtime reserved memory; not nvidia-smi process usage",
              "policy": "shared band birth and fixed deadline, no VoW policy; new config controls all implementations",
              "meaning": "mean-vs-window compares distinct architectures/weights, NOT equivalent-output optimization or accuracy benefit",
              "oracle_scope": "recomputed oracle retains cache allocations and rebuilds pairs on read; not a lean no-cache deployment",
              "work_count_scope": "work.pairs_prepared counts cache insertion; oracle_valid_pairs_recomputed additionally counts every valid pair rebuilt by oracle reads (padding slots excluded)",
              "probe_only": True, "execution_order": execution_order, "summaries": summaries,
              "runs": records, "comparisons": comparisons,
              "all_window_correctness_checks_passed": all(c["equivalent"] for c in comparisons)}
    out.mkdir(parents=True, exist_ok=True)
    for kind, values in first.items():
        np.savez_compressed(str(out/(kind+".npz")), **values)
    save_json(str(out/"profile.json"), report)
    if not report["all_window_correctness_checks_passed"]:
        raise AssertionError("correctness mismatch; inspect saved profile.json")
    print("WINDOW ARCHITECTURE PROFILE FINISHED: " + str(out), flush=True)


if __name__ == "__main__":
    main()
