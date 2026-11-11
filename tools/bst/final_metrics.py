"""Final raw-stream accuracy, delay, runtime, MAC subtotal and GPU-board energy.

Five default declared policies, never selected using test accuracy. Full raw recordings;
warmup per recording/policy and repeated synchronized timing. Accuracy uses only
the first measured pass; energy/MAC instrumentation has separate scope labels.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bst import infer as I
from tools.bst.energy_meter import NVMLBoard, EnergyInterval
from tools.bst.profile_inference import sync, window_timer, describe_windows, mac_hooks
from tools.bst.pipeline import load_model, torch_load, file_digest, checkpoint_scope, save_json

POLICIES = ("bir", "band_arrival", "band_fixed", "band_learned", "learned_learned")


def commit_policy(kind, continuation):
    """Age-zero SDU commits after its first read, before any later window."""
    if kind not in POLICIES:
        raise ValueError("unknown declared policy: " + str(kind))
    if kind == "band_arrival":
        return lambda features, risk, age: torch.ones_like(risk, dtype=torch.bool)
    return continuation if kind.endswith("learned") else None


def other_compute_pids(uuid):
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                       "--format=csv,noheader,nounits"], timeout=5).decode()
        return sorted({int(line.split(",")[1]) for line in text.splitlines()
                       if line.split(",")[0].strip() == uuid and int(line.split(",")[1]) != os.getpid()})
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def summarize(rows):
    if not rows or any(not r["repeats"] for r in rows):
        raise ValueError("performance summary requires measured recordings")
    events = sum(r["events"] for r in rows)
    # Sum per-recording median compute, not median per-event or mean of rates.
    seconds = sum(float(np.median([x["seconds"] for x in r["repeats"]])) for r in rows)
    duration = sum(r["duration_seconds"] for r in rows)
    energies = [np.median([x["energy"]["board_joules"] for x in r["repeats"]])
                if all(x["energy"]["board_joules"] is not None for x in r["repeats"]) else None for r in rows]
    joules = sum(energies) if all(e is not None for e in energies) else None
    return {"events": events, "recorded_seconds": duration, "sum_recording_median_compute_seconds": seconds,
            "events_per_second": events/seconds if seconds else None,
            "real_time_factor": seconds/duration if duration else None,
            "sum_recording_median_board_joules": joules,
            "board_joules_per_million_events": joules/events*1e6 if joules is not None and events else None,
            "board_energy_not_process_attribution": True}


def calibration_stats(logits, labels, bins=15):
    """Unweighted original-event probabilities; no fitting or test tuning."""
    z, y = np.asarray(logits, np.float64), np.asarray(labels, np.float64)
    probability = 1/(1+np.exp(-np.clip(z, -60, 60)))
    which = np.minimum((probability*bins).astype(int), bins-1)
    return {"count": len(z), "brier_sum": float(np.square(probability-y).sum()),
            "nll_sum": float((np.logaddexp(0,z)-y*z).sum()),
            "bin_count": np.bincount(which, minlength=bins).tolist(),
            "bin_probability_sum": np.bincount(which, weights=probability, minlength=bins).tolist(),
            "bin_label_sum": np.bincount(which, weights=y, minlength=bins).tolist()}


def pool_calibration(values):
    count = sum(x["count"] for x in values)
    b = np.sum([x["bin_count"] for x in values], axis=0)
    prob = np.sum([x["bin_probability_sum"] for x in values], axis=0)
    lab = np.sum([x["bin_label_sum"] for x in values], axis=0)
    return {"count": count, "brier": sum(x["brier_sum"] for x in values)/max(count,1),
            "nll": sum(x["nll_sum"] for x in values)/max(count,1),
            "ece_15_equal_width_bins": float(np.abs(prob-lab).sum()/max(count,1)),
            "bin_count": b.tolist(), "bin_probability_sum": prob.tolist(), "bin_label_sum": lab.tolist(),
            "scale": "sigmoid(z), not cost-shifted official report probability; approximate probability, no calibration guarantee"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ["checkpoint", "policy", "backbone", "data-root", "out"]:
        ap.add_argument("--"+key, required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--physical-gpu", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--allow-probe", action="store_true", help="local tests only; do not claim final experiment")
    ap.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    ap.add_argument("--skip-macs", action="store_true")
    args = ap.parse_args()
    if args.repeats < 1 or args.batch_size < 1 or args.threads < 1 or args.limit < 0:
        ap.error("invalid execution limits")
    if len(set(args.policies)) != len(args.policies):
        ap.error("policy names must be unique")
    outdir = Path(args.out)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError("choose a new metrics directory: " + args.out)
    board = NVMLBoard(args.physical_gpu)
    if not board.uuid:
        raise RuntimeError("Cannot verify requested physical GPU through NVML: " + str(board.error))
    if torch.cuda.is_initialized():
        raise RuntimeError("GPU identity must be bound before CUDA initialization")
    os.environ["CUDA_VISIBLE_DEVICES"] = board.uuid
    torch.set_num_threads(args.threads)
    device = torch.device("cuda:0")
    model, ck = load_model(args.checkpoint, device)
    cfg = ck["config"]
    if model.config.source_features != 2:
        raise ValueError("live BIR adapter supplies [z0, polarity] only")
    I.validate_official_threshold(cfg)
    scope = checkpoint_scope(ck, args.data_root)
    verified = I.verify_backbone_provenance(ck, file_digest(args.backbone))
    complete_priors = bool(ck.get("manifest", {}).get("prior_provenance", {}).get("complete", False))
    probe = scope["probe"] or not verified or not complete_priors or bool(args.limit)
    if probe and not args.allow_probe:
        raise ValueError("final measurement requires complete training and verified BIR identity")
    bb = torch_load(args.backbone, device)
    bcfg = dict(bb["config"], root=args.data_root, execution="step", input_device="cpu")
    if bcfg["window_ms"] != cfg["model"]["window_ms"]:
        raise ValueError("BIR/SDU window mismatch")
    log_w = math.log(float(bb["stats"]["pos_weight"]))
    if not np.isclose(log_w, ck["log_w"], atol=1e-6, rtol=0):
        raise ValueError("BIR prior scale mismatch")
    backbone = I.T.build_net(bcfg, device)
    backbone.load_state_dict(bb["model"])
    if not bool(backbone.gain_calibrated):
        raise ValueError("uncalibrated BIR gain")
    backbone.eval()
    continuation, birth_head, meta = I.load_policy(args.policy, args.checkpoint, model, device)
    if not np.isclose(log_w, meta["log_w"], atol=1e-6, rtol=0):
        raise ValueError("policy prior scale mismatch")
    pc = meta["config"]["policy"]
    q99 = np.asarray(bb["stats"]["q99"], np.float32)
    dataset = I.EvUAVStream(args.data_root, args.split, bcfg)
    names = dataset.names[:args.limit] if args.limit else dataset.names
    if not names:
        raise ValueError("no raw recordings in the requested split")
    grouped = set(I.dense_names(args.data_root, (args.split,), bcfg["height"], bcfg["width_px"])[args.split])
    metrics = {name: I.PolicyStats(name, bcfg["window_ms"]) for name in args.policies}
    rows = {name: [] for name in args.policies}
    outdir.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(args), "torch": torch.__version__, "hardware": torch.cuda.get_device_name(0),
              "cuda_runtime": torch.version.cuda,
              "parameter_counts": {name: sum(p.numel() for p in module.parameters()) for name, module in
                  (("backbone", backbone), ("SDU", model), ("VoW_continue", continuation), ("VoW_birth", birth_head))},
              "gpu_uuid": board.uuid, "checkpoint_sha256": file_digest(args.checkpoint),
              "backbone_sha256": file_digest(args.backbone), "policy_sha256": file_digest(args.policy),
              "config": cfg, "probe": probe, "complete_prior_provenance": complete_priors,
              "timing_scope": "synchronized serial windows: representation+BIR+neighbor/cache+SDU/VoW+CPU output; excludes file load/init/metrics/write",
              "energy_reference": "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html",
              "energy_scope": "gross GPU board energy; not CPU, wall-plug or neuromorphic energy; no idle subtraction",
              "mac_scope": "Conv2d/Linear subtotal only; not all ops/FLOPs; dense kernels, not spike-skipped hardware",
              "policy_protocol": "five_policy_with_age0_v2", "default_policy_set": list(POLICIES),
              "selection": "Predeclared policies in args.policies; test results do not select weights or thresholds",
              "age0_control": ("band_arrival runs SDU once; birth-referenced innovation must equal BIR at age zero, validating the zero-innovation boundary"
                               if model.config.innovation else
                               "band_arrival runs SDU once at admission, commits before later windows; separates arrival refinement from waiting for new evidence"),
              "policies": {}}
    for name in names:
        started = time.perf_counter()
        seq = dataset[dataset.names.index(name)]
        disk_seconds = time.perf_counter()-started
        rec, original = I.recording_from_sequence(seq, bcfg["window_ms"], log_w)
        if cfg.get("recording_n_windows") != rec.n_windows:
            raise ValueError("recording horizon mismatch")

        def execute(kind, measured=False):
            source = I.make_window_source(seq, bcfg, q99, device, next(backbone.parameters()).dtype)
            provider = I.BackboneArrivalProvider(backbone, source, rec, original, len(seq.t), log_w)
            birth, callback = "band", None
            if kind == "bir":
                birth, callback = "learned", lambda ids, live, w: np.zeros(len(ids), bool)
            elif kind == "learned_learned":
                birth, holder = "learned", []
                def callback(ids, live, w):
                    if not holder:
                        holder.append(I.make_birth_callback(birth_head, model, live, cfg["threshold"], log_w,
                                                            pc["init_cost"], meta["birth_context"], device))
                    return holder[0](ids, live, w)
            policy = commit_policy(kind, continuation)
            windows = []
            sync(device)
            torch.cuda.reset_peak_memory_stats(device)
            other0 = other_compute_pids(board.uuid) if measured else None
            meter = EnergyInterval(board)
            if measured:
                meter.begin()
            tick = time.perf_counter()
            try:
                with window_timer(device, windows):
                    output = I.run_recording(model, rec, cfg["evidence"], policy=policy, birth=birth,
                        threshold=cfg["threshold"], band_alpha=pc["band_alpha"], batch_size=args.batch_size,
                        device=device, wait_cost=pc["wait_cost"], birth_policy=callback,
                        arrival_provider=provider, band_kind=pc.get("band_kind", "sym"))
                sync(device)
            finally:
                seconds = time.perf_counter()-tick
                energy = meter.end() if measured else None
            if not np.isfinite(provider.arrival_scores).all() or len(windows) != rec.n_windows:
                raise RuntimeError("incomplete streaming execution")
            if measured:
                other1 = other_compute_pids(board.uuid)
                energy["foreign_compute_pids_before"] = other0
                energy["foreign_compute_pids_after"] = other1
                energy["no_other_compute_at_boundaries"] = other0 == [] and other1 == []
                energy["not_proof_of_exclusivity_between_boundaries"] = True
            return output, provider.arrival_scores, {"seconds": seconds, "backbone_seconds": provider.seconds,
                "energy": energy, "window_summary": describe_windows(windows, rec.window_ms), "windows": windows,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "work": output["stats"]}

        # Rotate policy order across recordings so BIR is not always measured first.
        start = names.index(name) % len(args.policies)
        order = args.policies[start:]+args.policies[:start]
        for kind in order:
            execute(kind)  # warmup, not reported
            repeats, first = [], None
            for repeat in range(args.repeats):
                output, prior, timing = execute(kind, measured=True)
                if first is None:
                    first = {"logits": output["logits"].copy(), "lags": output["lags"].copy(),
                             "reasons": output["reasons"].copy()}
                    scored = I.Recording(rec.name, rec.x, rec.y, rec.t, rec.p, prior, rec.label,
                                         rec.window_ms, rec.log_w, rec.n_windows, rec.target_id)
                    mr = I.metric_record(scored)
                    metrics[kind].add(mr, I.report_prob(output["logits"], log_w), output["lags"],
                                      I.references(mr, 0, rec.window_ms), dense=name in grouped)
                    dest = outdir/kind
                    dest.mkdir(exist_ok=True)
                    np.savez_compressed(dest/name, logits=output["logits"], z0=prior, lag=output["lags"],
                                        reasons=output["reasons"], labels=rec.label, original_ids=original,
                                        log_w=log_w, window_ms=rec.window_ms)
                    y = rec.label > .5
                    initial = I.report_prob(prior, log_w) >= np.float32(cfg["threshold"])
                    final = I.report_prob(output["logits"], log_w) >= np.float32(cfg["threshold"])
                    error_changes = {"rescued_prior_errors": int(((initial != y)&(final == y)).sum()),
                                     "introduced_errors": int(((initial == y)&(final != y)).sum()),
                                     "rescued_target_events": int((~initial & final & y).sum()),
                                     "harmed_target_events": int((initial & ~final & y).sum()),
                                     "removed_false_positives": int((initial & ~final & ~y).sum()),
                                     "introduced_false_positives": int((~initial & final & ~y).sum()),
                                     "false_negative_events": int((~final & y).sum()),
                                     "false_positive_events": int((final & ~y).sum())}
                    calibration = calibration_stats(output["logits"], rec.label)
                else:
                    np.testing.assert_array_equal(output["lags"], first["lags"])
                    np.testing.assert_array_equal(output["reasons"], first["reasons"])
                    np.testing.assert_allclose(output["logits"], first["logits"], atol=2e-6, rtol=2e-6)
                    np.testing.assert_array_equal(I.report_prob(output["logits"], log_w) >= np.float32(cfg["threshold"]),
                                                  I.report_prob(first["logits"], log_w) >= np.float32(cfg["threshold"]))
                repeats.append(timing)
                print("METRIC %s %s repeat=%d %.3fs energy=%s" % (name, kind, repeat, timing["seconds"], timing["energy"]["board_joules"]), flush=True)
            counts = defaultdict(int)
            if not args.skip_macs:
                with mac_hooks({"BIR": backbone, "SDU": model, "VoW_continue": continuation, "VoW_birth": birth_head}, counts):
                    execute(kind)
            rows[kind].append({"recording": name, "events": len(rec), "duration_seconds": rec.n_windows*rec.window_ms/1000.,
                               "disk_load_seconds_separate": disk_seconds, "repeats": repeats, "linear_conv_mac_subtotal": dict(counts),
                               "calibration": calibration, "actual_commit_error_changes": error_changes})
            report["policies"][kind] = {"accuracy_and_evidence_wait": metrics[kind].result(),
                                       "performance": summarize(rows[kind]), "recordings": rows[kind],
                                       "actual_commit_error_changes": {k: sum(r["actual_commit_error_changes"][k] for r in rows[kind]) for k in error_changes},
                                       "calibration": pool_calibration([r["calibration"] for r in rows[kind]])}
            report["complete"] = False
            save_json(str(outdir/"metrics.json"), report)
    report["complete"] = True
    save_json(str(outdir/"metrics.json"), report)
    print("BST FINAL METRICS FINISHED", flush=True)


if __name__ == "__main__":
    main()
