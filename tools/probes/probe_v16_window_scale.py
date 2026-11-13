"""Window-scale transfer diagnostic; this is NOT matched-window training.

Fit input normalization on train only, then test a fixed 100/50ms backbone on
three validation recordings selected by input event count. No test data, model
training, threshold selection, or formal model/configuration is modified.
Python 3.8 / torch 1.9 compatible.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dataset import stream_windows as sw
from dataset.overlap_windows import load_overlap_npz, update_overlap_histogram
from tools.v16 import train as tr


LABEL = "window shift diagnostic, unmatched frozen weights, cannot establish new-window trained accuracy"
WARNINGS = [
    LABEL,
    "Only validation records selected by input density are evaluated; no test data is used.",
    "Backbone weights and calibrated gains remain those trained at 100/50ms; larger windows are distribution shifts.",
    "time_bins=5 and input channel count remain fixed, so temporal bin width becomes 20/40/80ms.",
    "BIR dt equals stride, while learned tau in physical milliseconds stays fixed; the per-step decay therefore changes.",
    "Fresh train-fitted q99 changes input normalization, not trained weights or gain calibration.",
    "Results cannot isolate longer context from temporal coarsening, state-clock, normalization, or edge-coverage effects.",
    "Second-only inference has no learned paired fusion and cannot establish the benefit of the r4 fusion module.",
    "Pending event counts describe the two-observation design; second-only diagnostic inference does not store first-view features.",
    "Timing includes online representation, readouts, communication collection, and synchronization; excludes metrics computation.",
    "Acquisition latency is separate from runtime. Increased stride can improve throughput while worsening decision delay.",
]


def scaled_config(base, window):
    cfg = copy.deepcopy(base)
    stride = window // 2
    cfg.update(window_ms=window, stride_ms=stride,
               n_windows=(int(cfg["total_ms"]) - window) // stride + 1,
               revision_variant="second", timing_warmup_windows=1000 // stride)
    if "dt_ms" in cfg:
        cfg["dt_ms"] = stride
    return cfg


def hist_distribution(hist):
    hist = np.asarray(hist, np.int64)
    count = int(hist.sum())
    if not count:
        return {"count": 0}
    occupied = np.flatnonzero(hist)
    cumulative = hist.cumsum()
    def quantile(q):
        return int(np.searchsorted(cumulative, max(1, math.ceil(q * count))))
    return {"count": count, "min": int(occupied[0]), "mean": float(np.dot(hist, np.arange(len(hist))) / count),
            "p50": quantile(.5), "p90": quantile(.9), "p95": quantile(.95), "p99": quantile(.99),
            "max": int(occupied[-1])}


def record_properties(seq):
    paired = seq.coverage_count == 2
    # Events retained after their first observation, before next-window finalization.
    pending = np.bincount(seq.first_window[paired], minlength=seq.n_windows)
    latency = seq.output_time_ms - seq.t
    return {"record": seq.name, "events": seq.n_events, "windows": seq.n_windows,
            "expanded_reads": int(seq.order.size), "paired_events": int(paired.sum()),
            "single_events": int((~paired).sum()), "pending_events": tr.distribution(pending),
            "algorithmic_latency_ms": tr.distribution(latency)}, pending, latency


def fit_scale(cfg, paths):
    """One CPU pass computes the same input histogram as formal train stats."""
    started = time.perf_counter()
    hist = sw.new_count_histogram(sw.num_input_channels(cfg["time_bins"]))
    latency_hist = np.zeros(cfg["window_ms"] + 1, np.int64)
    pending_all, rows = [], []
    pos = neg = reads = windows = paired = singles = 0
    for i, path in enumerate(paths):
        seq = load_overlap_npz(path, cfg["height"], cfg["width_px"], cfg["window_ms"],
                               cfg["stride_ms"], cfg["total_ms"], cfg["time_bins"])
        for k in range(seq.n_windows):
            update_overlap_histogram(hist, seq, k, cfg["pad_height"], cfg["pad_width"])
        row, pending, latency = record_properties(seq)
        rows.append(row)
        pending_all.extend(pending.tolist())
        latency_hist += np.bincount(latency, minlength=len(latency_hist))
        n_pos = int((seq.label == 1).sum())
        pos += n_pos
        neg += seq.n_events - n_pos
        reads += seq.order.size
        windows += seq.n_windows
        paired += row["paired_events"]
        singles += row["single_events"]
        if (i + 1) % 20 == 0 or i + 1 == len(paths):
            print("STATS %d/%d W%d/S%d %.1fs" % (i + 1, len(paths), cfg["window_ms"], cfg["stride_ms"], time.perf_counter()-started), flush=True)
    return {"diagnostic_label": LABEL, "split": "train", "files": len(paths),
            "q99": [sw.quantile_from_histogram(h, .99) for h in hist],
            "nonzero_pixels_per_channel": [int(h[1:].sum()) for h in hist],
            "source_files": [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths],
            "window_ms": cfg["window_ms"], "stride_ms": cfg["stride_ms"],
            "time_bins": cfg["time_bins"], "bin_width_ms": cfg["window_ms"] / cfg["time_bins"],
            "n_windows_per_record": cfg["n_windows"], "windows": windows,
            "events": pos + neg, "train_pos": pos, "train_neg": neg,
            "expanded_reads": int(reads), "reads_per_event": reads / float(pos + neg),
            "paired_events": paired, "single_events": singles,
            "pending_events_per_step": tr.distribution(pending_all),
            "algorithmic_latency_ms": hist_distribution(latency_hist),
            "per_record": rows, "cpu_seconds": time.perf_counter()-started}


def density_selected_validation(base):
    records = tr.Records(base, "val")
    inventory = [(records[i].n_events, records.paths[i].name) for i in range(len(records))]
    inventory.sort()
    ranks = [0, len(inventory)//2, len(inventory)-1]
    return [{"density_rank": rank, "events": inventory[rank][0], "record": inventory[rank][1]}
            for rank in sorted(set(ranks))]


@torch.no_grad()
def infer_scale(cfg, checkpoint, q99, names, device, repeats):
    tr.seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    net, head = tr.make_models(cfg, device)
    net.load_state_dict(checkpoint["model"], strict=True)
    net.eval(); head.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dataset = tr.Records(cfg, "val", names)
    counts = np.zeros(3, np.int64)
    aggregate_roc = tr.FastRoc(cfg["pd_detT"], cfg["correct_thresh"], cfg["height"], cfg["width_px"])
    rows, timing_all, net_all = [], [], []
    communication = []
    for i in range(len(dataset)):
        seq = dataset[i]
        row, _, _ = record_properties(seq)
        runs = []
        first = None
        for repeat in range(repeats):
            tr.sync(device)
            started = time.perf_counter()
            probs, timing, messages = tr.infer_record(net, head, seq, cfg, q99, device,
                                                      modes=["second"], step=True, diagnostics=True)
            tr.sync(device)
            elapsed = time.perf_counter() - started
            probability = probs["second"]
            if first is None:
                first = probability
                c = np.asarray(tr.window_confusion(probability, seq.label, np.float32(cfg["threshold"]))[:3])
                counts += c
                roc = tr.FastRoc(cfg["pd_detT"], cfg["correct_thresh"], cfg["height"], cfg["width_px"])
                for metric in (roc, aggregate_roc):
                    metric.update(seq.t, probability, seq.target_id, seq.label, seq.x, seq.y, cfg["threshold"])
                row["metrics"] = tr.metric_row(c, roc)
            elif not np.array_equal(probability, first):
                raise RuntimeError("Non-repeatable online diagnostic: " + seq.name)
            completion, done = 0., []
            for end, duration in zip(seq.window_ends, timing["all_step_ms"]):
                completion = max(completion, float(end)) + duration
                done.append(completion)
            runs.append({"repeat": repeat + 1, "wall_seconds": elapsed,
                         "events_per_second": seq.n_events / elapsed,
                         "step_ms": tr.distribution(timing["step_ms"]),
                         "network_ms": tr.distribution(timing["network_ms"]),
                         "source_prepare_seconds": timing["source_prepare_seconds"],
                         "replay_latency_ms": tr.distribution(np.asarray(done)[seq.last_window] - seq.t),
                         "messages": tr.distribution(messages), "total_messages": float(np.sum(messages)),
                         "messages_per_second_of_recording": float(np.sum(messages)) / (cfg["total_ms"] / 1000.)})
            timing_all.extend(timing["step_ms"])
            net_all.extend(timing["network_ms"])
            communication.extend(messages)
        row["runs"] = runs
        rows.append(row)
        print("VAL W%d/S%d %s IoU %.5f recall %.5f FP%d %.2fs" %
              (cfg["window_ms"], cfg["stride_ms"], seq.name, row["metrics"]["iou"], row["metrics"]["acc"],
               row["metrics"]["fp"], runs[0]["wall_seconds"]), flush=True)
    reads = sum(row["expanded_reads"] for row in rows)
    readout_macs_per_event = (int(cfg["channels"][0]) + 2) * int(cfg["readout_hidden"]) + int(cfg["readout_hidden"])
    return {"readout": "second_only", "threshold": cfg["threshold"], "metrics": tr.metric_row(counts, aggregate_roc),
            "per_record": rows, "step_ms": tr.distribution(timing_all), "network_ms": tr.distribution(net_all),
            "communication_messages_per_step": tr.distribution(communication),
            "operations": {"backbone_parameters": tr.count_parameters(net),
                           "backbone_forward_steps_per_record": cfg["n_windows"],
                           "event_readouts_per_pass": reads,
                           "readout_linear_macs_per_pass": reads * readout_macs_per_event,
                           "fusion_macs": 0,
                           "scope": "Readout linear MACs exclude backbone, indexing, neuron dynamics and representation; no energy claim."},
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="run/local_checks/r4_local.yaml")
    parser.add_argument("--out", default="Res/v16_window_scale_diagnostic_20261113")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase", choices=("stats", "inference", "all"), default="stats")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.repeats < 1:
        parser.error("repeats must be positive")
    base = tr.load_flat_config(args.config)
    if (base["window_ms"], base["stride_ms"], base["time_bins"]) != (100, 50, 5):
        raise ValueError("Diagnostic requires the known 100/50ms 5-bin anchor")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(base["init_checkpoint"])
    checkpoint = tr.load_checkpoint(checkpoint_path, torch.device("cpu"))
    if (checkpoint["config"]["window_ms"], checkpoint["config"]["stride_ms"]) != (100, 50):
        raise ValueError("Expected 100/50ms-trained initialization")
    fingerprint = tr.sha256(checkpoint_path)
    stats = {}
    if args.phase in ("stats", "all"):
        paths = tr.Records(base, "train").paths
        for window in (100, 200, 400):
            stats[window] = fit_scale(scaled_config(base, window), paths)
            tr.write_json(out / ("train_stats_w%d.json" % window), stats[window])
        selected = density_selected_validation(base)
        tr.write_json(out / "validation_selection.json", {"criterion": "input event count min/median/max; no label selection", "records": selected})
    else:
        for window in (100, 200, 400):
            stats[window] = json.loads((out / ("train_stats_w%d.json" % window)).read_text(encoding="utf-8"))
        selected = json.loads((out / "validation_selection.json").read_text(encoding="utf-8"))["records"]
    base_q99 = np.asarray(checkpoint["stats"]["q99"], np.float32)
    if not np.allclose(base_q99, stats[100]["q99"], atol=1e-6, rtol=0):
        raise RuntimeError("100/50ms all-train q99 differs from frozen checkpoint; diagnose provenance first")
    result = {"diagnostic_label": LABEL, "warnings": WARNINGS, "checkpoint": str(checkpoint_path),
              "checkpoint_sha256": fingerprint, "checkpoint_epoch": checkpoint.get("epoch"),
              "checkpoint_design": checkpoint.get("design"),
              "anchor_window_ms": 100, "anchor_stride_ms": 50, "selected_validation_records": selected,
              "threshold": .9, "pd_detT_ms": base["pd_detT"], "device": args.device,
              "torch": torch.__version__, "statistics": {str(k): v for k,v in stats.items()}, "experiments": []}
    if args.phase in ("inference", "all"):
        names = [row["record"] for row in selected]
        for window in (100, 200, 400):
            cfg = scaled_config(base, window)
            variants = [("original_100_50_train_q99", base_q99)]
            if window != 100:
                variants.append(("new_window_train_q99", np.asarray(stats[window]["q99"], np.float32)))
            for norm, q99 in variants:
                value = infer_scale(cfg, checkpoint, q99, names, torch.device(args.device), args.repeats)
                value.update(window_ms=window, stride_ms=window//2, n_windows=cfg["n_windows"],
                             normalization=norm, q99=q99.tolist(), diagnostic_label=LABEL,
                             trained_for_this_window=(window==100))
                result["experiments"].append(value)
                tr.write_json(out / "summary.json", result)
    tr.write_json(out / "summary.json", result)
    print("DIAGNOSTIC COMPLETE " + str(out / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
