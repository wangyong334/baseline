"""Profile the actual V16 output policy, not the simultaneous ablation readouts.

Reports measured wall time, replay latency, memory and optional NVIDIA board
energy. Board joules include idle power and are not CPU/system or neuromorphic
energy. Input loading/preparation is reported separately from warmed step time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools.v16.train import (design_name, evidence_model, Records, infer_record, make_models,
                                     load_checkpoint, sync, distribution, validate_config)
from tools.v14.run_experiment import write_json, sha256
from train_stream_v1 import seed_everything
from utils.stream_common import load_flat_config


def smi(args):
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    return subprocess.check_output(["nvidia-smi"] + args, stderr=subprocess.DEVNULL,
                                   universal_newlines=True, timeout=5, **flags).strip()


def process_gpus():
    rows = smi(["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"])
    output = []
    for row in rows.splitlines():
        fields = [v.strip() for v in row.split(",")]
        if len(fields) == 2 and fields[0].isdigit():
            output.append((int(fields[0]), fields[1]))
    return output


class BoardPower(object):
    """Read-only, optional board power sampling. Never selects a GPU by guess."""
    def __init__(self, device, interval=.25):
        self.device, self.interval = device, interval
        self.samples, self.errors = [], []
        self.stop_event, self.thread = threading.Event(), None
        self.uuid, self.other_pids = None, []

    def sample(self):
        try:
            value = float(smi(["-i", self.uuid, "--query-gpu=power.draw", "--format=csv,noheader,nounits"]))
            if np.isfinite(value) and value >= 0:
                self.samples.append((time.perf_counter(), value))
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self.errors.append(str(error))

    def start(self):
        if self.device.type != "cuda" or not shutil.which("nvidia-smi"):
            return
        try:
            rows = process_gpus()
            ours = sorted(set(g for pid, g in rows if pid == os.getpid()))
            if len(ours) != 1:
                self.errors.append("Cannot uniquely resolve current process GPU UUID")
                return
            self.uuid = ours[0]
            self.other_pids = sorted({pid for pid, g in rows if g == self.uuid and pid != os.getpid()})
            self.sample()
            def worker():
                while not self.stop_event.wait(self.interval):
                    self.sample()
            self.thread = threading.Thread(target=worker, daemon=True)
            self.thread.start()
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self.errors.append(str(error))

    def finish(self, start, end):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=6)
        if self.uuid:
            self.sample()
            try:
                self.other_pids = sorted(set(self.other_pids) | {pid for pid, g in process_gpus()
                                                                if g == self.uuid and pid != os.getpid()})
            except (OSError, subprocess.SubprocessError):
                pass
        samples = np.asarray(self.samples, dtype=np.float64).reshape(-1, 2)
        result = {"gpu_uuid": self.uuid, "samples": len(samples), "board_joules": None,
                  "other_compute_pids_observed": self.other_pids, "errors": self.errors[:3],
                  "scope": "whole NVIDIA GPU board including idle; excludes CPU/system; not attributable to model if shared"}
        if len(samples) >= 2 and samples[0, 0] <= start and samples[-1, 0] >= end:
            inner = samples[(samples[:, 0] > start) & (samples[:, 0] < end)]
            times = np.r_[start, inner[:, 0], end]
            watts = np.interp(times, samples[:, 0], samples[:, 1])
            result["board_joules"] = float(np.sum(np.diff(times) * (watts[:-1] + watts[1:]) * .5))
            result["mean_board_watts"] = result["board_joules"] / max(end - start, 1e-9)
            result["method"] = "trapezoidal power integration with interpolated measurement boundaries"
        return result


def select_density_records(dataset, count):
    rows = []
    for i, path in enumerate(dataset.paths):
        # Select by input density only, never by prediction accuracy.
        with np.load(str(path)) as data:
            rows.append((int(data["ev_loc"].shape[0]), i))
    rows.sort()
    ranks = np.unique(np.linspace(0, len(rows) - 1, min(count, len(rows))).round().astype(int))
    return [rows[int(rank)][1] for rank in ranks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--records", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.records < 1 or args.repeats < 1:
        raise ValueError("records and repeats must be positive")
    destination = Path(args.out) / "profile.json"
    if destination.exists() and not args.overwrite:
        raise ValueError("Profile exists; use --overwrite deliberately")
    device = torch.device(args.device)
    ckpt = load_checkpoint(args.checkpoint, torch.device("cpu"))
    if ckpt.get("design") != design_name(ckpt.get("config", {})):
        raise ValueError("Not a V16 overlap checkpoint")
    supplied = load_flat_config(args.config)
    cfg = dict(ckpt["config"], root=str(Path(args.data_root).resolve()))
    for key in supplied:
        if key not in ("root", "save_root", "stats_path") and supplied[key] != cfg.get(key):
            raise ValueError("Profile config differs from checkpoint: " + key)
    validate_config(cfg)
    seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    net, head = make_models(cfg, device)
    net.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["fusion"])
    net.eval()
    head.eval()
    q99 = np.asarray(ckpt["stats"]["q99"], dtype=np.float32)
    del ckpt
    main_mode = "main"
    dataset = Records(cfg, args.split)
    selected = select_density_records(dataset, args.records)
    # One short real-input warm-up, outside measured energy/runtime.
    warm_seq = dataset[selected[0]]
    from dataset.overlap_windows import OverlapWindowSource
    source = OverlapWindowSource(warm_seq, cfg, q99, device)
    with torch.no_grad():
        x, events, _, _ = source.window(0)
        net(x, events, None)
    del source, x, events, warm_seq
    sync(device)
    power = BoardPower(device)
    power.start()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows, all_steps, all_net, all_latency = [], [], [], []
    total_events, total_reads, total_windows = 0, 0, 0
    start = time.perf_counter()
    for index in selected:
        reference = None
        for repeat in range(args.repeats):
            t0 = time.perf_counter()
            seq = dataset[index]
            load_seconds = time.perf_counter() - t0
            preds, timing, _ = infer_record(net, head, seq, cfg, q99, device,
                                             modes=[main_mode], step=True, diagnostics=False)
            sync(device)
            total_seconds = time.perf_counter() - t0
            pred = preds[main_mode]
            repeat_difference = 0. if reference is None else float(np.max(np.abs(pred - reference))) if pred.size else 0.
            if reference is None:
                reference = pred.copy()
            if repeat_difference > 1e-6:
                raise RuntimeError("Nondeterministic repeat outputs: " + seq.name)
            completion, finish = 0., []
            for window_end, cost in zip(seq.window_ends, timing["all_step_ms"]):
                completion = max(completion, float(window_end)) + cost
                finish.append(completion)
            latency = np.asarray(finish)[seq.last_window] - seq.t
            row = {"record": seq.name, "repeat": repeat, "events": seq.n_events,
                   "event_reads": len(seq.order) if evidence_model(cfg) else seq.n_events,
                   "windows": seq.n_windows,
                   "load_and_index_seconds": load_seconds,
                   "source_prepare_seconds": timing["source_prepare_seconds"],
                   "total_wall_seconds": total_seconds,
                   "events_per_second_including_preparation": seq.n_events / max(total_seconds, 1e-9),
                   "step_ms": distribution(timing["step_ms"]), "network_ms": distribution(timing["network_ms"]),
                   "event_replay_latency_ms": distribution(latency),
                   "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest(),
                   "repeat_max_abs_diff": repeat_difference}
            rows.append(row)
            all_steps.extend(timing["step_ms"])
            all_net.extend(timing["network_ms"])
            all_latency.append(latency)
            total_events += seq.n_events
            total_reads += len(seq.order)
            total_windows += seq.n_windows
            print("PROFILE %s repeat %d %.2fs, %d events, mean step %s ms" %
                  (seq.name, repeat + 1, total_seconds, seq.n_events, row["step_ms"]["mean"]), flush=True)
    sync(device)
    end = time.perf_counter()
    energy = power.finish(start, end)
    if energy["board_joules"] is not None:
        energy["board_joules_per_million_events"] = energy["board_joules"] * 1e6 / max(total_events, 1)
    dense_macs = 0 if evidence_model(cfg) else head.dense_macs(int(cfg["pad_height"]), int(cfg["pad_width"]))
    paired_events = total_reads - total_events
    fusion_macs = head.estimated_macs(paired_events) if evidence_model(cfg) else None
    result = {"design": design_name(cfg), "checkpoint_sha256": sha256(args.checkpoint), "seed": cfg["seed"],
              "device": str(device), "torch": torch.__version__,
              "precision": "float32; CUDA matmul and cuDNN TF32 disabled",
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "split": args.split, "selection": "input event-count quantiles (includes minimum/maximum for records>=2)",
              "repeats": args.repeats, "readout_policy": cfg["revision_variant"], "rows": rows,
              "events": total_events, "event_reads": total_reads if evidence_model(cfg) else total_events,
              "windows": total_windows, "context_dense_macs_per_window": dense_macs,
              "paired_events": paired_events, "fusion_event_macs": fusion_macs,
              "total_wall_seconds": end - start,
              "events_per_second_including_preparation": total_events / max(end - start, 1e-9),
              "step_ms": distribution(all_steps), "network_ms": distribution(all_net),
              "event_replay_latency_ms": distribution(np.concatenate(all_latency)),
              "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
              "energy": energy,
              "accounting": {"step": "input construction + backbone + trained fusion + output transfer, no diagnostics/metrics",
                             "preparation": "record load, stable event ID indexing and GPU sparse-input upload included in total wall",
                             "latency": "single-device arrival replay using measured step costs; excludes whole-record preload; not live sensor measurement",
                             "memory": "includes resident whole-record sparse input; pending decisions are only one half-window",
                             "power": "whole measured profile interval including preparation and reporting; optional nvidia-smi sampling overhead present; no theoretical MAC-to-joule conversion"}}
    write_json(destination, result)
    print("PROFILE COMPLETE " + str(destination), flush=True)


if __name__ == "__main__":
    main()
