"""V15 online latency/throughput and board-energy probe; no accuracy selection."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
import argparse
import json
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
from dataset.ev_uav_stream import EvUAVStream
from dataset.stream_source import make_window_source
from tools.v15.train import (DESIGN, make_models, load_checkpoint, infer_record, validate_config,
                             write_json, sha256, training_is_complete_scope)
from train_stream_v1 import seed_everything
from utils.stream_common import load_flat_config


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    return {"count": int(x.size), "mean": float(x.mean()) if x.size else None,
            "p50": float(np.percentile(x, 50)) if x.size else None,
            "p95": float(np.percentile(x, 95)) if x.size else None,
            "max": float(x.max()) if x.size else None}


def validate_profile_checkpoint(checkpoint, requested):
    """A measurement must identify the actual geometry and revision model."""
    if checkpoint.get("design") != DESIGN:
        raise ValueError("Checkpoint is not the current V15 native revision design")
    cfg = checkpoint["config"]
    keys = {key for key in set(cfg) | set(requested)
            if not key.startswith("_") and key not in ("root", "save_root", "stats_path")}
    for key in sorted(keys):
        if cfg.get(key) != requested.get(key):
            raise ValueError("Checkpoint/config mismatch: " + key)
    return dict(cfg)


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
        if end > start and len(samples) >= 2 and samples[0, 0] <= start and samples[-1, 0] >= end:
            inner = samples[(samples[:, 0] > start) & (samples[:, 0] < end)]
            times = np.r_[start, inner[:, 0], end]
            watts = np.interp(times, samples[:, 0], samples[:, 1])
            result["board_joules"] = float(np.sum(np.diff(times) * (watts[:-1] + watts[1:]) * .5))
            result["mean_board_watts"] = result["board_joules"] / max(end - start, 1e-9)
            result["method"] = "trapezoidal power integration with interpolated measurement boundaries"
        result["sharing_check"] = "compute-process snapshots before and after workload; transient sharing may be missed"
        return result


def density_records(dataset, count):
    rows = []
    for i, name in enumerate(dataset.names):
        with np.load(str(Path(dataset.directory) / name)) as data:
            rows.append((int(data["ev_loc"].shape[0]), i))
    rows.sort()
    ranks = np.unique(np.linspace(0, len(rows) - 1, min(count, len(rows))).round().astype(int))
    return [rows[int(rank)][1] for rank in ranks]


@torch.no_grad()
def infer_v5(net, seq, cfg, q99, device):
    """Same checkpoint, original native-step V5 path; overhead reference only."""
    sync(device)
    t0 = time.perf_counter()
    source = make_window_source(seq, cfg, q99, device, next(net.parameters()).dtype)
    sync(device)
    prepare = time.perf_counter() - t0
    states = None
    pred = np.empty(seq.n_events, dtype=np.float32)
    steps, network = [], []
    for k in range(seq.n_windows):
        sync(device)
        t0 = time.perf_counter()
        x, events, _, idx = source.window(k)
        sync(device)
        t1 = time.perf_counter()
        logits, states, _ = net(x, events, states)
        sync(device)
        pred[idx] = torch.sigmoid(logits).cpu().numpy()
        sync(device)
        network.append((time.perf_counter() - t1) * 1000.)
        steps.append((time.perf_counter() - t0) * 1000.)
    release = seq.t.astype(np.int64) // int(cfg["window_ms"])
    return {"final": pred}, {"all_step_ms": steps, "network_ms": network,
                             "source_prepare_seconds": prepare}, {
        "release_window": release, "output_time_ms": (release + 1) * cfg["window_ms"],
        "event_readouts": seq.n_events}


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
    if min(args.records, args.repeats) < 1:
        raise ValueError("Positive records/repeats required")
    target = Path(args.out) / "profile.json"
    if target.exists() and not args.overwrite:
        raise ValueError("Profile already exists")
    device = torch.device(args.device)
    ckpt = load_checkpoint(args.checkpoint, torch.device("cpu"))
    requested = load_flat_config(args.config)
    cfg = validate_profile_checkpoint(ckpt, requested)
    checkpoint_training_scope = cfg.get("_training_scope")
    checkpoint_probe_only = not training_is_complete_scope(cfg)
    cfg["root"] = args.data_root
    validate_config(cfg)
    seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    net, head = make_models(cfg, device)
    net.load_state_dict(ckpt["model"])
    if head is not None:
        head.load_state_dict(ckpt["revision"])
    q99 = np.asarray(ckpt["stats"]["q99"], dtype=np.float32)
    del ckpt
    net.eval()
    if head is not None:
        head.eval()
    dataset = EvUAVStream(cfg["root"], args.split, cfg)
    selected = density_records(dataset, args.records)
    warm = dataset[selected[0]]
    infer_record(net, head, warm, cfg, q99, device, modes=("final",), step=True, diagnostics=False)
    infer_v5(net, warm, cfg, q99, device)
    del warm
    variants = {}
    for variant in ("v5_same_backbone", "v15_final"):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        power = BoardPower(device)
        power.start()
        sync(device)
        start = time.perf_counter()
        rows, steps, networks, latencies, acquisition = [], [], [], [], []
        total_events, record_seconds = 0, 0.
        for index in selected:
            reference = None
            for repeat in range(args.repeats):
                t0 = time.perf_counter()
                seq = dataset[index]
                loading = time.perf_counter() - t0
                sync(device)
                t0 = time.perf_counter()
                if variant == "v15_final":
                    preds, timing, detail = infer_record(net, head, seq, cfg, q99, device,
                                                        modes=("final",), step=True, diagnostics=False)
                else:
                    preds, timing, detail = infer_v5(net, seq, cfg, q99, device)
                sync(device)
                seconds = time.perf_counter() - t0
                prediction = preds["final"]
                difference = 0. if reference is None or not prediction.size else float(np.max(np.abs(prediction - reference)))
                if not np.isfinite(prediction).all() or difference > 1e-6:
                    raise RuntimeError("Nonfinite/nondeterministic profile output: " + seq.name)
                reference = prediction.copy()
                if detail["event_readouts"] != seq.n_events:
                    raise RuntimeError("Main inference must read each event exactly once")
                done, completion = [], 0.
                for k, cost in enumerate(timing["all_step_ms"]):
                    completion = max(completion, (k + 1) * cfg["window_ms"]) + cost
                    done.append(completion)
                delay = np.asarray(done)[detail["release_window"]] - seq.t
                # Keep compact arrays: converting millions of event delays to
                # Python floats can dominate the measured workload/energy.
                latencies.append(delay)
                acquisition.append(np.asarray(detail["output_time_ms"]) - seq.t)
                steps.extend(timing["all_step_ms"])
                networks.extend(timing["network_ms"])
                total_events += seq.n_events
                record_seconds += seconds
                rows.append({"record": seq.name, "repeat": repeat, "events": seq.n_events,
                             "windows": seq.n_windows, "seconds": seconds, "disk_load_seconds": loading,
                             "source_prepare_seconds": timing["source_prepare_seconds"],
                             "repeat_max_abs_error": difference,
                             "prediction_sha256": __import__("hashlib").sha256(prediction.tobytes()).hexdigest()})
        sync(device)
        end = time.perf_counter()
        energy = power.finish(start, end)
        energy["measurement_scope"] = "entire repeat workload, including record loading, source preparation and validation bookkeeping"
        energy["workload_joules_per_event"] = (energy["board_joules"] / total_events
                                                if energy.get("board_joules") is not None and total_events else None)
        energy["model_only_joules"] = None
        event_readout_macs = total_events * ((int(cfg["channels"][0]) + 2) *
                                            int(cfg["readout_hidden"]) + int(cfg["readout_hidden"]))
        variants[variant] = {"records": rows, "step_ms": distribution(steps),
                             "network_ms": distribution(networks), "events": total_events,
                             "inference_seconds_including_source": record_seconds,
                             "events_per_second": total_events / max(record_seconds, 1e-9),
                             "acquisition_delay_ms": distribution(np.concatenate(acquisition)),
                             "single_device_replay_delay_ms": distribution(np.concatenate(latencies)),
                             "measurement_wall_seconds": end - start, "board_energy": energy,
                             "workload_events_per_second": total_events / max(end - start, 1e-9),
                             "event_readout_dense_macs": event_readout_macs,
                             "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None}
        print("PROFILE %s %.2f events/s" % (variant, variants[variant]["events_per_second"]), flush=True)
    result = {"design": DESIGN, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
              "config_sha256": sha256(args.config), "sensor_hw": [cfg["height"], cfg["width_px"]],
              "padded_canvas_hw": [cfg["pad_height"], cfg["pad_width"]],
              "checkpoint_probe_only": checkpoint_probe_only,
              "checkpoint_training_scope": checkpoint_training_scope,
              "split": args.split, "record_selection": "even ranks by event density, including extremes",
              "selected": [dataset.names[i] for i in selected], "variants": variants,
              "revision_parameters": sum(p.numel() for p in head.parameters()) if head is not None else 0,
              "revision_dense_macs_per_pair_plus_new_encoding": head.dense_macs(cfg["pad_height"], cfg["pad_width"]) if head is not None else 0,
              "backbone_parameters": sum(p.numel() for p in net.parameters()),
              "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "torch": torch.__version__, "precision": "float32; TF32 disabled",
              "protocol": "native 50 ms chunks, one backbone call per chunk; final main readout only",
              "operation_accounting": "Revision and event readout use dense ANN MACs in addition to the SNN backbone; MACs are not joules.",
              "throughput_scope": {"events_per_second": "record inference wall time including source construction and CPU prediction output; excludes disk loading",
                                   "workload_events_per_second": "complete power-metered repeat loop including disk loading and bookkeeping"},
              "limitations": ["Timing comparison uses the same learned backbone, not separately trained accuracy ablations.",
                              "Whole-record sparse arrays are preloaded; this is not an event-camera driver benchmark.",
                              "Board energy includes idle, input preparation, I/O and Python; CPU/system energy unmeasured.",
                              "Extra revision MACs exclude nonlinear/interpolation/memory costs; no neuromorphic energy claim.",
                              "Replay delay models arrival backlog using measured online steps; source-preload cost excluded.",
                              "Step/network timers include synchronization and CPU output transfer, not pure CUDA kernel time.",
                              "Variants run in fixed order after warmup; clock/thermal drift can affect close timing differences."]}
    write_json(target, result)
    print("PROFILE_SAVED " + str(target), flush=True)


if __name__ == "__main__":
    main()


