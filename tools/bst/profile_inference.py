"""Profile the complete online pipeline on raw recordings, without changing weights.

Times include window representation, BIR, neighbor search, transfers, SDU/VoW,
and cache/queue maintenance. Disk loading, initialization and output writing
are separate. Window timings synchronize CUDA. MAC hooks run in a separate
untimed pass and count Conv2d/Linear only, NOT full FLOPs or sparse SOPs.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import importlib.util
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bst import infer as I
from tools.bst.data import Recording
from tools.bst.data import NeighborIndex
from tools.bst.stream import CausalSourceCache, CausalPairCache
from model.bst import SharedSequentialModel, ContinuationValueHead, BirthValueHead
from tools.bst.pipeline import checkpoint_scope, file_digest, load_model, save_json, torch_load

TIMED_RECORDINGS = [Recording]
REFERENCE_MODEL_MODULE = None


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def install_reference(root):
    """Load a saved executor/index snapshot; never overwrite working sources."""
    global Recording, REFERENCE_MODEL_MODULE
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / "model/bst.py"
    if file_digest(path) != manifest["model/bst.py"]:
        raise ValueError("reference model snapshot changed")
    spec = importlib.util.spec_from_file_location("bst_reference_model", path)
    REFERENCE_MODEL_MODULE = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = REFERENCE_MODEL_MODULE
    spec.loader.exec_module(REFERENCE_MODEL_MODULE)
    loaded = {}
    for name in ("data", "stream"):
        path = root / "tools/bst" / (name + ".py")
        if file_digest(path) != manifest["tools/bst/" + name + ".py"]:
            raise ValueError("reference snapshot changed")
        spec = importlib.util.spec_from_file_location("bst_reference_" + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    loaded["stream"].NeighborIndex = loaded["data"].NeighborIndex
    loaded["stream"].Recording = loaded["data"].Recording
    Recording = loaded["data"].Recording
    TIMED_RECORDINGS.append(Recording)
    I.Recording = Recording
    I.run_recording = loaded["stream"].run_recording


@contextmanager
def window_timer(device, rows):
    originals = [(cls, cls.iter_windows) for cls in set(TIMED_RECORDINGS)]
    for cls, original in originals:
        def measured(rec, *args, _original=original, **kwargs):
            for window, ids in _original(rec, *args, **kwargs):
                sync(device)
                start = time.perf_counter()
                yield window, ids
                sync(device)
                rows.append({"window": int(window), "events": len(ids),
                             "ms": (time.perf_counter() - start) * 1000})
        cls.iter_windows = measured
    try:
        yield
    finally:
        for cls, original in originals:
            cls.iter_windows = original


def describe_windows(rows, window_ms):
    values = np.asarray([r["ms"] for r in rows])
    backlog, delays = 0., []
    # Single stream, arrivals every window_ms; no assumed overlap/pipelining.
    for duration in values:
        backlog = max(0., backlog - window_ms) + duration
        delays.append(backlog)
    return {"mean_ms": float(values.mean()), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "max_ms": float(values.max()),
            "over_window_fraction": float((values > window_ms).mean()),
            "serial_response_max_ms": float(max(delays)),
            "serial_unfinished_at_next_arrival_max_ms": float(max(0., max(delays) - window_ms))}


@contextmanager
def mac_hooks(modules, counts):
    handles = []
    for group, parent in modules.items():
        if parent is None:
            continue
        for module in parent.modules():
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue
            def count(layer, inputs, output, group=group):
                if isinstance(layer, torch.nn.Linear):
                    mac = output.numel() * layer.in_features
                else:
                    mac = output.numel() * (layer.in_channels // layer.groups) * math.prod(layer.kernel_size)
                counts[group] += int(mac)
            handles.append(module.register_forward_hook(count))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def stage_timers(device, totals):
    """Extra synchronizations: diagnostic pass, never mixed with main timing."""
    originals = []
    targets = [(I.BackboneArrivalProvider, "__call__", "representation_BIR"),
               (NeighborIndex, "query", "neighbor_query"),
               (NeighborIndex, "query_arriving", "neighbor_query"),
               (CausalSourceCache, "ensure", "source_encode_cache_insert"),
               (CausalSourceCache, "evict_before", "source_cache_evict"),
               (CausalSourceCache, "local_indices", "source_index_map"),
               (CausalPairCache, "allocate", "pair_cache_allocate"),
               (CausalPairCache, "append", "pair_cache_append_including_index_map"),
               (CausalPairCache, "read", "pair_cache_read"),
               (SharedSequentialModel, "reaudit", "SDU_reaudit"),
               (ContinuationValueHead, "forward", "VoW_continue"),
               (BirthValueHead, "forward", "VoW_birth")]
    # append invokes local_indices; omit that nested timer to keep disjoint sums.
    targets = [item for item in targets if item[2] != "source_index_map"]
    for cls, method, label in targets:
        original = getattr(cls, method)
        originals.append((cls, method, original))
        def measured(*args, _fn=original, _label=label, **kwargs):
            sync(device)
            started = time.perf_counter()
            try:
                return _fn(*args, **kwargs)
            finally:
                sync(device)
                totals[_label] += time.perf_counter() - started
        setattr(cls, method, measured)
    try:
        yield
    finally:
        for cls, method, original in originals:
            setattr(cls, method, original)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--policy")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--names", nargs="*", help="default: min/median/max event-count recordings")
    ap.add_argument("--modes", nargs="+", default=["fixed", "learned"], choices=["fixed", "learned"])
    ap.add_argument("--birth", choices=["band", "all", "learned"], default="band")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1024, help="current executor query batch; reference uses saved training batch")
    ap.add_argument("--count-macs", action="store_true")
    ap.add_argument("--stage-profile", action="store_true", help="extra instrumented run; synchronization adds overhead")
    ap.add_argument("--reference-dir", help="use the archived pre-optimization executor/index")
    ap.add_argument("--paired-reference", help="alternate archived/current runs in one process, capture and compare outputs")
    ap.add_argument("--capture-outputs", action="store_true", help="extra untimed pass: save all event outputs and all active-query traces")
    ap.add_argument("--compare-to", help="directory of reference output NPZ files; requires --capture-outputs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.repeats < 1 or args.threads < 1 or args.batch_size < 1:
        ap.error("repeats and threads must be positive")
    if ("learned" in args.modes or args.birth == "learned") and not args.policy:
        ap.error("learned policy requires --policy")
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    if args.compare_to and not args.capture_outputs:
        ap.error("--compare-to requires --capture-outputs")
    current_executor = I.run_recording
    if args.reference_dir and args.paired_reference:
        ap.error("choose either reference-only or paired comparison")
    if args.reference_dir or args.paired_reference:
        if args.stage_profile:
            if args.reference_dir:
                ap.error("stage timers target the current implementation")
        install_reference(args.reference_dir or args.paired_reference)
    executors = {"reference": I.run_recording, "optimized": current_executor} if args.paired_reference else {"selected": I.run_recording}
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, ck = load_model(args.checkpoint, device)
    reference_model = None
    if REFERENCE_MODEL_MODULE is not None:
        reference_model = REFERENCE_MODEL_MODULE.SharedSequentialModel(
            REFERENCE_MODEL_MODULE.BSTConfig(**ck["config"]["model"])).to(device)
        reference_model.load_state_dict(ck["model"])
    cfg = ck["config"]
    I.validate_official_threshold(cfg)
    verified = I.verify_backbone_provenance(ck, file_digest(args.backbone))
    bb = torch_load(args.backbone, device)
    bcfg = dict(bb["config"], root=args.data_root, execution="step", input_device="cpu")
    if int(bcfg["window_ms"]) != int(model.config.window_ms):
        raise ValueError("BIR/SDU window mismatch")
    backbone = I.T.build_net(bcfg, device)
    backbone.load_state_dict(bb["model"])
    backbone.eval()
    if not bool(backbone.gain_calibrated):
        raise ValueError("BIR gains not calibrated")
    log_w = math.log(float(bb["stats"]["pos_weight"]))
    if ck.get("log_w") is not None and not np.isclose(ck["log_w"], log_w):
        raise ValueError("prior scale mismatch")
    q99 = np.asarray(bb["stats"]["q99"], np.float32)
    continuation, birth_head, meta = None, None, None
    pc = cfg["policy"]
    if args.policy:
        continuation, birth_head, meta = I.load_policy(args.policy, args.checkpoint, model, device)
        pc = meta["config"]["policy"]
        if not np.isclose(meta["log_w"], log_w):
            raise ValueError("policy scale mismatch")
    dataset = I.EvUAVStream(args.data_root, args.split, bcfg)
    inventory = []
    for i, name in enumerate(dataset.names):
        seq = dataset[i]
        counts = np.bincount(seq.t // int(bcfg["window_ms"]), minlength=seq.n_windows)
        inventory.append({"name": name, "events": len(seq.t), "windows": seq.n_windows,
                          "duration_s": seq.n_windows * bcfg["window_ms"] / 1000.,
                          "max_events_per_window": int(counts.max()),
                          "p95_events_per_window": float(np.percentile(counts, 95))})
    ranked = sorted(inventory, key=lambda r: r["events"])
    names = args.names if args.names else list(dict.fromkeys(ranked[k]["name"] for k in [0, len(ranked)//2, -1]))
    result = {"schema": 1, "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "torch": torch.__version__, "device": str(device), "threads": args.threads,
              "dtype": "float32", "args": vars(args), "config": cfg,
              "backbone_sha256": file_digest(args.backbone), "checkpoint_sha256": file_digest(args.checkpoint),
              "log_w": log_w,
              "checkpoint_scope": checkpoint_scope(ck, args.data_root), "provenance_verified": verified,
              "selection": "event count only; no labels or accuracy used", "inventory": inventory,
              "timing_scope": "raw windows -> representation/BIR -> index/cache/SDU/VoW -> CPU outputs; excludes disk/init/metrics/write",
              "limitations": ["Local probe weights do not establish final model speed/accuracy.",
                              "Average throughput is not a per-window real-time guarantee.",
                              "MAC hooks count dense Conv2d/Linear only; exclude attention contractions and nonlinear/index operations.",
                              "No claim of hardware sparse execution from spike sparsity.",
                              "Window sync and Python instrumentation add overhead; no artificial sleeps."],
              "cases": []}
    for name in names:
        if name not in dataset.names:
            raise ValueError("unknown recording " + name)
        disk_start = time.perf_counter()
        seq = dataset[dataset.names.index(name)]
        disk_s = time.perf_counter() - disk_start
        rec, ids = I.recording_from_sequence(seq, bcfg["window_ms"], log_w)
        if cfg.get("recording_n_windows") not in (None, rec.n_windows):
            raise ValueError("recording horizon mismatch")

        def run(mode, capture=False, implementation="selected"):
            net = reference_model if implementation == "reference" or args.reference_dir else model
            source = I.make_window_source(seq, bcfg, q99, device, next(backbone.parameters()).dtype)
            provider = I.BackboneArrivalProvider(backbone, source, rec, ids, len(seq.t), log_w)
            holder = []
            def birth_callback(query_ids, live_rec, window):
                if not holder:
                    holder.append(I.make_birth_callback(birth_head, model, live_rec, cfg["threshold"], log_w,
                                                        pc["init_cost"], meta["birth_context"], device))
                return holder[0](query_ids, live_rec, window)
            rows = []
            sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            with window_timer(device, rows):
                out = executors[implementation](net, rec, cfg["evidence"],
                    policy=continuation if mode == "learned" else None, birth=args.birth,
                    threshold=cfg["threshold"], band_alpha=pc["band_alpha"],
                    batch_size=cfg["train"]["batch_size"] if implementation == "reference" or args.reference_dir else args.batch_size,
                    device=device, wait_cost=pc["wait_cost"],
                    birth_policy=birth_callback if args.birth == "learned" else None,
                    arrival_provider=provider, band_kind=pc.get("band_kind", "sym"), return_traces=capture)
            sync(device)
            elapsed = time.perf_counter() - start
            if len(rows) != rec.n_windows or not np.isfinite(provider.arrival_scores).all():
                raise RuntimeError("incomplete timed stream")
            if capture:
                arrays = {key: out[key] for key in ("logits", "lags", "reasons")}
                arrays["prior"] = provider.arrival_scores.copy()
                arrays["prediction"] = I.report_prob(out["logits"], log_w) >= np.float32(cfg["threshold"])
                for key in ("ids", "logits", "features", "committed", "updated"):
                    arrays["trace_" + key] = np.concatenate([t[key] for t in out["traces"]])
                arrays["trace_sizes"] = np.asarray([len(t["ids"]) for t in out["traces"]])
                folder = Path(args.out).with_suffix("")
                if args.paired_reference:
                    folder = folder / implementation
                folder.mkdir(exist_ok=True, parents=True)
                np.savez_compressed(folder / (name[:-4] + "_" + mode + ".npz"), **arrays)
                checks = {}
                compare_to = args.compare_to
                if args.paired_reference and implementation == "optimized":
                    compare_to = Path(args.out).with_suffix("") / "reference"
                if compare_to:
                    with np.load(Path(compare_to) / (name[:-4] + "_" + mode + ".npz"), allow_pickle=False) as baseline:
                        for key, values in arrays.items():
                            other = baseline[key]
                            if values.shape != other.shape:
                                raise AssertionError("changed trace shape: " + key)
                            exact = bool(np.array_equal(values, other))
                            entry = {"exact": exact}
                            if np.issubdtype(values.dtype, np.floating):
                                entry["max_abs_difference"] = float(np.max(np.abs(values-other))) if values.size else 0.
                                np.testing.assert_allclose(values, other, atol=2e-6, rtol=2e-6, err_msg=key)
                            else:
                                np.testing.assert_array_equal(values, other, err_msg=key)
                            checks[key] = entry
                return {"arrays": str(folder / (name[:-4] + "_" + mode + ".npz")), "comparison": checks,
                        "note": "separate correctness pass; capture/serialization excluded from timing runs"}
            return {"seconds": elapsed, "backbone_seconds": provider.seconds,
                    "other_pipeline_seconds": elapsed - provider.seconds, "stats": out["stats"],
                    "events_per_second": len(rec) / elapsed,
                    "real_time_factor": elapsed / (rec.n_windows * rec.window_ms / 1000.),
                    "window_summary": describe_windows(rows, rec.window_ms), "windows": rows,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}

        for mode in args.modes:
            implementations = list(executors)
            runs_by = {key: [] for key in implementations}
            for impl in implementations:
                print("WARMUP %s %s %s" % (name, mode, impl), flush=True)
                run(mode, implementation=impl)
            for repeat in range(args.repeats):
                for impl in (implementations if repeat % 2 == 0 else implementations[::-1]):
                    value = run(mode, implementation=impl)
                    runs_by[impl].append(value)
                    print("%s %s %s repeat=%d seconds=%.3f p95_window_ms=%.2f" %
                          (name, mode, impl, repeat, value["seconds"], value["window_summary"]["p95_ms"]), flush=True)
            for impl in implementations:
                runs = runs_by[impl]
                counts = defaultdict(int)
                if args.count_macs:
                    with mac_hooks({"BIR": backbone, "SDU": reference_model if impl == "reference" or args.reference_dir else model, "VoW_continue": continuation,
                                    "VoW_birth": birth_head}, counts):
                        run(mode, implementation=impl)
                stage_seconds = defaultdict(float)
                if args.stage_profile and impl != "reference":
                    with stage_timers(device, stage_seconds):
                        stage_result = run(mode, implementation=impl)
                    stage_seconds["other_queue_control_and_uninstrumented"] = stage_result["seconds"] - sum(stage_seconds.values())
                correctness = run(mode, capture=True, implementation=impl) if args.capture_outputs or args.paired_reference else None
                case = {"recording": name, "events": len(rec), "duration_s": rec.n_windows * rec.window_ms / 1000.,
                        "mode": mode, "implementation": impl, "disk_load_seconds_separate": disk_s,
                        "median_seconds": float(np.median([r["seconds"] for r in runs])),
                        "stage_seconds_separate_instrumented_pass": dict(stage_seconds),
                        "correctness": correctness,
                        "linear_conv_mac_subtotal": dict(counts), "runs": runs}
                result["cases"].append(case)
                save_json(args.out, result)
    print("WROTE " + str(Path(args.out).resolve()), flush=True)


if __name__ == "__main__":
    main()
