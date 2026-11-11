"""Synthetic execution-cost comparison of legacy V11 and shared BST.

This is NOT an accuracy-matched speedup claim. Randomly initialized architectures
have different widths/heads and are not trained. Data loading, BIR, neighbor
search, training backward, and VoW are excluded. Both models read the same
indexed source events and causal prefixes; no query exits early.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "deco"))
from model.bst import BSTConfig, SharedSequentialModel
from sdn_v11 import SpikingDecisionNeuronsV11


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, device, repeats, warmup):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        synchronize(device)
        walls, devices, peak_extra = [], [], []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                initial_memory = torch.cuda.memory_allocated(device)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
            synchronize(device)
            start = time.perf_counter()
            if device.type == "cuda":
                begin.record()
            output = fn()
            if device.type == "cuda":
                end.record()
            synchronize(device)
            walls.append((time.perf_counter() - start) * 1000)
            if device.type == "cuda":
                devices.append(float(begin.elapsed_time(end)))
                peak_extra.append(int(torch.cuda.max_memory_allocated(device) - initial_memory))
            del output
        result = {"wall_ms": walls, "wall_ms_median": statistics.median(walls),
                  "wall_ms_min": min(walls), "wall_ms_max": max(walls)}
        if devices:
            result.update({"cuda_elapsed_ms": devices,
                           "cuda_elapsed_ms_median": statistics.median(devices),
                           "peak_extra_allocated_bytes": max(peak_extra)})
        return result


def make_case(n_sources, n_queries, device, seed=37, past=10, depth=10, per_window=16):
    n_windows = past + depth + 1
    if n_sources < n_windows * per_window:
        raise ValueError("sources must be >= (past + depth + 1) * per_window")
    if n_queries < 1:
        raise ValueError("queries must be positive")
    g = torch.Generator().manual_seed(seed)
    source = torch.stack([torch.randn(n_sources, generator=g) * 4 - 3,
                          torch.randint(0, 2, (n_sources,), generator=g).float()], dim=1)
    xy = torch.rand(n_sources, 2, generator=g) * torch.tensor([346., 260.])
    windows = torch.div(torch.arange(n_sources) * n_windows, n_sources, rounding_mode="floor")
    times = (windows.float() + torch.rand(n_sources, generator=g)) * 50
    qxy = torch.rand(n_queries, 2, generator=g) * torch.tensor([346., 260.])
    qtime = (past + torch.rand(n_queries, generator=g)) * 50
    prior = torch.randn(n_queries, generator=g) * 3 - 2
    polarity = torch.randint(0, 2, (n_queries,), generator=g).float()
    picks = []
    for w in range(n_windows):
        ids = torch.nonzero(windows == w, as_tuple=False).flatten()
        # Sampling with replacement is explicit and identical for both models.
        picks.append(ids[torch.randint(len(ids), (n_queries, per_window), generator=g)])
    indices = torch.cat(picks, dim=1)
    dxdy = xy[indices] - qxy[:, None]
    dt = times[indices] - qtime[:, None]
    dw = (windows[indices] - past).float()
    relative = torch.cat([dxdy, dt[..., None], dw[..., None]], dim=-1)
    feat = torch.cat([relative, source[indices][..., 1:2], source[indices][..., :1]], dim=-1)
    unique = int(torch.unique(indices).numel())
    result = {"source": source.to(device), "indices": indices.to(device), "relative": relative.to(device),
              "legacy_features": feat.to(device), "prior": prior.to(device), "polarity": polarity.to(device),
              "ok": torch.ones_like(indices, dtype=torch.bool).to(device),
              "valid": torch.ones(n_queries, depth + 1, dtype=torch.bool).to(device),
              "dist2": dxdy.square().sum(-1).to(device), "dw": dw.to(device)}
    meta = {"sources": n_sources, "queries": n_queries, "past_steps": past, "depth": depth,
            "neighbors_per_window": per_window, "total_neighbor_slots": n_windows * per_window,
            "query_source_pairs": int(indices.numel()), "unique_referenced_sources": unique,
            "source_embedding_reuse_vs_all_sources": float(indices.numel()) / n_sources,
            "source_embedding_reuse_vs_referenced_sources": float(indices.numel()) / unique,
            "synthetic_sampling": "uniform source IDs within each temporal window, with replacement",
            "query_fraction_is_example_only": True}
    return result, meta


def benchmark_case(n_sources, n_queries, args, device):
    data, meta = make_case(n_sources, n_queries, device, seed=args.seed)
    depth, past, per_window = meta["depth"], meta["past_steps"], meta["neighbors_per_window"]
    torch.manual_seed(args.seed)
    legacy = SpikingDecisionNeuronsV11(past + depth + 1, per_window, sigma0=20., dim=64,
                                      heads=4, dmax=depth, audit="reaudit", rho="mean").to(device).eval()
    shared = SharedSequentialModel(BSTConfig(max_wait_steps=depth)).to(device).eval()
    with torch.no_grad():
        legacy.v_out.weight.normal_(0, 0.1)
        shared.value_weight.normal_(0, 0.1)
    batch = args.batch_size

    def run_legacy():
        final = []
        for begin in range(0, n_queries, batch):
            sl = slice(begin, min(begin + batch, n_queries))
            trajectory = legacy(data["legacy_features"][sl], data["ok"][sl], data["dw"][sl],
                                data["valid"][sl], data["prior"][sl] / 5., data["prior"][sl],
                                data["polarity"][sl], depth, data["dist2"][sl])
            final.append(trajectory[:, -1])
        return torch.cat(final)

    def encode():
        return shared.encode_sources(data["source"])

    with torch.no_grad():
        cache = encode()

    def run_shared_reaudit(bank, trim, use_pair_cache=False):
        final = []
        for begin in range(0, n_queries, batch):
            sl = slice(begin, min(begin + batch, n_queries))
            prior, p = data["prior"][sl], data["polarity"][sl]
            state = shared.make_state(prior, p)
            cache_pairs = shared.prepare_pairs(bank, data["indices"][sl], data["relative"][sl]) if use_pair_cache else None
            for age in range(depth + 1):
                end = (past + age + 1) * per_window if trim else meta["total_neighbor_slots"]
                pairs = ({key: value[:, :end] for key, value in cache_pairs.items()}
                         if cache_pairs is not None and trim else cache_pairs)
                score, state, _ = shared.reaudit(bank, data["indices"][sl, :end],
                                                 data["relative"][sl, :end], state, prior, p, age,
                                                 pair_cache=pairs)
            final.append(score)
        return torch.cat(final)

    def run_shared_full():
        return run_shared_reaudit(encode(), False)

    def run_shared_trim():
        return run_shared_reaudit(encode(), True)

    jobs = [("legacy_current_full_shape", run_legacy),
            ("shared_source_encoding_only", encode),
            ("shared_reaudit_only_full_shape", lambda: run_shared_reaudit(cache, False)),
            ("shared_total_full_shape", run_shared_full),
            ("shared_total_prefix_trimmed", run_shared_trim),
            ("shared_total_pair_cached_full_shape", lambda: run_shared_reaudit(encode(), False, True)),
            ("shared_total_pair_cached_prefix_trimmed", lambda: run_shared_reaudit(encode(), True, True))]
    result = {"case": meta, "batch_size": batch,
              "legacy_parameters": sum(p.numel() for p in legacy.parameters()),
              "shared_parameters": sum(p.numel() for p in shared.parameters()), "timings": {}}
    for name, job in jobs:
        timing = measure(job, device, args.repeats, args.warmup)
        result["timings"][name] = timing
        print("N=%d Q=%d %s median=%.3f ms" % (n_sources, n_queries, name, timing["wall_ms_median"]), flush=True)
    base = result["timings"]["legacy_current_full_shape"]["wall_ms_median"]
    for name in ("shared_total_full_shape", "shared_total_prefix_trimmed",
                 "shared_total_pair_cached_full_shape", "shared_total_pair_cached_prefix_trimmed"):
        result["timings"][name]["legacy_to_shared_wall_ratio"] = base / result["timings"][name]["wall_ms_median"]
    # This checks the execution optimization, not equivalence to the legacy model.
    with torch.no_grad():
        full = run_shared_reaudit(cache, False)
        trim = run_shared_reaudit(cache, True)
        pair_cached = run_shared_reaudit(cache, True, True)
        result["prefix_trim_max_abs_difference"] = float((full - trim).abs().max().cpu())
        result["pair_cache_max_abs_difference"] = float((full - pair_cached).abs().max().cpu())
        if not torch.allclose(full, trim, atol=2e-5, rtol=2e-5):
            raise RuntimeError("prefix trimming changed shared-model output")
        if not torch.allclose(full, pair_cached, atol=2e-5, rtol=2e-5):
            raise RuntimeError("pair caching changed shared-model output")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sources", type=int, nargs="+", default=[10000, 50000])
    parser.add_argument("--queries", type=int, nargs="+", default=[500, 2500])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--out", default=str(ROOT / "Res" / "bst_local_20261111" / "kernel_benchmark.json"))
    args = parser.parse_args()
    if len(args.sources) != len(args.queries):
        parser.error("--sources and --queries must have the same number of entries")
    if min(args.batch_size, args.repeats, args.threads) < 1 or args.warmup < 0:
        parser.error("batch, repeats and threads must be positive; warmup must be nonnegative")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but unavailable")
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
    result = {"kind": "untrained_architecture_execution_cost_only", "torch": torch.__version__,
              "python": sys.version, "device": str(device), "cpu_threads": args.threads,
              "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "dtype": "float32", "repeats": args.repeats, "warmup": args.warmup,
              "seed": args.seed, "causal_contract": "arrival-window complete; later windows masked",
              "excluded": ["BIR backbone", "data loading", "neighbor search", "training backward", "VoW", "early exit"],
              "limitations": ["Different model widths and heads; no matched accuracy or training.",
                              "Synthetic query/source counts are examples, not a production selection budget.",
                              "CUDA elapsed includes device scheduling between events; not a sum of profiled kernel times.",
                              "Prefix trimming is reported separately from the same full-shape comparison.",
                              "Memory figures are extra PyTorch allocated bytes beyond resident inputs/models, not end-to-end peak VRAM."],
              "cases": []}
    for n, q in zip(args.sources, args.queries):
        result["cases"].append(benchmark_case(n, q, args, device))
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE " + str(Path(args.out).resolve()), flush=True)


if __name__ == "__main__":
    main()
