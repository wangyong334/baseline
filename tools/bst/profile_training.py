"""Short paired training-step benchmark, including geometry preparation.

The archived model/index/rollout, current default rollout, and current bounded
geometry cache receive the same query IDs, inclusion weights and minibatches.
Each timing trial restarts model and AdamW. Correctness capture runs separately
so copying per-step logits/gradients/parameters does not pollute timing. This is
not an epoch-time prediction or a benchmark of converged accuracy.
"""
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import model.bst as current_model
import tools.bst.data as current_data
from tools.bst import pipeline as current_pipeline
from tools.bst.training_geometry import GeometryBatches

SOURCE_PATHS = ("model/bst.py", "tools/bst/data.py", "tools/bst/pipeline.py",
                "tools/bst/training_geometry.py")
LOADED_SOURCE_SHA256 = {name: current_pipeline.file_digest(ROOT/name) for name in SOURCE_PATHS}


KINDS = ("server_snapshot", "current_default", "current_geometry")


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def load_reference(snapshot):
    """Isolate archived modules and rebind globals after their ordinary imports."""
    snapshot = Path(snapshot)
    if (snapshot/"server_source").is_dir():
        snapshot = snapshot/"server_source"
    loaded, hashes = {}, {}
    for name, relative in (("model", "model/bst.py"), ("data", "tools/bst/data.py"),
                           ("pipeline", "tools/bst/pipeline.py")):
        path = snapshot/relative
        if not path.is_file():
            raise FileNotFoundError("missing archived training dependency: " + str(path))
        hashes[relative] = current_pipeline.file_digest(path)
        module_name = "bst_training_reference_"+name+"_"+hashes[relative][:12]
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    old = loaded["pipeline"]
    old.BSTConfig = loaded["model"].BSTConfig
    old.SharedSequentialModel = loaded["model"].SharedSequentialModel
    for name in ("NeighborIndex", "compact_neighbors", "load_recording", "sample_queries"):
        setattr(old, name, getattr(loaded["data"], name))
    return SimpleNamespace(**loaded, source_sha256=hashes, snapshot=str(snapshot.resolve()))


@contextmanager
def query_timer(index):
    original = index.query
    result = {"seconds": 0., "calls": 0, "rows": 0}
    def timed(ids, *args, **kwargs):
        tick = time.perf_counter()
        try:
            return original(ids, *args, **kwargs)
        finally:
            result["seconds"] += time.perf_counter()-tick
            result["calls"] += 1
            result["rows"] += len(ids)
    index.query = timed
    try:
        yield result
    finally:
        index.query = original


def flat_parameters(model, gradients=False):
    values = [(p.grad if p.grad is not None else torch.zeros_like(p)) if gradients else p
              for p in model.parameters()]
    return torch.cat([v.detach().reshape(-1) for v in values]).cpu().numpy().copy()


def run_sequence(model, rec, index, pipeline, ordered_ids, weights, config, device,
                 geometry=False, cache_mb=64., capture=False):
    """Measure one complete short SGD sequence; no model or index is reused."""
    device = torch.device(device)
    ids, weights = np.asarray(ordered_ids, np.int64), np.asarray(weights, np.float64)
    if not len(ids) or weights.shape != ids.shape:
        raise ValueError("nonempty ordered IDs and matching inclusion weights required")
    batch_size = int(config["train"]["batch_size"])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"],
                                  weight_decay=config["train"]["weight_decay"])
    prepared = GeometryBatches(index, ids, batch_size, cache_mb) if geometry else None
    iterator = iter(prepared) if prepared is not None else iter(
        (start, min(start+batch_size, len(ids)), None) for start in range(0, len(ids), batch_size))
    totals = {name: 0. for name in ("fetch_seconds", "forward_loss_seconds_excluding_query",
              "backward_seconds", "clip_seconds", "optimizer_seconds")}
    totals.update(batches=0, queries=len(ids), source_encoded_rows=0, valid_pair_visits=0)
    snapshots = []
    if device.type == "cuda":
        sync(device)
        torch.cuda.reset_peak_memory_stats(device)
    with query_timer(index) as queries:
        started = time.perf_counter()
        while True:
            tick = time.perf_counter()
            try:
                start, stop, batch = next(iterator)
            except StopIteration:
                totals["fetch_seconds"] += time.perf_counter()-tick
                break
            totals["fetch_seconds"] += time.perf_counter()-tick
            optimizer.zero_grad()
            sync(device)
            tick, before_query = time.perf_counter(), queries["seconds"]
            if prepared is None:
                output = pipeline.rollout(model, rec, index, ids[start:stop], device)
            else:
                output = pipeline.rollout(model, rec, index, ids[start:stop], device, prepared_batch=batch)
            labels = torch.as_tensor(rec.label[ids[start:stop]], dtype=torch.float32, device=device)
            inclusion = torch.as_tensor(weights[start:stop], dtype=torch.float32, device=device)
            loss = pipeline.weighted_event_loss(output["logits"], output["valid"], labels, inclusion) / float(batch_size)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite loss in training benchmark")
            sync(device)
            totals["forward_loss_seconds_excluding_query"] += time.perf_counter()-tick-(queries["seconds"]-before_query)
            tick = time.perf_counter()
            loss.backward()
            sync(device)
            totals["backward_seconds"] += time.perf_counter()-tick
            if capture:
                raw_gradients = flat_parameters(model, gradients=True)
            tick = time.perf_counter()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("nonfinite gradient in training benchmark")
            sync(device)
            totals["clip_seconds"] += time.perf_counter()-tick
            tick = time.perf_counter()
            optimizer.step()
            sync(device)
            totals["optimizer_seconds"] += time.perf_counter()-tick
            if capture:
                snapshots.append({"logits": output["logits"].detach().cpu().numpy().copy(),
                    "valid": output["valid"].detach().cpu().numpy().copy(), "gradients": raw_gradients,
                    "gradient_present": np.asarray([p.grad is not None for p in model.parameters()]),
                    "parameters": flat_parameters(model), "loss": np.asarray(float(loss.detach()))})
            totals["batches"] += 1
            totals["source_encoded_rows"] += int(output["source_count"])
            totals["valid_pair_visits"] += int(output["pair_count"])
            del output, loss, labels, inclusion
        sync(device)
        totals["wall_seconds"] = time.perf_counter()-started
        totals["query_seconds"] = queries["seconds"]
        totals["query_calls"] = queries["calls"]
        totals["query_rows"] = queries["rows"]
    totals["geometry_stats"] = dict(prepared.stats) if prepared is not None else None
    totals["geometry_materialize_seconds"] = prepared.stats["geometry_materialize_seconds"] if prepared is not None else 0.
    totals["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    totals["capture_pass_not_timing_result"] = bool(capture)
    return totals, snapshots


def compare_steps(actual, expected, atol=2e-6, rtol=2e-5):
    if len(actual) != len(expected):
        raise ValueError("training sequences have different batch counts")
    errors = {key: 0. for key in ("logits", "gradients", "parameters", "loss")}
    close = {key: True for key in errors}
    exact = True
    for a, b in zip(actual, expected):
        for key in ("valid", "gradient_present"):
            exact = exact and np.array_equal(a[key], b[key])
        for key in errors:
            if a[key].shape != b[key].shape:
                raise ValueError("training tensor shape differs: " + key)
            errors[key] = max(errors[key], float(np.max(np.abs(a[key]-b[key]))) if a[key].size else 0.)
            close[key] = close[key] and bool(np.allclose(a[key], b[key], atol=atol, rtol=rtol))
    return {"steps": len(actual), "maximum_absolute_differences": errors, "allclose": close,
            "valid_and_gradient_presence_exact": exact, "atol": atol, "rtol": rtol,
            "equivalent": exact and all(close.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data-root", "z0-root", "recording", "snapshot", "out"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-mb", type=float, default=64.)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args(argv)
    if min(args.batches, args.repeats, args.threads) < 1:
        parser.error("batches, repeats and threads must be positive")
    if Path(args.recording).name != args.recording or not args.recording.startswith("train_"):
        parser.error("recording must name one training NPZ, without directory components")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("choose an empty benchmark directory")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint = current_pipeline.torch_load(args.checkpoint, "cpu")
    config = checkpoint["config"]
    if config["model"].get("normalization", "mean") != "mean":
        raise ValueError("snapshot comparison requires the common mean architecture checkpoint")
    reference = load_reference(args.snapshot)
    pair = current_pipeline.paths(args.data_root, args.z0_root, "train", names=[args.recording])[0]
    current_rec, _ = current_pipeline.get_record(pair, config)
    old_rec, _ = reference.pipeline.get_record(pair, config)
    for key in ("x", "y", "t", "p", "label", "z0", "window"):
        if not np.array_equal(getattr(current_rec, key), getattr(old_rec, key)):
            raise ValueError("archived and current loaders disagree: " + key)
    if not np.isclose(current_rec.log_w, checkpoint["log_w"], atol=1e-6, rtol=0):
        raise ValueError("prior and checkpoint score scales differ")
    rng = np.random.default_rng(args.seed)
    sample = current_pipeline.choose_queries(current_rec, config["train"]["background_budget"], rng)
    order = rng.permutation(len(sample.ids))[:args.batches*config["train"]["batch_size"]]
    ids, weights = sample.ids[order], sample.weights[order]
    modules = {"server_snapshot": reference,
               "current_default": SimpleNamespace(model=current_model, data=current_data, pipeline=current_pipeline),
               "current_geometry": SimpleNamespace(model=current_model, data=current_data, pipeline=current_pipeline)}
    def trial(kind, capture=False):
        code = modules[kind]
        model = code.model.SharedSequentialModel(code.model.BSTConfig(**config["model"])).to(device)
        model.load_state_dict(checkpoint["model"])
        rec = old_rec if kind == "server_snapshot" else current_rec
        index = code.data.NeighborIndex(rec, **config["evidence"])
        sync(device)
        return run_sequence(model, rec, index, code.pipeline, ids, weights, config, device,
                            geometry=kind == "current_geometry", cache_mb=args.cache_mb, capture=capture)
    out.mkdir(parents=True, exist_ok=True)
    correctness = {}
    for kind in KINDS:
        _, correctness[kind] = trial(kind, capture=True)
        arrays = {"%s_%03d" % (key, i): value for i, step in enumerate(correctness[kind]) for key,value in step.items()}
        np.savez_compressed(str(out/("correctness_"+kind+".npz")), **arrays)
    comparisons = {kind: compare_steps(correctness[kind], correctness["server_snapshot"]) for kind in KINDS[1:]}
    if not all(row["equivalent"] for row in comparisons.values()):
        current_pipeline.save_json(str(out/"correctness_failure.json"), comparisons)
        raise RuntimeError("training equivalence failed; refusing performance claim")
    del correctness
    for kind in KINDS:
        trial(kind)  # independent warmup; no state carries into measured trials
    runs = {kind: [] for kind in KINDS}
    execution_order = []
    for repeat in range(args.repeats):
        offset = repeat % len(KINDS)
        sequence = KINDS[offset:]+KINDS[:offset]
        execution_order.append(list(sequence))
        for kind in sequence:
            row, _ = trial(kind)
            runs[kind].append(row)
            print("TRAIN PROFILE %s repeat=%d wall=%.3fs query=%.3fs" %
                  (kind, repeat, row["wall_seconds"], row["query_seconds"]), flush=True)
    medians = {kind: {key: float(np.median([row[key] for row in rows])) for key in rows[0]
                     if "seconds" in key and isinstance(rows[0][key], (int, float))}
               for kind, rows in runs.items()}
    report = {"complete": True, "args": vars(args), "torch": torch.__version__, "cuda": torch.version.cuda,
              "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "checkpoint_sha256": current_pipeline.file_digest(args.checkpoint),
              "raw_recording_sha256": current_pipeline.file_digest(pair[0]),
              "prior_file_sha256": current_pipeline.file_digest(pair[1]), "snapshot_sha256": reference.source_sha256,
              "current_source_sha256": LOADED_SOURCE_SHA256,
              "source_files_changed_during_probe": [name for name in SOURCE_PATHS
                  if current_pipeline.file_digest(ROOT/name) != LOADED_SOURCE_SHA256[name]],
              "actual_queries": len(ids), "batch_size": config["train"]["batch_size"],
              "query_ids_sha256": __import__("hashlib").sha256(ids.tobytes()).hexdigest(),
              "comparisons": comparisons, "execution_order": execution_order, "runs": runs, "medians": medians,
              "timing_scope": "Synchronized complete short training sequence, including cache construction/yield, H2D, forward, prefix BCE, backward, clip and AdamW. Excludes disk/model/optimizer initialization, warmups, correctness capture, serialization and epoch validation.",
              "breakdown_scope": "query_seconds is included in wall/fetch or rollout; forward excludes query. Archived/default query includes relative-geometry construction. Geometry materialize is a subset of cache fetch. Do not sum overlapping fields.",
              "optimizer_scope": "Same fresh AdamW, LR, weight decay, clipping and fixed batch denominator for every sequence. No epoch scheduler step inside this microbenchmark.",
              "interpretation": "Performance probe only; no promise for full epoch throughput, server hardware, convergence or accuracy. Does not change sampled IDs, batch boundaries, order or learned features across optimizer steps.",
              "prior_identity_verified": bool(current_rec.backbone_sha256 and current_rec.backbone_sha256 == checkpoint.get("manifest", {}).get("prior_provenance", {}).get("backbone_sha256"))}
    current_pipeline.save_json(str(out/"profile.json"), report)
    print("BST TRAIN PROFILE FINISHED: " + str(out), flush=True)


if __name__ == "__main__":
    main()
