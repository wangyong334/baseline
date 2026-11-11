"""Export complete BIR arrival trajectories with checked checkpoint provenance.

One .npz stores every original event in one recording. --limit restricts the
number of FULL recordings for a probe, never truncates rows in a training file.
The directory manifest records whether the entire requested split is complete.
An existing directory with missing or conflicting identity is never overwritten.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import train_stream_v1 as T
from dataset.ev_uav_stream import EvUAVStream
from dataset.stream_source import make_window_source
from tools.bst.infer import BackboneArrivalProvider, recording_from_sequence
from tools.bst.pipeline import file_digest, torch_load


def _write_manifest(path, manifest):
    temporary = str(path) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _identity(manifest):
    return (manifest.get("backbone_sha256"), manifest.get("window_ms"),
            manifest.get("recording_n_windows"), manifest.get("split"))


def check_existing_directory(out, desired):
    """Validate identity before creating or modifying export files."""
    path = os.path.join(out, "prior_provenance.json")
    if not os.path.isdir(out) or not os.listdir(out):
        return None
    if not os.path.isfile(path):
        raise ValueError("nonempty prior directory lacks provenance; choose a new directory: " + out)
    with open(path, encoding="utf-8") as source:
        existing = json.load(source)
    if existing.get("schema") != 1 or _identity(existing) != _identity(desired):
        raise ValueError("existing prior directory belongs to a different backbone/window/split")
    return existing


def validate_cached_file(path, checkpoint_hash, window_ms, n_windows, labels=None):
    with np.load(path, allow_pickle=False) as source:
        if "backbone_sha256" not in source or str(source["backbone_sha256"].item()) != checkpoint_hash:
            raise ValueError("cached recording has a different or missing backbone identity: " + str(path))
        if int(source["window_ms"]) != int(window_ms) or int(source["n_windows"]) != int(n_windows):
            raise ValueError("cached recording temporal metadata disagrees with BIR: " + str(path))
        if "complete" not in source or not bool(source["complete"].item()):
            raise ValueError("cached recording is not marked complete")
        trajectory = np.asarray(source["traj"])
        if trajectory.ndim != 2 or trajectory.shape[1] != 1 or not np.isfinite(trajectory).all():
            raise ValueError("cached recording is not a complete finite arrival trajectory")
        if labels is not None:
            if len(trajectory) != len(labels) or not np.array_equal(source["labels"], labels):
                raise ValueError("cached recording rows do not match the raw recording")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="first N complete recordings, probe only")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    if args.limit < 0 or args.threads < 1:
        raise ValueError("limit must be nonnegative and threads positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint_hash = file_digest(args.checkpoint)
    checkpoint = torch_load(args.checkpoint, device)
    cfg = dict(checkpoint["config"])
    cfg["root"], cfg["execution"], cfg["input_device"] = args.data_root, "step", "cpu"
    log_w = math.log(float(checkpoint["stats"]["pos_weight"]))
    q99 = np.asarray(checkpoint["stats"]["q99"], np.float32)
    dataset = EvUAVStream(args.data_root, args.split, cfg)
    names = list(dataset.names)
    selected = names[:args.limit] if args.limit else names
    declared_windows = int(cfg.get("n_windows", 0) or 0)
    manifest = {"schema": 1, "kind": "frozen_bir_arrival_prior", "backbone_sha256": checkpoint_hash,
                "window_ms": int(cfg["window_ms"]),
                "recording_n_windows": declared_windows if declared_windows > 0 else None,
                "split": args.split, "checkpoint": os.path.abspath(args.checkpoint),
                "data_root": os.path.abspath(args.data_root), "complete": False,
                "expected_recordings": names, "completed_recordings": [],
                "probe": bool(args.limit and len(selected) < len(names)), "log_w": log_w}
    existing = check_existing_directory(args.out, manifest)
    if existing is not None:
        if existing.get("expected_recordings") != names:
            raise ValueError("raw split membership changed since this directory was exported")
        completed = set(existing.get("completed_recordings", []))
        if not completed.issubset(set(names)):
            raise ValueError("existing provenance contains unknown recording names")
        # Do not trust a stale complete flag if a file was removed.
        if any(not os.path.isfile(os.path.join(args.out, name)) for name in completed):
            raise ValueError("provenance lists completed files that are missing")
    else:
        completed = set()
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "prior_provenance.json")
    manifest["completed_recordings"] = sorted(completed)
    manifest["complete"] = completed == set(names)
    _write_manifest(manifest_path, manifest)
    # Loading the backbone only when an export is needed avoids replaying a
    # completed same-identity directory, while every skipped file is validated.
    backbone = None
    start = time.perf_counter()
    for ri, name in enumerate(selected):
        seq = EvUAVStream(args.data_root, args.split, cfg, names=[name])[0]
        destination = os.path.join(args.out, name)
        if os.path.isfile(destination):
            validate_cached_file(destination, checkpoint_hash, cfg["window_ms"], seq.n_windows, seq.label)
            print("SKIP same-backbone complete recording %s" % name, flush=True)
        else:
            if backbone is None:
                backbone = T.build_net(cfg, device)
                backbone.load_state_dict(checkpoint["model"])
                if not bool(backbone.gain_calibrated):
                    raise ValueError("BIR checkpoint has no calibrated layer gains")
                backbone.eval()
            rec, original_ids = recording_from_sequence(seq, cfg["window_ms"], log_w)
            source = make_window_source(seq, cfg, q99, device, next(backbone.parameters()).dtype)
            provider = BackboneArrivalProvider(backbone, source, rec, original_ids, len(seq.t), log_w)
            for window, ids in rec.iter_windows(include_empty=True):
                provider(window, ids)
            if not np.isfinite(provider.arrival_scores).all():
                raise RuntimeError("refusing to save incomplete arrival scores")
            temporary = destination + ".tmp.npz"
            np.savez(temporary, traj=provider.arrival_scores[:, None], labels=seq.label,
                     log_w=np.float64(log_w), window_ms=np.int64(cfg["window_ms"]),
                     n_windows=np.int64(seq.n_windows), backbone_sha256=np.asarray(checkpoint_hash),
                     complete=np.asarray(True))
            os.replace(temporary, destination)
            print("EXPORTED %s %d/%d %s: %d events, %d windows, %.2f s" %
                  (args.split, ri + 1, len(selected), name, len(seq.t), seq.n_windows, provider.seconds), flush=True)
        completed.add(name)
        manifest["completed_recordings"] = sorted(completed)
        manifest["complete"] = completed == set(names)
        _write_manifest(manifest_path, manifest)
    print("BST PRIOR EXPORT FINISHED %s: %d/%d recordings, complete=%s, %.2f s" %
          (args.out, len(completed), len(names), manifest["complete"], time.perf_counter() - start), flush=True)


if __name__ == "__main__":
    main()
