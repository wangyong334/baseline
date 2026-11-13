"""Fail early on the real runner's CUDA/PyTorch before calibration/training.

Exercises nonzero fusion gradients and two optimizer steps across a TBPTT
boundary. It checks operator compatibility, not performance or final accuracy.
"""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from dataset.overlap_windows import OverlapSequence, OverlapWindowSource
from tools.v16 import train as tr
from utils.comm_budget import CommBudget


def check(config, device):
    cfg = dict(config)
    cfg.update(height=24, width_px=30, pad_height=24, pad_width=32,
               total_ms=250, n_windows=4, streams=2, tbptt_k=2,
               timing_warmup_windows=0, comm_budget_messages=20.)
    tr.validate_config(cfg)
    tr.seed_everything(int(cfg["seed"]), True)
    net, head = tr.make_models(cfg, device)
    records = []
    for seed in (37, 38):
        rng = np.random.RandomState(seed)
        n = 240
        t = np.concatenate([np.arange(0, 250, 5), rng.randint(0, 250, n-50)])
        records.append(OverlapSequence("preflight%d" % seed, rng.randint(0, 30, n),
                       rng.randint(0, 24, n), t, np.arange(n) % 2,
                       (np.arange(n) % 3 == 0).astype(np.float32), np.zeros(n), total_ms=250))
    q99 = np.full(12, np.log1p(2), dtype=np.float32)
    def factory():
        source = OverlapWindowSource(records[0], cfg, q99, device)
        def windows():
            for k in range(4):
                x, ev, _, _ = source.window(k)
                yield x, ev
        yield windows()
    tr.calibrate_gains(net, factory, .99, 32, 1, .05, 20., 4037)
    # Zero initialization would hide upstream head failures on step one.
    with torch.no_grad():
        if cfg.get("revision_architecture") == "evidence":
            if head.has_revision:
                head.output.weight.normal_(0., .01)
            if head.has_context:
                head.reliability[-1].weight.normal_(0., .01)
        else:
            head.revision[-1].weight.normal_(0., .01)
    optimizer = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=.001)
    result = tr.train_epoch(net, head, records, optimizer, cfg, q99, 1., device,
                            np.random.RandomState(37), CommBudget.from_config(cfg, net),
                            plan=[(0, [(0, 4)]), (0, [(1, 4)])])
    if result["updates"] != 2 or result["committed_events"] != 480:
        raise RuntimeError("Preflight did not execute both updates and commit every event")
    gradient_groups = dict(result["grad_groups_mean"])
    if cfg["revision_variant"] == "second":
        gradient_groups.pop("fusion", None)
    if not all(v > 0 for v in gradient_groups.values()):
        raise RuntimeError("Preflight has a dead backbone/readout/revision gradient")
    net.eval(); head.eval()
    first, _, _ = tr.infer_record(net, head, records[0], cfg, q99, device, modes=["main"], step=True)
    second, _, _ = tr.infer_record(net, head, records[0], cfg, q99, device, modes=["main"], step=True)
    if not np.array_equal(first["main"], second["main"]):
        raise RuntimeError("Online deterministic repeat mismatch")
    return {"passed": True, "torch": torch.__version__, "python": sys.version,
            "device": str(device), "variant": cfg["revision_variant"],
            "architecture": cfg.get("revision_architecture", "shared_feature_r1"),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "strict_deterministic": torch.are_deterministic_algorithms_enabled(),
            "checks": "nonzero revision; two Adam/TBPTT updates; original-ID coverage; repeatable online output",
            "training": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = check(tr.load_flat_config(args.config), torch.device(args.device))
    tr.write_json(Path(args.out), result)
    print("V16 RUNTIME PREFLIGHT PASSED " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
