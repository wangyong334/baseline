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


def activate_head_for_preflight(head, architecture):
    """Exercise upstream gradients hidden by zero-initialized output weights."""
    with torch.no_grad():
        if architecture == "hotelling":
            # Analytic observer with positive calibration scale is already live.
            # There is no zero-initialized MLP output to perturb.
            return
        if architecture in ("evidence", "anchored", "covariance"):
            if head.has_revision:
                head.output.weight.normal_(0., .01)
            # The anchored head has one correction path, not r2's reliability
            # blend. Do not require or silently synthesize a second module.
            if architecture == "evidence" and head.has_context:
                head.reliability[-1].weight.normal_(0., .01)
        else:
            head.revision[-1].weight.normal_(0., .01)


def fit_preflight_statistics(head, device):
    """Exercise train-only covariance fitting before its first forward pass.

    These labelled synthetic pairs belong only to this compatibility check.
    They are never serialized into the formal training checkpoint.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(4037)
    n, d = 256, int(head.feature_dim)
    labels = (torch.arange(n) % 2).float()
    first_h = torch.randn(n, d, generator=generator)
    first_h[:, 0] += labels
    second_h = .7 * first_h + .7 * torch.randn(n, d, generator=generator)
    second_h[:, 1] += labels
    first_z = first_h[:, 0] + .3 * torch.randn(n, generator=generator)
    second_z = second_h[:, 1] + .3 * torch.randn(n, generator=generator)
    observations = [value.to(device) for value in
                    (first_z, second_z, first_h, second_h, labels)]
    report = head.fit_statistics(*observations)
    if not report.get("fitted"):
        raise RuntimeError("Covariance preflight failed to fit synthetic training statistics")
    return report


def check(config, device):
    cfg = dict(config)
    stride = int(cfg["stride_ms"])
    total_ms = 5 * stride
    cfg.update(height=24, width_px=30, pad_height=24, pad_width=32,
               total_ms=total_ms, n_windows=4, streams=2, tbptt_k=2,
               timing_warmup_windows=0, comm_budget_messages=20.)
    tr.validate_config(cfg)
    tr.seed_everything(int(cfg["seed"]), True)
    net, head = tr.make_models(cfg, device)
    statistics = None
    if (cfg.get("revision_architecture") == "covariance" or
            (cfg.get("revision_architecture") == "hotelling" and head.has_revision)):
        statistics = fit_preflight_statistics(head, device)
    records = []
    for seed in (37, 38):
        rng = np.random.RandomState(seed)
        n = 240
        t = np.concatenate([np.arange(50) * (total_ms / 50.), rng.uniform(0, total_ms, n-50)])
        records.append(OverlapSequence("preflight%d" % seed, rng.randint(0, 30, n),
                       rng.randint(0, 24, n), t, np.arange(n) % 2,
                       (np.arange(n) % 3 == 0).astype(np.float32), np.zeros(n), total_ms=total_ms,
                       window_ms=cfg["window_ms"], stride_ms=stride))
    q99 = np.full(12, np.log1p(2), dtype=np.float32)
    def factory():
        source = OverlapWindowSource(records[0], cfg, q99, device)
        def windows():
            for k in range(4):
                x, ev, _, _ = source.window(k)
                yield x, ev
        yield windows()
    tr.calibrate_gains(net, factory, .99, 32, 1, .05, 20., 4037)
    activate_head_for_preflight(head, cfg.get("revision_architecture", "dense"))
    optimizer = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=.001)
    frozen_statistics = None
    if cfg.get("revision_architecture") == "hotelling":
        from tools.v16.train_hotelling import statistics_sha256
        frozen_statistics = statistics_sha256(head)
    result = tr.train_epoch(net, head, records, optimizer, cfg, q99, 1., device,
                            np.random.RandomState(37), CommBudget.from_config(cfg, net),
                            plan=[(0, [(0, 4)]), (0, [(1, 4)])])
    if result["updates"] != 2 or result["committed_events"] != 480:
        raise RuntimeError("Preflight did not execute both updates and commit every event")
    gradient_groups = dict(result["grad_groups_mean"])
    if not any(parameter.requires_grad for parameter in head.parameters()):
        gradient_groups.pop("fusion", None)
    if not all(v > 0 for v in gradient_groups.values()):
        raise RuntimeError("Preflight has a dead backbone/readout/revision gradient")
    net.eval(); head.eval()
    first, _, _ = tr.infer_record(net, head, records[0], cfg, q99, device, modes=["main"], step=True)
    second, _, _ = tr.infer_record(net, head, records[0], cfg, q99, device, modes=["main"], step=True)
    if not np.array_equal(first["main"], second["main"]):
        raise RuntimeError("Online deterministic repeat mismatch")
    if frozen_statistics is not None and statistics_sha256(head) != frozen_statistics:
        raise RuntimeError("Hotelling statistics mutated during optimization or online inference")
    return {"passed": True, "torch": torch.__version__, "python": sys.version,
            "device": str(device), "variant": cfg["revision_variant"],
            "architecture": cfg.get("revision_architecture", "shared_feature_r1"),
            "window_ms": cfg["window_ms"], "stride_ms": stride,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "strict_deterministic": torch.are_deterministic_algorithms_enabled(),
            "checks": "nonzero revision; two Adam/TBPTT updates; original-ID coverage; repeatable online output",
            "synthetic_training_statistics": statistics,
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
