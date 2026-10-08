"""Train a SPEED system: gain calibration, then per epoch one update per training recording (shuffled), validation
on the zero-wait readout (net) and checkpoint selection by validation IoU.

    python tools/train.py --config configs/base_evuav.yaml --root /path/to/EV-UAV-dataset --out runs/base_s37 \
        [--seed 37] [--set training.epochs=2] [--device cuda:1] [--max-train 4 --max-val 2 --max-steps 32]
Writes run_config.json, calibration.json, metrics.jsonl, best.pt (best validation IoU), last.pt.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")  # limits fragmentation on large canvases

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from speed.core.build import build_loss, build_system, load_config, with_overrides  # noqa: E402
from speed.core.runtime import memory_line, peak_memory_gib, seed_everything, write_json  # noqa: E402
from speed.core.training import (LayerMonitor, calibrate, linear_epoch_lr, save_checkpoint, select_subset,  # noqa: E402
                                 tau_statistics, train_stream)
from speed.data.dataset_card import list_recordings, load_card  # noqa: E402
from speed.data.readers import read_recording  # noqa: E402
from speed.eval.metrics import BenchmarkMetrics  # noqa: E402
from speed.slots.publish.readouts import NetReadout  # noqa: E402


class Recordings(torch.utils.data.Dataset):
    def __init__(self, card, items):
        self.card, self.items = card, list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, path = self.items[i]
        return read_recording(self.card, path, name)


def first_item(batch):
    return batch[0]


def loader(dataset, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=int(num_workers),
                                       collate_fn=first_item, generator=generator)


def validate(system, dataset, card, ev, carry, monitor):
    metrics = BenchmarkMetrics(card["sensor"]["width"], card["sensor"]["height"], int(ev["frame_ms"] * 1000),
                               ev["threshold"], ev["correct_thresh"])
    readout = NetReadout()
    was_training = system.network.training
    system.network.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            stream = dataset[i]
            results, _ = system.run_stream(stream, carry, monitor, readouts=[readout])
            metrics.update(stream, results["net"][0])
    system.network.train(was_training)
    r = metrics.result()
    return {k: r[k] for k in ("iou", "acc", "pd", "fa")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[], help="config overrides section.key=value")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train", type=int, default=0, help="smoke runs: first N training recordings")
    parser.add_argument("--max-val", type=int, default=0, help="smoke runs: first N validation recordings")
    parser.add_argument("--max-steps", type=int, default=None, help="smoke runs: steps per training recording")
    args = parser.parse_args()

    cfg = with_overrides(load_config(args.config), args.set)
    if args.seed is not None:
        cfg["training"]["seed"] = int(args.seed)
    tr, ev = cfg["training"], cfg["evaluation"]
    seed = int(tr["seed"])
    device = torch.device(args.device)
    if os.path.exists(os.path.join(args.out, "metrics.jsonl")):
        raise RuntimeError("output directory already has a run: %s" % args.out)
    os.makedirs(args.out, exist_ok=True)
    seed_everything(seed, bool(tr.get("deterministic", True)))
    system = build_system(cfg)
    system.to(device)
    card = load_card(cfg["dataset"])
    train_items = list_recordings(card, args.root, "train")
    val_items = list_recordings(card, args.root, "val")
    if args.max_train:
        train_items = train_items[:args.max_train]
    if args.max_val:
        val_items = val_items[:args.max_val]
    write_json(os.path.join(args.out, "run_config.json"), {
        "config": cfg, "args": vars(args), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "features": system.representation.feature_names(),
        "parameters": sum(p.numel() for p in system.network.parameters())})
    optimizer = torch.optim.Adam(system.network.parameters(), lr=float(tr["lr"]))
    calib_names = set(select_subset([n for n, _ in train_items], int(tr["calibration"]["sequences"]), seed + 1000))
    calib_streams = [read_recording(card, p, n) for n, p in train_items if n in calib_names]
    carry = bool(tr.get("carry", True))
    reports = calibrate(system, calib_streams, tr["calibration"], seed, carry)
    write_json(os.path.join(args.out, "calibration.json"), {"recordings": sorted(calib_names), "layers": reports})
    del calib_streams
    print(memory_line(device, "after calibration"), flush=True)
    loss_fn = build_loss(cfg["loss"])
    train_set, val_set = Recordings(card, train_items), Recordings(card, val_items)
    epochs, best = int(tr["epochs"]), -float("inf")
    for epoch in range(epochs):
        lr = linear_epoch_lr(epoch, epochs, float(tr["lr"]), float(tr["lr_end"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        rng = np.random.RandomState(seed * 1000 + epoch + 7)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        system.network.train()
        sums, events = {"loss_sum": 0.0, "mark_sum": 0.0, "intensity_sum": 0.0}, 0
        for stream in loader(train_set, seed * 1000 + epoch, tr.get("num_workers", 2)):
            out = train_stream(system, loss_fn, stream, optimizer, tr["tbptt_steps"], tr["grad_clip"], rng,
                               args.max_steps, carry, int(tr.get("checkpoint_steps", 0)), tr.get("update_steps"))
            for key in sums:
                sums[key] += out[key]
            events += out["events"]
        train_seconds = time.perf_counter() - t0
        print(memory_line(device, "after training"), flush=True)
        monitor = LayerMonitor(system.network.v_threshold)
        val = validate(system, val_set, card, ev, carry, monitor)
        print(memory_line(device, "after validation"), flush=True)
        record = {"epoch": epoch, "seed": seed, "lr": lr, "loss_per_event": sums["loss_sum"] / max(events, 1),
                  "mark_per_event": sums["mark_sum"] / max(events, 1),
                  "intensity_per_event": sums["intensity_sum"] / max(events, 1),
                  "val": val, "train_seconds": train_seconds, "epoch_seconds": time.perf_counter() - t0,
                  "peak_memory_gib": peak_memory_gib(device), "layers": monitor.summary(),
                  "tau": tau_statistics(system.network)}
        with open(os.path.join(args.out, "metrics.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if val["iou"] > best:
            best = val["iou"]
            save_checkpoint(os.path.join(args.out, "best.pt"), system.network, optimizer, epoch, best, cfg)
        save_checkpoint(os.path.join(args.out, "last.pt"), system.network, optimizer, epoch, best, cfg)
        print("epoch %d lr %.2e loss/event %.5f (mark %.5f, intensity %.5f) | val IoU %.4f ACC %.4f Pd %.4f Fa %.2e"
              " | %.0f s" % (epoch, lr, record["loss_per_event"], record["mark_per_event"],
                             record["intensity_per_event"], val["iou"], val["acc"], val["pd"], val["fa"],
                             record["epoch_seconds"]), flush=True)
    print("TRAINING FINISHED: best val IoU %.4f -> %s" % (best, args.out), flush=True)


if __name__ == "__main__":
    main()
