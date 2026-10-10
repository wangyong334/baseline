"""Train a SPEED system: gain calibration, then per epoch one update per training recording (shuffled), validation
and checkpoint selection by the validation IoU of evaluation.select_readout (default net, the zero-wait readout;
V4-2: pub) at the benchmark threshold, or by its mean IoU over thresholds 0.5-0.9 (select_metric mean_iou).
--init CKPT loads a trained SPEED checkpoint into the matching modules (e.g. the base network into a V4-3 network);
training.freeze lists module prefixes whose parameters stay fixed (V4-3: the trunk and the base heads). With --init
the gain calibration is skipped (the gains come with the checkpoint).

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
from speed.core.training import (LayerMonitor, calibrate, linear_epoch_lr, load_checkpoint, save_checkpoint,  # noqa: E402
                                 select_subset,
                                 tau_statistics, train_stream)
from speed.data.dataset_card import list_recordings, load_card  # noqa: E402
from speed.data.readers import read_recording  # noqa: E402
from speed.eval.metrics import BenchmarkMetrics  # noqa: E402
from speed.slots.publish.readouts import NetReadout  # noqa: E402

SELECT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


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


def readout_producing(system, name):
    for r in system.readouts:
        names = [r.name] if hasattr(r, "name") else ["%s%d" % (r.prefix, d) for d in r.delays]
        if name in names:
            return r
    raise ValueError("no readout produces %s (available %s)" % (name, system.readout_names()))


def validate(system, dataset, card, ev, carry, monitor):
    """-> (net metrics at the benchmark threshold, selection metrics of evaluation.select_readout)."""
    select, metric = ev.get("select_readout", "net"), ev.get("select_metric", "iou")
    if metric not in ("iou", "mean_iou"):
        raise ValueError("select_metric must be iou or mean_iou")
    make = lambda: BenchmarkMetrics(card["sensor"]["width"], card["sensor"]["height"],  # noqa: E731
                                    int(ev["frame_ms"] * 1000), ev["threshold"], ev["correct_thresh"])
    metrics = {"net": make()}
    readouts = [NetReadout()]
    if select != "net":
        metrics[select] = make()
        readouts.append(readout_producing(system, select))
    counts = {t: [0, 0, 0] for t in SELECT_THRESHOLDS}                    # tp, fp, fn of the selected readout
    was_training = system.network.training
    system.network.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            stream = dataset[i]
            results, _ = system.run_stream(stream, carry, monitor, readouts=readouts)
            for name, m in metrics.items():
                m.update(stream, results[name][0])
            if metric == "mean_iou":
                prob, target = results[select][0], stream.label == 1
                for t, c in counts.items():
                    pred = prob >= np.float32(t)
                    c[0] += int(np.count_nonzero(pred & target))
                    c[1] += int(np.count_nonzero(pred & ~target))
                    c[2] += int(np.count_nonzero(~pred & target))
    system.network.train(was_training)
    out = {name: {k: r[k] for k in ("iou", "acc", "pd", "fa")} for name, r in
           ((name, m.result()) for name, m in metrics.items())}
    sel = dict(out[select], readout=select, metric=metric)
    if metric == "mean_iou":
        ious = [c[0] / float(sum(c)) if sum(c) else 0.0 for c in counts.values()]
        sel["mean_iou"] = float(np.mean(ious))
    sel["value"] = sel["mean_iou"] if metric == "mean_iou" else sel["iou"]
    return out["net"], sel


def learned_parameter_summary(network):
    """Monitoring of the V4-3 learned constants (fusion and position weights, correlation gains)."""
    out = {}
    head, motion = getattr(network, "readout_head", None), getattr(network, "motion", None)
    with torch.no_grad():
        if head is not None:
            out["fusion"] = [float(v) for v in head.fusion.cpu()]
            out["position_weights"] = [round(float(v), 4) for v in torch.exp(head.position_log_weights()).cpu()]
        if motion is not None:
            out["correlation_gain"] = [float(v) for v in motion.gain.cpu()]
    return out


def load_initial_weights(network, path, device):
    """Loads a SPEED checkpoint into the matching part of the network (a base checkpoint's heads.net.* -> the V4-3
    heads.base.net.*). -> names of the loaded parameters and buffers."""
    state = load_checkpoint(path, device)["network"]
    own = network.state_dict()
    mapped = {}
    for key, value in state.items():
        target = key
        if key not in own and key.startswith("heads.net.") and "heads.base." + key[len("heads."):] in own:
            target = "heads.base." + key[len("heads."):]
        if target in own:
            if tuple(own[target].shape) != tuple(value.shape):
                raise ValueError("shape mismatch for %s: %s vs %s" % (target, tuple(own[target].shape), tuple(value.shape)))
            mapped[target] = value
    network.load_state_dict(mapped, strict=False)
    return sorted(mapped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[], help="config overrides section.key=value")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--max-train", type=int, default=0, help="smoke runs: first N training recordings")
    parser.add_argument("--max-val", type=int, default=0, help="smoke runs: first N validation recordings")
    parser.add_argument("--max-steps", type=int, default=None, help="smoke runs: steps per training recording")
    parser.add_argument("--init", default=None, help="SPEED checkpoint to start from (matching modules only)")
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
    train_items = list_recordings(card, args.root, args.train_split)
    val_items = list_recordings(card, args.root, args.val_split)
    if args.max_train:
        train_items = train_items[:args.max_train]
    if args.max_val:
        val_items = val_items[:args.max_val]
    write_json(os.path.join(args.out, "run_config.json"), {
        "config": cfg, "args": vars(args), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "features": system.representation.feature_names(),
        "parameters": sum(p.numel() for p in system.network.parameters())})
    carry = bool(tr.get("carry", True))
    if args.init:
        loaded = load_initial_weights(system.network, args.init, device)
        if not bool(system.network.backbone.gain_calibrated):
            raise RuntimeError("the initial checkpoint has no calibrated gains")
        write_json(os.path.join(args.out, "calibration.json"), {"init": os.path.abspath(args.init), "loaded": loaded})
        print("initialised from %s (%d tensors)" % (args.init, len(loaded)), flush=True)
    frozen = tuple(tr.get("freeze") or ())
    for name, p in system.network.named_parameters():
        if name.startswith(frozen):
            p.requires_grad_(False)
    trainable = [p for p in system.network.parameters() if p.requires_grad]
    print("trainable parameters: %d of %d" % (sum(p.numel() for p in trainable),
                                              sum(p.numel() for p in system.network.parameters())), flush=True)
    optimizer = torch.optim.Adam(trainable, lr=float(tr["lr"]))
    if not args.init:
        calib_names = set(select_subset([n for n, _ in train_items], int(tr["calibration"]["sequences"]), seed + 1000))
        calib_streams = [read_recording(card, p, n) for n, p in train_items if n in calib_names]
        reports = calibrate(system, calib_streams, tr["calibration"], seed, carry)
        write_json(os.path.join(args.out, "calibration.json"), {"recordings": sorted(calib_names), "layers": reports})
        del calib_streams
    print(memory_line(device, "after calibration"), flush=True)
    loss_fn = build_loss(cfg["loss"], float(cfg["clock"]["step_ms"]), system)
    scaler = torch.cuda.amp.GradScaler() if bool(tr.get("amp", False)) and device.type == "cuda" else None
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
                               args.max_steps, carry, int(tr.get("checkpoint_steps", 0)), tr.get("update_steps"),
                               scaler)
            for key, value in out.items():
                if key.endswith("_sum"):
                    sums[key] = sums.get(key, 0.0) + value
            events += out["events"]
        train_seconds = time.perf_counter() - t0
        print(memory_line(device, "after training"), flush=True)
        monitor = LayerMonitor(system.network.v_threshold)
        val, sel = validate(system, val_set, card, ev, carry, monitor)
        print(memory_line(device, "after validation"), flush=True)
        record = {"epoch": epoch, "seed": seed, "lr": lr, "loss_per_event": sums["loss_sum"] / max(events, 1)}
        for key in sorted(sums):
            if key.endswith("_sum") and key != "loss_sum":
                record[key[:-4] + "_per_event"] = sums[key] / max(events, 1)
        record.update({"val": val, "val_select": sel, "train_seconds": train_seconds,
                       "epoch_seconds": time.perf_counter() - t0, "peak_memory_gib": peak_memory_gib(device),
                       "layers": monitor.summary(), "tau": tau_statistics(system.network),
                       "learned": learned_parameter_summary(system.network)})
        with open(os.path.join(args.out, "metrics.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if sel["value"] > best:
            best = sel["value"]
            save_checkpoint(os.path.join(args.out, "best.pt"), system.network, optimizer, epoch, best, cfg)
        save_checkpoint(os.path.join(args.out, "last.pt"), system.network, optimizer, epoch, best, cfg)
        terms = ", ".join("%s %.5f" % (key[:-len("_per_event")], record[key]) for key in sorted(record)
                          if key.endswith("_per_event") and key != "loss_per_event")
        chosen = "" if sel["readout"] == "net" and sel["metric"] == "iou" else \
            " | select %s %s %.4f (IoU %.4f Fa %.2e)" % (sel["readout"], sel["metric"], sel["value"], sel["iou"],
                                                        sel["fa"])
        print("epoch %d lr %.2e loss/event %.5f (%s) | val IoU %.4f ACC %.4f Pd %.4f Fa %.2e%s | %.0f s" % (
            epoch, lr, record["loss_per_event"], terms, val["iou"], val["acc"], val["pd"], val["fa"], chosen,
            record["epoch_seconds"]), flush=True)
    print("TRAINING FINISHED: best val %s %s %.4f -> %s" % (ev.get("select_readout", "net"),
                                                            ev.get("select_metric", "iou"), best, args.out), flush=True)


if __name__ == "__main__":
    main()
