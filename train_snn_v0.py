"""Independent dual-FP32 local-rate SNN experiment. See README_SNN_V0.md."""
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
import numpy as np
from configs.configs import cfg
from model.evspsegnet_mp import evspsegnet_mp
from model.evspsegnet_snn_v0 import evspsegnet_snn_v0
from train_mp import (setup, finite, check_state, cpu_state, memory, clear_unused,
                      make_dataset, make_loader, forward_loss, check_gradients,
                      select_sample, linear_epoch_lr)
from utils.stcloss import STCLoss
from utils.eval import evalute
from utils.semantic_cpu import ForegroundMetrics


def emit(record, path=None):
    line = json.dumps(record, allow_nan=False)
    print(line, flush=True)
    if path:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def synchronize():
    for device in (0, 1):
        torch.cuda.synchronize(device)


def diagnostics(net, enabled):
    if hasattr(net, "diagnostics"):
        net.diagnostics(enabled)


def activity(net):
    return net.spike_stats() if hasattr(net, "spike_stats") else {}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(str(temporary), str(path))


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_recovery(net, optimizer, root, epoch, pending, best_loss, best_iou, mean_loss, variant, split):
    atomic_save({"format": "snn_v0_recovery_v1", "model": cpu_state(net),
                 "optimizer": cpu_tree(optimizer.state_dict()), "rng": rng_state(),
                 "epoch": epoch, "validation_pending": pending,
                 "best_loss": best_loss, "best_iou": best_iou, "mean_loss": mean_loss,
                 "variant": variant, "split": split, "config": vars(cfg),
                 "source_sha256": json.loads((root / "run_config.json").read_text(encoding="utf-8"))["source_sha256"]},
                root / "recovery_last.pt")


def validation_preflight(net):
    # Run the full metric path now, without affecting training RNG or BN state.
    before, rng, training = cpu_state(net), rng_state(), net.training
    try:
        result = evaluate(net, "val")
        for name, value in net.state_dict().items():
            if not torch.equal(value.detach().cpu(), before[name]):
                raise RuntimeError("Validation mutated model state: " + name)
        emit({"status": "VALIDATION_PREFLIGHT_OK", "validation": result,
              "note": "Untrained/resumed model diagnostic only; excluded from best-checkpoint selection"})
    finally:
        net.load_state_dict(before, strict=True)
        net.train(training)
        restore_rng(rng)
        clear_unused()


def update(net, criterion, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    synchronize()
    start = time.perf_counter()
    preds, loss = forward_loss(net, criterion, batch)
    loss.backward()
    counts = check_gradients(net)
    if not all(counts.values()):
        raise RuntimeError("Both devices must receive gradients")
    optimizer.step()
    check_state(net)
    synchronize()
    return {"loss": loss.item(), "step_seconds": time.perf_counter() - start,
            "events": batch["seg_label"].numel(),
            "voxels": batch["voxel_ev"].features.shape[0],
            "gradient_tensors": counts, "activity": activity(net)}


def evaluate(net, mode):
    net.eval()
    diagnostics(net, False)
    evaluator = evalute(cfg)
    semantic = ForegroundMetrics(threshold=0.9)
    with torch.no_grad():
        for index, batch in enumerate(make_loader(make_dataset(mode), False)):
            preds, voxel = net(batch["voxel_ev"])
            finite(preds, mode + " predictions")
            mapping = batch["p2v_map"].long().cuda(0)
            event_preds = preds[mapping].reshape(-1).cpu()
            labels = batch["seg_label"].reshape(-1).cpu()
            semantic.update(event_preds.numpy(), labels.numpy())
            if mode == "test" and cfg.roc:
                locs = batch["locs"].float().cpu()
                evaluator.roc_update(locs[:, 3], event_preds.clone(), batch["idx_label"], labels, locs)
            del preds, voxel, mapping, batch
    result = semantic.compute()
    if mode == "test" and cfg.roc:
        pd, fa = evaluator.cal_roc()
        result.update(pd=float(pd), fa=float(fa))
    check_state(net)
    return result


def snapshot(root, variant, mode, split):
    root.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parent
    names = ["train_snn_v0.py", "train_mp.py", "train.py", "model/lif_rate.py",
             "model/evspsegnet_snn_v0.py", "model/evspsegnet_mp.py",
             "model/evspsegnet.py", "model/basemodel.py", "utils/stcloss.py",
             "utils/eval.py", "dataset/ev_uav.py", "dataset/basedataset.py",
             "configs/configs.py", "utils/semantic_cpu.py"]
    hashes = {}
    for name in names:
        src = source_root / name
        dst = root / "source" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        hashes[name] = hashlib.sha256(src.read_bytes()).hexdigest()
    shutil.copy2(cfg.config, str(root / "config.yaml"))
    record = {"config": vars(cfg), "variant": variant, "mode": mode, "split": split,
              "output_root": str(root), "sample_override": os.environ.get("EVUAV_SAMPLE"),
              "precision": "FP32", "torch": torch.__version__,
              "gpus": [torch.cuda.get_device_name(d) for d in (0, 1)],
              "source_sha256": hashes,
              "protocol": "Local constant-current LIF rate code; no physical-time streaming; no energy claim"}
    (root / "run_config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main():
    if torch.cuda.device_count() < 2 or cfg.batch_size != 1:
        raise RuntimeError("Requires TWO CUDA GPUs and batch_size=1")
    torch.cuda.set_device(0)
    mode = os.environ.get("EVUAV_MODE", "smoke")
    variant = os.environ.get("EVUAV_VARIANT", "snn")
    if mode not in ("smoke", "overfit", "train", "test") or variant not in ("ann", "snn"):
        raise ValueError("mode=smoke/overfit/train/test; variant=ann/snn")
    split = int(os.environ.get("EVUAV_MP_SPLIT", "2"))
    if cfg.diagnostic_interval < 1:
        raise ValueError("diagnostic_interval must be positive")
    if not 0 <= cfg.validation_start < cfg.epochs:
        raise ValueError("validation_start must be within training epochs")
    linear_epoch_lr(0, cfg.epochs, cfg.lr, cfg.lr_end)
    if cfg.lr_schedule != "linear":
        raise ValueError("v0 uses linear LR only")
    setup(cfg.seed)
    resume_path = os.environ.get("EVUAV_RESUME")
    if resume_path and mode != "train":
        raise ValueError("EVUAV_RESUME is only supported for train")
    root = None
    if mode in ("train", "overfit"):
        default = cfg.model_save_root if mode == "train" else cfg.model_save_root + "_overfit"
        if variant == "ann" and not os.environ.get("EVUAV_RUN_DIR"):
            raise ValueError("ANN control requires its own EVUAV_RUN_DIR")
        root = Path(os.environ.get("EVUAV_RUN_DIR", default))
        snapshot(root, variant, mode, split)
    net = (evspsegnet_snn_v0(cfg, split) if variant == "snn" else evspsegnet_mp(cfg, split))
    emit({"mode": mode, "variant": variant, "seed": cfg.seed,
          "spike_sites": getattr(net, "spike_sites", []), "steps": cfg.snn_steps})
    if mode == "test":
        checkpoint = Path(os.environ.get("EVUAV_CHECKPOINT", cfg.model_path))
        metadata = checkpoint.parent / "run_config.json"
        if not metadata.exists():
            raise RuntimeError("Checkpoint run_config.json required to verify architecture")
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved["variant"] != variant:
            raise RuntimeError("Checkpoint variant differs from requested variant")
        for key in ("width", "input_channel", "snn_stages", "snn_steps", "snn_beta", "snn_threshold"):
            if saved["config"][key] != getattr(cfg, key):
                raise RuntimeError("Checkpoint config differs: " + key)
        net.load_state_dict(torch.load(str(checkpoint), map_location="cpu"), strict=True)
        check_state(net)
        emit({"checkpoint": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "test": evaluate(net, "test")})
        return
    criterion = STCLoss(k=cfg.k, t=cfg.t, cfg=cfg).cuda(0)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    dataset = make_dataset("train")
    if mode in ("smoke", "overfit"):
        sample = select_sample(dataset, largest=(mode == "smoke"))
        batch = dataset.custom_collate([sample])
        steps = 2 if mode == "smoke" else int(os.environ.get("EVUAV_OVERFIT_STEPS", "100"))
        if steps < 1:
            raise ValueError("overfit steps must be positive")
        net.train()
        memory(reset=True)
        for step in range(steps):
            diagnostics(net, True)
            before = cpu_state(net) if mode == "smoke" else None
            row = update(net, criterion, optimizer, batch)
            if before is not None:
                changed = {0: 0, 1: 0}
                for name, param in net.named_parameters():
                    if not torch.equal(before[name], param.detach().cpu()):
                        changed[param.device.index] += 1
                if not all(changed.values()):
                    raise RuntimeError("One device has no parameter update")
                row["updated_tensors"] = changed
            row.update(step=step, memory=memory())
            emit(row, root / "metrics.jsonl" if root else None)
        if root:
            torch.save(cpu_state(net), root / ("last_seed%d.pt" % cfg.seed))
        emit({"status": mode + " completed", "note": "overfit loss must be inspected; completion alone is not overfitting success"})
        return
    loader = make_loader(dataset, True)
    best_iou, best_loss = -float("inf"), float("inf")
    start_epoch, pending, resumed_mean = 0, False, None
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        if checkpoint.get("format") != "snn_v0_recovery_v1":
            raise ValueError("Resume requires recovery_last.pt, not weights-only .pt")
        if checkpoint["variant"] != variant or checkpoint["split"] != split:
            raise ValueError("Resume variant/split mismatch")
        for key, value in vars(cfg).items():
            if key not in ("config", "model_save_root", "model_path") and checkpoint["config"].get(key) != value:
                raise ValueError("Resume config mismatch: " + key)
        current_hashes = json.loads((root / "run_config.json").read_text(encoding="utf-8"))["source_sha256"]
        if current_hashes != checkpoint["source_sha256"]:
            raise ValueError("Resume source mismatch; do not silently change an experiment")
        if len(checkpoint["rng"]["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume must expose the same number of CUDA devices")
        net.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_iou, best_loss = checkpoint["best_iou"], checkpoint["best_loss"]
        pending, resumed_mean = checkpoint["validation_pending"], checkpoint["mean_loss"]
        start_epoch = checkpoint["epoch"] + (0 if pending else 1)
        if start_epoch >= cfg.epochs:
            raise ValueError("Checkpoint already finished all requested epochs")
        for name in ("best_loss_seed%d.pt" % cfg.seed, "best_iou_seed%d.pt" % cfg.seed):
            previous = Path(resume_path).parent / name
            if previous.exists():
                shutil.copy2(str(previous), str(root / name))
        restore_rng(checkpoint["rng"])
        del checkpoint
        emit({"resume": resume_path, "epoch": start_epoch, "validation_pending": pending})
    validation_preflight(net)
    for epoch in range(start_epoch, cfg.epochs):
        lr = linear_epoch_lr(epoch, cfg.epochs, cfg.lr, cfg.lr_end)
        for group in optimizer.param_groups:
            group["lr"] = lr
        net.train()
        memory(reset=True)
        total = 0.0
        start = time.perf_counter()
        batches = () if pending else loader
        for step, batch in enumerate(batches):
            diagnostics(net, step % cfg.diagnostic_interval == 0)
            row = update(net, criterion, optimizer, batch)
            total += row["loss"]
            if step % cfg.diagnostic_interval == 0:
                row.update(epoch=epoch, step=step)
                emit(row, root / "activity.jsonl")
            del batch
        optimizer.zero_grad(set_to_none=True)
        mean_loss = resumed_mean if pending else total / len(loader)
        if mean_loss < best_loss:
            best_loss = mean_loss
            atomic_save(cpu_state(net), root / ("best_loss_seed%d.pt" % cfg.seed))
        # Commit the completed training epoch BEFORE validation can fail.
        atomic_save(cpu_state(net), root / ("last_seed%d.pt" % cfg.seed))
        save_recovery(net, optimizer, root, epoch, True, best_loss, best_iou, mean_loss, variant, split)
        validation = evaluate(net, "val") if epoch >= cfg.validation_start else None
        if validation and validation["iou"] > best_iou:
            best_iou = validation["iou"]
            atomic_save(cpu_state(net), root / ("best_iou_seed%d.pt" % cfg.seed))
        save_recovery(net, optimizer, root, epoch, False, best_loss, best_iou, mean_loss, variant, split)
        pending = False
        emit({"epoch": epoch, "lr": lr, "mean_batch_loss": mean_loss,
              "validation": validation, "epoch_seconds": time.perf_counter() - start,
              "memory": memory()}, root / "metrics.jsonl")
        clear_unused()
    print("TRAINING FINISHED:", root, flush=True)


if __name__ == "__main__":
    main()
