"""Independent FP32 model-parallel entry. See README_MODEL_PARALLEL.md."""
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import tqdm
import spconv.pytorch as spconv

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from model.evspsegnet_mp import evspsegnet_mp
from train import setup
from utils.stcloss import STCLoss
from utils.eval import evalute


def finite(tensor, name):
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError("Nonfinite value: " + name)


def check_state(net):
    for name, tensor in net.state_dict().items():
        if tensor.is_floating_point():
            finite(tensor, name)
            if name.endswith("running_var") and (tensor < 0).any().item():
                raise RuntimeError("Negative BN variance: " + name)


def cpu_state(net):
    return {name: value.detach().cpu().clone()
            for name, value in net.state_dict().items()}


def memory(reset=False):
    result = {}
    for device in (0, 1):
        torch.cuda.synchronize(device)
        result[str(device)] = {
            "peak_allocated_GiB": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
            "peak_reserved_GiB": round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
        }
        if reset:
            torch.cuda.reset_peak_memory_stats(device)
    return result


def clear_unused():
    gc.collect()
    for device in (0, 1):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def make_dataset(mode):
    dataset = EvUAV(cfg, mode=mode)
    dataset.file_list = sorted(name for name in dataset.file_list if name.endswith(".npz"))
    if not dataset.file_list:
        raise RuntimeError("No NPZ samples in " + dataset.root)
    return dataset


def make_loader(dataset, shuffle):
    # custom_collate invokes CUDA; NEVER use multiprocess DataLoader workers here.
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=0, collate_fn=dataset.custom_collate,
    )


def forward_loss(net, criterion, batch):
    preds, voxel = net(batch["voxel_ev"])
    finite(preds, "predictions")
    check_state(net)
    label = batch["seg_label"].float().to("cuda:0")
    mapping = batch["p2v_map"].long().to("cuda:0")
    loss = criterion(voxel, mapping, preds, label)
    finite(loss, "loss")
    return preds, loss


def check_gradients(net):
    counts = {0: 0, 1: 0}
    for name, param in net.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                raise RuntimeError("Missing gradient: " + name)
            finite(param.grad, "gradient " + name)
            counts[param.device.index] += 1
    return counts


def select_sample(dataset, largest):
    name = os.environ.get("EVUAV_SAMPLE")
    if name is None and largest:
        records = []
        for candidate in dataset.file_list:
            with np.load(os.path.join(dataset.root, candidate)) as data:
                records.append((data["ev_loc"].shape[0], candidate))
        name = max(records)[1]
    if name is None:
        name = "train_000.npz"
    index = dataset.file_list.index(name)
    print("Sample:", name, flush=True)
    return dataset[index]


def tensor_metrics(reference, candidate):
    if reference.shape != candidate.shape:
        raise RuntimeError("Tensor shapes differ: %s vs %s" %
                           (tuple(reference.shape), tuple(candidate.shape)))
    if not reference.is_floating_point():
        return {"exact": torch.equal(reference, candidate)}
    if not torch.isfinite(reference).all().item() or not torch.isfinite(candidate).all().item():
        raise RuntimeError("Nonfinite tensor encountered during comparison")
    a = reference.reshape(-1).double()
    b = candidate.reshape(-1).double()
    diff = b - a
    a_norm = torch.norm(a).item()
    b_norm = torch.norm(b).item()
    diff_norm = torch.norm(diff).item()
    if a_norm == 0.0 and b_norm == 0.0:
        cosine = 1.0
    elif a_norm == 0.0 or b_norm == 0.0:
        cosine = 0.0
    else:
        cosine = torch.dot(a, b).item() / (a_norm * b_norm)
    return {
        "ref_max_abs": a.abs().max().item() if a.numel() else 0.0,
        "max_abs_error": diff.abs().max().item() if diff.numel() else 0.0,
        "relative_l2": diff_norm / max(a_norm, 1e-30),
        "cosine": max(-1.0, min(1.0, cosine)),
    }


def report_group(title, left, right, top=8):
    if torch.is_tensor(left):
        left, right = {title: left}, {title: right}
    if left.keys() != right.keys():
        raise RuntimeError("State/gradient keys differ in " + title)
    rows = []
    integer_mismatches = []
    for name in left:
        metrics = tensor_metrics(left[name], right[name])
        if "exact" in metrics:
            if not metrics["exact"]:
                integer_mismatches.append(name)
        else:
            rows.append((name, metrics))
    rows.sort(key=lambda item: item[1]["relative_l2"], reverse=True)
    print("\n---", title, "---", flush=True)
    if integer_mismatches:
        print("Non-floating mismatches:", integer_mismatches[:top], flush=True)
    for name, metrics in rows[:top]:
        print("%s ref_max=%.9g max_err=%.9g rel_l2=%.9g cosine=%.12f" % (
            name, metrics["ref_max_abs"], metrics["max_abs_error"],
            metrics["relative_l2"], metrics["cosine"]), flush=True)
    return rows, integer_mismatches


def compare(split):
    """Quantify single-card/two-card FP32 differences on one fixed voxel input."""
    sample = select_sample(make_dataset("train"), largest=False)

    # Voxelize exactly once. Reconstruct this same tensor for every trial so that
    # data preprocessing/voxel row order cannot explain a model difference.
    original = EvUAV.custom_collate([sample])
    fixed = {
        "features": original["voxel_ev"].features.detach().cpu().clone(),
        "indices": original["voxel_ev"].indices.detach().cpu().clone(),
        "spatial_shape": list(original["voxel_ev"].spatial_shape),
        "batch_size": original["voxel_ev"].batch_size,
        "label": original["seg_label"].detach().cpu().clone(),
        "p2v_map": original["p2v_map"].detach().cpu().clone(),
    }
    del original
    clear_unused()

    def run(label, parallel):
        setup(37)
        net = evspsegnet_mp(cfg, split) if parallel else evspsegnet(cfg).cuda(0)
        net.train()
        initial = cpu_state(net)
        criterion = STCLoss(k=cfg.k, t=cfg.t, cfg=cfg).cuda(0)
        with torch.cuda.device(0):
            voxel = spconv.SparseConvTensor(
                fixed["features"].cuda(0), fixed["indices"].int().cuda(0),
                fixed["spatial_shape"], fixed["batch_size"])
        batch = {"voxel_ev": voxel, "seg_label": fixed["label"],
                 "p2v_map": fixed["p2v_map"]}
        preds, loss = forward_loss(net, criterion, batch)
        loss.backward()
        check_gradients(net)
        result = {
            "preds": preds.detach().cpu().clone(),
            "loss": loss.detach().cpu().clone(),
            "grads": {k: p.grad.detach().cpu().clone()
                      for k, p in net.named_parameters() if p.grad is not None},
            "initial": initial,
            "state": cpu_state(net),
        }
        print(label, "loss=%.12g pred_min=%.9g pred_max=%.9g" % (
            result["loss"].item(), result["preds"].min().item(),
            result["preds"].max().item()), flush=True)
        del net, criterion, batch, voxel, preds, loss
        clear_unused()
        return result

    results = {
        "single_A": run("single_A", False),
        "single_B": run("single_B", False),
        "dual_A": run("dual_A", True),
        "dual_B": run("dual_B", True),
    }

    pairs = (
        ("SINGLE repeat", "single_A", "single_B"),
        ("DUAL repeat", "dual_A", "dual_B"),
        ("SINGLE vs DUAL", "single_A", "dual_A"),
    )
    summary = {}
    for pair_name, left_name, right_name in pairs:
        print("\n================", pair_name, "================", flush=True)
        left, right = results[left_name], results[right_name]
        pair_summary = {}
        for group in ("initial", "loss", "preds", "grads", "state"):
            rows, integer_mismatches = report_group(
                pair_name + " / " + group, left[group], right[group])
            pair_summary[group] = {
                "worst_relative_l2": rows[0][1]["relative_l2"] if rows else 0.0,
                "worst_cosine": min((row[1]["cosine"] for row in rows), default=1.0),
                "integer_mismatches": len(integer_mismatches),
            }
        for threshold in (0.5, 0.9):
            a = left["preds"] >= threshold
            b = right["preds"] >= threshold
            disagreements = (a != b).sum().item()
            print("threshold %.1f disagreements: %d / %d" %
                  (threshold, disagreements, a.numel()), flush=True)
        summary[pair_name] = pair_summary
    print("\nSUMMARY_JSON", json.dumps(summary, sort_keys=True), flush=True)
    print("FP32 COMPARISON DIAGNOSTIC COMPLETED", flush=True)


def smoke(split):
    setup(37)
    net = evspsegnet_mp(cfg, split).train()
    dataset = make_dataset("train")
    criterion = STCLoss(k=cfg.k, t=cfg.t, cfg=cfg).cuda(0)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sample = select_sample(dataset, largest=True)
    memory(reset=True)
    # Two updates also exercise already-initialized Adam state on the second step.
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        batch = dataset.custom_collate([sample])
        before = {k: p.detach().cpu().clone() for k, p in net.named_parameters()}
        print("Step", step, "events", batch["seg_label"].numel(),
              "voxels", batch["voxel_ev"].features.shape[0], flush=True)
        preds, loss = forward_loss(net, criterion, batch)
        print("Forward passed; loss =", loss.item(), flush=True)
        loss.backward()
        counts = check_gradients(net)
        if not all(counts.values()):
            raise RuntimeError("Both GPUs must receive gradients")
        print("Backward passed; gradient tensors:", counts, flush=True)
        optimizer.step()
        check_state(net)
        changed = {0: 0, 1: 0}
        for name, param in net.named_parameters():
            if not torch.equal(before[name], param.detach().cpu()):
                changed[param.device.index] += 1
        if not all(changed.values()):
            raise RuntimeError("No parameter update on one GPU")
        print("Adam updated parameter tensors:", changed, flush=True)
        print("Memory:", json.dumps(memory()), flush=True)
        del batch, preds, loss, before
    print("DUAL FP32 TWO-STEP SMOKE TEST: OK", flush=True)


def train(split):
    setup(37)
    # Exclusive new directory prevents accidental overwrite of previous experiments.
    root = Path(cfg.model_save_root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "run_config.json").write_text(json.dumps({
        "config": vars(cfg), "split": split, "precision": "FP32, AMP off",
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "gpu_names": [torch.cuda.get_device_name(i) for i in (0, 1)],
        "validation": "eval mode; epochs >= 40; foreground IoU threshold 0.9",
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
            for name in ("train_mp.py", "model/evspsegnet_mp.py", "model/evspsegnet.py",
                         "model/basemodel.py", "utils/stcloss.py", "dataset/ev_uav.py",
                         "dataset/basedataset.py")
        },
    }, indent=2), encoding="utf-8")
    net = evspsegnet_mp(cfg, split).train()
    train_data = make_dataset("train")
    loader = make_loader(train_data, True)
    criterion = STCLoss(k=cfg.k, t=cfg.t, cfg=cfg).cuda(0)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    val_loader = make_loader(make_dataset("val"), False)
    best_iou, best_loss = -float("inf"), float("inf")
    for epoch in range(cfg.epochs):
        net.train()
        total = 0.0
        memory(reset=True)
        progress = tqdm.tqdm(loader, desc="FP32 MP epoch %d" % epoch)
        for step, batch in enumerate(progress):
            optimizer.zero_grad(set_to_none=True)
            try:
                preds, loss = forward_loss(net, criterion, batch)
                loss.backward()
                check_gradients(net)
                optimizer.step()
                check_state(net)
            except Exception:
                print("FAILED epoch=%d step=%d events=%d" %
                      (epoch, step, batch["seg_label"].numel()), flush=True)
                raise
            value = loss.item()
            total += value
            progress.set_postfix(loss=value)
            del preds, loss, batch
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        mean_loss = total / len(loader)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(cpu_state(net), root / "best_loss_seed37.pt")
        val_iou = None
        if epoch >= 40:
            net.eval()
            evaluator = evalute(cfg)
            with torch.no_grad():
                for index, batch in enumerate(val_loader):
                    preds, voxel = net(batch["voxel_ev"])
                    finite(preds, "validation predictions")
                    mapping = batch["p2v_map"].long().cuda(0)
                    evaluator.matches[str(index)] = {
                        "seg_pred": preds[mapping].reshape(-1).cpu(),
                        "seg_gt": batch["seg_label"].reshape(-1).cpu(),
                    }
                    del preds, voxel, mapping, batch
            val_iou = evaluator.evaluate_semantic_segmantation_miou().item()
            if not np.isfinite(val_iou):
                raise RuntimeError("Nonfinite validation IoU")
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(cpu_state(net), root / "best_iou_seed37.pt")
            del evaluator
        torch.save(cpu_state(net), root / "last_seed37.pt")
        record = {"epoch": epoch, "mean_batch_loss": mean_loss,
                  "val_iou": val_iou, "next_lr": optimizer.param_groups[0]["lr"],
                  "memory": memory()}
        with (root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        clear_unused()
    print("TRAINING FINISHED:", root, flush=True)


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This entry requires two visible CUDA GPUs")
    torch.cuda.set_device(0)  # Dataset/HAIS_OP .cuda() allocations use logical GPU 0.
    if cfg.batch_size != 1:
        raise ValueError("Keep batch_size=1 for this initial model-parallel experiment")
    split = int(os.environ.get("EVUAV_MP_SPLIT", "2"))
    mode = os.environ.get("EVUAV_MODE", "smoke")
    print("Mode:", mode, "split:", split, "max_events:", cfg.max_events_num, flush=True)
    print("GPUs:", [torch.cuda.get_device_name(i) for i in (0, 1)], flush=True)
    if mode not in ("compare", "smoke", "train"):
        raise ValueError("EVUAV_MODE must be compare, smoke or train")
    try:
        {"compare": compare, "smoke": smoke, "train": train}[mode](split)
    except Exception:
        # No synchronize or new CUDA allocation here: preserve the original error.
        for device in (0, 1):
            print("Failure memory cuda:%d allocated=%.3f GiB reserved=%.3f GiB peak=%.3f GiB" % (
                device, torch.cuda.memory_allocated(device) / 2**30,
                torch.cuda.memory_reserved(device) / 2**30,
                torch.cuda.max_memory_allocated(device) / 2**30), flush=True)
        raise
