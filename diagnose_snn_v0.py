"""Read-only fixed TRAIN sample diagnosis of saved ANN/SNN overfit runs."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import spconv.pytorch as spconv
from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet_mp import evspsegnet_mp
from model.evspsegnet_snn_v0 import evspsegnet_snn_v0
from train_mp import setup, make_dataset, finite, check_state, clear_unused
from utils.stcloss import STCLoss


def emit(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def divide(a, b):
    return a / b if b else None


def scores(pred, truth, threshold):
    positive = pred >= threshold
    target = truth == 1
    tp = int((positive & target).sum())
    fp = int((positive & ~target).sum())
    fn = int((~positive & target).sum())
    tn = int((~positive & ~target).sum())
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": divide(tp, tp + fp + fn), "precision": divide(tp, tp + fp),
            "recall": divide(tp, tp + fn), "f1": divide(2 * tp, 2 * tp + fp + fn),
            "predicted_positive_fraction": divide(tp + fp, len(truth)),
            "event_fpr_not_official_Fa": divide(fp, fp + tn)}


def distribution(values):
    if not len(values):
        return None
    return {"count": len(values), "mean": float(values.mean()),
            "min": float(values.min()), "max": float(values.max()),
            "p10_p50_p90": [float(v) for v in np.quantile(values, [.1, .5, .9])]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate_bn(net, features, indices, shape, batch_size):
    """Single TRAIN-sample diagnostic: replace BN statistics, never save them.

    All BN layers use current-batch statistics during this pass. momentum=1
    replaces stale buffers immediately. Evaluation may not exactly equal train
    mode because running_var stores the unbiased rather than biased variance.
    This is not a generalization test or a deployment calibration recipe.
    """
    net.eval()
    before = {name: p.detach().cpu().clone() for name, p in net.named_parameters()}
    layers = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not layers or any(not m.track_running_stats for m in layers):
        raise RuntimeError("Expected BatchNorm with tracked running statistics")
    momenta = [m.momentum for m in layers]
    old_stats = [(m.running_mean.detach().clone(), m.running_var.detach().clone()) for m in layers]
    try:
        for m in layers:
            m.train()
            m.momentum = 1.0
        with torch.no_grad():
            x = spconv.SparseConvTensor(features.cuda(0), indices.int().cuda(0), shape, batch_size)
            preds, voxel = net(x)
            finite(preds, "BN calibration predictions")
            check_state(net)
            del preds, voxel, x
    finally:
        for m, momentum in zip(layers, momenta):
            m.momentum = momentum
        net.eval()
    for name, p in net.named_parameters():
        if not torch.equal(before[name], p.detach().cpu()):
            raise RuntimeError("Calibration changed learnable parameter: " + name)
    changed = sum(not torch.equal(mean, m.running_mean) or not torch.equal(var, m.running_var)
                  for m, (mean, var) in zip(layers, old_stats))
    return {"bn_layers": len(layers), "changed_bn_layers": changed,
            "passes": 1, "calibration_momentum": 1.0, "parameters_unchanged": True,
            "sample_scope": "same training sample only; in-memory diagnostic"}


def main():
    if torch.cuda.device_count() < 2 or cfg.batch_size != 1:
        raise RuntimeError("Requires two visible GPUs and batch_size=1")
    torch.cuda.set_device(0)
    sample_name = os.environ.get("EVUAV_SAMPLE", "train_000.npz")
    runs = {
        "ann": Path(os.environ.get("EVUAV_ANN_RUN", "log/ann_v0_overfit_seed37_trial1")),
        "snn": Path(os.environ.get("EVUAV_SNN_RUN", "log/snn_v0_overfit_seed37_trial1")),
    }
    source_root = Path(__file__).resolve().parent
    records = {}
    # Fail before inference if a checkpoint is missing or the experiment differs.
    for variant, root in runs.items():
        metadata = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
        if metadata["variant"] != variant or metadata["mode"] != "overfit":
            raise RuntimeError("Expected %s overfit checkpoint in %s" % (variant, root))
        recorded_sample = metadata.get("sample_override") or "train_000.npz"
        if recorded_sample != sample_name:
            raise RuntimeError("Requested sample differs from saved overfit sample")
        for key in ("seed", "width", "input_channel", "root", "max_events_num", "k", "t",
                    "snn_stages", "snn_steps", "snn_beta", "snn_threshold"):
            if metadata["config"][key] != getattr(cfg, key):
                raise RuntimeError("Config mismatch for %s: %s" % (variant, key))
        for name in ("model/evspsegnet.py", "model/basemodel.py", "model/evspsegnet_mp.py",
                     "model/evspsegnet_snn_v0.py", "model/lif_rate.py",
                     "dataset/ev_uav.py", "dataset/basedataset.py", "utils/stcloss.py"):
            if sha(source_root / name) != metadata["source_sha256"][name]:
                raise RuntimeError("Source changed since training: " + name)
        checkpoint = root / ("last_seed%d.pt" % cfg.seed)
        records[variant] = (metadata, checkpoint, sha(checkpoint))

    setup(cfg.seed)
    dataset = make_dataset("train")
    sample_path = Path(dataset.root) / sample_name
    with np.load(str(sample_path)) as raw:
        raw_count = len(raw["ev_loc"])
    # Existing runs did not save randomly subsampled inputs; don't claim exact
    # replay for such a sample. train_000 has 39169 events, below the cap.
    if raw_count >= cfg.max_events_num:
        raise RuntimeError("Randomly subsampled training input cannot be exactly replayed; use an uncapped sample")
    batch = EvUAV.custom_collate([dataset[dataset.file_list.index(sample_name)]])
    x = batch["voxel_ev"]
    features, indices = x.features.detach().cpu(), x.indices.detach().cpu()
    shape, batch_size = list(x.spatial_shape), x.batch_size
    mapping = batch["p2v_map"].long().cuda(0)
    labels = batch["seg_label"].float().reshape(-1).cuda(0)
    truth = labels.cpu().numpy()
    if not np.isin(truth, [0, 1]).all():
        raise RuntimeError("Expected binary event labels")
    del batch, x
    clear_unused()
    emit({"type": "sample", "name": sample_name, "split": "train",
          "sha256": sha(sample_path), "events": len(truth), "voxels": len(features),
          "positive_fraction": float(truth.mean()),
          "all_background": scores(np.zeros(len(truth)), truth, .5),
          "note": "null means undefined; event recall is not official object Pd"})
    criterion = STCLoss(k=cfg.k, t=cfg.t, cfg=cfg).cuda(0)
    for variant, (metadata, checkpoint, original_hash) in records.items():
        setup(cfg.seed)
        net = (evspsegnet_mp(cfg, metadata["split"]) if variant == "ann" else
               evspsegnet_snn_v0(cfg, metadata["split"]))
        state = torch.load(str(checkpoint), map_location="cpu")
        modes = ("eval", "train", "eval_bncal") if os.environ.get("EVUAV_BN_CALIBRATE", "0") == "1" else ("eval", "train")
        for mode in modes:
            # Restore ALL weights and BN buffers before every independent pass.
            net.load_state_dict(state, strict=True)
            net.train(mode == "train")
            check_state(net)
            if variant == "snn":
                net.diagnostics(True)
            if mode == "eval_bncal":
                result = calibrate_bn(net, features, indices, shape, batch_size)
                emit(dict(type="bn_calibration", variant=variant, **result))
            with torch.no_grad():
                voxel_input = spconv.SparseConvTensor(features.cuda(0), indices.int().cuda(0), shape, batch_size)
                preds, voxel = net(voxel_input)
                finite(preds, "predictions")
                check_state(net)
                loss = criterion(voxel, mapping, preds, labels)
                finite(loss, "loss")
                event_probs = preds[mapping].reshape(-1).cpu().numpy()
                zeros = torch.zeros_like(preds)
                background_loss = criterion(voxel.replace_feature(zeros), mapping, zeros, labels)
                finite(background_loss, "background loss")
                # Unweighted BCE supplies a second view; STC weights depend on
                # each prediction, so STC losses are not a fixed-weight score.
                p = np.clip(event_probs.astype(np.float64), 1e-7, 1 - 1e-7)
                bce = float(-(truth * np.log(p) + (1 - truth) * np.log(1 - p)).mean())
                emit({"type": "diagnosis", "variant": variant, "mode": mode,
                      "checkpoint": str(checkpoint), "checkpoint_sha256": original_hash,
                      "stc_loss": float(loss), "all_background_stc_loss": float(background_loss),
                      "unweighted_bce": bce,
                      "scores": [scores(event_probs, truth, t) for t in (.5, .9)],
                      "foreground_probabilities": distribution(event_probs[truth == 1]),
                      "background_probabilities": distribution(event_probs[truth == 0]),
                      "activity": net.spike_stats() if variant == "snn" else {}})
            del voxel_input, preds, voxel, loss, zeros, background_loss
        del net, state
        clear_unused()
        if sha(checkpoint) != original_hash:
            raise RuntimeError("Checkpoint changed during diagnosis")
    emit({"status": "diagnosis completed", "checkpoint_files_unchanged": True})


if __name__ == "__main__":
    main()
