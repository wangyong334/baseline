"""v6-2：V2-1 判决层 + V3 发布层接到冻结的 V5 骨干上的训练与评估（检验"之前的方案在 V5 上是否还有效"）。

骨干 = V5 定型权重（BIR + 峰值消息约束），冻结、不回传梯度；V5 的零等待输出原样保留并单列报告。
新增的只有 V2 的两个逐像素头（model/evidence_readout.EvidenceHeads，读 dec1 的 u_pre）：
    mark   逐事件 BCE（无 pos_weight），零等待读出 net
    log_g  本窗目标强度场，对 3x3 平滑后的目标计数做泊松负对数似然
判决层与发布层无可学习参数，逐式照搬 v3 分支（model/evidence_neuron.py、model/publish_readout.py）：
    fused_d{1..5}  V2-1：mark + 之后 d 窗沿 49 条速度管道的证据
    pub            V3：双边界发布 + 锚定证据链（年龄 0..5 窗择机发布）
模式（--mode）：
    smoke  2 个训练块（梯度只到两个头）+ 1 条验证序列（逐窗 / 逐片段两种执行的判定差异）
    train  与 V5 相同的多流排布逐块训练；每轮在 val 上按 net 的 IoU 选模型（V2 的规则）
    eval   逐窗执行（与 V5 正式评估同一路径）跑完整划分：V5、net、fused_d1..5、pub 的四个指标，
           相对 V5 的纠正 / 破坏数（目标 / 第一拍 / 目标旁背景 / 远处背景），pub 的发布年龄与原因
配置：configs/evisseg_stream_evidence.yaml 只有 EVIDENCE 一节，base_config 指向 V5 的配置。
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")   # 必须在导入 torch 之前设置

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_stream_v1 as V5  # noqa: E402
from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from dataset.stream_source import make_window_source  # noqa: E402
from model.evidence_readout import DEC1, EvidenceHeads, EvidenceRun, pixel_counts  # noqa: E402
from utils.evidence_loss import smoothed_target  # noqa: E402
from utils.evidence_metrics import (event_groups, flip_counts, iou_counts, publish_summary,  # noqa: E402
                                    split_report)
from utils.stream_common import linear_epoch_lr, load_flat_config  # noqa: E402

REQUIRED_KEYS = (
    "base_config", "backbone_checkpoint", "head_hidden", "mark_prior", "intensity_prior", "log_g_max",
    "loss_mark_weight", "loss_intensity_weight", "loss_intensity_smooth",
    "bg_fast_ms", "bg_slow_ms", "bg_prior", "bg_prior_windows", "bg_floor", "bg_smooth_radius",
    "cusum_axis_velocities", "cusum_footprint", "cusum_compensator", "cusum_nb_kappa", "cusum_aggregate",
    "cusum_track_tau_ms", "cusum_memory_gain", "cusum_gate_eps", "readout_delays", "fusion_weight",
    "publish_deadline", "publish_upper", "publish_lower", "publish_collapse", "publish_gate", "publish_anchor",
    "evidence_epochs", "evidence_lr", "evidence_lr_end", "evidence_grad_clip", "evidence_streams",
    "evidence_chunk", "evidence_eval_chunk", "evidence_save_root")
DATA_KEYS = ("height", "width_px", "pad_height", "pad_width", "window_ms", "n_windows", "time_bins", "input_clip")
EVAL_KEYS = ("threshold", "pd_detT", "correct_thresh", "rolling_window", "segments", "timing_warmup_windows")


def parse_args():
    parser = argparse.ArgumentParser(description="V2/V3 decision layer on the frozen V5 backbone")
    parser.add_argument("--config", required=True, help="configs/evisseg_stream_evidence.yaml")
    parser.add_argument("--mode", choices=("smoke", "train", "eval"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--reference", default=None,
                        help="eval：V5 同一权重的评估 JSON（eval_<split>_best_val_iou_seed37.json），核对 V5 行与之相同")
    parser.add_argument("--overwrite", action="store_true", help="eval：允许覆盖已有结果")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def build_config(path):
    """读取 EVIDENCE 配置并与 base_config 合并（两者不得有同名键）。"""
    ev = load_flat_config(path)
    missing = [k for k in REQUIRED_KEYS if k not in ev]
    unknown = sorted(set(ev) - set(REQUIRED_KEYS))
    if missing or unknown:
        raise ValueError("EVIDENCE 配置缺少 %s，多出 %s" % (missing, unknown))
    base = load_flat_config(ev["base_config"])
    clash = sorted(set(ev) & set(base))
    if clash:
        raise ValueError("EVIDENCE 配置与 base_config 有同名键: %s" % clash)
    if int(ev["evidence_epochs"]) < 1 or not ev["readout_delays"] or min(int(d) for d in ev["readout_delays"]) < 1:
        raise ValueError("需要 evidence_epochs >= 1，readout_delays 非空且都 >= 1")
    if int(ev["publish_deadline"]) != max(int(d) for d in ev["readout_delays"]):
        raise ValueError("publish_deadline 应等于最长的固定等待（与 V3 相同：D = 5）")
    cfg = dict(base)
    cfg.update(ev)
    return cfg


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_backbone(cfg, device, execution, input_device):
    """载入并冻结 V5 骨干。返回 (net, net_cfg, q99)；数据与输入键必须与 base_config 一致。"""
    ckpt = torch.load(cfg["backbone_checkpoint"], map_location=device)
    net_cfg = dict(ckpt["config"])
    if net_cfg.get("neuron") != "tif":
        raise ValueError("只接 V5 的 BIR（neuron: tif）骨干")
    differ = [k for k in DATA_KEYS if net_cfg.get(k) != cfg.get(k)]
    if differ:
        raise ValueError("checkpoint 与 base_config 的数据键不一致: %s" % differ)
    net_cfg["root"] = cfg["root"]
    for key in EVAL_KEYS:
        net_cfg[key] = cfg[key]
    net_cfg["execution"], net_cfg["input_device"] = execution, input_device
    V5.seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    net = V5.build_net(net_cfg, device)
    net.load_state_dict(ckpt["model"])
    if not bool(net.gain_calibrated):
        raise RuntimeError("checkpoint 中的增益未校准")
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    q99 = np.asarray(ckpt["stats"]["q99"], dtype=np.float32)
    return net, net_cfg, q99


def make_heads(cfg, net_cfg, device, dtype):
    torch.manual_seed(int(cfg["seed"]) + 7000)
    return EvidenceHeads(int(net_cfg["channels"][0]), int(cfg["head_hidden"]), float(cfg["mark_prior"]),
                         float(cfg["intensity_prior"]), float(cfg["log_g_max"])).to(device=device, dtype=dtype)


def decision_cfg(cfg, net_cfg):
    """判决层与发布层用到的配置（窗长、阈值取 V5 的口径）。"""
    out = dict(cfg)
    out["window_ms"] = float(net_cfg["window_ms"])
    out["threshold"] = float(net_cfg["threshold"])
    return out


# ---------------------------------------------------------------------------
# 两个头的损失（v3 utils/evidence_loss.py 的 mark_loss / intensity_loss，另按有效窗掩码）
# ---------------------------------------------------------------------------


def head_losses(heads, u7, events, labels, valid, smooth):
    """u7 [T,B,C,H,W]；events 的 t 为片段内窗号、b 为流号；valid [T,B] 有录像的窗为 1。
    返回 (bce_sum, nll_sum)：逐事件 BCE（无 pos_weight）之和、有效窗上目标强度泊松负对数似然之和。"""
    steps, batch = int(u7.shape[0]), int(u7.shape[1])
    H, W = int(u7.shape[3]), int(u7.shape[4])
    mark, log_g = heads(u7.reshape((steps * batch,) + tuple(u7.shape[2:])))
    flat = events["t"] * batch + events["b"]
    bce = F.binary_cross_entropy_with_logits(mark[flat, 0, events["y"], events["x"]], labels, reduction="sum") \
        if labels.numel() else mark.sum() * 0.0
    target = pixel_counts(events["y"], events["x"], flat, steps * batch, H, W, log_g.dtype, weight=labels)
    m = smoothed_target(target.view(steps, batch, 1, H, W), smooth).view(steps * batch, 1, H, W)
    per = (log_g.exp() - m * log_g).sum(dim=(1, 2, 3))
    nll = (per * valid.reshape(-1).to(per.dtype)).sum()
    return bce, nll


def train_epoch(net, heads, dataset, lengths, optimizer, cfg, net_cfg, q99, device, rng, max_chunks=None,
                plan=None):
    """一轮：B 条流逐块前向（骨干无梯度），损失 = (w_mark * BCE + w_int * NLL) / 本块事件数。"""
    k, n_streams = int(cfg["evidence_chunk"]), int(cfg["evidence_streams"])
    if plan is None:
        plan = V5.plan_streams(lengths, n_streams, k, rng, net_cfg.get("stream_layout", "greedy"))
    item_ends = {(index, first + n) for _, items in plan for index, first, n in map(V5.plan_item, items)}
    n_chunks = int(math.ceil(V5.stream_total_windows(plan) / float(k)))
    if max_chunks is not None:
        n_chunks = min(n_chunks, int(max_chunks))
    dtype = next(net.parameters()).dtype
    sources = V5.StreamSources(dataset, net_cfg, q99, device, dtype)
    w_mark, w_int = float(cfg["loss_mark_weight"]), float(cfg["loss_intensity_weight"])
    smooth, clip = int(cfg["loss_intensity_smooth"]), float(cfg["evidence_grad_clip"])
    stats = {"bce_sum": 0.0, "nll_sum": 0.0, "events": 0}
    net_states, updates = None, 0
    heads.train()
    for c in range(n_chunks):
        start = c * k
        pieces = V5.stream_pieces(plan, start, start + k)
        x, events, labels, keep = V5.assemble_streams(sources, pieces, k, net_cfg, device, dtype)
        valid = V5.stream_valid(pieces, k, device, dtype)
        with torch.no_grad():
            _, net_states, info = net.forward_chunk(x, events, net_states, keep=keep, collect=True)
        bce, nll = head_losses(heads, info["u_pre"][DEC1], events, labels, valid, smooth)
        n = int(labels.numel())
        loss = (w_mark * bce + w_int * nll) / float(max(n, 1))
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError("损失出现非有限值: 块 %d" % c)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), clip)
        optimizer.step()
        updates += 1
        stats["bce_sum"] += float(bce.detach())
        stats["nll_sum"] += float(nll.detach())
        stats["events"] += n
        for stream in pieces:
            for index, lo, hi, _, _ in stream:
                if (index, hi) in item_ends:
                    sources.release(index)
    stats.update(chunks=n_chunks, updates=updates)
    return stats


# ---------------------------------------------------------------------------
# 推理：一条序列，逐窗（step，与 V5 正式评估同一路径）或逐片段（layer）
# ---------------------------------------------------------------------------


def evaluate_sequence(net, heads, seq, cfg, net_cfg, q99, device, execution, decision=True):
    """推理一条序列。返回 dict：v5 [N]（V5 零等待概率）、probs {net, fused_d*, pub}（decision=False 时只有 net）、
    extra（pub 的 age / reason / z）、nll（目标强度的泊松负对数似然之和）。调用方负责 torch.no_grad()。"""
    dtype = next(net.parameters()).dtype
    source = make_window_source(seq, net_cfg, q99, device, dtype)
    H, W = int(net_cfg["pad_height"]), int(net_cfg["pad_width"])
    smooth = int(cfg["loss_intensity_smooth"])
    n, n_windows = seq.n_events, seq.n_windows
    run = EvidenceRun(decision_cfg(cfg, net_cfg), H, W, device, dtype) if decision else None
    v5 = np.full(n, np.nan, dtype=np.float32)
    net_parts = ([], [])
    nll_sum = 0.0
    net_states = None
    if execution == "step":
        spans = [(k, k + 1) for k in range(n_windows)]
    else:
        size = int(cfg["evidence_eval_chunk"])
        spans = [(s, min(s + size, n_windows)) for s in range(0, n_windows, size)]
    for start, end in spans:
        steps = end - start
        if execution == "step":
            x, events, labels, idx = source.window(start)
            logits, net_states, info = net(x, events, net_states, collect=True)
            u7 = info["u_pre"][DEC1].unsqueeze(0)
            events = dict(events, t=torch.zeros_like(events["b"]))
        else:
            x, events, labels, idx = source.chunk(start, end)
            logits, net_states, info = net.forward_chunk(x, events, net_states, collect=True)
            u7 = info["u_pre"][DEC1]
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size:
            v5[idx] = torch.sigmoid(logits).float().cpu().numpy()
        mark, log_g = heads(u7.reshape((steps,) + tuple(u7.shape[2:])))
        target = pixel_counts(events["y"], events["x"], events["t"], steps, H, W, dtype, weight=labels)
        m = smoothed_target(target.view(steps, 1, 1, H, W), smooth).view(steps, 1, H, W)
        nll_sum += float((log_g.exp() - m * log_g).sum())
        ml = mark[events["t"], 0, events["y"], events["x"]]
        if run is None:
            net_parts[0].append(idx)
            net_parts[1].append(torch.sigmoid(ml).float().cpu().numpy())
            continue
        t_np = events["t"].cpu().numpy()
        for t in range(steps):
            sel = np.flatnonzero(t_np == t)
            sel_t = torch.from_numpy(sel).to(device)
            run.step(start + t, log_g[t:t + 1], ml[sel_t], events["y"][sel_t], events["x"][sel_t], idx[sel])
    if np.isnan(v5).any():
        raise RuntimeError("序列 %s 的 V5 概率未填满" % seq.name)
    if run is None:
        from dataset.stream_windows import refill_by_index
        return {"v5": v5, "probs": {"net": refill_by_index(n, *net_parts)}, "extra": {}, "nll": nll_sum}
    probs, extra = run.outputs(n)
    return {"v5": v5, "probs": probs, "extra": extra, "nll": nll_sum}


def readout_names(cfg):
    return ["net"] + ["fused_d%d" % int(d) for d in cfg["readout_delays"]] + ["pub"]


def quick_val(net, heads, dataset, cfg, net_cfg, q99, device):
    """训练中验证（逐片段执行，不跑判决层）：net 的 IoU（全局拼接口径，阈值同 V5）与每事件的强度 NLL。"""
    thr = float(net_cfg["threshold"])
    counts = np.zeros(3, dtype=np.int64)
    nll, events = 0.0, 0
    heads.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            seq = dataset[i]
            res = evaluate_sequence(net, heads, seq, cfg, net_cfg, q99, device, "layer", decision=False)
            counts += iou_counts(res["probs"]["net"], seq.label, thr)
            nll += res["nll"]
            events += int(seq.n_events)
    heads.train()
    return float(counts[0]) / float(max(counts.sum(), 1)), nll / float(max(events, 1))


# ---------------------------------------------------------------------------
# 模式
# ---------------------------------------------------------------------------


def save_heads(path, heads, epoch, score, cfg):
    torch.save({"model": heads.state_dict(), "epoch": epoch, "score": score, "config": cfg,
                "backbone_sha256": file_sha256(cfg["backbone_checkpoint"])}, str(path))


def head_parameters(heads):
    return int(sum(p.numel() for p in heads.parameters()))


def mode_smoke(args, cfg):
    """2 个训练块 + 第 1 条验证序列（逐窗 / 逐片段两种执行比对，含判决层与发布层）。"""
    device = torch.device(args.device)
    net, net_cfg, q99 = load_backbone(cfg, device, "layer", "gpu")
    dtype = next(net.parameters()).dtype
    heads = make_heads(cfg, net_cfg, device, dtype)
    optimizer = torch.optim.Adam(heads.parameters(), lr=float(cfg["evidence_lr"]))
    names = sorted(n for n in os.listdir(os.path.join(cfg["root"], "train")) if n.endswith(".npz"))
    train_set = EvUAVStream(cfg["root"], "train", net_cfg, names[:int(cfg["evidence_streams"])])
    lengths = [int(train_set[i].n_windows) for i in range(len(train_set))]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    out = train_epoch(net, heads, train_set, lengths, optimizer, cfg, net_cfg, q99, device,
                      np.random.RandomState(int(cfg["seed"])), max_chunks=2)
    train_seconds = time.perf_counter() - t0
    leaked = [name for name, p in net.named_parameters() if p.grad is not None]
    if leaked:
        raise RuntimeError("骨干参数出现了梯度: %s" % leaked[:5])
    missing = [name for name, p in heads.named_parameters() if p.grad is None]
    if missing:
        raise RuntimeError("头的参数没有梯度: %s" % missing)
    seq = EvUAVStream(cfg["root"], "val", net_cfg)[0]
    heads.eval()
    with torch.no_grad():
        t1 = time.perf_counter()
        step = evaluate_sequence(net, heads, seq, cfg, dict(net_cfg, input_device="cpu"), q99, device, "step")
        step_seconds = time.perf_counter() - t1
        layer = evaluate_sequence(net, heads, seq, cfg, net_cfg, q99, device, "layer")
    thr = np.float32(net_cfg["threshold"])
    gap = {}
    for name in readout_names(cfg):
        a, b = step["probs"][name], layer["probs"][name]
        gap[name] = {"max_abs": float(np.abs(a - b).max()),
                     "decision_flip_frac": float(((a >= thr) != (b >= thr)).mean())}
    report = {"train": out, "train_seconds": train_seconds, "step_eval_seconds": step_seconds,
              "step_ms_per_window": 1e3 * step_seconds / max(seq.n_windows, 1), "step_vs_layer": gap,
              "publish_age_mean_windows": float(np.mean(step["extra"]["age_pub"])),
              "peak_memory_gib": V5.peak_memory_gib(device), "head_parameters": head_parameters(heads),
              "val_sequence": seq.name}
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    bad = {k: g for k, g in gap.items() if g["decision_flip_frac"] > 1e-3}
    if bad:
        raise RuntimeError("逐窗与逐片段两种执行的判定差异过大: %s" % bad)
    print("SMOKE TEST FINISHED", flush=True)


def mode_train(args, cfg):
    device = torch.device(args.device)
    net, net_cfg, q99 = load_backbone(cfg, device, "layer", "gpu")
    dtype = next(net.parameters()).dtype
    heads = make_heads(cfg, net_cfg, device, dtype)
    root = Path(cfg["evidence_save_root"])
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics.jsonl"
    if metrics_path.exists():
        raise RuntimeError("输出目录已有 metrics.jsonl，请换目录: %s" % root)
    V5.write_json(root / "run_config_train.json", {
        "config": cfg, "net_config": net_cfg, "head_parameters": head_parameters(heads),
        "backbone_sha256": file_sha256(cfg["backbone_checkpoint"]), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"})
    optimizer = torch.optim.Adam(heads.parameters(), lr=float(cfg["evidence_lr"]))
    train_set = EvUAVStream(cfg["root"], "train", net_cfg)
    val_set = EvUAVStream(cfg["root"], "val", net_cfg)
    lengths = [int(train_set[i].n_windows) for i in range(len(train_set))]
    seed, epochs = int(cfg["seed"]), int(cfg["evidence_epochs"])
    print("V2/V3 判决层接 V5：骨干 %s（冻结），头参数 %d | 训练集 %d 条录像 %d 窗，%d 条流、块 %d 窗" % (
        cfg["backbone_checkpoint"], head_parameters(heads), len(lengths), sum(lengths),
        int(cfg["evidence_streams"]), int(cfg["evidence_chunk"])), flush=True)
    best = -float("inf")
    for epoch in range(epochs):
        lr = linear_epoch_lr(epoch, epochs, float(cfg["evidence_lr"]), float(cfg["evidence_lr_end"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        out = train_epoch(net, heads, train_set, lengths, optimizer, cfg, net_cfg, q99, device,
                          np.random.RandomState(seed * 1000 + epoch + 7))
        train_seconds = time.perf_counter() - t0
        val_iou, val_nll = quick_val(net, heads, val_set, cfg, net_cfg, q99, device)
        ev = max(out["events"], 1)
        record = {"epoch": epoch, "lr": lr, "train_seconds": train_seconds,
                  "epoch_seconds": time.perf_counter() - t0, "updates": out["updates"],
                  "bce_per_event": out["bce_sum"] / ev, "nll_per_event": out["nll_sum"] / ev,
                  "val_net_iou": val_iou, "val_nll_per_event": val_nll,
                  "peak_memory_gib": V5.peak_memory_gib(device)}
        if val_iou > best:
            best = val_iou
            save_heads(root / "best_heads.pt", heads, epoch, val_iou, cfg)
        save_heads(root / "last_heads.pt", heads, epoch, val_iou, cfg)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print("epoch %d lr %.2e | bce %.4f nll %.4f（每事件）| val net IoU %.4f val nll %.4f | 更新 %d 次 | "
              "%.0fs（训练 %.0fs）| 显存峰值 %.2f GiB" % (
                  epoch, lr, record["bce_per_event"], record["nll_per_event"], val_iou, val_nll, out["updates"],
                  record["epoch_seconds"], train_seconds, record["peak_memory_gib"] or 0.0), flush=True)
    V5.write_json(root / "summary.json", {"best_val_net_iou": best, "epochs": epochs})
    print("TRAINING FINISHED:", root, flush=True)


def mode_eval(args, cfg):
    """逐窗执行跑完整划分：V5、net、fused_d1..5、pub 的四个指标与纠正 / 破坏数，pub 的发布统计。"""
    device = torch.device(args.device)
    net, net_cfg, q99 = load_backbone(cfg, device, "step", "cpu")
    dtype = next(net.parameters()).dtype
    root = Path(cfg["evidence_save_root"])
    out_path = root / ("eval_evidence_%s.json" % args.split)
    if out_path.exists() and not args.overwrite:
        raise RuntimeError("结果已存在: %s（使用 --overwrite 覆盖）" % out_path)
    heads = make_heads(cfg, net_cfg, device, dtype)
    ckpt = torch.load(str(root / "best_heads.pt"), map_location=device)
    if ckpt["backbone_sha256"] != file_sha256(cfg["backbone_checkpoint"]):
        raise RuntimeError("头训练时的骨干与当前 backbone_checkpoint 不是同一个文件")
    heads.load_state_dict(ckpt["model"])
    heads.eval()
    dataset = EvUAVStream(cfg["root"], args.split, net_cfg)
    window_ms = float(net_cfg["window_ms"])
    names = readout_names(cfg)
    seqs, v5, probs = [], [], {name: [] for name in names}
    ages, reasons = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(len(dataset)):
            seq = dataset[i]
            res = evaluate_sequence(net, heads, seq, cfg, net_cfg, q99, device, "step")
            seqs.append({"name": seq.name, "t": seq.t, "x": seq.x, "y": seq.y, "label": seq.label,
                         "target_id": seq.target_id, "groups": event_groups(seq, window_ms)})
            v5.append(res["v5"])
            for name in names:
                probs[name].append(res["probs"][name])
            ages.append(res["extra"]["age_pub"])
            reasons.append(res["extra"]["reason_pub"])
    seconds = time.perf_counter() - t0
    thr = float(net_cfg["threshold"])
    results = {"split": args.split, "backbone_checkpoint": cfg["backbone_checkpoint"], "epoch": int(ckpt["epoch"]),
               "threshold": thr, "seconds": seconds, "v5": split_report(seqs, v5, net_cfg), "readouts": {},
               "publish": publish_summary([s["label"] for s in seqs], ages, reasons, int(cfg["publish_deadline"]),
                                          window_ms)}
    for name in names:
        rep = split_report(seqs, probs[name], net_cfg)
        rep["flips_vs_v5"] = flip_counts(seqs, v5, probs[name], thr)
        results["readouts"][name] = rep
    if args.reference:
        ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))["carry"]
        check = {k: (results["v5"][k], ref[k]) for k in ("iou", "acc", "pd", "fa")}
        results["v5_vs_reference"] = {k: {"here": a, "reference": b, "equal": a == b} for k, (a, b) in check.items()}
        print("[V5] 零等待与 V5 正式评估：%s" % {k: ("相同" if a == b else "不同 %r vs %r" % (a, b))
                                              for k, (a, b) in check.items()}, flush=True)
    V5.write_json(out_path, results)
    r = results["v5"]
    print("[%s] V5 零等待   IoU %.4f ACC %.4f Pd %.4f Fa %.3e" % (args.split, r["iou"], r["acc"], r["pd"], r["fa"]),
          flush=True)
    fmt = "[%s] %-10s IoU %.4f ACC %.4f Pd %.4f Fa %.3e | 纠正/破坏 目标 %d/%d 第一拍 %d/%d 目标旁背景 %d/%d 远处背景 %d/%d"
    for name in names:
        r = results["readouts"][name]
        f = r["flips_vs_v5"]
        print(fmt % (args.split, name, r["iou"], r["acc"], r["pd"], r["fa"], f["target"]["corrected"],
                     f["target"]["broken"], f["onset"]["corrected"], f["onset"]["broken"], f["near_bg"]["corrected"],
                     f["near_bg"]["broken"], f["far_bg"]["corrected"], f["far_bg"]["broken"]), flush=True)
    p = results["publish"]
    print("[%s] pub 目标事件发布年龄：%s | 平均 %.1f ms（全部事件 %.1f ms）" % (
        args.split, "  ".join("%d窗 %.1f%%" % (d, 100 * v) for d, v in enumerate(p["age_hist_target"])),
        p["mean_delay_target_ms"], p["mean_delay_all_ms"]), flush=True)
    print("[%s] pub 目标事件发布原因：%s" % (
        args.split, "  ".join("%s %.1f%%" % (k, 100 * v) for k, v in p["reason_target"].items())), flush=True)
    print("EVAL FINISHED:", out_path, flush=True)


def main():
    args = parse_args()
    cfg = build_config(args.config)
    {"smoke": mode_smoke, "train": mode_train, "eval": mode_eval}[args.mode](args, cfg)


if __name__ == "__main__":
    main()
