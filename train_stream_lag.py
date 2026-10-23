"""V6 第一版：自定时判决神经元的滞后读出头（Stream-SNN-Innovations.md 13.12）。训练、评估与导出。

骨干 = V5 定型权重（BIR + 峰值消息约束），冻结、不回传梯度，零等待输出（d = 0）与 V5 逐位相同。
新增的只有滞后读出头（model/lag_readout.py）：第 k 窗出生的事件在第 k+d 窗（d = 1..D）再打一次分。
两个 variant 共用一次骨干前向、各自训练：pixel（只读原像素 dec1）与 multiscale（另读粗层与 enc4 邻域）。

模式（--mode）：
    smoke  2 个训练片段 + 1 条验证序列：检查梯度只到读出头、逐窗与逐层两种执行给出相同的分数、显存与耗时
    train  标准多流（与 V5 相同的排布）逐片段训练读出头；每轮在 val 上按各滞后的 IoU 选模型（每个 variant 各自）
    eval   逐窗执行（与 V5 正式评估相同的路径）跑完整划分：每个 variant、每个固定滞后 d = 0..D 的四个指标
           （延迟类指标与能耗一样放到最后阶段统一计算）；
           --dump 时导出逐事件、逐年龄的概率，供 tools/lag_publish.py 做自适应发布（在 val 上定规则，在 test 上报告）
配置：configs/evisseg_stream_lag*.yaml 只有 LAG 一节，base_config 指向 V5 的配置（数据路径、评测口径），
backbone_checkpoint 指向 V5 的权重；网络结构、q99、pos_weight 一律取自 checkpoint。
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
from model.lag_readout import (LAG_VARIANTS, LagBuffer, LagReadout, birth_features, lag_logits,  # noqa: E402
                               lag_maps)
from utils.lag_metrics import iou_acc_counts, split_report  # noqa: E402
from utils.stream_common import linear_epoch_lr, load_flat_config  # noqa: E402

REQUIRED_KEYS = ("base_config", "backbone_checkpoint", "lag_variants", "lags", "lag_hidden", "lag_epochs",
                 "lag_lr", "lag_lr_end", "lag_grad_clip", "lag_streams", "lag_chunk", "lag_save_root")
OPTIONAL_KEYS = ("lag_eval_chunk",)
# checkpoint 与 base_config 必须一致的数据与输入键（保证读的是训练骨干时的同一数据集）
DATA_KEYS = ("height", "width_px", "pad_height", "pad_width", "window_ms", "n_windows", "time_bins", "input_clip")
EVAL_KEYS = ("threshold", "pd_detT", "correct_thresh", "rolling_window", "segments", "timing_warmup_windows")


def parse_args():
    parser = argparse.ArgumentParser(description="V6 lag readout on the frozen V5 backbone")
    parser.add_argument("--config", required=True, help="configs/evisseg_stream_lag*.yaml")
    parser.add_argument("--mode", choices=("smoke", "train", "eval"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--dump", action="store_true", help="eval：导出逐事件、逐年龄的概率")
    parser.add_argument("--reference", default=None,
                        help="eval：V5 同一权重的评估 JSON（eval_<split>_best_val_iou_seed37.json），用于核对 d=0 与 V5 相同")
    parser.add_argument("--overwrite", action="store_true", help="eval：允许覆盖已有结果")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def build_config(path):
    """读取 LAG 配置并与 base_config 合并（两者不得有同名键）。"""
    lag = load_flat_config(path)
    missing = [k for k in REQUIRED_KEYS if k not in lag]
    unknown = sorted(set(lag) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
    if missing or unknown:
        raise ValueError("LAG 配置缺少 %s，多出 %s" % (missing, unknown))
    base = load_flat_config(lag["base_config"])
    clash = sorted(set(lag) & set(base))
    if clash:
        raise ValueError("LAG 配置与 base_config 有同名键: %s" % clash)
    variants = list(lag["lag_variants"])
    if not variants or len(set(variants)) != len(variants) or any(v not in LAG_VARIANTS for v in variants):
        raise ValueError("lag_variants 必须是 %s 的非空、无重复子集" % (LAG_VARIANTS,))
    if int(lag["lags"]) < 1 or int(lag["lag_epochs"]) < 2 or int(lag["lag_streams"]) < 1:
        raise ValueError("需要 lags >= 1、lag_epochs >= 2、lag_streams >= 1")
    if int(lag["lag_chunk"]) < int(lag["lags"]):
        raise ValueError("lag_chunk 不得小于 lags（每个 (事件, 滞后) 才能在一个片段里处理）")
    cfg = dict(base)
    cfg.update(lag)
    cfg.setdefault("lag_eval_chunk", 32)
    return cfg


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_backbone(cfg, device, execution, input_device):
    """载入并冻结 V5 骨干。返回 (net, net_cfg, q99, pos_weight)。

    net_cfg = checkpoint 的配置 + 当前数据根目录与评测口径 + 本次的执行方式；数据与输入键必须与 base_config 一致。
    """
    ckpt = torch.load(cfg["backbone_checkpoint"], map_location=device)
    net_cfg = dict(ckpt["config"])
    if net_cfg.get("neuron") != "tif":
        raise ValueError("V6 第一版只接 V5 的 BIR（neuron: tif）骨干")
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
    stats = ckpt["stats"]
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    pos_weight = min(float(stats["train_neg"]) / float(stats["train_pos"]), float(net_cfg["pos_weight_cap"]))
    return net, net_cfg, q99, pos_weight


def make_readouts(cfg, net_cfg, device, dtype):
    """每个 variant 一组滞后读出头（新模块在骨干构造之后创建，种子固定）。"""
    torch.manual_seed(int(cfg["seed"]) + 6000)
    return {v: LagReadout(tuple(net_cfg["channels"]), int(cfg["lags"]), v, int(cfg["lag_hidden"])).to(
        device=device, dtype=dtype) for v in cfg["lag_variants"]}


def readout_parameters(readouts):
    return {v: sum(p.numel() for p in r.parameters()) for v, r in readouts.items()}


# ---------------------------------------------------------------------------
# 训练：与 V5 相同的多流排布（plan_streams / stream_pieces / assemble_streams）
# ---------------------------------------------------------------------------


def item_end_table(plan, start, steps, device):
    """[steps, B]：流时间第 start+t 窗所在录像（计划里的一项）在流时间轴上的结束窗号（开区间）；空窗为 -1。"""
    table = torch.full((int(steps), len(plan)), -1, dtype=torch.long)
    stop = int(start) + int(steps)
    for b, (delay, items) in enumerate(plan):
        s = int(delay)
        for item in items:
            _, _, n = V5.plan_item(item)
            e = s + int(n)
            lo, hi = max(s, int(start)), min(e, stop)
            if lo < hi:
                table[lo - int(start):hi - int(start), b] = e
            s = e
            if s >= stop:
                break
    return table.to(device)


def train_epoch(net, readouts, dataset, lengths, optimizer, cfg, net_cfg, q99, pos_weight, device, rng,
                max_chunks=None, plan=None):
    """一轮：B 条流逐片段前向（骨干无梯度），对片段内到期的 (事件, 滞后) 算 BCE，只更新读出头。

    损失 = 每个 variant 的 BCE 之和 / 该 variant 本片段的 (事件, 滞后) 数，各 variant 互不影响（参数不共享，分别裁剪）。
    返回 {variant: {"loss_sum", "pairs", "loss_by_lag", "pairs_by_lag"}} 与 chunks、updates。
    """
    k, n_streams, lags = int(cfg["lag_chunk"]), int(cfg["lag_streams"]), int(cfg["lags"])
    if plan is None:
        plan = V5.plan_streams(lengths, n_streams, k, rng, net_cfg.get("stream_layout", "greedy"))
    item_ends = {(index, first + n) for _, items in plan for index, first, n in map(V5.plan_item, items)}
    n_chunks = int(math.ceil(V5.stream_total_windows(plan) / float(k)))
    if max_chunks is not None:
        n_chunks = min(n_chunks, int(max_chunks))
    dtype = next(net.parameters()).dtype
    sources = V5.StreamSources(dataset, net_cfg, q99, device, dtype)
    weight = torch.tensor([pos_weight], dtype=dtype, device=device)
    clip = float(cfg["lag_grad_clip"])
    variants = list(readouts)
    stats = {v: {"loss_sum": 0.0, "pairs": 0, "loss_by_lag": [0.0] * lags, "pairs_by_lag": [0] * lags}
             for v in variants}
    buffer, states, updates = LagBuffer(lags), None, 0
    for r in readouts.values():
        r.train()
    for c in range(n_chunks):
        start = c * k
        pieces = V5.stream_pieces(plan, start, start + k)
        x, events, labels, keep = V5.assemble_streams(sources, pieces, k, net_cfg, device, dtype)
        with torch.no_grad():
            logits, states, info = net.forward_chunk(x, events, states, keep=keep, collect=True)
            maps = lag_maps(info["u_pre"], variants)
            if labels.numel():
                ends = item_end_table(plan, start, k, device)
                t, b = events["t"], events["b"]
                buffer.add(g=start + t, end=ends[t, b], b=b, y=events["y"], x=events["x"], p=events["p"],
                           t_local=events["t_local"], z0=logits,
                           birth=birth_features(maps, t * len(plan) + b, events["y"], events["x"]),
                           label=labels, key=torch.zeros_like(t))
        groups = lag_logits(readouts, buffer, maps, start, k, len(plan))
        if groups:
            total = {v: 0.0 for v in variants}
            count = {v: 0 for v in variants}
            for v, d, rows, logit in groups:
                loss = F.binary_cross_entropy_with_logits(logit, rows["label"], pos_weight=weight, reduction="sum")
                total[v] = total[v] + loss
                count[v] += int(rows["label"].numel())
                stats[v]["loss_by_lag"][d - 1] += float(loss.detach())
                stats[v]["pairs_by_lag"][d - 1] += int(rows["label"].numel())
            objective = sum(total[v] / float(count[v]) for v in variants if count[v])
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            for v in variants:
                if count[v]:
                    torch.nn.utils.clip_grad_norm_(readouts[v].parameters(), clip)
                    value = float(total[v].detach())
                    if not math.isfinite(value):
                        raise RuntimeError("损失出现非有限值: 片段 %d, variant %s" % (c, v))
                    stats[v]["loss_sum"] += value
                    stats[v]["pairs"] += count[v]
            optimizer.step()
            updates += 1
        buffer.prune(start + k)
        for stream in pieces:
            for index, lo, hi, _, _ in stream:
                if (index, hi) in item_ends:
                    sources.release(index)
    return {"variants": stats, "chunks": n_chunks, "updates": updates}


# ---------------------------------------------------------------------------
# 推理：一条序列，逐窗（step，与 V5 正式评估同一路径）或逐片段（layer，训练中验证用）
# ---------------------------------------------------------------------------


def window_of_events(seq):
    """每个事件（文件原始顺序）所在的窗号。"""
    win = np.empty(seq.n_events, dtype=np.int64)
    win[seq.order] = np.repeat(np.arange(seq.n_windows, dtype=np.int64), np.diff(seq.bounds))
    return win


def evaluate_sequence(net, readouts, seq, cfg, net_cfg, q99, device, execution):
    """推理一条序列的全部窗口。返回 (probs {variant: [D+1, N] float32 numpy}, win [N], avail [N])。

    probs[v][0] 为 V5 读出头的概率（与 train_stream_v1.run_sequence 同一计算）；probs[v][d] 为第 d 个滞后的概率；
    录像末尾读不到的滞后用最后可读年龄 avail = min(D, n_windows-1-win) 的概率填充。调用方负责 torch.no_grad()。
    """
    lags = int(cfg["lags"])
    variants = list(readouts)
    dtype = next(net.parameters()).dtype
    source = make_window_source(seq, net_cfg, q99, device, dtype)
    n, n_windows = seq.n_events, seq.n_windows
    # 结果放在 CPU 的 numpy 数组里逐窗写入（避免设备端按下标赋值在确定性模式下报错）
    probs = {v: np.full((lags + 1, n), np.nan, dtype=np.float32) for v in variants}
    buffer, states = LagBuffer(lags), None
    if execution == "step":
        spans = [(k, k + 1) for k in range(n_windows)]
    else:
        size = int(cfg["lag_eval_chunk"])
        spans = [(s, min(s + size, n_windows)) for s in range(0, n_windows, size)]
    for start, end in spans:
        if execution == "step":
            x, events, labels, idx = source.window(start)
            logits, states, info = net(x, events, states, collect=True)
            events = dict(events, t=torch.zeros_like(events["b"]))
        else:
            x, events, labels, idx = source.chunk(start, end)
            logits, states, info = net.forward_chunk(x, events, states, collect=True)
        maps = lag_maps(info["u_pre"], variants)
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size:
            p0 = torch.sigmoid(logits).float().cpu().numpy()       # 与 train_stream_v1.run_sequence 相同的计算
            for v in variants:
                probs[v][0, idx] = p0
            t = events["t"]
            buffer.add(g=start + t, end=torch.full_like(t, n_windows), b=events["b"], y=events["y"], x=events["x"],
                       p=events["p"], t_local=events["t_local"], z0=logits,
                       birth=birth_features(maps, t + events["b"], events["y"], events["x"]),
                       label=labels, key=torch.from_numpy(idx).to(device))
        for v, d, rows, logit in lag_logits(readouts, buffer, maps, start, end - start, 1):
            probs[v][d, rows["key"].cpu().numpy()] = torch.sigmoid(logit).float().cpu().numpy()
        buffer.prune(end)
    win = window_of_events(seq)
    avail = np.minimum(lags, n_windows - 1 - win)
    out = {}
    for v in variants:
        p = probs[v]
        for d in range(1, lags + 1):
            late = avail < d
            p[d, late] = p[avail[late], np.flatnonzero(late)]
        if np.isnan(p).any():
            raise RuntimeError("序列 %s 有未填充的滞后概率（%s）" % (seq.name, v))
        out[v] = p
    return out, win, avail


def quick_lag_iou(net, readouts, dataset, cfg, net_cfg, q99, device):
    """训练中验证：逐片段执行跑完整划分，返回 {variant: [IoU(d) for d = 0..D]}（全局拼接口径，阈值 0.9）。"""
    lags, thr = int(cfg["lags"]), float(net_cfg["threshold"])
    counts = {v: np.zeros((lags + 1, 3), dtype=np.int64) for v in readouts}
    for r in readouts.values():
        r.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            seq = dataset[i]
            probs, _, _ = evaluate_sequence(net, readouts, seq, cfg, net_cfg, q99, device, "layer")
            for v, p in probs.items():
                for d in range(lags + 1):
                    counts[v][d] += iou_acc_counts(p[d], seq.label, thr)[2:]
    for r in readouts.values():
        r.train()
    return {v: [float(c[0]) / float(max(c.sum(), 1)) for c in counts[v]] for v in readouts}


# ---------------------------------------------------------------------------
# 模式
# ---------------------------------------------------------------------------


def save_readout(path, readout, epoch, score, cfg):
    torch.save({"model": readout.state_dict(), "variant": readout.variant, "lags": readout.lags,
                "epoch": epoch, "score": score, "config": cfg,
                "backbone_sha256": file_sha256(cfg["backbone_checkpoint"])}, str(path))


def mode_smoke(args, cfg):
    """2 个训练片段 + 第 1 条验证序列（逐窗 / 逐片段两种执行比对）。"""
    device = torch.device(args.device)
    net, net_cfg, q99, pos_weight = load_backbone(cfg, device, "layer", "gpu")
    dtype = next(net.parameters()).dtype
    readouts = make_readouts(cfg, net_cfg, device, dtype)
    optimizer = torch.optim.Adam([p for r in readouts.values() for p in r.parameters()], lr=float(cfg["lag_lr"]))
    names = sorted(n for n in os.listdir(os.path.join(cfg["root"], "train")) if n.endswith(".npz"))
    train_set = EvUAVStream(cfg["root"], "train", net_cfg, names[:int(cfg["lag_streams"])])
    lengths = [int(train_set[i].n_windows) for i in range(len(train_set))]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    out = train_epoch(net, readouts, train_set, lengths, optimizer, cfg, net_cfg, q99, pos_weight, device,
                      np.random.RandomState(int(cfg["seed"])), max_chunks=2)
    train_seconds = time.perf_counter() - t0
    frozen = [name for name, p in net.named_parameters() if p.grad is not None]
    if frozen:
        raise RuntimeError("骨干参数出现了梯度: %s" % frozen[:5])
    val = EvUAVStream(cfg["root"], "val", net_cfg)
    seq = val[0]
    for r in readouts.values():
        r.eval()
    with torch.no_grad():
        t1 = time.perf_counter()
        step, _, _ = evaluate_sequence(net, readouts, seq, cfg, dict(net_cfg, input_device="cpu"), q99, device, "step")
        step_seconds = time.perf_counter() - t1
        layer, _, _ = evaluate_sequence(net, readouts, seq, cfg, net_cfg, q99, device, "layer")
    # 两种执行数学等价，但 float32 下卷积的舍入不同；BIR 的量化 round(M/theta) 在 .5 边界附近会偶尔翻一级，
    # 个别事件的概率因此可以差得较多。只用于训练中选模型，所以按"阈值 0.9 下的判定与 IoU"核对，不按最大差值
    thr = float(net_cfg["threshold"])
    gap = {}
    for v in readouts:
        diff = np.abs(step[v] - layer[v])
        iou_step = [iou_acc_counts(step[v][d], seq.label, thr)[0] for d in range(step[v].shape[0])]
        iou_layer = [iou_acc_counts(layer[v][d], seq.label, thr)[0] for d in range(layer[v].shape[0])]
        gap[v] = {"max": float(diff.max()), "p9999": float(np.percentile(diff, 99.99)),
                  "frac_over_1e-2": float((diff > 1e-2).mean()),
                  "decision_flip_frac": float(((step[v] >= np.float32(thr)) != (layer[v] >= np.float32(thr))).mean()),
                  "iou_step": iou_step, "iou_layer": iou_layer,
                  "iou_max_diff": float(np.nanmax(np.abs(np.subtract(iou_step, iou_layer))))}
    report = {"train": out, "train_seconds": train_seconds, "step_eval_seconds": step_seconds,
              "step_ms_per_window": 1e3 * step_seconds / max(seq.n_windows, 1),
              "step_vs_layer": gap, "peak_memory_gib": V5.peak_memory_gib(device),
              "readout_parameters": readout_parameters(readouts), "val_sequence": seq.name}
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    bad = {v: g for v, g in gap.items() if not (g["iou_max_diff"] <= 2e-3 and g["decision_flip_frac"] <= 1e-3)}
    if bad:
        raise RuntimeError("逐窗与逐片段两种执行的判定差异过大: %s" % bad)
    print("SMOKE TEST FINISHED", flush=True)


def mode_train(args, cfg):
    device = torch.device(args.device)
    net, net_cfg, q99, pos_weight = load_backbone(cfg, device, "layer", "gpu")
    dtype = next(net.parameters()).dtype
    readouts = make_readouts(cfg, net_cfg, device, dtype)
    root = Path(cfg["lag_save_root"])
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics.jsonl"
    if metrics_path.exists():
        raise RuntimeError("输出目录已有 metrics.jsonl，请换目录: %s" % root)
    V5.write_json(root / "run_config_train.json", {
        "config": cfg, "net_config": net_cfg, "pos_weight": pos_weight,
        "readout_parameters": readout_parameters(readouts),
        "backbone_sha256": file_sha256(cfg["backbone_checkpoint"]), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"})
    optimizer = torch.optim.Adam([p for r in readouts.values() for p in r.parameters()], lr=float(cfg["lag_lr"]))
    train_set = EvUAVStream(cfg["root"], "train", net_cfg)
    val_set = EvUAVStream(cfg["root"], "val", net_cfg)
    lengths = [int(train_set[i].n_windows) for i in range(len(train_set))]
    seed, epochs, lags = int(cfg["seed"]), int(cfg["lag_epochs"]), int(cfg["lags"])
    print("V6 滞后读出头：骨干 %s（冻结），variant %s，D = %d，读出头参数 %s | 训练集 %d 条录像 %d 窗，%d 条流、片段 %d 窗" % (
        cfg["backbone_checkpoint"], " ".join(readouts), lags, readout_parameters(readouts), len(lengths),
        sum(lengths), int(cfg["lag_streams"]), int(cfg["lag_chunk"])), flush=True)
    best = {v: -float("inf") for v in readouts}
    for epoch in range(epochs):
        lr = linear_epoch_lr(epoch, epochs, float(cfg["lag_lr"]), float(cfg["lag_lr_end"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        out = train_epoch(net, readouts, train_set, lengths, optimizer, cfg, net_cfg, q99, pos_weight, device,
                          np.random.RandomState(seed * 1000 + epoch + 7))
        train_seconds = time.perf_counter() - t0
        val = quick_lag_iou(net, readouts, val_set, cfg, net_cfg, q99, device)
        record = {"epoch": epoch, "lr": lr, "train_seconds": train_seconds,
                  "epoch_seconds": time.perf_counter() - t0, "updates": out["updates"],
                  "peak_memory_gib": V5.peak_memory_gib(device), "val_iou": val, "variants": {}}
        for v, s in out["variants"].items():
            score = float(np.mean(val[v][1:]))
            record["variants"][v] = {
                "loss_per_pair": s["loss_sum"] / max(s["pairs"], 1),
                "loss_by_lag": [a / max(b, 1) for a, b in zip(s["loss_by_lag"], s["pairs_by_lag"])],
                "val_score": score}
            if score > best[v]:
                best[v] = score
                save_readout(root / ("best_lag_%s.pt" % v), readouts[v], epoch, score, cfg)
            save_readout(root / ("last_lag_%s.pt" % v), readouts[v], epoch, score, cfg)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print("epoch %d lr %.2e | %s | 更新 %d 次 | %.0fs（训练 %.0fs）| 显存峰值 %.2f GiB" % (
            epoch, lr, " | ".join("%s loss %.4f val IoU d0..%d %s" % (
                v, record["variants"][v]["loss_per_pair"], lags, " ".join("%.4f" % u for u in val[v]))
                for v in readouts), out["updates"], record["epoch_seconds"], train_seconds,
            record["peak_memory_gib"] or 0.0), flush=True)
    V5.write_json(root / "summary.json", {"best_val_score": best, "epochs": epochs})
    print("TRAINING FINISHED:", root, flush=True)


def mode_eval(args, cfg):
    """逐窗执行（与 V5 正式评估同一路径）跑完整划分，报告每个 variant、每个固定滞后的四个指标与机制诊断。"""
    device = torch.device(args.device)
    net, net_cfg, q99, _ = load_backbone(cfg, device, "step", "cpu")
    dtype = next(net.parameters()).dtype
    root = Path(cfg["lag_save_root"])
    out_path = root / ("eval_lag_%s.json" % args.split)
    if out_path.exists() and not args.overwrite:
        raise RuntimeError("结果已存在: %s（使用 --overwrite 覆盖）" % out_path)
    readouts, epochs = make_readouts(cfg, net_cfg, device, dtype), {}
    for v, r in readouts.items():
        ckpt = torch.load(str(root / ("best_lag_%s.pt" % v)), map_location=device)
        if ckpt["backbone_sha256"] != file_sha256(cfg["backbone_checkpoint"]):
            raise RuntimeError("读出头训练时的骨干与当前 backbone_checkpoint 不是同一个文件")
        r.load_state_dict(ckpt["model"])
        r.eval()
        epochs[v] = int(ckpt["epoch"])
    dataset = EvUAVStream(cfg["root"], args.split, net_cfg)
    lags = int(cfg["lags"])
    seqs, probs = [], {v: [] for v in readouts}
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(len(dataset)):
            seq = dataset[i]
            p, win, avail = evaluate_sequence(net, readouts, seq, cfg, net_cfg, q99, device, "step")
            seqs.append({"name": seq.name, "t": seq.t, "x": seq.x, "y": seq.y, "label": seq.label,
                         "target_id": seq.target_id, "win": win, "avail": avail})
            for v in readouts:
                probs[v].append(p[v])
    seconds = time.perf_counter() - t0
    first = next(iter(readouts))

    def report(v, d):
        return split_report(seqs, lambda i: probs[v][i][d], lambda i: np.minimum(d, seqs[i]["avail"]), net_cfg)

    results = {"split": args.split, "backbone_checkpoint": cfg["backbone_checkpoint"], "readout_epochs": epochs,
               "lags": lags, "threshold": float(net_cfg["threshold"]),
               "seconds": seconds, "d0": report(first, 0), "variants": {}}
    for v in readouts:
        results["variants"][v] = {str(d): report(v, d) for d in range(1, lags + 1)}
    # d = 0 的 IoU/ACC 与原 utils/eval.py 核对（同一份概率）
    from types import SimpleNamespace
    from utils.eval import evalute
    evaluator = evalute(SimpleNamespace(roc=False, pd_detT=net_cfg["pd_detT"], correct_thresh=net_cfg["correct_thresh"]))
    for i, s in enumerate(seqs):
        evaluator.matches[str(i)] = {"seg_pred": torch.from_numpy(probs[first][i][0]),
                                     "seg_gt": torch.from_numpy(s["label"])}
    iou, acc = evaluator.evaluate_iou_and_accuracy(thresh=float(net_cfg["threshold"]))
    results["d0_utils_eval"] = {"iou": float(iou), "acc": float(acc)}
    if (iou, acc) != (results["d0"]["iou"], results["d0"]["acc"]):
        raise RuntimeError("d=0 的 IoU/ACC 与 utils/eval.py 不一致: %s vs %s" % ((iou, acc), results["d0"]))
    if args.reference:
        ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))["carry"]
        check = {k: (results["d0"][k], ref[k]) for k in ("iou", "acc", "pd", "fa")}
        results["d0_vs_v5"] = {k: {"lag": a, "v5": b, "equal": a == b} for k, (a, b) in check.items()}
        print("[P0] d=0 与 V5 评估：%s" % {k: ("相同" if a == b else "不同 %r vs %r" % (a, b))
                                          for k, (a, b) in check.items()}, flush=True)
    V5.write_json(out_path, results)
    fmt = "  d=%d  IoU %.4f ACC %.4f Pd %.4f Fa %.3e | 误检中目标旁 %.1f%%"
    for v in readouts:
        print("[%s] %s" % (v, args.split), flush=True)
        for d in range(lags + 1):
            r = results["d0"] if d == 0 else results["variants"][v][str(d)]
            print(fmt % (d, r["iou"], r["acc"], r["pd"], r["fa"], 100.0 * r["fp_near_target_fraction"]), flush=True)
    if args.dump:
        offsets = np.cumsum([0] + [s["t"].shape[0] for s in seqs])
        meta = {"names": [s["name"] for s in seqs], "offsets": offsets.tolist(), "lags": lags,
                "split": args.split, "threshold": float(net_cfg["threshold"]),
                "pd_detT": net_cfg["pd_detT"], "correct_thresh": net_cfg["correct_thresh"],
                "height": net_cfg["height"], "width_px": net_cfg["width_px"], "window_ms": net_cfg["window_ms"],
                "backbone_checkpoint": cfg["backbone_checkpoint"]}
        cat = lambda key, dt: np.concatenate([s[key] for s in seqs]).astype(dt)  # noqa: E731
        # 整数时间戳（毫秒）与坐标都在这些类型的范围内；目标号是小整数，float32 精确
        common = dict(label=cat("label", np.int8), target_id=cat("target_id", np.float32), t=cat("t", np.int32),
                      x=cat("x", np.int16), y=cat("y", np.int16), win=cat("win", np.int32),
                      avail=cat("avail", np.int8))
        for v in readouts:
            path = root / ("dump_%s_%s.npz" % (args.split, v))
            np.savez(str(path), prob=np.concatenate(probs[v], axis=1).astype(np.float32),
                     meta=json.dumps(dict(meta, variant=v)), **common)
            print("导出:", path, flush=True)
    print("EVAL FINISHED:", out_path, flush=True)


def main():
    args = parse_args()
    cfg = build_config(args.config)
    {"smoke": mode_smoke, "train": mode_train, "eval": mode_eval}[args.mode](args, cfg)


if __name__ == "__main__":
    main()
