"""模块三：目标级航迹头（model/track_head.py）的训练与评估入口（Stream-SNN-Innovations.md 13.60～13.61）。

骨干（E3a 等已训练的流式网络）冻结、只读；只训练航迹头。每条录像从头到尾连续处理，骨干状态与航迹状态都跨片段延续。
模式：
    train   多流 TBPTT（与 train_stream_v1 的 trainer=streams 同一排法：B 条流、每段 k 窗、片段边界 detach），
            每轮在 val 上同时评估骨干原输出（base）与航迹头输出（track），按 track 的 val IoU 挑最好的一轮
    eval    在 val / test 上评估：四个指标（IoU / ACC / Pd / Fa，与正式评估同一口径）、误差类别（tools/error_budget.py），
            以及事先定好的关口是否通过；可选保存逐事件预测（--dump-dir）与逐窗航迹（--tracks-dir，可视化用）
decision_lag = 1 时：track = 一窗滞后的最终输出（最后一窗没有下一窗，用零等待输出），track_now = 零等待的暂定输出；
    训练损失 = now_loss_weight x 零等待 BCE + lag_loss_weight x 滞后 BCE。
骨干一律用逐层执行（execution=layer）：与逐窗执行数学等价（单元测试已核对），这里只为速度；base 一列即用来核对复现。
用法见 configs/track_head_v2.yaml 文件头。
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")   # 必须在导入 torch 之前设置

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_stream_v1 as T  # noqa: E402
from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from dataset.stream_source import make_window_source  # noqa: E402
from dataset.stream_windows import refill_by_index  # noqa: E402
from model.track_head import TrackHead  # noqa: E402
from utils.stream_common import linear_epoch_lr, load_flat_config  # noqa: E402
from utils.stream_metrics import FastRoc  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import error_budget  # noqa: E402

HEAD_KEYS = ("cell", "slots", "hidden", "history", "gate_sigma", "gate_max_px", "gate_floor", "birth_floor",
             "max_births", "death_exist", "max_coast", "match_min_px", "corr_hidden", "r_min", "recycle_exist",
             "decision_lag", "context", "existence")


# ---------------------------------------------------------------------------
# 骨干与航迹头
# ---------------------------------------------------------------------------


class ReadoutTap(object):
    """截取骨干读出层的输入（逐事件 dec1 特征），不改动骨干代码与输出。"""

    def __init__(self, net):
        self.feat = None
        original = net._readout

        def tapped(feat, events):
            self.feat = feat
            return original(feat, events)

        net._readout = tapped


def load_base(cfg, device):
    """读取冻结的骨干。网络结构、q99、pos_weight 取自骨干 checkpoint；数据根目录与评测口径取自本配置。"""
    ckpt = torch.load(cfg["base_checkpoint"], map_location=device)
    base = T.eval_train_config(ckpt["config"], cfg)
    base["execution"], base["input_device"] = "layer", cfg.get("input_device", "gpu")
    net = T.build_net(base, device)
    net.load_state_dict(ckpt["model"])
    if not bool(net.gain_calibrated):
        raise RuntimeError("骨干 checkpoint 的增益未校准")
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    stats = ckpt["stats"]
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    pos_weight = min(float(stats["train_neg"]) / float(stats["train_pos"]), float(base["pos_weight_cap"]))
    return net, ReadoutTap(net), base, q99, pos_weight, ckpt


def build_head(cfg, net, base, device):
    feat_dim = int(net.readout[0].in_features) - 2                 # 读出层输入 = [dec1 特征, p, t_local]
    kwargs = {key: cfg[key] for key in HEAD_KEYS if key in cfg}
    return TrackHead(feat_dim, int(base["pad_height"]), int(base["pad_width"]), **kwargs).to(device)


# ---------------------------------------------------------------------------
# 多流训练
# ---------------------------------------------------------------------------


class BaseOutputs(object):
    """冻结骨干的逐事件输出缓存：每条录像从头到尾连续推理一次（逐层执行、eval_chunk 窗一段、状态跨段延续），
    把 z0 与 dec1 特征按事件在文件中的下标存起来。

    多流训练里每条录像都从自己的开头起算（keep 在录像开头清零），状态跨片段延续，所以骨干对一条录像的输出与流怎么排、
    开头延迟几窗无关：算一次，各轮复用。这样每轮只剩航迹头的计算（骨干与模块二的背景估计不再重复）。
    """

    def __init__(self, net, tap, base, q99, device, dataset, keep_all=True):
        self.net, self.tap, self.base, self.q99, self.device = net, tap, base, q99, device
        self.dataset, self.keep_all = dataset, keep_all
        self.items = {}
        self.dtype = next(net.parameters()).dtype

    def get(self, index):
        if index in self.items:
            return self.items[index]
        seq = self.dataset[index]
        source = make_window_source(seq, self.base, self.q99, self.device, self.dtype)
        z0_all = torch.zeros(seq.n_events, dtype=torch.float32)
        feat_all = None
        chunk = int(self.base.get("eval_chunk", 32))
        states = None
        with torch.no_grad():
            for start in range(0, seq.n_windows, chunk):
                end = min(start + chunk, seq.n_windows)
                x, events, _, idx = source.chunk(start, end)
                z0, states, _ = self.net.forward_chunk(x, events, states)
                feat = self.tap.feat.float().cpu()
                if feat_all is None:
                    feat_all = torch.zeros((seq.n_events, int(feat.shape[1])), dtype=torch.float32)
                index_t = torch.from_numpy(np.asarray(idx, dtype=np.int64))
                z0_all[index_t] = z0.float().cpu()
                feat_all[index_t] = feat
        if feat_all is None:
            feat_all = torch.zeros((seq.n_events, int(self.net.readout[0].in_features) - 2), dtype=torch.float32)
        item = (seq, z0_all, feat_all)
        if self.keep_all:
            self.items[index] = item
        return item


def chunk_events(seq, lo, hi, offset=0, stream=0):
    """录像第 lo..hi-1 窗的事件（与 TorchWindowSource.chunk 同一顺序：先按窗、窗内按时间稳定排序），
    返回 numpy 字典 t（片段内窗号 + offset）/ b / x / y / p / t_local / label / tid，以及事件在文件中的下标。"""
    a, b = int(seq.bounds[lo]), int(seq.bounds[hi])
    idx = seq.order[a:b]
    window = np.repeat(np.arange(lo, hi, dtype=np.int64), np.diff(seq.bounds[lo:hi + 1]))
    return {"t": window - lo + int(offset), "b": np.full(b - a, stream, dtype=np.int64),
            "x": seq.x[idx].astype(np.int64), "y": seq.y[idx].astype(np.int64),
            "p": seq.p[idx].astype(np.float32), "t_local": seq.t_local[idx].astype(np.float32),
            "label": seq.label[idx].astype(np.float32), "tid": seq.target_id[idx].astype(np.float32)}, idx


def to_device(parts, device):
    events = {key: torch.from_numpy(np.ascontiguousarray(parts[key])).to(device)
              for key in ("t", "b", "x", "y", "p", "t_local")}
    return events, torch.from_numpy(parts["label"]).to(device), torch.from_numpy(parts["tid"]).to(device)


def assemble_cached(outputs, pieces, steps, device):
    """多流片段：拼接各流的事件与缓存的骨干输出。返回 (events, z0, feat, labels, keep [T, B], tid)。"""
    batch = len(pieces)
    keep = torch.ones((steps, batch), dtype=torch.float32, device=device)
    cols = {key: [] for key in ("t", "b", "x", "y", "p", "t_local", "label", "tid")}
    z0s, feats = [], []
    for b, stream in enumerate(pieces):
        for index, lo, hi, offset, new_start in stream:
            seq, z0_all, feat_all = outputs.get(index)
            part, idx = chunk_events(seq, lo, hi, offset, b)
            for key in cols:
                cols[key].append(part[key])
            index_t = torch.from_numpy(np.asarray(idx, dtype=np.int64))
            z0s.append(z0_all[index_t])
            feats.append(feat_all[index_t])
            if new_start:
                keep[offset, b] = 0
    if not z0s:
        empty = {key: np.zeros(0, dtype=np.int64 if key in ("t", "b", "x", "y") else np.float32) for key in cols}
        events, labels, tid = to_device(empty, device)
        return events, torch.zeros(0, device=device), torch.zeros(0, 1, device=device), labels, keep, tid
    events, labels, tid = to_device({key: np.concatenate(v) for key, v in cols.items()}, device)
    return events, torch.cat(z0s).to(device), torch.cat(feats).to(device), labels, keep, tid


def train_epoch(outputs, head, lengths, optimizer, cfg, pos_weight, device, rng, max_chunks=None):
    k, n_streams = int(cfg["tbptt_k"]), int(cfg["streams"])
    if cfg.get("stream_layout", "greedy") != "greedy":
        raise ValueError("骨干输出缓存要求每条录像整段放进一条流（stream_layout = greedy）")
    plan = T.plan_streams(lengths, n_streams, k, rng, "greedy")
    n_chunks = int(math.ceil(T.stream_total_windows(plan) / float(k)))
    if max_chunks is not None:
        n_chunks = min(n_chunks, int(max_chunks))
    weight = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    lam_e, lam_p, clip = float(cfg["lambda_exist"]), float(cfg["lambda_pos"]), float(cfg["grad_clip"])
    lam_d, lam_o = float(cfg.get("lambda_pd", 1.0)), float(cfg.get("lambda_origin", 1.0))   # 只在 existence = ipda 时有项
    w_now, w_lag = float(cfg.get("now_loss_weight", 1.0)), float(cfg.get("lag_loss_weight", 1.0))
    lag = int(cfg.get("decision_lag", 0)) == 1
    sums = {"event": 0.0, "events": 0, "lag": 0.0, "lag_n": 0, "exist": 0.0, "exist_n": 0, "pos": 0.0, "pos_n": 0,
            "pd": 0.0, "pd_n": 0, "org": 0.0, "org_n": 0,
            "updates": 0, "births": 0, "recycled": 0, "live": 0, "slot_windows": 0}
    norms = []
    hstate = None
    head.train()
    for c in range(n_chunks):
        pieces = T.stream_pieces(plan, c * k, (c + 1) * k)
        events, z0, feat, labels, keep, tid = assemble_cached(outputs, pieces, k, device)
        z, hstate, aux = head.forward_chunk(events, z0, feat, hstate, k, keep=keep, labels=labels, tid=tid,
                                            keys=labels if lag else None)
        for key in ("births", "recycled", "live", "slot_windows"):
            sums[key] += aux[key]
        if labels.numel():
            n_ev = int(labels.numel())
            ev_loss = F.binary_cross_entropy_with_logits(z, labels, pos_weight=weight, reduction="sum") / n_ev
            loss = w_now * ev_loss if lag else ev_loss
            if lag and aux["delayed_z"] is not None:
                n_lag = int(aux["delayed_z"].shape[0])
                lag_loss = F.binary_cross_entropy_with_logits(aux["delayed_z"], aux["delayed_key"], pos_weight=weight,
                                                              reduction="sum") / n_lag
                loss = loss + w_lag * lag_loss
                sums["lag"] += float(lag_loss.detach()) * n_lag
                sums["lag_n"] += n_lag
            if aux["exist_n"]:
                loss = loss + lam_e * aux["exist_sum"] / aux["exist_n"]
            if aux["pos_n"]:
                loss = loss + lam_p * aux["pos_sum"] / aux["pos_n"]
            if aux["pd_n"]:
                loss = loss + lam_d * aux["pd_sum"] / aux["pd_n"]
            if aux["org_n"]:
                loss = loss + lam_o * aux["org_sum"] / aux["org_n"]
            if not math.isfinite(float(loss.detach())):
                raise RuntimeError("损失出现非有限值：片段 %d" % c)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norms.append(float(torch.nn.utils.clip_grad_norm_(head.parameters(), clip)))
            optimizer.step()
            sums["event"] += float(ev_loss.detach()) * n_ev
            sums["events"] += n_ev
            sums["exist"] += float(aux["exist_sum"].detach())
            sums["exist_n"] += aux["exist_n"]
            sums["pos"] += float(aux["pos_sum"].detach())
            sums["pos_n"] += aux["pos_n"]
            for key in ("pd", "org"):
                sums[key] += float(aux[key + "_sum"].detach())
                sums[key + "_n"] += aux[key + "_n"]
            sums["updates"] += 1
        hstate = head.detach_state(hstate)
    windows = max(sums["slot_windows"], 1)
    return {"event_loss": sums["event"] / max(sums["events"], 1),
            "lag_loss": sums["lag"] / max(sums["lag_n"], 1) if lag else None,
            "exist_loss": sums["exist"] / max(sums["exist_n"], 1),
            "pos_loss": sums["pos"] / max(sums["pos_n"], 1),
            "pd_loss": sums["pd"] / sums["pd_n"] if sums["pd_n"] else None,
            "origin_loss": sums["org"] / sums["org_n"] if sums["org_n"] else None,
            "updates": sums["updates"], "chunks": n_chunks,
            "grad_norm_median": float(np.median(norms)) if norms else 0.0,
            "births_per_window": sums["births"] / windows, "recycled_per_window": sums["recycled"] / windows,
            "live_per_window": sums["live"] / windows}


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------


def run_split(outputs, head, base, device, split, full=False, dump_dir=None, tracks_dir=None, cache=None):
    """整段连续推理一个划分；返回 base（骨干原输出）、track（航迹头最终输出）的指标，滞后判决时另有 track_now（零等待）。

    cache（训练中每轮验证用的字典，跨轮保留）：骨干冻结，base 的计数与误差类别每轮都一样，只在第一轮算一次；
    给出 cache 时 track_now 只算 IoU、不做误差分类（省时间）。正式评估（mode_eval）不传 cache，全部都算。
    """
    dataset = outputs.dataset
    thr = float(base["threshold"])
    chunk = int(base.get("eval_chunk", 32))
    lag = head.decision_lag == 1
    kinds = ("base", "track", "track_now") if lag else ("base", "track")
    counts = {kind: np.zeros(3, dtype=np.int64) for kind in kinds}                   # TP, FP, FN
    rocs = {kind: FastRoc(base["pd_detT"], base["correct_thresh"], base["height"], base["width_px"])
            for kind in kinds} if full else None
    rows = {kind: [] for kind in kinds}
    stats = {"births": 0, "recycled": 0, "live": 0, "slot_windows": 0}
    was_training = head.training
    head.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            seq, z0_all, feat_all = outputs.get(i)
            hstate = head.init_state(1, device)
            idx_parts, p0_parts, p1_parts, lag_key, lag_prob = [], [], [], [], []
            log = [] if tracks_dir else None
            for start in range(0, seq.n_windows, chunk):
                end = min(start + chunk, seq.n_windows)
                part, idx = chunk_events(seq, start, end)
                events, _, _ = to_device(part, device)
                index_t = torch.from_numpy(np.asarray(idx, dtype=np.int64))
                z0, feat = z0_all[index_t].to(device), feat_all[index_t].to(device)
                keys = index_t.to(device) if lag else None
                z, hstate, aux = head.forward_chunk(events, z0, feat, hstate, end - start, log=log, keys=keys)
                for key in stats:
                    stats[key] += aux[key]
                idx_parts.append(idx)
                p0_parts.append(torch.sigmoid(z0).float().cpu().numpy())
                p1_parts.append(torch.sigmoid(z).float().cpu().numpy())
                if lag and aux["delayed_z"] is not None:
                    lag_key.append(aux["delayed_key"].cpu().numpy().astype(np.int64))
                    lag_prob.append(torch.sigmoid(aux["delayed_z"]).float().cpu().numpy())
            probs = {"base": refill_by_index(seq.n_events, idx_parts, p0_parts),
                     "track": refill_by_index(seq.n_events, idx_parts, p1_parts)}
            if lag:
                probs["track_now"] = probs["track"].copy()
                if lag_key:                                   # 最后一窗没有下一窗：保留零等待输出
                    probs["track"][np.concatenate(lag_key)] = np.concatenate(lag_prob)
            pos = seq.label > 0.5
            for kind in kinds:
                prob = probs[kind]
                cached = cache is not None and kind == "base" and seq.name in cache
                if cached:
                    count, row = cache[seq.name]
                else:
                    pred = prob >= thr
                    count = np.array([int((pred & pos).sum()), int((pred & ~pos).sum()), int((~pred & pos).sum())])
                    row = None
                    if cache is None or kind != "track_now":
                        row = error_budget.analyse_arrays(
                            seq.name, seq.x, seq.y, seq.t, pos, prob, seq.target_id, thr, 10, float(base["window_ms"]),
                            (1,), int(base["height"]), int(base["width_px"]))[0]
                    if cache is not None and kind == "base":
                        cache[seq.name] = (count, row)
                counts[kind] += count
                if full:
                    rocs[kind].update(seq.t, prob, seq.target_id, seq.label, seq.x, seq.y, thr)
                if row is not None:
                    rows[kind].append(row)
            if dump_dir:
                extra = {"now_probabilities": probs["track_now"].astype(np.float32)} if lag else {}
                np.savez(os.path.join(dump_dir, seq.name),
                         locs=np.stack([np.zeros(seq.n_events, dtype=np.int64), seq.x, seq.y, seq.t], 1),
                         labels=seq.label.astype(np.float32), probabilities=probs["track"].astype(np.float32),
                         base_probabilities=probs["base"].astype(np.float32),
                         target_id=seq.target_id.astype(np.float64), **extra)
            if tracks_dir:
                np.savez(os.path.join(tracks_dir, seq.name),
                         **{key: np.stack([w[key][0] for w in log])
                            for key in ("active", "pos", "vel", "ex", "pd", "beta")})
    head.train(was_training)
    windows = max(stats["slot_windows"], 1)
    out = {"split": split, "n_sequences": len(dataset),
           "head_stats": {"births_per_window": stats["births"] / windows,
                          "recycled_per_window": stats["recycled"] / windows,
                          "live_per_window": stats["live"] / windows}}
    for kind, (tp, fp, fn) in counts.items():
        res = {"iou": tp / max(tp + fp + fn, 1), "acc": tp / max(tp + fn, 1), "tp": int(tp), "fp": int(fp),
               "fn": int(fn)}
        if full:
            pd, fa = rocs[kind].result()
            res.update(pd=float(pd), fa=float(fa))
        if rows[kind]:
            res["fp_cat"] = {c: sum(r["fp_cat"][c] for r in rows[kind]) for c in ("near", "tail", "ahead", "far")}
            res["fn_cat"] = {c: sum(r["fn_cat"][c] for r in rows[kind]) for c in ("never", "same", "onset", "gap")}
        out[kind] = res
    return out


def gate_check(result, cfg):
    """事先定好的关口（13.60）：拖尾误报、断续漏检各降 >= gate_tail_drop / gate_gap_drop；IoU 不低于 base；
    目标附近的误报增加 <= gate_near_rise。"""
    b, t = result["base"], result["track"]

    def drop(before, after):
        return (before - after) / float(before) if before else 0.0

    checks = {
        "tail_fp_drop": drop(b["fp_cat"]["tail"], t["fp_cat"]["tail"]),
        "gap_fn_drop": drop(b["fn_cat"]["gap"], t["fn_cat"]["gap"]),
        "iou_delta": t["iou"] - b["iou"],
        "near_fp_rise": -drop(b["fp_cat"]["near"], t["fp_cat"]["near"]),
    }
    checks["pass"] = bool(checks["tail_fp_drop"] >= float(cfg["gate_tail_drop"])
                          and checks["gap_fn_drop"] >= float(cfg["gate_gap_drop"])
                          and checks["iou_delta"] >= 0.0
                          and checks["near_fp_rise"] <= float(cfg["gate_near_rise"]))
    return checks


# ---------------------------------------------------------------------------
# 模式
# ---------------------------------------------------------------------------


def mode_train(args, cfg):
    device = torch.device(args.device)
    T.seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    net, tap, base, q99, pos_weight, _ = load_base(cfg, device)
    head = build_head(cfg, net, base, device)
    root = Path(cfg["save_root"])
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics.jsonl"
    if metrics_path.exists():
        raise RuntimeError("输出目录已有 metrics.jsonl，请换目录")
    T.write_json(root / "run_config.json", cfg)
    optimizer = torch.optim.Adam(head.parameters(), lr=float(cfg["lr"]))
    train_set = EvUAVStream(base["root"], "train", base)
    lengths = [int(train_set[i].n_windows) for i in range(len(train_set))]
    train_out = BaseOutputs(net, tap, base, q99, device, train_set)
    val_out = BaseOutputs(net, tap, base, q99, device, EvUAVStream(base["root"], "val", base))
    seed, epochs = int(cfg["seed"]), int(cfg["epochs"])
    print("航迹头训练：骨干 %s（冻结）| 航迹头参数 %d | context %s | decision_lag %d | %d 条流、k=%d | "
          "训练集 %d 条录像 %d 窗 | pos_weight %.2f" % (
              cfg["base_checkpoint"], sum(p.numel() for p in head.parameters()), head.context, head.decision_lag,
              int(cfg["streams"]), int(cfg["tbptt_k"]), len(lengths), sum(lengths), pos_weight), flush=True)
    best, val_cache = -float("inf"), {}
    for epoch in range(epochs):
        lr = linear_epoch_lr(epoch, epochs, float(cfg["lr"]), float(cfg["lr_end"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        out = train_epoch(train_out, head, lengths, optimizer, cfg, pos_weight, device,
                          np.random.RandomState(seed * 1000 + epoch + 7), args.max_chunks)
        train_s = time.perf_counter() - t0
        val = run_split(val_out, head, base, device, "val", cache=val_cache)
        record = dict(epoch=epoch, lr=lr, train=out, val=val, gate=gate_check(val, cfg), train_seconds=train_s,
                      epoch_seconds=time.perf_counter() - t0, peak_memory_gib=T.peak_memory_gib(device))
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        state = {"head": head.state_dict(), "epoch": epoch, "val": val, "config": cfg}
        if val["track"]["iou"] > best:
            best = val["track"]["iou"]
            torch.save(dict(state, best_val_iou=best), str(root / ("best_val_iou_seed%d.pt" % seed)))
        torch.save(dict(state, best_val_iou=best), str(root / "last.pt"))
        b, t = val["base"], val["track"]
        now = (" | 零等待 IoU %.4f" % val["track_now"]["iou"]) if "track_now" in val else ""
        calib = "" if out["pd_loss"] is None else " 检测 %.4f 来源 %.4f" % (out["pd_loss"], out["origin_loss"] or 0.0)
        print("epoch %d lr %.2e | 损失 事件 %.4f%s 存在 %.4f 位置 %.4f%s | val IoU base %.4f -> track %.4f%s | "
              "拖尾误报 %d -> %d 断续漏检 %d -> %d 目标附近误报 %d -> %d | 每窗 航迹 %.2f 出生 %.3f 回收 %.4f | "
              "%.0fs（训练 %.0fs）| 显存峰值 %.2f GiB" % (
                  epoch, lr, out["event_loss"], (" 滞后 %.4f" % out["lag_loss"]) if out["lag_loss"] is not None else "",
                  out["exist_loss"], out["pos_loss"], calib, b["iou"], t["iou"], now,
                  b["fp_cat"]["tail"], t["fp_cat"]["tail"], b["fn_cat"]["gap"], t["fn_cat"]["gap"],
                  b["fp_cat"]["near"], t["fp_cat"]["near"], val["head_stats"]["live_per_window"],
                  val["head_stats"]["births_per_window"], val["head_stats"]["recycled_per_window"],
                  record["epoch_seconds"], train_s, record["peak_memory_gib"] or 0.0), flush=True)
    print("TRAINING FINISHED:", root, flush=True)


def mode_eval(args, cfg):
    device = torch.device(args.device)
    T.seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    net, tap, base, q99, _, _ = load_base(cfg, device)
    head = build_head(cfg, net, base, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    head.load_state_dict(ckpt["head"])
    for d in (args.dump_dir, args.tracks_dir):
        if d:
            os.makedirs(d, exist_ok=True)
    outputs = BaseOutputs(net, tap, base, q99, device, EvUAVStream(base["root"], args.split, base), keep_all=False)
    result = run_split(outputs, head, base, device, args.split, full=True, dump_dir=args.dump_dir,
                       tracks_dir=args.tracks_dir)
    result.update(checkpoint=args.checkpoint, epoch=ckpt.get("epoch"), base_checkpoint=cfg["base_checkpoint"],
                  gate=gate_check(result, cfg), context=head.context, decision_lag=head.decision_lag)
    name = Path(args.checkpoint).stem
    path = Path(args.checkpoint).parent / ("eval_%s_%s.json" % (args.split, name))
    T.write_json(path, result)
    for kind in ("base", "track", "track_now"):
        if kind in result:
            r = result[kind]
            print("%s %-9s IoU %.4f ACC %.4f Pd %.4f Fa %.3e | 拖尾误报 %d 断续漏检 %d 目标附近误报 %d" % (
                args.split, kind, r["iou"], r["acc"], r["pd"], r["fa"], r["fp_cat"]["tail"], r["fn_cat"]["gap"],
                r["fp_cat"]["near"]), flush=True)
    print("航迹统计：", json.dumps(result["head_stats"], ensure_ascii=False), flush=True)
    print("关口：", json.dumps(result["gate"], ensure_ascii=False), flush=True)
    print("结果写入", path, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("train", "eval"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", default=None, help="eval：航迹头 checkpoint")
    parser.add_argument("--dump-dir", default=None, help="eval：保存逐事件预测（含 base_probabilities）")
    parser.add_argument("--tracks-dir", default=None, help="eval：保存逐窗航迹（active / pos / vel / ex）")
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None, help="train：每轮最多几个片段（冒烟用）")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_flat_config(args.config)
    if args.save_root:
        cfg["save_root"] = args.save_root
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.mode == "train":
        mode_train(args, cfg)
    else:
        if not args.checkpoint:
            raise ValueError("eval 需要 --checkpoint")
        mode_eval(args, cfg)


if __name__ == "__main__":
    main()
