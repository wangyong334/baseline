"""全局端到端训练（模块一 + 二 + 三，v2；Stream-SNN-Innovations.md 13.67）：流式脉冲骨干与目标神经元层一起训练。

两条通路，一个损失：
    像素通路 = 流式脉冲 U-Net（模块一：BIR 神经元 + 每窗消息峰值约束）+ 不变输入接口（模块二 NISI），给每个事件的证据 z0；
    目标通路 = 目标 token 层（head_kind: tokens，model/object_tokens.py，v6 网络写法）：事件对"背景 token + 槽 token"的
        交叉注意力、GRU 校正、槽间自注意力、每个槽一个脉冲神经元（膜电位 = 存在对数几率，越过 Wald 界 +-A 发放）、
        学出的提议；读骨干的 z0、dec1 特征，背景 logit 另读模块二的本地背景率；
        head_kind: object 为 v5 目标神经元层（model/object_layer.py，手写滤波程序，只作对照）；
    损失 = 像素通路读出的逐事件 BCE（保持为校准的逐事件证据）+ 目标通路零等待 / 终判输出的 BCE（都按本片段事件总数归一）
        + 目标通路的监督（匈牙利匹配后的存在 BCE、归属交叉熵、位置 Smooth-L1）
        + 模块一的每窗消息峰值约束（CommBudget，乘子从骨干训练结束时的值接着走）。
两种起点：
    从头训练（配置给 backbone_config，SPRINT 整体模型）：按骨干配置新建网络 -> 增益校准 -> 骨干、杂波场、目标层一起训练；
        clutter_estimator = learned 时模块二的背景率由学出来的杂波场给出（model/clutter_field.py，泊松最大似然，
        损失另加 lambda_clutter x 杂波场的泊松负对数似然），不变接口与目标层都只读它；
    微调（配置给 base_checkpoint）：骨干从已训练的 checkpoint（如 E3a）初始化，学习率乘 backbone_lr_scale。
目标层从头训练（v6 只用标准初始化；v5 对照的期望事件数初值取训练集统计）。
多流 TBPTT 与 train_stream_v1 的 trainer = streams 相同（B 条流、每段 k 窗、片段边界 detach、录像开头 keep 清零），每个片段更新一次。
模式：train（每轮在 val 上评估：base = 像素通路读出，track_now = 零等待，track = 终判；按 track 的 val IoU 挑最好的一轮）；
      eval（val / test：四个指标、误差类别、关口、可选逐事件预测与逐窗目标槽）。
用法见 configs/full_model_v2.yaml 文件头。
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")   # 必须在导入 torch 之前设置

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_stream_v1 as T  # noqa: E402
import train_track_head as TH  # noqa: E402
from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from dataset.stream_source import make_window_source  # noqa: E402
from dataset.stream_windows import num_input_channels  # noqa: E402
from model.clutter_field import ClutterField, event_count_maps  # noqa: E402
from model.comod import REF_RATE  # noqa: E402
from model.lif2d_stream import detach_states  # noqa: E402
from utils.comm_budget import CommBudget  # noqa: E402
from utils.stream_common import linear_epoch_lr, load_flat_config  # noqa: E402
from utils.train_numerics import clip_grad_norm_checked_  # noqa: E402


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


OVERRIDE_KEYS = ("root", "seed", "deterministic", "input_device", "eval_chunk", "clutter_estimator", "save_root",
                 "threshold", "pd_detT", "correct_thresh", "rolling_window", "segments", "timing_warmup_windows")


def scratch_config(cfg):
    """从头训练：骨干的结构与数据设置取自 backbone_config（如 E3a 的 YAML），本配置覆盖数据根目录、种子、评测口径等。"""
    base = load_flat_config(cfg["backbone_config"])
    base.update({key: cfg[key] for key in OVERRIDE_KEYS if key in cfg})
    base["execution"] = "layer"
    return base


def build_models(cfg, base, device, q99):
    """按骨干配置建网络（不读权重）+ 目标神经元层 + 杂波场（clutter_estimator = learned 时）。"""
    if cfg.get("head_kind") not in ("object", "tokens"):
        raise ValueError("端到端训练只用于目标层（head_kind: tokens = v6 目标 token 层，object = v5 对照）")
    net = T.build_net(base, device)
    if net.invariant is not None:
        net.invariant.set_q99(q99)                                   # 模块二：查表用的 q99
    tap = TH.ReadoutTap(net)
    head = TH.build_head(cfg, net, base, device)
    field = None
    if base.get("clutter_estimator", "online") == "learned":
        if net.invariant is None:
            raise ValueError("clutter_estimator = learned 需要骨干开不变接口（invariant_input: true）")
        field = ClutterField(int(base["pad_height"]), int(base["pad_width"]),
                             sensor_hw=(int(base["height"]), int(base["width_px"])),
                             ref_rate=tuple(base.get("invariant_ref_rate", REF_RATE))).to(device)
    return net, tap, head, field


def load_models(cfg, device):
    """训练用：从头（backbone_config）或从骨干 checkpoint（base_checkpoint）起步。
    返回 (net, tap, head, field, base, q99, pos_weight, stats, ckpt)；从头时 ckpt 为 None、网络尚未做增益校准。"""
    if cfg.get("backbone_config"):
        base = scratch_config(cfg)
        stats, q99, pos_weight = T.load_stats(base)
        net, tap, head, field = build_models(cfg, base, device, q99)
        return net, tap, head, field, base, q99, pos_weight, stats, None
    ckpt = torch.load(cfg["base_checkpoint"], map_location=device)
    base = T.eval_train_config(ckpt["config"], cfg)
    base["execution"], base["input_device"] = "layer", cfg.get("input_device", "gpu")
    if cfg.get("clutter_estimator"):
        base["clutter_estimator"] = cfg["clutter_estimator"]
    stats = ckpt["stats"]
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    pos_weight = min(float(stats["train_neg"]) / float(stats["train_pos"]), float(base["pos_weight_cap"]))
    net, tap, head, field = build_models(cfg, base, device, q99)
    net.load_state_dict(ckpt["model"])
    if not bool(net.gain_calibrated):
        raise RuntimeError("骨干 checkpoint 的增益未校准")
    return net, tap, head, field, base, q99, pos_weight, stats, ckpt


def comm_from_backbone(base, net, cfg, ckpt):
    """模块一的消息峰值约束；乘子 mu 从骨干那次训练里 checkpoint 所在一轮结束时的值接着走（没有记录时从 0 开始）。"""
    comm = CommBudget.from_config(base, net)
    if comm is None or ckpt is None:                                   # 从头训练：乘子从 0 开始（与骨干单独训练相同）
        return comm
    path = Path(cfg["base_checkpoint"]).parent / "metrics.jsonl"
    found = False
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("epoch") == ckpt.get("epoch") and rec.get("comm"):
                comm.mu, found = float(rec["comm"]["mu_end"]), True
    if not found:
        print("注意：没找到骨干训练记录 %s 里第 %s 轮的乘子，消息约束的乘子从 0 开始" % (path, ckpt.get("epoch")), flush=True)
    return comm


# ---------------------------------------------------------------------------
# 多流数据
# ---------------------------------------------------------------------------


class Sources(T.StreamSources):
    """与 train_stream_v1.StreamSources 相同，另存每条录像的 target_id（目标通路的监督要用）。"""

    def __init__(self, dataset, cfg, q99, device, dtype):
        super(Sources, self).__init__(dataset, cfg, q99, device, dtype)
        self.tids = {}

    def get(self, index):
        if index not in self.sources:
            seq = self.dataset[index]
            self.sources[index] = make_window_source(seq, self.cfg, self.q99, self.device, self.dtype)
            self.bounds[index] = seq.bounds
            self.tids[index] = np.asarray(seq.target_id, dtype=np.float32)
        return self.sources[index]

    def release(self, index):
        super(Sources, self).release(index)
        self.tids.pop(index, None)


def assemble(sources, pieces, steps, cfg, device, dtype):
    """与 train_stream_v1.assemble_streams 相同（x [T,B,C,H,W]、events、labels、keep [T,B]），另返回每个事件的 target_id。"""
    batch = len(pieces)
    x = torch.zeros((steps, batch, num_input_channels(cfg["time_bins"]), int(cfg["pad_height"]),
                     int(cfg["pad_width"])), dtype=dtype, device=device)
    keep = torch.ones((steps, batch), dtype=dtype, device=device)
    parts, labels, tids = {key: [] for key in ("t", "b", "y", "x", "p", "t_local")}, [], []
    bg = None
    for b, stream in enumerate(pieces):
        for index, lo, hi, offset, new_start in stream:
            xs, events, lab, idx = sources.get(index).chunk(lo, hi)
            x[offset:offset + (hi - lo), b] = xs[:, 0]
            if "bg" in events:
                if bg is None:
                    bg = torch.zeros((steps, batch) + tuple(events["bg"].shape[2:]), dtype=torch.float32, device=device)
                bg[offset:offset + (hi - lo), b] = events["bg"][:, 0]
            if new_start:
                keep[offset, b] = 0
            parts["t"].append(events["t"] + int(offset))
            parts["b"].append(torch.full_like(events["t"], b))
            for key in ("y", "x", "p", "t_local"):
                parts[key].append(events[key])
            labels.append(lab)
            tids.append(torch.from_numpy(sources.tids[index][np.asarray(idx, dtype=np.int64)]).to(device))
    if labels:
        events = {key: torch.cat(value) for key, value in parts.items()}
        labels, tid = torch.cat(labels), torch.cat(tids).to(dtype)
    else:
        long_empty = torch.zeros(0, dtype=torch.long, device=device)
        float_empty = torch.zeros(0, dtype=dtype, device=device)
        events = {"t": long_empty, "b": long_empty, "y": long_empty, "x": long_empty,
                  "p": float_empty, "t_local": float_empty}
        labels, tid = float_empty, float_empty
    if bg is not None:
        events["bg"] = bg
    return x, events, labels, keep, tid


def object_events(events):
    """骨干的事件字典 -> 目标层的事件字典（另加每个事件的背景率，来自模块二）。"""
    ev = {key: events[key] for key in ("t", "b", "x", "y", "p", "t_local")}
    if events.get("bg") is not None:
        ev["bg_event"] = TH.event_background(events)
    return ev


# ---------------------------------------------------------------------------
# 训练与评估
# ---------------------------------------------------------------------------


def train_epoch(net, tap, head, dataset, lengths, optimizer, cfg, base, q99, pos_weight, device, rng, comm,
                max_chunks=None, field=None):
    k, n_streams = int(base["tbptt_k"]), int(base["streams"])
    if int(base.get("accumulate", 1)) != 1:
        raise ValueError("端到端训练每个片段更新一次（accumulate = 1）")
    plan = T.plan_streams(lengths, n_streams, k, rng, base.get("stream_layout", "greedy"))
    item_ends = {(index, first + n) for _, items in plan for index, first, n in map(T.plan_item, items)}
    n_chunks = int(math.ceil(T.stream_total_windows(plan) / float(k)))
    if max_chunks is not None:
        n_chunks = min(n_chunks, int(max_chunks))
    dtype = next(net.parameters()).dtype
    sources = Sources(dataset, base, q99, device, dtype)
    weight = torch.tensor([pos_weight], dtype=dtype, device=device)
    named_params = [(prefix + name, p) for prefix, module in (("backbone.", net), ("head.", head), ("field.", field))
                    if module is not None for name, p in module.named_parameters() if p.requires_grad]
    H, W = int(base["pad_height"]), int(base["pad_width"])
    lam_clutter = float(cfg.get("lambda_clutter", 1.0))
    clip = float(cfg["grad_clip"])
    w_pix, w_now, w_lag = float(cfg["lambda_pixel"]), float(cfg["now_loss_weight"]), float(cfg["lag_loss_weight"])
    lam = {"exist": float(cfg["lambda_exist"]), "pos": float(cfg["lambda_pos"]), "mem": float(cfg["lambda_membership"])}
    sums = {key: 0.0 for key in ("pix", "now", "lag", "exist", "pos", "mem", "clutter")}
    counts = {key: 0 for key in sums}
    stat = {"births": 0, "live": 0, "slot_windows": 0, "updates": 0}
    norms = []
    if comm is not None:
        comm.start_epoch()
    states = hstate = fstate = None
    net.train()
    head.train()
    if field is not None:
        field.train()
    for c in range(n_chunks):
        pieces = T.stream_pieces(plan, c * k, (c + 1) * k)
        x, events, labels, keep, tid = assemble(sources, pieces, k, base, device, dtype)
        clutter = None
        if field is not None:                                        # 模块二 v2：学出来的杂波场给出背景率（只读给骨干、目标层）
            cmaps = event_count_maps(events, k, len(pieces), H, W)
            log_rate, fstate = field.forward_chunk(cmaps, fstate, keep)
            events["bg"] = torch.exp(log_rate).detach()
            if labels.numel():
                background = event_count_maps(events, k, len(pieces), H, W, weight=(labels < 0.5).float())
                clutter = field.nll(log_rate, background, T.stream_valid(pieces, k, device, dtype))
        prev = states
        logits, states, info = net.forward_chunk(x, events, states, keep=keep, collect=comm is not None)
        if labels.numel():
            z, hstate, aux = head.forward_chunk(object_events(events), logits, tap.feat, hstate, k, keep=keep,
                                                labels=labels, tid=tid, keys=labels)
            for key in ("births", "live", "slot_windows"):
                stat[key] += aux[key]
            n_ev = int(labels.numel())
            pix = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight, reduction="sum") / n_ev
            now = F.binary_cross_entropy_with_logits(z, labels, pos_weight=weight, reduction="sum") / n_ev
            loss = w_pix * pix + w_now * now
            parts = {"pix": (pix, n_ev), "now": (now, n_ev)}
            if aux["delayed_z"] is not None:
                # same normalisation as the per-event losses (/ all events of the chunk): every event's final answer
                # weighs as much as its zero-wait answer; averaging over the few waiting events alone would give each
                # of them ~100x weight and let them dominate the backbone's gradient
                n_lag = int(aux["delayed_z"].shape[0])
                lag = F.binary_cross_entropy_with_logits(aux["delayed_z"], aux["delayed_key"], pos_weight=weight,
                                                         reduction="sum") / n_ev
                loss = loss + w_lag * lag
                parts["lag"] = (lag * n_ev / n_lag, n_lag)                 # logged per waiting event
            for key in ("exist", "pos", "mem"):
                if aux[key + "_n"]:
                    term = aux[key + "_sum"] / aux[key + "_n"]
                    loss = loss + lam[key] * term
                    parts[key] = (term, aux[key + "_n"])
            if clutter is not None:
                loss = loss + lam_clutter * clutter
                parts["clutter"] = (clutter, 1)
            if comm is not None:
                penalty = comm.chunk_loss(net, info["u_pre"], prev, keep, 1, T.stream_valid(pieces, k, device, dtype))
                if penalty is not None:
                    loss = loss + penalty
            if not math.isfinite(float(loss.detach())):
                raise RuntimeError("损失出现非有限值：片段 %d" % c)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norms.append(clip_grad_norm_checked_(named_params, clip, "end2end chunk %d" % c))
            optimizer.step()
            if comm is not None:
                comm.end_update()
            stat["updates"] += 1
            for key, (value, n) in parts.items():
                sums[key] += float(value.detach()) * n
                counts[key] += n
        states = detach_states(states)
        hstate = head.detach_state(hstate)
        if field is not None:
            fstate = field.detach_state(fstate)
        for stream in pieces:
            for index, lo, hi, _, _ in stream:
                if (index, hi) in item_ends:
                    sources.release(index)
    windows = max(stat["slot_windows"], 1)
    out = {key + "_loss": (sums[key] / counts[key] if counts[key] else None) for key in sums}
    out.update(updates=stat["updates"], chunks=n_chunks, grad_norm_median=float(np.median(norms)) if norms else 0.0,
               births_per_window=stat["births"] / windows, live_per_window=stat["live"] / windows,
               comm=comm.summary() if comm is not None else None)
    return out


def evaluate(net, tap, head, base, q99, device, split, full=False, dump_dir=None, tracks_dir=None, dataset=None,
             field=None):
    """整段连续推理一个划分（杂波场 + 骨干 + 目标层）；指标与 train_track_head.run_split 同一口径。"""
    was_training = net.training
    net.eval()
    if field is not None:
        field.eval()
    dataset = dataset if dataset is not None else EvUAVStream(base["root"], split, base)
    outputs = TH.BaseOutputs(net, tap, base, q99, device, dataset, keep_all=False, with_bg=TH.needs_background(head),
                             field=field)
    result = TH.run_split(outputs, head, base, device, split, full=full, dump_dir=dump_dir, tracks_dir=tracks_dir)
    net.train(was_training)
    if field is not None:
        field.train(was_training)
    return result


def save_checkpoint(path, net, head, field, epoch, val, best, cfg, base, stats, comm):
    torch.save({"model": net.state_dict(), "head": head.state_dict(),
                "field": field.state_dict() if field is not None else None, "epoch": epoch, "val": val,
                "best_val_iou": best, "config": cfg, "base_config": base, "stats": stats,
                "comm_mu": float(comm.mu) if comm is not None else None}, str(path))


def mode_train(args, cfg):
    device = torch.device(args.device)
    T.seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    net, tap, head, field, base, q99, pos_weight, stats, ckpt = load_models(cfg, device)
    root = Path(cfg["save_root"])
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics.jsonl"
    if metrics_path.exists():
        raise RuntimeError("输出目录已有 metrics.jsonl，请换目录")
    T.write_json(root / "run_config.json", {"config": cfg, "base_config": base})
    if ckpt is None:                                                  # 从头训练：一次性增益校准（杂波场初值 = 参照率，接口近似恒等）
        T.run_calibration(net, base, q99, device, root)
    train_set = EvUAVStream(base["root"], "train", base)
    lengths = [int(train_set[i].n_windows) for i in range(len(train_set))]
    rate0 = None
    if hasattr(head, "init_rate"):                                    # v5 only: its expected-count init is a data statistic
        rate0 = TH.mean_target_events(train_set)
        head.init_rate(rate0)
    comm = comm_from_backbone(base, net, cfg, ckpt)
    lr0, lr1, scale = float(cfg["lr"]), float(cfg["lr_end"]), float(cfg["backbone_lr_scale"])
    groups = [{"params": net.parameters(), "lr": lr0 * scale, "scale": scale},
              {"params": head.parameters(), "lr": lr0, "scale": 1.0}]
    if field is not None:
        groups.append({"params": field.parameters(), "lr": lr0, "scale": 1.0})
    optimizer = torch.optim.Adam(groups)
    seed, epochs = int(cfg["seed"]), int(cfg["epochs"])
    val_set = EvUAVStream(base["root"], "val", base)
    print("端到端训练：骨干 %s（学习率 x %.3g）+ 目标层 %s（参数 %d；杂波 %s；分离 %s）+ 杂波场 %s | "
          "%d 条流、k=%d | 训练集 %d 条录像 %d 窗 | pos_weight %.2f | 期望事件数初值 %s | 长度单位 %.0f px | 消息约束 %s" % (
              cfg.get("base_checkpoint") or ("从头（%s）" % cfg["backbone_config"]), scale, cfg["head_kind"],
              sum(p.numel() for p in head.parameters()), head.clutter_source, head.separation,
              "学出来（参数 %d）" % sum(p.numel() for p in field.parameters()) if field is not None else "手写估计器",
              int(base["streams"]), int(base["tbptt_k"]), len(lengths), sum(lengths), pos_weight,
              "无" if rate0 is None else "%.2f" % rate0, head.unit,
              "关" if comm is None else "开（mu 从 %.4f 起）" % comm.mu), flush=True)
    best = -float("inf")
    for epoch in range(epochs):
        lr = linear_epoch_lr(epoch, epochs, lr0, lr1)
        for group in optimizer.param_groups:
            group["lr"] = lr * group["scale"]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        out = train_epoch(net, tap, head, train_set, lengths, optimizer, cfg, base, q99, pos_weight, device,
                          np.random.RandomState(seed * 1000 + epoch + 7), comm, args.max_chunks, field=field)
        train_s = time.perf_counter() - t0
        val = evaluate(net, tap, head, base, q99, device, "val", dataset=val_set, field=field)
        record = dict(epoch=epoch, lr=lr, train=out, val=val, gate=TH.gate_check(val, cfg), train_seconds=train_s,
                      epoch_seconds=time.perf_counter() - t0, peak_memory_gib=T.peak_memory_gib(device),
                      evidence=head.evidence_summary(),
                      clutter_decays=field.decays().tolist() if field is not None else None)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if val["track"]["iou"] > best:
            best = val["track"]["iou"]
            save_checkpoint(root / ("best_val_iou_seed%d.pt" % seed), net, head, field, epoch, val, best, cfg, base,
                            stats, comm)
        save_checkpoint(root / "last.pt", net, head, field, epoch, val, best, cfg, base, stats, comm)
        hs, ev = val["head_stats"], record["evidence"]
        loss_txt = " ".join("%s %.4f" % (key, out[key + "_loss"])
                            for key in ("pix", "now", "lag", "exist", "pos", "mem", "clutter")
                            if out[key + "_loss"] is not None)
        comm_txt = ""
        if out["comm"] is not None:
            comm_txt = " | 消息 p95 %.4g / 预算 %.4g mu %.4f" % (out["comm"].get("messages_window_p95") or 0.0,
                                                           out["comm"]["budget"], out["comm"]["mu_end"])
        ev_txt = " ".join("%s %.3f" % (name, ev[key]) for key, name in
                          (("orphan_ref", "l0"), ("llr_scale", "wL"), ("clutter_gain", "杂波增益")) if key in ev)
        print("epoch %d lr %.2e | 损失 %s | val IoU base %.4f now %.4f track %.4f | 延后 %.2f%%（目标事件 %.1f%%，平均 "
              "%.3f 窗）| 每窗 槽 %.2f 出生 %.3f | %s%s | %.0fs（训练 %.0fs）| 显存峰值 %.2f GiB" % (
                  epoch, lr, loss_txt, val["base"]["iou"], val["track_now"]["iou"], val["track"]["iou"],
                  100.0 * hs["deferred_frac"], 100.0 * hs["target_deferred_frac"], hs["target_mean_delay_windows"],
                  hs["live_per_window"], hs["births_per_window"], ev_txt, comm_txt, record["epoch_seconds"], train_s,
                  record["peak_memory_gib"] or 0.0),
              flush=True)
    print("TRAINING FINISHED:", root, flush=True)


def mode_eval(args, cfg):
    device = torch.device(args.device)
    T.seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    ckpt = torch.load(args.checkpoint, map_location=device)
    base = dict(ckpt["base_config"])                                   # 训练时的骨干配置；数据根目录与评测口径取本配置
    base.update({key: cfg[key] for key in OVERRIDE_KEYS if key in cfg and key not in ("clutter_estimator", "save_root")})
    q99 = np.asarray(ckpt["stats"]["q99"], dtype=np.float32)
    net, tap, head, field = build_models(cfg, base, device, q99)
    net.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    if field is not None:
        field.load_state_dict(ckpt["field"])
    for d in (args.dump_dir, args.tracks_dir):
        if d:
            os.makedirs(d, exist_ok=True)
    result = evaluate(net, tap, head, base, q99, device, args.split, full=True, dump_dir=args.dump_dir,
                      tracks_dir=args.tracks_dir, field=field)
    result.update(checkpoint=args.checkpoint, epoch=ckpt.get("epoch"),
                  base_checkpoint=cfg.get("base_checkpoint") or cfg.get("backbone_config"),
                  clutter_estimator=base.get("clutter_estimator", "online"),
                  clutter_decays=field.decays().tolist() if field is not None else None,
                  gate=TH.gate_check(result, cfg), context=head.context, decision_lag=head.decision_lag,
                  defer_max=head.defer_max, defer_alpha=cfg.get("defer_alpha"), existence_calibration=None,
                  defer_revision=head.defer_revision, evidence=head.evidence_summary(),
                  clutter_source=head.clutter_source, continuous_time=head.continuous_time)
    path = Path(args.checkpoint).parent / ("eval_%s_%s.json" % (args.split, Path(args.checkpoint).stem))
    T.write_json(path, result)
    for kind in ("base", "track_now", "track"):
        r = result[kind]
        print("%s %-9s IoU %.4f ACC %.4f Pd %.4f Fa %.3e | 拖尾误报 %d 断续漏检 %d 目标附近误报 %d" % (
            args.split, kind, r["iou"], r["acc"], r["pd"], r["fa"], r["fp_cat"]["tail"], r["fn_cat"]["gap"],
            r["fp_cat"]["near"]), flush=True)
    print("目标层统计：", json.dumps(result["head_stats"], ensure_ascii=False), flush=True)
    print("结果写入", path, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("train", "eval"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", default=None, help="eval：端到端 checkpoint（含骨干与目标层）")
    parser.add_argument("--dump-dir", default=None, help="eval：保存逐事件预测")
    parser.add_argument("--tracks-dir", default=None, help="eval：保存逐窗目标槽")
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-checkpoint", default=None, help="覆盖配置中的骨干初始化 checkpoint")
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
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.base_checkpoint:
        cfg["base_checkpoint"] = args.base_checkpoint
        cfg.pop("backbone_config", None)
    if args.mode == "train":
        if int(cfg["epochs"]) < 2:
            raise ValueError("epochs 至少为 2（学习率按轮线性下降）")
        mode_train(args, cfg)
    else:
        if not args.checkpoint:
            raise ValueError("eval 需要 --checkpoint")
        mode_eval(args, cfg)


if __name__ == "__main__":
    main()
