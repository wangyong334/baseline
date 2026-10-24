"""v6-2：V2-1 判决层 + V3 发布层接到 V5 上的训练与评估（检验"之前的方案在 V5 上是否还有效"）。

两种骨干（evidence_backbone）：
    frozen  V5 定型权重冻结，只训练 V2 的两个逐像素头（第一轮，configs/evisseg_stream_evidence.yaml）
    joint   与 V3 最终版相同的训练方式：骨干（V5 结构：BIR + 每窗消息数的峰值约束）与两个头一起从头训练，
            损失 = mark 的逐事件 BCE（无 pos_weight）+ 目标强度的泊松 NLL + V5 的消息约束；V5 自带的读出头不参与
两种网络输入（evidence_input）：
    frame     V5 的 12 通道计数帧
    frontend  V3 最终版的输入：V2 证据前端（dataset/stream_features.py，从 v3 逐字节照搬）的 22 通道特征
两个头（model/evidence_readout.EvidenceHeads，读 dec1 的 u_pre）：mark（零等待读出 net）、log_g（目标强度场）。
判决层与发布层无可学习参数，从 v3 逐字节照搬（model/evidence_neuron.py、model/publish_readout.py）：
    fused_d{1..5}  V2-1：mark + 之后 d 窗沿 49 条速度管道的证据
    pub            V3：双边界发布 + 锚定证据链（年龄 0..5 窗择机发布）
V5 参照：backbone_checkpoint 指向 V5 定型权重。frozen 时它就是骨干；joint 时评估另跑一份冻结的 V5，
    给出逐事件的 V5 概率（纠正 / 破坏数的基准），并与 V5 正式评估 JSON 核对。
模式（--mode）：
    smoke  2 个训练块 + 1 条验证序列（逐窗 / 逐片段两种执行的判定差异）
    train  与 V5 相同的多流排布逐块训练；每轮在 val 上按 net 的 IoU 选模型（V2 的规则）
    eval   逐窗执行跑完整划分：V5、net、fused_d1..5、pub 的四个指标，相对 V5 的纠正 / 破坏数，pub 的发布统计
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
from dataset.stream_features import EventChunkSource, EvidenceFrontEnd  # noqa: E402
from dataset.stream_source import make_window_source  # noqa: E402
from dataset.stream_windows import num_input_channels, refill_by_index  # noqa: E402
from model.evidence_readout import DEC1, EvidenceHeads, EvidenceRun, pixel_counts  # noqa: E402
from model.evspsegnet_stream import EvSpSegNetStream, calibrate_gains  # noqa: E402
from utils.comm_budget import CommBudget  # noqa: E402
from utils.evidence_loss import smoothed_target  # noqa: E402
from utils.evidence_metrics import (event_groups, flip_counts, iou_counts, publish_summary,  # noqa: E402
                                    split_report)
from utils.stream_common import linear_epoch_lr, load_flat_config, select_subset  # noqa: E402

BACKBONES = ("frozen", "joint")
INPUTS = ("frame", "frontend")
REQUIRED_KEYS = (
    "base_config", "backbone_checkpoint", "evidence_backbone", "evidence_input",
    "head_hidden", "mark_prior", "intensity_prior", "log_g_max",
    "loss_mark_weight", "loss_intensity_weight", "loss_intensity_smooth",
    "bg_fast_ms", "bg_slow_ms", "bg_prior", "bg_prior_windows", "bg_floor", "bg_smooth_radius",
    "cusum_axis_velocities", "cusum_footprint", "cusum_compensator", "cusum_nb_kappa", "cusum_aggregate",
    "cusum_track_tau_ms", "cusum_memory_gain", "cusum_gate_eps", "readout_delays", "fusion_weight",
    "publish_deadline", "publish_upper", "publish_lower", "publish_collapse", "publish_gate", "publish_anchor",
    "evidence_epochs", "evidence_lr", "evidence_lr_end", "evidence_grad_clip", "evidence_streams",
    "evidence_chunk", "evidence_eval_chunk", "evidence_save_root")
FRONTEND_KEYS = ("fe_taus_ms", "fe_dipole_taus_ms", "fe_dipole_radius", "fe_features")
DATA_KEYS = ("height", "width_px", "pad_height", "pad_width", "window_ms", "n_windows", "time_bins", "input_clip")
EVAL_KEYS = ("threshold", "pd_detT", "correct_thresh", "rolling_window", "segments", "timing_warmup_windows")


def parse_args():
    parser = argparse.ArgumentParser(description="V2/V3 decision layer on V5")
    parser.add_argument("--config", required=True, help="configs/evisseg_stream_evidence*.yaml")
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
    if ev.get("evidence_backbone") not in BACKBONES or ev.get("evidence_input") not in INPUTS:
        raise ValueError("evidence_backbone 必须是 %s 之一，evidence_input 必须是 %s 之一" % (BACKBONES, INPUTS))
    frontend = ev["evidence_input"] == "frontend"
    allowed = set(REQUIRED_KEYS) | (set(FRONTEND_KEYS) if frontend else set())
    missing = [k for k in sorted(allowed) if k not in ev]
    unknown = sorted(set(ev) - allowed)
    if missing or unknown:
        raise ValueError("EVIDENCE 配置缺少 %s，多出 %s" % (missing, unknown))
    if ev["evidence_backbone"] == "frozen" and frontend:
        raise ValueError("冻结的 V5 骨干只能接 V5 的输入（evidence_input: frame）")
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


def is_joint(cfg):
    return cfg["evidence_backbone"] == "joint"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_backbone(cfg, device, execution, input_device):
    """载入并冻结 V5 定型权重。返回 (net, net_cfg, q99)；数据与输入键必须与 base_config 一致。"""
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


def build_frontend(cfg, device):
    """V2 证据前端（与 v3 train_stream_v2.build_frontend 相同的参数顺序；bg_mode 固定 adaptive）。"""
    return EvidenceFrontEnd(cfg["fe_taus_ms"], cfg["window_ms"], cfg["fe_dipole_taus_ms"], cfg["fe_dipole_radius"],
                            cfg["bg_fast_ms"], cfg["bg_slow_ms"], cfg["bg_prior"], cfg["bg_prior_windows"],
                            cfg["bg_floor"], cfg["bg_smooth_radius"], cfg["fe_features"], "adaptive").to(device)


def input_channels(cfg, frontend):
    return frontend.n_features if frontend is not None else num_input_channels(cfg["time_bins"])


def build_joint_net(cfg, frontend, device):
    """V5 结构的骨干（与 train_stream_v1.build_net 相同，只是输入通道数按 evidence_input 决定），随机初始化。"""
    net = EvSpSegNetStream(
        in_channels=input_channels(cfg, frontend), channels=tuple(cfg["channels"]),
        neuron=cfg["neuron"], norm=cfg["norm"], state_mode=cfg["state_mode"],
        readout_hidden=int(cfg["readout_hidden"]), dt_ms=float(cfg["window_ms"]),
        tau_init_ms=float(cfg["tau_init_ms"]), tau_min_ms=float(cfg["tau_min_ms"]),
        tau_max_ms=float(cfg["tau_max_ms"]), v_threshold=float(cfg["v_threshold"]),
        merged_decoder=cfg.get("merged_decoder", False), dual_memory=cfg.get("dual_memory"),
        subthreshold_residual=cfg.get("subthreshold_residual"),
        tif=cfg.get("tif") if cfg["neuron"] == "tif" else None)
    return net.to(device)


def joint_net_cfg(cfg, execution, input_device):
    out = dict(cfg)
    out["execution"], out["input_device"] = execution, input_device
    return out


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


def empty_events(device, dtype):
    long_empty = torch.zeros(0, dtype=torch.long, device=device)
    float_empty = torch.zeros(0, dtype=dtype, device=device)
    return {"b": long_empty, "y": long_empty, "x": long_empty, "p": float_empty, "t_local": float_empty}


def calibrate_joint(net, cfg, frontend, q99, device, out_dir):
    """一次性逐层增益校准（与 V5 / V2 相同的规则与数据：固定种子的训练序列，每条取前 calib_windows 窗）。"""
    if frontend is None:
        return V5.run_calibration(net, cfg, q99, device, out_dir)
    names = select_subset(V5.train_file_names(cfg), int(cfg["calib_sequences"]), int(cfg["seed"]) + 1000)
    dataset = EvUAVStream(cfg["root"], "train", cfg, names)
    n_windows = int(cfg["calib_windows"])
    H, W = int(cfg["pad_height"]), int(cfg["pad_width"])
    dtype = next(net.parameters()).dtype

    def windows_of(seq):
        source = EventChunkSource(seq, cfg, device, dtype)
        state = frontend.init_state(1, H, W, device, dtype)
        _, feats, _, _ = frontend.run_chunk(state, source.chunk(0, min(n_windows, seq.n_windows)))
        for t in range(int(feats.shape[0])):
            yield feats[t], empty_events(device, dtype)

    def factory():
        for i in range(len(dataset)):
            yield windows_of(dataset[i])

    reports = calibrate_gains(net, factory, float(cfg["calib_quantile"]), int(cfg["calib_samples_per_channel"]),
                              int(cfg["calib_min_positive"]), float(cfg["calib_gain_min"]),
                              float(cfg["calib_gain_max"]), int(cfg["seed"]) + 4000)
    V5.write_json(Path(out_dir) / "calibration.json", {"sequences": names, "layers": reports})
    return reports


# ---------------------------------------------------------------------------
# 输入：多流训练的拼批，与一条序列的逐窗 / 逐片段输入
# ---------------------------------------------------------------------------


class FrontendStreams(object):
    """V2 前端输入的多流拼批：每段录像一个 EventChunkSource，每条流一份前端状态（开始新录像时清零）。
    片段内的窗口、事件、keep 的排布与 V5.assemble_streams 相同；没有录像的空窗输入为 0。"""

    def __init__(self, dataset, cfg, frontend, device, dtype):
        self.dataset, self.cfg, self.frontend, self.device, self.dtype = dataset, cfg, frontend, device, dtype
        self.sources, self.states = {}, {}

    def get(self, index):
        if index not in self.sources:
            self.sources[index] = EventChunkSource(self.dataset[index], self.cfg, self.device, self.dtype)
        return self.sources[index]

    def release(self, index):
        self.sources.pop(index, None)

    def assemble(self, pieces, steps):
        H, W = int(self.cfg["pad_height"]), int(self.cfg["pad_width"])
        batch = len(pieces)
        x = torch.zeros((steps, batch, self.frontend.n_features, H, W), dtype=self.dtype, device=self.device)
        keep = torch.ones((steps, batch), dtype=self.dtype, device=self.device)
        parts, labels = {key: [] for key in ("t", "b", "y", "x", "p", "t_local")}, []
        with torch.no_grad():
            for b, stream in enumerate(pieces):
                for index, lo, hi, offset, new_start in stream:
                    chunk = self.get(index).chunk(lo, hi)
                    if new_start or b not in self.states:
                        self.states[b] = self.frontend.init_state(1, H, W, self.device, self.dtype)
                    if new_start:
                        keep[offset, b] = 0
                    for t in range(hi - lo):
                        self.states[b], feats, _, _ = self.frontend.step(
                            self.states[b], chunk["counts"][t], chunk["moment_a"][t], chunk["moment_b"][t])
                        x[offset + t, b] = feats[0]
                    ev = chunk["events"]
                    parts["t"].append(ev["t"] + int(offset))
                    parts["b"].append(torch.full_like(ev["t"], b))
                    for key in ("y", "x", "p", "t_local"):
                        parts[key].append(ev[key])
                    labels.append(chunk["labels"])
        if labels:
            return x, {key: torch.cat(v) for key, v in parts.items()}, torch.cat(labels), keep
        events = dict(empty_events(self.device, self.dtype), t=torch.zeros(0, dtype=torch.long, device=self.device))
        return x, events, torch.zeros(0, dtype=self.dtype, device=self.device), keep


class FrameStreams(object):
    """V5 输入的多流拼批（V5.StreamSources + V5.assemble_streams，逐位相同）。"""

    def __init__(self, dataset, net_cfg, q99, device, dtype):
        self.sources = V5.StreamSources(dataset, net_cfg, q99, device, dtype)
        self.net_cfg, self.device, self.dtype = net_cfg, device, dtype

    def release(self, index):
        self.sources.release(index)

    def assemble(self, pieces, steps):
        return V5.assemble_streams(self.sources, pieces, steps, self.net_cfg, self.device, self.dtype)


class SequenceInput(object):
    """一条序列的网络输入：frame 用 V5 的窗口来源；frontend 用 V2 的 EventChunkSource + 前端递推。"""

    def __init__(self, seq, cfg, net_cfg, q99, frontend, device, dtype):
        self.frontend = frontend
        if frontend is None:
            self.source = make_window_source(seq, net_cfg, q99, device, dtype)
        else:
            self.source = EventChunkSource(seq, cfg, device, dtype)
            self.state = frontend.init_state(1, int(cfg["pad_height"]), int(cfg["pad_width"]), device, dtype)

    def window(self, k):
        """第 k 窗：(x [1,C,H,W], events（无 t）, labels, idx)。"""
        if self.frontend is None:
            return self.source.window(k)
        x, events, labels, idx = self.chunk(k, k + 1)
        events = {key: value for key, value in events.items() if key != "t"}
        return x[0], events, labels, idx

    def chunk(self, start, end):
        """第 start..end-1 窗：(x [T,1,C,H,W], events（t 为片段内窗号）, labels, idx)。"""
        if self.frontend is None:
            return self.source.chunk(start, end)
        chunk = self.source.chunk(start, end)
        self.state, feats, _, _ = self.frontend.run_chunk(self.state, chunk)
        return feats, chunk["events"], chunk["labels"], chunk["idx"]


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


def train_epoch(net, heads, dataset, lengths, optimizer, cfg, net_cfg, q99, device, rng, frontend=None, comm=None,
                max_chunks=None, plan=None):
    """一轮：B 条流逐块前向。frozen 时骨干无梯度、只更新两个头；joint 时骨干与头一起更新，另加 V5 的消息约束。
    损失 = (w_mark * BCE + w_int * NLL) / 本块事件数（+ 消息约束项）；没有事件的块不更新（与 V5 相同）。"""
    joint = is_joint(cfg)
    k, n_streams = int(cfg["evidence_chunk"]), int(cfg["evidence_streams"])
    if plan is None:
        plan = V5.plan_streams(lengths, n_streams, k, rng, net_cfg.get("stream_layout", "greedy"))
    item_ends = {(index, first + n) for _, items in plan for index, first, n in map(V5.plan_item, items)}
    n_chunks = int(math.ceil(V5.stream_total_windows(plan) / float(k)))
    if max_chunks is not None:
        n_chunks = min(n_chunks, int(max_chunks))
    dtype = next(heads.parameters()).dtype
    feeder = (FrontendStreams(dataset, cfg, frontend, device, dtype) if frontend is not None
              else FrameStreams(dataset, net_cfg, q99, device, dtype))
    w_mark, w_int = float(cfg["loss_mark_weight"]), float(cfg["loss_intensity_weight"])
    smooth, clip = int(cfg["loss_intensity_smooth"]), float(cfg["evidence_grad_clip"])
    params = list(heads.parameters()) + (list(net.parameters()) if joint else [])
    stats = {"bce_sum": 0.0, "nll_sum": 0.0, "events": 0}
    net_states, updates = None, 0
    if comm is not None:
        comm.start_epoch()
    heads.train()
    if joint:
        net.train()
    for c in range(n_chunks):
        start = c * k
        pieces = V5.stream_pieces(plan, start, start + k)
        x, events, labels, keep = feeder.assemble(pieces, k)
        valid = V5.stream_valid(pieces, k, device, dtype)
        prev_states = net_states
        if joint:
            _, net_states, info = net.forward_chunk(x, events, net_states, keep=keep, collect=True)
        else:
            with torch.no_grad():
                _, net_states, info = net.forward_chunk(x, events, net_states, keep=keep, collect=True)
        n = int(labels.numel())
        if n:
            bce, nll = head_losses(heads, info["u_pre"][DEC1], events, labels, valid, smooth)
            loss = (w_mark * bce + w_int * nll) / float(n)
            if comm is not None:
                penalty = comm.chunk_loss(net, info["u_pre"], prev_states, keep, 1, valid)
                if penalty is not None:
                    loss = loss + penalty
            if not math.isfinite(float(loss.detach())):
                raise RuntimeError("损失出现非有限值: 块 %d" % c)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            optimizer.step()
            updates += 1
            if comm is not None:
                comm.end_update()
            stats["bce_sum"] += float(bce.detach())
            stats["nll_sum"] += float(nll.detach())
            stats["events"] += n
        if joint:
            net_states = V5.detach_states(net_states)
        for stream in pieces:
            for index, lo, hi, _, _ in stream:
                if (index, hi) in item_ends:
                    feeder.release(index)
    stats.update(chunks=n_chunks, updates=updates)
    if comm is not None:
        stats["comm"] = comm.summary()
    return stats


# ---------------------------------------------------------------------------
# 推理：一条序列，逐窗（step，与 V5 正式评估同一路径）或逐片段（layer）
# ---------------------------------------------------------------------------


def evaluate_sequence(net, heads, seq, cfg, net_cfg, q99, device, execution, decision=True, frontend=None,
                      reference=None):
    """推理一条序列。返回 dict：v5 [N]（V5 零等待概率；joint 且未给 reference 时为 None）、
    probs {net, fused_d*, pub}（decision=False 时只有 net）、extra（pub 的 age / reason / z）、nll（目标强度 NLL 之和）。
    reference = (ref_net, ref_cfg, ref_q99)：joint 时另跑一份冻结的 V5（逐窗执行）给出 V5 概率。调用方负责 torch.no_grad()。"""
    dtype = next(heads.parameters()).dtype
    inputs = SequenceInput(seq, cfg, net_cfg, q99, frontend, device, dtype)
    H, W = int(net_cfg["pad_height"]), int(net_cfg["pad_width"])
    smooth = int(cfg["loss_intensity_smooth"])
    n, n_windows = seq.n_events, seq.n_windows
    run = EvidenceRun(decision_cfg(cfg, net_cfg), H, W, device, dtype) if decision else None
    own_v5 = not is_joint(cfg)
    v5 = np.full(n, np.nan, dtype=np.float32) if (own_v5 or reference is not None) else None
    ref_source = ref_states = None
    if reference is not None:
        ref_net, ref_cfg, ref_q99 = reference
        ref_source = make_window_source(seq, ref_cfg, ref_q99, device, dtype)
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
            x, events, labels, idx = inputs.window(start)
            logits, net_states, info = net(x, events, net_states, collect=True)
            u7 = info["u_pre"][DEC1].unsqueeze(0)
            events = dict(events, t=torch.zeros_like(events["b"]))
        else:
            x, events, labels, idx = inputs.chunk(start, end)
            logits, net_states, info = net.forward_chunk(x, events, net_states, collect=True)
            u7 = info["u_pre"][DEC1]
        idx = np.asarray(idx, dtype=np.int64)
        if own_v5 and idx.size:
            v5[idx] = torch.sigmoid(logits).float().cpu().numpy()
        if ref_source is not None:
            for k in range(start, end):
                rx, rev, _, ridx = ref_source.window(k)
                rlog, ref_states, _ = ref_net(rx, rev, ref_states)
                ridx = np.asarray(ridx, dtype=np.int64)
                if ridx.size:
                    v5[ridx] = torch.sigmoid(rlog).float().cpu().numpy()
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
    if v5 is not None and np.isnan(v5).any():
        raise RuntimeError("序列 %s 的 V5 概率未填满" % seq.name)
    if run is None:
        return {"v5": v5, "probs": {"net": refill_by_index(n, *net_parts)}, "extra": {}, "nll": nll_sum}
    probs, extra = run.outputs(n)
    return {"v5": v5, "probs": probs, "extra": extra, "nll": nll_sum}


def readout_names(cfg):
    return ["net"] + ["fused_d%d" % int(d) for d in cfg["readout_delays"]] + ["pub"]


def quick_val(net, heads, dataset, cfg, net_cfg, q99, device, frontend=None):
    """训练中验证（逐片段执行，不跑判决层）：net 的 IoU（全局拼接口径，阈值同 V5）与每事件的强度 NLL。"""
    thr = float(net_cfg["threshold"])
    counts = np.zeros(3, dtype=np.int64)
    nll, events = 0.0, 0
    was_training = net.training
    heads.eval()
    net.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            seq = dataset[i]
            res = evaluate_sequence(net, heads, seq, cfg, net_cfg, q99, device, "layer", decision=False,
                                    frontend=frontend)
            counts += iou_counts(res["probs"]["net"], seq.label, thr)
            nll += res["nll"]
            events += int(seq.n_events)
    heads.train()
    net.train(was_training)
    return float(counts[0]) / float(max(counts.sum(), 1)), nll / float(max(events, 1))


# ---------------------------------------------------------------------------
# 构造与保存
# ---------------------------------------------------------------------------


def head_parameters(heads):
    return int(sum(p.numel() for p in heads.parameters()))


def setup(cfg, device, execution, input_device, calibrate_dir=None):
    """返回 (net, heads, net_cfg, q99, frontend)。frozen：载入冻结的 V5；joint：新建 V5 结构的骨干，
    calibrate_dir 给出时做一次增益校准（训练 / 冒烟），否则由调用方载入已训练的权重（评估）。"""
    if not is_joint(cfg):
        net, net_cfg, q99 = load_backbone(cfg, device, execution, input_device)
        heads = make_heads(cfg, net_cfg, device, next(net.parameters()).dtype)
        return net, heads, net_cfg, q99, None
    net_cfg = joint_net_cfg(cfg, execution, input_device)
    V5.seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    frontend = build_frontend(cfg, device) if cfg["evidence_input"] == "frontend" else None
    _, q99, _ = V5.load_stats(cfg)
    net = build_joint_net(cfg, frontend, device)
    heads = make_heads(cfg, net_cfg, device, next(net.parameters()).dtype)
    if calibrate_dir is not None:
        calibrate_joint(net, cfg, frontend, q99, device, calibrate_dir)
    return net, heads, net_cfg, q99, frontend


def save_checkpoint(path, net, heads, epoch, score, cfg):
    payload = {"heads": heads.state_dict(), "epoch": epoch, "score": score, "config": cfg,
               "backbone_sha256": file_sha256(cfg["backbone_checkpoint"])}
    if is_joint(cfg):
        payload["model"] = net.state_dict()
    torch.save(payload, str(path))


def checkpoint_name(cfg, kind):
    return ("%s_joint.pt" if is_joint(cfg) else "%s_heads.pt") % kind


# ---------------------------------------------------------------------------
# 模式
# ---------------------------------------------------------------------------


def mode_smoke(args, cfg):
    """2 个训练块 + 第 1 条验证序列（逐窗 / 逐片段两种执行比对，含判决层与发布层）。"""
    device = torch.device(args.device)
    tmp = Path(cfg["evidence_save_root"] + "_smoke")
    tmp.mkdir(parents=True, exist_ok=True)
    net, heads, net_cfg, q99, frontend = setup(cfg, device, "layer", "gpu", calibrate_dir=tmp)
    params = list(heads.parameters()) + (list(net.parameters()) if is_joint(cfg) else [])
    optimizer = torch.optim.Adam(params, lr=float(cfg["evidence_lr"]))
    comm = CommBudget.from_config(cfg, net) if is_joint(cfg) else None
    names = sorted(n for n in os.listdir(os.path.join(cfg["root"], "train")) if n.endswith(".npz"))
    train_set = EvUAVStream(cfg["root"], "train", net_cfg, names[:int(cfg["evidence_streams"])])
    lengths = [int(train_set[i].n_windows) for i in range(len(train_set))]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    out = train_epoch(net, heads, train_set, lengths, optimizer, cfg, net_cfg, q99, device,
                      np.random.RandomState(int(cfg["seed"])), frontend=frontend, comm=comm, max_chunks=2)
    train_seconds = time.perf_counter() - t0
    with_grad = [name for name, p in net.named_parameters() if p.grad is not None]
    if is_joint(cfg):
        lacking = [name for name, p in net.named_parameters() if p.grad is None and not name.startswith("readout.")]
        if lacking:
            raise RuntimeError("骨干参数没有梯度: %s" % lacking[:5])
    elif with_grad:
        raise RuntimeError("冻结的骨干出现了梯度: %s" % with_grad[:5])
    missing = [name for name, p in heads.named_parameters() if p.grad is None]
    if missing:
        raise RuntimeError("头的参数没有梯度: %s" % missing)
    seq = EvUAVStream(cfg["root"], "val", net_cfg)[0]
    heads.eval()
    net.eval()
    with torch.no_grad():
        t1 = time.perf_counter()
        step = evaluate_sequence(net, heads, seq, cfg, dict(net_cfg, input_device="cpu"), q99, device, "step",
                                 frontend=frontend)
        step_seconds = time.perf_counter() - t1
        layer = evaluate_sequence(net, heads, seq, cfg, net_cfg, q99, device, "layer", frontend=frontend)
    thr = np.float32(net_cfg["threshold"])
    gap = {}
    for name in readout_names(cfg):
        a, b = step["probs"][name], layer["probs"][name]
        gap[name] = {"max_abs": float(np.abs(a - b).max()),
                     "decision_flip_frac": float(((a >= thr) != (b >= thr)).mean())}
    report = {"backbone": cfg["evidence_backbone"], "input": cfg["evidence_input"], "train": out,
              "train_seconds": train_seconds, "step_eval_seconds": step_seconds,
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
    root = Path(cfg["evidence_save_root"])
    root.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics.jsonl"
    if metrics_path.exists():
        raise RuntimeError("输出目录已有 metrics.jsonl，请换目录: %s" % root)
    net, heads, net_cfg, q99, frontend = setup(cfg, device, "layer", "gpu", calibrate_dir=root)
    joint = is_joint(cfg)
    comm = CommBudget.from_config(cfg, net) if joint else None
    params = list(heads.parameters()) + (list(net.parameters()) if joint else [])
    V5.write_json(root / "run_config_train.json", {
        "config": cfg, "net_config": net_cfg, "head_parameters": head_parameters(heads),
        "backbone_parameters": int(sum(p.numel() for p in net.parameters())) if joint else 0,
        "input_channels": input_channels(cfg, frontend),
        "frontend_features": frontend.feature_names() if frontend is not None else None,
        "backbone_sha256": file_sha256(cfg["backbone_checkpoint"]), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"})
    optimizer = torch.optim.Adam(params, lr=float(cfg["evidence_lr"]))
    train_set = EvUAVStream(cfg["root"], "train", net_cfg)
    val_set = EvUAVStream(cfg["root"], "val", net_cfg)
    lengths = [int(train_set[i].n_windows) for i in range(len(train_set))]
    seed, epochs = int(cfg["seed"]), int(cfg["evidence_epochs"])
    print("V2/V3 判决层接 V5：骨干 %s、输入 %s（%d 通道），头参数 %d | 训练集 %d 条录像 %d 窗，%d 条流、块 %d 窗%s" % (
        cfg["evidence_backbone"], cfg["evidence_input"], input_channels(cfg, frontend), head_parameters(heads),
        len(lengths), sum(lengths), int(cfg["evidence_streams"]), int(cfg["evidence_chunk"]),
        "" if comm is None else "，消息约束：每窗预算 %.4g、违约比例上限 %.0f%%" % (comm.budget, 100 * comm.alpha)),
        flush=True)
    best = -float("inf")
    for epoch in range(epochs):
        lr = linear_epoch_lr(epoch, epochs, float(cfg["evidence_lr"]), float(cfg["evidence_lr_end"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        out = train_epoch(net, heads, train_set, lengths, optimizer, cfg, net_cfg, q99, device,
                          np.random.RandomState(seed * 1000 + epoch + 7), frontend=frontend, comm=comm)
        train_seconds = time.perf_counter() - t0
        val_iou, val_nll = quick_val(net, heads, val_set, cfg, net_cfg, q99, device, frontend=frontend)
        ev = max(out["events"], 1)
        record = {"epoch": epoch, "lr": lr, "train_seconds": train_seconds,
                  "epoch_seconds": time.perf_counter() - t0, "updates": out["updates"],
                  "bce_per_event": out["bce_sum"] / ev, "nll_per_event": out["nll_sum"] / ev,
                  "val_net_iou": val_iou, "val_nll_per_event": val_nll, "comm": out.get("comm"),
                  "peak_memory_gib": V5.peak_memory_gib(device)}
        if val_iou > best:
            best = val_iou
            save_checkpoint(root / checkpoint_name(cfg, "best"), net, heads, epoch, val_iou, cfg)
        save_checkpoint(root / checkpoint_name(cfg, "last"), net, heads, epoch, val_iou, cfg)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        c = out.get("comm")
        print("epoch %d lr %.2e | bce %.4f nll %.4f（每事件）| val net IoU %.4f val nll %.4f | 更新 %d 次 | "
              "%.0fs（训练 %.0fs）| 显存峰值 %.2f GiB%s" % (
                  epoch, lr, record["bce_per_event"], record["nll_per_event"], val_iou, val_nll, out["updates"],
                  record["epoch_seconds"], train_seconds, record["peak_memory_gib"] or 0.0,
                  "" if c is None else " | 每窗消息 均值 %.4g p95 %.4g 违约 %.1f%% mu %.4f" % (
                      c["messages_est_mean"] or 0.0, c["messages_window_p95"] or 0.0,
                      100.0 * (c["violation_mean"] or 0.0), c["mu_end"])), flush=True)
    V5.write_json(root / "summary.json", {"best_val_net_iou": best, "epochs": epochs})
    print("TRAINING FINISHED:", root, flush=True)


def mode_eval(args, cfg):
    """逐窗执行跑完整划分：V5、net、fused_d1..5、pub 的四个指标与纠正 / 破坏数，pub 的发布统计。"""
    device = torch.device(args.device)
    root = Path(cfg["evidence_save_root"])
    out_path = root / ("eval_evidence_%s.json" % args.split)
    if out_path.exists() and not args.overwrite:
        raise RuntimeError("结果已存在: %s（使用 --overwrite 覆盖）" % out_path)
    net, heads, net_cfg, q99, frontend = setup(cfg, device, "step", "cpu")
    ckpt = torch.load(str(root / checkpoint_name(cfg, "best")), map_location=device)
    if ckpt["backbone_sha256"] != file_sha256(cfg["backbone_checkpoint"]):
        raise RuntimeError("训练时的 V5 参照与当前 backbone_checkpoint 不是同一个文件")
    heads.load_state_dict(ckpt["heads"] if "heads" in ckpt else ckpt["model"])   # 第一轮（frozen）的权重存在 model 下
    heads.eval()
    reference = None
    if is_joint(cfg):
        net.load_state_dict(ckpt["model"])
        if not bool(net.gain_calibrated):
            raise RuntimeError("checkpoint 中的增益未校准")
        net.eval()
        ref_net, ref_cfg, ref_q99 = load_backbone(cfg, device, "step", "cpu")
        reference = (ref_net, ref_cfg, ref_q99)
    dataset = EvUAVStream(cfg["root"], args.split, net_cfg)
    window_ms = float(net_cfg["window_ms"])
    names = readout_names(cfg)
    seqs, v5, probs = [], [], {name: [] for name in names}
    ages, reasons = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(len(dataset)):
            seq = dataset[i]
            res = evaluate_sequence(net, heads, seq, cfg, net_cfg, q99, device, "step", frontend=frontend,
                                    reference=reference)
            seqs.append({"name": seq.name, "t": seq.t, "x": seq.x, "y": seq.y, "label": seq.label,
                         "target_id": seq.target_id, "groups": event_groups(seq, window_ms)})
            v5.append(res["v5"])
            for name in names:
                probs[name].append(res["probs"][name])
            ages.append(res["extra"]["age_pub"])
            reasons.append(res["extra"]["reason_pub"])
    seconds = time.perf_counter() - t0
    thr = float(net_cfg["threshold"])
    results = {"split": args.split, "backbone": cfg["evidence_backbone"], "input": cfg["evidence_input"],
               "backbone_checkpoint": cfg["backbone_checkpoint"], "epoch": int(ckpt["epoch"]),
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
