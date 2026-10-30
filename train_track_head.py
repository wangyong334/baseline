"""模块三：目标级航迹头（model/track_head.py）的训练与评估入口（Stream-SNN-Innovations.md 13.60）。

骨干（E3a 等已训练的流式网络）冻结、只读；只训练航迹头。每条录像从头到尾连续处理，骨干状态与航迹状态都跨片段延续。
模式：
    train   多流 TBPTT（与 train_stream_v1 的 trainer=streams 同一排法：B 条流、每段 k 窗、片段边界 detach），
            每轮在 val 上同时评估骨干原输出（base）与航迹头输出（track），按 track 的 val IoU 挑最好的一轮
    eval    在 val / test 上评估：四个指标（IoU / ACC / Pd / Fa，与正式评估同一口径）、误差类别（tools/error_budget.py），
            以及事先定好的关口是否通过；可选保存逐事件预测（--dump-dir）与逐窗航迹（--tracks-dir，可视化用）
骨干一律用逐层执行（execution=layer）：与逐窗执行数学等价（单元测试已核对），这里只为速度；base 一列即用来核对复现。
用法见 configs/track_head_e3a.yaml 文件头。
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")   # 必须在导入 torch 之前设置

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_stream_v1 as T  # noqa: E402
from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from dataset.stream_source import make_window_source  # noqa: E402
from dataset.stream_windows import refill_by_index  # noqa: E402
from model.lif2d_stream import detach_states  # noqa: E402
from model.track_head import TrackHead  # noqa: E402
from utils.stream_common import linear_epoch_lr, load_flat_config  # noqa: E402
from utils.stream_metrics import FastRoc  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import error_budget  # noqa: E402

HEAD_KEYS = ("cell", "slots", "hidden", "history", "gate_sigma", "gate_max_px", "gate_floor", "birth_floor",
             "max_births", "death_exist", "max_coast", "match_min_px", "corr_hidden", "r_min")


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


class TidSources(T.StreamSources):
    """与 train_stream_v1.StreamSources 相同，另外记住录像本身（取 target_id 用）。"""

    def __init__(self, dataset, cfg, q99, device, dtype):
        super(TidSources, self).__init__(dataset, cfg, q99, device, dtype, None)
        self.seqs = {}

    def get(self, index):
        if index not in self.sources:
            seq = self.dataset[index]
            self.seqs[index] = seq
            self.sources[index] = make_window_source(seq, self.cfg, self.q99, self.device, self.dtype)
            self.bounds[index] = seq.bounds
        return self.sources[index]

    def release(self, index):
        super(TidSources, self).release(index)
        self.seqs.pop(index, None)


def assemble_with_tid(sources, pieces, steps, cfg, device, dtype):
    """train_stream_v1.assemble_streams 的同口径拼批，另返回逐事件 target_id。"""
    batch = len(pieces)
    x = torch.zeros((steps, batch, T.num_input_channels(cfg["time_bins"]), int(cfg["pad_height"]),
                     int(cfg["pad_width"])), dtype=dtype, device=device)
    keep = torch.ones((steps, batch), dtype=dtype, device=device)
    parts, labels, tids, bg = {key: [] for key in ("t", "b", "y", "x", "p", "t_local")}, [], [], None
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
            tids.append(torch.from_numpy(np.asarray(sources.seqs[index].target_id[idx], dtype=np.float32)).to(device))
    if labels:
        events = {key: torch.cat(value) for key, value in parts.items()}
        labels, tid = torch.cat(labels), torch.cat(tids)
    else:
        long_empty = torch.zeros(0, dtype=torch.long, device=device)
        float_empty = torch.zeros(0, dtype=dtype, device=device)
        events = {"t": long_empty, "b": long_empty, "y": long_empty, "x": long_empty,
                  "p": float_empty, "t_local": float_empty}
        labels, tid = float_empty, float_empty
    if bg is not None:
        events["bg"] = bg
    return x, events, labels, keep, tid


def train_epoch(net, tap, head, dataset, lengths, optimizer, cfg, base, q99, pos_weight, device, rng, max_chunks=None):
    k, n_streams = int(cfg["tbptt_k"]), int(cfg["streams"])
    plan = T.plan_streams(lengths, n_streams, k, rng, cfg.get("stream_layout", "greedy"))
    item_ends = {(index, first + n) for _, items in plan for index, first, n in map(T.plan_item, items)}
    n_chunks = int(math.ceil(T.stream_total_windows(plan) / float(k)))
    if max_chunks is not None:
        n_chunks = min(n_chunks, int(max_chunks))
    dtype = next(net.parameters()).dtype
    sources = TidSources(dataset, base, q99, device, dtype)
    weight = torch.tensor([pos_weight], dtype=dtype, device=device)
    lam_e, lam_p, clip = float(cfg["lambda_exist"]), float(cfg["lambda_pos"]), float(cfg["grad_clip"])
    sums = {"event": 0.0, "exist": 0.0, "pos": 0.0, "events": 0, "exist_n": 0, "pos_n": 0, "updates": 0}
    norms = []
    bstates, hstate = None, None
    head.train()
    for c in range(n_chunks):
        pieces = T.stream_pieces(plan, c * k, (c + 1) * k)
        x, events, labels, keep, tid = assemble_with_tid(sources, pieces, k, base, device, dtype)
        with torch.no_grad():
            z0, bstates, _ = net.forward_chunk(x, events, bstates, keep=keep)
            feat = tap.feat
        z, hstate, aux = head.forward_chunk(events, z0, feat, hstate, k, keep=keep, labels=labels, tid=tid)
        if labels.numel():
            n_ev = int(labels.numel())
            ev_loss = F.binary_cross_entropy_with_logits(z, labels, pos_weight=weight, reduction="sum") / n_ev
            loss = ev_loss
            if aux["exist_n"]:
                loss = loss + lam_e * aux["exist_sum"] / aux["exist_n"]
            if aux["pos_n"]:
                loss = loss + lam_p * aux["pos_sum"] / aux["pos_n"]
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
            sums["updates"] += 1
        bstates = detach_states(bstates)
        hstate = head.detach_state(hstate)
        for stream in pieces:
            for index, lo, hi, _, _ in stream:
                if (index, hi) in item_ends:
                    sources.release(index)
    return {"event_loss": sums["event"] / max(sums["events"], 1),
            "exist_loss": sums["exist"] / max(sums["exist_n"], 1),
            "pos_loss": sums["pos"] / max(sums["pos_n"], 1),
            "updates": sums["updates"], "chunks": n_chunks,
            "grad_norm_median": float(np.median(norms)) if norms else 0.0}


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------


def run_split(net, tap, head, base, q99, device, split, names=None, full=False, dump_dir=None, tracks_dir=None):
    """整段连续推理一个划分；返回 base（骨干原输出）与 track（航迹头输出）的指标。"""
    dataset = EvUAVStream(base["root"], split, base, names)
    thr = float(base["threshold"])
    chunk = int(base.get("eval_chunk", 32))
    counts = {"base": np.zeros(3, dtype=np.int64), "track": np.zeros(3, dtype=np.int64)}   # TP, FP, FN
    rocs = {kind: FastRoc(base["pd_detT"], base["correct_thresh"], base["height"], base["width_px"])
            for kind in counts} if full else None
    rows = {"base": [], "track": []}
    dtype = next(net.parameters()).dtype
    was_training = head.training
    head.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            seq = dataset[i]
            source = make_window_source(seq, base, q99, device, dtype)
            bstates, hstate = None, head.init_state(1, device)
            idx_parts, p0_parts, p1_parts, log = [], [], [], ([] if tracks_dir else None)
            for start in range(0, seq.n_windows, chunk):
                end = min(start + chunk, seq.n_windows)
                x, events, _, idx = source.chunk(start, end)
                z0, bstates, _ = net.forward_chunk(x, events, bstates)
                z, hstate, _ = head.forward_chunk(events, z0, tap.feat, hstate, end - start, log=log)
                idx_parts.append(idx)
                p0_parts.append(torch.sigmoid(z0).float().cpu().numpy())
                p1_parts.append(torch.sigmoid(z).float().cpu().numpy())
            probs = {"base": refill_by_index(seq.n_events, idx_parts, p0_parts),
                     "track": refill_by_index(seq.n_events, idx_parts, p1_parts)}
            pos = seq.label > 0.5
            for kind, prob in probs.items():
                pred = prob >= thr
                counts[kind] += [int((pred & pos).sum()), int((pred & ~pos).sum()), int((~pred & pos).sum())]
                if full:
                    rocs[kind].update(seq.t, prob, seq.target_id, seq.label, seq.x, seq.y, thr)
                rows[kind].append(error_budget.analyse_arrays(
                    seq.name, seq.x, seq.y, seq.t, pos, prob, seq.target_id, thr, 10, float(base["window_ms"]),
                    (1,), int(base["height"]), int(base["width_px"]))[0])
            if dump_dir:
                np.savez(os.path.join(dump_dir, seq.name),
                         locs=np.stack([np.zeros(seq.n_events, dtype=np.int64), seq.x, seq.y, seq.t], 1),
                         labels=seq.label.astype(np.float32), probabilities=probs["track"].astype(np.float32),
                         base_probabilities=probs["base"].astype(np.float32),
                         target_id=seq.target_id.astype(np.float64))
            if tracks_dir:
                np.savez(os.path.join(tracks_dir, seq.name),
                         **{key: np.stack([w[key][0] for w in log]) for key in ("active", "pos", "vel", "ex")})
    head.train(was_training)
    out = {"split": split, "n_sequences": len(dataset)}
    for kind, (tp, fp, fn) in counts.items():
        res = {"iou": tp / max(tp + fp + fn, 1), "acc": tp / max(tp + fn, 1), "tp": int(tp), "fp": int(fp),
               "fn": int(fn)}
        if full:
            pd, fa = rocs[kind].result()
            res.update(pd=float(pd), fa=float(fa))
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
    seed, epochs = int(cfg["seed"]), int(cfg["epochs"])
    print("航迹头训练：骨干 %s（冻结）| 航迹头参数 %d | %d 条流、k=%d | 训练集 %d 条录像 %d 窗 | pos_weight %.2f" % (
        cfg["base_checkpoint"], sum(p.numel() for p in head.parameters()), int(cfg["streams"]), int(cfg["tbptt_k"]),
        len(lengths), sum(lengths), pos_weight), flush=True)
    best = -float("inf")
    for epoch in range(epochs):
        lr = linear_epoch_lr(epoch, epochs, float(cfg["lr"]), float(cfg["lr_end"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        out = train_epoch(net, tap, head, train_set, lengths, optimizer, cfg, base, q99, pos_weight, device,
                          np.random.RandomState(seed * 1000 + epoch + 7), args.max_chunks)
        train_s = time.perf_counter() - t0
        val = run_split(net, tap, head, base, q99, device, "val")
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
        print("epoch %d lr %.2e | 损失 事件 %.4f 存在 %.4f 位置 %.4f | val IoU base %.4f -> track %.4f | "
              "拖尾误报 %d -> %d 断续漏检 %d -> %d 目标附近误报 %d -> %d | %.0fs（训练 %.0fs）| 显存峰值 %.2f GiB" % (
                  epoch, lr, out["event_loss"], out["exist_loss"], out["pos_loss"], b["iou"], t["iou"],
                  b["fp_cat"]["tail"], t["fp_cat"]["tail"], b["fn_cat"]["gap"], t["fn_cat"]["gap"],
                  b["fp_cat"]["near"], t["fp_cat"]["near"], record["epoch_seconds"], train_s,
                  record["peak_memory_gib"] or 0.0), flush=True)
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
    result = run_split(net, tap, head, base, q99, device, args.split, full=True, dump_dir=args.dump_dir,
                       tracks_dir=args.tracks_dir)
    result.update(checkpoint=args.checkpoint, epoch=ckpt.get("epoch"), base_checkpoint=cfg["base_checkpoint"],
                  gate=gate_check(result, cfg))
    name = Path(args.checkpoint).stem
    path = Path(args.checkpoint).parent / ("eval_%s_%s.json" % (args.split, name))
    T.write_json(path, result)
    for kind in ("base", "track"):
        r = result[kind]
        print("%s %-5s IoU %.4f ACC %.4f Pd %.4f Fa %.3e | 拖尾误报 %d 断续漏检 %d 目标附近误报 %d" % (
            args.split, kind, r["iou"], r["acc"], r["pd"], r["fa"], r["fp_cat"]["tail"], r["fn_cat"]["gap"],
            r["fp_cat"]["near"]), flush=True)
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
