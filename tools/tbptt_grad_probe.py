"""TBPTT 梯度探针：同一份权重、同一批训练序列，比较不同截断长度 k 算出的梯度（只算梯度，不更新参数）。

要回答的问题：k 越短效果越好（16 > 32 > 64），是因为长程时间梯度本身没用 / 是噪声，还是另有原因
（例如长程项把总范数撑大、grad_clip=1.0 裁剪后有效学习率变小）。

对每条序列、每个 k，按训练时的同一口径（per_sequence：片段边界 detach，损失除以整条序列事件数）
计算梯度 g_k；片段从第 0 窗起按 k 对齐（first_len = k），不随机化，保证可复现。参照为 g_ref
（默认 ref_k = 160 = 整段 8 s 的精确 BPTT）。k = 1 即"不做时间回传，只保留空间方向梯度"（SLTT 式）。

报告（逐序列取平均，另报"把所有序列梯度相加后"的方向，后者更接近一轮训练的平均更新方向）：
    |g_k|                 裁剪前的总范数（大于 grad_clip 就会被裁剪；超裁剪占比 = 这样的序列比例）
    |g_k| / |g_ref|       范数比
    cos(g_k, g_ref)       与精确梯度的夹角余弦；接近 1 说明截断几乎不改变方向
    cos_sum               所有序列梯度之和的余弦
    temporal_share        |g_k - g_1| / |g_k|：时间方向梯度在 g_k 中的占比
    分组余弦              enc1..dec1、up*、readout 各自与 g_ref 的余弦（看截断影响落在哪些层）

用法（服务器，在仓库根目录）:
    python tools/tbptt_grad_probe.py --checkpoint log/stream_v1_seed37/best_val_iou_seed37.pt \
        --out log/tbptt_probe/v1_k16_best.json [--n-seq 8] [--ks 1 2 4 8 16 32 64] [--ref-k 160]
显存不够跑 ref_k 时会报出并改用能跑通的最大 k 作参照（结果里记录 ref_k_used）。
"""
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")   # 必须在导入 torch 之前设置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_stream_v1 as T  # noqa: E402
from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from utils.stream_common import load_flat_config, select_subset  # noqa: E402


def flat_grads(net):
    """按顶层模块分组取出梯度，返回 {组名: float64 numpy 向量}；没有梯度的参数记为 0。"""
    groups = {}
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        group = name.split(".")[0]
        g = np.zeros(param.numel()) if param.grad is None else param.grad.detach().double().cpu().numpy().ravel()
        groups.setdefault(group, []).append(g)
    return {k: np.concatenate(v) for k, v in groups.items()}


def concat_all(grads):
    """把分组梯度按组名排序后拼成一个向量。"""
    return np.concatenate([grads[k] for k in sorted(grads)])


def cosine(a, b):
    """两向量的余弦；任一为零向量时返回 NaN。"""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def compare(grads, ref):
    """g 与参照 g_ref 的比较：总余弦、范数比、各组余弦。"""
    a, b = concat_all(grads), concat_all(ref)
    nb = float(np.linalg.norm(b))
    return {"cos": cosine(a, b), "norm": float(np.linalg.norm(a)),
            "norm_ratio": float(np.linalg.norm(a)) / nb if nb > 0 else float("nan"),
            "group_cos": {k: cosine(grads[k], ref[k]) for k in sorted(grads)}}


def temporal_share(grads, spatial):
    """|g_k - g_1| / |g_k|：g_1 只含空间方向梯度，差值即长度 2..k 的时间路径贡献。"""
    a, s = concat_all(grads), concat_all(spatial)
    na = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - s)) / na if na > 0 else float("nan")


def is_oom(error):
    """torch 1.9 没有 OutOfMemoryError 类型，按报错文本识别显存不足。"""
    return "out of memory" in str(error).lower()


def sequence_gradient(net, seq, cfg, q99, pos_weight, device, k):
    """按 per_sequence 口径、片段从第 0 窗起按 k 对齐，计算一条序列的梯度（不更新参数）。"""
    cfg_k = dict(cfg, tbptt_k=int(k), optimizer_step="per_sequence")
    net.zero_grad(set_to_none=True)
    out = T.train_sequence(net, seq, None, cfg_k, q99, pos_weight, device, None, first_len=int(k))
    return flat_grads(net), out


def mean_of(values):
    """忽略 NaN 的均值；全为 NaN 时返回 NaN。"""
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def summarize(per_seq, sums, ks, ref_k, clip):
    """逐序列结果取平均，并比较各 k 的"梯度之和"与参照之和。"""
    table = []
    for k in ks:
        rows = [r["k"][str(k)] for r in per_seq if str(k) in r["k"]]
        if not rows:
            continue
        entry = {"k": k, "n_seq": len(rows),
                 "norm_mean": mean_of([r["norm"] for r in rows]),
                 "norm_ratio_mean": mean_of([r["norm_ratio"] for r in rows]),
                 "cos_mean": mean_of([r["cos"] for r in rows]),
                 "temporal_share_mean": mean_of([r.get("temporal_share", float("nan")) for r in rows]),
                 "clipped_frac": mean_of([1.0 if r["norm"] > clip else 0.0 for r in rows]),
                 "group_cos_mean": {}}
        for group in rows[0]["group_cos"]:
            entry["group_cos_mean"][group] = mean_of([r["group_cos"][group] for r in rows])
        if k in sums and ref_k in sums:
            entry["cos_sum"] = cosine(concat_all(sums[k]), concat_all(sums[ref_k]))
        table.append(entry)
    return table


def main():
    parser = argparse.ArgumentParser(description="TBPTT gradient probe (no parameter update)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/evisseg_stream_v1.yaml", help="只取数据根目录")
    parser.add_argument("--n-seq", type=int, default=8, help="固定种子抽取的训练序列条数")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--ref-k", type=int, default=160, help="参照截断长度，160 = 整段精确 BPTT")
    parser.add_argument("--execution", choices=T.EXECUTIONS, default=None, help="默认沿用 checkpoint 的设置")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = dict(ckpt["config"])
    cfg["root"] = load_flat_config(args.config)["root"]
    if args.execution is not None:
        cfg["execution"] = args.execution
    T.seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    stats = ckpt["stats"]
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    pos_weight = min(float(stats["train_neg"]) / float(stats["train_pos"]), float(cfg["pos_weight_cap"]))
    net = T.build_net(cfg, device)
    net.load_state_dict(ckpt["model"])
    if not bool(net.gain_calibrated):
        raise RuntimeError("checkpoint 中的增益未校准")
    net.train()

    ks = sorted(set(int(k) for k in args.ks))
    if min(ks) < 1:
        raise ValueError("--ks 必须 >= 1")
    ref_k = int(args.ref_k)
    names = select_subset(T.train_file_names(cfg), int(args.n_seq), int(cfg["seed"]) + 3000)
    dataset = EvUAVStream(cfg["root"], "train", cfg, names)
    print("checkpoint %s | epoch %s | 训练时 k=%s | 序列 %d 条 | ks %s | 参照 k=%d | 执行 %s" % (
        args.checkpoint, ckpt.get("epoch"), cfg.get("tbptt_k"), len(dataset), ks, ref_k,
        T.speed_options(cfg)[0]), flush=True)

    per_seq, sums, skipped = [], {}, {}
    for i in range(len(dataset)):
        seq = dataset[i]
        grads, seconds = {}, {}
        for k in ks + ([ref_k] if ref_k not in ks else []):
            if k in skipped:
                continue
            t0 = time.perf_counter()
            try:
                grads[k], _ = sequence_gradient(net, seq, cfg, q99, pos_weight, device, k)
            except RuntimeError as error:
                if not is_oom(error):
                    raise
                skipped[k] = str(error).splitlines()[0]
                net.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print("  k=%d 显存不足，跳过: %s" % (k, skipped[k]), flush=True)
                continue
            seconds[k] = time.perf_counter() - t0
            sums[k] = grads[k] if k not in sums else {g: sums[k][g] + grads[k][g] for g in grads[k]}
        ref_used = max(grads) if ref_k not in grads else ref_k
        record = {"sequence": seq.name, "ref_k_used": ref_used, "seconds": seconds, "k": {}}
        for k in ks:
            if k not in grads:
                continue
            entry = compare(grads[k], grads[ref_used])
            if 1 in grads:
                entry["temporal_share"] = temporal_share(grads[k], grads[1])
            record["k"][str(k)] = entry
        per_seq.append(record)
        print("[%d/%d] %s 参照 k=%d | " % (i + 1, len(dataset), seq.name, ref_used) + " ".join(
            "k%d:|g|=%.3f cos=%.3f" % (k, record["k"][str(k)]["norm"], record["k"][str(k)]["cos"])
            for k in ks if str(k) in record["k"]), flush=True)

    ref_used = ref_k if ref_k not in skipped else max(k for k in sums)
    table = summarize(per_seq, sums, ks, ref_used, float(cfg["grad_clip"]))
    print("\n%4s %9s %11s %9s %9s %10s %8s" % ("k", "|g|", "|g|/|gref|", "cos", "cos_sum", "时间占比", "超裁剪占比"))
    for row in table:
        print("%4d %9.4f %11.4f %9.4f %9.4f %10.4f %8.2f" % (
            row["k"], row["norm_mean"], row["norm_ratio_mean"], row["cos_mean"], row.get("cos_sum", float("nan")),
            row["temporal_share_mean"], row["clipped_frac"]))
    print("\n分组余弦（相对参照）:")
    for row in table:
        print("  k=%-3d " % row["k"] + " ".join("%s %.3f" % (g, v) for g, v in row["group_cos_mean"].items()))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    T.write_json(args.out, {"checkpoint": args.checkpoint, "epoch": ckpt.get("epoch"),
                            "trained_tbptt_k": cfg.get("tbptt_k"), "ks": ks, "ref_k": ref_k,
                            "ref_k_used": ref_used, "skipped": skipped, "sequences": names,
                            "summary": table, "per_sequence": per_seq})
    print("PROBE FINISHED:", args.out, flush=True)


if __name__ == "__main__":
    main()
