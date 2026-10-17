"""整段录像（EV-UAV NPZ 格式，如 EV-Flying x4、EBSSA DAVIS）-> K5 基线用的 8 s 块（10-08 协议：K5 按原样每 8 s 一块训练 / 推理）。

切块规则（每段录像，事件顺序不变，不丢不增）：
    块号 b = floor(t_ms / block_ms)，只写有事件的块；块内时间 t' = t_ms - b * block_ms（0..block_ms-1）
    ev_loc   int64   [N,3]  x, y, t'                    （K5 体素坐标，画布 352x288x8192 写死在 dataset/basedataset.py）
    evs_norm float32 [N,6]  x/W, y/H, t'/8192, p, label, target_id（W、H 为 K5 归一化尺寸，EV-Flying x4 320x192、EBSSA 256x192）
    cls      int8    [N]    原文件有就照抄
文件名 <录像名>_b<块号三位>.npz；manifest.json 记录每块的来源录像、块号、时间偏移、事件数，供拼回整段评测用。
写完后把每段录像的各块按块号拼接，核对与原录像逐事件相同（x、y、t、p、label、target_id）。

用法（在仓库根目录）:
    python tools/make_blocks.py --src /path/evflying_x4 --dst /path/evflying_x4_blocks8s --k5-width 320 --k5-height 192
"""
import argparse
import json
import os
import sys
import time

import numpy as np

BLOCK_MS = 8000
T_NORM = 8192.0


def split_blocks(ev_loc, evs_norm, block_ms=BLOCK_MS):
    """返回 [(块号, 事件下标 slice)]；要求 t 单调不减（项目的转换工具都保证这一点）。"""
    t = ev_loc[:, 2].astype(np.int64)
    if t.size and (t.min() < 0 or np.any(np.diff(t) < 0)):
        raise ValueError("时间为负或不单调")
    block = t // int(block_ms)
    edges = np.flatnonzero(np.r_[True, block[1:] != block[:-1], True])
    return [(int(block[a]), slice(int(a), int(b))) for a, b in zip(edges[:-1], edges[1:])]


def block_arrays(ev_loc, evs_norm, b, sl, width, height, block_ms=BLOCK_MS):
    loc = ev_loc[sl].astype(np.int64).copy()
    loc[:, 2] -= b * int(block_ms)
    norm = np.stack([loc[:, 0] / float(width), loc[:, 1] / float(height), loc[:, 2] / T_NORM,
                     evs_norm[sl, 3], evs_norm[sl, 4], evs_norm[sl, 5]], 1).astype(np.float32)
    return loc, norm


def main():
    parser = argparse.ArgumentParser(description="whole recordings -> 8 s blocks for the K5 baseline")
    parser.add_argument("--src", required=True, help="整段录像根目录（train/ val/ test/）")
    parser.add_argument("--dst", required=True)
    parser.add_argument("--k5-width", type=int, required=True)
    parser.add_argument("--k5-height", type=int, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    t0 = time.time()
    manifest = {"rule": "block = floor(t_ms / %d); t' = t_ms - block * %d; empty blocks skipped" % (BLOCK_MS, BLOCK_MS),
                "evs_norm_columns": ["x/%d" % args.k5_width, "y/%d" % args.k5_height, "t'/%d" % int(T_NORM),
                                     "p", "label", "target_id"],
                "source": os.path.abspath(args.src), "splits": {}}
    for split in args.splits:
        src_dir, dst_dir = os.path.join(args.src, split), os.path.join(args.dst, split)
        os.makedirs(dst_dir, exist_ok=False)
        entries = []
        for name in sorted(n for n in os.listdir(src_dir) if n.endswith(".npz")):
            with np.load(os.path.join(src_dir, name)) as z:
                ev_loc, evs_norm = np.asarray(z["ev_loc"]), np.asarray(z["evs_norm"])
                cls = np.asarray(z["cls"]) if "cls" in z.files else None
            stem = name[:-4]
            parts = []
            for b, sl in split_blocks(ev_loc, evs_norm):
                loc, norm = block_arrays(ev_loc, evs_norm, b, sl, args.k5_width, args.k5_height)
                out = "%s_b%03d.npz" % (stem, b)
                fields = {"ev_loc": loc, "evs_norm": norm}
                if cls is not None:
                    fields["cls"] = cls[sl]
                np.savez(os.path.join(dst_dir, out), **fields)
                parts.append((b, out))
                entries.append({"file": out, "recording": name, "block": b, "offset_ms": b * BLOCK_MS,
                                "events": int(loc.shape[0]), "target_events": int(norm[:, 4].sum())})
            # 读回各块按块号拼接，核对与原录像逐事件相同
            loc_all, norm_all = [], []
            for b, out in parts:
                with np.load(os.path.join(dst_dir, out)) as z:
                    l = np.asarray(z["ev_loc"]).copy()
                    l[:, 2] += b * BLOCK_MS
                    loc_all.append(l)
                    norm_all.append(np.asarray(z["evs_norm"])[:, 3:6])
            if not (np.array_equal(np.concatenate(loc_all), ev_loc.astype(np.int64))
                    and np.array_equal(np.concatenate(norm_all), evs_norm[:, 3:6].astype(np.float32))):
                raise RuntimeError("%s 各块拼回后与原录像不一致" % name)
            print("%-5s %-34s %3d 块 %9d 事件" % (split, name, len(parts), ev_loc.shape[0]), flush=True)
        manifest["splits"][split] = entries
        print("%-5s 共 %d 块" % (split, len(entries)), flush=True)
    with open(os.path.join(args.dst, "manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=1, ensure_ascii=False)
    print("BLOCKS FINISHED: %s（%.0f s）" % (args.dst, time.time() - t0), flush=True)


if __name__ == "__main__":
    sys.exit(main())
