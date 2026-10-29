"""误检来源分解（只读 --dump-dir 的逐事件预测，CPU，numpy 即可）。13.51 的口径：
    附近        离任一标注目标事件 <= radius 像素，且时间差 <= dt_ms（目标边界、光晕、可能的漏标）
    路径上时间不符  在某个目标事件 radius 像素内，但时间差都 > dt_ms（轨迹起止、交叉、停留处）
    远离        离所有目标事件都 > radius 像素（背景噪声、闪烁）
另给出：把"附近"的误检都当作标注上限、不计入误检时的序列 IoU（标注上限下的 IoU）。
用法：python tools/fp_source.py --dump-dir e3b=run/dump/e3b_test_L0 --names test_022 test_023 --out run/fp_source.json
"""
import argparse
import json
import os

import numpy as np


def decompose(path, threshold, radius, dt_ms, height=260, edge_rows=15, chunk=2048):
    d = np.load(path)
    x, y, t = (d["locs"][:, i].astype(np.float64) for i in (1, 2, 3))
    pos = d["labels"] > 0.5
    pred = d["probabilities"] >= threshold
    fp = pred & ~pos
    px, py, pt = x[pos], y[pos], t[pos]
    fx, fy, ft = x[fp], y[fp], t[fp]
    near = np.zeros(fx.size, bool)
    path_ = np.zeros(fx.size, bool)
    for s in range(0, fx.size, chunk):
        e = slice(s, s + chunk)
        close = np.maximum(np.abs(fx[e, None] - px[None]), np.abs(fy[e, None] - py[None])) <= radius
        path_[e] = close.any(1)
        near[e] = (close & (np.abs(ft[e, None] - pt[None]) <= dt_ms)).any(1)
    tp = int((pred & pos).sum()); fn = int((~pred & pos).sum()); n = int(fx.size)
    far = int((~path_).sum()); n_near = int(near.sum())
    bottom = int((fy >= height - edge_rows).sum())                 # 下边缘误检（10-29：画布补零边）
    return {"name": os.path.splitext(os.path.basename(path))[0], "fp": n, "tp": tp, "fn": fn,
            "near": n_near, "path_wrong_time": int((path_ & ~near).sum()), "far": far, "bottom_edge": bottom,
            "iou": tp / (tp + n + fn) if tp + n + fn else None,
            "iou_without_near_fp": tp / (tp + n - n_near + fn) if tp + n - n_near + fn else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dump-dir", action="append", required=True, help="名称=目录")
    parser.add_argument("--names", nargs="*", default=None, help="序列名（不带 .npz），默认全部")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--radius", type=float, default=10.0, help="像素（切比雪夫距离）")
    parser.add_argument("--dt-ms", type=float, default=500.0, help="事件时间戳单位为毫秒")
    parser.add_argument("--height", type=int, default=260, help="传感器高度（像素），用于统计最下面 15 行的误检")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = {}
    print("%-5s %-9s %6s %6s %6s %6s | %5s %6s %5s | %6s %8s %6s" % (
        "模型", "序列", "TP", "FP", "FN", "IoU", "附近", "路径上", "远离", "远离数", "去附近IoU", "下边缘"))
    for item in args.dump_dir:
        name, _, folder = item.rpartition("=")
        name = name or os.path.basename(os.path.normpath(folder))
        rows = []
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".npz") or f.endswith(".with_p.npz"):
                continue
            if args.names and os.path.splitext(f)[0] not in args.names:
                continue
            r = decompose(os.path.join(folder, f), args.threshold, args.radius, args.dt_ms, args.height)
            rows.append(r)
            k = max(r["fp"], 1)
            print("%-5s %-9s %6d %6d %6d %6.3f | %4.0f%% %5.0f%% %4.0f%% | %6d %8.3f %6d" % (
                name, r["name"], r["tp"], r["fp"], r["fn"], r["iou"], 100 * r["near"] / k,
                100 * r["path_wrong_time"] / k, 100 * r["far"] / k, r["far"], r["iou_without_near_fp"], r["bottom_edge"]))
        report[name] = rows
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
