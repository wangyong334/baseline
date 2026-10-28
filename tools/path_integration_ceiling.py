"""方向 B 的上限检查：沿真实轨迹积累证据，能不能在极密噪声下把目标和噪声分开、要几窗（Inn 13.47）。

原理：雷达的长时间积累 / 检测前跟踪。固定位置上的积累受目标停留时间限制（停留定律）；沿目标轨迹积累则不受限，
信噪比随积累窗数 K 按 sqrt(K) 增长。本工具用标签给出的真实轨迹作"神谕"，测两种积累方式的可分性上限：
    path   沿目标过去 K 窗的真实位置积累：sum_j count(box at c(k-j), window k-j)
    fixed  固定在目标当前位置积累：      sum_j count(box at c(k),   window k-j)
零假设：把整条（K 窗）轨迹平移到远离所有目标的随机位置，用同样的框、同样的积累方式。
指标：配对 AUC = 每个样本的目标积累量大于它自己的 M 个零假设积累量的比例（平局记 0.5），对样本取平均；
      另给 d' = (目标 - 零假设均值) / 零假设标准差 的中位数。
只用标签与事件计数，不涉及网络；结果只作原理检验，不进方法。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.stream_windows import load_npz_events  # noqa: E402

H, W, WINDOW_MS, N_WINDOWS, TIME_BINS = 260, 346, 50, 160, 5
KS = (1, 2, 4, 8, 16, 32)


def integral_images(seq):
    """每窗全部事件的计数图的积分图 [n_windows, H+1, W+1]（int32）。"""
    out = np.zeros((N_WINDOWS, H + 1, W + 1), dtype=np.int32)
    for k in range(N_WINDOWS):
        idx = seq.window_index(k)
        counts = np.zeros((H, W), dtype=np.int32)
        np.add.at(counts, (seq.y[idx], seq.x[idx]), 1)
        out[k, 1:, 1:] = counts.cumsum(0).cumsum(1)
    return out


def box_sum(ii, k, cx, cy, hx, hy):
    """窗 k、中心 (cx, cy)、半宽 (hx, hy) 的方框内事件数（框截到画面内）。参数可为数组。"""
    x0 = np.clip(np.round(cx - hx).astype(np.int64), 0, W)
    x1 = np.clip(np.round(cx + hx).astype(np.int64) + 1, 0, W)
    y0 = np.clip(np.round(cy - hy).astype(np.int64), 0, H)
    y1 = np.clip(np.round(cy + hy).astype(np.int64) + 1, 0, H)
    return ii[k, y1, x1] - ii[k, y0, x1] - ii[k, y1, x0] + ii[k, y0, x0]


def target_tracks(seq, min_events):
    """每个目标每窗的质心与半宽：{tid: {k: (cx, cy, hx, hy, n)}}；少于 min_events 个目标事件的窗不记。"""
    tracks = {}
    pos = seq.label == 1
    for k in range(N_WINDOWS):
        idx = seq.window_index(k)
        idx = idx[pos[idx]]
        if idx.size == 0:
            continue
        tids = seq.target_id[idx]
        for tid in np.unique(tids):
            sel = idx[tids == tid]
            if sel.size < min_events:
                continue
            xs, ys = seq.x[sel], seq.y[sel]
            hx = max(1.0, (xs.max() - xs.min()) / 2.0 + 0.5)
            hy = max(1.0, (ys.max() - ys.min()) / 2.0 + 0.5)
            tracks.setdefault(int(tid), {})[k] = (xs.mean(), ys.mean(), hx, hy, sel.size)
    return tracks


def evaluate_sequence(seq, n_null, rng, min_events, ks):
    ii = integral_images(seq)
    tracks = target_tracks(seq, min_events)
    centers = {k: [] for k in range(N_WINDOWS)}                      # 所有目标在各窗的位置（零假设要避开）
    for track in tracks.values():
        for k, (cx, cy, hx, hy, _) in track.items():
            centers[k].append((cx, cy, max(hx, hy)))
    result = {}
    for K in ks:
        rows = {"path": [], "fixed": []}
        for track in tracks.values():
            for k in sorted(track):
                past = [k - j for j in range(K)]
                if past[-1] < 0 or any(p not in track for p in past):
                    continue                                          # 过去 K 窗都要有轨迹（神谕上限）
                cx, cy, hx, hy, _ = track[k]
                path_c = np.array([track[p][:2] for p in past])       # [K, 2]
                fixed_c = np.repeat(np.array([[cx, cy]]), K, axis=0)
                # 零假设偏移：整条轨迹平移后都在画面内，且离所有目标 > 3 个框宽
                offsets, tries = [], 0
                while len(offsets) < n_null and tries < 50 * n_null:
                    tries += 1
                    d = rng.uniform([-W, -H], [W, H])
                    moved = np.concatenate([path_c, fixed_c]) + d
                    if np.any(moved[:, 0] < hx) or np.any(moved[:, 0] > W - 1 - hx) or \
                            np.any(moved[:, 1] < hy) or np.any(moved[:, 1] > H - 1 - hy):
                        continue
                    ok = True
                    for j, p in enumerate(past):
                        for (ox, oy, s) in centers[p]:
                            for c in (path_c[j] + d, fixed_c[j] + d):
                                if abs(c[0] - ox) < 3 * (hx + s) and abs(c[1] - oy) < 3 * (hy + s):
                                    ok = False
                    if ok:
                        offsets.append(d)
                if len(offsets) < max(2, n_null // 2):
                    continue
                offsets = np.array(offsets)
                ks_arr = np.array(past)
                for name, cs in (("path", path_c), ("fixed", fixed_c)):
                    target = int(box_sum(ii, ks_arr, cs[:, 0], cs[:, 1], hx, hy).sum())
                    nulls = np.array([box_sum(ii, ks_arr, cs[:, 0] + d[0], cs[:, 1] + d[1], hx, hy).sum()
                                      for d in offsets], dtype=np.float64)
                    auc = float(np.mean((target > nulls) + 0.5 * (target == nulls)))
                    sd = float(nulls.std())
                    dprime = (target - nulls.mean()) / sd if sd > 0 else float("inf") if target > nulls.mean() else 0.0
                    rows[name].append((auc, dprime, target, float(nulls.mean())))
        result[K] = {}
        for name, r in rows.items():
            if not r:
                result[K][name] = None
                continue
            a = np.array(r, dtype=np.float64)
            finite = a[np.isfinite(a[:, 1]), 1]
            result[K][name] = {"n": len(r), "auc": float(a[:, 0].mean()),
                               "frac_auc1": float(np.mean(a[:, 0] >= 1.0)),
                               "dprime_median": float(np.median(finite)) if finite.size else None,
                               "target_mean": float(a[:, 2].mean()), "null_mean": float(a[:, 3].mean())}
    neg = int((seq.label == 0).sum())
    return {"neg_per_window": neg / float(N_WINDOWS), "n_tracks": len(tracks), "by_K": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="含 val/ test/ 的 EV-UAV 数据目录")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--names", nargs="*", default=None, help="只跑这些文件（不含目录）")
    parser.add_argument("--n-null", type=int, default=16)
    parser.add_argument("--min-events", type=int, default=3)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rng = np.random.RandomState(args.seed)
    report = {}
    for split in args.splits:
        folder = os.path.join(args.root, split)
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".npz") or (args.names and name not in args.names):
                continue
            seq = load_npz_events(os.path.join(folder, name), H, W, WINDOW_MS, N_WINDOWS, TIME_BINS)
            report[name] = evaluate_sequence(seq, args.n_null, rng, args.min_events, KS)
            r = report[name]["by_K"]
            print("%-14s neg/w %6.0f  " % (name, report[name]["neg_per_window"]) + "  ".join(
                "K%-2d p %.3f f %.3f" % (K, (r[K]["path"] or {}).get("auc", float("nan")),
                                         (r[K]["fixed"] or {}).get("auc", float("nan"))) for K in KS), flush=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1)
    return report


if __name__ == "__main__":
    main()
