"""负事件（label != 1，即背景与噪声）的类型画像：只读数据，不加载模型、不训练（Stream-SNN-Innovations.md 12.7、12.19）。

目的：事件表示层的候选方案只对某一类负事件有效，设计前先查清极密序列（val_023、test_022/023）的负事件属于哪一类。
    传感器背景活动噪声（BA）：逐像素近似泊松、空间均匀、时间平稳、彼此独立 -> B1 逐像素速率归一化 / B2 超出随机巧合的支持度
    闪烁光源：固定像素、周期性（100/120 Hz 的事件间隔约 10/8.3 ms）、极性交替
    背景运动 / 纹理：事件在时空上成团、相邻窗的事件图存在非零的整体位移
    热点像素：少数像素占大量事件
每个序列输出下列统计（只用负事件；正事件的时空支持度另列作对照）：
    rate        每窗负事件数的均值、变异系数 cv、前后四分之一的相对变化 trend、相邻窗的自相关 lag1
    spatial     有事件的像素占比 coverage、前 1% 像素的事件占比 top1、像素总数的离散指数 var/mean（均匀泊松约为 1）、基尼系数
    temporal    活跃像素（每窗至少 0.5 个事件）各自跨窗计数的离散指数（泊松约为 1；成串爆发 > 1）的中位数
    polarity    ON 占比；同一像素相邻两个事件极性改变的比例与独立假设 2p(1-p) 的比值（闪烁明显 > 1）
    interval    同一像素相邻事件的时间间隔（毫秒）：众数、落在 8~11 ms（闪烁）与 1~3 ms（爆发）的比例
    support     10 ms 时间片内 3x3 邻域的其他事件数，观测均值 / 独立泊松（按各像素平均速率）下的期望；约 1 = 彼此独立
    motion      相邻两窗负事件图在 ±3 像素平移范围内的归一化互相关：零位移相关、最佳相关、最佳位移的增益，
                以及"最佳相关显著（> 5/sqrt(重叠像素数)，纯噪声下相关的 5 倍标准差）且最佳位移非零"的窗对占比
                （有整体运动的背景 / 纹理）。支持度的"机会水平"按每个像素自身的平均速率定义：固定不动的一团事件
                支持度约为 1，只有超出各位置平均速率的时空聚集（运动的边缘、目标）才 > 1。
用法:
    python tools/noise_profile.py --config configs/evisseg_stream_tif_notransport.yaml --split val test \\
        --out log/noise_profile_evuav.json
    python tools/noise_profile.py --config configs/evisseg_stream_tif_notransport_evflying.yaml --split val test \\
        --out log/noise_profile_evflying.json        # EV-Flying：按录像自身长度，分块计算
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

FINE_MS = 10
SHIFT = 3
ACTIVE_RATE = 0.5


def _box3(maps):
    """3x3 邻域求和（含中心），maps [..., H, W]，边界外为 0。"""
    padded = np.pad(maps, [(0, 0)] * (maps.ndim - 2) + [(1, 1), (1, 1)])
    h, w = maps.shape[-2:]
    out = np.zeros(maps.shape, dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            out += padded[..., dy:dy + h, dx:dx + w]
    return out


def _gini(values):
    v = np.sort(values.astype(np.float64))
    n = v.size
    if n == 0 or v.sum() <= 0:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def _dispersion(values):
    m = float(values.mean())
    return float(values.var() / m) if m > 0 else 0.0


def rate_stats(counts):
    """counts: 每窗事件数 [n_windows]。"""
    mean = float(counts.mean())
    n = counts.size
    q = max(n // 4, 1)
    lag1 = float(np.corrcoef(counts[:-1], counts[1:])[0, 1]) if n > 2 and counts.std() > 0 else 0.0
    return {"per_window": mean, "cv": float(counts.std() / mean) if mean > 0 else 0.0,
            "trend": float((counts[-q:].mean() - counts[:q].mean()) / mean) if mean > 0 else 0.0, "lag1": lag1}


def spatial_stats(totals):
    """totals: 每个像素的事件总数 [H, W]。"""
    flat = totals.ravel().astype(np.float64)
    total = flat.sum()
    k = max(int(round(flat.size * 0.01)), 1)
    return {"coverage": float((flat > 0).mean()),
            "top1": float(np.sort(flat)[-k:].sum() / total) if total > 0 else 0.0,
            "dispersion": _dispersion(flat), "gini": _gini(flat)}


def temporal_stats(total, square, n_windows):
    """由各像素跨窗计数的和与平方和（[H, W]）算活跃像素的离散指数。"""
    mean = total / float(n_windows)
    active = mean >= ACTIVE_RATE
    if not active.any():
        return {"active_pixels": 0, "dispersion_median": None}
    var = square[active] / float(n_windows) - mean[active] ** 2
    return {"active_pixels": int(active.sum()), "dispersion_median": float(np.median(var / mean[active]))}


def interval_polarity_stats(x, y, t, p, width):
    """同一像素相邻事件：时间间隔与极性改变。"""
    if t.size < 2:
        return {"on_fraction": float(p.mean()) if p.size else None, "alternation": None, "alternation_ratio": None,
                "interval_mode_ms": None, "interval_8_11": None, "interval_1_3": None}
    pixel = y.astype(np.int64) * width + x.astype(np.int64)
    order = np.lexsort((t, pixel))
    pix, ts, ps = pixel[order], t[order], p[order]
    same = pix[1:] == pix[:-1]
    gaps = (ts[1:] - ts[:-1])[same]
    flips = (ps[1:] != ps[:-1])[same]
    on = float(p.mean())
    expected = 2.0 * on * (1.0 - on)
    out = {"on_fraction": on, "alternation": float(flips.mean()) if flips.size else None}
    out["alternation_ratio"] = (out["alternation"] / expected) if (flips.size and expected > 0) else None
    if gaps.size:
        positive = gaps[(gaps >= 2) & (gaps <= 40)]
        out["interval_mode_ms"] = int(np.bincount(positive).argmax()) if positive.size else None
        out["interval_8_11"] = float(((gaps >= 8) & (gaps <= 11)).mean())
        out["interval_1_3"] = float(((gaps >= 1) & (gaps <= 3)).mean())
    else:
        out.update(interval_mode_ms=None, interval_8_11=None, interval_1_3=None)
    return out


def support_ratio(x, y, t, height, width, duration_ms):
    """10 ms 时间片内 3x3 邻域的其他事件数：观测 / 独立泊松期望（各像素按序列平均速率）。"""
    if t.size < 2:
        return None
    n_bins = max(int(np.ceil(duration_ms / float(FINE_MS))), 1)
    fine = np.minimum(t // FINE_MS, n_bins - 1).astype(np.int64)
    y, x = y.astype(np.int64), x.astype(np.int64)
    rate = np.zeros((height, width), dtype=np.float64)
    np.add.at(rate, (y, x), 1.0)
    rate /= n_bins                                                      # 每像素每个时间片的平均事件数
    expected = _box3(rate[None])[0][y, x]                               # 邻域（含本像素的其他事件）的期望
    observed = np.zeros(t.size, dtype=np.float64)
    step = 50                                                           # 分块：每次 50 个时间片，控制内存
    for lo in range(0, n_bins, step):
        sel = np.nonzero((fine >= lo) & (fine < lo + step))[0]
        if sel.size == 0:
            continue
        local = fine[sel] - lo
        maps = np.bincount((local * height + y[sel]) * width + x[sel],
                           minlength=step * height * width).reshape(step, height, width).astype(np.float32)
        observed[sel] = _box3(maps)[local, y[sel], x[sel]] - 1.0         # 去掉事件自己
    return float(observed.mean() / expected.mean()) if expected.mean() > 0 else None


def motion_pair(a, b):
    """相邻两窗事件图 a、b（[H, W]）在 ±SHIFT 平移范围内的归一化互相关。

    返回 (零位移相关, 最佳相关, 是否显著的非零位移)；任一窗没有事件时返回 None。
    显著 = 最佳位移非零且最佳相关 > 5/sqrt(重叠像素数)（纯噪声下相关的 5 倍标准差）。
    """
    h, w = a.shape
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.sum() == 0 or b.sum() == 0:
        return None
    best, best_shift, zero, overlap = -2.0, (0, 0), None, a.size
    for dy in range(-SHIFT, SHIFT + 1):
        for dx in range(-SHIFT, SHIFT + 1):
            ya, yb = slice(max(0, -dy), h - max(0, dy)), slice(max(0, dy), h - max(0, -dy))
            xa, xb = slice(max(0, -dx), w - max(0, dx)), slice(max(0, dx), w - max(0, -dx))
            u, v = a[ya, xa].ravel(), b[yb, xb].ravel()
            if u.std() == 0 or v.std() == 0:
                continue
            c = float(np.corrcoef(u, v)[0, 1])
            if (dy, dx) == (0, 0):
                zero = c
            if c > best:
                best, best_shift, overlap = c, (dy, dx), u.size
    if zero is None:
        return None
    return zero, best, bool(best_shift != (0, 0) and best > 5.0 / np.sqrt(overlap))


def motion_summary(pairs):
    """motion_pair 结果列表的汇总：中位数与显著位移占比。"""
    pairs = [r for r in pairs if r is not None]
    if not pairs:
        return {"zero_corr": None, "best_corr": None, "shift_gain": None, "moved_frac": None, "pairs": 0}
    zero, best, moved = (np.array([r[i] for r in pairs], dtype=np.float64) for i in range(3))
    return {"zero_corr": float(np.median(zero)), "best_corr": float(np.median(best)),
            "shift_gain": float(np.median(best - zero)), "moved_frac": float(moved.mean()), "pairs": len(pairs)}


def profile_sequence(seq, height, width, window_ms, n_windows=None, max_pairs=400, block=200):
    """一个序列（录像）的负事件画像（另附正事件的时空支持度作对照）。

    n_windows 为 None 或 <= 0 时用录像自身的窗数（EV-Flying 等长录像）。按 block 窗分块累计，内存与录像长度无关；
    相邻窗互相关最多取约 max_pairs 对（等间隔抽取），长录像也只需几分钟。
    """
    if not n_windows or n_windows <= 0:
        n_windows = int(seq.n_windows)
    neg = seq.label != 1
    x, y, t, p = seq.x[neg], seq.y[neg], seq.t[neg], seq.p[neg].astype(np.float64)
    window = np.minimum(t // window_ms, n_windows - 1).astype(np.int64)
    counts = np.bincount(window, minlength=n_windows).astype(np.float64)
    total = np.zeros((height, width), dtype=np.float64)
    square = np.zeros((height, width), dtype=np.float64)
    stride = max(1, (n_windows - 1) // max_pairs)
    pairs, previous = [], None
    order = np.argsort(window, kind="stable")
    starts = list(range(0, n_windows, block))
    edges = np.searchsorted(window[order], starts + [n_windows])
    for i, lo in enumerate(starts):
        hi = min(lo + block, n_windows)
        sel = order[edges[i]:edges[i + 1]]
        cube = np.bincount(((window[sel] - lo) * height + y[sel]) * width + x[sel],
                           minlength=(hi - lo) * height * width).reshape(hi - lo, height, width).astype(np.float64)
        total += cube.sum(axis=0)
        square += (cube ** 2).sum(axis=0)
        for k in range(hi - lo):
            if previous is not None and (lo + k - 1) % stride == 0:
                pairs.append(motion_pair(previous, cube[k]))
            previous = cube[k]
    duration = window_ms * n_windows
    pos = ~neg
    return {"name": seq.name, "windows": n_windows, "neg_events": int(neg.sum()),
            "pos_per_window": float(pos.sum()) / n_windows,
            "rate": rate_stats(counts),
            "spatial": spatial_stats(total),
            "temporal": temporal_stats(total, square, n_windows),
            "polarity_interval": interval_polarity_stats(x, y, t, p, width),
            "support_neg": support_ratio(x, y, t, height, width, duration),
            "support_pos": support_ratio(seq.x[pos], seq.y[pos], seq.t[pos], height, width, duration),
            "motion": motion_summary(pairs)}


def fmt(value, spec="%.2f"):
    return "-" if value is None else spec % value


def print_table(rows):
    head = ("序列", "负/窗", "cv", "trend", "覆盖", "top1", "空间离散", "基尼", "时间离散", "ON", "极性改变比",
            "间隔众数", "8-11ms", "支持度负", "支持度正", "零位移相关", "最佳相关", "位移增益", "显著位移")
    print(" | ".join(head))
    for r in sorted(rows, key=lambda r: -r["rate"]["per_window"]):
        pi = r["polarity_interval"]
        print(" | ".join([r["split"] + "/" + r["name"], "%.0f" % r["rate"]["per_window"], fmt(r["rate"]["cv"]),
                          fmt(r["rate"]["trend"]), fmt(r["spatial"]["coverage"]), fmt(r["spatial"]["top1"]),
                          fmt(r["spatial"]["dispersion"], "%.1f"), fmt(r["spatial"]["gini"]),
                          fmt(r["temporal"]["dispersion_median"]), fmt(pi["on_fraction"]),
                          fmt(pi["alternation_ratio"]), fmt(pi["interval_mode_ms"], "%d"), fmt(pi["interval_8_11"], "%.3f"),
                          fmt(r["support_neg"]), fmt(r["support_pos"]), fmt(r["motion"]["zero_corr"]), fmt(r["motion"]["best_corr"]),
                          fmt(r["motion"]["shift_gain"], "%.3f"), fmt(r["motion"]["moved_frac"])]))


def main():
    from dataset.ev_uav_stream import EvUAVStream
    from utils.stream_common import load_flat_config
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/evisseg_stream_tif_notransport.yaml")
    parser.add_argument("--split", nargs="+", default=["val", "test"])
    parser.add_argument("--names", nargs="*", default=None, help="只跑这些文件（例如 val_023.npz）")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.out) and not args.overwrite:
        raise RuntimeError("结果文件已存在（加 --overwrite 覆盖）: %s" % args.out)
    cfg = load_flat_config(args.config)
    rows = []
    for split in args.split:
        dataset = EvUAVStream(cfg["root"], split, cfg)
        names = dataset.names if not args.names else [n for n in dataset.names if n in args.names]
        for i, name in enumerate(dataset.names):
            if name not in names:
                continue
            seq = dataset[i]
            row = profile_sequence(seq, int(cfg["height"]), int(cfg["width_px"]), int(cfg["window_ms"]),
                                   int(cfg["n_windows"]))                 # 0 = 按录像自身长度（EV-Flying）
            row["split"] = split
            rows.append(row)
            print("[%s] %s 负事件 %d" % (split, name, row["neg_events"]), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump({"config": args.config, "fine_ms": FINE_MS, "shift": SHIFT, "rows": rows}, stream, indent=1,
                  ensure_ascii=False)
    print_table(rows)
    print("NOISE PROFILE FINISHED:", args.out, flush=True)


if __name__ == "__main__":
    main()
