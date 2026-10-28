"""E2 之后的诊断（Inn 13.50）：极密序列上的虚警是什么——偶然的噪声团，还是持续运动的东西？

输入：train_stream_v1.py --mode eval --dump-dir 导出的逐事件预测（每个序列一个 NPZ：locs [b,x,y,t]、labels、probabilities）。
做法（阈值 0.9）：
    1. 每个 50 ms 窗里，把预测为目标的事件画到像素图上，8 连通成团；团内多数是负事件的记为"虚警团"，否则记"目标团"。
    2. 团的持续性（雷达 M/N 航迹确认的原理）：把相邻窗里中心距离 <= max_step 像素的团串成链（贪心最近邻），
       统计每个团所在链的长度。偶然噪声团的链短；运动目标（无论有没有标注）的链长。
    3. 每种"至少持续 M 窗才确认"的规则下：虚警团（按事件数计）被去掉多少，目标团保住多少——这是航迹确认能拿到的上限。
    4. 虚警团处的局部背景密度（21x21 邻域本窗全部事件数）与全图平均之比：检验"噪声在空间上不均匀、虚警落在局部密集处"。
    5. 虚警事件的极性配对：同一像素 5 ms 内有相反极性事件的比例（散粒噪声 ON/OFF 成对），与目标事件比较。
只作诊断，不进方法。
"""
import argparse
import json
import os

import numpy as np

H, W, WINDOW_MS = 260, 346, 50
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def components(ys, xs):
    """像素坐标集合的 8 连通分量标号（返回每个像素的分量号）。"""
    index = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    label = -np.ones(len(index), dtype=np.int64)
    n = 0
    for start in range(len(label)):
        if label[start] >= 0:
            continue
        stack, label[start] = [start], n
        while stack:
            i = stack.pop()
            y, x = int(ys[i]), int(xs[i])
            for dy, dx in NEIGHBORS:
                j = index.get((y + dy, x + dx))
                if j is not None and label[j] < 0:
                    label[j] = n
                    stack.append(j)
        n += 1
    return label, n


def pairing_fraction(x, y, t, p, sel, gap_ms=5):
    """sel 中的事件里，同一像素 gap_ms 内存在相反极性事件的比例。"""
    if sel.sum() == 0:
        return None
    key = y.astype(np.int64) * W + x
    order = np.lexsort((t, key))
    k, tt, pp = key[order], t[order], p[order]
    paired = np.zeros(len(order), dtype=bool)
    same = k[1:] == k[:-1]
    close = (tt[1:] - tt[:-1]) <= gap_ms
    flip = pp[1:] != pp[:-1]
    hit = same & close & flip
    paired[1:] |= hit
    paired[:-1] |= hit
    back = np.empty_like(paired)
    back[order] = paired
    return float(back[sel].mean())


def analyze(path, threshold, max_step, ms):
    d = np.load(path)
    locs, labels, prob = d["locs"], d["labels"] > 0.5, d["probabilities"]
    x, y, t = locs[:, 1].astype(np.int64), locs[:, 2].astype(np.int64), locs[:, 3].astype(np.int64)
    p = d["p"] if "p" in d.files else None
    pred = prob >= threshold
    win = t // WINDOW_MS
    n_win = int(win.max()) + 1
    blobs = []                                                         # (window, cy, cx, n_events, n_neg, n_pos, local_ratio)
    for k in range(n_win):
        in_k = win == k
        sel = np.where(in_k & pred)[0]
        if sel.size == 0:
            continue
        counts = np.zeros((H, W), dtype=np.int64)
        np.add.at(counts, (y[in_k], x[in_k]), 1)
        ii = np.zeros((H + 1, W + 1), dtype=np.int64)
        ii[1:, 1:] = counts.cumsum(0).cumsum(1)
        mean_density = counts.sum() / float(H * W) * 21 * 21
        pix = np.unique(np.stack([y[sel], x[sel]], 1), axis=0)
        lab, n = components(pix[:, 0], pix[:, 1])
        pix_of = {(int(a), int(b)): int(c) for (a, b), c in zip(pix, lab)}
        comp = np.array([pix_of[(int(a), int(b))] for a, b in zip(y[sel], x[sel])])
        for c in range(n):
            ev = sel[comp == c]
            cy, cx = y[ev].mean(), x[ev].mean()
            y0, y1 = int(max(cy - 10, 0)), int(min(cy + 11, H))
            x0, x1 = int(max(cx - 10, 0)), int(min(cx + 11, W))
            local = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
            npos = int(labels[ev].sum())
            blobs.append((k, cy, cx, ev.size, ev.size - npos, npos, local / max(mean_density, 1e-9)))
    blobs = np.array(blobs, dtype=np.float64) if blobs else np.zeros((0, 7))
    # 链：相邻窗、中心距离 <= max_step 的团贪心相连
    chain_len = np.ones(len(blobs), dtype=np.int64)
    if len(blobs):
        by_win = {}
        for i, b in enumerate(blobs):
            by_win.setdefault(int(b[0]), []).append(i)
        nxt = -np.ones(len(blobs), dtype=np.int64)
        prv = -np.ones(len(blobs), dtype=np.int64)
        for k in sorted(by_win):
            cur, later = by_win[k], by_win.get(k + 1, [])
            pairs = sorted((np.hypot(blobs[i, 1] - blobs[j, 1], blobs[i, 2] - blobs[j, 2]), i, j)
                           for i in cur for j in later)
            for dist, i, j in pairs:
                if dist <= max_step and nxt[i] < 0 and prv[j] < 0:
                    nxt[i], prv[j] = j, i
        for i in range(len(blobs)):
            if prv[i] < 0:
                members, j = [], i
                while j >= 0:
                    members.append(j)
                    j = nxt[j]
                chain_len[members] = len(members)
    fa = blobs[:, 4] > blobs[:, 5] if len(blobs) else np.zeros(0, dtype=bool)
    # 链的运动一致性（雷达航迹确认里的运动模型）：长度 >= 3 的链，质心对时间做匀速直线拟合
    chains = []
    if len(blobs):
        for i in range(len(blobs)):
            if prv[i] >= 0:
                continue
            members, j = [], i
            while j >= 0:
                members.append(j)
                j = nxt[j]
            if len(members) < 3:
                continue
            b = blobs[members]
            tt = b[:, 0] - b[0, 0]
            resid = []
            for col in (1, 2):
                coef = np.polyfit(tt, b[:, col], 1)
                resid.append(b[:, col] - np.polyval(coef, tt))
            rms = float(np.sqrt(np.mean(np.square(resid[0]) + np.square(resid[1]))))
            speed = float(np.hypot(b[-1, 1] - b[0, 1], b[-1, 2] - b[0, 2]) / (len(members) - 1))
            chains.append((len(members), rms, speed, bool(fa[members].sum() * 2 > len(members)),
                           float(b[:, 4].sum()), float(b[:, 5].sum())))
    out = {"name": os.path.basename(path), "fp_events": int((pred & ~labels).sum()), "tp_events": int((pred & labels).sum()),
           "fa_blobs": int(fa.sum()), "target_blobs": int((~fa).sum())}
    out["confirm"] = {}
    for m in ms:
        keep = chain_len >= m
        out["confirm"][m] = {
            "fp_events_removed_frac": float(blobs[fa & ~keep, 4].sum() / max(blobs[fa, 4].sum(), 1)),
            "tp_events_kept_frac": float(blobs[~fa & keep, 5].sum() / max(blobs[~fa, 5].sum(), 1))}
    out["fa_chain_len_median"] = float(np.median(chain_len[fa])) if fa.any() else None
    out["target_chain_len_median"] = float(np.median(chain_len[~fa])) if (~fa).any() else None
    out["fa_blob_events_median"] = float(np.median(blobs[fa, 3])) if fa.any() else None
    out["fa_local_density_ratio_median"] = float(np.median(blobs[fa, 6])) if fa.any() else None
    out["target_local_density_ratio_median"] = float(np.median(blobs[~fa, 6])) if (~fa).any() else None
    def auc(a, b):                                                    # P(a > b)，平局 0.5
        a, b = np.asarray(a), np.asarray(b)
        if not len(a) or not len(b):
            return None
        return float(((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()))
    if chains:
        c = np.array(chains, dtype=np.float64)
        fa_c, tg_c = c[c[:, 3] > 0.5], c[c[:, 3] <= 0.5]
        out["chains"] = {"fa": int(len(fa_c)), "target": int(len(tg_c)),
                         "fa_rms_median": float(np.median(fa_c[:, 1])) if len(fa_c) else None,
                         "target_rms_median": float(np.median(tg_c[:, 1])) if len(tg_c) else None,
                         "fa_speed_median": float(np.median(fa_c[:, 2])) if len(fa_c) else None,
                         "target_speed_median": float(np.median(tg_c[:, 2])) if len(tg_c) else None,
                         "auc_rms_fa_gt_target": auc(fa_c[:, 1], tg_c[:, 1]),
                         "auc_speed_target_gt_fa": auc(tg_c[:, 2], fa_c[:, 2]),
                         "fa_events_in_chains": float(fa_c[:, 4].sum()) if len(fa_c) else 0.0}
    if p is not None:
        out["pairing_fp"] = pairing_fraction(x, y, t, p, pred & ~labels)
        out["pairing_tp"] = pairing_fraction(x, y, t, p, pred & labels)
        out["pairing_neg"] = pairing_fraction(x, y, t, p, ~labels)
    return out


def fmt(v):
    return "-" if v is None else "%.2f" % v


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--data-dir", default=None, help="原始 NPZ 目录（用来取极性；dump 里没有 p 时需要）")
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--max-step", type=float, default=8.0, help="相邻窗团中心的最大位移（像素）")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    ms = (1, 2, 3, 4, 6)
    report = []
    for name in sorted(os.listdir(args.dump_dir)):
        if not name.endswith(".npz") or (args.names and name not in args.names):
            continue
        path = os.path.join(args.dump_dir, name)
        if args.data_dir:                                             # 把原始极性并进来
            raw = np.load(os.path.join(args.data_dir, name))
            d = dict(np.load(path))
            d["p"] = raw["evs_norm"][:, 3].astype(np.int8)
            path = path + ".with_p.npz"
            np.savez(path, **d)
        r = analyze(path, args.threshold, args.max_step, ms)
        report.append(r)
        print("%-14s FP %6d TP %6d | 虚警团 %5d 目标团 %5d | 链长中位 虚警 %s 目标 %s | 局部密度比 虚警 %s 目标 %s | "
              "极性配对 FP %s TP %s 负 %s" % (
                  name, r["fp_events"], r["tp_events"], r["fa_blobs"], r["target_blobs"],
                  r["fa_chain_len_median"], r["target_chain_len_median"],
                  None if r["fa_local_density_ratio_median"] is None else round(r["fa_local_density_ratio_median"], 2),
                  None if r["target_local_density_ratio_median"] is None else round(r["target_local_density_ratio_median"], 2),
                  r.get("pairing_fp"), r.get("pairing_tp"), r.get("pairing_neg")), flush=True)
        if "chains" in r:
            ch = r["chains"]
            print("    链（>= 3 窗）：虚警链 %d、目标链 %d | 匀速拟合残差中位 虚警 %s 目标 %s 像素 | 净速度中位 虚警 %s 目标 %s 像素/窗"
                  " | AUC（残差区分）%s、AUC（速度区分）%s" % (
                      ch["fa"], ch["target"], fmt(ch["fa_rms_median"]), fmt(ch["target_rms_median"]),
                      fmt(ch["fa_speed_median"]), fmt(ch["target_speed_median"]),
                      fmt(ch["auc_rms_fa_gt_target"]), fmt(ch["auc_speed_target_gt_fa"])), flush=True)
        for m, c in r["confirm"].items():
            print("    至少持续 %d 窗才确认：去掉虚警事件 %.0f%%，保住目标事件 %.0f%%" % (
                m, 100 * c["fp_events_removed_frac"], 100 * c["tp_events_kept_frac"]), flush=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
