"""模块三误差预算（10-30）：零等待的错误里，有多少能靠"多看几个窗"或目标级判决改对。只读 --dump-dir 的逐事件预测，numpy 即可。

口径（窗 = t // window_ms；r = 切比雪夫半径，像素；阈值默认 0.9）：
  漏检（标签为目标、预测为背景），按所属目标 target_id 的检出情况分：
    never   整个目标从未被检出（任何窗都没有命中事件）            —— 判决层救不回
    same    同一窗里这个目标已有命中事件（空间范围没覆盖全）     —— 目标级判决可能补，不靠等待
    onset   目标首次检出之前（第一拍）；首次检出在 D 窗内则可回溯补判
    gap     首次检出之后的断续；前 4 窗内有检出 = 跟踪外推可补（D = 0），否则看后 D 窗内有没有检出
  误检（标签为背景、预测为目标），按位置与标注目标的关系分：
    near    本窗 ±1 窗内、r 以内有标注目标事件（目标边界、光晕、漏标） —— 标注上限，不可修
    tail    不属 near；前 2～20 窗内这里有过目标（目标离开后的残留）    —— 航迹删除可修（D = 0）
    ahead   不属以上；后 2～20 窗内目标会来这里（目标到达前、多为漏标） —— 标注上限，不可修
    far     其余：后 D 窗里不到 ceil(D/2) 窗在 r 以内有预测为目标的事件 = 不持续（M/N 确认可修），否则为持续误检（不可修）
每个 D 给出"把可修的错误全部改对"的 IoU 上限（神谕：用标签判断哪些可修，所以只是天花板，不是可达值）。
另给出单一全局阈值的最优 IoU（输出校准的空间，只作诊断）。
用法：python tools/error_budget.py --dump-dir v5=Res/v5_dump_local/v5_test --dense test_022 test_023 --out Res/v5_dump_local/budget_test.json
"""
import argparse
import glob
import json
import os

import numpy as np

PAST_COAST = 4          # 跟踪外推：前 4 窗（200 ms）内检出过，就认为跟踪器还记得它
TAIL_RANGE = (2, 20)    # 残留：目标在 2～20 窗（0.1～1 s）前在这里
AHEAD_RANGE = (2, 20)   # 到达前：目标在 2～20 窗后来到这里


def dilate(mask, r):
    """二维布尔图按 (2r+1)^2 方框膨胀（积分图实现）。"""
    if r <= 0:
        return mask.copy()
    h, w = mask.shape
    c = np.zeros((h + 1, w + 1), np.int32)
    c[1:, 1:] = mask.astype(np.int32).cumsum(0).cumsum(1)
    y0 = np.clip(np.arange(h) - r, 0, h); y1 = np.clip(np.arange(h) + r + 1, 0, h)
    x0 = np.clip(np.arange(w) - r, 0, w); x1 = np.clip(np.arange(w) + r + 1, 0, w)
    s = c[y1][:, x1] - c[y0][:, x1] - c[y1][:, x0] + c[y0][:, x0]
    return s > 0


def window_maps(win, y, x, sel, n_win, h, w, r):
    """每窗一张"sel 事件在 r 以内"的布尔图。"""
    maps = np.zeros((n_win, h, w), bool)
    for k in np.unique(win[sel]):
        m = np.zeros((h, w), bool)
        idx = sel & (win == k)
        m[y[idx], x[idx]] = True
        maps[k] = dilate(m, r)
    return maps


def count_in_range(maps, win, y, x, idx, lo, hi):
    """对事件 idx：窗口 k+lo .. k+hi（含两端）里有几张图在 (y, x) 为真。"""
    out = np.zeros(idx.sum(), np.int32)
    kk, yy, xx = win[idx], y[idx], x[idx]
    n = maps.shape[0]
    for off in range(lo, hi + 1):
        j = kk + off
        ok = (j >= 0) & (j < n)
        out[ok] += maps[j[ok], yy[ok], xx[ok]]
    return out


def any_in_range(maps, win, y, x, idx, lo, hi):
    return count_in_range(maps, win, y, x, idx, lo, hi) > 0


def analyse(path, threshold, radius, window_ms, delays, h, w):
    d = np.load(path)
    return analyse_arrays(os.path.splitext(os.path.basename(path))[0], d["locs"][:, 1], d["locs"][:, 2],
                          d["locs"][:, 3], d["labels"] > 0.5, d["probabilities"], d["target_id"],
                          threshold, radius, window_ms, delays, h, w)


def analyse_arrays(name, x, y, t_ms, pos, prob, tid, threshold, radius=10, window_ms=50.0, delays=(1, 2, 4, 8),
                   h=260, w=346):
    """与 analyse 相同，输入为内存中的数组（x、y 像素，t_ms 毫秒，pos 标签布尔，prob 概率，tid 目标号）。"""
    x = np.asarray(x).astype(np.int64); y = np.asarray(y).astype(np.int64)
    win = (np.asarray(t_ms) // window_ms).astype(np.int64)
    pos = np.asarray(pos, bool)
    prob = np.asarray(prob)
    tid = np.asarray(tid).astype(np.int64)
    pred = prob >= threshold
    n_win = int(win.max()) + 1 if win.size else 1
    tp, fp, fn = pred & pos, pred & ~pos, ~pred & pos
    out = {"name": name, "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum())}

    # ---- 漏检：按目标的检出窗口分类
    det = {}                                  # target_id -> 有命中事件的窗口（排序）
    for g in np.unique(tid[pos]):
        det[g] = np.unique(win[tp & (tid == g)])
    fn_idx = np.flatnonzero(fn)
    cat = np.empty(fn_idx.size, object)
    need = np.full(fn_idx.size, np.inf)       # 需要等待的窗数（0 = 不用等）
    for i, e in enumerate(fn_idx):
        dw, k = det[tid[e]], win[e]
        if dw.size == 0:
            cat[i] = "never"
        elif np.any(dw == k):
            cat[i] = "same"
        elif k < dw[0]:
            cat[i] = "onset"; need[i] = dw[0] - k
        else:
            cat[i] = "gap"
            past = dw[(dw < k) & (dw >= k - PAST_COAST)]
            fut = dw[dw > k]
            need[i] = 0 if past.size else (fut[0] - k if fut.size else np.inf)
    out["fn_cat"] = {c: int((cat == c).sum()) for c in ("never", "same", "onset", "gap")}

    # ---- 误检：按与标注目标的位置关系分类
    lab_maps = window_maps(win, y, x, pos, n_win, h, w, radius)
    near = any_in_range(lab_maps, win, y, x, fp, -1, 1)
    tail = ~near & any_in_range(lab_maps, win, y, x, fp, -TAIL_RANGE[1], -TAIL_RANGE[0])
    ahead = ~near & ~tail & any_in_range(lab_maps, win, y, x, fp, AHEAD_RANGE[0], AHEAD_RANGE[1])
    far = ~near & ~tail & ~ahead
    del lab_maps
    pred_maps = window_maps(win, y, x, pred, n_win, h, w, radius)
    out["fp_cat"] = {"near": int(near.sum()), "tail": int(tail.sum()), "ahead": int(ahead.sum()), "far": int(far.sum())}

    # ---- 每个等待窗数 D 的可修数量
    out["fixable"] = {}
    for D in [0] + list(delays):
        # M/N 确认：后 D 窗里至少 ceil(D/2) 窗在 r 以内有预测为目标的事件才算持续；不持续的 far 误检可修
        transient = far & (count_in_range(pred_maps, win, y, x, fp, 1, D) < (D + 1) // 2) if D > 0 else np.zeros_like(far)
        fix_fp = {"tail": int(tail.sum()), "far_transient": int(transient.sum())}
        fix_fn = {"onset": int(((cat == "onset") & (need <= D)).sum()), "gap": int(((cat == "gap") & (need <= D)).sum())}
        out["fixable"][str(D)] = {"fp": fix_fp, "fn": fix_fn}
    return out, prob, pos


def iou(tp, fp, fn):
    return tp / (tp + fp + fn) if tp + fp + fn else float("nan")


def pooled(rows, delays):
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    res = {"tp": tp, "fp": fp, "fn": fn, "iou": iou(tp, fp, fn),
           "fn_cat": {c: sum(r["fn_cat"][c] for r in rows) for c in rows[0]["fn_cat"]},
           "fp_cat": {c: sum(r["fp_cat"][c] for r in rows) for c in rows[0]["fp_cat"]}, "ceiling": {}}
    for D in [0] + list(delays):
        f = [r["fixable"][str(D)] for r in rows]
        tail = sum(x["fp"]["tail"] for x in f); tr = sum(x["fp"]["far_transient"] for x in f)
        on = sum(x["fn"]["onset"] for x in f); gp = sum(x["fn"]["gap"] for x in f)
        res["ceiling"][str(D)] = {
            "fp_fixed": tail + tr, "fn_fixed": on + gp,
            "iou_fp_only": iou(tp, fp - tail - tr, fn),
            "iou_fn_only": iou(tp + on + gp, fp, fn - on - gp),
            "iou_all": iou(tp + on + gp, fp - tail - tr, fn - on - gp)}
    return res


def best_threshold_iou(probs, labels, grid=np.round(np.arange(0.05, 0.96, 0.05), 2)):
    p = np.concatenate(probs); l = np.concatenate(labels)
    best = (None, -1.0)
    for t in grid:
        pr = p >= t
        v = iou(int((pr & l).sum()), int((pr & ~l).sum()), int((~pr & l).sum()))
        if v > best[1]:
            best = (float(t), v)
    return {"threshold": best[0], "iou": best[1]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dump-dir", required=True, help="名称=目录")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--radius", type=int, default=10, help="像素（切比雪夫距离）")
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 2, 4, 8], help="允许等待的窗数")
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--width", type=int, default=346)
    parser.add_argument("--dense", nargs="*", default=[], help="单独汇总的序列名（如极密序列）")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    name, folder = args.dump_dir.split("=", 1)
    files = sorted(f for f in glob.glob(os.path.join(folder, "*.npz")) if not f.endswith(".with_p.npz"))
    rows, probs, labels = [], [], []
    for f in files:
        r, p, l = analyse(f, args.threshold, args.radius, args.window_ms, args.delays, args.height, args.width)
        rows.append(r); probs.append(p); labels.append(l)
    groups = {"all": rows}
    if args.dense:
        groups["regular"] = [r for r in rows if r["name"] not in args.dense]
        groups["dense"] = [r for r in rows if r["name"] in args.dense]
    report = {"name": name, "threshold": args.threshold, "radius": args.radius, "delays": args.delays,
              "groups": {g: pooled(v, args.delays) for g, v in groups.items() if v},
              "best_global_threshold": best_threshold_iou(probs, labels), "sequences": rows}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    for g, v in report["groups"].items():
        print("== %s %s：TP %d FP %d FN %d IoU %.4f" % (name, g, v["tp"], v["fp"], v["fn"], v["iou"]))
        print("   漏检分类", v["fn_cat"], " 误检分类", v["fp_cat"])
        for D, c in v["ceiling"].items():
            print("   D=%s 窗：可修 FP %d、FN %d -> IoU 上限 %.4f（只修 FP %.4f，只修 FN %.4f）"
                  % (D, c["fp_fixed"], c["fn_fixed"], c["iou_all"], c["iou_fp_only"], c["iou_fn_only"]))
    b = report["best_global_threshold"]
    print("   单一全局阈值最优：%.2f -> IoU %.4f（阈值 %.2f 时 %.4f）" % (b["threshold"], b["iou"], args.threshold,
                                                               report["groups"]["all"]["iou"]))


if __name__ == "__main__":
    main()
