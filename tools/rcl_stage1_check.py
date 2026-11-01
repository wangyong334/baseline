"""RCL 第一阶段关口（RCL_V1_PLAN_20261101.md §8，固定滞后对 V5 零等待）：读 tools/run_track_head.sh 存的逐事件预测
（run/dump/<名字>_<划分>/*.npz，含 base_probabilities / now_probabilities / probabilities），每个划分报告：
    IoU          骨干（= 零等待暂定输出）与终判；与正式评估同一口径（阈值 0.9，>=，全划分累计 TP / FP / FN）
    普通录像     去掉 --dense 列出的极密录像后的 IoU：终判不低于骨干
    极密录像     目标附近误报（tools/error_budget 的 near 类，10 像素）：终判比骨干增加 <= 10%
    第一拍召回   以骨干对每个目标的首检窗为统一参照，取首检窗及之前的该目标事件：终判比零等待高 >= 0.08
    另报         往上 / 往下翻转的事件（改对 / 改错）；被救回的目标事件（零等待 0、终判 1）等了几窗
最后汇总五条关口：val IoU >= --val-iou-min、test 不低于骨干、普通录像两个划分都不降、极密录像目标附近误报 <= +10%、
第一拍召回 >= +0.08。不过关就停（模块三转为分析），不去调提交规则。
用法：python tools/rcl_stage1_check.py --dump run/dump/changepoint_v1 [--dense val_023 test_022 test_023]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import error_budget  # noqa: E402

DENSE = ("val_023", "test_022", "test_023")       # EV-UAV 的 3 条极密录像（CLAUDE.md 第 1 节）


def iou(c):
    return c[0] / float(max(c[0] + c[1] + c[2], 1))


def counts(pred, pos):
    return np.array([int((pred & pos).sum()), int((pred & ~pos).sum()), int((~pred & pos).sum())], dtype=np.int64)


def first_beat_mask(win, pos, tid, base_hit):
    """每个目标：骨干首检窗及之前的该目标事件（骨干从未检出的目标不计）。"""
    early = np.zeros(pos.shape, bool)
    for g in np.unique(tid[pos]):
        hit = win[pos & base_hit & (tid == g)]
        if hit.size:
            early |= pos & (tid == g) & (win <= hit.min())
    return early


def check_split(directory, dense, thr=0.9, window_ms=50.0, height=260, width=346):
    names = sorted(f for f in os.listdir(directory) if f.endswith(".npz"))
    if not names:
        raise RuntimeError("目录里没有逐事件预测：%s" % directory)
    tot = {k: np.zeros(3, dtype=np.int64) for k in ("base", "final", "base_normal", "final_normal")}
    near = {"base": 0, "final": 0}
    early = {"n": 0, "now": 0, "final": 0, "base": 0}
    flips = {"up_tp": 0, "up_fp": 0, "down_tn": 0, "down_fn": 0}
    waits, dense_seen = [], []
    for name in names:
        with np.load(os.path.join(directory, name)) as d:
            locs, pos = d["locs"], d["labels"] > 0.5
            tid = d["target_id"].astype(np.int64)
            base_p, fin_p = d["base_probabilities"], d["probabilities"]
            now_p = d["now_probabilities"] if "now_probabilities" in d.files else base_p
            wait = d["decision_wait"].astype(np.int64) if "decision_wait" in d.files else None
        x, y, t = locs[:, 1], locs[:, 2], locs[:, 3]
        win = (t // int(window_ms)).astype(np.int64)
        base, fin, now = base_p >= thr, fin_p >= thr, now_p >= thr
        cb, cf = counts(base, pos), counts(fin, pos)
        tot["base"] += cb
        tot["final"] += cf
        stem = name[:-4]
        if stem in dense:
            dense_seen.append(stem)
            for kind, prob in (("base", base_p), ("final", fin_p)):
                row = error_budget.analyse_arrays(stem, x, y, t, pos, prob, tid, thr, 10, window_ms, (1,),
                                                  height, width)[0]
                near[kind] += int(row["fp_cat"]["near"])
        else:
            tot["base_normal"] += cb
            tot["final_normal"] += cf
        e = first_beat_mask(win, pos, tid, base)
        early["n"] += int(e.sum())
        early["base"] += int((e & base).sum())
        early["now"] += int((e & now).sum())
        early["final"] += int((e & fin).sum())
        up, down = ~now & fin, now & ~fin
        flips["up_tp"] += int((up & pos).sum())
        flips["up_fp"] += int((up & ~pos).sum())
        flips["down_tn"] += int((down & ~pos).sum())
        flips["down_fn"] += int((down & pos).sum())
        if wait is not None:
            waits.append(wait[up & pos])
    n = float(max(early["n"], 1))
    rw = np.concatenate(waits) if waits else np.zeros(0, dtype=np.int64)
    out = {"n_sequences": len(names), "dense": dense_seen,
           "iou": {"base": iou(tot["base"]), "final": iou(tot["final"])},
           "iou_normal": {"base": iou(tot["base_normal"]), "final": iou(tot["final_normal"])},
           "dense_near_fp": dict(near),
           "dense_near_fp_rise": (near["final"] - near["base"]) / float(near["base"]) if near["base"] else None,
           "first_beat": {"n": early["n"], "base": early["base"] / n, "zero_wait": early["now"] / n,
                          "final": early["final"] / n, "gain": (early["final"] - early["now"]) / n},
           "flips": flips,
           "rescued": {"n": int(rw.size), "mean_wait_windows": float(rw.mean()) if rw.size else None,
                       "wait_hist": np.bincount(rw).tolist() if rw.size else []},
           "counts": {k: v.tolist() for k, v in tot.items()}}
    return out


def gates(res, val_iou_min, near_rise_max=0.10, first_beat_min=0.08):
    g = {}
    if "val" in res:
        g["val_iou >= %.4f" % val_iou_min] = res["val"]["iou"]["final"] >= val_iou_min
    if "test" in res:
        g["test_iou >= base"] = res["test"]["iou"]["final"] >= res["test"]["iou"]["base"]
    g["normal_not_lower"] = all(r["iou_normal"]["final"] >= r["iou_normal"]["base"] for r in res.values())
    rises = [r["dense_near_fp_rise"] for r in res.values() if r["dense_near_fp_rise"] is not None]
    g["dense_near_fp_rise <= %.2f" % near_rise_max] = bool(rises) and all(x <= near_rise_max for x in rises)
    g["first_beat_gain >= %.2f" % first_beat_min] = all(r["first_beat"]["gain"] >= first_beat_min
                                                         for r in res.values())
    g = {k: bool(v) for k, v in g.items()}
    g["pass"] = all(g.values())
    return g


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dump", required=True, help="前缀：读 <前缀>_val/ 与 <前缀>_test/")
    parser.add_argument("--dense", nargs="*", default=list(DENSE), help="极密录像名（不带 .npz）")
    parser.add_argument("--val-iou-min", type=float, default=0.8768, help="V5 val 0.8668 + 0.010")
    parser.add_argument("--out", default=None, help="结果 JSON（默认 <前缀>_stage1.json）")
    args = parser.parse_args(argv)
    res = {}
    for split in ("val", "test"):
        d = "%s_%s" % (args.dump, split)
        if os.path.isdir(d):
            res[split] = check_split(d, set(args.dense))
    if not res:
        raise RuntimeError("找不到 %s_val / %s_test" % (args.dump, args.dump))
    for split, r in res.items():
        fb, fl, rs = r["first_beat"], r["flips"], r["rescued"]
        print("%s（%d 条，极密 %s）IoU 骨干 %.4f -> 终判 %.4f | 普通录像 %.4f -> %.4f | 极密目标附近误报 %s -> %s（%s）"
              % (split, r["n_sequences"], ",".join(r["dense"]) or "-", r["iou"]["base"], r["iou"]["final"],
                 r["iou_normal"]["base"], r["iou_normal"]["final"], r["dense_near_fp"]["base"],
                 r["dense_near_fp"]["final"],
                 "%+.1f%%" % (100 * r["dense_near_fp_rise"]) if r["dense_near_fp_rise"] is not None else "-"))
        print("    第一拍（%d 个事件）召回 骨干 %.3f 零等待 %.3f 终判 %.3f（%+.3f）| 翻转 上 对 %d 错 %d 下 对 %d 错 %d | "
              "救回目标事件 %d，平均等 %s 窗" % (
                  fb["n"], fb["base"], fb["zero_wait"], fb["final"], fb["gain"], fl["up_tp"], fl["up_fp"],
                  fl["down_tn"], fl["down_fn"], rs["n"],
                  "%.2f" % rs["mean_wait_windows"] if rs["mean_wait_windows"] is not None else "-"))
    g = gates(res, args.val_iou_min)
    print("关口：", json.dumps(g, ensure_ascii=False))
    path = args.out or (args.dump + "_stage1.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump({"splits": res, "gates": g}, stream, ensure_ascii=False, indent=1)
    print("结果写入", path)
    return res, g


if __name__ == "__main__":
    main()
