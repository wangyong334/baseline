"""V6 的自适应发布（13.12；选择规则 10-23 按 13.15 修订）：在 val 导出上定规则，在 test 导出上报告四个指标。

输入为 train_stream_lag.py --mode eval --dump 导出的 dump_<split>_<variant>.npz（同一 variant 的 val 与 test）。
规则与选法（看 test 之前写死）：
    1. 截止 D* = val 上固定滞后 IoU 最高的 d（并列取小）。
    2. 发布规则 = utils.lag_metrics.adaptive_publish（双边界、随年龄线性收拢、到 D* 强制发布、发布即最终）。
    3. 边距 m = 候选网格中，val 上保住等待增益的比例 >= rho 的最小值：IoU(m) >= IoU(0) + rho * (IoU(D*) - IoU(0))
       （m 越小发布越早；m = 0 即零等待，m = inf 即固定 D*）。rho 是需求常数（默认 0.9：保住九成等待增益）。
       V6a 用的是绝对容差 IoU(D*) - 0.005：等待增益本身只有 0.007 时，容差吃掉了大半增益，选出的 m 几乎不等（13.15）。
       D* = 0（等待没有增益）时 m 无意义，全部在出生窗发布。
    4. 把 (D*, m) 用到 test：报告四个指标，以及机制诊断（各年龄发布比例、提前发布相对截止判决的翻转率、误检中目标旁的比例）。
固定滞后 d = 0..D 的四个指标同时列出作参照（d = 0 即 V5）。延迟类指标与能耗一样放到最后阶段统一计算。

用法:
    python tools/lag_publish.py --val log/<run>/dump_val_multiscale.npz --test log/<run>/dump_test_multiscale.npz \
        --out log/<run>/publish_multiscale.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lag_metrics import adaptive_publish, iou_acc_counts, split_report  # noqa: E402

DEFAULT_MARGINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, math.inf)


def load_dump(path):
    """读取导出：返回 (prob [D+1, N], 每条序列的字典列表, meta)。"""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    prob = data["prob"]
    offsets = meta["offsets"]
    seqs = []
    for i in range(len(offsets) - 1):
        a, b = offsets[i], offsets[i + 1]
        seqs.append({"name": meta["names"][i], "slice": (a, b), "t": data["t"][a:b], "x": data["x"][a:b],
                     "y": data["y"][a:b], "label": data["label"][a:b].astype(np.float32),
                     "target_id": data["target_id"][a:b].astype(np.float64), "win": data["win"][a:b],
                     "avail": data["avail"][a:b].astype(np.int64)})
    return prob, seqs, meta


def concat(seqs, key):
    return np.concatenate([s[key] for s in seqs])


def choose_rule(prob, seqs, threshold, margins, retain):
    """在 val 上定 (D*, m)。返回 (deadline, margin, 选择过程的记录)。"""
    if not 0.0 <= float(retain) <= 1.0:
        raise ValueError("retain 必须在 [0, 1] 内")
    if not margins or any(not (m >= 0) for m in margins) or list(margins) != sorted(margins):
        raise ValueError("候选边距必须非负且从小到大排列")
    label, avail = concat(seqs, "label"), concat(seqs, "avail")
    fixed = [iou_acc_counts(prob[d], label, threshold)[0] for d in range(prob.shape[0])]
    deadline = int(np.argmax(fixed))                     # argmax 并列时取第一个，即最小的 d
    target = fixed[0] + float(retain) * (fixed[deadline] - fixed[0])
    rows, chosen = [], None
    for m in margins:
        pub, age, _ = adaptive_publish(prob, avail, deadline, m, threshold)
        iou = iou_acc_counts(pub, label, threshold)[0]
        rows.append({"margin": m if math.isfinite(m) else "inf", "iou": iou, "mean_age": float(age.mean())})
        if chosen is None and iou >= target:
            chosen = m
    if chosen is None:
        chosen = math.inf
    return deadline, chosen, {"fixed_iou": fixed, "deadline": deadline, "retain": float(retain), "target_iou": target,
                              "margins": rows, "margin": chosen if math.isfinite(chosen) else "inf"}


def apply_rule(prob, seqs, cfg, deadline, margin):
    """在一个划分上按 (D*, m) 发布并报告；另报提前发布相对截止判决的翻转率。"""
    thr = float(cfg["threshold"])
    pubs, ages = [], []
    for s in seqs:
        a, b = s["slice"]
        pub, age, _ = adaptive_publish(prob[:, a:b], s["avail"], deadline, margin, thr)
        pubs.append(pub)
        ages.append(age)
    report = split_report(seqs, lambda i: pubs[i], lambda i: ages[i], cfg)
    age, pub, avail = np.concatenate(ages), np.concatenate(pubs), concat(seqs, "avail")
    final = prob[np.minimum(deadline, avail), np.arange(prob.shape[1])] >= np.float32(thr)
    early = age < np.minimum(deadline, avail)
    flips = (pub >= np.float32(thr)) != final
    report["early_fraction"] = float(early.mean()) if early.size else float("nan")
    report["flip_rate_vs_deadline"] = float(flips[early].mean()) if early.any() else 0.0
    return report


def main():
    parser = argparse.ArgumentParser(description="V6 adaptive publishing replay (val -> test)")
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--retain", type=float, default=0.9,
                        help="val 上至少保住的等待增益比例 rho：IoU(m) >= IoU(0) + rho * (IoU(D*) - IoU(0))")
    parser.add_argument("--margins", default=None, help="逗号分隔的候选边距（logit 单位，可含 inf）")
    args = parser.parse_args()
    margins = DEFAULT_MARGINS if args.margins is None else tuple(float(v) for v in args.margins.split(","))
    val_prob, val_seqs, val_meta = load_dump(args.val)
    test_prob, test_seqs, test_meta = load_dump(args.test)
    if val_meta["variant"] != test_meta["variant"] or val_meta["lags"] != test_meta["lags"]:
        raise ValueError("val 与 test 的导出不是同一个 variant / D")
    cfg = {k: test_meta[k] for k in ("threshold", "pd_detT", "correct_thresh", "height", "width_px")}
    deadline, margin, selection = choose_rule(val_prob, val_seqs, float(cfg["threshold"]), margins, args.retain)
    fixed = {}
    for d in range(test_prob.shape[0]):
        fixed[str(d)] = split_report(test_seqs, lambda i, d=d: test_prob[d, slice(*test_seqs[i]["slice"])],
                                     lambda i, d=d: np.minimum(d, test_seqs[i]["avail"]), cfg)
    adaptive_test = apply_rule(test_prob, test_seqs, cfg, deadline, margin)
    adaptive_val = apply_rule(val_prob, val_seqs, cfg, deadline, margin)
    out = {"variant": test_meta["variant"], "lags": test_meta["lags"], "retain": args.retain,
           "selection_on_val": selection, "adaptive_val": adaptive_val,
           "adaptive_test": adaptive_test, "fixed_test": fixed}
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(out, stream, indent=2, ensure_ascii=False)
    fmt = "%-10s IoU %.4f ACC %.4f Pd %.4f Fa %.3e | 平均发布年龄 %.3f 窗 | 误检中目标旁 %.1f%%"
    print("variant %s：val 上选出 D* = %d、m = %s（保住等待增益的 %.0f%%：val IoU 目标 %.4f）" % (
        out["variant"], deadline, selection["margin"], 100.0 * args.retain, selection["target_iou"]))
    for row in selection["margins"]:
        print("  val m=%-5s IoU %.4f 平均发布年龄 %.4f 窗" % (row["margin"], row["iou"], row["mean_age"]))
    for d in range(test_prob.shape[0]):
        r = fixed[str(d)]
        print(fmt % ("test d=%d" % d, r["iou"], r["acc"], r["pd"], r["fa"], r["mean_age"],
                     100.0 * r["fp_near_target_fraction"]))
    r = adaptive_test
    print(fmt % ("test 自适应", r["iou"], r["acc"], r["pd"], r["fa"], r["mean_age"],
                 100.0 * r["fp_near_target_fraction"]))
    print("提前发布比例 %.3f，提前发布相对截止判决的翻转率 %.4f，各年龄发布数 %s" % (
        r["early_fraction"], r["flip_rate_vs_deadline"], r["age_hist"]))
    print("PUBLISH FINISHED:", args.out)


if __name__ == "__main__":
    main()
