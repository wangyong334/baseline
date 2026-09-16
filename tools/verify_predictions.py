"""独立核验：只读取导出的逐事件预测 NPZ，从零重新计算指标，不依赖任何训练或评估代码。

回答四个问题：
    1. 两个方法评估的是不是同一批事件、同一份标签？（逐文件核对 locs 与 labels 是否完全相同）
    2. 指标算得对不对？（纯 numpy/scipy 从零实现 IoU/ACC/Pd/Fa；若能导入原 utils/eval.py，再算一遍对比）
    3. 指标本身是否正常？（对真实标签构造全 0、全 1、随机、完美、加噪声等平凡预测，检查指标取值范围）
    4. 提升来自哪里？（逐序列 IoU 对比、胜负统计、提升与序列属性的相关性，可选画图人工查看）

用法:
    python tools/verify_predictions.py \
        --pred baseline=log/verify/baseline_seed37_test \
        --pred stream_v1=log/verify/stream_v1_seed37_test \
        --plot-dir log/verify/plots
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WIDTH, HEIGHT = 346, 260


def parse_args():
    """解析参数。--pred 可以给多次，格式为 名称=目录。"""
    parser = argparse.ArgumentParser(description="逐事件预测的独立核验")
    parser.add_argument("--pred", action="append", required=True, help="名称=预测目录，可重复")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--pd-detT", type=float, default=50.0)
    parser.add_argument("--correct-thresh", type=float, default=0.0001)
    parser.add_argument("--window-ms", type=int, default=50)
    parser.add_argument("--plot-dir", default=None, help="若给出，为提升最大/最小的序列画 x-t 散点图")
    parser.add_argument("--plot-n", type=int, default=3)
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def load_dir(directory):
    """读取目录下全部 NPZ，返回 {文件名: 字典}；检查必需字段与长度一致。"""
    files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not files:
        raise SystemExit("目录中没有 NPZ: %s" % directory)
    out = {}
    for f in files:
        with np.load(f) as d:
            item = {k: np.asarray(d[k]) for k in d.files}
        for key in ("locs", "labels", "probabilities"):
            if key not in item:
                raise SystemExit("%s 缺少字段 %s" % (f, key))
        n = item["labels"].shape[0]
        if item["locs"].shape != (n, 4) or item["probabilities"].reshape(-1).shape[0] != n:
            raise SystemExit("%s 字段长度不一致" % f)
        item["probabilities"] = item["probabilities"].reshape(-1)
        out[os.path.basename(f)] = item
    return out


def check_alignment(methods):
    """核对所有方法的文件集合、事件坐标与标签是否逐元素完全相同。返回报告字典。

    这是整个核验的前提：只有确认两个方法面对的是同一批事件和同一份标签，指标对比才有意义。
    """
    names = list(methods)
    ref_name = names[0]
    ref = methods[ref_name]
    report = {"reference": ref_name, "files": len(ref), "events": int(sum(v["labels"].size for v in ref.values())),
              "mismatches": []}
    for other in names[1:]:
        cur = methods[other]
        if set(cur) != set(ref):
            report["mismatches"].append("%s 与 %s 的文件集合不同: 仅 %s 有 %s" % (
                other, ref_name, other, sorted(set(cur) - set(ref))[:3]))
            continue
        for fname in sorted(ref):
            a, b = ref[fname], cur[fname]
            if a["labels"].size != b["labels"].size:
                report["mismatches"].append("%s: 事件数 %d vs %d" % (fname, a["labels"].size, b["labels"].size))
                continue
            if not np.array_equal(a["locs"][:, 1:4], b["locs"][:, 1:4]):
                report["mismatches"].append("%s: 事件坐标/顺序不同" % fname)
            if not np.array_equal(a["labels"], b["labels"]):
                report["mismatches"].append("%s: 标签不同" % fname)
    return report


def iou_acc_from_scratch(items, threshold):
    """从零计算全局 IoU 与 ACC（所有序列的事件拼接后统计，与原仓库口径一致）。

    IoU = TP / (TP + FP + FN)，ACC = TP / 正事件数（正事件召回率）。
    """
    tp = fp = fn = 0
    for it in items:
        pred = it["probabilities"] >= threshold
        pos = it["labels"] == 1
        tp += int(np.count_nonzero(pred & pos))
        fp += int(np.count_nonzero(pred & ~pos))
        fn += int(np.count_nonzero(~pred & pos))
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    acc = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"iou": iou, "acc": acc, "tp": tp, "fp": fp, "fn": fn}


def pd_fa_from_scratch(items, threshold, det_t, correct_thresh):
    """从零复现原 utils/eval.py 的 roc_update + cal_roc（使用 scipy 连通域，不依赖 cv2）。

    逐项保留原实现的口径：
      * 帧划分用严格不等号 i*det_t < t < (i+1)*det_t（时间恰为 det_t 整数倍的事件不计入）
      * 帧总数用 int((t_max - t_min) / det_t)，循环帧数为该值 + 1
      * 目标检出：帧内该目标 id(!=0) 的事件中，预测等于标签的数量 / 标签和 >= correct_thresh
      * 虚警：帧内 label=0 且预测为正的事件映射到 260x346 图像，8 连通域个数
    若 NPZ 没有 target_id 字段则返回 None。
    """
    from scipy import ndimage
    if any("target_id" not in it for it in items):
        return None
    structure = np.ones((3, 3), dtype=int)
    obj_num = correct = false_num = frame_num = 0
    for it in items:
        t = it["locs"][:, 3].astype(np.float64)
        x = it["locs"][:, 1].astype(np.int64)
        y = it["locs"][:, 2].astype(np.int64)
        pred = (it["probabilities"] >= threshold).astype(np.float64)
        label = it["labels"].astype(np.float64)
        tid = it["target_id"].astype(np.float64)
        span = t.max() - t.min()
        frame_num += int(span / det_t)
        for i in range(int(span / det_t + 1)):
            m = (t > i * det_t) & (t < (i + 1) * det_t)
            if not np.any(m):
                continue
            tid_f, pred_f, lab_f = tid[m], pred[m], label[m]
            for target in set(tid_f.tolist()):
                if target == 0:
                    continue
                obj_num += 1
                sel = tid_f == target
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.sum(pred_f[sel] == lab_f[sel]) / np.sum(lab_f[sel])
                if ratio >= correct_thresh:
                    correct += 1
            false_sel = (lab_f == 0) & (pred_f == 1)
            mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            np.add.at(mask, (y[m][false_sel], x[m][false_sel]), 1)
            _, n_comp = ndimage.label(mask > 0, structure=structure)
            false_num += n_comp
    pd = correct / obj_num if obj_num else float("nan")
    fa = false_num / (frame_num * WIDTH * HEIGHT) if frame_num else float("nan")
    return {"pd": pd, "fa": fa, "targets_frames": obj_num, "false_components": false_num}


def metrics_with_original_code(items, args):
    """尝试用原仓库 utils/eval.py 的 evalute 再算一遍（需要 torch、cv2、pandas）。失败则返回原因字符串。

    调用方式与原 test.py 相同：matches 放预测与标签，miou/accuracy 全局计算，roc_update 逐序列调用。
    """
    try:
        import torch
        from types import SimpleNamespace
        from utils.eval import evalute
    except Exception as error:                       # 本地没有 torch/cv2 时跳过
        return "无法导入原评估代码: %s" % error
    ev = evalute(SimpleNamespace(roc=True, pd_detT=args.pd_detT, correct_thresh=args.correct_thresh))
    for i, it in enumerate(items):
        probs = torch.from_numpy(it["probabilities"].astype(np.float32))
        label = torch.from_numpy(it["labels"].astype(np.float32))
        ev.matches[str(i)] = {"seg_pred": probs.clone(),
                              "seg_gt": label.cuda() if torch.cuda.is_available() else label}
        if "target_id" in it:
            locs = torch.from_numpy(it["locs"].astype(np.float32))
            ev.roc_update(locs[:, 3], probs.clone(), it["target_id"], label, locs, thresh=args.threshold)
    out = {"iou": float(ev.evaluate_semantic_segmantation_miou(thresh=args.threshold))}
    if torch.cuda.is_available():
        out["acc"] = float(ev.evaluate_semantic_segmantation_accuracy(thresh=args.threshold))
    if all("target_id" in it for it in items):
        pd, fa = ev.cal_roc()
        out.update({"pd": float(pd), "fa": float(fa)})
    return out


def sanity_predictors(items, threshold, seed=0):
    """对同一份真实标签构造平凡预测，检查指标取值是否符合预期（证明指标实现本身正常）。

    期望：全 0 -> IoU 0；全 1 -> IoU = 正样本比例；随机 -> 很低；完美 -> 1；
          完美预测再随机翻转 1%/10% 的标签 -> IoU 单调下降。
    """
    rng = np.random.RandomState(seed)
    pos_total = sum(int(np.sum(it["labels"] == 1)) for it in items)
    n_total = sum(it["labels"].size for it in items)

    def run(make):
        fake = [{"labels": it["labels"], "probabilities": make(it)} for it in items]
        return iou_acc_from_scratch(fake, threshold)["iou"]

    def flipped(it, rate):
        prob = it["labels"].astype(np.float32).copy()
        flip = rng.rand(prob.size) < rate
        prob[flip] = 1.0 - prob[flip]
        return prob

    return {
        "正样本比例": pos_total / n_total,
        "全0预测": run(lambda it: np.zeros(it["labels"].size, np.float32)),
        "全1预测": run(lambda it: np.ones(it["labels"].size, np.float32)),
        "随机预测": run(lambda it: rng.rand(it["labels"].size).astype(np.float32)),
        "完美预测": run(lambda it: it["labels"].astype(np.float32)),
        "完美预测翻转1%": run(lambda it: flipped(it, 0.01)),
        "完美预测翻转10%": run(lambda it: flipped(it, 0.10)),
    }


def sequence_properties(item, window_ms):
    """计算一个序列的属性：事件数、前景比例、平均每窗事件数、平均每窗目标事件数（只数有目标的窗）。"""
    t = item["locs"][:, 3]
    lab = item["labels"] == 1
    n_win = int(t.max() // window_ms) + 1 if t.size else 1
    per_win = np.bincount(t // window_ms, minlength=n_win)
    tgt_win = np.bincount(t[lab] // window_ms, minlength=n_win)
    return {"events": int(t.size), "fg_ratio": float(lab.mean()) if t.size else 0.0,
            "events_per_window": float(per_win.mean()),
            "target_events_per_window": float(tgt_win[tgt_win > 0].mean()) if np.any(tgt_win > 0) else 0.0}


def spearman(a, b):
    """斯皮尔曼秩相关系数（用 numpy 实现，样本少时仅供参考）。"""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def plot_sequence(path, name, item_by_method, threshold, max_background=20000, seed=0):
    """画一个序列的 x-t 散点图：第一行为真实标签，其后每行一个方法（绿=TP 橙=FP 蓝=FN，灰=其余背景）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.RandomState(seed)
    methods = list(item_by_method)
    ref = item_by_method[methods[0]]
    t, x, lab = ref["locs"][:, 3], ref["locs"][:, 1], ref["labels"] == 1
    bg = np.flatnonzero(~lab)
    if bg.size > max_background:
        bg = rng.choice(bg, max_background, replace=False)
    fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(14, 3.2 * (len(methods) + 1)), sharex=True)
    axes[0].scatter(t[bg], x[bg], s=0.3, c="0.8")
    axes[0].scatter(t[lab], x[lab], s=1.0, c="red")
    axes[0].set_title("%s  ground truth (red = target)" % name)   # 图中文字用英文，避免服务器缺中文字体
    for ax, m in zip(axes[1:], methods):
        pred = item_by_method[m]["probabilities"] >= threshold
        tp, fp, fn = pred & lab, pred & ~lab, ~pred & lab
        iou = tp.sum() / max((tp | fp | fn).sum(), 1)
        ax.scatter(t[bg], x[bg], s=0.3, c="0.85")
        ax.scatter(t[fn], x[fn], s=1.0, c="tab:blue", label="FN")
        ax.scatter(t[fp], x[fp], s=1.0, c="tab:orange", label="FP")
        ax.scatter(t[tp], x[tp], s=1.0, c="tab:green", label="TP")
        ax.set_title("%s  IoU %.3f  TP %d  FP %d  FN %d" % (m, iou, tp.sum(), fp.sum(), fn.sum()))
        ax.legend(loc="upper right", markerscale=8)
    axes[-1].set_xlabel("t (ms)")
    for ax in axes:
        ax.set_ylabel("x (px)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    """入口：加载 -> 对齐核对 -> 从零算指标 -> 原代码复算 -> 平凡预测 -> 逐序列分析 -> 可选画图。"""
    args = parse_args()
    methods = {}
    for spec in args.pred:
        if "=" not in spec:
            raise SystemExit("--pred 格式应为 名称=目录")
        name, directory = spec.split("=", 1)
        methods[name] = load_dir(directory)
    names = list(methods)
    result = {}

    print("=" * 96)
    print("【1】数据对齐核对（各方法是否评估同一批事件、同一份标签）")
    align = check_alignment(methods)
    result["alignment"] = align
    print("  参照: %s | 文件数 %d | 事件总数 %d" % (align["reference"], align["files"], align["events"]))
    if align["mismatches"]:
        for m in align["mismatches"][:10]:
            print("  ✗", m)
        print("  ⚠ 存在不一致，后续对比不可信")
    else:
        print("  ✓ 所有方法的文件、事件坐标与顺序、标签逐元素完全一致")

    print("=" * 96)
    print("【2】指标：从零实现 vs 原仓库 utils/eval.py（阈值 %.2f）" % args.threshold)
    result["metrics"] = {}
    for m in names:
        items = [methods[m][f] for f in sorted(methods[m])]
        scratch = iou_acc_from_scratch(items, args.threshold)
        roc = pd_fa_from_scratch(items, args.threshold, args.pd_detT, args.correct_thresh)
        if roc:
            scratch.update(roc)
        original = metrics_with_original_code(items, args)
        result["metrics"][m] = {"from_scratch": scratch, "original_code": original}
        line = "  %-12s 从零: IoU %.4f ACC %.4f" % (m, scratch["iou"], scratch["acc"])
        if roc:
            line += " Pd %.4f Fa %.3e" % (roc["pd"], roc["fa"])
        print(line)
        if isinstance(original, dict):
            keys = [k for k in ("iou", "acc", "pd", "fa") if k in original]
            diffs = {k: abs(original[k] - scratch[k]) for k in keys if k in scratch}
            print("  %-12s 原代码: %s | 最大差异 %.2e" % (
                "", " ".join("%s %.6g" % (k.upper(), original[k]) for k in keys), max(diffs.values())))
        else:
            print("  %-12s 原代码: %s" % ("", original))

    print("=" * 96)
    print("【3】指标自检：同一份标签上的平凡预测")
    ref_items = [methods[names[0]][f] for f in sorted(methods[names[0]])]
    sanity = sanity_predictors(ref_items, args.threshold)
    result["sanity"] = sanity
    for k, v in sanity.items():
        print("  %-16s %.4f" % (k, v))

    print("=" * 96)
    print("【4】逐序列 IoU")
    files = sorted(methods[names[0]])
    rows = []
    header = "  %-16s %8s %7s %8s %8s" % ("序列", "事件数", "前景%", "事件/窗", "目标/窗") + \
             "".join(" %11s" % m[:11] for m in names)
    print(header)
    for f in files:
        props = sequence_properties(methods[names[0]][f], args.window_ms)
        ious = {m: iou_acc_from_scratch([methods[m][f]], args.threshold)["iou"] for m in names}
        rows.append((f, props, ious))
        print("  %-16s %8d %6.2f%% %8.1f %8.1f" % (f[:16], props["events"], 100 * props["fg_ratio"],
                                                 props["events_per_window"], props["target_events_per_window"]) +
              "".join(" %11.4f" % ious[m] for m in names))
    result["per_sequence"] = [{"file": f, **p, "iou": i} for f, p, i in rows]

    if len(names) >= 2:
        a, b = names[0], names[-1]
        gains = np.array([r[2][b] - r[2][a] for r in rows])
        valid = ~np.isnan(gains)
        print("=" * 96)
        print("【5】%s 相对 %s 的逐序列提升" % (b, a))
        print("  胜 %d / 平 %d / 负 %d（|差值|<0.01 记为平）" % (
            int(np.sum(gains[valid] >= 0.01)), int(np.sum(np.abs(gains[valid]) < 0.01)),
            int(np.sum(gains[valid] <= -0.01))))
        print("  提升 中位数 %+.4f | 最小 %+.4f | 最大 %+.4f" % (
            np.median(gains[valid]), gains[valid].min(), gains[valid].max()))
        for key, label in (("events_per_window", "每窗事件数"), ("fg_ratio", "前景比例"),
                           ("target_events_per_window", "每窗目标事件数")):
            vals = np.array([r[1][key] for r in rows])[valid]
            print("  提升与%-8s的秩相关 %+.2f" % (label, spearman(vals, gains[valid])))
        base_iou = np.array([r[2][a] for r in rows])[valid]
        print("  提升与%s自身IoU的秩相关 %+.2f（负值表示：%s越差的序列提升越大）" % (a, spearman(base_iou, gains[valid]), a))
        result["gain_summary"] = {"method_a": a, "method_b": b, "median_gain": float(np.median(gains[valid]))}

        if args.plot_dir:
            os.makedirs(args.plot_dir, exist_ok=True)
            order = np.argsort(gains)
            picks = list(order[-args.plot_n:][::-1]) + list(order[:args.plot_n])
            for k in dict.fromkeys(picks):
                f = rows[k][0]
                out = os.path.join(args.plot_dir, "%s_gain%+.3f.png" % (f.replace(".npz", ""), gains[k]))
                plot_sequence(out, f, {m: methods[m][f] for m in names}, args.threshold)
                print("  已画图:", out)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, default=float)
        print("结果已写入", args.out_json)
    print("=" * 96)


if __name__ == "__main__":
    main()
