"""逐事件错误类型分解（V2 定稿设计第 11 节阶段 0 的 D1；只读 eval 导出，纯 CPU，几分钟）。

回答"剩下的错误在哪里、修掉哪一类能涨多少 IoU"，用来决定时序后端该解决什么，而不是凭假设设计。
对每个读出、每个阈值，把逐事件错误分成（优先级从上到下，一个事件只进一类）：
    漏检 FN（目标事件判成背景）
        start     序列开头 start_ms 内（网络状态还没建立）
        onset     该目标首次出现后 onset_ms 内（目标刚出现）
        edge      该目标本窗事件中离质心较远的外圈（距离 > 本窗均方根半径，且本窗该目标 >= 4 个事件）
        core      其余（稳定跟踪的目标上的漏检）
    误检 FP（背景事件判成目标），按到本窗最近目标事件的距离：
        adjacent  <= near_px[0] 像素（紧贴目标）
        near      <= near_px[1] 像素
        far       更远
        no_target 本窗没有目标事件（纯背景窗）
      另报 far / no_target 中"同一像素在该序列里有误检的不同窗数 >= repeat_min"的数量（热像素 / 静态闪烁；一次突发只算一窗）。
每一类给出：事件数、占该类错误的比例、"只修好这一类时的 IoU"（IoU = TP/(TP+FP+FN) 的逐事件口径，与原评估一致）。
多个读出时，另报相对第一个读出每一类被修好 / 新增的数量（例如 net -> fused_d2 修掉了哪类漏检）。

用法（服务器，读已有导出，不用 GPU）：
    python tools/error_breakdown.py --dump-dir log/verify/v21_floor4_base_s37_test \\
        log/verify/v21_floor4_base_s37_test:prob_fused_d2 --thresholds 0.9 0.7 --out log/energy/errors_s37_test.json
目录写法与 tools/sweep_threshold.py 相同（目录:字段，默认 probabilities）。
字段可以写成几个 logit 字段相加（目录:logit_net+evidence_d2+delta_attr_d2），概率取 sigmoid(和)，
用来离线试组合读出（例如 V2-1 融合 + V2-2 修正），不用 GPU。--names 只看指定序列，--per-sequence 另报逐序列结果。
"""
import argparse
import json
import os
from collections import OrderedDict

import numpy as np

FN_CLASSES = ("start", "onset", "edge", "core")
FP_CLASSES = ("adjacent", "near", "far", "no_target")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="逐事件错误类型分解")
    parser.add_argument("--dump-dir", nargs="+", required=True, help="目录或 目录:字段（第一个读出作为比较基准）")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.9])
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--start-ms", type=float, default=800.0)
    parser.add_argument("--onset-ms", type=float, default=250.0)
    parser.add_argument("--near-px", nargs=2, type=float, default=[3.0, 10.0])
    parser.add_argument("--repeat-min", type=int, default=5)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--names", nargs="+", default=None, help="只看这些序列（文件名，可省略 .npz）")
    parser.add_argument("--per-sequence", action="store_true", help="另报每条序列在第一个阈值下的结果")
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def split_spec(spec):
    """"目录"或"目录:字段" -> (目录, 字段)。目录本身存在时不拆（兼容 Windows 盘符）。"""
    if os.path.isdir(spec) or ":" not in spec:
        return spec, "probabilities"
    directory, key = spec.rsplit(":", 1)
    return directory, key


def read_field(data, key):
    """读一个概率字段；key 含 "+" 时把这些 logit 字段相加再取 sigmoid。"""
    if "+" not in key:
        return np.asarray(data[key]).astype(np.float32)
    total = sum(np.asarray(data[k]).astype(np.float64) for k in key.split("+"))
    return (1.0 / (1.0 + np.exp(-total))).astype(np.float32)


def load_sequences(specs, max_sequences=0, names_filter=None):
    """按目录读取导出：每个目录只读一次坐标、标签、目标编号，再取各读出字段。返回 (序列列表, 读出名列表)。"""
    by_dir = OrderedDict()
    for spec in specs:
        directory, key = split_spec(spec)
        by_dir.setdefault(directory, []).append((spec, key))
    names, sequences = [], None
    for directory, fields in by_dir.items():
        files = sorted(n for n in os.listdir(directory) if n.endswith(".npz"))
        if names_filter:
            wanted = set(n if n.endswith(".npz") else n + ".npz" for n in names_filter)
            files = [n for n in files if n in wanted]
        if max_sequences:
            files = files[:int(max_sequences)]
        if not files:
            raise RuntimeError("目录里没有 NPZ: %s" % directory)
        loaded = []
        for fname in files:
            with np.load(os.path.join(directory, fname)) as data:
                seq = {"name": fname, "locs": np.asarray(data["locs"]), "labels": np.asarray(data["labels"]) > 0.5,
                       "target_id": np.asarray(data["target_id"]).astype(np.int64), "probs": {}}
                for spec, key in fields:
                    seq["probs"][spec] = read_field(data, key)
            loaded.append(seq)
        if sequences is None:
            sequences = loaded
        else:
            if [s["name"] for s in sequences] != [s["name"] for s in loaded]:
                raise RuntimeError("各目录的序列不一致: %s" % directory)
            for base, extra in zip(sequences, loaded):
                if base["labels"].shape != extra["labels"].shape:
                    raise RuntimeError("事件数不一致: %s / %s" % (directory, extra["name"]))
                base["probs"].update(extra["probs"])
        names += [spec for spec, _ in fields]
    return sequences, names


def event_classes(seq, window_ms, start_ms, onset_ms):
    """目标事件的漏检类别（按优先级）与每个事件的窗号。返回 (fn_class [N] 整数，-1 表示背景事件, window [N])。"""
    locs, lab, tid = seq["locs"], seq["labels"], seq["target_id"]
    x, y, t = locs[:, 1].astype(np.float64), locs[:, 2].astype(np.float64), locs[:, 3].astype(np.float64)
    window = np.floor(t / window_ms).astype(np.int64)
    cls = np.full(lab.shape[0], -1, dtype=np.int64)
    idx = np.nonzero(lab)[0]
    if idx.size == 0:
        return cls, window
    ids = tid[idx]
    first = {}
    for i, tt in zip(ids.tolist(), t[idx].tolist()):
        first[i] = min(first.get(i, tt), tt)
    age = t[idx] - np.array([first[i] for i in ids.tolist()])
    # 每个 (目标, 窗) 的质心与均方根半径
    _, group = np.unique(ids * 1000003 + window[idx], return_inverse=True)
    count = np.bincount(group)
    cx = np.bincount(group, x[idx]) / count
    cy = np.bincount(group, y[idx]) / count
    d2 = (x[idx] - cx[group]) ** 2 + (y[idx] - cy[group]) ** 2
    rms = np.sqrt(np.bincount(group, d2) / count)
    edge = (count[group] >= 4) & (np.sqrt(d2) > rms[group])
    c = np.full(idx.size, FN_CLASSES.index("core"), dtype=np.int64)
    c[edge] = FN_CLASSES.index("edge")
    c[age < onset_ms] = FN_CLASSES.index("onset")
    c[t[idx] < start_ms] = FN_CLASSES.index("start")
    cls[idx] = c
    return cls, window


def fp_classes(seq, window, fp_mask, near_px, repeat_min):
    """误检事件的类别 [N]（非误检为 -1）与"重复像素"标记 [N]。"""
    locs, lab = seq["locs"], seq["labels"]
    x, y = locs[:, 1].astype(np.float64), locs[:, 2].astype(np.float64)
    cls = np.full(lab.shape[0], -1, dtype=np.int64)
    fp_idx = np.nonzero(fp_mask)[0]
    repeat = np.zeros(lab.shape[0], dtype=bool)
    if fp_idx.size == 0:
        return cls, repeat
    # 重复像素按"该像素有误检的不同窗数"计（一次突发在同一窗里产生多个事件只算一窗）
    pix = (y[fp_idx] * 100000 + x[fp_idx]).astype(np.int64)
    pix_windows = np.unique(pix * 4096 + window[fp_idx])
    uniq_pix, n_windows = np.unique(pix_windows // 4096, return_counts=True)
    repeat[fp_idx] = n_windows[np.searchsorted(uniq_pix, pix)] >= int(repeat_min)
    tgt = np.nonzero(lab)[0]
    order = np.argsort(window[tgt], kind="stable")
    tw = window[tgt][order]
    tx, ty = x[tgt][order], y[tgt][order]
    for w in np.unique(window[fp_idx]):
        sel = fp_idx[window[fp_idx] == w]
        lo, hi = np.searchsorted(tw, w, "left"), np.searchsorted(tw, w, "right")
        if hi == lo:
            cls[sel] = FP_CLASSES.index("no_target")
            continue
        dist = np.sqrt((x[sel, None] - tx[None, lo:hi]) ** 2 + (y[sel, None] - ty[None, lo:hi]) ** 2).min(1)
        c = np.full(sel.size, FP_CLASSES.index("far"), dtype=np.int64)
        c[dist <= near_px[1]] = FP_CLASSES.index("near")
        c[dist <= near_px[0]] = FP_CLASSES.index("adjacent")
        cls[sel] = c
    return cls, repeat


def breakdown(sequences, names, thresholds, args):
    """全部读出 x 阈值的分解结果（跨序列累加）。"""
    cache = [event_classes(s, args.window_ms, args.start_ms, args.onset_ms) for s in sequences]
    out = OrderedDict()
    for th in thresholds:
        per = OrderedDict()
        base_pred = None
        for ni, name in enumerate(names):
            tot = {"tp": 0, "fp": 0, "fn": 0}
            fn_cnt = dict.fromkeys(FN_CLASSES, 0)
            fp_cnt = dict.fromkeys(FP_CLASSES, 0)
            fp_rep = {"far": 0, "no_target": 0}
            target_cnt = dict.fromkeys(FN_CLASSES, 0)
            fixed = {"fn": dict.fromkeys(FN_CLASSES, 0), "fp": dict.fromkeys(FP_CLASSES, 0)}
            new = {"fn": dict.fromkeys(FN_CLASSES, 0), "fp": dict.fromkeys(FP_CLASSES, 0)}
            preds = []
            for si, (seq, (tcls, window)) in enumerate(zip(sequences, cache)):
                lab = seq["labels"]
                pred = seq["probs"][name] >= th
                preds.append(pred)
                tp, fn, fp = lab & pred, lab & ~pred, ~lab & pred
                tot["tp"] += int(tp.sum())
                tot["fn"] += int(fn.sum())
                tot["fp"] += int(fp.sum())
                for k, c in enumerate(FN_CLASSES):
                    fn_cnt[c] += int((fn & (tcls == k)).sum())
                    target_cnt[c] += int(lab[tcls == k].sum())
                fcls, rep = fp_classes(seq, window, fp, args.near_px, args.repeat_min)
                for k, c in enumerate(FP_CLASSES):
                    fp_cnt[c] += int((fcls == k).sum())
                for c in ("far", "no_target"):
                    fp_rep[c] += int(((fcls == FP_CLASSES.index(c)) & rep).sum())
                if base_pred is not None:
                    bp = base_pred[si]
                    bfn, bfp = lab & ~bp, ~lab & bp
                    bfcls, _ = fp_classes(seq, window, bfp, args.near_px, args.repeat_min)
                    for k, c in enumerate(FN_CLASSES):
                        m = tcls == k
                        fixed["fn"][c] += int((bfn & m & pred).sum())
                        new["fn"][c] += int((~bfn & m & fn).sum())
                    for k, c in enumerate(FP_CLASSES):
                        fixed["fp"][c] += int(((bfcls == k) & ~pred).sum())
                        new["fp"][c] += int(((fcls == k) & ~bp).sum())
            if base_pred is None:
                base_pred = preds
            denom = float(max(tot["tp"] + tot["fp"] + tot["fn"], 1))
            row = {"iou": tot["tp"] / denom, "recall": tot["tp"] / float(max(tot["tp"] + tot["fn"], 1)),
                   "precision": tot["tp"] / float(max(tot["tp"] + tot["fp"], 1)), "counts": tot,
                   "targets": target_cnt,
                   "fn": {c: {"n": fn_cnt[c], "share": fn_cnt[c] / float(max(tot["fn"], 1)),
                              "miss_rate": fn_cnt[c] / float(max(target_cnt[c], 1)),
                              "iou_if_fixed": (tot["tp"] + fn_cnt[c]) / denom} for c in FN_CLASSES},
                   "fp": {c: {"n": fp_cnt[c], "share": fp_cnt[c] / float(max(tot["fp"], 1)),
                              "iou_if_fixed": tot["tp"] / max(denom - fp_cnt[c], 1.0)} for c in FP_CLASSES},
                   "fp_repeat_pixels": fp_rep}
            if ni > 0:
                row["vs_first"] = {"fixed": fixed, "new": new}
            per[name] = row
        out["%g" % th] = per
    return out


def per_sequence(sequences, names, threshold, args):
    """每条序列、每个读出的 IoU 与主要错误数（紧贴 / 附近误检、边缘漏检）。"""
    rows = []
    for seq in sequences:
        tcls, window = event_classes(seq, args.window_ms, args.start_ms, args.onset_ms)
        lab = seq["labels"]
        row = {"name": seq["name"], "readouts": {}}
        for name in names:
            pred = seq["probs"][name] >= threshold
            tp, fn, fp = int((lab & pred).sum()), lab & ~pred, ~lab & pred
            fcls, _ = fp_classes(seq, window, fp, args.near_px, args.repeat_min)
            row["readouts"][name] = {
                "iou": tp / float(max(tp + int(fp.sum()) + int(fn.sum()), 1)), "tp": tp,
                "fp": int(fp.sum()), "fn": int(fn.sum()),
                "fp_adjacent": int((fcls == FP_CLASSES.index("adjacent")).sum()),
                "fp_near": int((fcls == FP_CLASSES.index("near")).sum()),
                "fn_edge": int((fn & (tcls == FN_CLASSES.index("edge"))).sum())}
        rows.append(row)
    return rows


def print_per_sequence(rows, names, threshold):
    short = [n.rsplit(":", 1)[-1] if ":" in n and not os.path.isdir(n) else "probabilities" for n in names]
    print("\n=== 逐序列（阈值 %g）：IoU | 紧贴误检 / 附近误检 / 边缘漏检 ===" % threshold)
    print("%-14s" % "序列" + "".join("%-30s" % s[:28] for s in short))
    for row in rows:
        cells = []
        for name in names:
            r = row["readouts"][name]
            cells.append("%.4f | %d/%d/%d" % (r["iou"], r["fp_adjacent"], r["fp_near"], r["fn_edge"]))
        print("%-14s" % row["name"][:-4] + "".join("%-30s" % c for c in cells))


def print_report(result, names):
    for th, per in result.items():
        print("\n=== 阈值 %s ===" % th)
        for name in names:
            r = per[name]
            c = r["counts"]
            print("\n%s\n  IoU %.4f  召回 %.4f  精确率 %.4f  | TP %d FP %d FN %d" % (
                name, r["iou"], r["recall"], r["precision"], c["tp"], c["fp"], c["fn"]))
            print("  漏检  " + "  ".join("%s %d（占 %.0f%%，该类漏检率 %.1f%%，修好后 IoU %.4f）" % (
                k, v["n"], 100 * v["share"], 100 * v["miss_rate"], v["iou_if_fixed"]) for k, v in r["fn"].items()))
            print("  误检  " + "  ".join("%s %d（占 %.0f%%，修好后 IoU %.4f）" % (
                k, v["n"], 100 * v["share"], v["iou_if_fixed"]) for k, v in r["fp"].items()))
            print("  误检中的重复像素（>= repeat_min 个不同窗）：far %d，no_target %d" % (
                r["fp_repeat_pixels"]["far"], r["fp_repeat_pixels"]["no_target"]))
            if "vs_first" in r:
                f, n = r["vs_first"]["fixed"], r["vs_first"]["new"]
                print("  相对第一个读出：修好漏检 " + "，".join("%s %d/新增 %d" % (k, f["fn"][k], n["fn"][k]) for k in FN_CLASSES)
                      + "；修好误检 " + "，".join("%s %d/新增 %d" % (k, f["fp"][k], n["fp"][k]) for k in FP_CLASSES))


def main(argv=None):
    args = parse_args(argv)
    sequences, names = load_sequences(args.dump_dir, args.max_sequences, args.names)
    print("%d 条序列，%d 个事件，读出：%s" % (len(sequences), sum(s["labels"].shape[0] for s in sequences), names))
    result = breakdown(sequences, names, args.thresholds, args)
    print_report(result, names)
    rows = None
    if args.per_sequence:
        rows = per_sequence(sequences, names, args.thresholds[0], args)
        print_per_sequence(rows, names, args.thresholds[0])
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump({"args": vars(args), "readouts": names, "result": result, "per_sequence": rows}, stream,
                      indent=2, ensure_ascii=False)
        print("\n报告:", args.out)
    return result


if __name__ == "__main__":
    main()
