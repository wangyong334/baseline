"""持续性门控的离线检验（只读已有 eval 导出，纯 CPU；不用 GPU、不改模型）。

问题（09-26 错误分解）：V2-1 融合读出 z = m + w*F 假设背景事件在时间上独立（泊松），闪烁点、热像素这类原地持续的杂波
违反这一点，同一位置的同一种活动被一次次当成新证据，把它推过阈值；融合读出新增的远处误检里约六成落在反复误报的像素上，
密集序列 test_022/023 的误检也集中在这类位置。目标在运动（每窗中位位移约 2.6 像素），原地持续"像目标"的响应多半是杂波。

持续性（只用过去的窗，因果）：
    D_k(x) = 1[ x 的 3x3 邻域内有像素在第 k 窗的原始融合概率 sigmoid(m + w*F) >= tau_h ]
    h_k(x) = rho * h_{k-1}(x) + D_{k-1}(x),  rho = exp(-1 / tau_w)      （按窗计数，一次突发只算一窗）
    q = 1 / (1 + (h / h0)^4)                                            （h << h0 时 q ≈ 1，h >> h0 时 q ≈ 0）
    计数用未抑制的原始分数，避免"压下去就不再计数、恢复后又误报"的振荡。第 k 窗事件在第 k+d 窗末发布，
    用到的第 j <= k-1 窗的 F 在第 j+d 窗末已知，满足因果。
两个版本（m = logit_net，F = evidence_d{d}，w = 1 即 V2-1 的 fused_d{d}）：
    A 保守版  z = m + w * (q * max(F, 0) + min(F, 0))    只削减融合读出加上的正分，SNN 初判与负证据不动
    B 总分版  z = m + w * F - beta * (1 - q)             对总分扣分，也能压 SNN 自己在重复位置的误检（可能误伤慢速/悬停目标）
选择：参数与阈值在 val 上按 IoU 选定、冻结到 test（协议 "val"）；另报原协议阈值 0.9（参数在 val 上按 0.9 选）。
报告：逐事件 IoU / 召回（= ACC）/ 精确率，Pd / Fa（原 utils/eval.py，只算最终选定的几组，可 --no-pd 跳过），
      指定序列的 IoU，以及错误分类的变化（tools/error_breakdown.py 的口径；相对 fused 修好 / 新增了哪类错误，真实目标被误伤多少）。

用法（服务器，读 baseline 已有导出）：
    python tools/persistence_gate.py --val-dump log/verify/v21_floor4_base_s37_val \\
        --test-dump log/verify/v21_floor4_base_s37_test --out log/energy/persistence_gate_s37.json
"""
import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from tools import error_breakdown as eb  # noqa: E402

HEIGHT, WIDTH = 260, 346
DEFAULT_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]      # 0.50 ... 0.95


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="持续性门控的离线检验")
    parser.add_argument("--val-dump", required=True)
    parser.add_argument("--test-dump", required=True)
    parser.add_argument("--delay", type=int, default=2, help="用 evidence_d{d}（与 fused_d{d} 对应）")
    parser.add_argument("--weight", type=float, default=1.0, help="融合权重 w（1 = V2-1 fused）")
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--n-windows", type=int, default=160)
    parser.add_argument("--tau-h", type=float, nargs="+", default=[0.5, 0.8], help="计入持续性的原始融合概率门槛")
    parser.add_argument("--tau-w", type=float, nargs="+", default=[20.0, 40.0, 80.0], help="泄漏计数的时间常数（窗）")
    parser.add_argument("--h0", type=float, nargs="+", default=[4.0, 8.0, 12.0, 20.0], help="持续性尺度（窗）")
    parser.add_argument("--beta", type=float, nargs="+", default=[2.0, 4.0, 8.0], help="B 版的扣分强度（logit）")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--fixed-threshold", type=float, default=0.9, help="原协议阈值")
    parser.add_argument("--seqs", nargs="*", default=["test_022", "test_023", "test_003", "test_008"],
                        help="单独报告 IoU 的测试序列")
    parser.add_argument("--no-pd", action="store_true", help="跳过 Pd/Fa（原评估代码较慢）")
    parser.add_argument("--pd-detT", type=int, default=50)
    parser.add_argument("--correct-thresh", type=float, default=1e-4)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------- 数据
def load_split(directory, delay, max_sequences=0):
    """读一个导出目录：坐标、标签、目标编号、logit_net、evidence_d{d}，并核对 sigmoid(m + F) 与导出的 fused 概率一致。"""
    names = sorted(n for n in os.listdir(directory) if n.endswith(".npz"))
    if max_sequences:
        names = names[:int(max_sequences)]
    if not names:
        raise RuntimeError("目录里没有 NPZ: %s" % directory)
    out, worst = [], 0.0
    for name in names:
        with np.load(os.path.join(directory, name)) as data:
            for key in ("logit_net", "evidence_d%d" % delay):
                if key not in data.files:
                    raise SystemExit("%s 缺少字段 %s（需要 V2 的 eval 导出）" % (name, key))
            seq = {"name": name, "locs": np.asarray(data["locs"]), "labels": np.asarray(data["labels"]) > 0.5,
                   "target_id": np.asarray(data["target_id"]).astype(np.int64),
                   "m": np.asarray(data["logit_net"]).astype(np.float64),
                   "F": np.asarray(data["evidence_d%d" % delay]).astype(np.float64)}
            key = "prob_fused_d%d" % delay
            if key in data.files:
                ref = np.asarray(data[key]).astype(np.float64)
                worst = max(worst, float(np.abs(sigmoid(seq["m"] + seq["F"]) - ref).max()) if ref.size else 0.0)
        out.append(seq)
    return out, worst


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    return math.log(p / (1.0 - p))


def dilate3(D):
    """[T,H,W] 布尔数组的 3x3 膨胀（画面外按 False）。"""
    out = D.copy()
    H, W = D.shape[1], D.shape[2]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out[:, max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] |= \
                D[:, max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return out


def persistence(seq, tau_h, tau_ws, weight, window_ms, n_windows):
    """每个事件处的持续性计数 h（每个 tau_w 一个数组）。只用第 k 窗之前的窗。"""
    x = seq["locs"][:, 1].astype(np.int64)
    y = seq["locs"][:, 2].astype(np.int64)
    k = np.clip((seq["locs"][:, 3] // window_ms).astype(np.int64), 0, n_windows - 1)
    H, W = max(HEIGHT, int(y.max()) + 1 if y.size else HEIGHT), max(WIDTH, int(x.max()) + 1 if x.size else WIDTH)
    raw = sigmoid(seq["m"] + weight * seq["F"]) >= tau_h
    D = np.zeros((n_windows, H, W), dtype=bool)
    D[k[raw], y[raw], x[raw]] = True
    D = dilate3(D)
    rhos = np.array([math.exp(-1.0 / float(t)) for t in tau_ws])
    h = np.zeros((len(tau_ws), H, W), dtype=np.float32)
    out = np.zeros((len(tau_ws), x.size), dtype=np.float32)
    order = np.argsort(k, kind="stable")
    bounds = np.searchsorted(k[order], np.arange(n_windows + 1), side="left")
    for kk in range(n_windows):
        sel = order[bounds[kk]:bounds[kk + 1]]
        if sel.size:
            out[:, sel] = h[:, y[sel], x[sel]]
        h = (rhos[:, None, None] * h + D[kk][None]).astype(np.float32)
    return {t: out[i] for i, t in enumerate(tau_ws)}


def gate(h, h0):
    return 1.0 / (1.0 + (h / float(h0)) ** 4)


def scores(m, F, weight, variant, q=None, beta=0.0):
    """各版本的 logit：net / fused / A / B。"""
    if variant == "net":
        return m
    if variant == "fused":
        return m + weight * F
    if variant == "A":
        return m + weight * (q * np.maximum(F, 0.0) + np.minimum(F, 0.0))
    if variant == "B":
        return m + weight * F - beta * (1.0 - q)
    raise ValueError(variant)


def iou_curve(z, labels, thresholds):
    """各阈值下的 (IoU, 召回, 精确率, TP, FP, FN)。"""
    out = []
    pos = labels
    for th in thresholds:
        pred = z >= logit(th)
        tp = int(np.count_nonzero(pos & pred))
        fp = int(np.count_nonzero(~pos & pred))
        fn = int(np.count_nonzero(pos & ~pred))
        out.append((tp / float(max(tp + fp + fn, 1)), tp / float(max(tp + fn, 1)), tp / float(max(tp + fp, 1)),
                    tp, fp, fn))
    return out


# ---------------------------------------------------------------------------- 选择
def configurations(args):
    """版本 -> [(参数字典)]。"""
    out = OrderedDict([("net", [{}]), ("fused", [{}])])
    out["A"] = [{"tau_h": th, "tau_w": tw, "h0": h0} for th in args.tau_h for tw in args.tau_w for h0 in args.h0]
    out["B"] = [{"tau_h": th, "tau_w": tw, "h0": h0, "beta": b}
                for th in args.tau_h for tw in args.tau_w for h0 in args.h0 for b in args.beta]
    return out


def prepare(sequences, args):
    """拼接整个划分，并预先算好各 (tau_h, tau_w) 的持续性。"""
    cat = {"m": np.concatenate([s["m"] for s in sequences]), "F": np.concatenate([s["F"] for s in sequences]),
           "labels": np.concatenate([s["labels"] for s in sequences])}
    hist = {}
    for tau_h in args.tau_h:
        per = [persistence(s, tau_h, args.tau_w, args.weight, args.window_ms, args.n_windows) for s in sequences]
        for tau_w in args.tau_w:
            hist[(tau_h, tau_w)] = np.concatenate([p[tau_w] for p in per])
    cat["hist"] = hist
    return cat


def variant_logit(cat, variant, params, weight):
    if variant in ("net", "fused"):
        return scores(cat["m"], cat["F"], weight, variant)
    q = gate(cat["hist"][(params["tau_h"], params["tau_w"])], params["h0"])
    return scores(cat["m"], cat["F"], weight, variant, q, params.get("beta", 0.0))


def select(val, configs, args):
    """每个版本在 val 上选两种：固定阈值（原协议）下 IoU 最高的参数；参数 + 阈值一起选。"""
    chosen = OrderedDict()
    fixed = float(args.fixed_threshold)
    thresholds = sorted(set(args.thresholds) | {fixed})
    for variant, grid in configs.items():
        best_fixed, best_free = None, None
        for params in grid:
            curve = iou_curve(variant_logit(val, variant, params, args.weight), val["labels"], thresholds)
            for th, row in zip(thresholds, curve):
                if th == fixed and (best_fixed is None or row[0] > best_fixed[2]):
                    best_fixed = (params, th, row[0])
                if th in args.thresholds and (best_free is None or row[0] > best_free[2]):
                    best_free = (params, th, row[0])
        chosen[(variant, "0.9")] = best_fixed
        chosen[(variant, "val")] = best_free
    return chosen


# ---------------------------------------------------------------------------- 报告
def detection_metrics(sequences, probs, threshold, args):
    """Pd / Fa：调用原 utils/eval.py 的 roc_update 与 cal_roc（与 tools/sweep_threshold.py 相同）。"""
    import torch
    from utils.eval import evalute
    evaluator = evalute(SimpleNamespace(roc=True, pd_detT=args.pd_detT, correct_thresh=args.correct_thresh))
    for seq, p in zip(sequences, probs):
        locs = seq["locs"]
        zeros = np.zeros(locs.shape[0], dtype=np.float32)
        ev_locs = torch.from_numpy(np.stack([zeros, locs[:, 1], locs[:, 2], locs[:, 3]], 1).astype(np.float32))
        evaluator.roc_update(ev_locs[:, 3], torch.from_numpy(p.astype(np.float32)), seq["target_id"],
                             torch.from_numpy(seq["labels"].astype(np.float32)), ev_locs, thresh=float(threshold))
    pd, fa = evaluator.cal_roc()
    return float(pd), float(fa)


def split_by_sequence(z, sequences):
    out, start = [], 0
    for s in sequences:
        n = s["labels"].shape[0]
        out.append(z[start:start + n])
        start += n
    return out


def describe(params):
    return " ".join("%s=%g" % kv for kv in params.items()) if params else "-"


def main(argv=None):
    args = parse_args(argv)
    val_seqs, val_err = load_split(args.val_dump, args.delay, args.max_sequences)
    test_seqs, test_err = load_split(args.test_dump, args.delay, args.max_sequences)
    print("val %d 条 / test %d 条序列；sigmoid(m + F) 与导出 fused_d%d 的最大差 %.2e / %.2e" % (
        len(val_seqs), len(test_seqs), args.delay, val_err, test_err), flush=True)
    if max(val_err, test_err) > 1e-4:
        print("警告：导出的 fused 概率与 sigmoid(logit_net + evidence) 不一致，检查 --delay / --weight", flush=True)
    val, test = prepare(val_seqs, args), prepare(test_seqs, args)
    configs = configurations(args)
    chosen = select(val, configs, args)
    seq_names = [n if n.endswith(".npz") else n + ".npz" for n in args.seqs]
    report = OrderedDict()
    binary = OrderedDict()
    print("\n=== 参数与阈值在 val 上选定、冻结到 test（逐事件 IoU / 召回 / 精确率%s）===" % ("" if args.no_pd else " / Pd / Fa"))
    header = "%-6s %-5s %-38s %5s %8s %8s %7s %7s" % ("版本", "协议", "参数（val 选定）", "阈值", "val IoU", "test IoU",
                                                     "召回", "精确率")
    if not args.no_pd:
        header += " %7s %9s" % ("Pd", "Fa")
    header += "".join(" %9s" % n[:-4] for n in seq_names)
    print(header)
    for (variant, proto), (params, th, val_iou) in chosen.items():
        z = variant_logit(test, variant, params, args.weight)
        iou, rec, prec, tp, fp, fn = iou_curve(z, test["labels"], [th])[0]
        row = {"variant": variant, "protocol": proto, "params": params, "threshold": th, "val_iou": val_iou,
               "test": {"iou": iou, "recall": rec, "precision": prec, "tp": tp, "fp": fp, "fn": fn}}
        per_seq = split_by_sequence(z, test_seqs)
        if not args.no_pd:
            row["test"]["pd"], row["test"]["fa"] = detection_metrics(test_seqs, [sigmoid(v) for v in per_seq], th, args)
        row["sequences"] = {}
        for seq, zs in zip(test_seqs, per_seq):
            if seq["name"] in seq_names:
                row["sequences"][seq["name"][:-4]] = iou_curve(zs, seq["labels"], [th])[0][0]
        report["%s@%s" % (variant, proto)] = row
        binary["%s@%s" % (variant, proto)] = [(zs >= logit(th)).astype(np.float32) for zs in per_seq]
        line = "%-6s %-5s %-38s %5.2f %8.4f %8.4f %7.4f %7.4f" % (variant, proto, describe(params)[:38], th, val_iou,
                                                               iou, rec, prec)
        if not args.no_pd:
            line += " %7.4f %9.2e" % (row["test"]["pd"], row["test"]["fa"])
        line += "".join(" %9.4f" % row["sequences"].get(n[:-4], float("nan")) for n in seq_names)
        print(line, flush=True)

    # 错误分类的变化（相对 fused，同一协议）：真实目标被误伤多少、修掉了哪类误检
    ns = SimpleNamespace(window_ms=args.window_ms, start_ms=800.0, onset_ms=250.0, near_px=[3.0, 10.0], repeat_min=5)
    breakdowns = OrderedDict()
    for proto in ("val", "0.9"):
        names = ["fused@" + proto, "A@" + proto, "B@" + proto]
        seqs = [{"name": s["name"], "locs": s["locs"], "labels": s["labels"], "target_id": s["target_id"],
                 "probs": {n: binary[n][i] for n in names}} for i, s in enumerate(test_seqs)]
        print("\n=== test 错误分类（协议 %s，各自选定的阈值；第一个为 fused，后两个给出相对 fused 的修好 / 新增）===" % proto)
        result = eb.breakdown(seqs, names, [0.5], ns)
        eb.print_report(result, names)
        breakdowns[proto] = result["0.5"]
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump({"args": vars(args), "results": report, "breakdown": breakdowns}, stream, indent=2,
                      ensure_ascii=False)
        print("\n报告:", args.out)
    return report


if __name__ == "__main__":
    main()
