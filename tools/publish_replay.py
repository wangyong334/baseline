"""V3 阶段 1 的离线回放：用评估导出的逐事件证据，重放与在线发布层同一个决策函数，得到精度–延迟–虚警的完整比较。

在线发布层（train_stream_v2.py --publish on）一次只给一个工作点；阈值 θ 改变时发布时刻也跟着变，必须重放整个策略，
不能对已经停下的分数再扫阈值。本工具读取 `--mode eval --dump-dir` 的导出（需要 readout_delays 覆盖 1..D，例如
--readout-delays 1 2 3 4 5），第 d 窗的证据分数 z(d) = mark + w·F(d)，调用 model/publish_readout.PublishRule.replay。

做的事：
    1 val：固定延迟 d = 0..D 与各发布配置，都在"基线 net@threshold 的虚警率"上二分出阈值；按"超出可实现前沿的 IoU"
      选一个配置（可实现前沿 = 固定延迟各点 + 相邻两个固定延迟的随机混合，每个事件按固定种子随机取其一，实际重算指标）
    2 test：冻结该配置，两种口径——部署口径（θ 用 val 选出的值）与同虚警率口径（在 test 上重新二分到基线虚警率）
    3 等待安全曲线：固定边界 logit(threshold) 下，Fa(d) / Fa(0)
    4 首次检出延迟：net、固定延迟、发布策略（按逐事件发布时刻，utils/stream_metrics.first_detection_latencies_by_event）
    5 一致性（可选）：导出里有在线运行的 label_pub / age_pub 时，与同参数的回放逐事件比对（--check-json 给出该次评估的结果 JSON）
Pd / Fa 与原仓库 utils/eval.py 口径逐位相同（向量化实现；tests/test_stream_v3_publish.py 对照）。

用法（服务器，先导出 val 与 test）：
    python tools/publish_replay.py --val <val 导出目录> --test <test 导出目录> --deadline 5 --out <结果 json>
"""
import argparse
import glob
import itertools
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.publish_readout import REASONS, PublishRule  # noqa: E402
from utils.stream_metrics import (first_detection_latencies_by_event, first_detection_latencies_published,  # noqa: E402
                                  summarize_latencies)

H, W = 260, 346


def parse_args():
    p = argparse.ArgumentParser(description="V3 发布层的离线回放")
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--deadline", type=int, default=5)
    p.add_argument("--collapse", nargs="+", default=["linear", "step"])
    p.add_argument("--upper", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--lower", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--gate", nargs="+", default=["none"], help="归属门槛（logit）列表，none = 不门控")
    p.add_argument("--weight", type=float, default=1.0, help="融合权重 w（与评估时的 fusion_weight 相同）")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--window-ms", type=int, default=50)
    p.add_argument("--n-windows", type=int, default=160, help="每条序列的窗数（与 YAML 的 n_windows 相同）")
    p.add_argument("--pd-detT", type=int, default=50)
    p.add_argument("--correct-thresh", type=float, default=1e-4)
    p.add_argument("--mix", type=float, nargs="+", default=[0.25, 0.5, 0.75], help="相邻固定延迟随机混合的比例")
    p.add_argument("--check-json", default=None, help="在线评估的结果 JSON（含 publish_cfg），用于一致性比对")
    p.add_argument("--max-seqs", type=int, default=0)
    p.add_argument("--out", default="publish_replay.json")
    return p.parse_args()


class EventSet(object):
    """一个划分的全部事件：各年龄的证据 F[d]、标签与 utils/eval.py 口径的 IoU / ACC / Pd / Fa（输入为布尔预测）。"""

    def __init__(self, directory, args):
        files = sorted(glob.glob(os.path.join(directory, "*.npz")))
        if args.max_seqs:
            files = files[:args.max_seqs]
        if not files:
            raise SystemExit("目录里没有 NPZ: %s" % directory)
        D, T = int(args.deadline), int(args.window_ms)
        cols = {k: [] for k in ("label", "logit", "prob", "k", "t", "tid", "avail", "seq")}
        F = [[] for _ in range(D + 1)]
        online = {k: [] for k in ("label_pub", "age_pub", "reason_pub")}
        fid_all, pix_all, valid_all = [], [], []
        frame_base, frames = 0, 0
        self.seq_slices = []
        start = 0
        for i, f in enumerate(files):
            z = np.load(f)
            missing = ["evidence_d%d" % d for d in range(1, D + 1) if "evidence_d%d" % d not in z.files]
            if missing:
                raise SystemExit("%s 缺少 %s：评估时加 --readout-delays %s" % (
                    os.path.basename(f), missing, " ".join(str(d) for d in range(1, D + 1))))
            locs = z["locs"]
            x, y, t = locs[:, 1].astype(np.int64), locs[:, 2].astype(np.int64), locs[:, 3].astype(np.int64)
            k = t // T
            cols["label"].append(z["labels"] > 0.5)
            cols["logit"].append(z["logit_net"].astype(np.float64))
            cols["prob"].append(z["probabilities"].astype(np.float32))
            cols["k"].append(k)
            cols["t"].append(t)
            cols["tid"].append(z["target_id"].astype(np.float64))
            cols["avail"].append(np.minimum(D, int(args.n_windows) - 1 - k))
            cols["seq"].append(np.full(x.size, i, np.int32))
            F[0].append(np.zeros(x.size))
            for d in range(1, D + 1):
                F[d].append(z["evidence_d%d" % d].astype(np.float64))
            for key in online:
                online[key].append(z[key] if key in z.files else None)
            tf = locs[:, 3].astype(np.float32)
            n_fr = int((tf.max() - tf.min()) / args.pd_detT)
            frames += n_fr
            fr = t // args.pd_detT
            valid_all.append((t % args.pd_detT != 0) & (fr <= n_fr))
            fid_all.append(frame_base + fr)
            frame_base += n_fr + 1
            pix_all.append(y * W + x)
            self.seq_slices.append(slice(start, start + x.size))
            start += x.size
        for key, v in cols.items():
            setattr(self, key, np.concatenate(v))
        self.F = np.stack([np.concatenate(v) for v in F])                   # [D+1, N]
        self.online = {key: np.concatenate(v) if all(a is not None for a in v) else None for key, v in online.items()}
        self.n_seqs, self.frames, self.D = len(files), frames, D
        valid = np.concatenate(valid_all)
        fid, pix = np.concatenate(fid_all), np.concatenate(pix_all)
        neg = np.flatnonzero(valid & ~self.label)
        order = np.argsort(fid[neg], kind="stable")
        self.neg_idx, self.neg_fid, self.neg_pix = neg[order], fid[neg][order], pix[neg][order]
        obj = np.flatnonzero(valid & (self.tid != 0))
        _, self.obj_group = np.unique(fid[obj] * 100000 + self.tid[obj].astype(np.int64), return_inverse=True)
        self.obj_idx = obj
        self.n_obj = int(self.obj_group.max()) + 1 if obj.size else 0
        self.obj_lsum = np.bincount(self.obj_group, self.label[obj].astype(np.float64), minlength=self.n_obj)
        self.buf = np.zeros((H, W), np.uint8)

    def fa(self, pred):
        keep = pred[self.neg_idx]
        f, p = self.neg_fid[keep], self.neg_pix[keep]
        if f.size == 0:
            return 0.0
        cut = np.flatnonzero(np.diff(f)) + 1
        flat, total = self.buf.ravel(), 0
        for s, e in zip(np.concatenate([[0], cut]), np.concatenate([cut, [f.size]])):
            if e - s == 1:
                total += 1
                continue
            flat[p[s:e]] = 1
            total += cv2.connectedComponents(self.buf, connectivity=8)[0] - 1
            flat[p[s:e]] = 0
        return total / float(self.frames * W * H)

    def metrics(self, pred, delay, args):
        lab = self.label
        tp = int(np.count_nonzero(pred & lab))
        fp = int(np.count_nonzero(pred & ~lab))
        fn = int(np.count_nonzero(~pred & lab))
        correct = (pred[self.obj_idx] == lab[self.obj_idx]).astype(np.float64)
        csum = np.bincount(self.obj_group, correct, minlength=self.n_obj)
        with np.errstate(divide="ignore", invalid="ignore"):
            pd = float(np.count_nonzero(csum / self.obj_lsum >= args.correct_thresh)) / max(self.n_obj, 1)
        publish = self.k + delay
        prob = pred.astype(np.float32)
        records = []
        for sl in self.seq_slices:
            records += first_detection_latencies_by_event(self.t[sl], lab[sl].astype(np.float32), self.tid[sl], prob[sl],
                                                          args.window_ms, 0.5, args.correct_thresh, publish[sl])
        lat = summarize_latencies(records)
        return {"iou": tp / float(max(tp + fp + fn, 1)), "acc": tp / float(max(tp + fn, 1)), "pd": pd, "fa": self.fa(pred),
                "delay_target_ms": float(args.window_ms) * float(delay[lab].mean()),
                "delay_all_ms": float(args.window_ms) * float(delay.mean()),
                "first_detection_median_ms": lat.get("latency_median_ms"), "first_detection_mean_ms": lat.get("latency_mean_ms")}


def z_by_age(es, rule):
    """[D+1, N]：年龄 0 为 mark，其余为 rule.score(mark, F(d))（与在线发布层同一表达式）。"""
    return np.stack([es.logit] + [rule.score(es.logit, es.F[d]) for d in range(1, es.D + 1)])


def bisect(es, policy, target, lo=-30.0, hi=60.0, steps=30):
    """policy(θ) -> (布尔预测, 逐事件发布年龄)。找 Fa 不超过 target 的最低 θ。"""
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if es.fa(policy(mid)[0]) <= target:
            hi = mid
        else:
            lo = mid
    return hi


def fixed_policy(es, d, weight):
    z = es.logit + weight * es.F[d]
    delay = np.minimum(d, es.avail).astype(np.float64)
    return lambda theta: (z >= theta, delay)


def mixed_policy(es, d_lo, d_hi, q, weight, seed=0):
    """相邻两个固定延迟的随机混合：每个事件按固定种子以概率 q 取 d_hi，否则 d_lo（可实现的前沿点）。"""
    pick = np.random.RandomState(seed + 1000 * d_hi).rand(es.label.size) < q
    z = np.where(pick, es.logit + weight * es.F[d_hi], es.logit + weight * es.F[d_lo])
    delay = np.minimum(np.where(pick, d_hi, d_lo), es.avail).astype(np.float64)
    return lambda theta: (z >= theta, delay)


def publish_policy(es, spec, weight):
    D, collapse, a, b, gate = spec

    def run(theta):
        rule = PublishRule(theta, a, b, D, collapse, gate, weight)
        label, age, _, _ = rule.replay(z_by_age(es, rule), es.avail)
        return label, age.astype(np.float64)
    return run


def frontier(es, target, args):
    """可实现前沿：固定延迟 0..D 与相邻两者的随机混合，各自在 target 虚警率上的指标，按平均目标延迟排序。"""
    pts = []
    for d in range(es.D + 1):
        pol = fixed_policy(es, d, args.weight)
        m = es.metrics(*pol(bisect(es, pol, target)), args)
        pts.append(dict(m, name="d%d" % d))
    for d in range(es.D):
        for q in args.mix:
            pol = mixed_policy(es, d, d + 1, q, args.weight)
            m = es.metrics(*pol(bisect(es, pol, target)), args)
            pts.append(dict(m, name="d%d/d%d@%.2f" % (d, d + 1, q)))
    return sorted(pts, key=lambda r: r["delay_target_ms"])


def front_at(pts, delay_ms, key):
    """前沿在给定平均目标延迟处的值：可实现点之间线性插值（点足够密，插值误差小）。"""
    xs = [p["delay_target_ms"] for p in pts]
    return float(np.interp(delay_ms, xs, [p[key] for p in pts]))


def line(tag, m, pts=None):
    s = "  %-34s IoU %.4f ACC %.4f Pd %.4f Fa %.2e | 目标平均发布延迟 %5.1f ms | 首次检出延迟中位 %s ms" % (
        tag, m["iou"], m["acc"], m["pd"], m["fa"], m["delay_target_ms"],
        "—" if m["first_detection_median_ms"] is None else "%.0f" % m["first_detection_median_ms"])
    if pts is not None:
        s += " | 超出可实现前沿 IoU %+.4f Pd %+.4f" % (m["iou"] - front_at(pts, m["delay_target_ms"], "iou"),
                                                   m["pd"] - front_at(pts, m["delay_target_ms"], "pd"))
    return s


def main():
    args = parse_args()
    t0 = time.time()
    sets = {"val": EventSet(args.val, args), "test": EventSet(args.test, args)}
    report = {"args": vars(args)}
    thr_logit = math.log(args.threshold / (1.0 - args.threshold))
    gates = [None if str(g).lower() in ("none", "null") else float(g) for g in args.gate]
    specs = list(itertools.product([args.deadline], args.collapse, args.upper, args.lower, gates))
    targets = {name: es.fa(es.prob >= np.float32(args.threshold)) for name, es in sets.items()}
    for name, es in sets.items():
        print("[%s] %d 条序列，%d 个事件；同虚警率目标（net@%.2f）%.3e（累计 %.0f s）" % (
            name, es.n_seqs, es.label.size, args.threshold, targets[name], time.time() - t0), flush=True)

    # 1 val：前沿与配置选择
    ev = sets["val"]
    pts_val = frontier(ev, targets["val"], args)
    print("\n== val 可实现前沿（同虚警率 %.3e）" % targets["val"])
    for p in pts_val:
        if "/" not in p["name"]:
            print(line("固定 " + p["name"], p))
    rows, best = {}, None
    for spec in specs:
        pol = publish_policy(ev, spec, args.weight)
        theta = bisect(ev, pol, targets["val"])
        m = dict(ev.metrics(*pol(theta), args), theta=theta)
        m["margin_iou"] = m["iou"] - front_at(pts_val, m["delay_target_ms"], "iou")
        key = "D%d_%s_a%g_b%g_g%s" % spec
        rows[key] = m
        print(line(key, m, pts_val), flush=True)
        if best is None or m["margin_iou"] > rows[best]["margin_iou"]:
            best = key
    spec = specs[list(rows).index(best)]
    print("  val 选出：%s（θ = %.4f，超出前沿 IoU %+.4f）" % (best, rows[best]["theta"], rows[best]["margin_iou"]))
    report["val"] = {"target_fa": targets["val"], "frontier": pts_val, "rows": rows, "chosen": best}

    # 2 test：部署口径与同虚警率口径
    et = sets["test"]
    pol = publish_policy(et, spec, args.weight)
    deploy = dict(et.metrics(*pol(rows[best]["theta"]), args), theta=rows[best]["theta"])
    pts_test = frontier(et, targets["test"], args)
    theta_eq = bisect(et, pol, targets["test"])
    equal = dict(et.metrics(*pol(theta_eq), args), theta=theta_eq)
    best_fixed = max((p for p in pts_test if "/" not in p["name"]), key=lambda p: p["iou"])
    print("\n== test：冻结 %s" % best)
    for p in pts_test:
        if "/" not in p["name"]:
            print(line("固定 %s（同虚警率 %.3e）" % (p["name"], targets["test"]), p))
    print(line("发布（同虚警率，θ=%.3f）" % theta_eq, equal, pts_test))
    print(line("发布（部署口径，θ 取 val 的 %.3f）" % deploy["theta"], deploy))
    print("  对最好的固定延迟 %s：IoU %+.4f，Pd %+.4f，目标平均发布延迟 %.1f ms 对 %.1f ms" % (
        best_fixed["name"], equal["iou"] - best_fixed["iou"], equal["pd"] - best_fixed["pd"],
        equal["delay_target_ms"], best_fixed["delay_target_ms"]))
    report["test"] = {"target_fa": targets["test"], "frontier": pts_test, "equal_fa": equal, "deploy": deploy,
                      "best_fixed": best_fixed}

    # 3 等待安全曲线
    print("\n== 等待安全：固定边界 logit(%.2f) 下 Fa(d) / Fa(0)" % args.threshold)
    safety = {}
    for name, es in sets.items():
        base = es.fa(es.logit >= thr_logit)
        curve = [es.fa(es.logit + args.weight * es.F[d] >= thr_logit) / max(base, 1e-12) for d in range(es.D + 1)]
        safety[name] = curve
        print("  %-5s " % name + " / ".join("d%d %.2f" % (d, v) for d, v in enumerate(curve)))
    report["wait_safety"] = safety

    # 4 一致性：在线发布层 vs 回放
    if args.check_json:
        results = json.load(open(args.check_json, encoding="utf-8"))
        cfg = results["publish_cfg"]
        if cfg is None:
            raise SystemExit("%s 不是 --publish on 的评估结果" % args.check_json)
        rule = PublishRule(cfg["theta"], cfg["upper"], cfg["lower"], cfg["deadline"], cfg["collapse"], cfg["gate"],
                           cfg["weight"])
        split = next(v["split"] for v in results.values() if isinstance(v, dict) and "split" in v)
        for name, es in sets.items():
            if name != split or es.online["label_pub"] is None:     # 只比对这次在线评估对应的划分
                continue
            if rule.deadline != es.D:
                print("  一致性：在线期限 %d 与 --deadline %d 不同，跳过" % (rule.deadline, es.D))
                continue
            label, age, reason, _ = rule.replay(z_by_age(es, rule), es.avail)
            same = (label == (es.online["label_pub"] > 0.5)) & (age == es.online["age_pub"].astype(np.int64)) & (
                reason == es.online["reason_pub"].astype(np.int64))
            print("  一致性（%s）：在线与回放逐事件相同 %.5f%%（%d / %d 不同；差异只应来自 float32/float64 在边界上的取整）" % (
                name, 100.0 * same.mean(), int((~same).sum()), same.size))
            report.setdefault("consistency", {})[name] = float(same.mean())
    report["reasons"] = list(REASONS)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print("\n报告: %s（用时 %.0f s）" % (args.out, time.time() - t0))


if __name__ == "__main__":
    main()
