"""V2 的三段能耗口径：骨干 + 证据前端 + 判决层，按每窗与每 8 秒汇总。

为什么需要它：`tools/stream_energy.py` 只换算骨干（`operations_backbone`），而 v2-1 在骨干之外
新增了两块每窗都要跑的稠密实数运算——前端的逐像素时间矩递推/背景估计/极性偶极卷积，以及判决层在
V 个速度假设 × 全画面上的证据累加。不把它们计进来，能耗数字是不成立的。

口径（与 tools/baseline_energy.py、tools/stream_energy.py 一致）：
    MAC 4.6 pJ，AC（加法/比较等 elementwise）0.9 pJ
    exp/log/log1p/softplus 没有公认的单位代价，用 --transcendental-macs 折算成 MAC（默认 10），
    并同时报告 1 / 10 / 20 三档，避免结论依赖这一个假设。
骨干有"稠密"与"事件驱动"两种口径（见 stream_energy.py）；前端的逐事件部分本来就按事件计，
判决层是完全稠密的——事件再稀疏，每窗仍要在 V × H × W 上更新，这一点必须在论文里写明。

用法:
    python tools/energy_v2.py --eval-json log/v21_seed37/eval_test_best_val_iou_seed37.json \
        --windows 160 --out log/energy/energy_v2_test.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import json  # noqa: E402

MAC_PJ = 4.6
AC_PJ = 0.9
TRANSCENDENTAL_SCALES = (1.0, 10.0, 20.0)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="V2 三段能耗（骨干 / 前端 / 判决层）")
    parser.add_argument("--eval-json", nargs="+", required=True, help="train_stream_v2.py --mode eval 的结果文件")
    parser.add_argument("--state-mode", choices=("carry", "reset_each_window"), default="carry")
    parser.add_argument("--windows", type=int, default=160, help="每条序列的窗数（8 秒 / 50 ms = 160）")
    parser.add_argument("--transcendental-macs", type=float, default=10.0,
                        help="一次 exp/log 折算成几次 MAC（默认 10；报告里同时给 1/10/20 三档）")
    parser.add_argument("--baseline-mj", type=float, default=174.1,
                        help="基线 K5 每 8 秒能耗（mJ），默认用 CLAUDE.md 记录的 174.1")
    parser.add_argument("--out", default="log/energy/energy_v2.json")
    return parser.parse_args()


def energy_pj(mac, ac, transcendental, scale):
    """理论能耗（pJ）。transcendental 按 scale 倍 MAC 折算。"""
    return float(mac) * MAC_PJ + float(ac) * AC_PJ + float(transcendental) * scale * MAC_PJ


def backbone_terms(ops, event_driven):
    """骨干的 (mac, ac, transcendental)。事件驱动口径把 enc1 换成只算非零输入。"""
    mac = float(ops["mac_event_driven"]) if (event_driven and "mac_event_driven" in ops) else float(ops["mac"])
    return mac, float(ops["sop"]), 0.0


def module_terms(ops):
    """前端/判决层的 (mac, ac, transcendental)。"""
    return float(ops["mac"]), float(ops["elementwise"]), float(ops["transcendental"])


def analyse(result, windows, scale):
    """把一份 eval 结果里的三段运算量换算成每窗/每 8 秒能耗。"""
    rows = []
    for name, terms in (("骨干（稠密）", backbone_terms(result["operations_backbone"], False)),
                        ("骨干（事件驱动）", backbone_terms(result["operations_backbone"], True)),
                        ("证据前端", module_terms(result["operations_frontend"])),
                        ("判决层", module_terms(result["operations_decision"]))):
        mac, ac, tr = terms
        per_window = energy_pj(mac, ac, tr, scale) * 1e-9        # pJ -> mJ
        rows.append({"part": name, "mac": mac, "ac": ac, "transcendental": tr,
                     "mj_per_window": per_window, "mj_per_8s": per_window * windows})
    dense = rows[0]["mj_per_8s"] + rows[2]["mj_per_8s"] + rows[3]["mj_per_8s"]
    event = rows[1]["mj_per_8s"] + rows[2]["mj_per_8s"] + rows[3]["mj_per_8s"]
    return {"rows": rows, "total_dense_mj_per_8s": dense, "total_event_driven_mj_per_8s": event}


def sparse_decision(result, windows, scale, total_without_decision):
    """稀疏同步执行下的判决层能耗：eval 结果里 operations_decision_sparse 的每个门控档位一行。

    活跃比例取自 ε = 0 的运行时是"门控到 ε 之后"的比例（门控后的 G 恰好是未门控 G 的截断），
    但精度要用 --cusum-gate-eps ε 的运行另测。旧结果没有这个字段时返回空字典。
    """
    out = {}
    for eps, ops in sorted(result.get("operations_decision_sparse", {}).items(), key=lambda kv: float(kv[0])):
        mac, ac, tr = module_terms(ops)
        mj = energy_pj(mac, ac, tr, scale) * 1e-9 * windows
        out[eps] = {"active_fraction": float(ops.get("active_fraction", 1.0)), "decision_mj_per_8s": mj,
                    "total_event_driven_mj_per_8s": total_without_decision + mj}
    return out


def print_report(path, result, report, baseline_mj, scale):
    """终端表格。"""
    print("\n########", path)
    print("神经元 %s | 每窗事件 %.0f | 输入非零占比 %.4f | exp/log 折算 %.0f 倍 MAC" % (
        result.get("neuron"), result.get("events_per_window", float("nan")),
        result.get("input_nonzero_fraction", float("nan")), scale))
    print("%-18s %12s %12s %14s %14s %10s" % ("部分", "MAC", "AC", "exp/log", "mJ/8s", "占比"))
    total = report["total_event_driven_mj_per_8s"]
    for row in report["rows"]:
        share = "—" if row["part"] == "骨干（稠密）" else "%.1f%%" % (100 * row["mj_per_8s"] / max(total, 1e-12))
        print("%-18s %12.4g %12.4g %14.4g %14.3f %10s" % (
            row["part"], row["mac"], row["ac"], row["transcendental"], row["mj_per_8s"], share))
    print("合计（骨干稠密口径）      %.2f mJ/8s" % report["total_dense_mj_per_8s"])
    print("合计（骨干事件驱动口径）  %.2f mJ/8s" % total)
    if baseline_mj > 0:
        print("对基线 K5（%.1f mJ/8s）的倍数：稠密 %.2fx，事件驱动 %.2fx" % (
            baseline_mj, report["total_dense_mj_per_8s"] / baseline_mj, total / baseline_mj))


def main():
    """入口。"""
    args = parse_args()
    out = {"transcendental_macs": args.transcendental_macs, "windows": args.windows, "runs": {}}
    for path in args.eval_json:
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        result = data[args.state_mode]
        missing = [k for k in ("operations_backbone", "operations_frontend", "operations_decision")
                   if k not in result]
        if missing:
            print("跳过 %s：缺少 %s（需要用新版 train_stream_v2.py 重新评估）" % (path, missing))
            continue
        result = dict(result)
        result["neuron"] = data.get("neuron")
        report = analyse(result, args.windows, args.transcendental_macs)
        report["sensitivity"] = {
            str(s): analyse(result, args.windows, s)["total_event_driven_mj_per_8s"]
            for s in TRANSCENDENTAL_SCALES}
        print_report(path, result, report, args.baseline_mj, args.transcendental_macs)
        rows = {row["part"]: row["mj_per_8s"] for row in report["rows"]}
        report["decision_sparse"] = sparse_decision(result, args.windows, args.transcendental_macs,
                                                    rows["骨干（事件驱动）"] + rows["证据前端"])
        for eps, row in report["decision_sparse"].items():
            print("判决层稀疏同步（门控 ε=%s，足迹活跃比例 %.4f）：%.2f mJ/8s；整机（事件驱动口径）%.2f mJ/8s" % (
                eps, row["active_fraction"], row["decision_mj_per_8s"], row["total_event_driven_mj_per_8s"]))
        print("exp/log 折算敏感性（事件驱动口径 mJ/8s）：" +
              "，".join("%.0fx -> %.2f" % (s, report["sensitivity"][str(s)]) for s in TRANSCENDENTAL_SCALES))
        out["runs"][path] = report
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(out, stream, indent=1, ensure_ascii=False)
    print("\n报告:", args.out)


if __name__ == "__main__":
    main()
