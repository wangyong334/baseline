"""把 log/ 下所有 --mode eval 的结果汇总成一张对比表（按运行名分组、跨种子求均值）。

每次做完一批实验都要手写脚本读 JSON、按种子求均值、算能耗，这个工具把那件事固定下来。
读取 `log/<运行名>_seed<N>/eval_<划分>_best_val_iou_seed<N>.json`，按去掉种子后的运行名分组，输出：
    IoU / ACC / Pd / Fa（阈值 0.9，按各运行训练时的状态模式）、换成另一种状态模式后的 IoU
    每 8 秒理论能耗：稠密口径与事件驱动口径（enc1 只计非零输入）
    分段 IoU（序列前 16 窗 / 中间 / 末 16 窗）、首次检出延迟中位数、学到的 tau 范围、best checkpoint 的轮次
新结果的 JSON 里带 input_nonzero_fraction（评估时实测），旧结果没有，用 --density 给一个默认值。

用法:
    python tools/summarize_runs.py --split test                     # 全部运行
    python tools/summarize_runs.py --split val --pattern relu       # 只看某一类
    python tools/summarize_runs.py --split test --markdown --out log/energy/summary_test.json
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

MAC_PJ, AC_PJ = 4.6, 0.9
NAME_RE = re.compile(r"^eval_(val|test)_best_val_iou_seed(\d+)\.json$")


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="汇总所有 eval 结果")
    parser.add_argument("--log-root", default="log")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--pattern", default="", help="只统计运行名包含该字符串的目录")
    parser.add_argument("--windows", type=int, default=160, help="每条序列的窗口数（换算每 8 秒能耗）")
    parser.add_argument("--density", type=float, default=0.0009274,
                        help="结果文件里没有实测非零输入占比时使用的默认值（test 集实测 0.093%）")
    parser.add_argument("--markdown", action="store_true", help="输出 Markdown 表格，方便贴进 CLAUDE.md")
    parser.add_argument("--out", default=None, help="同时写一份 JSON")
    return parser.parse_args()


def run_name_and_seed(directory):
    """log/stream_v1_fast_seed37 -> ("stream_v1_fast", 37)；没有 seed 后缀时种子记为 None。"""
    match = re.match(r"^(.*)_seed(\d+)$", directory)
    return (match.group(1), int(match.group(2))) if match else (directory, None)


def energy_mj_per_8s(operations, density, windows):
    """(稠密口径, 事件驱动口径) 的每 8 秒能耗（mJ）。"""
    sop = float(operations["sop"])
    dense = (float(operations["mac"]) * MAC_PJ + sop * AC_PJ) * 1e-9 * windows
    mac_event = operations.get("mac_event_driven")
    if mac_event is None:
        enc1 = [r for r in operations["per_layer"] if r["layer"] == "enc1"][0]
        mac_event = float(operations["mac"]) - float(enc1["mac"]) + float(enc1["dense"]) * float(density)
    return dense, (float(mac_event) * MAC_PJ + sop * AC_PJ) * 1e-9 * windows


def collect(log_root, split, pattern):
    """扫描目录，返回 {运行名: [(种子, 结果字典), ...]}。"""
    runs = {}
    for directory in sorted(os.listdir(log_root)):
        if pattern and pattern not in directory:
            continue
        path = os.path.join(log_root, directory)
        if not os.path.isdir(path):
            continue
        for name in sorted(os.listdir(path)):
            match = NAME_RE.match(name)
            if not match or match.group(1) != split:
                continue
            with open(os.path.join(path, name), "r", encoding="utf-8") as stream:
                data = json.load(stream)
            run, seed = run_name_and_seed(directory)
            if seed is not None and seed != int(match.group(2)):
                continue                                   # 目录名与文件名的种子不一致，跳过
            runs.setdefault(run, []).append((seed, data))
    return runs


OTHER_MODE = {"carry": "reset_each_window", "reset_each_window": "carry"}


def summarize(entries, density, windows):
    """把一个运行的若干种子汇总成一行。

    主指标取该运行"训练时用的状态模式"（reset 训练的模型按 carry 评估没有意义），另一种模式单列一栏。
    """
    mode = entries[0][1].get("trained_state_mode", "carry")
    carry = [d[mode] for _, d in entries]
    row = {"seeds": [s for s, _ in entries], "n": len(entries), "state_mode": mode,
           "epochs": [d.get("epoch") for _, d in entries],
           "neuron": entries[0][1].get("neuron"), "parameters": entries[0][1]["parameters"]["total"]}
    for key in ("iou", "acc", "pd", "fa"):
        values = [c[key] for c in carry if c.get(key) is not None]
        if values:
            row[key] = float(np.mean(values))
            row[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[key + "_seeds"] = [float(v) for v in values]
    row["other_mode_iou"] = float(np.mean([d[OTHER_MODE[mode]]["iou"] for _, d in entries]))
    if carry[0].get("segment_iou"):
        keys = list(carry[0]["segment_iou"].keys())
        row["segment_iou"] = {k: float(np.mean([c["segment_iou"][k] for c in carry])) for k in keys}
    if carry[0].get("latency"):
        row["latency_median_ms"] = float(np.mean([c["latency"]["latency_median_ms"] for c in carry]))
        row["detection_rate"] = float(np.mean([c["latency"]["detection_rate"] for c in carry]))
    if carry[0].get("operations"):
        pairs = [energy_mj_per_8s(c["operations"], c.get("input_nonzero_fraction", density), windows)
                 for c in carry]
        row["energy_dense_mj"] = float(np.mean([p[0] for p in pairs]))
        row["energy_event_mj"] = float(np.mean([p[1] for p in pairs]))
        row["measured_density"] = all("input_nonzero_fraction" in c for c in carry)
    taus = [v["tau_mean"] for c in carry for v in (c.get("tau") or {}).values()]
    row["tau_ms"] = [float(min(taus)), float(max(taus))] if taus else None
    rates = [c["layers"]["dec1"]["firing_rate"] for c in carry if c.get("layers")]
    row["dec1_firing_rate"] = float(np.mean(rates)) if rates else None
    return row


def main():
    """入口：扫描 -> 汇总 -> 打印表格（可选 Markdown / JSON）。"""
    args = parse_args()
    runs = collect(args.log_root, args.split, args.pattern)
    if not runs:
        raise SystemExit("没有找到 eval 结果：%s（--split %s，--pattern %r）" % (args.log_root, args.split, args.pattern))
    rows = {name: summarize(entries, args.density, args.windows)
            for name, entries in sorted(runs.items())}
    order = sorted(rows, key=lambda n: -rows[n].get("iou", 0.0))
    header = ("运行", "n", "IoU", "±", "ACC", "Pd", "Fa", "换状态", "能耗 mJ/8s（稠密/事件）", "延迟", "tau ms")
    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
    else:
        print("%-34s %2s %7s %7s %7s %7s %9s %7s %22s %6s %10s" % header)
    for name in order:
        r = rows[name]
        energy = ("%.1f / %.2f" % (r["energy_dense_mj"], r["energy_event_mj"])) if "energy_dense_mj" in r else "-"
        tau = ("%.0f-%.0f" % tuple(r["tau_ms"])) if r["tau_ms"] else "-"
        cells = (name, r["n"], "%.4f" % r.get("iou", float("nan")), "%.4f" % r.get("iou_std", 0.0),
                 "%.4f" % r.get("acc", float("nan")), "%.4f" % r.get("pd", float("nan")),
                 "%.2e" % r.get("fa", float("nan")), "%.4f" % r["other_mode_iou"], energy,
                 "%.0fms" % r.get("latency_median_ms", float("nan")), tau)
        if args.markdown:
            print("| " + " | ".join(str(c) for c in cells) + " |")
        else:
            print("%-34s %2d %7s %7s %7s %7s %9s %7s %22s %6s %10s" % cells)
    print("\n阈值 0.9，主指标按各运行训练时的状态模式；换状态列是同一权重改用另一种状态模式的 IoU；"
          "能耗按每 8 秒（%d 窗）折算，事件驱动口径下 enc1 只计非零输入" % args.windows)
    estimated = [n for n in order if rows[n].get("measured_density") is False]
    if estimated:
        print("以下运行的结果文件里没有实测非零输入占比，事件驱动口径用了默认值 %.5f%%：%s"
              % (100 * args.density, ", ".join(estimated)))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump({"split": args.split, "windows": args.windows, "runs": rows}, stream,
                      indent=2, ensure_ascii=False)
        print("报告:", args.out)


if __name__ == "__main__":
    main()
