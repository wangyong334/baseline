"""把流式 SNN 的评估结果换算成每 8 秒能耗，分"稠密"和"事件驱动"两种口径，并与原基线同口径对比。

输入: train_stream_v1.py --mode eval 生成的 eval_<split>_*.json（其中 operations 是单个窗口的理论运算量）。
口径（常数与 tools/baseline_energy.py 相同：MAC 4.6 pJ，AC 0.9 pJ）：
    dense  原 estimate_operations：enc1 在整张 264x352 画布（含 99% 以上的空像素）上计 MAC
    event  事件驱动：enc1 只对非零输入元素计算。卷积按输入散射时，一个非零输入元素驱动 C_out × k² 次乘加，
           所以运算数 = 非零输入元素数 × C_out × k² = 稠密运算数 × 非零输入占比。
           其余层本来就按"发放率 × 稠密运算数"计 SOP，读出按事件计，两种口径相同。
           基线的稀疏卷积只在有事件的体素上计算，对应的就是这个口径。
非零输入占比由数据直接算出，与权重无关（计数 > 0 的元素归一化后仍 > 0）。
每 8 秒 = 每窗 × 窗数（160）。
另报告"膜电位衰减乘法"上界：每窗每个状态元素一次乘法（按 MAC）。常见 SNN 能耗估计不计这一项，
基线同样不计 ReLU 与残差加法，这里单列、不计入总数，供论文讨论口径时引用。

用法（服务器，只用 CPU）:
    python tools/stream_energy.py --config configs/evisseg_stream_v1.yaml --split test \
        --eval-json log/stream_v1_seed37/eval_test_best_val_iou_seed37.json \
        --baseline-json log/baseline_k5_repolr_seed37/energy_test_best_iou_seed37.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from dataset.stream_windows import count_channels  # noqa: E402

MAC_PJ = 4.6
AC_PJ = 0.9


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="流式 SNN 能耗换算（稠密 / 事件驱动口径）与基线对比")
    parser.add_argument("--config", required=True, help="流式配置（提供数据路径与窗口参数）")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--eval-json", nargs="+", required=True, help="一个或多个 --mode eval 的结果文件")
    parser.add_argument("--baseline-json", nargs="*", default=[], help="tools/baseline_energy.py 的结果文件")
    parser.add_argument("--state-mode", choices=("carry", "reset_each_window"), default="carry")
    parser.add_argument("--max-sequences", type=int, default=0, help="计算非零输入占比时只用前几条序列（0 = 全部）")
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default=None, help="默认 log/energy/stream_energy_<split>.json")
    return parser.parse_args()


def energy_mj(mac, sop):
    """理论能耗（mJ）。"""
    return (float(mac) * MAC_PJ + float(sop) * AC_PJ) * 1e-9


def input_nonzero_fraction(sequences, cfg):
    """统计所有窗口 12 通道输入中非零元素的占比（画布含补零区域，与 estimate_operations 的稠密运算数一致）。

    sequences 为可迭代的 StreamSequence。返回 (占比, 窗口数)。
    """
    nonzero, total, windows = 0, 0, 0
    for seq in sequences:
        for k in range(seq.n_windows):
            idx = seq.window_index(k)
            counts = count_channels(seq.x[idx], seq.y[idx], seq.p[idx], seq.inner_bin[idx], cfg["time_bins"],
                                    cfg["pad_height"], cfg["pad_width"])
            nonzero += int(np.count_nonzero(counts))
            total += int(counts.size)
            windows += 1
    return nonzero / float(max(total, 1)), windows


def account(operations, nonzero_fraction, n_windows):
    """按两种口径换算一个 operations 字典（单窗），返回每窗与每 8 秒的 MAC / SOP / 能耗。

    event 口径只改 enc1：MAC = enc1 稠密运算数 × 非零输入占比。
    """
    layers = {row["layer"]: row for row in operations["per_layer"]}
    if "enc1" not in layers:
        raise ValueError("operations 中没有 enc1")
    enc1 = layers["enc1"]
    event_mac = float(operations["mac"]) - float(enc1["mac"]) + float(enc1["dense"]) * float(nonzero_fraction)
    out = {"nonzero_input_fraction": float(nonzero_fraction), "windows_per_sequence": int(n_windows)}
    for name, mac in (("dense", float(operations["mac"])), ("event", event_mac)):
        sop = float(operations["sop"])
        out[name] = {"mac_per_window": mac, "sop_per_window": sop,
                     "energy_mj_per_window": energy_mj(mac, sop),
                     "mac_per_8s": mac * n_windows, "sop_per_8s": sop * n_windows,
                     "energy_mj_per_8s": energy_mj(mac, sop) * n_windows,
                     "enc1_share": (float(enc1["dense"]) * (1.0 if name == "dense" else float(nonzero_fraction))
                                    * MAC_PJ * 1e-9) / max(energy_mj(mac, sop), 1e-30)}
    state = int(operations.get("state_elements", 0))
    out["membrane_leak_upper_bound_mj_per_8s"] = energy_mj(state * n_windows, 0.0)
    return out


def main():
    """入口：计算非零输入占比 -> 逐个换算 eval 结果 -> 读取基线结果 -> 打印对比表并写 JSON。"""
    args = parse_args()
    missing = [path for path in list(args.eval_json) + list(args.baseline_json) if not os.path.isfile(path)]
    if missing:
        raise SystemExit("找不到以下结果文件，请先生成或修正路径: %s" % missing)
    from dataset.ev_uav_stream import EvUAVStream
    from utils.stream_common import load_flat_config

    cfg = load_flat_config(args.config)
    if args.root:
        cfg["root"] = args.root
    dataset = EvUAVStream(cfg["root"], args.split, cfg)
    count = len(dataset) if not args.max_sequences else min(int(args.max_sequences), len(dataset))
    fraction, windows = input_nonzero_fraction((dataset[i] for i in range(count)), cfg)
    n_windows = int(cfg["n_windows"])
    print("非零输入占比 %.5f%%（%s，%d 条序列，%d 窗）" % (100 * fraction, args.split, count, windows), flush=True)

    rows = []
    for path in args.eval_json:
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        result = data[args.state_mode]
        if result.get("split") != args.split:
            raise ValueError("%s 是 %s 划分的结果，与 --split %s 不一致" % (path, result.get("split"), args.split))
        rows.append({"name": path, "kind": "stream_snn", "iou": result["iou"],
                     **account(result["operations"], fraction, n_windows)})
    baselines = []
    for path in args.baseline_json:
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        if data.get("split") != args.split:
            raise ValueError("%s 是 %s 划分的结果，与 --split %s 不一致" % (path, data.get("split"), args.split))
        baselines.append({"name": path, "kind": "baseline_sparse_ann", "iou": data["iou"],
                          "mac_per_8s": data["per_8s"]["mac"], "energy_mj_per_8s": data["per_8s"]["energy_mj"]})

    print("\n%-64s %8s %14s %14s %12s" % ("结果文件", "IoU", "稠密口径 mJ/8s", "事件驱动 mJ/8s", "衰减乘法上界"))
    for row in rows:
        print("%-64s %8.4f %14.2f %14.2f %12.2f" % (row["name"][-64:], row["iou"], row["dense"]["energy_mj_per_8s"],
                                                    row["event"]["energy_mj_per_8s"],
                                                    row["membrane_leak_upper_bound_mj_per_8s"]))
    for row in baselines:
        print("%-64s %8.4f %14s %14.2f %12s" % (row["name"][-64:], row["iou"], "-", row["energy_mj_per_8s"], "-"))
    print("（基线 IoU 来自 baseline_energy.py 调用的原 miou 函数；SNN IoU 来自 eval 结果，两者都是阈值 0.9 的正类 IoU）")

    out_path = args.out or os.path.join("log", "energy", "stream_energy_%s.json" % args.split)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump({"split": args.split, "state_mode": args.state_mode, "constants_pj": {"mac": MAC_PJ, "ac": AC_PJ},
                   "nonzero_input_fraction": fraction, "sequences_for_fraction": count,
                   "stream": rows, "baselines": baselines}, stream, indent=2, ensure_ascii=False)
    print("报告:", out_path)


if __name__ == "__main__":
    main()
