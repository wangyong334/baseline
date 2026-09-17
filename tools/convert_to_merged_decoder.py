"""把已训练的原版 V1 checkpoint 无损换算为电流合并解码器（merged_decoder=True），不需要重新训练。

原版解码阶段：I = Conv3x3( cat[ ConvT2x2(S_deep), S_skip ] )。ConvT 的输出是实数，
所以解码卷积有一半输入是实数，只能按 MAC（实数乘加）计。
ConvT 与解码卷积之间没有非线性，两次线性运算可以精确合成一次：
    I = ConvT4x4_s2_p1(S_deep) + Conv3x3(S_skip)
合并后所有突触运算的输入都是 0/1 脉冲，网络函数不变（推导见 merge_decoder_state_dict）。

流程：读取 checkpoint -> float64 换算权重 -> 在真实数据窗口上逐窗做两级核对 -> 通过才保存。
      数学核对：原权重转 float64 后换算，合并核不经 float32 舍入，必须零脉冲翻转、误差不超过 1e-9；
      部署核对：实际保存的 float32 合并核在 float32 下对比，只报告（舍入可能让极少数临界神经元翻转，
               最终以评估 IoU 与原版是否一致为准）。

用法（服务器；默认用 CPU，不占用训练中的 GPU）：
    python tools/convert_to_merged_decoder.py --checkpoint log/stream_v1_seed37/best_val_iou_seed37.pt
默认输出 log/stream_v1_seed37/best_val_iou_seed37_merged.pt 与同名 _conversion.json，之后照常评估：
    python train_stream_v1.py --mode eval --checkpoint log/stream_v1_seed37/best_val_iou_seed37_merged.pt --split test
"""
import argparse
import copy
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_stream_v1 import build_net, write_json  # noqa: E402  先导入：它在导入 torch 前设置环境变量

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataset.ev_uav_stream import EvUAVStream, window_to_device  # noqa: E402
from model.evspsegnet_stream import count_parameters, merge_decoder_state_dict  # noqa: E402

FLOAT64_TOLERANCE = 1e-9


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="原版 V1 -> 电流合并解码器的无损权重换算")
    parser.add_argument("--checkpoint", required=True, help="原版 V1 的权重文件（train 模式保存的 .pt）")
    parser.add_argument("--out", default=None, help="输出路径，默认 <原名>_merged.pt")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="用哪个划分的真实窗口核对")
    parser.add_argument("--sequences", type=int, default=2, help="核对用的序列数；0 表示只用合成输入")
    parser.add_argument("--windows", type=int, default=160, help="每条序列核对的窗口数（带状态连续前向）")
    parser.add_argument("--root", default=None, help="覆盖 checkpoint 中记录的数据根目录（只用于核对）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    return parser.parse_args()


def convert_checkpoint(ckpt, source=None):
    """返回换算后的 checkpoint 字典，不修改输入。

    保留 epoch / best_val_iou / stats 等元信息；配置中写入 merged_decoder=True，eval 据此重建结构。
    优化器状态对应旧参数形状，不保留，因此换算后的权重只用于评估，不能 --resume 续训。
    """
    cfg = dict(ckpt["config"])
    if cfg.get("spiking_decoder", False):
        raise ValueError("这是已删除的 10 层脉冲解码器权重，ConvT 与解码卷积之间有 LIF，不能换算")
    if cfg.get("merged_decoder", False):
        raise ValueError("该 checkpoint 已经是电流合并解码器")
    new_cfg = dict(cfg)
    new_cfg["merged_decoder"] = True
    converted = {key: value for key, value in ckpt.items() if key not in ("model", "optimizer", "config")}
    converted.update({
        "model": merge_decoder_state_dict(ckpt["model"]),
        "config": new_cfg,
        "conversion": {"source": source,
                       "method": "Conv3x3(cat[ConvT2x2(S_deep), S_skip]) -> ConvT4x4_s2_p1(S_deep) + Conv3x3(S_skip)",
                       "optimizer": "未保留（参数形状已改变），不能用于 --resume"},
    })
    return converted


def real_sequences(cfg, stats, split, n_sequences, n_windows, device):
    """返回无参函数；每次调用重新产出真实数据的若干序列，每个序列是按时间顺序的 (x, events) 生成器。"""
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    dataset = EvUAVStream(cfg["root"], split, cfg)
    count = min(int(n_sequences), len(dataset))

    def windows_of(seq):
        """按时间顺序产出一个序列的窗口输入。"""
        for k in range(min(int(n_windows), seq.n_windows)):
            x, events, _, _ = window_to_device(seq, k, q99, cfg, device)
            yield x, events

    def factory():
        for i in range(count):
            yield windows_of(dataset[i])
    return factory


def synthetic_sequences(channels_in, height, width, n_sequences, n_windows, device,
                        density=0.05, events=600, seed=0):
    """返回无参函数，产出固定种子的合成窗口（数据目录不可用时的兜底核对，也供单元测试使用）。"""
    def windows_of(generator):
        """产出一个合成序列的窗口：约 density 比例的像素有计数，每窗 events 个事件。"""
        for _ in range(int(n_windows)):
            mask = (torch.rand(1, 1, height, width, generator=generator) < density).float()
            x = mask * torch.rand(1, channels_in, height, width, generator=generator) * 3.0
            ev = {"b": torch.zeros(events, dtype=torch.long),
                  "y": torch.randint(0, height, (events,), generator=generator),
                  "x": torch.randint(0, width, (events,), generator=generator),
                  "p": torch.randint(0, 2, (events,), generator=generator).float(),
                  "t_local": torch.rand(events, generator=generator)}
            yield x.to(device), {key: value.to(device) for key, value in ev.items()}

    def factory():
        generator = torch.Generator().manual_seed(int(seed))
        for _ in range(int(n_sequences)):
            yield windows_of(generator)
    return factory


def compare_networks(reference, candidate, sequences, dtype):
    """在同一批窗口上逐窗对比两个网络：每条序列开头清空状态，窗口之间传递状态。

    返回最大 logit / 膜电位 / 增益前电流误差、脉冲翻转数（按"是否发放"比较）、
    参与比较的神经元总数与其中发放的数量（用来确认核对不是在全网沉默时做的）。
    """
    reference = copy.deepcopy(reference).to(dtype).eval()
    candidate = copy.deepcopy(candidate).to(dtype).eval()
    report = {"dtype": str(dtype).replace("torch.", ""), "sequences": 0, "windows": 0,
              "logit_max_error": 0.0, "state_max_error": 0.0, "current_max_error": 0.0,
              "spike_flips": 0, "compared_neurons": 0, "active_neurons": 0}
    with torch.no_grad():
        for windows in sequences:
            report["sequences"] += 1
            state_a = state_b = None
            for x, events in windows:
                x = x.to(dtype)
                events = {key: (value.to(dtype) if value.is_floating_point() else value)
                          for key, value in events.items()}
                logits_a, state_a, info_a = reference(x, events, state_a, collect=True)
                logits_b, state_b, info_b = candidate(x, events, state_b, collect=True)
                report["windows"] += 1
                if logits_a.numel():
                    report["logit_max_error"] = max(report["logit_max_error"],
                                                    float((logits_a - logits_b).abs().max()))
                for a, b in zip(state_a, state_b):
                    if a is not None:
                        report["state_max_error"] = max(report["state_max_error"], float((a - b).abs().max()))
                for a, b in zip(info_a["current"], info_b["current"]):
                    report["current_max_error"] = max(report["current_max_error"], float((a - b).abs().max()))
                for a, b in zip(info_a["spikes"], info_b["spikes"]):
                    report["spike_flips"] += int(((a > 0) != (b > 0)).sum())
                    report["compared_neurons"] += int(a.numel())
                    report["active_neurons"] += int((a > 0).sum())
    return report


def verify_conversion(reference, converted_cfg, original_state, stored_state, sequences, device):
    """两级核对，返回 (reports, passed)。

    数学核对：把原权重转成 float64 再换算，合并核不经过 float32 舍入；与原网络在 float64 下对比。
             必须零脉冲翻转且误差不超过 FLOAT64_TOLERANCE——证明换算公式对这份权重严格成立。
             （若直接用保存的 float32 合并核做 float64 对比，比较的其实是 float32 舍入误差，
              在上亿个神经元里总会碰到离阈值极近的，会误判为不等价。）
    部署核对：用实际保存的 float32 合并核在 float32 下对比，只报告不作门槛。
    sequences 为无参函数，每次调用重新产出同一批窗口。
    """
    exact_state = merge_decoder_state_dict(OrderedDict(
        (key, value.double() if value.is_floating_point() else value) for key, value in original_state.items()))
    exact = build_net(converted_cfg, device).double()
    exact.load_state_dict(exact_state)
    deployed = build_net(converted_cfg, device)
    deployed.load_state_dict(stored_state)
    math_report = compare_networks(reference, exact, sequences(), torch.float64)
    math_report["check"] = "数学核对（float64 合并核）"
    deploy_report = compare_networks(reference, deployed, sequences(), torch.float32)
    deploy_report["check"] = "部署核对（保存的 float32 合并核）"
    passed = (math_report["windows"] > 0 and math_report["spike_flips"] == 0
              and math_report["logit_max_error"] <= FLOAT64_TOLERANCE
              and math_report["state_max_error"] <= FLOAT64_TOLERANCE)
    return [math_report, deploy_report], passed


def main():
    """入口：换算 -> 核对 -> 通过才保存。"""
    args = parse_args()
    out_path = args.out or os.path.splitext(args.checkpoint)[0] + "_merged.pt"
    report_path = os.path.splitext(out_path)[0] + "_conversion.json"
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit("输出已存在: %s（确认要覆盖请加 --overwrite）" % out_path)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    converted = convert_checkpoint(ckpt, source=args.checkpoint)
    cfg = dict(ckpt["config"])
    if args.root:
        cfg["root"] = args.root
    reference = build_net(cfg, device)
    reference.load_state_dict(ckpt["model"])
    candidate = build_net(converted["config"], device)
    candidate.load_state_dict(converted["model"])
    if not bool(candidate.gain_calibrated):
        raise SystemExit("checkpoint 中的增益未校准，这不是训练完成的权重")

    split_dir = os.path.join(cfg["root"], args.split)
    if args.sequences > 0 and os.path.isdir(split_dir):
        sequences = real_sequences(cfg, ckpt["stats"], args.split, args.sequences, args.windows, device)
        data_note = "真实数据 %s：前 %d 条序列，每条前 %d 窗" % (args.split, args.sequences, args.windows)
    else:
        if args.sequences > 0:
            print("找不到数据目录 %s，改用合成输入核对（可用 --root 指定数据根目录）" % split_dir, flush=True)
        sequences = synthetic_sequences(2 + 2 * int(cfg["time_bins"]), int(cfg["pad_height"]),
                                        int(cfg["pad_width"]), 2, 40, device)
        data_note = "合成输入：2 条序列 × 40 窗"
    print("核对数据:", data_note, flush=True)

    reports, passed = verify_conversion(reference, converted["config"], ckpt["model"], converted["model"],
                                        sequences, device)
    for r in reports:
        print("  %s | %d 窗 | logit 最大误差 %.2e | 膜电位 %.2e | 电流 %.2e | 脉冲翻转 %d / %d（发放 %.2f%%）" % (
            r["check"], r["windows"], r["logit_max_error"], r["state_max_error"], r["current_max_error"],
            r["spike_flips"], r["compared_neurons"], 100.0 * r["active_neurons"] / max(r["compared_neurons"], 1)),
            flush=True)

    summary = {"source": args.checkpoint, "output": out_path, "verification_data": data_note,
               "float64_tolerance": FLOAT64_TOLERANCE, "passed": passed, "reports": reports,
               "parameters": {"original": count_parameters(reference)["total"],
                              "merged_stored": count_parameters(candidate)["total"],
                              "note": "合并核由原参数算出，自由度不变；存储数量因 4x4 核而增加"},
               "epoch": ckpt.get("epoch"), "best_val_iou": ckpt.get("best_val_iou")}
    if not passed:
        write_json(report_path, summary)
        raise SystemExit("float64 核对未通过，未保存权重。报告: %s" % report_path)
    if reports[1]["spike_flips"]:
        print("  注意：float32 下有 %d 个神经元因舍入翻转；评估后请核对 IoU 是否与原版一致"
              % reports[1]["spike_flips"], flush=True)
    torch.save(converted, out_path)
    write_json(report_path, summary)
    print("参数量: 原版 %s -> 合并形式存储 %s" % (
        "{:,}".format(summary["parameters"]["original"]), "{:,}".format(summary["parameters"]["merged_stored"])))
    print("CONVERSION PASSED:", out_path, flush=True)


if __name__ == "__main__":
    main()
