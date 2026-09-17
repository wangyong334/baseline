"""用真实权重和真实数据核对训练加速选项与原实现等价（在服务器 GPU 上运行）。

对照：原实现 = execution=step + input_device=cpu；加速 = execution=layer + input_device=gpu。
四项核对，结果写入 JSON：
    1. 输入（门槛）：numpy 逐窗构造与设备端片段构造，一条序列全部窗口逐位相同。
       与训练使用相同的 deterministic 设置，因此也验证了 GPU 上 index_put_(accumulate=True) 在该设置下可用
    2. 逻辑（门槛，float64）：从第 --offset 窗开始连续 2 个片段，比较 logits、各层 u_pre、脉冲、片段末膜电位、
       参数梯度；再用 train_sequence 训练同一条序列前 --train-windows 窗，比较一次更新后的参数。
       float64 下舍入可忽略，这一项检验的是实现逻辑。
    3. 数值（只报告，float32，关闭 TF32）：同上，看实际训练精度下的误差量级。
       batch 大小不同时 cuDNN 选的算法不同，float32 下不可能逐位相同，临界神经元可能翻转。
    4. 端到端（float32，默认设置）：若干条序列的 run_sequence；加 --full-split 时整个划分的 evaluate_split，
       门槛为 |ΔIoU| <= 1e-3，同时记录两种实现的耗时。
TF32：torch 1.9 默认允许 cuDNN 卷积使用 TF32，V1 就是在这个默认设置下训练的；第 3 项临时关闭，第 4 项保持默认。

用法（服务器）:
    CUDA_VISIBLE_DEVICES=1 python tools/check_stream_execution.py \
        --checkpoint log/stream_v1_seed37/best_val_iou_seed37.pt --split val --full-split
"""
import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_stream_v1 as T  # noqa: E402  先导入：它在导入 torch 前设置环境变量

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from dataset.stream_source import NumpyWindowSource, TorchWindowSource  # noqa: E402
from model.evspsegnet_stream import LAYER_NAMES  # noqa: E402
from utils.stream_metrics import iou_from_counts  # noqa: E402

FLOAT64_TOLERANCE = 1e-9          # logits / u_pre / 膜电位 / 参数梯度的相对误差
PARAM_TOLERANCE = 1e-9            # 一次 Adam 更新后参数的绝对误差
IOU_TOLERANCE = 1e-3
REFERENCE = {"execution": "step", "input_device": "cpu"}
FAST = {"execution": "layer", "input_device": "gpu"}


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="核对逐层执行 / 设备端输入与原实现等价")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sequence-index", type=int, default=0, help="第 1-3 项使用的序列（按文件名排序）")
    parser.add_argument("--offset", type=int, default=64, help="第 2、3 项从第几窗开始（之前的窗口无梯度预热状态）")
    parser.add_argument("--chunk", type=int, default=16, help="第 2、3 项的片段长度")
    parser.add_argument("--train-windows", type=int, default=48, help="train_sequence 核对使用的窗口数")
    parser.add_argument("--sequences", type=int, default=2, help="第 4 项逐序列对比的序列数")
    parser.add_argument("--full-split", action="store_true", help="第 4 项追加整个划分的 evaluate_split 对比")
    parser.add_argument("--root", default=None, help="覆盖 checkpoint 中记录的数据根目录")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--logic-device", default=None,
                        help="第 2 项 float64 核对用的设备，默认同 --device；GPU 上 float64 卷积不可用时改为 cpu（较慢）")
    parser.add_argument("--out", default=None, help="报告路径，默认 <checkpoint 目录>/execution_check_<split>.json")
    return parser.parse_args()


class TF32(object):
    """临时设置 TF32 开关（matmul 与 cuDNN），退出时恢复。"""

    def __init__(self, enabled):
        self.enabled = bool(enabled)
        self.saved = None

    def __enter__(self):
        self.saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = self.enabled
        torch.backends.cudnn.allow_tf32 = self.enabled
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.saved
        return False


def relative_error(a, b, floor=1.0):
    """max|a-b| / max(max|a|, floor)。logits、膜电位量级约为 1，用 floor=1；梯度量级很小，用很小的 floor。"""
    if a.numel() == 0:
        return 0.0
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max()) / max(float(a.abs().max()), float(floor))


def check_inputs(seq, cfg, q99, device, chunk):
    """第 1 项：两种输入构造逐窗逐位比较，另比较一个片段。"""
    reference = NumpyWindowSource(seq, cfg, q99, device)
    candidate = TorchWindowSource(seq, cfg, q99, device)
    report = {"windows": seq.n_windows, "bitwise_equal_windows": 0, "input_max_abs_error": 0.0,
              "event_fields_equal": True, "chunk_equal": True, "nonzero_input_fraction": 0.0}
    nonzero, total = 0, 0
    for k in range(seq.n_windows):
        a, b = reference.window(k), candidate.window(k)
        if torch.equal(a[0], b[0]):
            report["bitwise_equal_windows"] += 1
        report["input_max_abs_error"] = max(report["input_max_abs_error"], float((a[0] - b[0]).abs().max()))
        same = set(a[1]) == set(b[1]) and all(torch.equal(a[1][key], b[1][key]) for key in a[1])
        same = same and torch.equal(a[2], b[2]) and np.array_equal(a[3], b[3])
        report["event_fields_equal"] = report["event_fields_equal"] and bool(same)
        nonzero += int((a[0] != 0).sum())
        total += int(a[0].numel())
    end = min(chunk, seq.n_windows)
    a, b = reference.chunk(0, end), candidate.chunk(0, end)
    report["chunk_equal"] = bool(torch.equal(a[0], b[0]) and all(torch.equal(a[1][key], b[1][key]) for key in a[1])
                                 and torch.equal(a[2], b[2]) and np.array_equal(a[3], b[3]))
    report["nonzero_input_fraction"] = nonzero / float(max(total, 1))
    report["passed"] = (report["bitwise_equal_windows"] == seq.n_windows and report["event_fields_equal"]
                        and report["chunk_equal"])
    return report


def chunk_loss(logits, labels, pos_weight):
    """与 train_sequence 相同的片段损失（求和再除以事件数，事件数为 0 时返回 None）。"""
    if labels.numel() == 0:
        return None
    weight = torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight, reduction="sum") / labels.numel()


def compare_chunks(net, seq, cfg, q99, device, dtype, offset, chunk, pos_weight):
    """第 2、3 项：预热到 offset 窗后，连续 2 个片段分别用逐窗 forward 与 forward_chunk 计算并反向，逐项比较。"""
    source = TorchWindowSource(seq, cfg, q99, device, dtype)       # 输入已由第 1 项核对，这里两边共用
    net_a = copy.deepcopy(net).to(dtype).train()
    net_b = copy.deepcopy(net).to(dtype).train()
    offset = max(0, min(int(offset), seq.n_windows - 2 * int(chunk)))
    states = None
    with torch.no_grad():
        for k in range(offset):
            x, events, _, _ = source.window(k)
            _, states, _ = net_a(x, events, states)
    states_a = states_b = states
    report = {"dtype": str(dtype).replace("torch.", ""), "windows": [offset, offset + 2 * int(chunk)],
              "logit_rel_error": 0.0, "u_pre_rel_error": dict.fromkeys(LAYER_NAMES, 0.0),
              "state_rel_error": 0.0, "spike_flips": dict.fromkeys(LAYER_NAMES, 0),
              "spikes": dict.fromkeys(LAYER_NAMES, 0), "events": 0}
    for start in (offset, offset + int(chunk)):
        end = start + int(chunk)
        logits_a, labels_a, kept = [], [], {"spikes": [[] for _ in LAYER_NAMES], "u_pre": [[] for _ in LAYER_NAMES]}
        loss_a = None
        for k in range(start, end):
            x, events, labels, _ = source.window(k)
            logits, states_a, info = net_a(x, events, states_a, collect=True)
            logits_a.append(logits)
            labels_a.append(labels)
            for layer in range(len(LAYER_NAMES)):
                kept["spikes"][layer].append((info["spikes"][layer] > 0).detach())
                kept["u_pre"][layer].append(info["u_pre"][layer].detach())
        logits_a, labels_a = torch.cat(logits_a), torch.cat(labels_a)
        loss_a = chunk_loss(logits_a, labels_a, pos_weight)
        if loss_a is not None:
            loss_a.backward()

        x, events, labels_b, _ = source.chunk(start, end)
        logits_b, states_b, info_b = net_b.forward_chunk(x, events, states_b, collect=True)
        loss_b = chunk_loss(logits_b, labels_b, pos_weight)
        if loss_b is not None:
            loss_b.backward()

        report["events"] += int(labels_b.numel())
        report["logit_rel_error"] = max(report["logit_rel_error"], relative_error(logits_a, logits_b))
        for layer, name in enumerate(LAYER_NAMES):
            spikes_a = torch.stack(kept["spikes"][layer])
            spikes_b = info_b["spikes"][layer] > 0
            report["spike_flips"][name] += int((spikes_a != spikes_b).sum())
            report["spikes"][name] += int(spikes_a.sum())
            report["u_pre_rel_error"][name] = max(report["u_pre_rel_error"][name],
                                                  relative_error(torch.stack(kept["u_pre"][layer]),
                                                                 info_b["u_pre"][layer]))
        for a, b in zip(states_a, states_b):
            if a is not None:
                report["state_rel_error"] = max(report["state_rel_error"], relative_error(a, b))
        del kept, info_b
        states_a = [None if s is None else s.detach() for s in states_a]
        states_b = [None if s is None else s.detach() for s in states_b]
    worst, worst_name = 0.0, None
    for (name, pa), (_, pb) in zip(net_a.named_parameters(), net_b.named_parameters()):
        if (pa.grad is None) != (pb.grad is None):
            worst, worst_name = float("inf"), name
            break
        if pa.grad is not None:
            error = relative_error(pa.grad, pb.grad, floor=1e-12)
            if error > worst:
                worst, worst_name = error, name
    report["grad_rel_error"], report["grad_worst_parameter"] = worst, worst_name
    report["u_pre_rel_error_max"] = max(report["u_pre_rel_error"].values())
    report["spike_flips_total"] = sum(report["spike_flips"].values())
    return report


def compare_train_sequence(net, seq, cfg, q99, device, dtype, windows, pos_weight):
    """第 2 项的一部分：同一初始权重、同一随机数，分别用两种实现训练 windows 个窗口，比较更新后的参数。"""
    report = {"dtype": str(dtype).replace("torch.", ""), "windows": int(windows)}
    nets, outputs = [], []
    for option in (REFERENCE, FAST):
        run_cfg = dict(cfg, **option)
        copy_net = copy.deepcopy(net).to(dtype).train()
        optimizer = torch.optim.Adam(copy_net.parameters(), lr=float(cfg["lr"]))
        outputs.append(T.train_sequence(copy_net, seq, optimizer, run_cfg, q99, pos_weight, device,
                                        np.random.RandomState(int(cfg["seed"])), max_windows=int(windows)))
        nets.append(copy_net)
    worst, worst_name = 0.0, None
    for (name, pa), (_, pb) in zip(nets[0].named_parameters(), nets[1].named_parameters()):
        error = float((pa.detach() - pb.detach()).abs().max())
        if error > worst:
            worst, worst_name = error, name
    report.update({"param_max_abs_error": worst, "param_worst": worst_name,
                   "loss_sum": [outputs[0]["loss_sum"], outputs[1]["loss_sum"]],
                   "steps": [outputs[0]["steps"], outputs[1]["steps"]],
                   "missing_grads": [outputs[0]["missing_grads"], outputs[1]["missing_grads"]]})
    return report


def compare_sequences(net, dataset, cfg, q99, device, count, mode):
    """第 4 项：若干条完整序列的 run_sequence（无梯度），比较概率、IoU 与发放率，并计时。"""
    rows = []
    for i in range(min(int(count), len(dataset))):
        seq = dataset[i]
        row = {"sequence": seq.name}
        results = []
        for label, option in (("reference", REFERENCE), ("fast", FAST)):
            monitor = T.LayerMonitor(net.v_threshold)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                probs, confusion = T.run_sequence(net, seq, dict(cfg, **option), q99, device, mode, monitor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row[label + "_seconds"] = time.perf_counter() - t0
            tp, fp, fn = (int(confusion[:, j].sum()) for j in range(3))
            row[label + "_iou"] = iou_from_counts(tp, fp, fn)
            row[label + "_firing_rate"] = {name: s["firing_rate"] for name, s in monitor.summary().items()}
            results.append((probs, confusion))
        row["prob_max_abs_error"] = float(np.abs(results[0][0] - results[1][0]).max()) if seq.n_events else 0.0
        row["threshold_decisions_changed"] = int(((results[0][0] >= float(cfg["threshold"]))
                                                  != (results[1][0] >= float(cfg["threshold"]))).sum())
        rows.append(row)
    return rows


def main():
    """入口：加载权重与数据 -> 四项核对 -> 写报告并打印结论。"""
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = dict(ckpt["config"])
    if args.root:
        cfg["root"] = args.root
    cfg.setdefault("threshold", 0.9)
    T.seed_everything(int(cfg["seed"]), bool(cfg["deterministic"]))
    q99 = np.asarray(ckpt["stats"]["q99"], dtype=np.float32)
    pos_weight = min(float(ckpt["stats"]["train_neg"]) / float(ckpt["stats"]["train_pos"]), float(cfg["pos_weight_cap"]))
    net = T.build_net(dict(cfg, **REFERENCE), device)
    net.load_state_dict(ckpt["model"])
    if not bool(net.gain_calibrated):
        raise SystemExit("checkpoint 中的增益未校准")
    dataset = EvUAVStream(cfg["root"], args.split, cfg)
    seq = dataset[args.sequence_index]
    out_path = args.out or os.path.join(os.path.dirname(args.checkpoint), "execution_check_%s.json" % args.split)
    report = {"checkpoint": args.checkpoint, "split": args.split, "sequence": seq.name, "torch": torch.__version__,
              "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "deterministic": bool(cfg["deterministic"]), "state_mode": cfg["state_mode"],
              "default_tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                               "cudnn": torch.backends.cudnn.allow_tf32},
              "tolerances": {"float64_relative": FLOAT64_TOLERANCE, "param_abs": PARAM_TOLERANCE,
                             "iou_abs": IOU_TOLERANCE}}
    print("序列 %s | 事件 %d | 设备 %s | torch %s | deterministic=%s（与训练相同）" % (
        seq.name, seq.n_events, report["device"], torch.__version__, cfg["deterministic"]), flush=True)

    report["1_inputs"] = check_inputs(seq, cfg, q99, device, args.chunk)
    r = report["1_inputs"]
    print("[1 输入] 逐位相同 %d/%d 窗 | 最大误差 %.2e | 事件字段一致 %s | 片段一致 %s | 非零输入占比 %.4f%% -> %s" % (
        r["bitwise_equal_windows"], r["windows"], r["input_max_abs_error"], r["event_fields_equal"],
        r["chunk_equal"], 100 * r["nonzero_input_fraction"], "通过" if r["passed"] else "未通过"), flush=True)

    logic_device = torch.device(args.logic_device) if args.logic_device else device
    with TF32(False):
        logic_net = copy.deepcopy(net).to(logic_device)
        logic = compare_chunks(logic_net, seq, cfg, q99, logic_device, torch.float64, args.offset, args.chunk,
                               pos_weight)
        logic_train = compare_train_sequence(logic_net, seq, cfg, q99, logic_device, torch.float64,
                                             args.train_windows, pos_weight)
        del logic_net
        numeric = compare_chunks(net, seq, cfg, q99, device, torch.float32, args.offset, args.chunk, pos_weight)
    logic["device"] = str(logic_device)
    logic["passed"] = (logic["logit_rel_error"] <= FLOAT64_TOLERANCE and logic["u_pre_rel_error_max"] <= FLOAT64_TOLERANCE
                       and logic["state_rel_error"] <= FLOAT64_TOLERANCE and logic["grad_rel_error"] <= FLOAT64_TOLERANCE
                       and logic["spike_flips_total"] == 0 and logic["events"] > 0
                       and min(logic["spikes"].values()) > 0)
    logic_train["passed"] = (logic_train["param_max_abs_error"] <= PARAM_TOLERANCE
                             and logic_train["steps"][0] == logic_train["steps"][1]
                             and logic_train["missing_grads"][0] == logic_train["missing_grads"][1])
    report["2_logic_float64"], report["2_logic_train_sequence"], report["3_numeric_float32_no_tf32"] = (
        logic, logic_train, numeric)
    for title, r in (("[2 逻辑 float64]", logic), ("[3 数值 float32]", numeric)):
        print("%s 窗 %d-%d，事件 %d | logits %.2e | u_pre %.2e | 膜电位 %.2e | 梯度 %.2e（%s）| 脉冲翻转 %d / 发放 %d%s" % (
            title, r["windows"][0], r["windows"][1], r["events"], r["logit_rel_error"], r["u_pre_rel_error_max"],
            r["state_rel_error"], r["grad_rel_error"], r["grad_worst_parameter"], r["spike_flips_total"],
            sum(r["spikes"].values()), (" -> " + ("通过" if r["passed"] else "未通过")) if "passed" in r else "（只报告）"),
            flush=True)
    print("[2 逻辑 float64] train_sequence 前 %d 窗：更新后参数最大误差 %.2e（%s）| 损失 %.6f / %.6f -> %s" % (
        logic_train["windows"], logic_train["param_max_abs_error"], logic_train["param_worst"],
        logic_train["loss_sum"][0], logic_train["loss_sum"][1], "通过" if logic_train["passed"] else "未通过"), flush=True)
    del logic, numeric

    net.eval()
    rows = compare_sequences(net, dataset, cfg, q99, device, args.sequences, cfg["state_mode"])
    report["4_sequences"] = rows
    for row in rows:
        print("[4 整序列] %s | IoU %.4f / %.4f | 概率最大差 %.2e | 阈值判定变化 %d 个事件 | 耗时 %.1fs / %.1fs" % (
            row["sequence"], row["reference_iou"], row["fast_iou"], row["prob_max_abs_error"],
            row["threshold_decisions_changed"], row["reference_seconds"], row["fast_seconds"]), flush=True)
    if args.full_split:
        split_rows = {}
        for label, option in (("reference", REFERENCE), ("fast", FAST)):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            result = T.evaluate_split(net, dict(cfg, **option), q99, device, args.split, cfg["state_mode"],
                                      collect_layers=True)
            split_rows[label] = {"iou": result["iou"], "acc": result["acc"], "seconds": time.perf_counter() - t0,
                                 "firing_rate": {n: s["firing_rate"] for n, s in result["layers"].items()}}
        split_rows["passed"] = abs(split_rows["reference"]["iou"] - split_rows["fast"]["iou"]) <= IOU_TOLERANCE
        report["4_full_split"] = split_rows
        print("[4 整个划分] IoU %.4f / %.4f | ACC %.4f / %.4f | 耗时 %.0fs / %.0fs -> %s" % (
            split_rows["reference"]["iou"], split_rows["fast"]["iou"], split_rows["reference"]["acc"],
            split_rows["fast"]["acc"], split_rows["reference"]["seconds"], split_rows["fast"]["seconds"],
            "通过" if split_rows["passed"] else "未通过"), flush=True)

    gates = [report["1_inputs"]["passed"], report["2_logic_float64"]["passed"],
             report["2_logic_train_sequence"]["passed"]]
    if args.full_split:
        gates.append(report["4_full_split"]["passed"])
    report["passed"] = all(gates)
    T.write_json(out_path, report)
    print("报告:", out_path)
    print("EXECUTION CHECK %s" % ("PASSED" if report["passed"] else "FAILED"), flush=True)


if __name__ == "__main__":
    main()
