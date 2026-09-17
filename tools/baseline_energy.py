"""统计原论文 EV-SpSegNet（3D 稀疏卷积 ANN）处理每条 8 秒序列的理论运算量与能耗，用于和流式 SNN 同口径对比。

只使用原仓库的模型、数据集、体素化与评估代码（与 tools/dump_baseline_predictions.py 相同的推理流程）。
计数口径（常数与流式 SNN 相同：MAC 4.6 pJ）：
    稀疏卷积  实际连接数 × C_in × C_out。输入特征是实数，全部记 MAC。
             测量方法：把每个稀疏卷积复制一份，权重全置 1，在同一组坐标上输入全 1 特征；
             输出特征之和恰好等于 Σ(每个输出位置的有效输入邻居数) × C_in × C_out。
             连接关系由 spconv 自己计算，不在这里重写几何规则。
             1x1 子流形卷积的结果必须等于 体素数 × C_in × C_out，作为自检。
    线性层    行数 × in × out（SE 的全连接、逐体素分类层）
    注意力    nn.MultiheadAttention，序列长 L、token 数 N、维度 E：N·L·4E²（q/k/v/输出投影）+ N·L²·2E
             （本实现 unsqueeze(0) 后 L=1）
    SE 缩放   特征逐元素乘，N × C
    不计      BatchNorm（推理时并入卷积）、ReLU、残差加法、最大池化比较、体素化预处理。
             流式 SNN 的 estimate_operations 同样不计增益、膜电位更新与预处理。
输出 JSON：每条序列的事件数、体素数、分模块/分类型运算量；整个划分平均每条序列（8 秒）的 MAC 与能耗；
另用原评估函数计算 IoU/ACC，确认加载的是正确权重（K5 seed37 test 应为 0.8143）。

用法（服务器）:
    CUDA_VISIBLE_DEVICES=1 python tools/baseline_energy.py \
        --config configs/evisseg_evuav_baseline_v2_repolr.yaml \
        --checkpoint log/baseline_k5_repolr_seed37/best_iou_seed37.pt --split test
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAC_PJ = 4.6
AC_PJ = 0.9


def parse_args():
    """解析本脚本自己的参数。必须在导入原仓库模块之前完成，因为 configs/configs.py 在导入时会解析命令行。"""
    parser = argparse.ArgumentParser(description="原 EV-SpSegNet 的理论运算量与能耗")
    parser.add_argument("--config", required=True, help="原基线的 YAML 配置")
    parser.add_argument("--checkpoint", required=True, help="原基线的权重文件")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--max-sequences", type=int, default=0, help="只统计前几条序列（0 = 全部）")
    parser.add_argument("--out", default=None, help="默认 <checkpoint 目录>/energy_<split>_<checkpoint 名>.json")
    return parser.parse_args()


def attention_operations(length, tokens, dim):
    """nn.MultiheadAttention 一次前向的乘法数：投影 N·L·4E² + 注意力矩阵与加权求和 N·L²·2E。"""
    return int(tokens) * int(length) * 4 * int(dim) * int(dim) + int(tokens) * int(length) ** 2 * 2 * int(dim)


def linear_operations(rows, in_features, out_features):
    """线性层乘法数：行数 × in × out。"""
    return int(rows) * int(in_features) * int(out_features)


class OperationCounter(object):
    """给网络挂钩子，按模块累计一次前向的运算量。"""

    def __init__(self, net, spconv, se_type):
        import torch.nn as nn
        self.spconv = spconv
        self.records = {}
        self.checks = {"subm1x1_checked": 0, "subm1x1_mismatch": []}
        self.probes = {}
        self.handles = []
        for name, module in net.named_modules():
            if self.is_sparse_conv(module):
                probe = copy.deepcopy(module).eval()
                probe.weight.data.fill_(1.0)
                if getattr(probe, "bias", None) is not None:
                    probe.bias.data.zero_()
                self.probes[name] = probe
        for name, module in net.named_modules():
            if name in self.probes:
                self.handles.append(module.register_forward_hook(self.conv_hook(name)))
            elif isinstance(module, nn.MultiheadAttention):
                self.handles.append(module.register_forward_hook(self.attention_hook(name)))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self.linear_hook(name)))
            elif isinstance(module, se_type):
                self.handles.append(module.register_forward_hook(self.se_hook(name)))

    def is_sparse_conv(self, module):
        """spconv 的稀疏卷积（含子流形、下采样、逆卷积）；最大池化没有 weight，被排除。"""
        return (isinstance(module, self.spconv.SparseModule) and hasattr(module, "inverse")
                and hasattr(module, "kernel_size") and getattr(module, "weight", None) is not None)

    def add(self, name, kind, ops):
        """累计一个模块的运算量。"""
        entry = self.records.setdefault(name, {"kind": kind, "ops": 0, "calls": 0})
        entry["ops"] += int(ops)
        entry["calls"] += 1

    def conv_hook(self, name):
        """稀疏卷积：在同一坐标上运行全 1 权重的副本，输出之和即连接数 × C_in × C_out。"""
        import numpy as np
        import torch

        def hook(module, inputs, output):
            x = inputs[0]
            ones = torch.ones_like(x.features)
            if module.inverse:
                probe_input = x.replace_feature(ones)          # 逆卷积需要沿用配对下采样的规则表
            else:
                probe_input = self.spconv.SparseConvTensor(ones, x.indices, x.spatial_shape, x.batch_size)
            with torch.no_grad():
                probe_output = self.probes[name](probe_input)
            ops = int(round(float(probe_output.features.double().sum())))
            self.add(name, "sparse_conv", ops)
            if getattr(module, "subm", False) and int(np.prod(module.kernel_size)) == 1:
                expected = int(x.features.shape[0]) * int(x.features.shape[1]) * int(output.features.shape[1])
                self.checks["subm1x1_checked"] += 1
                if expected != ops:
                    self.checks["subm1x1_mismatch"].append([name, expected, ops])
        return hook

    def linear_hook(self, name):
        def hook(module, inputs, output):
            rows = int(inputs[0].numel()) // int(module.in_features)
            self.add(name, "linear", linear_operations(rows, module.in_features, module.out_features))
        return hook

    def attention_hook(self, name):
        def hook(module, inputs, output):
            length, tokens, dim = (int(v) for v in output[0].shape)       # (L, N, E)，batch_first=False
            self.add(name, "attention", attention_operations(length, tokens, dim))
        return hook

    def se_hook(self, name):
        def hook(module, inputs, output):
            features = inputs[0].features
            self.add(name, "se_scale", int(features.shape[0]) * int(features.shape[1]))
        return hook

    def pop(self):
        """取出本次前向的记录并清空。"""
        records, self.records = self.records, {}
        return records


def summarize_records(records):
    """把逐模块记录汇总为：总运算量、按顶层模块、按运算类型。"""
    by_group, by_kind, total = {}, {}, 0
    for name, entry in records.items():
        group = name.split(".")[0]
        by_group[group] = by_group.get(group, 0) + entry["ops"]
        by_kind[entry["kind"]] = by_kind.get(entry["kind"], 0) + entry["ops"]
        total += entry["ops"]
    return {"mac": total, "by_group": by_group, "by_kind": by_kind}


def main():
    """入口：加载原网络与权重 -> 逐序列推理并计数 -> 汇总能耗 -> 原评估函数核对 IoU/ACC。"""
    args = parse_args()
    sys.argv = [sys.argv[0], "--config", args.config]

    import numpy as np
    import spconv.pytorch as spconv
    import torch
    from configs.configs import cfg
    from dataset.ev_uav import EvUAV
    from model.basemodel import SEModule
    from model.evspsegnet import evspsegnet
    from utils.eval import evalute

    net = evspsegnet(cfg).eval().cuda()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cuda"))
    counter = OperationCounter(net, spconv, SEModule)
    dataset = EvUAV(cfg, mode=args.split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.custom_collate)
    evaluator = evalute(cfg)
    rows = []
    with torch.no_grad():
        for i, ev in enumerate(loader):
            if args.max_sequences and i >= args.max_sequences:
                break
            name = dataset.file_list[i]
            label = ev["seg_label"].float()
            counter.pop()
            preds, _ = net(ev["voxel_ev"])
            summary = summarize_records(counter.pop())
            preds = preds[ev["p2v_map"].long().cuda()].squeeze(-1).cpu()
            evaluator.matches[str(i)] = {"seg_pred": preds, "seg_gt": label.cuda()}
            row = {"sequence": name, "events": int(label.numel()),
                   "voxels": int(ev["voxel_ev"].features.shape[0]), **summary,
                   "energy_mj": summary["mac"] * MAC_PJ * 1e-9}
            rows.append(row)
            print("[%d/%d] %s | 事件 %d | 体素 %d | MAC %.3e | %.2f mJ" % (
                i + 1, len(dataset), name, row["events"], row["voxels"], row["mac"], row["energy_mj"]), flush=True)
    iou = evaluator.evaluate_semantic_segmantation_miou()
    acc = evaluator.evaluate_semantic_segmantation_accuracy()
    groups = sorted({g for r in rows for g in r["by_group"]})
    kinds = sorted({k for r in rows for k in r["by_kind"]})
    mean_mac = float(np.mean([r["mac"] for r in rows]))
    result = {
        "checkpoint": args.checkpoint, "config": args.config, "split": args.split, "sequences": len(rows),
        "constants_pj": {"mac": MAC_PJ, "ac": AC_PJ},
        "per_8s": {"mac": mean_mac, "sop": 0.0, "energy_mj": mean_mac * MAC_PJ * 1e-9,
                   "events": float(np.mean([r["events"] for r in rows])),
                   "voxels": float(np.mean([r["voxels"] for r in rows])),
                   "by_group": {g: float(np.mean([r["by_group"].get(g, 0) for r in rows])) for g in groups},
                   "by_kind": {k: float(np.mean([r["by_kind"].get(k, 0) for r in rows])) for k in kinds}},
        "iou": float(iou), "acc": float(acc), "checks": counter.checks, "rows": rows,
        "note": "稀疏卷积按实际连接数计 MAC（事件驱动口径）；不计 BN（可并入卷积）、ReLU、残差加法、池化比较与体素化",
    }
    out_path = args.out or os.path.join(os.path.dirname(args.checkpoint), "energy_%s_%s.json" % (
        args.split, os.path.splitext(os.path.basename(args.checkpoint))[0]))
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    p = result["per_8s"]
    print("平均每条序列（8 s）：事件 %.0f | 体素 %.0f | MAC %.3e | 能耗 %.2f mJ" % (
        p["events"], p["voxels"], p["mac"], p["energy_mj"]))
    print("按类型:", {k: "%.2e" % v for k, v in p["by_kind"].items()})
    print("按模块:", {g: "%.2e" % v for g, v in p["by_group"].items()})
    print("自检：1x1 子流形卷积 %d 次，不一致 %d 次" % (counter.checks["subm1x1_checked"],
                                               len(counter.checks["subm1x1_mismatch"])))
    print("原评估函数: iou %.4f, acc %.4f" % (iou, acc))
    print("报告:", out_path)
    print("BASELINE ENERGY FINISHED", flush=True)


if __name__ == "__main__":
    main()
