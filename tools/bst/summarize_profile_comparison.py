"""Summarize paired raw-video inference measurements and strict output checks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def representative(case):
    return min(case["runs"], key=lambda r: abs(r["seconds"] - case["median_seconds"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile")
    ap.add_argument("--out", required=True)
    ap.add_argument("--projection-profile")
    args = ap.parse_args()
    source = Path(args.profile)
    p = json.loads(source.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for c in p["cases"]:
        grouped.setdefault((c["recording"], c["mode"]), {})[c["implementation"]] = c
    if len(grouped) != 6 or any(set(g) != {"reference", "optimized"} for g in grouped.values()):
        raise ValueError("expected all three recordings, both policies, both implementations")
    rows = []
    logical = ["events", "births", "queries_updated", "decision_checks", "pair_visits", "max_active",
               "source_encoded", "deadline_commits", "policy_commits", "birth_commits", "remaining_active"]
    for (name, mode), pair in grouped.items():
        old, new = pair["reference"], pair["optimized"]
        a, b = representative(old), representative(new)
        for key in logical:
            assert a["stats"][key] == b["stats"][key], (name, mode, key)
            assert all(run["stats"][key] == a["stats"][key] for c in pair.values() for run in c["runs"])
        check = new["correctness"]["comparison"]
        for key in ["prior", "prediction", "lags", "reasons", "trace_ids", "trace_sizes", "trace_committed", "trace_updated"]:
            assert check[key]["exact"], (name, mode, key)
        rows.append({"recording": name, "mode": mode, "events": old["events"],
                     "seconds_before": old["median_seconds"], "seconds_after": new["median_seconds"],
                     "speed_ratio": old["median_seconds"] / new["median_seconds"],
                     "time_reduction_percent": 100*(1-new["median_seconds"]/old["median_seconds"]),
                     "p95_ms_before": a["window_summary"]["p95_ms"], "p95_ms_after": b["window_summary"]["p95_ms"],
                     "max_ms_after": b["window_summary"]["max_ms"],
                     "over_50ms_fraction_after": b["window_summary"]["over_window_fraction"],
                     "peak_allocated_MiB_before": a["peak_allocated_bytes"]/2**20,
                     "peak_allocated_MiB_after": b["peak_allocated_bytes"]/2**20,
                     "comparisons": check, "work_before": a["stats"], "work_after": b["stats"],
                     "mac_before": old["linear_conv_mac_subtotal"], "mac_after": new["linear_conv_mac_subtotal"],
                     "stages_after": new["stage_seconds_separate_instrumented_pass"]})
    result = {"source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "hardware": p["hardware"], "rows": rows,
              "max_logit_abs_error": max(r["comparisons"]["logits"]["max_abs_difference"] for r in rows),
              "max_trace_logit_abs_error": max(r["comparisons"]["trace_logits"]["max_abs_difference"] for r in rows),
              "max_trace_feature_abs_error": max(r["comparisons"]["trace_features"]["max_abs_difference"] for r in rows)}
    (out/"comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# 推理流程优化与整体模型迭代：验收报告", "",
             "## 改动及主线", "",
             "1. 邻域检索：不同年龄的待定事件共用当前到达窗口的批量 KNN；返回紧凑的新增槽位。过去选中的邻居身份和顺序不变。",
             "2. 因果缓存：每个源事件只编码一次，只扫描新增邻居；固定源编码与相对位置投影保存在可复用的查询槽位中，避免每窗复制整个历史邻居队列。提交后释放查询槽位，源观测仍可供其他事件使用。",
             "3. SDU：每次新证据到来，仍用当前状态对完整已到达证据重新赋权、重新估值。缓存不含随后验变化的权重或证据值；先验加当前证据修正的公式不变，不重复累加旧修正。",
             "4. VoW：合并每窗的小批次结果及有效性检查，减少 CPU/GPU 同步；停止条件、成本和截止时间不变。",
             "5. 推理执行：查询批次独立于训练，默认 1024，可用 --batch-size 调整；源事件编码批次为 4096。这些是执行分块，不是事件采样或证据预算。",
             "6. 整体模型迭代：将 SDU 的源特征、相对位置、查询特征的同输入并行线性投影合并为 GEMM。只在 eval 且无梯度时启用；训练保持原路径，参数名与检查点兼容，权重版本/设备/精度改变时刷新缓存。BIR 递归神经元和先验输出保持原样。", "",
             "本轮是保持模型决策含义的实现优化，没有新增训练、测试集调参或可声称为论文新贡献的机制。CLAUDE.md 未更新。", "",
             "## 同录像交替计时", "",
             "硬件：%s；PyTorch %s；float32；同一进程、同一组权重，预热后交替改变旧/新运行次序，各测 %d 次，报告中位数。旧实现查询批次为保存的训练批次 256，新实现为 1024。" % (p["hardware"], p["torch"], p["args"]["repeats"]), "",
             "计时包括事件表示、BIR、索引/缓存、SDU/VoW 和输出回 CPU；不含磁盘加载、权重初始化、指标计算及写盘。同步逐窗测量；正确性抓取、MAC 计数、分阶段计时在额外的独立前向进行。", "",
             "| 录像事件数 | 停止方式 | 优化前 s | 优化后 s | 加速比 | 耗时下降 | 窗 p95：前→后 ms |", "|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| %s | %s | %.3f | %.3f | %.2f× | %.1f%% | %.1f → %.1f |" %
                     (format(r["events"], ","), r["mode"], r["seconds_before"], r["seconds_after"],
                      r["speed_ratio"], r["time_reduction_percent"], r["p95_ms_before"], r["p95_ms_after"]))
    lines += ["", "每段录像均为 8 秒。整段耗时 <8 秒表示平均吞吐跟得上，不代表逐窗硬实时；max 和超 50 ms 比例见 comparison.json。", "",
              "## 输出与中间状态一致性", "",
              "- 三段录像共 651,810 个事件，固定/学习型停止各跑一遍，共比较 1,303,620 个事件输出。",
              "- 先验分数、官方阈值分类、逐事件等待窗数、提交原因、每窗活跃事件身份、更新/提交布尔掩码：完全一致。",
              "- 最终后验分数最大绝对差：%.9g；所有中间后验最大差：%.9g；VoW 输入特征最大差：%.9g。未宣称浮点逐位相同。" %
              (result["max_logit_abs_error"], result["max_trace_logit_abs_error"], result["max_trace_feature_abs_error"]),
              "- 出生数、SDU 更新数、旧证据访问数、源编码数和提交数一致，速度收益不是通过少处理事件、缩短等待或删旧证据得到的。",
              "- 单元测试另外覆盖真实提前停止/缓存槽位复用、空窗、距离并列、自身排除、未来扰动、重复证据无漂移、所有 reweight/revalue 消融及投影缓存刷新。", "",
              "## 计算与内存的取舍", ""]
    for r in rows:
        if r["events"] != max(x["events"] for x in rows):
            continue
        w = r["work_after"]
        lines += ["- %s：SDU 线性乘加小计 %.3f → %.3f GMAC；BIR %.3f GMAC 不变。动态重审仍访问 %s 对证据，但不变的关系编码仅准备 %s 对。" %
                  (r["mode"], r["mac_before"]["SDU"]/1e9, r["mac_after"]["SDU"]/1e9, r["mac_after"]["BIR"]/1e9,
                   format(w["pair_visits"], ","), format(w["pairs_prepared"], ",")),
                  "  PyTorch 峰值 allocated：%.1f → %.1f MiB；查询缓存约 %.1f MiB，容量 %d 个槽位。" %
                  (r["peak_allocated_MiB_before"], r["peak_allocated_MiB_after"], w["pair_cache_bytes"]/2**20, w["pair_cache_capacity"])]
    lines += ["", "缓存用额外显存换取较少重复计算，容量随历史最大活跃队列增长，不随已提交事件总数无限增长；极高事件率仍可能增加显存需求。MAC 只含 Conv2d/Linear，不是完整 FLOPs 或能耗。投影融合不改变 MAC 数，计数阶段使用未融合等价线性路径以保留 hooks。", "",
              "## 适用边界", "",
              "- 使用本地小样本探针权重，尚不是服务器最终权重；这组权重缺完整骨干配对元数据。相同权重前后差分有效，但不能据此宣称最终精度或服务器速度。",
              "- 当前探针学习型停止没有提前提交，本轮加速来自执行优化。没有用测试集改 VoW，也没有把运行波动算作策略收益。",
              "- BIR 仍运行密集卷积，SDU 仍进行动态旧证据重审；没有把通信稀疏误报为 GPU 跳算。后续准确率/机制迭代须使用训练/验证划分，等待服务器正式结果。", "",
              "## 文件与复现", "",
              "- 原始计时及逐窗记录：`%s`。" % source.resolve(),
              "- 同名目录下 reference/optimized 保存逐事件输出和全部活跃查询轨迹。",
              "- tools/bst/profile_inference.py 的 --paired-reference 指向保存的 reference 代码快照即可复跑，--batch-size 控制新版执行批次。",
              "- 模型/代码检查点与具体参数均在原始 JSON args、config 和 SHA256 字段中。"]
    if args.projection_profile:
        micro = json.loads(Path(args.projection_profile).read_text(encoding="utf-8"))
        lines += ["", "## 第二轮投影合并的单独验证", "", "同权重、合成输入，开关融合交替计时 21 次；仅源编码、关系准备与一次 SDU 重审，不含 BIR、检索或 VoW，不能当作整网延迟。", ""]
        for case in micro["cases"]:
            lines.append("- %d 个查询：未合并 %.3f ms，合并 %.3f ms，耗时下降 %.1f%%；本探针输出最大差 %.9g。" %
                         (case["queries"], case["unfused_median_ms"], case["fused_median_ms"],
                          100*(1-case["fused_median_ms"]/case["unfused_median_ms"]), case["max_abs_difference"]))
        lines += ["", "该项收益较小，主要整体收益来自第一轮缓存/检索/执行分块优化；不把不同时间测量的两轮总耗时之差归因于投影融合。"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, mode in zip(axes, ["fixed", "learned"]):
        subset = [r for r in rows if r["mode"] == mode]
        x = np.arange(len(subset))
        ax.bar(x-.18, [r["seconds_before"] for r in subset], .36, label="Before", color="#909da8")
        ax.bar(x+.18, [r["seconds_after"] for r in subset], .36, label="Optimized", color="#247e8a")
        ax.axhline(8, color="#ba493d", linestyle="--", label="8 s input duration")
        ax.set_xticks(x, [format(r["events"], ",") for r in subset])
        ax.set(xlabel="Events per recording", ylabel="Complete processing time (s)", title=mode + " stopping")
        ax.legend(fontsize=8)
    fig.suptitle("Paired online inference | same weights and videos | RTX 5060 Laptop", fontsize=12)
    fig.savefig(out/"speed_comparison.png", dpi=170)
    fig.savefig(out/"speed_comparison.svg")
    print("WROTE", out.resolve())


if __name__ == "__main__":
    main()
