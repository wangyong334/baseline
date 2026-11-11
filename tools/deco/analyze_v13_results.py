"""Audit downloaded V13 results and aggregate event-scale execution workload.

No model fitting or test-set policy selection. Saved endpoints are NOT the
actual early-commit predictions; endpoint diagnostics are counterfactual.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quantiles(x):
    return {k: float(v) for k, v in zip(["p50", "p95", "max"], np.percentile(x, [50, 95, 100]))} if len(x) else {}


def analyze(root):
    rows, identities, aligned = [], set(), {}
    for path in sorted(root.glob("run/deco/*/*/vow_best/vow.json")):
        backbone = path.parents[2].name
        seed = int(re.search(r"_s(\d+)_", path.parents[1].name)[1])
        assert (backbone, seed) not in identities
        identities.add((backbone, seed))
        report = read(path)
        summary = read(path.parents[1] / "summary.json")
        evidence = summary["evidence"]
        with np.load(path.with_name("vow_test_events.npz"), allow_pickle=False) as source:
            a = {k: source[k].copy() for k in source.files}
        n = len(a["m0"])
        assert all(len(v) == n for v in a.values())
        assert all(np.isfinite(a[k]).all() for k in ["m0", "U0", "UF", "q_lo", "q_hi"])
        assert ((0 <= a["L"]) & (a["L"] <= a["F"]) & (a["F"] <= evidence["D"])).all()
        comparison = {k: a[k] for k in ["m0", "label", "rec", "F"]}
        if backbone in aligned:
            assert all(np.array_equal(v, aligned[backbone][k]) for k, v in comparison.items())
        else:
            aligned[backbone] = comparison
        test = report["test"]
        total = int(round(test["vow"]["spawned"] / test["vow"]["births"]))
        for policy in ["vow", "sym"]:
            selection = a["spawn_" + policy].astype(bool)
            assert int(selection.sum()) == test[policy]["spawned"]
            wait = float(a["L"][selection].astype(np.int64).sum() * evidence["window_ms"] / total)
            assert np.isclose(wait, test[policy]["mean_wait_ms_all_events"], atol=1e-9)
        y = a["label"].astype(bool)
        p, initial, final = a["m0"] > 0, a["U0"] > 0, a["UF"] > 0
        safe = (a["m0"] != 0) & (a["U0"] != 0) & (a["UF"] != 0)
        spawn = a["spawn_vow"].astype(bool)
        rescue = (p != y) & (final == y) & safe
        harm = (p == y) & (final != y) & safe
        groups = {}
        for name, sel in {"band_all": np.ones(n, bool), "band_target": y, "band_background": ~y,
                          "band_onset": a["onset"].astype(bool), "band_persistent": a["persistent"].astype(bool)}.items():
            group_wait = np.where(spawn, a["L"], 0).astype(np.float64)[sel] * evidence["window_ms"]
            groups[name] = {"events": int(sel.sum()), "spawned": int((spawn & sel).sum()),
                            "mean_wait_ms": float(group_wait.mean()) if len(group_wait) else None,
                            "wait_ms_quantiles": quantiles(group_wait),
                            "spawned_wait_ms_quantiles": quantiles(a["L"][spawn & sel].astype(float) * evidence["window_ms"])}
        # Defaults verified against saved V11 training config: dim64, heads4.
        assert summary["args"]["design"] == "v11" and summary["args"]["v11_audit"] == "reaudit"
        dim, heads = 64, 4
        slots = (evidence["past"] + evidence["D"] + 1) * (evidence["k_near"] + evidence["k_evid"])
        entry = 2*dim + dim*dim + slots*(6*dim + 4*dim*dim)
        step_linear = 3*dim*dim + 2*heads*dim + 4*dim*dim + slots*dim*heads
        step_attention = 2*slots*dim
        mac = entry + (evidence["D"]+1)*(step_linear+step_attention)
        rows.append({"backbone": backbone, "sdu_seed": seed, "source": str(path),
                     "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "n_events": total, "n_band": n, "best_epoch": summary["best_epoch"],
                     "policies_test": test, "policies_val": report["val"],
                     "checks": {"array_lengths_finite_ages": True, "within_backbone_rows_aligned": True,
                                "spawn_wait_reconciled": True,
                                "endpoint_sign_flip_disagreement": int(((p != final) != a["flip"].astype(bool)).sum()),
                                "zero_endpoint_rows_excluded_from_transitions": int((~safe).sum())},
                     "endpoint_counterfactuals": {"prior_to_deadline_rescues": int(rescue.sum()),
                         "prior_to_deadline_harms": int(harm.sum()),
                         "blocked_deadline_rescues": int((rescue & ~spawn).sum()),
                         "avoided_deadline_harms": int((harm & ~spawn).sum()),
                         "arrival_sdu_rescues": int(((p != y) & (initial == y) & safe).sum()),
                         "arrival_sdu_harms": int(((p == y) & (initial != y) & safe).sum()),
                         "future_rescues_vs_arrival_sdu": int(((initial != y) & (final == y) & safe).sum()),
                         "future_harms_vs_arrival_sdu": int(((initial == y) & (final != y) & safe).sum())},
                     "groups": groups,
                     "work": {"slots_per_query": slots, "sdu_mac_subtotal_per_full_query": mac,
                              "entry_mac": entry, "linear_mac_per_age": step_linear,
                              "attention_mac_per_age": step_attention,
                              "full_trace_query_updates": n*(evidence["D"]+1),
                              "full_trace_mac_subtotal": n*mac,
                              "hypothetical_stopped_query_updates": int((a["L"][spawn].astype(np.int64)+1).sum()),
                              "warning": "Old evaluator computes full traces before simulating birth/stop. Hypothetical updates are not measured savings."}})
    if not rows:
        raise ValueError("no vow_best results found")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile")
    ap.add_argument("--stage-profile")
    args = ap.parse_args()
    rows = analyze(Path(args.root))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for backbone in sorted({r["backbone"] for r in rows}):
        subset = [r for r in rows if r["backbone"] == backbone]
        groups[backbone] = {}
        for policy in ["arrival", "sym", "vow", "vow_z0", "margin_m", "wait_all"]:
            values = [r["policies_test"][policy]["official"]["iou"]*100 for r in subset]
            groups[backbone][policy] = {"mean_iou_percent": float(np.mean(values)),
                                      "sd_iou_pp": float(np.std(values, ddof=1)) if len(values)>1 else None}
    data = {"rows": rows, "within_backbone_descriptive_statistics": groups,
            "independent_backbones": len(groups), "seed_nesting": "SDU seeds are nested within BIR seeds",
            "limits": ["No saved actual commit logits U(L); endpoint transitions cannot reconstruct early-commit errors.",
                       "FP/Fa and IoU use original official definitions; ACC is target-event recall.",
                       "Old sym baseline uses V11 birth band plus learned commit, not fixed-wait V11.",
                       "MAC subtotal excludes BIR/VoW/nonlinearities/lookup/backward; timing needs separate measurement."]}
    lines = ["# V13 结果与整条模型推理开销核查", "", "本报告不修改模型、阈值或 CLAUDE.md。旧 V13 下载结果与新共享模型的本地运行探针分开解释。", "",
             "## 1. 下载结果（官方测试 IoU，单位 %）", "",
             "6 组运行来自 2 个 BIR 骨干种子，每个骨干配 3 个 SDU 种子；不是 6 个独立骨干实验。sym 是 V11 进入区间搭配学习型停止。", "",
             "| BIR / SDU | BIR | sym | VoW | 等出生数 margin | VoW 出生比例 | VoW 全事件平均等待 ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        t = r["policies_test"]
        lines.append("| %s / %s | %.2f | %.2f | %.2f | %.2f | %.2f%% | %.3f |" %
                     (r["backbone"], r["sdu_seed"], *[t[p]["official"]["iou"]*100 for p in ["arrival","sym","vow","margin_m"]],
                      t["vow"]["births"]*100, t["vow"]["mean_wait_ms_all_events"]))
    lines += ["", "这张表中的等待是证据等待，不是 GPU 推理时间。相对等出生数 margin，VoW 的 IoU 增益为：" +
              "、".join("%+.3f pp" % ((r["policies_test"]["vow"]["iou_pooled"]-r["policies_test"]["margin_m"]["iou_pooled"])*100) for r in rows) + "。", "",
              "VoW 相比 sym 在六组中的 Fa 都更高；当前结果体现召回与虚警的权衡，不能写成 VoW 全面降低虚警。", "",
              "## 2. 事件数量如何放大旧实现的计算", "",
              "V11 每候选事件最多有 336 个证据槽，计算 11 个年龄状态；一次完整轨迹约 7,378,560 MAC（线性层和两次注意力收缩的小计）。MAC 指一次乘加，按 2 FLOPs/MAC 换算仍不是完整 FLOPs。", "",
              "不是对所有事件做全连接两两比较，因此不能简单说 O(N²)。但固定邻居数和等待上限后，总代价仍随候选事件数近似线性增长；重复处理旧邻居和 Python/CUDA 调度开销不能忽略。", "",
              "| 骨干 | 测试事件总数 | 宽区间候选数 | 实际完整轨迹更新次数 | SDU MAC 小计 / 单次测试轨迹 |", "|---|---:|---:|---:|---:|"]
    for backbone in groups:
        r = next(r for r in rows if r["backbone"]==backbone)
        w = r["work"]
        lines.append("| %s | %s | %s | %s | %.3f TMAC |" % (backbone, r["n_events"], r["n_band"], w["full_trace_query_updates"], w["full_trace_mac_subtotal"]/1e12))
    lines += ["", "关键缺陷：旧 fit_vow/evaluate 先算完整轨迹，再模拟进入/停止。以上工作量不会因为表中的 VoW 出生比例变小就自动减少。它不是实测在线省算。", "",
              "## 3. 新共享模型：真实录像完整推理", ""]
    if args.profile:
        profile = read(args.profile)
        assert sum(r["events"] for r in profile["inventory"]) == rows[0]["n_events"], "raw split denominator mismatch"
        data["profile_source"] = str(Path(args.profile).resolve())
        lines += ["硬件：%s；PyTorch %s；float32；每组预热 1 次、计时 %s 次。选取测试集中事件数最少、中位、最多的三段完整录像，选取依据只有事件数。" %
                  (profile["hardware"], profile["torch"], profile["args"]["repeats"]), "",
                  "计时包含事件表示、BIR、邻域检索、共享缓存、SDU、VoW、CPU/GPU 传输和输出回 CPU；不包含磁盘载入、权重加载、指标计算和结果写盘。逐窗 CUDA 同步可能增加开销。", "",
                  "**使用本地小样本训练的探针权重，不是服务器最终训练权重；不能作为论文最终性能，也不能声称与旧 V13 等精度加速。**", "",
                  "| 录像 / 停止方式 | 事件数 | 视频秒数 | 推理中位秒数 | 实时因子 | 窗耗时 p95 / 最大 ms* | 出生事件数 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for case in profile["cases"]:
            rr = min(case["runs"], key=lambda r: abs(r["seconds"]-case["median_seconds"]))
            lines.append("| %s / %s | %d | %.1f | %.3f | %.3f | %.1f / %.1f | %d |" %
                         (case["recording"],case["mode"],case["events"],case["duration_s"],case["median_seconds"],
                          case["median_seconds"]/case["duration_s"],rr["window_summary"]["p95_ms"],rr["window_summary"]["max_ms"],rr["stats"]["births"]))
        lines += ["", "*窗口统计取总耗时位于中位数的那次运行。实时因子=处理耗时/录像时长，小于 1 只说明整段平均跟得上，不保证所有 50 ms 窗口按时完成。JSON 中记录了每窗耗时和串行队列积压估算。", "",
                  "线性/卷积 MAC 用独立不计时前向统计，不把 hook 开销混进速度。只统计 Conv2d/Linear，未包括所有注意力、非线性、检索运算；BIR 使用密集卷积，消息稀疏不意味着这些内核跳算。", ""]
        for case in profile["cases"]:
            lines.append("- %s / %s：%s" % (case["recording"], case["mode"],
                          "，".join("%s %.3f GMAC" % (k,v/1e9) for k,v in case["linear_conv_mac_subtotal"].items())))
        learned = [c for c in profile["cases"] if c["mode"] == "learned"]
        if learned and all(r["stats"]["policy_commits"] == 0 for c in learned for r in c["runs"]):
            lines += ["", "**本次探针的学习型停止没有提前提交任何事件**，三段录像的 SDU 更新数与固定等待相同。因此两种模式的时差不能归因于 VoW 省算，而是运行波动和附加检查开销的综合结果；未充分训练的策略尚未提供计算节省。"]
        lines += ["", "本地检查点缺少完整的骨干身份元数据（provenance_verified=false）；使用指定 BIR 的计算探针可运行，但不能把它当作已经验证配对的正式精度实验。原始 24 段录像合计事件数与旧结果分母 2,074,586 一致。"]
        # Scientific figure: measured runtime and the dense-stream deadline.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        cases = [c for c in profile["cases"] if c["mode"] == "fixed"]
        ax[0].bar(np.arange(len(cases)), [c["median_seconds"] for c in cases], color="#287b8e")
        ax[0].axhline(8, color="#bd4338", linestyle="--", label="8 s recording duration")
        ax[0].set_xticks(np.arange(len(cases)), [format(c["events"], ",") for c in cases])
        ax[0].set(xlabel="Events per complete recording", ylabel="Pipeline processing time (s)", title="Shared model, fixed deadline")
        ax[0].legend(fontsize=8)
        dense = max(cases, key=lambda c: c["events"])
        run = min(dense["runs"], key=lambda r: abs(r["seconds"]-dense["median_seconds"]))
        ax[1].plot([r["window"] for r in run["windows"]], [r["ms"] for r in run["windows"]], color="#287b8e", linewidth=1.5)
        ax[1].axhline(50, color="#bd4338", linestyle="--", label="50 ms input interval")
        ax[1].set(xlabel="Window index", ylabel="Processing time (ms)", title="Dense recording: %s events" % format(dense["events"], ","))
        ax[1].legend(fontsize=8)
        fig.suptitle("RTX 5060 Laptop | local probe weights | synchronized end-to-end computation", fontsize=11)
        fig.savefig(out/"inference_cost.png", dpi=170)
        fig.savefig(out/"inference_cost.svg")
        plt.close(fig)
        lines += ["", "![实测整段与逐窗计算时间](inference_cost.png)"]
    else:
        lines.append("在线计时另见 profile_inference.py 输出；旧下载包不含完整在线延迟测量。")
    if args.stage_profile:
        stages = read(args.stage_profile)
        data["stage_profile_source"] = str(Path(args.stage_profile).resolve())
        lines += ["", "### 稠密录像分阶段定位（额外同步的诊断前向）", "", "单独运行，额外同步会扰动速度，不能与上表各秒数直接相减。", ""]
        for c in stages["cases"]:
            lines.append("- %s / %s：%s" % (c["recording"], c["mode"], "；".join("%s %.3f s" % (k,v) for k,v in c["stage_seconds_separate_instrumented_pass"].items())))
    lines += ["", "## 4. 解释边界和改进优先级", "",
              "1. 保持每事件的分类输出，但昂贵特征编码应共享；只对尚未提交的候选继续审查旧证据。进入比例、活跃队列大小和更新次数一起决定负载，单看显存不足以判断轻量或实时。",
              "2. 稠密 BIR 每 50 ms 执行一次，与事件数增长不完全同比；事件读出、证据检索和 SDU 则随事件/候选密度增长。继续优化前应对照分阶段耗时，不能只减少参数。",
              "3. 分别报告切窗收集等待、等待未来证据、实际计算、计算队列积压。全事件平均等待会被立即输出的背景事件稀释；目标组和出生组分位数已保存在 analysis.json。",
              "4. NPZ 仅保存先验、到达时 SDU、截止时 SDU 和停止年龄；缺少实际停止分数 U(L)。不能据此精确恢复提前停止造成的所有误报漏检。端点纠错/损伤统计只能当作反事实诊断。",
              "5. 完整实时结论还需服务器最终权重、完整测试集、同硬件重复测量和更密集流压力测试。本地三个密度点不是最大稳定吞吐率或普遍实时保证。", "",
              "## 来源", "", "- 原始结果根目录：`%s`" % Path(args.root).resolve(),
              "- 每次运行的来源路径、摘要 SHA256、配置口径、数组校验和诊断见 analysis.json。",
              "- 计算来源：tools/deco/sdn_v11.py、tools/deco/fit_vow.py；在线执行：tools/bst/infer.py、tools/bst/stream.py。"]
    (out/"analysis.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("WROTE", out.resolve())


if __name__ == "__main__":
    main()
