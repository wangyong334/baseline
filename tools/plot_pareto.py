"""精度—能耗帕累托图：回答"是不是靠流式算 160 次堆出来的精度"。

每个点 = 一种方法；横轴 = 每 8 秒理论能耗（mJ，对数轴，事件驱动口径，常数与 tools/energy_v2.py 相同），
纵轴 = test IoU。两个面板：
    (a) 阈值 0.9（原基准协议）
    (b) 等虚警率 Fa = 3e-6（基线 K5 自己的工作点；阈值扫描后在 log Fa 上插值）
数值全部从服务器的结果文件现算（扫阈值 JSON、能耗 JSON），多种子时取均值；只在原始文件缺失时使用 MANUAL 中
写明来源的记录值。可选叠加 tools/sliding_baseline.py 的结果（离线 ANN 改成每 50 ms 因果输出时的代价）。

用法（服务器或本地解压的结果目录，--log-root 可给多个，按顺序查找）:
    python tools/plot_pareto.py --log-root log --out outputs/figures/pareto
    python tools/plot_pareto.py --log-root log --sliding log/sliding_k5/summary_test.json \
        --sliding-sweep log/energy/sweep_sliding_k5_test.json --out outputs/figures/pareto
输出: <out>.png / <out>.pdf / <out>.json（点表，含每个数的来源与种子数）
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.energy_v2 import energy_pj  # noqa: E402

TARGET_FA = 3e-6
WINDOWS = 160
# 原始文件里没有、只在 CLAUDE.md 第 7 节"效率"留有记录的数（写明来源，便于核对）
MANUAL = {"v1_relu_energy": (781.0, "CLAUDE.md §7 效率：ReLU 对照 8 s 781 mJ（全部按 MAC 计）")}
FRONTEND, DECISION, BACKBONE_EVENT = "证据前端", "判决层", "骨干（事件驱动）"


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="精度—能耗帕累托图")
    parser.add_argument("--log-root", nargs="+", default=["log"], help="结果根目录，可给多个，按顺序查找文件")
    parser.add_argument("--sliding", nargs="*", default=[], help="tools/sliding_baseline.py 的 summary JSON")
    parser.add_argument("--sliding-sweep", nargs="*", default=[], help="对滑窗导出目录做的扫阈值 JSON（用于面板 b）")
    parser.add_argument("--out", default="outputs/figures/pareto", help="输出前缀（不含扩展名）")
    return parser.parse_args()


def find(roots, relative):
    """在各个根目录下找文件，返回第一个存在的路径；都没有时返回 None。"""
    for root in roots:
        path = os.path.join(root, relative)
        if os.path.exists(path):
            return path
    return None


def load(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def sweep_rows(roots, relative, key):
    """取扫阈值 JSON 里某个导出目录（或"目录:字段"）的阈值行；缺失时返回 None。"""
    path = find(roots, relative)
    if path is None:
        return None
    result = load(path)["results"].get(key)
    return None if result is None else result["rows"]


def iou_at_threshold(rows, threshold=0.9):
    for row in rows:
        if abs(row["threshold"] - threshold) < 1e-9:
            return row["iou"], row.get("fa")
    return None, None


def iou_at_fa(rows, target):
    """在 log(Fa) 上线性插值得到目标虚警率处的 IoU；目标不在扫描范围内时返回 None（不外推）。"""
    points = sorted([r for r in rows if r.get("fa", 0) > 0], key=lambda r: r["fa"])
    for low, high in zip(points, points[1:]):
        if low["fa"] <= target <= high["fa"]:
            span = math.log(high["fa"]) - math.log(low["fa"])
            w = 0.0 if span == 0 else (math.log(target) - math.log(low["fa"])) / span
            return low["iou"] + w * (high["iou"] - low["iou"])
    return None


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def v2_energy(roots, run_path, with_decision):
    """V2 的每 8 秒能耗：骨干（事件驱动）+ 前端（+ 判决层）。另给 exp/log 按 1× / 20× MAC 折算的范围。"""
    path = find(roots, "energy/energy_v2_round5.json")
    if path is None:
        return None
    report = load(path)
    scale = float(report.get("transcendental_macs", 10.0))
    run = report["runs"].get(run_path)
    if run is None:
        return None
    rows = {r["part"]: r for r in run["rows"]}
    parts = [BACKBONE_EVENT, FRONTEND] + ([DECISION] if with_decision else [])

    def total(s):
        return sum(energy_pj(rows[p]["mac"], rows[p]["ac"], rows[p]["transcendental"], s) for p in parts) * 1e-9 * WINDOWS

    return {"value": total(scale), "low": total(1.0), "high": total(20.0), "source": path}


def collect(roots):
    """按方法汇总点表。每项：名称、类别、延迟、能耗、两个口径的 IoU 及种子数、来源。"""
    points = []
    stream = find(roots, "energy/stream_energy_test.json")
    stream_report = load(stream) if stream else None

    def add(name, family, latency, energy, sweeps):
        rows = [r for r in (sweep_rows(roots, rel, key) for rel, key in sweeps) if r is not None]
        iou09, n09 = mean_or_none([iou_at_threshold(r)[0] for r in rows])
        fa09, _ = mean_or_none([iou_at_threshold(r)[1] for r in rows])
        iou_fa, n_fa = mean_or_none([iou_at_fa(r, TARGET_FA) for r in rows])
        if energy is None or not rows:
            print("跳过 %s：缺少%s" % (name, "能耗" if energy is None else "扫阈值结果"))
            return
        points.append({"name": name, "family": family, "latency": latency, "energy_mj": energy["value"],
                       "energy_low": energy.get("low", energy["value"]), "energy_high": energy.get("high", energy["value"]),
                       "iou_0.9": iou09, "fa_0.9": fa09, "seeds_0.9": n09, "iou_at_fa": iou_fa, "seeds_at_fa": n_fa,
                       "energy_source": energy["source"], "sweeps": ["%s | %s" % s for s in sweeps]})

    if stream_report:
        k5 = stream_report["baselines"][0]
        add("EV-SpSegNet K5 (offline)", "ANN", "offline 8 s",
            {"value": k5["energy_mj_per_8s"], "source": stream + " | baselines[0]"},
            [("energy/threshold_sweep_test.json", "log/verify/baseline_k5_s37_test")])
        merged = [s for s in stream_report["stream"] if s["name"].endswith("_merged.json")]
        if merged:
            add("Streaming V1 (SNN)", "SNN", "50 ms",
                {"value": merged[0]["event"]["energy_mj_per_8s"], "source": stream + " | " + merged[0]["name"]},
                [("energy/threshold_sweep_test.json", "log/verify/v1_s%d_test" % s) for s in (37, 38, 39)])
    value, note = MANUAL["v1_relu_energy"]
    add("Streaming V1-ReLU (ANN)", "ANN", "50 ms", {"value": value, "source": note},
        [("energy/threshold_sweep_test.json", "log/verify/relu_s%d_test" % s) for s in (37, 38, 39)])

    v21 = "log/v21_seed37/eval_test_best_val_iou_seed37_ops.json"
    v21_sweeps = [("energy/threshold_sweep_v21_all_test.json", "log/verify/stream_v21_s%d_test" % s) for s in (37, 38, 39)]
    add("V2 LIF, net", "SNN", "50 ms", v2_energy(roots, v21, False), v21_sweeps)
    add("V2 LIF, d2", "SNN", "150 ms", v2_energy(roots, v21, True), [(a, b + ":prob_fused_d2") for a, b in v21_sweeps])
    floor = "log/v21_floor4_seed37/eval_test_best_val_iou_seed37.json"
    floor_sweeps = [("energy/sweep_v21_floor4_s37_test.json", "log/verify/v21_floor4_s37_test")]
    add("V2 LIF+floor, net", "SNN", "50 ms", v2_energy(roots, floor, False), floor_sweeps)
    add("V2 LIF+floor, d2", "SNN", "150 ms", v2_energy(roots, floor, True), [(a, b + ":prob_fused_d2") for a, b in floor_sweeps])
    relu = "log/v21_relu_seed37/eval_test_best_val_iou_seed37.json"
    relu_sweeps = [("energy/sweep_v21_relu_s%d_test.json" % s, "log/verify/v21_relu_s%d_test" % s) for s in (37, 38, 39)]
    add("V2 ReLU, net", "ANN", "50 ms", v2_energy(roots, relu, False), relu_sweeps)
    add("V2 ReLU, d2", "ANN", "150 ms", v2_energy(roots, relu, True), [(a, b + ":prob_fused_d2") for a, b in relu_sweeps])
    return points


def sliding_points(summaries, sweeps):
    """滑窗基线（离线 ANN 改为每 stride 因果输出）的点：每个上下文长度一个。"""
    tables = [load(p) for p in sweeps]
    points = []
    for path in summaries:
        summary = load(path)
        for entry in summary["contexts"]:
            if "energy_mj_per_8s" not in entry:
                print("跳过滑窗 context %d ms：summary 里没有能耗（sliding_baseline.py 需加 --count-ops）" % entry["context_ms"])
                continue
            rows = None
            for table in tables:
                if entry["dump_dir"] in table["results"]:
                    rows = table["results"][entry["dump_dir"]]["rows"]
            points.append({"name": "K5 sliding %g s" % (entry["context_ms"] / 1000.0), "family": "ANN",
                           "latency": "%g ms" % summary["stride_ms"], "energy_mj": entry["energy_mj_per_8s"],
                           "energy_low": entry["energy_mj_per_8s"], "energy_high": entry["energy_mj_per_8s"],
                           "iou_0.9": entry["iou"], "fa_0.9": entry["fa"], "seeds_0.9": 1,
                           "iou_at_fa": iou_at_fa(rows, TARGET_FA) if rows else None, "seeds_at_fa": 1 if rows else 0,
                           "energy_source": path, "sweeps": [entry["dump_dir"]], "context_ms": entry["context_ms"]})
    return points


# 标签相对点的偏移（点，水平对齐）；右上角几个点挤在一起，逐个指定方向避免重叠
LABEL_OFFSETS = {"V2 LIF+floor, d2": (-7, 7, "right"), "V2 ReLU, d2": (6, -13, "left"),
                 "V2 ReLU, net": (-7, 7, "right"), "V2 LIF, d2": (7, -13, "left"),
                 "Streaming V1-ReLU (ANN)": (-7, 6, "right")}
STYLE = {  # (颜色, 标记)；SNN 用青色，ANN 用灰 / 橙；标记表示延迟
    "SNN": "#1d7f92", "ANN": "#b0632a", "offline 8 s": "^", "50 ms": "o", "150 ms": "s"}


def plot(points, out):
    """两个面板：阈值 0.9 与等虚警率 3e-6。同一模型的 net → d2 用细线相连（判决层的代价与收益）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    panels = (("iou_0.9", "(a) test IoU at threshold 0.9 (benchmark protocol)"),
              ("iou_at_fa", "(b) test IoU at equal false-alarm rate Fa = 3e-6"))
    for ax, (key, title) in zip(axes, panels):
        by_name = {p["name"]: p for p in points}
        for base in ("V2 LIF", "V2 LIF+floor", "V2 ReLU"):
            a, b = by_name.get(base + ", net"), by_name.get(base + ", d2")
            if a and b and a[key] is not None and b[key] is not None:
                ax.plot([a["energy_mj"], b["energy_mj"]], [a[key], b[key]], color="#9aa3ab", lw=0.9, zorder=1)
        slides = sorted([p for p in points if "context_ms" in p and p[key] is not None], key=lambda p: p["context_ms"])
        if slides:
            ax.plot([p["energy_mj"] for p in slides], [p[key] for p in slides], color=STYLE["ANN"], lw=1.0,
                    ls="--", zorder=1)
        for p in points:
            if p[key] is None:
                continue
            marker = STYLE.get(p["latency"], "D")
            color = STYLE[p["family"]]
            filled = p["family"] == "SNN"
            if p["energy_high"] > p["energy_low"] * 1.01:
                ax.errorbar(p["energy_mj"], p[key], xerr=[[p["energy_mj"] - p["energy_low"]],
                                                          [p["energy_high"] - p["energy_mj"]]],
                            fmt="none", ecolor=color, elinewidth=0.8, alpha=0.5, zorder=2)
            ax.scatter(p["energy_mj"], p[key], s=58, marker=marker, zorder=3, linewidths=1.4,
                       facecolors=color if filled else "white", edgecolors=color)
            seeds = p["seeds_0.9"] if key == "iou_0.9" else p["seeds_at_fa"]
            label = p["name"] + ("" if seeds >= 3 else " (%d seed%s)" % (seeds, "" if seeds == 1 else "s"))
            dx, dy, ha = LABEL_OFFSETS.get(p["name"], (6, 4, "left"))
            ax.annotate(label, (p["energy_mj"], p[key]), textcoords="offset points", xytext=(dx, dy), fontsize=7.2,
                        color="#333333", ha=ha)
        ax.set_xscale("log")
        ax.set_xlim(4, 3000)
        ax.set_xlabel("theoretical energy per 8 s (mJ, event-driven accounting)")
        ax.set_title(title, fontsize=9.5)
        ax.grid(True, which="both", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("test IoU")
    axes[0].axhline(0.7590, color="#999999", lw=0.8, ls=":")
    axes[0].text(4.5, 0.7612, "PACT 75.90 (offline, energy n/a; code keeps the baseline downsampling bug)",
                 fontsize=6.8, color="#666666")
    axes[0].axhline(0.7556, color="#999999", lw=0.8, ls=":")
    axes[0].text(4.5, 0.7505, "PointEvent 75.56 (offline, energy n/a)", fontsize=6.8, color="#666666")
    axes[0].set_ylim(0.742, 0.908)
    axes[1].set_ylim(0.782, 0.905)
    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor=STYLE["SNN"], markeredgecolor=STYLE["SNN"],
                          label="spiking (SNN)"),
               plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=STYLE["ANN"], label="ANN"),
               plt.Line2D([], [], marker="^", ls="", color="#555555", markerfacecolor="white", label="offline, output after 8 s"),
               plt.Line2D([], [], marker="o", ls="", color="#555555", markerfacecolor="white", label="streaming, 50 ms"),
               plt.Line2D([], [], marker="s", ls="", color="#555555", markerfacecolor="white",
                          label="streaming, 150 ms (d2 readout)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.075, "error bars on d2 points: exp/log counted as 1x-20x MAC (point at 10x); grey line joins net -> d2 "
                        "of the same weights (cost and gain of the decision layer)", ha="center", fontsize=7, color="#666666")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".png", dpi=220)
    fig.savefig(out + ".pdf")
    plt.close(fig)


def main():
    """入口：收集点表 -> 打印 -> 画图 -> 写 JSON。"""
    args = parse_args()
    points = collect(args.log_root) + sliding_points(args.sliding, args.sliding_sweep)
    if not points:
        raise SystemExit("没有找到任何结果文件，检查 --log-root")
    print("%-28s %-5s %-12s %10s %10s %6s %12s %6s" % ("方法", "类别", "延迟", "mJ/8s", "IoU@0.9", "种子", "IoU@Fa3e-6", "种子"))
    for p in points:
        print("%-28s %-5s %-12s %10.1f %10s %6d %12s %6d" % (
            p["name"], p["family"], p["latency"], p["energy_mj"],
            "-" if p["iou_0.9"] is None else "%.4f" % p["iou_0.9"], p["seeds_0.9"],
            "-" if p["iou_at_fa"] is None else "%.4f" % p["iou_at_fa"], p["seeds_at_fa"]))
    plot(points, args.out)
    with open(args.out + ".json", "w", encoding="utf-8") as stream:
        json.dump({"target_fa": TARGET_FA, "windows": WINDOWS, "points": points}, stream, indent=1, ensure_ascii=False)
    print("图: %s.png / .pdf；点表: %s.json" % (args.out, args.out))


if __name__ == "__main__":
    main()
