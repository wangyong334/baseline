"""Paper tables and figures from the V9-final results (FINAL_PLAN_20261104.md §8). CPU only.

Reads a manifest (JSON; configs/final/paper_manifest.json lists the final-round paths) and writes, next to --out:
    tables.md            T1 main table (EV-UAV test; offline methods = numbers from their papers only, no K5),
                         T2 module-three controls, T3 commit policies, T4 module-two 2x2 + noise, T5 25 ms vs 50 ms
    fig_frontier.png     IoU against the mean decision latency of target events (log x), per policy family
    fig_info_arrival.png IoU and corrections against the commit lag (fixed denominator)
    fig_robustness.png   IoU and Fa against injected background noise (Hz/px), one panel each (no dual axis)
    fig_wait.png         adaptive waits by recording group and target phase
    fig_teaser.png       online vs offline: published offline IoU at their decision latency (output at the end of an
                         8 s clip: ~4 s on average per event) against our streaming operating points
Missing inputs are skipped with a note, so the script runs before every result exists.
A manifest value that is a list of paths (several seeds) is reported as mean +- std.
"""
import argparse
import json
import math
import os

import numpy as np

PUBLISHED = [   # test IoU / ACC / Pd / Fa as reported by the original papers (EV-UAV); offline, 8 s clips
    {"name": "EV-SpSegNet (ICCV'25)", "iou": 0.5518, "acc": None, "pd": 0.7753, "fa": 1.63e-4},
    {"name": "SpTopoNet (CVPR'26)", "iou": 0.6662, "acc": 0.7443, "pd": 0.8336, "fa": 1.29e-4},
    {"name": "PointEvent", "iou": 0.7556, "acc": None, "pd": None, "fa": None},
]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]     # reference palette, fixed order (validated)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
HZ = (0.0, 0.8, 1.6, 2.4, 4.0)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _paths(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def load_json(path, base):
    p = path if os.path.isabs(path) else os.path.join(base, path)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def loads(value, base):
    out = [load_json(p, base) for p in _paths(value)]
    return [o for o in out if o is not None]


def backbone_metrics(d):
    """carry-mode metrics of a train_stream_v1 eval JSON."""
    c = d["carry"]
    lat = c.get("latency") or {}
    return {"iou": c["iou"], "acc": c["acc"], "pd": c.get("pd"), "fa": c.get("fa"),
            "first_det_ms": lat.get("latency_median_ms"), "messages": c.get("messages_mean"),
            "decision_ms": None}


def head_metrics(d, kind="track"):
    r = d[kind]
    return {"iou": r["iou"], "acc": r["acc"], "pd": r.get("pd"), "fa": r.get("fa"), "tp": r.get("tp"),
            "fp": r.get("fp"), "fn": r.get("fn")}


def mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None)


def fmt(vals, digits=4, sci=False):
    m, s = mean_std(vals if isinstance(vals, list) else [vals])
    if m is None:
        return "-"
    f = ("%%.%de" % 2) if sci else ("%%.%df" % digits)
    return (f % m) + ((" ± " + (f % s)) if s is not None else "")


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def table_main(man, base, notes):
    rows = ["| 方法 | 因果 / 流式 | IoU | ACC | Pd | Fa | 首检中位 ms | 目标事件判决延迟 均值 ms |", "|---|---|---|---|---|---|---|---|"]
    for p in PUBLISHED:
        rows.append("| %s | 否（离线 8 s 片段） | %.4f | %s | %s | %s | 约 4000（片段结束才出结果） | - |" % (
            p["name"], p["iou"], "%.4f" % p["acc"] if p["acc"] else "-", "%.4f" % p["pd"] if p["pd"] else "-",
            "%.2e" % p["fa"] if p["fa"] else "-"))
    for r in man.get("main", []):
        ms = []
        if r.get("backbone_eval"):
            ms = [backbone_metrics(d) for d in loads(r["backbone_eval"], base)]
        elif r.get("head_eval"):
            ms = [head_metrics(d) for d in loads(r["head_eval"], base)]
        if not ms:
            notes.append("T1 缺：%s" % r["name"])
            continue
        lat, fd = [], []
        for a in loads(r.get("analysis"), base):
            pol = a["policies"][r.get("policy", "zero")]
            lat.append(pol["latency_ms"]["target_mean"])
            fd.append(pol["first_detection_ms"]["median"])
        if not fd:
            fd = [m.get("first_det_ms") for m in ms]
        rows.append("| %s | 是 | %s | %s | %s | %s | %s | %s |" % (
            r["name"], fmt([m["iou"] for m in ms]), fmt([m["acc"] for m in ms]), fmt([m["pd"] for m in ms]),
            fmt([m["fa"] for m in ms], sci=True), fmt(fd, 1), fmt(lat, 1)))
    return "### T1 主表（EV-UAV test，阈值 0.9；离线方法只用原论文的数字）\n\n" + "\n".join(rows)


def table_controls(man, base, notes):
    rows = ["| 方法 | val IoU | test IoU | test 普通 / 极密 | test 第一拍召回 | test 救回（同窗 / 第一拍 / 断续） | 误压 | 压掉 / 新增误报 | 目标延迟 均值 ms |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in man.get("controls", []):
        cells = {}
        for split in ("val", "test"):
            an = loads(r.get("analysis_" + split), base)
            pols = [a["policies"][r.get("policy", "fixed_%d" % a["D"])] for a in an]
            if not pols and r.get("b1"):
                b1 = loads(r["b1"], base)
                pols = [b["chosen_on_val"][str(r.get("b1_lag", 10))][split] for b in b1]
            if not pols and r.get("eval_" + split):                    # eval JSON only: IoU and groups
                pols = [eval_as_policy(d, r.get("dense_" + split, [])) for d in loads(r["eval_" + split], base)]
            cells[split] = pols
        if not cells["test"]:
            notes.append("T2 缺：%s" % r["name"])
            continue
        t = cells["test"]
        if any("flips" not in p for p in t):
            rows.append("| %s | %s | %s | %s / %s | - | - | - | - | - |" % (
                r["name"], fmt([p["iou"] for p in cells["val"]]), fmt([p["iou"] for p in t]),
                fmt([p["groups"]["normal"] for p in t]), fmt([p["groups"]["dense"] for p in t])))
            continue
        fl = [p["flips"] for p in t]
        rows.append("| %s | %s | %s | %s / %s | %s | %s / %s / %s | %s | %s / %s | %s |" % (
            r["name"], fmt([p["iou"] for p in cells["val"]]), fmt([p["iou"] for p in t]),
            fmt([p["groups"]["normal"] for p in t]), fmt([p["groups"]["dense"] for p in t]),
            fmt([p["first_beat_recall"] for p in t], 3),
            fmt([f["rescued"]["same"] for f in fl], 0), fmt([f["rescued"]["onset"] for f in fl], 0),
            fmt([f["rescued"]["gap"] for f in fl], 0), fmt([f["suppressed"] for f in fl], 0),
            fmt([f["fp_removed"] for f in fl], 0), fmt([f["fp_added"] for f in fl], 0),
            fmt([p["latency_ms"]["target_mean"] for p in t], 1)))
    return "### T2 模块三的对照（同一骨干、同未来预算 500 ms；B1 / 等待窗数按 val 选）\n\n" + "\n".join(rows)


def eval_as_policy(d, dense):
    """A head eval JSON as a partial policy record (IoU and group IoU from the per-sequence counts)."""
    from utils.recording_groups import group_report
    r = d["track"]
    seqs = (d.get("per_sequence") or {}).get("track")
    g = group_report(seqs, dense) if seqs else {"normal": None, "dense": None}
    return {"iou": r["iou"], "groups": {"normal": g["normal"], "dense": g["dense"]}}


def table_policies(man, base, notes):
    an = loads(man.get("policies_analysis"), base)
    if not an:
        notes.append("T3 缺：policies_analysis")
        return ""
    a = an[0]
    D = a["D"]
    order = (["zero"] + ["fixed_%d" % k for k in (1, 3, 5, D)] + ["simple_0.05_1", "simple_0.05_%d" % D]
             + ["adaptive_%g" % x for x in (0.2, 0.1, 0.05, 0.01)] + ["slice_0.2", "slice_0.05", "window_0.05"])
    rows = ["| 提交策略 | IoU | 普通 / 极密 | 目标延迟 均值 / 中位 / p90 ms | 到达即判 | 首检中位 ms | 第一拍召回 |",
            "|---|---|---|---|---|---|---|"]
    for n in order:
        if n not in a["policies"]:
            continue
        p = a["policies"][n]
        src = p.get("target_commit_source", {})
        arrival = ("%.1f%%" % (100 * src["arrival"]) if src else
                   "%.1f%%" % (100 * p["wait_windows"]["target_zero_frac"]) if n.startswith(("simple", "zero")) else "-")
        rows.append("| %s | %.4f | %.4f / %s | %.1f / %.1f / %.1f | %s | %s | %.3f |" % (
            n, p["iou"], p["groups"]["normal"], "%.4f" % p["groups"]["dense"] if p["groups"]["dense"] == p["groups"]["dense"] else "-",
            p["latency_ms"]["target_mean"], p["latency_ms"]["target_median"], p["latency_ms"]["target_p90"], arrival,
            "%.1f" % p["first_detection_ms"]["median"] if p["first_detection_ms"]["median"] is not None else "-",
            p["first_beat_recall"]))
    title = "### T3 同一份权重的提交策略（%s，%d ms 窗，D = %d）\n\n" % (man.get("policies_title", ""), a["window_ms"], D)
    return title + "\n".join(rows)


def noise_file(d, stem, hz, window_ms, tag=""):
    """eval JSON of one backbone (or head, tag = "_adaptive") at hz (0 = clean); new name (Hz) first, then legacy."""
    if hz == 0:
        return os.path.join(d, "eval_test_%s%s.json" % (stem, tag))
    if tag:
        return os.path.join(d, "eval_test_%s%s_noise%ghz_seed0.json" % (stem, tag, hz))
    new = os.path.join(d, "eval_test_%s_noise%ghz_seed0.json" % (stem, hz))
    if os.path.exists(new):
        return new
    return os.path.join(d, "eval_test_%s_noiseL%g_seed0.json" % (stem, round(25.0 * hz * window_ms / 1000.0, 6)))


def robustness_rows(man, base):
    out = []
    for s in man.get("robustness", []):
        pts = []
        for hz in HZ:
            tag = s.get("tag", "")
            path = s["clean"] if (hz == 0 and s.get("clean")) else noise_file(s["dir"], s["stem"], hz,
                                                                              s.get("window_ms", 50), tag)
            d = load_json(path, base)
            if d is None:
                pts.append(None)
            elif "carry" in d:
                pts.append(backbone_metrics(d))
            else:                                                       # full model (head eval JSON)
                m = head_metrics(d)
                m["messages"] = None
                pts.append(m)
        out.append((s["name"], pts))
    return out


def table_module2(man, base, notes):
    rows = ["| 骨干 | val IoU（普通 / 极密） | test 干净 | 0.8 | 1.6 | 2.4 | 4 Hz/px | Fa 0 → 4 Hz/px | 每窗消息 0 → 4 Hz/px |",
            "|---|---|---|---|---|---|---|---|---|"]
    rob = dict(robustness_rows(man, base))
    for s in man.get("robustness", []):
        pts = rob.get(s["name"])
        val = loads(s.get("val_eval"), base)
        vtxt = "-"
        if val:
            c = val[0]["carry"]
            groups = s.get("val_groups")
            vtxt = "%.4f" % c["iou"]
            if "per_sequence" in c and groups is not None:
                from_counts = _group_iou(c["per_sequence"], groups)
                vtxt += "（%s / %s）" % from_counts
        if not pts or all(p is None for p in pts):
            notes.append("T4 缺：%s" % s["name"])
            continue
        cells = ["%.4f" % p["iou"] if p else "-" for p in pts]
        fa = "%s → %s" % ("%.2e" % pts[0]["fa"] if pts[0] else "-", "%.2e" % pts[-1]["fa"] if pts[-1] else "-")
        msg = ("×%.2f" % (pts[-1]["messages"] / pts[0]["messages"])) if pts[0] and pts[-1] and pts[0]["messages"] else "-"
        rows.append("| %s | %s | %s | %s |" % (s["name"], vtxt, " | ".join(cells), fa + " | " + msg))
    return "### T4 模块二：匹配 2x2 与受控噪声（test，注入种子 0）\n\n" + "\n".join(rows)


def _group_iou(per_seq, dense):
    from utils.recording_groups import group_report  # noqa: WPS433
    g = group_report(per_seq, dense)
    f = lambda v: "-" if v is None else "%.4f" % v              # noqa: E731
    return f(g["normal"]), f(g["dense"])


def table_window(man, base, notes):
    rows = ["| 配置 | 窗长 | val IoU | test IoU | 目标延迟 均值 / 中位 ms | 首检中位 ms | 每秒消息 |", "|---|---|---|---|---|---|---|"]
    for r in man.get("window_length", []):
        cells = {}
        for split in ("val", "test"):
            an = loads(r.get("analysis_" + split), base)
            cells[split] = [a["policies"][r["policy"]] for a in an]
        if not cells["test"]:
            notes.append("T5 缺：%s" % r["name"])
            continue
        t = cells["test"]
        msg = loads(r.get("backbone_eval"), base)
        mps = [backbone_metrics(d)["messages"] * 1000.0 / float(r["window_ms"]) for d in msg if backbone_metrics(d)["messages"]]
        rows.append("| %s | %d ms | %s | %s | %s / %s | %s | %s |" % (
            r["name"], r["window_ms"], fmt([p["iou"] for p in cells["val"]]), fmt([p["iou"] for p in t]),
            fmt([p["latency_ms"]["target_mean"] for p in t], 1), fmt([p["latency_ms"]["target_median"] for p in t], 1),
            fmt([p["first_detection_ms"]["median"] for p in t], 1), fmt(mps, 0) if mps else "-"))
    return "### T5 切得更细：25 ms 对 50 ms（模块一 + 三；未来预算都是 500 ms）\n\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _axes(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def fig_frontier(man, base, path, notes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = [(s, loads(s.get("analysis"), base)) for s in man.get("frontier", [])]
    series = [(s, a[0]) for s, a in series if a]
    if not series:
        notes.append("图 frontier 缺数据")
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _axes(ax, "目标事件的平均判决延迟（ms，对数轴；不含计算时间）", "IoU（test，阈值 0.9）")
    for i, (s, a) in enumerate(series):
        col = SERIES[i % len(SERIES)]
        D = a["D"]
        fixed = ["zero"] + ["fixed_%d" % k for k in range(1, D + 1)]
        xs = [a["policies"][n]["latency_ms"]["target_mean"] for n in fixed]
        ys = [a["policies"][n]["iou"] for n in fixed]
        ax.plot(xs, ys, color=col, linewidth=1.2, linestyle=(0, (3, 2)), marker="o", markersize=3.5,
                markerfacecolor=SURFACE, markeredgecolor=col, label="%s：每个事件固定等 k 窗（k = 0..%d）" % (s["name"], D))
        ax.annotate("不等（骨干）", (xs[0], ys[0]), xytext=(5, -2), textcoords="offset points", fontsize=7, color=INK2)
        ax.annotate("等 %d 窗" % D, (xs[-1], ys[-1]), xytext=(-4, -11), textcoords="offset points", fontsize=7,
                    color=INK2, ha="right")
        alphas = (0.2, 0.1, 0.05, 0.01)
        ad = ["adaptive_%g" % x for x in alphas]
        xa = [a["policies"][n]["latency_ms"]["target_mean"] for n in ad]
        ya = [a["policies"][n]["iou"] for n in ad]
        ax.plot(xa, ya, color=col, linewidth=1.8, marker="D", markersize=4.5,
                label="%s：逐事件自适应提交" % s["name"])
        for x, y, al in zip(xa, ya, alphas):
            ax.annotate("α %g" % al, (x, y), xytext=(0, 5), textcoords="offset points", fontsize=6.5, color=INK2,
                        ha="center")
        sl = ["slice_%g" % x for x in alphas if "slice_%g" % x in a["policies"]]
        if sl:
            xs_ = [a["policies"][n]["latency_ms"]["target_mean"] for n in sl]
            ys_ = [a["policies"][n]["iou"] for n in sl]
            ax.plot(xs_, ys_, color=col, linewidth=1.0, marker="s", markersize=5, markerfacecolor="none",
                    label="%s：切片级自适应（整窗一起放行）" % s["name"])
    for p in man.get("frontier_points", []):
        d = loads(p.get("analysis"), base)
        if d:
            pol = d[0]["policies"][p["policy"]]
            ax.plot([pol["latency_ms"]["target_mean"]], [pol["iou"]], marker="^", markersize=5, color=INK2,
                    linestyle="none")
            ax.annotate(p["name"], (pol["latency_ms"]["target_mean"], pol["iou"]), xytext=(5, -3),
                        textcoords="offset points", fontsize=7, color=INK2)
    for b in loads(man.get("frontier_b1"), base)[:1]:                  # B1: one point per val-chosen lag
        pts = sorted((int(k), v["test"]) for k, v in b["chosen_on_val"].items())
        xb = [v["latency_ms"]["target_mean"] for _, v in pts]
        yb = [v["iou"] for _, v in pts]
        ax.plot(xb, yb, color=INK2, linewidth=1.0, linestyle=":", marker="^", markersize=4.5,
                label="B1：不学习的 M-of-N 延迟确认（参数按 val 选）")
    ax.set_xscale("log")
    ticks = [12.5, 25, 50, 100, 200, 500]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%g" % t for t in ticks])
    ax.minorticks_off()
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, labelcolor=INK)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_teaser(man, base, path, notes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = []
    for p in man.get("teaser", []):
        d = loads(p.get("analysis"), base)
        if d:
            pol = d[0]["policies"][p["policy"]]
            pts.append((p["name"], pol["latency_ms"]["target_mean"], pol["iou"]))
    if not pts:
        notes.append("图 teaser 缺数据")
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _axes(ax, "每个事件拿到判决的平均延迟（ms，对数轴）", "IoU（EV-UAV test）")
    clip_ms = 4000.0                    # offline: output at the end of an 8 s clip, events uniform in the clip
    for i, q in enumerate(PUBLISHED):
        ax.plot([clip_ms], [q["iou"]], marker="o", markersize=5, color=INK2, linestyle="none")
        ax.annotate(q["name"], (clip_ms, q["iou"]), xytext=(-6, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=7, color=INK2)
    xs, ys = [x for _, x, _ in pts], [y for _, _, y in pts]
    ax.plot(xs, ys, color=SERIES[0], linewidth=1.8, marker="D", markersize=4.5)
    for name, x, y in pts:
        ax.annotate(name, (x, y), xytext=(4, -10), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xscale("log")
    ticks = [25, 50, 100, 250, 500, 1000, 4000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%g" % t for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(15, 9000)
    ax.axvline(clip_ms, color=INK2, linewidth=0.8, linestyle=":")
    ax.annotate("离线方法：看完 8 s 片段才出结果", (clip_ms, max(ys)), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_info_arrival(man, base, path, notes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(s, loads(s.get("analysis"), base)) for s in man.get("info_arrival", [])]
    rows = [(s, a[0]) for s, a in rows if a]
    if not rows:
        notes.append("图 info_arrival 缺数据")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _axes(axes[0], "提交滞后（ms）", "IoU")
    _axes(axes[1], "提交滞后（ms）", "相对 V5 的改判（事件数）")
    for i, (s, a) in enumerate(rows):
        col = SERIES[i % len(SERIES)]
        ia = a["info_arrival"]
        ms = [r["ms"] for r in ia]
        axes[0].plot(ms, [r["iou"] for r in ia], color=col, linewidth=1.8, marker="o", markersize=3.5, label=s["name"])
        axes[0].annotate(s["name"], (ms[-1], ia[-1]["iou"]), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK)
        if i == 0:
            resc = [sum(r["flips"]["rescued"].values()) for r in ia]
            supp = [r["flips"]["suppressed"] for r in ia]
            fpr = [r["flips"]["fp_removed"] - r["flips"]["fp_added"] for r in ia]
            axes[1].plot(ms, resc, color=SERIES[0], linewidth=1.8, marker="o", markersize=3.5, label="救回的目标事件")
            axes[1].plot(ms, supp, color=SERIES[1], linewidth=1.8, marker="s", markersize=3.5, label="误压的目标事件")
            axes[1].plot(ms, fpr, color=SERIES[2], linewidth=1.8, marker="^", markersize=3.5, label="净压掉的误报")
            axes[1].set_title(s["name"], fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, labelcolor=INK)
    axes[1].legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_robustness(man, base, path, notes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rob = [(n, p) for n, p in robustness_rows(man, base) if any(x is not None for x in p)]
    if not rob:
        notes.append("图 robustness 缺数据")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _axes(axes[0], "注入的背景噪声（Hz / 像素）", "IoU（test）")
    _axes(axes[1], "注入的背景噪声（Hz / 像素）", "Fa（对数轴）")
    marks = "osD^v"
    for i, (name, pts) in enumerate(rob):
        col = SERIES[i % len(SERIES)]
        xs = [h for h, p in zip(HZ, pts) if p]
        axes[0].plot(xs, [p["iou"] for p in pts if p], color=col, linewidth=1.8, marker=marks[i % 5], markersize=4,
                     label=name)
        axes[1].plot(xs, [p["fa"] for p in pts if p], color=col, linewidth=1.8, marker=marks[i % 5], markersize=4,
                     label=name)
        last = [p for p in pts if p][-1]
        axes[0].annotate(name, (xs[-1], last["iou"]), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=7, frameon=False, labelcolor=INK)
    axes[1].legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_wait(man, base, path, notes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    an = loads(man.get("policies_analysis"), base)
    if not an:
        notes.append("图 wait 缺数据")
        return
    w = an[0]["wait_vs_context"]
    names = {"normal": "普通录像", "dense": "极密录像", "first_beat": "第一拍事件", "tracked": "已跟上的事件",
             "gap": "断续漏检处"}
    groups = [("按录像", [(names[k], w["by_group"][k]["mean_wait"]) for k in ("normal", "dense") if k in w["by_group"]]),
              ("按目标阶段", [(names[k], w["by_phase"][k]["mean_wait"]) for k in ("first_beat", "tracked", "gap")
                          if k in w["by_phase"]])]
    fig, ax = plt.subplots(figsize=(5.4, 2.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _axes(ax, "", "自适应提交的平均等待（窗）")
    xs, labels, x = [], [], 0.0
    top = max(v for _, items in groups for _, v in items) if any(items for _, items in groups) else 1.0
    for g, (title, items) in enumerate(groups):
        start = x
        for name, v in items:
            ax.bar([x], [v], width=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=1.0)
            ax.annotate("%.2f" % v, (x, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5,
                        color=INK)
            xs.append(x)
            labels.append(name)
            x += 1.0
        ax.annotate(title, ((start + x - 1.0) / 2.0, top * 1.12), ha="center", fontsize=8, color=INK2,
                    annotation_clip=False)
        x += 0.6
    ax.set_ylim(0, top * 1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--base", default=".", help="directory the manifest paths are relative to")
    args = parser.parse_args(argv)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with open(args.manifest, encoding="utf-8") as fh:
        man = json.load(fh)
    os.makedirs(args.out, exist_ok=True)
    try:                                                    # CJK labels: use a font that has them when available
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except ImportError:
        pass
    notes = []
    parts = [table_main(man, args.base, notes), table_controls(man, args.base, notes),
             table_policies(man, args.base, notes), table_module2(man, args.base, notes),
             table_window(man, args.base, notes)]
    for fn, name in ((fig_teaser, "fig_teaser.png"), (fig_frontier, "fig_frontier.png"),
                     (fig_info_arrival, "fig_info_arrival.png"),
                     (fig_robustness, "fig_robustness.png"), (fig_wait, "fig_wait.png")):
        try:
            fn(man, args.base, os.path.join(args.out, name), notes)
        except ImportError as error:
            notes.append("%s 未画：%s" % (name, error))
    text = "\n\n".join(p for p in parts if p)
    if notes:
        text += "\n\n缺的输入：\n" + "\n".join("- " + n for n in notes)
    with open(os.path.join(args.out, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    print("PAPER TABLES FINISHED:", args.out, flush=True)


if __name__ == "__main__":
    main()
