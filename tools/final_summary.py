"""One-page summary of the V9-final runs on the server (Markdown to stdout; FINAL_SERVER_COMMANDS_20261104.md).

Reads what exists under --log-root (default log/final) and --run-root (default run/final):
    backbones  <log>/bb_*_seed<S>/       selection.json, eval_{val,test}_selected_seed<S>[_noise<hz>hz_seed0].json
    heads      <log>/{rcl,b2}_*_seed<S>/ eval_{val,test}_selected_seed<S>[_adaptive|_past][_noise...].json
    analysis   <run>/analysis/<run>_seed<S>_{val,test}.json, b1_on_*.json; backbone_choice_seed<S>.json
Symlinked runs (bb_m2_50, rcl_feature_50_on_m2) are listed once, under their real names, with a marker.
Group IoU (normal / dense) uses the shot-noise rule of utils/recording_groups.py on --data-root (EV-UAV only).
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.recording_groups import dense_names, group_report  # noqa: E402


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def f4(v):
    return "-" if v is None else "%.4f" % v


def groups(seqs, dense):
    if not seqs or dense is None:
        return "-", "-"
    g = group_report(seqs, dense)
    return f4(g["normal"]), f4(g["dense"])


def runs(log_root, prefixes):
    """[(name, seed, dir, link_names)] of real run directories; symlinks pointing at them become markers."""
    links = {}
    for path in glob.glob(os.path.join(log_root, "*_seed*")):
        if os.path.islink(path):
            links.setdefault(os.path.basename(os.path.realpath(path)), []).append(os.path.basename(path))
    out = []
    for path in sorted(glob.glob(os.path.join(log_root, "*_seed*"))):
        base = os.path.basename(path)
        m = re.match(r"(.+)_seed(\d+)$", base)
        if os.path.islink(path) or not m or not os.path.isdir(path) or not base.startswith(prefixes):
            continue
        out.append((m.group(1), int(m.group(2)), path, links.get(base, [])))
    return out


def backbone_table(log_root, dense):
    rows = ["| 骨干 | 选中轮次（只看 val 会选） | val IoU | val 普通 / 极密 | test IoU / ACC / Pd / Fa | test 普通 / 极密 |"
            " 4 Hz/px IoU / Fa | 每窗消息 | 首检中位 ms（含每窗计算） |", "|---|---|---|---|---|---|---|---|---|"]
    for name, seed, path, marks in runs(log_root, ("bb_",)):
        stem = os.path.join(path, "eval_%%s_selected_seed%d%%s.json" % seed)
        sel = load(os.path.join(path, "selection.json")) or {}
        val, test = load(stem % ("val", "")), load(stem % ("test", ""))
        noisy = load(stem % ("test", "_noise4hz_seed0"))
        if val is None and test is None:
            continue
        v, t = (val or {}).get("carry", {}), (test or {}).get("carry", {})
        vn, vd = groups(v.get("per_sequence"), dense.get("val"))
        tn, td = groups(t.get("per_sequence"), dense.get("test"))
        n = (noisy or {}).get("carry", {})
        lat = (t.get("latency") or {}).get("latency_median_ms")
        rows.append("| %s%s | %s（%s） | %s | %s / %s | %s / %s / %s / %s | %s / %s | %s / %s | %s | %s |" % (
            name, " ← %s" % ", ".join(marks) if marks else "", sel.get("epoch", "-"), sel.get("best_val_epoch", "-"),
            f4(v.get("iou")), vn, vd, f4(t.get("iou")), f4(t.get("acc")), f4(t.get("pd")),
            "%.2e" % t["fa"] if "fa" in t else "-", tn, td, f4(n.get("iou")), "%.2e" % n["fa"] if "fa" in n else "-",
            "%.3g" % t["messages_mean"] if t.get("messages_mean") else "-", "%.1f" % lat if lat else "-"))
    return "### 骨干（模块一 / 模块二）\n\n" + "\n".join(rows)


def head_table(log_root, run_root, dense):
    rows = ["| 判决头（骨干） | val 不等 → 固定 D | test 不等 → 固定 D | test 普通 / 极密 | test 自适应 α 0.05 | 目标延迟 均值 / 中位 ms |"
            " test 不给未来 | 第一拍召回 不等 → D | test 救回 / 误压 / 压掉误报 / 新增误报 | 完整模型 4 Hz/px |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, seed, path, marks in runs(log_root, ("rcl_", "b2_")):
        stem = os.path.join(path, "eval_%%s_selected_seed%d%%s.json" % seed)
        val, test = load(stem % ("val", "")), load(stem % ("test", ""))
        if test is None:
            continue
        ada, past = load(stem % ("test", "_adaptive")), load(stem % ("test", "_past"))
        noisy = load(stem % ("test", "_adaptive_noise4hz_seed0"))
        an = load(os.path.join(run_root, "analysis", "%s_seed%d_test.json" % (name, seed)))
        tn, td = groups((test.get("per_sequence") or {}).get("track"), dense.get("test"))
        pol = (an or {}).get("policies", {})
        a05, zero = pol.get("adaptive_0.05", {}), pol.get("zero", {})
        fixed = pol.get("fixed_%d" % an["D"], {}) if an else {}
        fl = fixed.get("flips") or {}
        lat = a05.get("latency_ms") or {}
        base = os.path.basename(os.path.dirname(test.get("base_checkpoint", "")))
        rows.append("| %s%s（%s） | %s → %s | %s → %s | %s / %s | %s | %s / %s | %s | %s → %s | %s / %s / %s / %s | %s |" % (
            name, " ← %s" % ", ".join(marks) if marks else "", base or "-",
            f4((val or {}).get("base", {}).get("iou")), f4((val or {}).get("track", {}).get("iou")),
            f4(test["base"]["iou"]), f4(test["track"]["iou"]), tn, td, f4((ada or {}).get("track", {}).get("iou")),
            "%.1f" % lat["target_mean"] if lat.get("target_mean") is not None else "-",
            "%.1f" % lat["target_median"] if lat.get("target_median") is not None else "-",
            f4((past or {}).get("track", {}).get("iou")),
            "%.3f" % zero["first_beat_recall"] if "first_beat_recall" in zero else "-",
            "%.3f" % fixed["first_beat_recall"] if "first_beat_recall" in fixed else "-",
            sum(fl.get("rescued", {}).values()) if fl else "-", fl.get("suppressed", "-"), fl.get("fp_removed", "-"),
            fl.get("fp_added", "-"), f4((noisy or {}).get("track", {}).get("iou"))))
    return "### 判决头（模块三与对照）\n\n" + "\n".join(rows)


def extras(run_root):
    out = []
    for path in sorted(glob.glob(os.path.join(run_root, "backbone_choice_seed*.json"))):
        c = load(path)
        if c:
            out.append("### 选骨干（%s）\n\n- 选定：**%s**（%s）\n- 模块二主张分支：%s\n%s" % (
                os.path.basename(path), c.get("winner"), c.get("winner_run"), c.get("claim_branch"),
                "\n".join("- " + line for line in c.get("log", []))))
    for path in sorted(glob.glob(os.path.join(run_root, "analysis", "b1_on_*.json"))):
        b = load(path)
        if not b:
            continue
        lines = ["### B1 不学习的延迟确认（%s；参数按 val 选）\n" % os.path.basename(path),
                 "| 等待窗数 | val IoU | test IoU | test 第一拍召回 |", "|---|---|---|---|"]
        for lag, r in sorted(b.get("chosen_on_val", {}).items(), key=lambda kv: int(kv[0])):
            lines.append("| %s | %s | %s | %s |" % (lag, f4(r["val"]["iou"]), f4(r["test"]["iou"]),
                                                    "%.3f" % r["test"]["first_beat_recall"]))
        out.append("\n".join(lines))
    return "\n\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log-root", default="log/final")
    parser.add_argument("--run-root", default="run/final")
    parser.add_argument("--data-root", default=None, help="EV-UAV data root for the normal / dense groups")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    dense = {}
    if args.data_root:
        try:
            dense = dense_names(args.data_root, ("val", "test"))
        except Exception as error:                          # summary only: never fail on the grouping
            print("（分组规则读不了数据：%s）" % error, file=sys.stderr)
    parts = ["# V9-final 结果汇总", "", backbone_table(args.log_root, dense), "",
             head_table(args.log_root, args.run_root, dense), "", extras(args.run_root)]
    print("\n".join(parts))


if __name__ == "__main__":
    main()
