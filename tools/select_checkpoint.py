"""Checkpoint selection on val (FINAL_PLAN_20261104.md §6). The same rule for backbones and decision heads.

Rules:
    guarded (default; pre-registered before the final round):
        1. per epoch, from metrics.jsonl: official pooled val IoU, normal-group IoU, dense-group IoU, macro IoU
           (groups: utils/recording_groups.py, physical shot-noise rule; no test data is read);
        2. guardrail: keep epochs whose normal-group IoU >= (best normal-group IoU over all epochs) - delta
           (delta = 0.005 is the tolerance for "no regression on normal recordings");
        3. among those, take the highest official val IoU (ties -> the later epoch).
        Runs without per-sequence counts (legacy) fall back to the best val IoU.
        Lesson behind it: the pooled val IoU is dominated by val_023 (5990 target events), and its spikes picked bad
        epochs: E1 21 (test 0.8533 vs 0.8836), E3b 24 (0.8668 vs 0.8788), V5@25 18 (val_023 0.42 vs 0.32).
    plain   (--rule plain or SELECT_RULE=plain): the epoch with the highest official pooled val IoU; on ties the first
            such epoch, i.e. the epoch of the trainer's own best_val_iou checkpoint.
    last    (--rule last or SELECT_RULE=last; 11-08): the last epoch of the fixed schedule, no selection. For datasets
            without an official val split (EV-Flying: our 4 held-out train recordings are too few to rank epochs;
            they are kept for fitting the decision layer only).
Replica of an earlier run (--reference-metrics <its metrics.jsonl>; plain rule, the reference has no per-sequence
counts): training is deterministic, so a run stopped early that reproduces a complete earlier run bit for bit is
selected on the earlier run's full record. Every common epoch must have the identical official val IoU, and the chosen
epoch must already be in this run (its weights come from <run>/epochs).

Usage:
    python tools/select_checkpoint.py --run log/<name>_seed37 [--data-root <dataset root>] [--copy]
    --copy writes <run>/selected_seed<seed>.pt (from epochs/epoch_XXX.pt, or best / last when they coincide).
Works on train_stream_v1.py runs (val_sequences per epoch) and train_track_head.py runs (val.per_sequence.track).
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.recording_groups import dense_names, group_report  # noqa: E402

DELTA = 0.005
RULES = ("plain", "guarded", "last")


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def epoch_table(records):
    """[(epoch, per-sequence counts or None, official val IoU)] for either trainer's metrics.jsonl."""
    out = []
    for r in records:
        if "val_sequences" in r:                                    # backbone (train_stream_v1.py)
            out.append((int(r["epoch"]), r["val_sequences"], float(r["val_iou"])))
        elif isinstance(r.get("val"), dict) and "track" in r["val"]:  # decision head (train_track_head.py)
            seqs = (r["val"].get("per_sequence") or {}).get("track")
            out.append((int(r["epoch"]), seqs, float(r["val"]["track"]["iou"])))
        elif "val_iou" in r:                                        # legacy backbone run: no per-sequence counts
            out.append((int(r["epoch"]), None, float(r["val_iou"])))
    return out


def select(table, dense=None, delta=DELTA, rule="guarded"):
    """Apply a rule. Returns a dict with the chosen epoch, the plain best-val epoch and per-epoch values."""
    if not table:
        raise ValueError("metrics.jsonl 里没有可用的轮次")
    if rule == "last":
        best = max(v for _, _, v in table)
        first_best = min(e for e, _, v in table if v == best)
        epoch, val = max((e, v) for e, _, v in table)
        return {"rule": "last epoch of the fixed schedule (no official val split; no selection)", "epoch": epoch,
                "best_val_epoch": first_best, "chosen": {"epoch": epoch, "val_iou": val},
                "epochs": [{"epoch": e, "val_iou": v} for e, _, v in table]}
    if rule == "plain":
        best = max(v for _, _, v in table)
        epoch = min(e for e, _, v in table if v == best)            # first epoch reaching it = trainer's best_val_iou
        return {"rule": "max official val IoU (first epoch reaching it)", "epoch": epoch, "best_val_epoch": epoch,
                "chosen": {"epoch": epoch, "val_iou": best}, "epochs": [{"epoch": e, "val_iou": v} for e, _, v in table]}
    if rule != "guarded":
        raise ValueError("未知的选择规则：%s（%s）" % (rule, " / ".join(RULES)))
    best_plain = max(table, key=lambda r: (r[2], r[0]))[0]
    if any(seqs is None for _, seqs, _ in table):
        return {"rule": "best_val_iou (legacy run: no per-sequence val counts)", "epoch": best_plain,
                "best_val_epoch": best_plain, "epochs": [{"epoch": e, "val_iou": v} for e, _, v in table]}
    rows = []
    for epoch, seqs, val in table:
        g = group_report(seqs, dense)
        if g["all"] is not None and abs(g["all"] - val) > 1e-6:
            raise RuntimeError("第 %d 轮：逐录像计数合成的 IoU %.6f 与官方 val IoU %.6f 不一致" % (epoch, g["all"], val))
        rows.append(dict(g, epoch=epoch, val_iou=val))
    normals = [r["normal"] for r in rows if r["normal"] is not None]
    if not normals:                                                 # no normal recording: the guardrail is vacuous
        floor, keep = None, rows
    else:
        floor = max(normals) - float(delta)
        keep = [r for r in rows if r["normal"] is not None and r["normal"] >= floor]
    chosen = max(keep, key=lambda r: (r["val_iou"], r["epoch"]))
    plain = next(r for r in rows if r["epoch"] == best_plain)
    return {"rule": "max official val IoU s.t. normal-group IoU >= best normal - %.4f" % float(delta),
            "delta": float(delta), "normal_floor": floor, "epoch": chosen["epoch"], "chosen": chosen,
            "best_val_epoch": best_plain, "best_val": plain, "candidates": [r["epoch"] for r in keep],
            "epochs": rows}


def check_replica(records, reference):
    """Number of epochs compared; raises unless every epoch of the run has the reference's val IoU bit for bit."""
    ref = {e: v for e, _, v in epoch_table(reference)}
    run = epoch_table(records)
    if not run:
        raise ValueError("本运行还没有完成的轮次")
    for epoch, _, val in run:
        if epoch not in ref:
            raise RuntimeError("第 %d 轮不在参考运行的记录里" % epoch)
        if val != ref[epoch]:
            raise RuntimeError("第 %d 轮 val IoU %r 与参考运行的 %r 不同：不是逐位复现，不能借用参考运行的记录" % (
                epoch, val, ref[epoch]))
    return len(run)


def run_seed(run, records):
    """Seed of a run: backbone records carry it; head runs keep it in run_config.json."""
    if "seed" in records[-1]:
        return int(records[-1]["seed"])
    for name in ("run_config.json", "run_config_train.json"):
        path = run / name
        if path.exists():
            cfg = json.loads(path.read_text(encoding="utf-8"))
            cfg = cfg.get("config", cfg)
            if "seed" in cfg:
                return int(cfg["seed"])
    return 37


def checkpoint_for(run, epoch, record_last, seed, best_epoch):
    """Path of the weights of `epoch`: epochs/epoch_XXX.pt, else best / last when they are that epoch."""
    path = run / "epochs" / ("epoch_%03d.pt" % epoch)
    if path.exists():
        return path
    if epoch == best_epoch and (run / ("best_val_iou_seed%d.pt" % seed)).exists():
        return run / ("best_val_iou_seed%d.pt" % seed)
    if epoch == record_last and (run / "last.pt").exists():
        return run / "last.pt"
    raise FileNotFoundError("第 %d 轮没有权重（训练时要开 save_epochs: true）：%s" % (epoch, path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="训练输出目录（含 metrics.jsonl）")
    parser.add_argument("--rule", help="guarded = 事先定的普通组护栏（默认）；plain = 只看 val IoU；"
                                       "环境变量 SELECT_RULE 改默认；给了 --reference-metrics 时默认 plain")
    parser.add_argument("--data-root", help="数据根目录（只有 guarded 要用：读 train / val 的事件数，定极密分组）")
    parser.add_argument("--reference-metrics", help="本运行是这次完整旧运行的逐位复现、提前停下：核对每一轮后按它的完整记录选")
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--height", type=int, default=260, help="传感器高（像素），背景率换算 Hz/px 用")
    parser.add_argument("--width", type=int, default=346, help="传感器宽（像素）")
    parser.add_argument("--copy", action="store_true", help="把选中的轮次复制为 selected_seed<seed>.pt")
    args = parser.parse_args(argv)
    args.rule = args.rule or os.environ.get("SELECT_RULE") or ("plain" if args.reference_metrics else "guarded")
    if args.rule not in RULES:
        parser.error("--rule / SELECT_RULE 只能是 %s" % " / ".join(RULES))
    run = Path(args.run)
    if not (run / "metrics.jsonl").exists():           # frozen run kept without its training log (the 10-23 V5)
        frozen = sorted(run.glob("best_val_iou_seed*.pt"))
        if not frozen:
            raise FileNotFoundError("%s 既没有 metrics.jsonl，也没有 best_val_iou_seed*.pt" % run)
        src = frozen[0]
        seed = int(re.search(r"seed(\d+)", src.name).group(1))
        result = {"rule": "frozen checkpoint without metrics.jsonl: used as is (its own best-val selection)",
                  "epoch": None, "best_val_epoch": None, "frozen": True, "run": str(run), "seed": seed, "epochs": []}
        if args.copy:
            dst = run / ("selected_seed%d.pt" % seed)
            shutil.copyfile(str(src), str(dst))
            result.update(selected_checkpoint=str(dst), source_checkpoint=str(src))
        (run / "selection.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print("没有 metrics.jsonl：直接用冻结的权重 %s（它自己按 val 选的轮次）" % src.name)
        print("SELECTION FINISHED:", run / "selection.json", flush=True)
        return
    records = read_records(run / "metrics.jsonl")
    last = int(records[-1]["epoch"])
    names = None
    if args.reference_metrics:
        if args.rule != "plain":
            parser.error("--reference-metrics 只用 plain 规则（参考运行没有逐录像计数）")
        reference = read_records(args.reference_metrics)
        compared = check_replica(records, reference)
        result = select(epoch_table(reference), rule="plain")
        if result["epoch"] > last:
            raise RuntimeError("参考运行选中第 %d 轮，本运行只训到第 %d 轮：还没有这一轮的权重" % (result["epoch"], last))
        result.update(reference=str(args.reference_metrics), replica_epochs_compared=compared,
                      reference_epochs=len(reference))
    elif args.rule == "guarded":
        if not args.data_root:
            parser.error("guarded 规则需要 --data-root")
        names = dense_names(args.data_root, ("val",), args.height, args.width)
        result = select(epoch_table(records), names["val"], args.delta, rule="guarded")
        result.update(dense_rule="shot-noise regime (utils/recording_groups.py)", val_dense=names["val"])
    else:
        result = select(epoch_table(records), rule=args.rule)
    seed = run_seed(run, records)
    result.update(run=str(run), seed=seed)
    if args.copy:
        src = checkpoint_for(run, result["epoch"], last, seed, result["best_val_epoch"])
        dst = run / ("selected_seed%d.pt" % seed)
        shutil.copyfile(str(src), str(dst))
        result["selected_checkpoint"], result["source_checkpoint"] = str(dst), str(src)
    (run / "selection.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    ch = result.get("chosen")
    print("选择规则：%s" % result["rule"])
    if result.get("reference"):
        print("参考运行：%s（本运行的 %d 轮 val IoU 与它逐位相同；按它完整的 %d 轮记录选）" % (
            result["reference"], result["replica_epochs_compared"], result["reference_epochs"]))
    if names is not None:
        print("极密分组（暗光散粒噪声规则）：%s" % (" ".join(names["val"]) or "无"))
    if ch and "normal" in ch:
        bv = result["best_val"]
        fmt = lambda v: "-" if v is None else "%.4f" % v                               # noqa: E731
        print("选中第 %d 轮：val %.4f | 普通 %s | 极密 %s | 宏平均 %s（普通组下限 %s，候选 %d 轮）" % (
            ch["epoch"], ch["val_iou"], fmt(ch["normal"]), fmt(ch["dense"]), fmt(ch["macro"]),
            fmt(result["normal_floor"]), len(result["candidates"])))
        print("只看 val IoU 会选第 %d 轮：val %.4f | 普通 %s | 极密 %s" % (
            bv["epoch"], bv["val_iou"], fmt(bv["normal"]), fmt(bv["dense"])))
    elif ch:
        print("选中第 %d 轮：val %.4f（记录共 %d 轮）" % (ch["epoch"], ch["val_iou"], len(result["epochs"])))
    else:
        print("旧运行没有逐录像计数：退回 val IoU 最好的第 %d 轮" % result["epoch"])
    if args.copy:
        print("已复制：%s -> %s" % (result["source_checkpoint"], result["selected_checkpoint"]))
    print("SELECTION FINISHED:", run / "selection.json", flush=True)


if __name__ == "__main__":
    main()
