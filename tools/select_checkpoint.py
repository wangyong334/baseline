"""Guarded checkpoint selection on val (FINAL_PLAN_20261104.md §6; pre-registered before the final round).

Rule (the same for backbones and decision heads):
    1. per epoch, from metrics.jsonl: official pooled val IoU, normal-group IoU, dense-group IoU, macro IoU
       (groups: utils/recording_groups.py, physical shot-noise rule; no test data is read);
    2. guardrail: keep epochs whose normal-group IoU >= (best normal-group IoU over all epochs) - delta
       (delta = 0.005 is the tolerance for "no regression on normal recordings");
    3. among those, take the highest official val IoU (ties -> the later epoch).
Lesson behind it: E3b's best-val epoch (24) was a val_023 spike while the normal recordings dropped 0.0085 below V5;
its test IoU was 0.8668 against 0.8821 at the last epoch. Plain best-val-IoU selection is reported alongside.

Usage:
    python tools/select_checkpoint.py --run log/<name>_seed37 --data-root <dataset root> [--copy]
    --copy writes <run>/selected_seed<seed>.pt (from epochs/epoch_XXX.pt, or best / last when they coincide).
Works on train_stream_v1.py runs (val_sequences per epoch) and train_track_head.py runs (val.per_sequence.track).
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.recording_groups import dense_names, group_report  # noqa: E402

DELTA = 0.005


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


def select(table, dense, delta=DELTA):
    """Apply the guarded rule. Returns a dict with the chosen epoch, the plain best-val epoch and per-epoch groups."""
    if not table:
        raise ValueError("metrics.jsonl 里没有可用的轮次")
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
    parser.add_argument("--data-root", required=True, help="数据根目录（只读 train / val 的事件数，定极密分组）")
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--height", type=int, default=260, help="传感器高（像素），背景率换算 Hz/px 用")
    parser.add_argument("--width", type=int, default=346, help="传感器宽（像素）")
    parser.add_argument("--copy", action="store_true", help="把选中的轮次复制为 selected_seed<seed>.pt")
    args = parser.parse_args(argv)
    run = Path(args.run)
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    names = dense_names(args.data_root, ("val",), args.height, args.width)
    result = select(epoch_table(records), names["val"], args.delta)
    seed = run_seed(run, records)
    result.update(dense_rule="shot-noise regime (utils/recording_groups.py)", val_dense=names["val"], run=str(run),
                  seed=seed)
    if args.copy:
        src = checkpoint_for(run, result["epoch"], int(records[-1]["epoch"]), seed, result["best_val_epoch"])
        dst = run / ("selected_seed%d.pt" % seed)
        shutil.copyfile(str(src), str(dst))
        result["selected_checkpoint"], result["source_checkpoint"] = str(dst), str(src)
    (run / "selection.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    ch = result.get("chosen")
    print("选择规则：%s" % result["rule"])
    print("极密分组（暗光散粒噪声规则）：%s" % (" ".join(names["val"]) or "无"))
    if ch:
        bv = result["best_val"]
        fmt = lambda v: "-" if v is None else "%.4f" % v                               # noqa: E731
        print("选中第 %d 轮：val %.4f | 普通 %s | 极密 %s | 宏平均 %s（普通组下限 %s，候选 %d 轮）" % (
            ch["epoch"], ch["val_iou"], fmt(ch["normal"]), fmt(ch["dense"]), fmt(ch["macro"]),
            fmt(result["normal_floor"]), len(result["candidates"])))
        print("只看 val IoU 会选第 %d 轮：val %.4f | 普通 %s | 极密 %s" % (
            bv["epoch"], bv["val_iou"], fmt(bv["normal"]), fmt(bv["dense"])))
    else:
        print("旧运行没有逐录像计数：退回 val IoU 最好的第 %d 轮" % result["epoch"])
    if args.copy:
        print("已复制：%s -> %s" % (result["source_checkpoint"], result["selected_checkpoint"]))
    print("SELECTION FINISHED:", run / "selection.json", flush=True)


if __name__ == "__main__":
    main()
