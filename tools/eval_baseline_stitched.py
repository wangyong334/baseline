"""K5 / 原版 EV-SpSegNet 基线在整段长录像上的评测：按 8 s 块推理，逐事件拼回整段，再用与流式 V1 相同的代码算指标。

协议（10-08 用户定）：离线方法按自身设计每 8 s 一块推理再拼回整段；我们的方法整段连续处理。
切块规则与 tools/make_blocks.py 相同（直接调用它的函数），所以不需要单独的块文件：从整段录像现场切块。
指标与 train_stream_v1.evaluate_split 同一口径：
    IoU / ACC  utils/eval.py 的 evaluate_iou_and_accuracy（全部录像拼接，阈值 0.9）
    Pd / Fa    utils/stream_metrics.FastRoc（与原 roc_update / cal_roc 逐位等价，传感器尺寸取 --height/--width，
               在整段录像上计帧，与 V1 完全相同；原 utils/eval.py 的 Fa 分母写死 346x260，不能用于其他传感器）
    逐窗 TP/FP/FN（窗号对齐累加，与 V1 的 per_window 相同，可直接比较随时间的变化）
    首次检出：判定与 V1 相同；离线延迟 = 检出窗所在 8 s 块的结束时刻 - 目标第一个事件时刻（离线方法要等块结束）
结果写到 checkpoint 同目录 eval_stitched_<split>_<checkpoint 名>.json；--dump-dir 时另存逐事件概率（与 V1 --dump-dir 字段相同）。

用法（单卡，在仓库根目录）:
    CUDA_VISIBLE_DEVICES=0 python tools/eval_baseline_stitched.py --config configs/evisseg_evflying_x4_k5.yaml \
        --checkpoint log/k5_evflying_x4_seed37/best_iou_seed37.pt --split test \
        --recordings /path/evflying_x4 --height 180 --width 320 --k5-width 320 --k5-height 192
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """必须在导入原仓库模块之前解析：configs/configs.py 在 import 时会解析命令行（只认 --config）。"""
    parser = argparse.ArgumentParser(description="stitched whole-recording evaluation of the offline baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--recordings", required=True, help="整段录像根目录（含 val/ test/）")
    parser.add_argument("--height", type=int, required=True, help="传感器有效高度（Fa 分母）")
    parser.add_argument("--width", type=int, required=True, help="传感器有效宽度（Fa 分母）")
    parser.add_argument("--k5-width", type=int, required=True, help="evs_norm 的 x 归一化（须与训练块相同）")
    parser.add_argument("--k5-height", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--window-ms", type=int, default=50)
    parser.add_argument("--dump-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.argv = [sys.argv[0], "--config", args.config]

    from types import SimpleNamespace

    import numpy as np
    import torch
    from configs.configs import cfg
    from dataset.basedataset import BaseDataLoader
    from model.evspsegnet import evspsegnet
    from tools.make_blocks import BLOCK_MS, block_arrays, split_blocks
    from utils.eval import evalute
    from utils.stream_metrics import (FastRoc, first_detection_latencies, summarize_latencies,
                                      window_confusion)

    out_path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "eval_stitched_%s_%s.json" % (
        args.split, os.path.splitext(os.path.basename(args.checkpoint))[0]))
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit("结果已存在: %s（加 --overwrite 覆盖）" % out_path)
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
    torch.cuda.set_device(0)
    net = evspsegnet(cfg).cuda(0).eval()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cuda:0"))
    evaluator = evalute(SimpleNamespace(roc=False))
    roc = FastRoc(cfg.pd_detT, cfg.correct_thresh, args.height, args.width)
    split_dir = os.path.join(args.recordings, args.split)
    names = sorted(n for n in os.listdir(split_dir) if n.endswith(".npz"))
    confusion_sum = np.zeros((0, 4), dtype=np.int64)
    records, per_recording, block_seconds, n_blocks = [], [], 0.0, 0
    t_start = time.time()
    with torch.no_grad():
        for i, name in enumerate(names):
            with np.load(os.path.join(split_dir, name)) as z:
                ev_loc, evs_norm = np.asarray(z["ev_loc"]).astype(np.int64), np.asarray(z["evs_norm"])
            x, y, t = ev_loc[:, 0], ev_loc[:, 1], ev_loc[:, 2]
            label = evs_norm[:, 4].astype(np.float32)
            target_id = evs_norm[:, 5].astype(np.float64)
            probs = np.full(ev_loc.shape[0], np.nan, dtype=np.float32)
            for b, sl in split_blocks(ev_loc, evs_norm):
                loc, norm = block_arrays(ev_loc, evs_norm, b, sl, args.k5_width, args.k5_height)
                sample = {"ev_loc": loc, "evs_norm": norm[:, 0:4], "seg_label": norm[:, 4], "idx": norm[:, 5]}
                batch = BaseDataLoader.custom_collate([sample])
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                preds, _ = net(batch["voxel_ev"])
                block_probs = preds[batch["p2v_map"].long().cuda(0)].reshape(-1).cpu().numpy()
                torch.cuda.synchronize()
                block_seconds += time.perf_counter() - t0
                n_blocks += 1
                if block_probs.shape[0] != sl.stop - sl.start:
                    raise RuntimeError("%s 块 %d 的预测数与事件数不一致" % (name, b))
                probs[sl] = block_probs
                del batch, preds
            if np.isnan(probs).any():
                raise RuntimeError("%s 有事件没有拿到预测" % name)
            evaluator.matches[str(i)] = {"seg_pred": torch.from_numpy(probs), "seg_gt": torch.from_numpy(label)}
            roc.update(t, probs, target_id, label, x, y, args.threshold)
            windows = t // args.window_ms
            n_win = int(windows.max()) + 1
            confusion = np.zeros((n_win, 4), dtype=np.int64)
            bounds = np.searchsorted(windows, np.arange(n_win + 1))
            for k in range(n_win):
                a, c = bounds[k], bounds[k + 1]
                confusion[k] = window_confusion(probs[a:c], label[a:c], args.threshold)
            if n_win > confusion_sum.shape[0]:
                confusion_sum = np.concatenate([confusion_sum, np.zeros((n_win - confusion_sum.shape[0], 4), np.int64)])
            confusion_sum[:n_win] += confusion
            tp, fp, fn = (int(confusion[:, j].sum()) for j in range(3))
            per_recording.append({"recording": name, "events": int(t.size), "tp": tp, "fp": fp, "fn": fn,
                                  "iou": tp / float(max(tp + fp + fn, 1))})
            for r in first_detection_latencies(t, label, target_id, probs, args.window_ms, args.threshold,
                                               cfg.correct_thresh, 0.0):
                if r["detect_window"] is not None:
                    release = ((r["detect_window"] * args.window_ms) // BLOCK_MS + 1) * BLOCK_MS
                    r["latency_ms"] = float(release - r["t_first_ms"])
                records.append(r)
            if args.dump_dir:
                zeros = np.zeros(t.size, dtype=np.int64)
                np.savez(os.path.join(args.dump_dir, name), locs=np.stack([zeros, x, y, t], 1),
                         labels=label, probabilities=probs, target_id=target_id)
            print("[%d/%d] %s 事件 %d IoU %.4f" % (i + 1, len(names), name, t.size, per_recording[-1]["iou"]), flush=True)
    iou, acc = evaluator.evaluate_iou_and_accuracy(thresh=args.threshold)
    pd, fa = roc.result()
    result = {"checkpoint": args.checkpoint, "config": args.config, "split": args.split,
              "down_kernel": getattr(cfg, "down_kernel", [3, 3, 5]), "threshold": args.threshold,
              "protocol": "offline 8 s blocks (tools/make_blocks.py rule), stitched back to whole recordings",
              "sensor": [args.width, args.height], "k5_norm": [args.k5_width, args.k5_height],
              "n_sequences": len(names), "n_blocks": n_blocks,
              "iou": float(iou), "acc": float(acc), "pd": float(pd), "fa": float(fa),
              "latency_offline": summarize_latencies(records),
              "net_ms_per_block_mean": 1e3 * block_seconds / max(n_blocks, 1),
              "per_recording": per_recording,
              "per_window": {k: confusion_sum[:, j].tolist() for j, k in enumerate(("tp", "fp", "fn", "pos"))},
              "seconds": round(time.time() - t_start, 1)}
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, ensure_ascii=False)
    print("[stitched] IoU %.4f ACC %.4f Pd %.4f Fa %.3e | 检出 %s | %d 块，每块 %.0f ms" % (
        iou, acc, pd, fa, result["latency_offline"].get("detection_rate"), n_blocks, result["net_ms_per_block_mean"]))
    print("EVAL FINISHED:", out_path, flush=True)


if __name__ == "__main__":
    main()
