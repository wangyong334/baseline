"""把原论文 EV-SpSegNet（离线 3D 稀疏卷积 ANN）改成因果滑动窗口推理：离线 ANN 要做到流式同等延迟，要付出什么代价。

做法（除了每次能看到的事件范围，其余与原 test.py / tools/dump_baseline_predictions.py 完全相同）：
    对每条序列，每隔 stride（默认 50 ms）在时刻 t_end 前向一次，输入只含 [t_end - context, t_end) 内的事件；
    坐标与特征（含按整段 8 s 归一化的时间 t/T）一律原样保留、只做截取，所以唯一的变量是"能看到多长的历史"；
    只保留最新 stride 内（[t_end - stride, t_end)）事件的预测。每个事件恰好被预测一次；输入里没有任何 t >= t_end 的事件。
    context = 8000 即"因果全历史"（每次看到从序列开头到现在的全部事件）；
    context = stride = 8000 即原来的离线单次前向（用来核对：IoU 应与原 test.py 相同，K5 s37 test 为 0.8143）。
输出（每个 context 一个目录，字段与 dump_baseline_predictions.py 相同，可直接交给 tools/sweep_threshold.py）:
    <out-root>/ctx<context>_s<stride>_<split>/<序列名>.npz
    <out-root>/summary_<split>.json：每个 context 的 IoU / ACC / Pd / Fa（原评估函数，阈值 0.9）、每 8 s 前向次数、
        平均每次输入事件数、单次前向耗时；--count-ops 时另给每 8 s 的 MAC 与能耗（与 tools/baseline_energy.py 同一计数器）。
注意：基线是在完整 8 s 序列上训练的，短上下文属于"不重训、直接截断"的设定，掉点里有一部分来自分布偏移，报告时要说明。

用法（服务器，单卡；先跑第一行核对与离线结果一致，再跑正式的上下文扫描）:
    CUDA_VISIBLE_DEVICES=0 python tools/sliding_baseline.py --config configs/evisseg_evuav_baseline_v2_repolr.yaml \
        --checkpoint log/baseline_k5_repolr_seed37/best_iou_seed37.pt --split test \
        --contexts-ms 8000 --stride-ms 8000 --out-root log/sliding_k5_check
    CUDA_VISIBLE_DEVICES=0 python tools/sliding_baseline.py --config configs/evisseg_evuav_baseline_v2_repolr.yaml \
        --checkpoint log/baseline_k5_repolr_seed37/best_iou_seed37.pt --split test \
        --contexts-ms 250 500 1000 2000 8000 --stride-ms 50 --count-ops --out-root log/sliding_k5
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAC_PJ = 4.6


def parse_args():
    """解析本脚本自己的参数。必须在导入原仓库模块之前完成，因为 configs/configs.py 在导入时会解析命令行。"""
    parser = argparse.ArgumentParser(description="EV-SpSegNet 的因果滑动窗口推理")
    parser.add_argument("--config", required=True, help="原基线的 YAML 配置")
    parser.add_argument("--checkpoint", required=True, help="原基线的权重文件")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--contexts-ms", type=int, nargs="+", default=[250, 500, 1000, 2000, 8000],
                        help="每次前向能看到的历史长度（毫秒），每个值单独导出一个目录")
    parser.add_argument("--stride-ms", type=int, default=50, help="每隔多久输出一次（= 输出延迟的上限）")
    parser.add_argument("--count-ops", action="store_true", help="逐次统计实际连接数的 MAC（约慢一倍）")
    parser.add_argument("--max-sequences", type=int, default=0, help="只跑前几条序列（0 = 全部，调试用）")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有的导出目录")
    return parser.parse_args()


def window_schedule(t, context_ms, stride_ms):
    """因果滑窗的调度：返回 [(t_end, 输入事件下标, 其中要保留预测的位置)]，没有新事件的窗口直接跳过。

    输入: t 为事件时间（毫秒，非负整数，任意顺序）。
    第 w 个窗口 t_end = (w+1)*stride；输入事件 = t 在 [max(0, t_end - context), t_end) 内（按时间稳定排序）；
    保留 = 其中 t 在 [t_end - stride, t_end) 内的那一段（排序后恰好是最后一段）。
    """
    context_ms, stride_ms = int(context_ms), int(stride_ms)
    if stride_ms < 1 or context_ms < stride_ms:
        raise ValueError("需要 stride >= 1 且 context >= stride（否则最新 stride 内的事件不全在输入里）")
    t = np.asarray(t, dtype=np.int64)
    if t.size == 0:
        return []
    if int(t.min()) < 0:
        raise ValueError("事件时间不能为负")
    order = np.argsort(t, kind="stable")
    ts = t[order]
    windows = []
    n_windows = int(ts[-1]) // stride_ms + 1
    for w in range(n_windows):
        t_end = (w + 1) * stride_ms
        keep_lo = np.searchsorted(ts, t_end - stride_ms, side="left")
        hi = np.searchsorted(ts, t_end, side="left")
        if hi == keep_lo:
            continue
        ctx_lo = np.searchsorted(ts, max(0, t_end - context_ms), side="left")
        windows.append((t_end, order[ctx_lo:hi], np.arange(keep_lo - ctx_lo, hi - ctx_lo)))
    return windows


def check_coverage(n_events, windows):
    """断言每个事件恰好被保留一次（漏掉或重复都说明调度写错了）。"""
    hits = np.zeros(int(n_events), dtype=np.int64)
    for _, ctx, keep in windows:
        np.add.at(hits, ctx[keep], 1)
    if not np.all(hits == 1):
        raise AssertionError("滑窗覆盖错误：%d 个事件未预测，%d 个事件重复预测"
                             % (int(np.sum(hits == 0)), int(np.sum(hits > 1))))


def main():
    """入口：加载原网络 -> 逐序列、逐 context 滑窗推理 -> 导出 NPZ -> 原评估函数算指标 -> 写 summary。"""
    args = parse_args()
    for path in (args.config, args.checkpoint):
        if not os.path.isfile(path):
            raise SystemExit("找不到文件: %s" % path)
    sys.argv = [sys.argv[0], "--config", args.config]

    import time
    import torch
    from configs.configs import cfg
    from dataset.ev_uav import EvUAV
    from model.evspsegnet import evspsegnet
    from utils.eval import evalute

    net = evspsegnet(cfg).eval().cuda()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cuda"))
    counter, summarize_records = None, None
    if args.count_ops:
        import spconv.pytorch as spconv
        from model.basemodel import SEModule
        from tools.baseline_energy import OperationCounter, summarize_records
        counter = OperationCounter(net, spconv, SEModule)
    dataset = EvUAV(cfg, mode=args.split)
    n_seq = len(dataset) if not args.max_sequences else min(len(dataset), int(args.max_sequences))
    contexts = sorted(set(int(c) for c in args.contexts_ms))
    dump_dirs = {c: os.path.join(args.out_root, "ctx%d_s%d_%s" % (c, args.stride_ms, args.split)) for c in contexts}
    for c, d in dump_dirs.items():
        if os.path.isdir(d) and os.listdir(d) and not args.overwrite:
            raise SystemExit("导出目录已有文件: %s（加 --overwrite 覆盖）" % d)
        os.makedirs(d, exist_ok=True)
    evaluators = {c: evalute(cfg) for c in contexts}
    stats = {c: {"passes": [], "events_per_pass": [], "mac": [], "collate_ms": [], "forward_ms": []} for c in contexts}

    with torch.no_grad():
        for i in range(n_seq):
            ev = dataset[i]
            name = dataset.file_list[i]
            ev_loc, evs_norm = np.asarray(ev["ev_loc"]), np.asarray(ev["evs_norm"])
            seg_label, idx = np.asarray(ev["seg_label"]), np.asarray(ev["idx"])
            t = np.round(ev_loc[:, 2]).astype(np.int64)
            locs = np.hstack([np.zeros((ev_loc.shape[0], 1)), ev_loc]).astype(np.int64)
            for c in contexts:
                windows = window_schedule(t, c, args.stride_ms)
                check_coverage(t.shape[0], windows)
                prob = np.full(t.shape[0], np.nan, dtype=np.float32)
                mac, collate_ms, forward_ms, sizes = 0, 0.0, 0.0, []
                for t_end, ctx, keep in windows:
                    assert int(t[ctx].max()) < t_end, "输入里出现了未来事件"
                    sub = {"ev_loc": ev_loc[ctx], "evs_norm": evs_norm[ctx], "seg_label": seg_label[ctx], "idx": idx[ctx]}
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    batch = dataset.custom_collate([sub])
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    if counter is not None:
                        counter.pop()
                    preds, _ = net(batch["voxel_ev"])
                    preds = preds[batch["p2v_map"].long().cuda()].squeeze(-1)
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    if counter is not None:
                        mac += summarize_records(counter.pop())["mac"]
                    prob[ctx[keep]] = preds.cpu().numpy()[keep]
                    collate_ms += 1e3 * (t1 - t0)
                    forward_ms += 1e3 * (t2 - t1)
                    sizes.append(int(ctx.shape[0]))
                if np.isnan(prob).any():
                    raise AssertionError("%s context=%d 有事件没有得到预测" % (name, c))
                np.savez(os.path.join(dump_dirs[c], name), locs=locs, labels=seg_label.astype(np.float32),
                         probabilities=prob, target_id=idx.astype(np.float64))
                e = evaluators[c]
                label_t = torch.from_numpy(seg_label.astype(np.float32))
                pred_t = torch.from_numpy(prob)
                e.matches[str(i)] = {"seg_pred": pred_t, "seg_gt": label_t.cuda()}
                locs_t = torch.from_numpy(locs).float()
                e.roc_update(locs_t[:, 3], pred_t, idx, label_t, locs_t)
                s = stats[c]
                s["passes"].append(len(windows))
                s["events_per_pass"].append(float(np.mean(sizes)) if sizes else 0.0)
                s["mac"].append(mac)
                s["collate_ms"].append(collate_ms / max(len(windows), 1))
                s["forward_ms"].append(forward_ms / max(len(windows), 1))
                print("[%d/%d] %s context %5d ms | 前向 %3d 次 | 每次平均 %.0f 个事件 | %.1f ms/次%s" % (
                    i + 1, n_seq, name, c, len(windows), s["events_per_pass"][-1],
                    s["collate_ms"][-1] + s["forward_ms"][-1],
                    (" | MAC %.3e" % mac) if counter is not None else ""), flush=True)

    summary = {"split": args.split, "checkpoint": args.checkpoint, "config": args.config, "stride_ms": args.stride_ms,
               "sequences": n_seq, "constants_pj": {"mac": MAC_PJ}, "contexts": [],
               "note": "因果滑窗：输入只含 [t_end-context, t_end) 的事件，只保留最新 stride 的预测；基线未针对短上下文重训；"
                       "能耗按实际连接数计 MAC（与 tools/baseline_energy.py 同口径），未计体素化"}
    for c in contexts:
        e, s = evaluators[c], stats[c]
        iou = float(e.evaluate_semantic_segmantation_miou())
        acc = float(e.evaluate_semantic_segmantation_accuracy())
        pd, fa = e.cal_roc()
        row = {"context_ms": c, "dump_dir": dump_dirs[c], "iou": iou, "acc": acc, "pd": float(pd), "fa": float(fa),
               "passes_per_8s": float(np.mean(s["passes"])), "events_per_pass": float(np.mean(s["events_per_pass"])),
               "ms_per_pass": float(np.mean(s["collate_ms"]) + np.mean(s["forward_ms"])),
               "forward_ms_per_pass": float(np.mean(s["forward_ms"]))}
        if counter is not None:
            row["mac_per_8s"] = float(np.mean(s["mac"]))
            row["energy_mj_per_8s"] = row["mac_per_8s"] * MAC_PJ * 1e-9
        summary["contexts"].append(row)
        print("context %5d ms | IoU %.4f ACC %.4f Pd %.4f Fa %.2e | 每 8 s 前向 %.0f 次%s" % (
            c, iou, acc, pd, fa, row["passes_per_8s"],
            (" | %.1f mJ/8s" % row["energy_mj_per_8s"]) if counter is not None else ""), flush=True)
    out = os.path.join(args.out_root, "summary_%s.json" % args.split)
    with open(out, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print("报告:", out)
    print("SLIDING BASELINE FINISHED", flush=True)


if __name__ == "__main__":
    main()
