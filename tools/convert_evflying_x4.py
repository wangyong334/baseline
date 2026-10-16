"""EV-Flying -> EV-UAV 格式的 4x4 合并版本（方案 B，10-16 用户确认）。

转换规则（每个事件一一对应，顺序不变，不丢不增）：
    x' = floor(x / 4), y' = floor(y / 4)          1280x720 -> 有效区域 320x180；网络画布 320x184（只在下方补 4 行，
                                                  与 EV-UAV 346x260 -> 352x264 同一规则：补到 8 的倍数）
    t_ms = floor(t_us / 1000)                     与 EV-UAV 一样用整数毫秒（窗内 10 ms 时间仓不受影响）
    label = 1 当且仅当 id != -1                   target_id = id（背景记 0）；类别另存为 cls（0 背景 / 1 鸟 / 2 昆虫）
输出与 EV-UAV NPZ 字段一致（流式读取器只用 ev_loc 与 evs_norm[:,3:6]；K5 读取器用 ev_loc 与 evs_norm[:,0:4]）：
    ev_loc   int64   [N,3]  x', y', t_ms
    evs_norm float32 [N,6]  x'/320, y'/192, t_ms/8192, p, label, target_id
             （前 3 列只给 K5：其体素画布按 32 的倍数取 320x192，时间按 8 s 块归一；整段录像的 t/8192 会大于 1，
               K5 训练前需另切 8 s 块重新归一）
    cls      int8    [N]    额外字段，读取器忽略
EV-UAV 文件里的结构化数组 ev 没有被任何读取器使用，这里不写。

划分（我们的协议，官方 Train/Test 不变）：
    train 17 段：官方 Train 去掉 val 的 4 段     val 4 段：官方 Train 的 3、10、17、21     test 6 段：5、9、13、23、24、26
    不转换 Train/29、Test/28（另加的 FRED 无人机片段，不属于官方 27 段）。

每个文件写完后用项目自己的读取器 dataset/stream_windows.load_npz_events 读回校验；另写数据卡 data_card.json
（逐段事件数、时长、窗数、目标事件占比、目标数、类别计数、背景事件率、各数组内容的 SHA-256）。

用法（在仓库根目录；源目录下是 Train/<id>/<id>.npy 与 Test/<id>/<id>.npy）:
    python tools/convert_evflying_x4.py --src /path/to/evflying --dst /path/to/evflying_x4
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from dataset.stream_windows import load_npz_events  # noqa: E402

FACTOR = 4
SRC_W, SRC_H = 1280, 720
VALID_W, VALID_H = SRC_W // FACTOR, SRC_H // FACTOR          # 320 x 180
CANVAS_W, CANVAS_H = 320, 184                                 # 网络画布：补到 8 的倍数
K5_W, K5_H, K5_T = 320, 192, 8192.0                           # evs_norm 前 3 列的归一化（只给 K5）
WINDOW_MS = 50
SPLITS = {
    "train": [("Train", i) for i in (0, 1, 2, 4, 6, 7, 8, 11, 12, 14, 15, 18, 19, 20, 22, 25, 27)],
    "val": [("Train", i) for i in (3, 10, 17, 21)],
    "test": [("Test", i) for i in (5, 9, 13, 23, 24, 26)],
}
SKIPPED = ["Train/29", "Test/28"]


def sha256_array(a):
    """数组内容（dtype、形状、字节）的 SHA-256，与 NPZ 的压缩方式和时间戳无关，可用于服务器端核对。"""
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def convert_arrays(raw):
    """把原始 [N,6]（x, y, p, t_us, id, class）转换成 (ev_loc, evs_norm, cls)，并检查原始数据的取值。"""
    x, y, p, t, ids, cls = (raw[:, j] for j in range(6))
    for name, v in (("x", x), ("y", y), ("p", p), ("id", ids), ("class", cls)):
        if not np.all(v == np.round(v)):
            raise ValueError("%s 含非整数" % name)
    if x.min() < 0 or x.max() >= SRC_W or y.min() < 0 or y.max() >= SRC_H:
        raise ValueError("坐标越界")
    if not np.isin(p, (0, 1)).all():
        raise ValueError("极性不是 0/1")
    if t.min() < 0 or np.any(np.diff(t) < 0):
        raise ValueError("时间为负或不单调")
    if np.any(ids == 0) or np.any((ids < 0) & (ids != -1)):
        raise ValueError("id 只能是 -1（背景）或 >= 1")
    background = ids == -1
    if np.any(background != (cls == 0)):
        raise ValueError("id 与类别不一致（背景必须 class 0）")
    xs = (x.astype(np.int64) // FACTOR)
    ys = (y.astype(np.int64) // FACTOR)
    t_ms = np.floor(t / 1000.0).astype(np.int64)
    label = (~background).astype(np.float32)
    target_id = np.where(background, 0, ids).astype(np.float32)
    ev_loc = np.stack([xs, ys, t_ms], 1)
    evs_norm = np.stack([xs / float(K5_W), ys / float(K5_H), t_ms / K5_T, p, label, target_id], 1).astype(np.float32)
    return ev_loc, evs_norm, cls.astype(np.int8)


def sequence_card(ev_loc, evs_norm, cls, n_windows):
    """一段录像的统计（写进数据卡）。"""
    label = evs_norm[:, 4] == 1
    ids = evs_norm[:, 5][label]
    duration_s = n_windows * WINDOW_MS / 1000.0
    n_bg = int((~label).sum())
    return {"events": int(ev_loc.shape[0]), "n_windows": int(n_windows), "duration_s": duration_s,
            "t_ms_min": int(ev_loc[:, 2].min()), "t_ms_max": int(ev_loc[:, 2].max()),
            "target_events": int(label.sum()), "target_fraction": float(label.mean()),
            "n_targets": int(np.unique(ids).size),
            "class_events": {str(c): int((cls == c).sum()) for c in (0, 1, 2)},
            "background_rate_per_px_s": n_bg / float(VALID_W * VALID_H * duration_s),
            "x_max": int(ev_loc[:, 0].max()), "y_max": int(ev_loc[:, 1].max())}


def main():
    parser = argparse.ArgumentParser(description="EV-Flying -> 4x4 binned EV-UAV-format NPZ")
    parser.add_argument("--src", required=True, help="EV-Flying 根目录（含 Train/、Test/）")
    parser.add_argument("--dst", required=True, help="输出根目录（写 train/ val/ test/ 与 data_card.json）")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    card = {"source": "EV-Flying (arXiv 2506.04048), official 27 recordings (Train 21 + Test 6); extra FRED clips skipped",
            "skipped": SKIPPED, "binning": FACTOR, "source_resolution": [SRC_W, SRC_H],
            "valid_resolution": [VALID_W, VALID_H], "canvas": [CANVAS_W, CANVAS_H], "window_ms": WINDOW_MS,
            "time_unit": "integer ms (floor of us / 1000)", "label_rule": "label = id != -1; target_id = id (bg 0)",
            "evs_norm_columns": ["x/%d" % K5_W, "y/%d" % K5_H, "t_ms/%d" % int(K5_T), "p", "label", "target_id"],
            "splits": {}, "sequences": {}}
    t_start = time.time()
    for split, items in SPLITS.items():
        out_dir = os.path.join(args.dst, split)
        os.makedirs(out_dir, exist_ok=True)
        card["splits"][split] = []
        for folder, idx in items:
            src = os.path.join(args.src, folder, str(idx), "%d.npy" % idx)
            name = "evflying_%s_%02d.npz" % (folder.lower(), idx)
            dst = os.path.join(out_dir, name)
            if os.path.exists(dst) and not args.overwrite:
                raise RuntimeError("已存在 %s（加 --overwrite 覆盖）" % dst)
            raw = np.load(src, allow_pickle=False)
            if raw.ndim != 2 or raw.shape[1] != 6:
                raise ValueError("%s 形状异常 %s" % (src, raw.shape))
            ev_loc, evs_norm, cls = convert_arrays(raw)
            n_windows = int(math.ceil((int(ev_loc[:, 2].max()) + 1) / float(WINDOW_MS)))
            np.savez(dst, ev_loc=ev_loc, evs_norm=evs_norm, cls=cls)
            # 用项目自己的流式读取器读回：坐标在有效区域内、时间在窗内、p/label 合法
            seq = load_npz_events(dst, VALID_H, VALID_W, WINDOW_MS, n_windows, 5)
            if seq.n_events != raw.shape[0] or int(seq.label.sum()) != int((raw[:, 4] != -1).sum()):
                raise RuntimeError("%s 读回的事件数或目标事件数不一致" % dst)
            if not (np.array_equal(seq.x, raw[:, 0].astype(np.int64) // FACTOR)
                    and np.array_equal(seq.y, raw[:, 1].astype(np.int64) // FACTOR)):
                raise RuntimeError("%s 读回的坐标与 floor(原坐标/4) 不一致" % dst)
            entry = sequence_card(ev_loc, evs_norm, cls, n_windows)
            entry.update({"source": "%s/%d/%d.npy" % (folder, idx, idx), "file": "%s/%s" % (split, name),
                          "source_rows_sha256": sha256_array(raw),
                          "ev_loc_sha256": sha256_array(ev_loc), "evs_norm_sha256": sha256_array(evs_norm),
                          "cls_sha256": sha256_array(cls)})
            card["sequences"][name] = entry
            card["splits"][split].append(name)
            print("%-5s %-26s %9d 事件 %6.1f s %5d 窗 目标 %.2f%% 目标数 %3d 背景 %.4f /px/s" % (
                split, name, entry["events"], entry["duration_s"], entry["n_windows"],
                100 * entry["target_fraction"], entry["n_targets"], entry["background_rate_per_px_s"]), flush=True)
            del raw, ev_loc, evs_norm, cls, seq
    totals = {}
    for split, names in card["splits"].items():
        seqs = [card["sequences"][n] for n in names]
        events = sum(s["events"] for s in seqs)
        totals[split] = {"sequences": len(seqs), "events": events,
                         "duration_min": sum(s["duration_s"] for s in seqs) / 60.0,
                         "target_fraction": sum(s["target_events"] for s in seqs) / float(max(events, 1))}
    card["totals"] = totals
    with open(os.path.join(args.dst, "data_card.json"), "w", encoding="utf-8") as stream:
        json.dump(card, stream, indent=2, ensure_ascii=False)
    for split, tot in totals.items():
        print("%-5s %2d 段 %10d 事件 %6.1f min 目标事件 %.2f%%" % (
            split, tot["sequences"], tot["events"], tot["duration_min"], 100 * tot["target_fraction"]))
    print("CONVERT FINISHED: %s（%.0f s）" % (args.dst, time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
