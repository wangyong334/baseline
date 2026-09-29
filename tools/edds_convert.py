"""把 EDDS（Event based Drone Detection and Segmentation，Zenodo 10.5281/zenodo.18629643，CC-BY-4.0；
Prophesee EVK4 1280x720，EVT 3.0 RAW）转换成 EV-UAV 的 NPZ 格式，用于跨数据集零样本测试（模型只在 EV-UAV 上训练）。

处理步骤（全部与标签无关，跑之前写死）：
    1 解码 EVT 3.0 RAW（纯 numpy、分块、块间状态接续，与整文件解码逐位相同）。
      EDDS 标注文件的时间 = RAW 时间 - ((首个 TIME_HIGH - 1) << 12) 微秒（实测 15 段录像全部逐事件精确对齐）。
    2 屏蔽热像素：全程事件率 > hot_rate（默认 1000 次/秒）的像素。它们是同一台相机上固定的坏点
      （例如 (789, 499) 约 850 万次/秒；普通像素 99.9% 低于 16 次/秒），落在上面的目标事件为 0（本工具逐段核对）。
    3 空间合并：坐标整除 factor（默认 4：1280x720 -> 320x180，放进 EV-UAV 的 346x260 传感器画布）。
      时间、极性、标签都不变，不删除任何事件。
    4 逐事件标签：(x, y, t, p) 出现在该段录像任一 segmentation-*.csv 里的事件为目标（target_id = 1），其余为背景。
    5 切成 8 s 的序列（160 窗 x 50 ms，与 EV-UAV 相同），时间转为序列内的毫秒。
输出：<out>/test/edds_<录像>_<序号>.npz，字段与 EV-UAV 相同：ev_loc int32 [x, y, t_ms]，
evs_norm float32 [x/352, y/288, t/8192, p, label, target_id]（读取见 dataset/stream_windows.load_npz_events，
每个文件写完都用它校验一遍）；<out>/manifest.json 记录处理参数、热像素、标签核对、各序列统计与分辨率上限。
分辨率上限：模型对每个（窗, 像素）只给一个分数，同一格里的目标与背景事件拿到同一个分数；按每格多数标签给出的
"最好情况"逐事件 IoU / ACC，分别在原生分辨率与合并后的分辨率上计算，用来说明合并本身的代价。

用法（本地）：python tools/edds_convert.py --src <edds 解压目录> --out <输出目录> [--factor 4] [--names stationary01 ...]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.stream_windows import load_npz_events  # noqa: E402

WIDTH, HEIGHT = 1280, 720                 # EVK4 / IMX636
BITS12 = np.arange(12, dtype=np.int64)
STATIONARY = ["stationary%02d" % i for i in range(1, 9)]


# ---------------------------------------------------------------------------
# EVT 3.0 解码
# ---------------------------------------------------------------------------
# 16 位字，类型在高 4 位：0 ADDR_Y、2 ADDR_X（单个事件）、3 VECT_BASE_X、4 VECT_12、5 VECT_8、6 TIME_LOW、8 TIME_HIGH；
# 其余类型（7/A/E/F 等）不产生 CD 事件，忽略。时间 = 回绕次数 * 2^24 + TIME_HIGH << 12 + TIME_LOW；
# TIME_HIGH 之后、下一个 TIME_LOW 之前低 12 位取 0（与 OpenEB 的解码器一致）。
# 块间状态用"在块首补几个不产生事件的合成字"接续，所以任意分块与整文件解码逐位相同（tests/test_stream_edds.py 守着）。

def header_size(path):
    """RAW 文件开头以 % 开头的文本头的字节数。"""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    i = 0
    while head[i:i + 1] == b"%":
        i = head.index(b"\n", i) + 1
    return i


class DecoderState(object):
    """块间接续的解码状态。"""

    def __init__(self):
        self.y, self.th, self.loops = 0, 0, 0
        self.tl, self.tl_valid = 0, False
        self.base, self.pol = 0, 0


def decode_words(words, st):
    """解码一块 16 位字（uint16 数组），原地更新 st。返回 (x, y, t, p)：int64 x/y/t（微秒），int8 p。"""
    pre = [(0x0 << 12) | st.y, (0x8 << 12) | st.th]
    if st.tl_valid:
        pre.append((0x6 << 12) | st.tl)
    pre.append((0x3 << 12) | (st.pol << 11) | min(st.base, 0x7FF))
    w = np.concatenate([np.array(pre, np.uint16), np.asarray(words, np.uint16)]).astype(np.int64)
    idx = np.arange(w.shape[0], dtype=np.int64)
    typ = w >> 12

    def last_idx(mask):
        return np.maximum.accumulate(np.where(mask, idx, -1))

    y = w[last_idx(typ == 0)] & 0x7FF
    is_th = typ == 8
    th_seq = w[is_th] & 0xFFF
    prev = np.concatenate([[st.th], th_seq[:-1]])
    loops_seq = st.loops + np.cumsum(th_seq < prev - 2048)            # TIME_HIGH 从 4095 回到 0：回绕一次
    th_rank = np.cumsum(is_th) - 1
    th_last = last_idx(is_th)
    tl_last = last_idx(typ == 6)
    low = np.where(tl_last > th_last, w[np.maximum(tl_last, 0)] & 0xFFF, 0)
    t = (loops_seq[th_rank] << 24) + (th_seq[th_rank] << 12) + low

    inc = np.where(typ == 4, 12, np.where(typ == 5, 8, 0))
    cs = np.cumsum(inc)
    cs_ex = cs - inc
    vb_last = last_idx(typ == 3)
    base = (w[vb_last] & 0x7FF) + (cs_ex - cs_ex[vb_last])
    vpol = (w[vb_last] >> 11) & 1

    is_x = typ == 2
    xs, ys, ts, ps = [w[is_x] & 0x7FF], [y[is_x]], [t[is_x]], [(w[is_x] >> 11) & 1]
    is_v = (typ == 4) | (typ == 5)
    if is_v.any():
        mask = np.where(typ[is_v] == 4, w[is_v] & 0xFFF, w[is_v] & 0xFF)
        r, c = np.nonzero(((mask[:, None] >> BITS12) & 1).astype(bool))
        xs.append(base[is_v][r] + c); ys.append(y[is_v][r]); ts.append(t[is_v][r]); ps.append(vpol[is_v][r])
    # 块内先放单个事件、再放向量事件，所以输出不是严格的时间顺序；需要时间顺序的调用方自己排序（convert_recording 会排）
    x, yy, tt, pp = (np.concatenate(a) for a in (xs, ys, ts, ps))
    ok = (x < WIDTH) & (yy < HEIGHT)

    st.y = int(y[-1]); st.th = int(th_seq[-1]); st.loops = int(loops_seq[-1])
    st.tl_valid = bool(tl_last[-1] > th_last[-1]); st.tl = int(w[tl_last[-1]] & 0xFFF) if st.tl_valid else 0
    st.base = int((w[vb_last[-1]] & 0x7FF) + (cs[-1] - cs_ex[vb_last[-1]])); st.pol = int(vpol[-1])
    return x[ok], yy[ok], tt[ok], pp[ok].astype(np.int8)


def first_time_high(path):
    """RAW 文件里第一个 TIME_HIGH 的值（12 位）。"""
    off = header_size(path)
    w = np.fromfile(path, dtype="<u2", count=1 << 20, offset=off).astype(np.int64)
    th = w[(w >> 12) == 8]
    if th.size == 0:
        raise ValueError("文件开头 100 万字里没有 TIME_HIGH: %s" % path)
    return int(th[0] & 0xFFF)


def label_time_shift(path):
    """EDDS 标注文件的时间原点（微秒）：(首个 TIME_HIGH - 1) << 12。"""
    return (first_time_high(path) - 1) << 12


def iter_events(path, chunk_words=20_000_000, shift=0):
    """逐块产出 (x, y, t, p)，t 已减去 shift（微秒）。首个 TIME_HIGH 之前的字没有完整时间戳，丢弃（与 OpenEB 一致）。"""
    off = header_size(path)
    st = DecoderState()
    th0 = first_time_high(path)
    started = False
    with open(path, "rb") as f:
        f.seek(off)
        while True:
            buf = f.read(int(chunk_words) * 2)
            if len(buf) < 2:
                break
            words = np.frombuffer(buf[:len(buf) // 2 * 2], dtype="<u2")
            if not started:
                th_pos = np.flatnonzero((words >> 12) == 8)
                if th_pos.size == 0:
                    continue
                words, st.th, started = words[th_pos[0]:], th0, True
            x, y, t, p = decode_words(words, st)
            yield x, y, t - int(shift), p


# ---------------------------------------------------------------------------
# 标签、热像素、分辨率上限
# ---------------------------------------------------------------------------

def event_key(x, y, t, p):
    """(x, y, t, p) 的唯一整数键（t 为微秒，x < 2048、y < 2048）。"""
    return (np.asarray(t, np.int64) << 23) | (np.asarray(y, np.int64) << 12) | (np.asarray(x, np.int64) << 1) \
        | np.asarray(p, np.int64)


def load_segmentation(directory):
    """读取一段录像的全部 segmentation-*.csv（列 x;y;t;p），返回 (排好序的唯一键, 行数)。"""
    files = sorted(glob.glob(os.path.join(directory, "segmentation-*.csv")))
    if not files:
        raise RuntimeError("没有 segmentation-*.csv: %s" % directory)
    rows = [np.loadtxt(f, delimiter=";", skiprows=1, dtype=np.int64, ndmin=2) for f in files]
    s = np.concatenate(rows)
    return np.unique(event_key(s[:, 0], s[:, 1], s[:, 2], s[:, 3])), int(s.shape[0])


def is_member(keys, sorted_keys):
    """keys 中每个元素是否在 sorted_keys 里（sorted_keys 必须升序）。"""
    if sorted_keys.size == 0:
        return np.zeros(keys.shape, bool)
    pos = np.searchsorted(sorted_keys, keys)
    return sorted_keys[np.minimum(pos, sorted_keys.size - 1)] == keys


def resolution_ceiling(t_us, x, y, label, window_us, width):
    """每个（窗, 像素）只给一个分数时，按每格多数标签（平局判目标）能达到的最好逐事件 IoU 与 ACC。"""
    cell = (np.asarray(t_us, np.int64) // int(window_us)) * (int(width) * 4096) \
        + np.asarray(y, np.int64) * int(width) + np.asarray(x, np.int64)
    _, inv = np.unique(cell, return_inverse=True)
    lab = np.asarray(label, bool)
    n_target = np.bincount(inv, weights=lab.astype(np.float64))
    n_all = np.bincount(inv).astype(np.float64)
    pick = n_target >= n_all - n_target
    tp = float(n_target[pick].sum())
    fp = float((n_all - n_target)[pick].sum())
    fn = float(n_target[~pick].sum())
    total = float(lab.sum())
    mixed = (n_target > 0) & (n_target < n_all)
    return {"iou": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
            "acc": tp / total if total else float("nan"),
            "target_events_in_mixed_cells": float(n_target[mixed].sum()) / total if total else float("nan")}


# ---------------------------------------------------------------------------
# 转换
# ---------------------------------------------------------------------------

def convert_recording(src, name, out_dir, factor=4, hot_rate=1000.0, segment_ms=8000, window_ms=50,
                      canvas=(346, 260), chunk_words=20_000_000, verify=True):
    """转换一段录像，写出若干 8 s 序列的 NPZ，返回这段录像的清单（dict）。"""
    t0 = time.time()
    raw = os.path.join(src, name + ".raw")
    shift = label_time_shift(raw)
    # 第一遍：逐像素事件数与时长 -> 热像素
    pix = np.zeros(WIDTH * HEIGHT, np.int64)
    t_lo, t_hi, n_raw = None, None, 0
    for x, y, t, _ in iter_events(raw, chunk_words, shift):
        if x.size:
            n_raw += int(x.size)
            pix += np.bincount(y * WIDTH + x, minlength=WIDTH * HEIGHT)
            t_lo = int(t.min()) if t_lo is None else min(t_lo, int(t.min()))
            t_hi = int(t.max()) if t_hi is None else max(t_hi, int(t.max()))
    duration_s = (t_hi - t_lo) / 1e6
    rate = pix / max(duration_s, 1e-9)
    hot = rate > float(hot_rate)
    hot_idx = np.flatnonzero(hot)
    seg_keys, seg_rows = load_segmentation(os.path.join(src, name))
    # 第二遍：打标签、去热像素
    xs, ys, ts, ps, ls = [], [], [], [], []
    found = np.zeros(seg_keys.size, bool)
    target_on_hot = 0
    for x, y, t, p in iter_events(raw, chunk_words, shift):
        if x.size == 0:
            continue
        k = event_key(x, y, t, p)
        lab = is_member(k, seg_keys)
        if lab.any():
            found[np.searchsorted(seg_keys, k[lab])] = True
        keep = ~hot[y * WIDTH + x]
        target_on_hot += int(np.count_nonzero(lab & ~keep))
        xs.append(x[keep].astype(np.int16)); ys.append(y[keep].astype(np.int16)); ts.append(t[keep])
        ps.append(p[keep]); ls.append(lab[keep])
    x, y, t, p, lab = (np.concatenate(a) for a in (xs, ys, ts, ps, ls))
    if t.size and t.min() < 0:
        raise RuntimeError("%s：平移后出现负时间（%d us），时间原点规则不成立" % (name, int(t.min())))
    order = np.argsort(t, kind="stable")
    x, y, t, p, lab = x[order], y[order], t[order], p[order], lab[order]
    ceiling_native = resolution_ceiling(t, x, y, lab, window_ms * 1000, WIDTH)
    xb, yb = x.astype(np.int64) // int(factor), y.astype(np.int64) // int(factor)
    width_b, height_b = -(-WIDTH // int(factor)), -(-HEIGHT // int(factor))
    if width_b > canvas[0] or height_b > canvas[1]:
        raise ValueError("合并 %d 倍后 %dx%d 放不进 %dx%d 的画布" % (factor, width_b, height_b, canvas[0], canvas[1]))
    ceiling_binned = resolution_ceiling(t, xb, yb, lab, window_ms * 1000, width_b)
    # 切成 8 s 序列
    t_ms = t // 1000
    seq_idx = t_ms // int(segment_ms)
    os.makedirs(out_dir, exist_ok=True)
    sequences = []
    for s in np.unique(seq_idx).tolist():
        m = seq_idx == s
        local_ms = t_ms[m] - s * int(segment_ms)
        local_us = t[m] - s * int(segment_ms) * 1000
        lab_s = lab[m].astype(np.float32)
        ev_loc = np.stack([xb[m], yb[m], local_ms], 1).astype(np.int32)
        evs_norm = np.stack([xb[m] / 352.0, yb[m] / 288.0, local_us / 1000.0 / 8192.0,
                             p[m].astype(np.float64), lab_s, lab_s], 1).astype(np.float32)
        fname = "edds_%s_%02d.npz" % (name, s)
        path = os.path.join(out_dir, fname)
        np.savez_compressed(path, ev_loc=ev_loc, evs_norm=evs_norm)
        if verify:
            seq = load_npz_events(path, canvas[1], canvas[0], window_ms, segment_ms // window_ms, 5)
            if seq.n_events != int(m.sum()) or int(seq.label.sum()) != int(lab_s.sum()):
                raise RuntimeError("%s 读回后事件数或标签数不一致" % fname)
        sequences.append({"file": fname, "events": int(m.sum()), "target_events": int(lab_s.sum()),
                          "duration_ms": int(local_ms.max()) + 1,
                          "windows_with_target": int(np.unique(local_ms[lab[m]] // int(window_ms)).size)})
    return {"name": name, "duration_s": round(duration_s, 3), "label_time_shift_us": int(shift),
            "raw_events": n_raw, "hot_pixels": [[int(i % WIDTH), int(i // WIDTH), round(float(rate[i]), 1)]
                                                for i in hot_idx],
            "hot_events_removed": int(pix[hot].sum()), "events_kept": int(t.size),
            "segmentation_rows": seg_rows, "segmentation_unique_events": int(seg_keys.size),
            "segmentation_found_in_raw": round(float(found.mean()), 6),
            "target_events_kept": int(lab.sum()), "target_events_on_hot_pixels": target_on_hot,
            "ceiling_native": ceiling_native, "ceiling_binned": ceiling_binned,
            "binned_size": [width_b, height_b], "sequences": sequences, "seconds": round(time.time() - t0, 1)}


def parse_args():
    p = argparse.ArgumentParser(description="EDDS -> EV-UAV NPZ（热像素屏蔽 + 空间合并 + 8 s 切分），用于零样本测试")
    p.add_argument("--src", required=True, help="edds.zip 解压后的目录（含 *.raw 与各录像子目录）")
    p.add_argument("--out", required=True, help="输出根目录；NPZ 写到 <out>/test/")
    p.add_argument("--names", nargs="+", default=STATIONARY, help="要转换的录像（默认固定机位 8 段）")
    p.add_argument("--factor", type=int, default=4, help="空间合并倍数（默认 4：1280x720 -> 320x180）")
    p.add_argument("--hot-rate", type=float, default=1000.0, help="热像素阈值（全程事件率，次/秒）")
    p.add_argument("--segment-ms", type=int, default=8000, help="序列长度（毫秒，与 EV-UAV 的 160 x 50 ms 相同）")
    p.add_argument("--window-ms", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    out_test = os.path.join(args.out, "test")
    if os.path.isdir(out_test) and os.listdir(out_test):
        raise RuntimeError("输出目录已有文件，请换一个 --out：%s" % out_test)
    manifest = {"source": "EDDS, Zenodo 10.5281/zenodo.18629643 (CC-BY-4.0)", "factor": args.factor,
                "hot_rate": args.hot_rate, "segment_ms": args.segment_ms, "window_ms": args.window_ms,
                "canvas": [346, 260], "recordings": []}
    for name in args.names:
        rec = convert_recording(args.src, name, out_test, args.factor, args.hot_rate, args.segment_ms,
                                args.window_ms)
        manifest["recordings"].append(rec)
        print("%s：%.1f s，热像素 %s，保留 %d 个事件（目标 %d），标注在 RAW 里找到 %.4f%%，落在热像素上的目标 %d；"
              "分辨率上限 IoU 原生 %.4f / 合并 %.4f；%d 个序列，用时 %.0f s" % (
                  name, rec["duration_s"], [h[:2] for h in rec["hot_pixels"]], rec["events_kept"],
                  rec["target_events_kept"], 100 * rec["segmentation_found_in_raw"],
                  rec["target_events_on_hot_pixels"], rec["ceiling_native"]["iou"], rec["ceiling_binned"]["iou"],
                  len(rec["sequences"]), rec["seconds"]), flush=True)
        with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
    n_seq = sum(len(r["sequences"]) for r in manifest["recordings"])
    print("完成：%d 段录像 -> %d 个序列，写在 %s；清单 %s" % (
        len(manifest["recordings"]), n_seq, out_test, os.path.join(args.out, "manifest.json")))


if __name__ == "__main__":
    main()
