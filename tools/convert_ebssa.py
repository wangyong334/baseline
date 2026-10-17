"""EBSSA（DAVIS 240x180 部分）-> EV-UAV 格式 NPZ（10-16 用户确认的方案）。

数据：labelled_ebssa.h5（tonic 用的 Google Drive 文件，12,106,528,328 字节），每段录像一个组：
    TD/{x,y,p,ts}    事件；x/y 从 1 起（MATLAB），p 为 -1/1，ts 为 µs 绝对时间
    Obj/{id,ts,x,y}  专家标注点（标出进入/离开点后线性插值，约 100 Hz）；所有可见点源都标
                     （被跟踪的卫星、凝视的恒星、漂移的恒星、划过的卫星）
只转换 DAVIS（xMax = 240）且有 Obj 的 34 段；ATIS 的 14 段与无标注的 36 段不转换。

转换规则（每段录像，事件顺序不变）：
    1. 时间裁到标注区间 [min Obj.ts, max Obj.ts]（漏标集中在区间外）；区间内保持连续，不切段
    2. 坐标 x-1, y-1（0 起）；p: -1 -> 0, 1 -> 1；t_ms = floor((ts - min Obj.ts) / 1000)
    3. 标签（tonic 加载器的 10x10 框约定）：每条轨迹的标注点按时间线性插值到事件自身时刻（只在该轨迹首末点之间），
       |x - x_c| <= 5 且 |y - y_c| <= 5 即目标；落在多个框内时取欧氏距离最近的轨迹（距离相同取先出现的轨迹）；
       target_id = Obj.id + 1（背景 0）。标签在 µs 精度上计算，之后才取整到毫秒
    4. cls（只用于分析，读取器忽略）：0 背景；按所属轨迹的净速度（首末点位移 / 时长，像素每 50 ms）
       1 静止 (< 0.1)、2 漂移 ([0.1, 1))、3 快速 (>= 1)
    不去热像素、不去噪：约 8% 的目标事件落在 > 100 Hz 的像素上（静止点源与热像素相似）；去噪本身是任务的一部分。
输出字段与 EV-Flying x4 相同（流式读取器只用 ev_loc 与 evs_norm[:,3:6]；K5 读取器用 ev_loc 与 evs_norm[:,0:4]）：
    ev_loc   int64   [N,3]  x, y, t_ms
    evs_norm float32 [N,6]  x/256, y/192, t_ms/8192, p, label, target_id
             （前 3 列只给 K5：体素画布按 32 的倍数取 256x192；整段的 t/8192 会大于 1，K5 训练前需另切 8 s 块）
    cls      int8    [N]
有效区域 240x180；网络画布 240x184（只在下方补 4 行，与 EV-UAV、EV-Flying 同一规则：补到 8 的倍数）。

划分（按录像，与 EV-UAV、EV-Flying 一致）：按运动类别分层（含快速轨迹 / 只有静止 / 其余），用 balanced_split
（RandomState(0)，200000 次抽样）找时长、事件数、目标事件数三项都最接近 60/20/20 的一组，且每个子集三层都有；
结果写死在 SPLITS（--search-split 可从 h5 复现），转换后重新统计并写进数据卡。

每个文件写完后用项目自己的读取器 dataset/stream_windows.load_npz_events 读回校验；另写 data_card.json
（规则、原始 h5 的 SHA-256、逐段统计与各数组内容的 SHA-256）和 SHA256SUMS（可在服务器上 sha256sum -c）。

用法（在仓库根目录；需要 h5py）:
    python tools/convert_ebssa.py --h5 /path/labelled_ebssa.h5 --dst /path/ebssa_davis
    python tools/convert_ebssa.py --h5 /path/labelled_ebssa.h5 --search-split     # 只复现划分，不写文件
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from dataset.stream_windows import load_npz_events  # noqa: E402

SENSOR_W, SENSOR_H = 240, 180                                  # DAVIS240，有效区域
CANVAS_W, CANVAS_H = 240, 184                                  # 网络画布：补到 8 的倍数
K5_W, K5_H, K5_T = 256, 192, 8192.0                            # evs_norm 前 3 列的归一化（只给 K5）
WINDOW_MS = 50
BOX_HALF = 5.0                                                 # tonic: 10x10 框，中心为标注点
STATIC_MAX, DRIFT_MAX = 0.1, 1.0                               # 运动类别阈值（像素每 50 ms，只用于分析）
CLASS_NAMES = {0: "background", 1: "static", 2: "drift", 3: "fast"}
SPLIT_FRACTIONS = (("train", 0.6), ("val", 0.2), ("test", 0.2))
SPLITS = {
    "train": ["20170214-21-15_SL8RB_21938", "SL14RB_leap_1933_davis_td", "SL8RB_1957_davis_td",
              "archenar_leos_1149_davis_td", "archenar_leos_1152_davis_td", "archenar_leos_1159_davis_td",
              "ariane_1946_davis_td", "atlasrb_1851]_davis_td", "canopus_leos_11_26_davis_td",
              "canopus_leos_1320_davis_td", "canopus_streaks_1603_davis_td", "cosmos_1848_davis_td",
              "geoslew_2103_davis_td", "geostarelong_abs6_2022_davis_td", "mag4.46hip54682_1959_davis_td",
              "mag7.41hip50559_2003_davis_td", "optusc1d2_stare_2028_davis_td", "resursdk1_davis_td",
              "skymuster2_2138_davis_td", "sl14rb_1901_davis_td", "sl3rb_00877_1940_davis_td"],
    "val": ["SL_2ARB_davis_td", "archenar_leos_1223_davis_td", "beidu_2001_davis_td", "canopus_streaks_1558_davis_td",
            "daytime_geo_stare_1700_davis_td", "mag5.52hip50456_2001_davis_td", "slew2107_davis_td"],
    "test": ["20170214-20-58_22285_SL-16RB", "archenar_leos_11_33_davis_td", "geo_abs6_25924_2017_davis_td",
             "iridium76_sat_davis_td", "refslew_2106_davis_td", "skymuster_stare_2033_davis_td"],
}


def sha256_array(a):
    """数组内容（dtype、形状、字节）的 SHA-256，与 NPZ 的压缩方式和时间戳无关，可用于服务器端核对。"""
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_name(key):
    """录像名 -> 输出文件名（去掉文件名里不安全的字符，如 atlasrb_1851] 的 ']'）。"""
    return "ebssa_%s.npz" % re.sub(r"[^A-Za-z0-9._-]", "", key)


def motion_class(tt, xx, yy):
    """轨迹净速度（首末点位移 / 时长，像素每 50 ms） -> 1 静止 / 2 漂移 / 3 快速。"""
    v = math.hypot(xx[-1] - xx[0], yy[-1] - yy[0]) / max(tt[-1] - tt[0], 1.0) * (WINDOW_MS * 1000.0)
    return (1 if v < STATIC_MAX else (2 if v < DRIFT_MAX else 3)), v


def build_tracks(obj_id, obj_ts, obj_x, obj_y):
    """标注点 -> 轨迹列表 [(id, tt, xx, yy, cls, v)]：按时间排序、去掉重复时刻（保留第一个），至少 2 个点。"""
    tracks = []
    for j in np.unique(obj_id):
        m = obj_id == j
        order = np.argsort(obj_ts[m], kind="stable")
        tt, xx, yy = obj_ts[m][order], obj_x[m][order], obj_y[m][order]
        keep = np.concatenate([[True], np.diff(tt) > 0])
        tt, xx, yy = tt[keep], xx[keep], yy[keep]
        if tt.size < 2:
            continue
        cls, v = motion_class(tt, xx, yy)
        tracks.append((int(j), tt, xx, yy, cls, v))
    return tracks


def box_labels(ts, x, y, tracks):
    """10x10 框标签：返回 (label bool, target_id float32, cls int8)。ts 必须单调不减；x/y 与轨迹同一坐标系。"""
    n = ts.size
    best = np.full(n, np.inf)
    target_id = np.zeros(n, dtype=np.float32)
    cls = np.zeros(n, dtype=np.int8)
    for j, tt, xx, yy, c, _ in tracks:
        a, b = np.searchsorted(ts, tt[0], side="left"), np.searchsorted(ts, tt[-1], side="right")
        if b <= a:
            continue
        px, py = np.interp(ts[a:b], tt, xx), np.interp(ts[a:b], tt, yy)
        dx, dy = x[a:b] - px, y[a:b] - py
        d = np.hypot(dx, dy)
        win = (np.abs(dx) <= BOX_HALF) & (np.abs(dy) <= BOX_HALF) & (d < best[a:b])
        idx = np.flatnonzero(win) + a
        best[idx] = d[win]
        target_id[idx] = j + 1
        cls[idx] = c
    return np.isfinite(best), target_id, cls


def check_recording(td, obj):
    """原始取值检查：坐标从 1 起且在传感器内、极性 -1/1、时间单调、标注 id 为非负整数、标注时间落在录像内。"""
    x, y, p, ts = td["x"], td["y"], td["p"], td["ts"]
    for name, v in (("x", x), ("y", y), ("p", p), ("Obj.id", obj["id"])):
        if not np.all(v == np.round(v)):
            raise ValueError("%s 含非整数" % name)
    if x.min() < 1 or x.max() > SENSOR_W or y.min() < 1 or y.max() > SENSOR_H:
        raise ValueError("坐标越界（应为 1..%d x 1..%d）" % (SENSOR_W, SENSOR_H))
    if not np.isin(p, (-1, 1)).all():
        raise ValueError("极性不是 -1/1")
    if np.any(np.diff(ts) < 0):
        raise ValueError("时间不单调")
    if obj["id"].min() < 0:
        raise ValueError("Obj.id 为负")
    # Obj.ts 以 float32 存储（绝对时间约 1e9 µs 时间隔 64–256 µs），允许 1 ms 的舍入误差
    if obj["ts"].min() < ts[0] - 1000.0 or obj["ts"].max() > ts[-1] + 1000.0:
        raise ValueError("标注时间超出录像范围")


def convert_recording(td, obj):
    """一段录像 -> (ev_loc, evs_norm, cls, info)。td/obj 为 float64 一维数组的字典（h5 原值，未做任何换算）。"""
    check_recording(td, obj)
    tracks = build_tracks(obj["id"], obj["ts"], obj["x"], obj["y"])
    a0, a1 = float(obj["ts"].min()), float(obj["ts"].max())
    lo, hi = np.searchsorted(td["ts"], a0, side="left"), np.searchsorted(td["ts"], a1, side="right")
    ts = td["ts"][lo:hi]
    x, y, p = td["x"][lo:hi], td["y"][lo:hi], td["p"][lo:hi]
    label, target_id, cls = box_labels(ts, x, y, tracks)
    xs, ys = x.astype(np.int64) - 1, y.astype(np.int64) - 1
    t_ms = np.floor((ts - a0) / 1000.0).astype(np.int64)
    p01 = (p > 0).astype(np.float32)
    ev_loc = np.stack([xs, ys, t_ms], 1)
    evs_norm = np.stack([xs / float(K5_W), ys / float(K5_H), t_ms / K5_T, p01, label.astype(np.float32), target_id],
                        1).astype(np.float32)
    info = {"annotated_start_us": a0, "annotated_end_us": a1, "annotated_offset_s": (a0 - float(td["ts"][0])) / 1e6,
            "source_events": int(td["ts"].size), "dropped_before": int(lo), "dropped_after": int(td["ts"].size - hi),
            "tracks": [{"id": j, "points": int(tt.size), "span_s": float((tt[-1] - tt[0]) / 1e6),
                        "speed_px_per_50ms": float(v), "class": CLASS_NAMES[c]} for j, tt, _, _, c, v in tracks],
            "stratum": stratum_of({c for _, _, _, _, c, _ in tracks})}
    return ev_loc, evs_norm, cls, info


def stratum_of(classes):
    """划分分层：有快速轨迹 -> F；只有静止 -> S；其余 -> D。"""
    if 3 in classes:
        return "F"
    return "S" if classes == {1} else "D"


def balanced_split(rows, seed=0, draws=200000):
    """rows: [{"key","stratum","duration","events","targets"}]（顺序固定）。
    随机把每段分到 train/val/test（概率 0.6/0.2/0.2），要求每个子集三层都有，取三项占比与 60/20/20 最大偏差最小的一组。
    返回 ({key: split}, 最大偏差)。"""
    names = [s for s, _ in SPLIT_FRACTIONS]
    target = np.array([f for _, f in SPLIT_FRACTIONS])
    strata = np.array([r["stratum"] for r in rows])
    m = np.array([[r["duration"], r["events"], r["targets"]] for r in rows], dtype=np.float64)
    total = m.sum(0)
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(draws):
        a = rng.choice(len(names), len(rows), p=target)
        if not all(((a == k) & (strata == s)).any() for k in range(len(names)) for s in "SDF"):
            continue
        frac = np.stack([m[a == k].sum(0) / total for k in range(len(names))])
        dev = float(np.abs(frac - target[:, None]).max())
        if best is None or dev < best[0]:
            best = (dev, a.copy())
    if best is None:
        raise RuntimeError("没有找到满足分层约束的划分")
    return {r["key"]: names[k] for r, k in zip(rows, best[1])}, best[0]


def read_group(group):
    td = {n: np.asarray(group["TD"][n][()], dtype=np.float64).ravel() for n in ("x", "y", "p", "ts")}
    obj = {n: np.asarray(group["Obj"][n][()], dtype=np.float64).ravel() for n in ("id", "ts", "x", "y")}
    return td, obj


def davis_labelled_keys(h5):
    return [k for k in h5.keys() if "Obj" in h5[k] and int(np.asarray(h5[k]["xMax"][()]).ravel()[0]) == SENSOR_W]


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
            "class_events": {CLASS_NAMES[c]: int((cls == c).sum()) for c in CLASS_NAMES},
            "background_rate_per_px_s": n_bg / float(SENSOR_W * SENSOR_H * duration_s),
            "on_fraction": float(evs_norm[:, 3].mean()),
            "x_max": int(ev_loc[:, 0].max()), "y_max": int(ev_loc[:, 1].max())}


def search_split(h5):
    rows = []
    for key in davis_labelled_keys(h5):
        ev_loc, evs_norm, _, info = convert_recording(*read_group(h5[key]))
        rows.append({"key": key, "stratum": info["stratum"], "events": int(ev_loc.shape[0]),
                     "duration": (info["annotated_end_us"] - info["annotated_start_us"]) / 1e6,
                     "targets": int(evs_norm[:, 4].sum())})
    split, dev = balanced_split(rows)
    print("最大偏差 %.4f" % dev)
    for name, _ in SPLIT_FRACTIONS:
        keys = sorted(k for k, s in split.items() if s == name)
        print("%-5s %2d 段 %s %s" % (name, len(keys), keys, "== SPLITS" if keys == sorted(SPLITS[name]) else "!= SPLITS"))


def main():
    parser = argparse.ArgumentParser(description="EBSSA DAVIS -> EV-UAV-format NPZ")
    parser.add_argument("--h5", required=True, help="labelled_ebssa.h5")
    parser.add_argument("--dst", default=None, help="输出根目录（写 train/ val/ test/、data_card.json、SHA256SUMS）")
    parser.add_argument("--search-split", action="store_true", help="只从 h5 复现划分并与 SPLITS 比较")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    import h5py

    t_start = time.time()
    with h5py.File(args.h5, "r") as h5:
        keys = davis_labelled_keys(h5)
        if sorted(keys) != sorted(k for v in SPLITS.values() for k in v):
            raise RuntimeError("h5 里的 DAVIS 有标注录像与 SPLITS 不一致")
        if args.search_split:
            search_split(h5)
            return
        if not args.dst:
            raise SystemExit("需要 --dst")
        card = {"source": "EBSSA labelled_ebssa.h5 (Afshar et al., IEEE Sensors J. 2020; tonic Google Drive id "
                          "1lCh2HWvxEzzaBHT5TlPuyUn6XPM5OVWN); DAVIS 240x180 recordings with Obj labels only",
                "source_h5_bytes": os.path.getsize(args.h5), "source_h5_sha256": sha256_file(args.h5),
                "sensor_resolution": [SENSOR_W, SENSOR_H], "canvas": [CANVAS_W, CANVAS_H], "window_ms": WINDOW_MS,
                "time_rule": "crop to [min Obj.ts, max Obj.ts]; t_ms = floor((ts - min Obj.ts) / 1000)",
                "coordinate_rule": "x - 1, y - 1 (h5 is 1-based); p: -1 -> 0, 1 -> 1",
                "label_rule": "tonic 10x10 box: track points linearly interpolated to each event time (inside the "
                              "track span); |dx| <= 5 and |dy| <= 5 -> target; nearest track wins; target_id = Obj.id + 1",
                "cls_rule": "0 bg; track net speed px/50ms: 1 static < %.1f <= 2 drift < %.1f <= 3 fast (analysis only)"
                            % (STATIC_MAX, DRIFT_MAX),
                "filtering": "none (no hot-pixel removal, no denoising)",
                "split_rule": "by recording; strata F (has fast track) / S (static only) / D; balanced_split seed 0, "
                              "200000 draws, minimise max deviation of duration/events/targets from 60/20/20",
                "evs_norm_columns": ["x/%d" % K5_W, "y/%d" % K5_H, "t_ms/%d" % int(K5_T), "p", "label", "target_id"],
                "splits": {}, "sequences": {}}
        sums = []
        for split, split_keys in SPLITS.items():
            out_dir = os.path.join(args.dst, split)
            os.makedirs(out_dir, exist_ok=True)
            card["splits"][split] = []
            for key in split_keys:
                name = file_name(key)
                dst = os.path.join(out_dir, name)
                if os.path.exists(dst) and not args.overwrite:
                    raise RuntimeError("已存在 %s（加 --overwrite 覆盖）" % dst)
                td, obj = read_group(h5[key])
                ev_loc, evs_norm, cls, info = convert_recording(td, obj)
                n_windows = int(ev_loc[:, 2].max()) // WINDOW_MS + 1
                np.savez(dst, ev_loc=ev_loc, evs_norm=evs_norm, cls=cls)
                # 用项目自己的流式读取器读回：坐标在有效区域内、时间在窗内、p/label 合法、事件与标签数一致
                seq = load_npz_events(dst, SENSOR_H, SENSOR_W, WINDOW_MS, 0, 5)
                if seq.n_windows != n_windows or seq.n_events != ev_loc.shape[0]:
                    raise RuntimeError("%s 读回的窗数或事件数不一致" % dst)
                if int(seq.label.sum()) != int(evs_norm[:, 4].sum()) or not np.array_equal(seq.x, ev_loc[:, 0]):
                    raise RuntimeError("%s 读回的标签或坐标不一致" % dst)
                entry = sequence_card(ev_loc, evs_norm, cls, n_windows)
                entry.update(info)
                entry.update({"source": key, "file": "%s/%s" % (split, name),
                              "source_td_sha256": sha256_array(np.stack([td[n] for n in ("x", "y", "p", "ts")], 1)),
                              "ev_loc_sha256": sha256_array(ev_loc), "evs_norm_sha256": sha256_array(evs_norm),
                              "cls_sha256": sha256_array(cls)})
                card["sequences"][name] = entry
                card["splits"][split].append(name)
                sums.append((sha256_file(dst), "%s/%s" % (split, name)))
                print("%-5s %-44s %9d 事件 %6.1f s %5d 窗 目标 %.2f%% 目标数 %2d 层 %s" % (
                    split, name, entry["events"], entry["duration_s"], entry["n_windows"],
                    100 * entry["target_fraction"], entry["n_targets"], entry["stratum"]), flush=True)
                del td, obj, ev_loc, evs_norm, cls, seq
    totals = {}
    all_seqs = list(card["sequences"].values())
    grand = {k: float(sum(s[k] for s in all_seqs)) for k in ("duration_s", "events", "target_events")}
    for split, names in card["splits"].items():
        seqs = [card["sequences"][n] for n in names]
        tot = {k: float(sum(s[k] for s in seqs)) for k in ("duration_s", "events", "target_events")}
        totals[split] = {"sequences": len(seqs), "events": int(tot["events"]), "windows": sum(s["n_windows"] for s in seqs),
                         "duration_min": tot["duration_s"] / 60.0,
                         "target_fraction": tot["target_events"] / max(tot["events"], 1.0),
                         "share_of_duration": tot["duration_s"] / grand["duration_s"],
                         "share_of_events": tot["events"] / grand["events"],
                         "share_of_target_events": tot["target_events"] / grand["target_events"],
                         "strata": {s: sum(x["stratum"] == s for x in seqs) for s in "SDF"}}
    card["totals"] = totals
    card_path = os.path.join(args.dst, "data_card.json")
    with open(card_path, "w", encoding="utf-8") as stream:
        json.dump(card, stream, indent=2, ensure_ascii=False)
    sums.append((sha256_file(card_path), "data_card.json"))
    with open(os.path.join(args.dst, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as stream:
        stream.writelines("%s  %s\n" % item for item in sums)
    for split, tot in totals.items():
        print("%-5s %2d 段 %10d 事件 %6.1f min %6d 窗 目标事件 %.2f%% | 时长/事件/目标占比 %.3f/%.3f/%.3f | S/D/F %s" % (
            split, tot["sequences"], tot["events"], tot["duration_min"], tot["windows"], 100 * tot["target_fraction"],
            tot["share_of_duration"], tot["share_of_events"], tot["share_of_target_events"],
            "/".join(str(tot["strata"][s]) for s in "SDF")))
    print("CONVERT FINISHED: %s（%.0f s）" % (args.dst, time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
