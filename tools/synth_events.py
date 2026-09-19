"""按点过程生成模型合成 EV-UAV 格式的事件序列（真值精确、参数可控），用于定理级核对与本地端到端冒烟。

每条序列 8 s（可改），由以下来源叠加：
    背景      均匀泊松，每像素每 ms 率 bg_rate；dispersion > 0 时每 50 ms 窗的整体强度乘以 Gamma(1/d, d)（过离散杂波）
    热像素    n_hot 个固定像素，率 hot_rate
    闪烁      可选的静止矩形，强度按 (1+sin(2*pi*f*t))/2 调制；亮度上升段为 ON、下降段为 OFF（静止的二维结构，
              矩形内像素同步变化——空间相关的背景）
    成团杂波  clusters > 0 时每窗随机出现若干矩形小块（6-16 像素见方），块内背景率放大 cluster_gain 倍（空间相关的突发）
    背景突变  bg_step > 1 时在随机时刻之后背景率整体乘以 bg_step
    闪现目标  blink=True 时目标只在"亮"的时段发事件（亮/灭时长各 100-400 ms 随机），检验目标短暂消失
    目标      n_targets 个小圆盘（半径 radius），匀速直线运动（速度 speed 像素/ms，方向随机），存活一段时间；
              事件均匀落在圆盘内，率 target_rate（每 ms），极性按"暗目标衬亮背景"：前半圆 OFF、后半圆 ON
              （bright_prob 的概率改为亮目标，极性反号）
输出 NPZ 字段与原数据集一致：ev_loc [N,3] = (x, y, t)，evs_norm [N,6] = (x/W, y/H, t/T, p, label, target_id)。

用法:
    python tools/synth_events.py --out /path/synth --n-train 20 --n-val 5 --n-test 5 --preset mixed
    预设：sparse / dense（过离散）/ flicker（同步闪烁）/ clutter（成团突发）/ bgstep（背景突变）/ blink（目标闪现）/ mixed
    python tools/synth_events.py --out /path/synth_small --height 56 --width 72 --n-train 6 --n-val 2 --n-test 2
"""
import argparse
import json
import os

import numpy as np

PRESETS = {
    "sparse": dict(bg_rate=0.003 / 50, dispersion=0.0, n_hot=10, flicker=False),
    "dense": dict(bg_rate=0.04 / 50, dispersion=0.5, n_hot=30, flicker=False),
    "flicker": dict(bg_rate=0.003 / 50, dispersion=0.0, n_hot=10, flicker=True),
    "clutter": dict(bg_rate=0.003 / 50, dispersion=0.0, n_hot=10, clusters=2.0),
    "bgstep": dict(bg_rate=0.003 / 50, dispersion=0.0, n_hot=10, bg_step=8.0),
    "blink": dict(bg_rate=0.003 / 50, dispersion=0.0, n_hot=10, blink=True),
}


def poisson_times(rng, rate, t0, t1):
    """[t0, t1) 内率为 rate（每 ms）的齐次泊松事件时间。"""
    n = rng.poisson(max(rate, 0.0) * max(t1 - t0, 0.0))
    return rng.uniform(t0, t1, n)


def generate_sequence(rng, height=260, width=346, duration_ms=8000.0, bg_rate=0.003 / 50, dispersion=0.0,
                      n_hot=10, hot_rate=0.02, flicker=False, flicker_rate=0.05, flicker_hz=(5.0, 20.0),
                      n_targets=(1, 3), radius=(1.5, 3.0), speed=(0.007, 0.08), target_rate=(0.2, 0.6),
                      bright_prob=0.2, window_ms=50.0, clusters=0.0, cluster_gain=20.0, bg_step=1.0, blink=False):
    """生成一条序列，返回 dict(x, y, t, p, label, target_id)（numpy，t 为整数毫秒）。"""
    xs, ys, ts, ps, labels, tids = [], [], [], [], [], []

    def add(x, y, t, p, label, tid):
        keep = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (t >= 0) & (t < duration_ms)
        xs.append(x[keep]); ys.append(y[keep]); ts.append(t[keep]); ps.append(p[keep])
        labels.append(np.full(int(keep.sum()), label, np.float32)); tids.append(np.full(int(keep.sum()), tid, np.float64))

    n_windows = int(round(duration_ms / window_ms))
    step_at = rng.uniform(0.3, 0.7) * duration_ms if bg_step > 1 else duration_ms
    for w in range(n_windows):                                             # 背景（可过离散、可突变）
        scale = rng.gamma(1.0 / dispersion, dispersion) if dispersion > 0 else 1.0
        if w * window_ms >= step_at:
            scale *= bg_step
        n = rng.poisson(bg_rate * height * width * window_ms * scale)
        t = rng.uniform(w * window_ms, (w + 1) * window_ms, n)
        add(rng.uniform(0, width, n), rng.uniform(0, height, n), t, rng.randint(0, 2, n), 0.0, 0.0)
        for _ in range(rng.poisson(clusters)):                             # 成团突发
            cw, ch = rng.randint(6, 17), rng.randint(6, 17)
            cx, cy = rng.uniform(0, max(width - cw, 1)), rng.uniform(0, max(height - ch, 1))
            m = rng.poisson(bg_rate * cluster_gain * cw * ch * window_ms)
            add(cx + rng.uniform(0, cw, m), cy + rng.uniform(0, ch, m),
                rng.uniform(w * window_ms, (w + 1) * window_ms, m), rng.randint(0, 2, m), 0.0, 0.0)
    for _ in range(int(n_hot)):                                            # 热像素
        t = poisson_times(rng, hot_rate, 0.0, duration_ms)
        add(np.full(t.size, rng.randint(0, width)) + 0.5, np.full(t.size, rng.randint(0, height)) + 0.5, t,
            rng.randint(0, 2, t.size), 0.0, 0.0)
    if flicker:                                                            # 静止闪烁矩形
        fw, fh = rng.randint(6, 18), rng.randint(10, 36)
        fx, fy = rng.randint(0, max(width - fw, 1)), rng.randint(0, max(height - fh, 1))
        freq = rng.uniform(*flicker_hz) / 1000.0
        n = rng.poisson(flicker_rate * fw * fh * duration_ms)
        t = rng.uniform(0, duration_ms, n)
        accept = rng.uniform(0, 1, n) < 0.5 * (1.0 + np.sin(2 * np.pi * freq * t))
        t = t[accept]
        p = (np.cos(2 * np.pi * freq * t) > 0).astype(np.int64)            # 亮度上升段 ON
        add(fx + rng.uniform(0, fw, t.size), fy + rng.uniform(0, fh, t.size), t, p, 0.0, 0.0)
    for i in range(rng.randint(n_targets[0], n_targets[1] + 1)):          # 运动小目标
        r = rng.uniform(*radius)
        v = rng.uniform(*speed)
        angle = rng.uniform(0, 2 * np.pi)
        vx, vy = v * np.cos(angle), v * np.sin(angle)
        t0 = rng.uniform(0, 0.5 * duration_ms)
        t1 = min(duration_ms, t0 + rng.uniform(0.25, 1.0) * duration_ms)
        x0, y0 = rng.uniform(0.2 * width, 0.8 * width), rng.uniform(0.2 * height, 0.8 * height)
        t = poisson_times(rng, rng.uniform(*target_rate), t0, t1)
        if blink:                                                          # 亮/灭交替，只保留亮的时段
            edges, on, cursor = [], True, t0
            while cursor < t1:
                length = rng.uniform(100.0, 400.0)
                if on:
                    edges.append((cursor, cursor + length))
                cursor, on = cursor + length, not on
            keep = np.zeros(t.size, dtype=bool)
            for a, b in edges:
                keep |= (t >= a) & (t < b)
            t = t[keep]
        rho, phi = r * np.sqrt(rng.uniform(0, 1, t.size)), rng.uniform(0, 2 * np.pi, t.size)
        ox, oy = rho * np.cos(phi), rho * np.sin(phi)
        leading = (ox * vx + oy * vy) > 0
        dark = rng.uniform() >= bright_prob
        p = np.where(leading, 0, 1) if dark else np.where(leading, 1, 0)     # 暗目标：前沿 OFF、后沿 ON
        add(x0 + vx * (t - t0) + ox, y0 + vy * (t - t0) + oy, t, p, 1.0, float(i + 1))
    x = np.floor(np.concatenate(xs)).astype(np.int64)
    y = np.floor(np.concatenate(ys)).astype(np.int64)
    t = np.floor(np.concatenate(ts)).astype(np.int64)
    order = np.argsort(t, kind="stable")
    return {"x": x[order], "y": y[order], "t": t[order], "p": np.concatenate(ps)[order].astype(np.int64),
            "label": np.concatenate(labels)[order], "target_id": np.concatenate(tids)[order]}


def save_npz(path, seq, height, width, duration_ms):
    """按原数据集字段写 NPZ（load_npz_events 读取 ev_loc 与 evs_norm[:,3:6]）。"""
    ev_loc = np.stack([seq["x"], seq["y"], seq["t"]], 1).astype(np.int64)
    evs_norm = np.stack([seq["x"] / float(width), seq["y"] / float(height), seq["t"] / float(duration_ms),
                         seq["p"], seq["label"], seq["target_id"]], 1).astype(np.float32)
    np.savez(path, ev_loc=ev_loc, evs_norm=evs_norm)


def main():
    """命令行入口：按预设生成 train/val/test 三个划分。"""
    parser = argparse.ArgumentParser(description="合成 EV-UAV 格式的事件序列")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--n-val", type=int, default=5)
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--preset", choices=tuple(PRESETS) + ("mixed",), default="mixed")
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--width", type=int, default=346)
    parser.add_argument("--duration-ms", type=float, default=8000.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)
    manifest = {}
    for split, count in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        for i in range(count):
            preset = args.preset if args.preset != "mixed" else list(PRESETS)[rng.randint(len(PRESETS))]
            seq = generate_sequence(rng, args.height, args.width, args.duration_ms, **PRESETS[preset])
            name = "%s_%03d.npz" % (split, i)
            save_npz(os.path.join(args.out, split, name), seq, args.height, args.width, args.duration_ms)
            manifest[split + "/" + name] = {"preset": preset, "events": int(seq["t"].size),
                                            "target_events": int(seq["label"].sum())}
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print("写入 %d 条序列到 %s" % (len(manifest), args.out))


if __name__ == "__main__":
    main()
