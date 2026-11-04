"""Module-three evaluation suite from per-lag trajectories (FINAL_PLAN_20261104.md §4.4 / §7; CPU, numpy).

Input: the trajectories written by `train_track_head.py --mode eval --trajectory-dir` (one fixed-D pass: column d of
traj is the calibrated logit of every event at lag d, column 0 = V5's provisional value) and the dataset NPZ files
(positions, times, target ids; the trajectory keeps the file's event order).
From that single pass every commit policy is reproduced exactly (the head's outputs do not depend on commits):
    zero        no wait (V5)                                           lag 0
    fixed_k     every event waits k windows (k = 1..D)                 = --commit-lag k
    adaptive_a  RCL's rule: commit 1 at P >= hi, 0 at (provisional 1 and P <= lo), else at D;
                hi / lo = logit tau* +- logit(1 - a); arrival commit when V5 >= hi   = commit: adaptive
    simple_a_K  arrival commit when V5 >= hi, everything else waits K windows (review 11.23's simple baseline)
    window_a    adaptive outputs, but a whole window is released together, when its slowest event commits
                (provisional negatives never commit early, so this is always D: an upper reference only)
    slice_a     the fair slice-granularity analogue: a whole window is released at the first lag at which every one
                of its events is confident either way (P >= hi or P <= lo, symmetric), else at D; every event
                of the window takes its value at that lag (adaptive slicing moved to the output side)
Per policy: IoU / ACC / Pd / Fa at the report threshold (official code paths), normal / dense groups, per-event
decision latency (ms, publish time - event time, compute time excluded), first-detection latency, first-beat recall,
commit sources and flips against V5 by error class (never / same / onset / gap, as tools/error_budget.py).
Also: the information-arrival curve (fixed denominator, k = 0..D) and the adaptive wait against local background
density and target phase (analysis only).
Usage:
    python tools/final_analysis.py --traj-dir run/traj/<name>_val --split val --data-root <root> --window-ms 50 \
        --out run/analysis/<name>_val.json
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.recording_groups import dense_names  # noqa: E402

THRESHOLD = 0.9
ALPHAS = (0.01, 0.05, 0.1, 0.2)
FN_CATS = ("never", "same", "onset", "gap")


def _logit(p):
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_split(traj_dir, data_root, split):
    """Recordings of one split: trajectory + positions / times / target ids from the dataset NPZ."""
    recs = []
    for path in sorted(glob.glob(os.path.join(traj_dir, "*.npz"))):
        name = os.path.basename(path)
        with np.load(path) as tr:
            traj, log_w, labels = np.asarray(tr["traj"], dtype=np.float32), float(tr["log_w"]), np.asarray(tr["labels"])
        with np.load(os.path.join(data_root, split, name)) as d:
            loc, norm = np.asarray(d["ev_loc"]), np.asarray(d["evs_norm"])
        if loc.shape[0] != traj.shape[0]:
            raise RuntimeError("%s：轨迹 %d 个事件、数据 %d 个（加噪评估的轨迹不能用本工具）" % (name, traj.shape[0], loc.shape[0]))
        pos = norm[:, 4] > 0.5
        if not np.array_equal(pos, labels > 0.5):
            raise RuntimeError("%s：轨迹里的标签与数据不一致" % name)
        recs.append({"name": name, "x": loc[:, 0].astype(np.int64), "y": loc[:, 1].astype(np.int64),
                     "t": loc[:, 2].astype(np.int64), "tid": norm[:, 5].astype(np.float64), "pos": pos,
                     "traj": traj, "log_w": log_w})
    if not recs:
        raise RuntimeError("没有轨迹文件：%s" % traj_dir)
    return recs


def report_prob(z, log_w):
    """Calibrated logit -> report-scale probability, float32 as in train_track_head.run_split."""
    import torch
    z32 = np.asarray(z, dtype=np.float32) + np.float32(log_w)
    return torch.sigmoid(torch.from_numpy(np.ascontiguousarray(z32))).numpy()


# ---------------------------------------------------------------------------
# policies: per-event committed lag L (and the value is traj[e, L])
# ---------------------------------------------------------------------------


def available_lag(traj):
    """Last lag with a value (events near the end of a recording are flushed with their last estimate)."""
    return (~np.isnan(traj)).sum(1) - 1


def lag_fixed(traj, k):
    return np.minimum(int(k), available_lag(traj))


def commit_bounds(tau_cal, alpha):
    half = _logit(1.0 - float(alpha))
    return tau_cal - half, tau_cal + half


def lag_adaptive(traj, tau_cal, alpha):
    """RCL's adaptive commit (model/changepoint_layer.py forward_chunk). Returns (L, source):
    source 0 = at arrival, 1 = mid-wait by the head, 2 = deadline D, 3 = flushed at the end of the recording."""
    lo, hi = commit_bounds(tau_cal, alpha)
    depth = traj.shape[1] - 1
    avail = available_lag(traj)
    z0 = traj[:, 0]
    prov = z0 >= tau_cal
    decided = z0 >= hi
    lag = np.where(decided, 0, np.minimum(depth, avail))
    source = np.where(decided, 0, np.where(avail >= depth, 2, 3))
    with np.errstate(invalid="ignore"):
        for d in range(1, depth + 1):
            v = traj[:, d]
            hit = ~decided & (d <= avail) & ((v >= hi) | (prov & (v <= lo)))
            lag[hit] = d
            source[hit] = np.where(d < depth, 1, 2)
            decided |= hit
    return lag, source


def lag_simple(traj, tau_cal, alpha, k):
    """Arrival commit for V5-confident events (same bound as adaptive), all others wait k windows."""
    _, hi = commit_bounds(tau_cal, alpha)
    return np.where(traj[:, 0] >= hi, 0, lag_fixed(traj, k))


def lag_window_shared(lag, window):
    """A window's events are released together, when its slowest event commits."""
    out = np.zeros_like(lag)
    if lag.size:
        top = np.zeros(int(window.max()) + 1, dtype=lag.dtype)
        np.maximum.at(top, window, lag)
        out = top[window]
    return out


def lag_slice(traj, tau_cal, alpha, window):
    """slice_a: per window, the first lag at which all its events are confident (>= hi or <= lo), else D / flush."""
    lo, hi = commit_bounds(tau_cal, alpha)
    depth = traj.shape[1] - 1
    avail = available_lag(traj)
    n_win = int(window.max()) + 1 if window.size else 1
    release = np.full(n_win, depth, dtype=np.int64)
    last = np.zeros(n_win, dtype=np.int64)
    np.maximum.at(last, window, avail)
    release = np.minimum(release, last)                       # a window near the end is flushed at its last lag
    done = np.zeros(n_win, dtype=bool)
    with np.errstate(invalid="ignore"):
        for d in range(0, depth + 1):
            v = traj[:, d]
            unsure = (d <= avail) & ~((v >= hi) | (v <= lo))
            bad = np.zeros(n_win, dtype=bool)
            np.logical_or.at(bad, window, unsure)
            ok = ~bad & ~done & (d <= last)
            release[ok] = d
            done |= ok
    return np.minimum(release[window], avail)


def value_at(traj, lag):
    return traj[np.arange(traj.shape[0]), lag]


# ---------------------------------------------------------------------------
# per-event references from V5 (zero wait)
# ---------------------------------------------------------------------------


NO_WINDOW = np.iinfo(np.int64).max


def target_index(tid, pos):
    """Dense target index 0..G-1 of every target event (one per distinct target id among them; EV-Flying ids may be
    0), -1 elsewhere. Returns (index, G). Vectorised: long EV-Flying recordings have hundreds of targets."""
    idx = np.full(tid.shape[0], -1, dtype=np.int64)
    if not pos.any():
        return idx, 0
    _, inv = np.unique(tid[pos], return_inverse=True)
    idx[pos] = inv
    return idx, int(inv.max()) + 1


def first_hit_window(win, hit, g, n_targets):
    """[G] first window with a hit of each target (NO_WINDOW when never hit)."""
    first = np.full(max(n_targets, 1), NO_WINDOW, dtype=np.int64)
    if hit.any():
        np.minimum.at(first, g[hit], win[hit])
    return first


def fn_categories(win, pos, pred, tid):
    """Per event: 0 = not a V5 miss, 1 never / 2 same / 3 onset / 4 gap (tools/error_budget.py definitions)."""
    cat = np.zeros(pos.shape[0], dtype=np.int8)
    fn = pos & ~pred
    if not fn.any():
        return cat
    g, n = target_index(tid, pos)
    tp = pos & pred
    first = first_hit_window(win, tp, g, n)
    stride = int(win.max()) + 1
    hits = np.unique(g[tp] * stride + win[tp])                  # (target, window) pairs with a hit
    gi, w = g[fn], win[fn]
    same = np.isin(gi * stride + w, hits)
    cat[fn] = np.where(first[gi] == NO_WINDOW, 1, np.where(same, 2, np.where(w < first[gi], 3, 4))).astype(np.int8)
    return cat


def first_beat_mask(win, pos, pred, tid):
    """Target events in or before V5's first detection window of their target (targets V5 detects at some point)."""
    out = np.zeros(pos.shape[0], dtype=bool)
    g, n = target_index(tid, pos)
    first = first_hit_window(win, pos & pred, g, n)
    sel = pos & (first[np.maximum(g, 0)] != NO_WINDOW)
    out[sel] = win[sel] <= first[g[sel]]
    return out


def local_background(x, y, win, pos, radius=8):
    """Per event: background events of the same window within +-radius px (box), from a per-window count image."""
    out = np.zeros(x.shape[0], dtype=np.int32)
    bg = ~pos
    h, w = int(y.max()) + 1 if y.size else 1, int(x.max()) + 1 if x.size else 1
    order = np.argsort(win, kind="stable")
    bounds = np.searchsorted(win[order], np.arange(int(win.max()) + 2 if win.size else 1))
    for k in range(len(bounds) - 1):
        idx = order[bounds[k]:bounds[k + 1]]
        if idx.size == 0:
            continue
        img = np.zeros((h + 1, w + 1), dtype=np.int32)
        sel = idx[bg[idx]]
        np.add.at(img, (y[sel] + 1, x[sel] + 1), 1)
        c = img.cumsum(0).cumsum(1)
        yy, xx = y[idx], x[idx]
        y0, y1 = np.clip(yy - radius, 0, h), np.clip(yy + radius + 1, 0, h)
        x0, x1 = np.clip(xx - radius, 0, w), np.clip(xx + radius + 1, 0, w)
        out[idx] = c[y1, x1] - c[y0, x1] - c[y1, x0] + c[y0, x0]
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def counts(pred, pos):
    return np.array([int((pred & pos).sum()), int((pred & ~pos).sum()), int((~pred & pos).sum())], dtype=np.int64)


def iou_of(c):
    return float(c[0]) / float(c.sum()) if c.sum() else float("nan")


class PolicyStats(object):
    """Accumulates one policy over the recordings of a split."""

    def __init__(self, name, window_ms, full=True, sensor=(260, 346)):
        from utils.stream_metrics import FastRoc
        self.name, self.window_ms = name, float(window_ms)
        self.c = np.zeros(3, dtype=np.int64)
        self.groups = {"normal": np.zeros(3, dtype=np.int64), "dense": np.zeros(3, dtype=np.int64)}
        self.roc = FastRoc(50, 1e-4, int(sensor[0]), int(sensor[1])) if full else None   # pd_detT 50 ms, as configs
        self.lat_t, self.lat_all, self.wait_t, self.first_det, self.n_targets = [], [], [], [], 0
        self.fb = np.zeros(2, dtype=np.int64)
        self.flips = {"rescued": {c: 0 for c in FN_CATS}, "suppressed": 0, "fp_removed": 0, "fp_added": 0}
        self.sources = np.zeros(4, dtype=np.int64)
        self.rescued_wait = []

    def add(self, rec, prob, lag, ref, source=None, dense=False):
        pos = rec["pos"]
        pred = prob >= np.float32(THRESHOLD)
        c = counts(pred, pos)
        self.c += c
        self.groups["dense" if dense else "normal"] += c
        if self.roc is not None:
            self.roc.update(rec["t"], prob, rec["tid"], pos.astype(np.float32), rec["x"], rec["y"], THRESHOLD)
        W = self.window_ms
        win = ref["win"]
        publish = (win + 1 + lag) * W                       # end of the event's window + waited windows
        lat = publish - rec["t"]
        self.lat_all.append(lat)
        self.lat_t.append(lat[pos])
        self.wait_t.append(lag[pos])
        g, n = ref["target"], ref["n_targets"]               # first correct positive output about each target
        self.n_targets += n
        if n:
            t_first = np.full(n, np.inf)
            np.minimum.at(t_first, g[pos], rec["t"][pos].astype(np.float64))
            first_pub = np.full(n, np.inf)
            ok = pos & pred
            np.minimum.at(first_pub, g[ok], publish[ok].astype(np.float64))
            det = np.isfinite(first_pub)
            self.first_det.extend((first_pub[det] - t_first[det]).tolist())
        fb = ref["first_beat"]
        self.fb += np.array([int((pred & fb).sum()), int(fb.sum())])
        base = ref["pred0"]
        up, down = ~base & pred, base & ~pred
        for i, name in enumerate(FN_CATS, start=1):
            self.flips["rescued"][name] += int((up & pos & (ref["fn_cat"] == i)).sum())
        self.flips["suppressed"] += int((down & pos).sum())
        self.flips["fp_removed"] += int((down & ~pos).sum())
        self.flips["fp_added"] += int((up & ~pos).sum())
        self.rescued_wait.append(lag[up & pos])
        if source is not None:
            self.sources += np.bincount(source[pos], minlength=4)[:4]

    def result(self):
        lat_t = np.concatenate(self.lat_t) if self.lat_t else np.zeros(0)
        lat_all = np.concatenate(self.lat_all) if self.lat_all else np.zeros(0)
        wait_t = np.concatenate(self.wait_t) if self.wait_t else np.zeros(0)
        rw = np.concatenate(self.rescued_wait) if self.rescued_wait else np.zeros(0)
        fd = np.asarray(self.first_det)
        tp, fp, fn = (int(v) for v in self.c)
        out = {"iou": iou_of(self.c), "acc": tp / float(max(tp + fn, 1)), "tp": tp, "fp": fp, "fn": fn,
               "groups": {g: iou_of(v) for g, v in self.groups.items()},
               "latency_ms": {"target_mean": float(lat_t.mean()) if lat_t.size else None,
                              "target_median": float(np.median(lat_t)) if lat_t.size else None,
                              "target_p90": float(np.percentile(lat_t, 90)) if lat_t.size else None,
                              "all_mean": float(lat_all.mean()) if lat_all.size else None},
               "wait_windows": {"target_mean": float(wait_t.mean()) if wait_t.size else None,
                                "target_zero_frac": float((wait_t == 0).mean()) if wait_t.size else None,
                                "rescued_mean": float(rw.mean()) if rw.size else None},
               "first_detection_ms": {"median": float(np.median(fd)) if fd.size else None,
                                      "p90": float(np.percentile(fd, 90)) if fd.size else None,
                                      "detected_frac": fd.size / float(max(self.n_targets, 1))},
               "first_beat_recall": self.fb[0] / float(max(self.fb[1], 1)),
               "flips": self.flips}
        if self.roc is not None:
            pd, fa = self.roc.result()
            out.update(pd=float(pd), fa=float(fa))
        if self.sources.sum():
            s = self.sources / float(self.sources.sum())
            out["target_commit_source"] = {"arrival": float(s[0]), "mid_wait": float(s[1]), "deadline": float(s[2]),
                                           "flushed": float(s[3])}
        return out


def references(rec, tau_cal, window_ms):
    win = rec["t"] // int(window_ms)
    pred0 = report_prob(rec["traj"][:, 0], rec["log_w"]) >= np.float32(THRESHOLD)
    target, n_targets = target_index(rec["tid"], rec["pos"])
    return {"win": win, "pred0": pred0, "fn_cat": fn_categories(win, rec["pos"], pred0, rec["tid"]),
            "first_beat": first_beat_mask(win, rec["pos"], pred0, rec["tid"]), "target": target,
            "n_targets": n_targets}


def policy_table(depth, alphas=ALPHAS, simple_k=(1, 3, 5)):
    names = ["zero"] + ["fixed_%d" % k for k in range(1, depth + 1)]
    names += ["adaptive_%g" % a for a in alphas] + ["window_%g" % a for a in alphas] + ["slice_%g" % a for a in alphas]
    names += ["simple_%g_%d" % (a, k) for a in alphas for k in sorted(set(list(simple_k) + [depth]))]
    return names


def policy_lag(name, traj, tau_cal, win):
    """(L, source or None) of a named policy."""
    if name == "zero":
        return np.zeros(traj.shape[0], dtype=np.int64), None
    kind, _, rest = name.partition("_")
    if kind == "fixed":
        return lag_fixed(traj, int(rest)), None
    if kind == "adaptive":
        return lag_adaptive(traj, tau_cal, float(rest))
    if kind == "window":
        lag, source = lag_adaptive(traj, tau_cal, float(rest))
        return lag_window_shared(lag, win), None
    if kind == "slice":
        return lag_slice(traj, tau_cal, float(rest), win), None
    if kind == "simple":
        a, k = rest.split("_")
        return lag_simple(traj, tau_cal, float(a), int(k)), None
    raise ValueError(name)


def analyse(recs, dense, window_ms, alphas=ALPHAS, full=True, sensor=(260, 346)):
    depth = recs[0]["traj"].shape[1] - 1
    log_w = recs[0]["log_w"]
    tau_cal = _logit(THRESHOLD) - log_w
    names = policy_table(depth, alphas)
    stats = {n: PolicyStats(n, window_ms, full=full, sensor=sensor) for n in names}
    density_rows = []
    for rec in recs:
        ref = references(rec, tau_cal, window_ms)
        is_dense = rec["name"] in dense
        for n in names:
            lag, source = policy_lag(n, rec["traj"], tau_cal, ref["win"])
            prob = report_prob(value_at(rec["traj"], lag), rec["log_w"])
            stats[n].add(rec, prob, lag, ref, source, is_dense)
        lag05, src05 = lag_adaptive(rec["traj"], tau_cal, 0.05)
        pos = rec["pos"]
        bgc = local_background(rec["x"], rec["y"], ref["win"], pos)
        phase = np.where(ref["first_beat"], 0, np.where(ref["fn_cat"] == 4, 2, np.where(ref["fn_cat"] == 1, 3, 1)))
        density_rows.append((lag05[pos], src05[pos], bgc[pos], phase[pos], np.full(int(pos.sum()), is_dense)))
    result = {"window_ms": float(window_ms), "D": depth, "tau_cal": tau_cal, "log_w": log_w,
              "n_events": int(sum(r["traj"].shape[0] for r in recs)), "n_target": int(sum(r["pos"].sum() for r in recs)),
              "dense": sorted(dense), "policies": {n: s.result() for n, s in stats.items()}}
    result["info_arrival"] = [dict(k=k, ms=k * float(window_ms), **{key: result["policies"][n][key] for key in
                                   ("iou", "groups", "flips", "first_beat_recall")})
                              for k, n in enumerate(["zero"] + ["fixed_%d" % k for k in range(1, depth + 1)])]
    result["wait_vs_context"] = wait_vs_context(density_rows)
    return result


def wait_vs_context(rows):
    """Adaptive (alpha 0.05) target waits by local background quartile and by target phase (analysis only)."""
    lag = np.concatenate([r[0] for r in rows])
    src = np.concatenate([r[1] for r in rows])
    bg = np.concatenate([r[2] for r in rows])
    phase = np.concatenate([r[3] for r in rows])
    dense = np.concatenate([r[4] for r in rows])
    out = {"by_background": [], "by_phase": {}, "by_group": {}}
    if lag.size == 0:
        return out
    edges = np.unique(np.percentile(bg, [0, 25, 50, 75, 100]))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (bg >= lo) & ((bg <= hi) if hi == edges[-1] else (bg < hi))
        if sel.any():
            out["by_background"].append({"bg_lo": float(lo), "bg_hi": float(hi), "n": int(sel.sum()),
                                         "mean_wait": float(lag[sel].mean()),
                                         "arrival_frac": float((src[sel] == 0).mean())})
    for code, name in enumerate(("first_beat", "tracked", "gap", "never_detected")):
        sel = phase == code
        if sel.any():
            out["by_phase"][name] = {"n": int(sel.sum()), "mean_wait": float(lag[sel].mean()),
                                     "arrival_frac": float((src[sel] == 0).mean())}
    for name, sel in (("normal", ~dense), ("dense", dense)):
        if sel.any():
            out["by_group"][name] = {"n": int(sel.sum()), "mean_wait": float(lag[sel].mean()),
                                     "arrival_frac": float((src[sel] == 0).mean())}
    return out


def markdown(result, title):
    p = result["policies"]
    lines = ["### %s（%d ms 窗，D = %d，%d 个事件 / %d 个目标事件）" % (title, result["window_ms"], result["D"],
                                                                result["n_events"], result["n_target"]),
             "", "| 策略 | IoU | ACC | Pd | Fa | 普通 / 极密 | 目标事件延迟 均值 / 中位 / p90 ms | 目标平均等待（窗） | 首检中位 ms | 第一拍召回 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    show = ["zero"] + ["fixed_%d" % k for k in (1, 3, 5, result["D"])] + ["adaptive_0.05", "simple_0.05_%d" % result["D"],
                                                                         "window_0.05", "slice_0.05", "slice_0.2"]
    for n in show:
        if n not in p:
            continue
        r = p[n]
        lines.append("| %s | %.4f | %.4f | %s | %s | %.4f / %s | %.1f / %.1f / %.1f | %.2f | %s | %.3f |" % (
            n, r["iou"], r["acc"], "%.4f" % r["pd"] if "pd" in r else "-", "%.3e" % r["fa"] if "fa" in r else "-",
            r["groups"]["normal"], "%.4f" % r["groups"]["dense"] if r["groups"]["dense"] == r["groups"]["dense"] else "-",
            r["latency_ms"]["target_mean"], r["latency_ms"]["target_median"], r["latency_ms"]["target_p90"],
            r["wait_windows"]["target_mean"],
            "%.1f" % r["first_detection_ms"]["median"] if r["first_detection_ms"]["median"] is not None else "-",
            r["first_beat_recall"]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--traj-dir", required=True)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--window-ms", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--no-roc", action="store_true", help="skip Pd / Fa (faster)")
    parser.add_argument("--height", type=int, default=260, help="sensor height (px): Fa and the dense rule")
    parser.add_argument("--width", type=int, default=346, help="sensor width (px)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    recs = load_split(args.traj_dir, args.data_root, args.split)
    names = dense_names(args.data_root, (args.split,), args.height, args.width)
    result = analyse(recs, set(names[args.split]), args.window_ms, full=not args.no_roc, sensor=(args.height, args.width))
    result.update(traj_dir=args.traj_dir, split=args.split)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    md = markdown(result, args.title or os.path.basename(args.traj_dir.rstrip("/\\")))
    with open(os.path.splitext(args.out)[0] + ".md", "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print(md)
    print("ANALYSIS FINISHED:", args.out, flush=True)


if __name__ == "__main__":
    main()
