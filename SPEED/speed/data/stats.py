import numpy as np

QUANTILES = (10, 50, 90)


def _pct(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"p%d" % q: None for q in QUANTILES}
    return {"p%d" % q: float(np.percentile(values, q)) for q in QUANTILES}


def object_window_rows(stream, window_us):
    """One row per (target, window) with >= 1 event: [id, window, n_events, extent_x, extent_y, cx, cy, cls]."""
    target = stream.label == 1
    if not target.any():
        return np.zeros((0, 8))
    ids = stream.target_id[target]
    win = stream.t[target] // int(window_us)
    x, y, cls = stream.x[target], stream.y[target], stream.cls[target]
    order = np.lexsort((win, ids))
    ids, win, x, y, cls = ids[order], win[order], x[order], y[order], cls[order]
    starts = np.flatnonzero(np.r_[True, (np.diff(ids) != 0) | (np.diff(win) != 0)])
    ends = np.r_[starts[1:], ids.size]
    counts = (ends - starts).astype(np.float64)
    xmin, xmax = np.minimum.reduceat(x, starts), np.maximum.reduceat(x, starts)
    ymin, ymax = np.minimum.reduceat(y, starts), np.maximum.reduceat(y, starts)
    cx = np.add.reduceat(x.astype(np.float64), starts) / counts
    cy = np.add.reduceat(y.astype(np.float64), starts) / counts
    return np.stack([ids[starts], win[starts], counts, xmax - xmin + 1, ymax - ymin + 1, cx, cy, cls[starts]], 1)


def motion_from_rows(rows, window_us):
    """Speeds (px/s), relative speeds (body lengths/s) from consecutive windows, and track durations (s)."""
    window_s = float(window_us) * 1e-6
    speeds, rel, durations = [], [], []
    size = np.maximum(rows[:, 3], rows[:, 4]) if rows.size else rows
    for ident in np.unique(rows[:, 0]) if rows.size else []:
        m = rows[:, 0] == ident
        s, z = rows[m], size[m]
        durations.append((s[:, 1].max() - s[:, 1].min() + 1) * window_s)
        step = np.diff(s[:, 1]) == 1
        disp = np.hypot(np.diff(s[:, 5]), np.diff(s[:, 6]))[step]
        speeds.extend(disp / window_s)
        rel.extend((disp / np.maximum((z[1:] + z[:-1]) / 2.0, 1.0)[step]) / window_s)
    return np.asarray(speeds), np.asarray(rel), np.asarray(durations)


def recording_background(stream, hot_hz=1000.0):
    duration_s = max(stream.span_us, 1) * 1e-6
    bg = stream.label == 0
    pixels = stream.width * stream.height
    rate = np.bincount(stream.y[bg] * stream.width + stream.x[bg], minlength=pixels) / duration_s
    return {"bg_rate_per_px_s": float(np.count_nonzero(bg)) / (pixels * duration_s),
            "bg_rate_p99_per_px_s": float(np.percentile(rate, 99)),
            "hot_pixels": int(np.count_nonzero(rate > hot_hz)),
            "duration_s": duration_s}


def split_statistics(streams, window_us=50000):
    """Scale statistics of a split in physical units; the analysis window only sets the motion sampling step."""
    rows_all, per_rec = [], []
    n_events = n_target = 0
    speeds, rel, durations = [], [], []
    for k, stream in enumerate(streams):
        rows = object_window_rows(stream, window_us)
        if rows.size:
            rows = rows.copy()
            rows[:, 0] += k * 10 ** 7  # keep ids unique across recordings
        s, r, d = motion_from_rows(rows, window_us)
        speeds.append(s)
        rel.append(r)
        durations.append(d)
        rows_all.append(rows)
        bg = recording_background(stream)
        bg.update(stream.summary())
        per_rec.append(bg)
        n_events += stream.n_events
        n_target += int(np.count_nonzero(stream.label == 1))
    rows = np.concatenate([r for r in rows_all if r.size]) if any(r.size for r in rows_all) else np.zeros((0, 8))
    window_s = float(window_us) * 1e-6
    size = np.maximum(rows[:, 3], rows[:, 4]) if rows.size else []
    classes = {}
    if rows.size:
        for c in np.unique(rows[:, 7]):
            m = rows[:, 7] == c
            classes[str(int(c))] = {"targets": int(np.unique(rows[m, 0]).size), "size_px": _pct(size[m])}
    return {
        "analysis_window_us": int(window_us),
        "n_recordings": len(per_rec),
        "duration_s": float(sum(r["duration_s"] for r in per_rec)),
        "n_events": int(n_events),
        "target_event_share": float(n_target) / max(n_events, 1),
        "n_targets": int(np.unique(rows[:, 0]).size) if rows.size else 0,
        "target_size_px": _pct(size),
        "target_events_per_s": _pct(rows[:, 2] / window_s if rows.size else []),
        "target_speed_px_s": _pct(np.concatenate(speeds) if speeds else []),
        "target_rel_speed_per_s": _pct(np.concatenate(rel) if rel else []),
        "target_duration_s": _pct(np.concatenate(durations) if durations else []),
        "bg_rate_per_px_s": _pct([r["bg_rate_per_px_s"] for r in per_rec]),
        "bg_rate_p99_per_px_s": _pct([r["bg_rate_p99_per_px_s"] for r in per_rec]),
        "hot_pixels_max": int(max((r["hot_pixels"] for r in per_rec), default=0)),
        "per_class": classes,
        "recordings": per_rec,
    }
