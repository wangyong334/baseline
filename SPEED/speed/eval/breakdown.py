"""Target-event recall broken down by target speed, cross-track size and class.

Kinematics follow the V4 problem analysis (M1/E): per (target, window) centroid, velocity by central difference
of consecutive window centroids (px/ms), size = median over the track of the p5-p95 extent of events projected on
the normal to the motion (+1 px). Events of tracks seen in a single isolated window have unknown velocity.
"""
import numpy as np

SPEED_EDGES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)   # px per step
SIZE_EDGES = (0.0, 3.0, 6.0, 12.0, 24.0)                     # px


def target_kinematics(stream, window_us=50000):
    """-> per-event arrays vx, vy (px/ms) and size (px); nan for background and undefined values."""
    n = stream.n_events
    vx, vy, size = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    tgt = np.flatnonzero(stream.label == 1)
    if tgt.size == 0:
        return vx, vy, size
    ids, win = stream.target_id[tgt], stream.t[tgt] // int(window_us)
    order = np.lexsort((win, ids))
    gid, gwin = ids[order], win[order]
    start = np.flatnonzero(np.r_[True, (gid[1:] != gid[:-1]) | (gwin[1:] != gwin[:-1])])
    count = np.diff(np.r_[start, order.size])
    xs, ys = stream.x[tgt][order].astype(np.float64), stream.y[tgt][order].astype(np.float64)
    cx, cy = np.add.reduceat(xs, start) / count, np.add.reduceat(ys, start) / count
    g_id, g_win = gid[start], gwin[start]

    # central difference over adjacent windows of the same track
    link = (g_id[1:] == g_id[:-1]) & (g_win[1:] == g_win[:-1] + 1)
    idx = np.arange(start.size)
    lo, hi = idx - np.r_[False, link], idx + np.r_[link, False]
    dt_ms = (g_win[hi] - g_win[lo]) * (window_us / 1000.0)
    ok = hi > lo
    gvx, gvy = np.full(start.size, np.nan), np.full(start.size, np.nan)
    gvx[ok] = (cx[hi[ok]] - cx[lo[ok]]) / dt_ms[ok]
    gvy[ok] = (cy[hi[ok]] - cy[lo[ok]]) / dt_ms[ok]

    widths = {}
    for g in np.flatnonzero((count >= 5) & ok):
        sl = slice(start[g], start[g] + count[g])
        sp = np.hypot(gvx[g], gvy[g])
        if sp < 1e-6:
            w = max(np.percentile(xs[sl], 95) - np.percentile(xs[sl], 5),
                    np.percentile(ys[sl], 95) - np.percentile(ys[sl], 5)) + 1.0
        else:
            proj = xs[sl] * (-gvy[g] / sp) + ys[sl] * (gvx[g] / sp)
            w = np.percentile(proj, 95) - np.percentile(proj, 5) + 1.0
        widths.setdefault(int(g_id[g]), []).append(w)
    track_size = {k: float(np.median(v)) for k, v in widths.items()}

    group = np.empty(order.size, dtype=np.int64)
    group[order] = np.repeat(idx, count)
    vx[tgt], vy[tgt] = gvx[group], gvy[group]
    size[tgt] = np.array([track_size.get(int(k), np.nan) for k in ids])
    return vx, vy, size


def bin_labels(edges):
    return ["%g-%g" % (a, b) for a, b in zip(edges[:-1], edges[1:])] + ["%g+" % edges[-1], "unknown"]


def bin_index(values, edges):
    """Bin index per value; len(edges) for nan (unknown)."""
    v = np.asarray(values, dtype=np.float64)
    out = np.clip(np.searchsorted(np.asarray(edges, np.float64), v, side="right") - 1, 0, len(edges) - 1)
    out[~np.isfinite(v)] = len(edges)
    return out


class RecallBreakdown(object):
    """Accumulates target-event recall per bin for several readouts (and keep rate against a reference readout)."""

    def __init__(self, readouts, reference=None, step_ms=50.0, window_us=50000):
        self.readouts, self.reference = list(readouts), reference
        self.step_ms, self.window_us = float(step_ms), int(window_us)
        self.tables = {}
        self.false_events = {r: 0 for r in self.readouts}
        self.background_events = 0

    def _add(self, table, key, bins, n_bins, detected, ref):
        t = self.tables.setdefault((table, key), {"n": np.zeros(n_bins, np.int64),
                                                   "hit": {r: np.zeros(n_bins, np.int64) for r in self.readouts},
                                                   "ref": np.zeros(n_bins, np.int64),
                                                   "kept": {r: np.zeros(n_bins, np.int64) for r in self.readouts}})
        t["n"] += np.bincount(bins, minlength=n_bins)
        for r in self.readouts:
            t["hit"][r] += np.bincount(bins, weights=detected[r], minlength=n_bins).astype(np.int64)
            if ref is not None:
                t["kept"][r] += np.bincount(bins, weights=detected[r] & ref, minlength=n_bins).astype(np.int64)
        if ref is not None:
            t["ref"] += np.bincount(bins, weights=ref, minlength=n_bins).astype(np.int64)

    def update(self, stream, decisions):
        """decisions: readout name -> bool array over all events of the stream."""
        tgt = stream.label == 1
        bg = ~tgt
        self.background_events += int(bg.sum())
        for r in self.readouts:
            self.false_events[r] += int((decisions[r] & bg).sum())
        if not tgt.any():
            return
        vx, vy, size = target_kinematics(stream, self.window_us)
        det = {r: decisions[r][tgt] for r in self.readouts}
        ref = det[self.reference] if self.reference else None
        per_step = self.step_ms
        speed = np.hypot(vx[tgt], vy[tgt]) * per_step
        cheb = np.maximum(np.abs(vx[tgt]), np.abs(vy[tgt])) * per_step
        cls = stream.cls[tgt]
        n_s, n_z = len(SPEED_EDGES) + 1, len(SIZE_EDGES) + 1
        s_bin, c_bin, z_bin = bin_index(speed, SPEED_EDGES), bin_index(cheb, SPEED_EDGES), bin_index(size[tgt], SIZE_EDGES)
        for key in ["all"] + ["class%d" % c for c in np.unique(cls)]:
            m = np.ones(cls.size, bool) if key == "all" else cls == int(key[5:])
            sub = {r: det[r][m] for r in self.readouts}
            sref = ref[m] if ref is not None else None
            self._add("speed", key, s_bin[m], n_s, sub, sref)
            self._add("cheb", key, c_bin[m], n_s, sub, sref)
            self._add("size", key, z_bin[m], n_z, sub, sref)

    def result(self):
        labels = {"speed": bin_labels(SPEED_EDGES), "cheb": bin_labels(SPEED_EDGES), "size": bin_labels(SIZE_EDGES)}
        out = {"readouts": self.readouts, "reference": self.reference, "step_ms": self.step_ms,
               "speed_unit": "px per step", "tables": {},
               "false_events": self.false_events, "background_events": self.background_events}
        for (table, key), t in sorted(self.tables.items()):
            rows = []
            for i, lab in enumerate(labels[table]):
                n = int(t["n"][i])
                row = {"bin": lab, "events": n,
                       "recall": {r: (float(t["hit"][r][i]) / n if n else None) for r in self.readouts}}
                if self.reference:
                    nr = int(t["ref"][i])
                    row["kept_vs_reference"] = {r: (float(t["kept"][r][i]) / nr if nr else None)
                                                for r in self.readouts}
                rows.append(row)
            out["tables"].setdefault(table, {})[key] = rows
        return out


def format_table(result, table, key):
    """Recall per bin, then (with a reference) the fraction of reference-detected events each readout keeps."""
    names = result["readouts"]
    cols = " ".join("%9s" % n[:9] for n in names)
    head = "%-9s %9s | recall %s" % (table, "events", cols)
    if result["reference"]:
        head += " | kept %s" % cols
    lines = [head]
    fmt = lambda v: "%9s" % "-" if v is None else "%9.3f" % v  # noqa: E731
    for row in result["tables"][table][key]:
        if not row["events"]:
            continue
        line = "%-9s %9d |        %s" % (row["bin"], row["events"], " ".join(fmt(row["recall"][n]) for n in names))
        if result["reference"]:
            line += " |      %s" % " ".join(fmt(row["kept_vs_reference"][n]) for n in names)
        lines.append(line)
    return "\n".join(lines)


class MotionAccuracy(object):
    """Learned motion vs label velocity on target events, by speed (px per step): direction error and speed ratio of the
    most probable anchor (with its residual), |error|, and coverage = how often the label's own anchor (the dominant
    one of its soft target) is among the anchors the verifier kept for the event."""

    def __init__(self, table, step_ms=50.0, window_us=50000):
        self.table = table
        self.step_ms, self.window_us = float(step_ms), int(window_us)
        self.rows, self.cover = [], []

    def update(self, stream, velocity, anchors):
        """velocity [N, 2] (vy, vx px/ms) and kept anchor indices [N, K] per event, file order."""
        import torch
        vx, vy, _ = target_kinematics(stream, self.window_us)
        sel = (stream.label == 1) & np.isfinite(vx)
        if not sel.any():
            return
        true = np.c_[vy[sel], vx[sel]] * self.step_ms
        _, dom, _ = self.table.soft_target(torch.from_numpy(true))
        self.rows.append(np.c_[true, velocity[sel] * self.step_ms])
        self.cover.append((anchors[sel] == dom.numpy().reshape(-1, 1)).any(1))

    def result(self):
        if not self.rows:
            return {}
        r = np.concatenate(self.rows).astype(np.float64)
        cover = np.concatenate(self.cover)
        true, est = r[:, 0:2], r[:, 2:4]
        sp_t, sp_e = np.hypot(*true.T), np.hypot(*est.T)
        err = np.hypot(*(est - true).T)
        cos = (est * true).sum(1) / np.maximum(sp_t * sp_e, 1e-12)
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        bins = bin_index(sp_t, SPEED_EDGES)
        out = {"events": int(len(r)), "unit": "px per step", "bins": []}
        for i, lab in enumerate(bin_labels(SPEED_EDGES)):
            m = bins == i
            if not m.any():
                continue
            mov = m & (sp_t >= 1.0)
            out["bins"].append({"bin": lab, "events": int(m.sum()),
                                "angle_median": float(np.median(ang[mov])) if mov.any() else None,
                                "speed_ratio_median": float(np.median(sp_e[mov] / sp_t[mov])) if mov.any() else None,
                                "error_median": float(np.median(err[m])), "coverage": float(np.mean(cover[m]))})
        mov = sp_t >= 1.0
        out.update(angle_median=float(np.median(ang[mov])) if mov.any() else None,
                   speed_ratio_median=float(np.median(sp_e[mov] / sp_t[mov])) if mov.any() else None,
                   error_median=float(np.median(err)), coverage=float(np.mean(cover)))
        return out
