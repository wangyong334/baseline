"""3-D event-stream figures in the style of the EV-UAV paper: events in (x, t, y), grey = background, red = events
classified (or labelled) as target, one column per method plus ground truth, optional zoom box with an inset."""
import numpy as np

GREY, RED, GREEN = "#9a9a9a", "#d62728", "#2ca02c"


def panel(stream, red_mask, t0_us=None, t1_us=None, title=""):
    t0 = int(stream.t.min()) if t0_us is None else int(t0_us)
    t1 = int(stream.t.max()) + 1 if t1_us is None else int(t1_us)
    keep = (stream.t >= t0) & (stream.t < t1)
    return {"title": title, "x": stream.x[keep], "y": stream.y[keep], "t": stream.t[keep] * 1e-6,
            "red": np.asarray(red_mask, dtype=bool)[keep], "width": stream.width, "height": stream.height}


def auto_zoom(gt_panel, span_s=0.5, margin_px=6):
    """Box (x0, x1, y0, y1, t0, t1): a time slab of span_s seconds centred on the busiest stretch of target events,
    spatially fitted to the target events inside it; None without target events."""
    red = gt_panel["red"]
    if not red.any():
        return None
    x, y, t = gt_panel["x"][red], gt_panel["y"][red], gt_panel["t"][red]
    order = np.sort(t)
    count = np.searchsorted(order, order + span_s) - np.arange(order.size)
    t0 = order[int(np.argmax(count))]
    t1 = t0 + span_s
    m = (t >= t0) & (t <= t1)
    return (max(x[m].min() - margin_px, 0), min(x[m].max() + margin_px, gt_panel["width"] - 1),
            max(y[m].min() - margin_px, 0), min(y[m].max() + margin_px, gt_panel["height"] - 1), t0, t1)


def _scatter(ax, p, rng, max_background, select=None, size=0.3):
    sel = np.ones(p["x"].size, bool) if select is None else select
    bg = np.flatnonzero(sel & ~p["red"])
    if bg.size > max_background:
        bg = rng.choice(bg, max_background, replace=False)
    fg = np.flatnonzero(sel & p["red"])
    ax.computed_zorder = False  # keep target events on top of the background cloud
    ax.scatter(p["x"][bg], p["t"][bg], p["y"][bg], s=size, c=GREY, alpha=0.35, linewidths=0, depthshade=False, zorder=1)
    ax.scatter(p["x"][fg], p["t"][fg], p["y"][fg], s=size * 4, c=RED, linewidths=0, depthshade=False, zorder=2)


def _box_edges(ax, box):
    x0, x1, y0, y1, t0, t1 = box
    for xs, ts, ys in (([x0, x1], [t0, t0], [y0, y0]), ([x0, x1], [t1, t1], [y0, y0]), ([x0, x1], [t0, t0], [y1, y1]),
                       ([x0, x1], [t1, t1], [y1, y1]), ([x0, x0], [t0, t1], [y0, y0]), ([x1, x1], [t0, t1], [y0, y0]),
                       ([x0, x0], [t0, t1], [y1, y1]), ([x1, x1], [t0, t1], [y1, y1]), ([x0, x0], [t0, t0], [y0, y1]),
                       ([x1, x1], [t0, t0], [y0, y1]), ([x0, x0], [t1, t1], [y0, y1]), ([x1, x1], [t1, t1], [y0, y1])):
        ax.plot(xs, ts, ys, color=GREEN, linewidth=0.8)


def render(rows, out_path, zoom=None, elev=18, azim=-62, max_background=60000, panel_size=3.2, dpi=150, seed=0):
    """rows: list of lists of panels (one row per recording / time range); zoom: box, 'auto' or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    fig = plt.figure(figsize=(panel_size * n_cols, panel_size * n_rows))
    for i, row in enumerate(rows):
        box = auto_zoom(row[-1]) if zoom == "auto" else zoom
        for j, p in enumerate(row):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection="3d")
            _scatter(ax, p, rng, max_background)
            ax.set_xlim(0, p["width"])
            ax.set_zlim(p["height"], 0)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
            if i == 0:
                ax.set_title(p["title"], fontsize=10)
            if box is not None:
                _box_edges(ax, box)
                pos = ax.get_position()
                inset = fig.add_axes([pos.x0, pos.y0 + pos.height * 0.58, pos.width * 0.42, pos.height * 0.42],
                                     projection="3d")
                x0, x1, y0, y1, t0, t1 = box
                sel = (p["x"] >= x0) & (p["x"] <= x1) & (p["y"] >= y0) & (p["y"] <= y1) & (p["t"] >= t0) & (p["t"] <= t1)
                _scatter(inset, p, rng, max_background, select=sel, size=2.0)
                inset.patch.set_alpha(0.0)
                inset.set_xlim(x0, x1), inset.set_ylim(t0, t1), inset.set_zlim(y1, y0)
                inset.view_init(elev=elev, azim=azim)
                inset.set_xticks([]), inset.set_yticks([]), inset.set_zticks([])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
