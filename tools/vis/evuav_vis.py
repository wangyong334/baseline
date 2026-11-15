"""Event visualisation for EV-UAV, built on the official EV-SpSegNet tool.

Copied on 11-15 from the EV-SpSegNet repository (utils/vis.py; the PACT repository ships the same file) and extended
in place. Only display choices were added; the unused open3d / cv2 imports of the original were dropped so that the
module runs with matplotlib alone.
    show_points_matplt   the original function, unchanged: one matplotlib scatter3D over (x, y, t), label 1 red,
                         the rest grey, axes off
    draw_events          the same scatter3D with one category per event (PACT colour scheme: correct green, false
                         red, missed blue, background grey), axis panes and X / Y / T labels kept (SpTopoNet style),
                         time in seconds; background events can be subsampled for display only
    outcome              categories of a prediction against the labels
    draw_box             the edges of a zoom box (x, y, t) in a panel
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SENSOR_W, SENSOR_H = 346, 260
GREY = '#A5A5A5'                        # the original tool's colour for unselected events
COLOURS = {"input": GREY, "background": '#C8C8C8', "correct": '#2CA02C', "false": '#D62728', "missed": '#1F5FBF',
           "target": '#2CA02C'}
SIZES = {"input": 0.4, "background": 0.3, "correct": 1.6, "false": 2.0, "missed": 2.0, "target": 1.6}
ORDER = ("input", "background", "correct", "missed", "false", "target")   # drawing order: errors on top


def show_points_matplt(points, label):
    """The original EV-SpSegNet function (points [N, 3] = x, y, t; label 1 red, 0 grey), kept for reference."""
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection="3d")

    events_selected = points[label == 1]
    events_no_selected = points[label == 0]

    ax.scatter3D(events_selected[:, 0], events_selected[:, 1], events_selected[:, 2], '.', s=1, c='r', edgecolor='none')
    ax.scatter3D(events_no_selected[:, 0], events_no_selected[:, 1], events_no_selected[:, 2], '.', s=2, c='#A5A5A5',
                 edgecolor='none')

    ax.set_xlim([0, 350])
    ax.set_ylim([0, 260])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    plt.savefig('img.jpg', dpi=600, bbox_inches='tight', pad_inches=0)
    return fig


def outcome(pred, label):
    """Per-event category of a binary prediction: correct (target found), false, missed, background."""
    pred, label = np.asarray(pred, bool), np.asarray(label, bool)
    cat = np.full(pred.shape, "background", dtype=object)
    cat[pred & label] = "correct"
    cat[pred & ~label] = "false"
    cat[~pred & label] = "missed"
    return cat


def draw_events(ax, x, y, t_ms, cat, max_background=60000, seed=0, view=(18, -58), t_range=None, xy_range=None,
                labels=True, rasterized=True):
    """Scatter the events of one panel. cat: per-event category (keys of COLOURS; any other value, e.g. "skip", is
    not drawn). Categories "input" and "background" are subsampled to at most max_background points for display
    (fixed seed); targets, correct, false and missed events are always drawn. t in seconds on the vertical axis, as in
    the original tool (x, y, t)."""
    rng = np.random.RandomState(seed)
    t = np.asarray(t_ms, np.float64) / 1000.0
    cat = np.asarray(cat, dtype=object)
    for c in ORDER:
        m = np.flatnonzero(cat == c)
        if not len(m):
            continue
        if c in ("input", "background") and max_background and len(m) > max_background:
            m = np.sort(rng.choice(m, max_background, replace=False))
        ax.scatter3D(x[m], y[m], t[m], s=SIZES[c], c=COLOURS[c], edgecolor='none', depthshade=False,
                     rasterized=rasterized)
    x0, x1, y0, y1 = xy_range if xy_range is not None else (0, SENSOR_W, 0, SENSOR_H)
    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])
    if t_range is not None:
        ax.set_zlim(t_range)
    ax.view_init(*view)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(length=0, pad=-4)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor('#9A9A9A')
        axis._axinfo["grid"].update(linewidth=0.3, color='#D0D0D0')
    if labels:
        ax.set_xlabel('X', labelpad=-12, fontsize=7)
        ax.set_ylabel('Y', labelpad=-12, fontsize=7)
        ax.set_zlabel('T', labelpad=-12, fontsize=7)
    return ax


def draw_box(ax, box, colour='#202020', lw=0.7):
    """Edges of the zoom box (x0, x1, y0, y1, t0_ms, t1_ms) in a 3D panel."""
    x0, x1, y0, y1, t0, t1 = box
    t0, t1 = t0 / 1000.0, t1 / 1000.0
    for xs, ys, zs in (([x0, x1], [y0, y0], [t0, t0]), ([x0, x1], [y1, y1], [t0, t0]), ([x0, x0], [y0, y1], [t0, t0]),
                       ([x1, x1], [y0, y1], [t0, t0]), ([x0, x1], [y0, y0], [t1, t1]), ([x0, x1], [y1, y1], [t1, t1]),
                       ([x0, x0], [y0, y1], [t1, t1]), ([x1, x1], [y0, y1], [t1, t1]), ([x0, x0], [y0, y0], [t0, t1]),
                       ([x1, x1], [y0, y0], [t0, t1]), ([x0, x0], [y1, y1], [t0, t1]), ([x1, x1], [y1, y1], [t0, t1])):
        ax.plot(xs, ys, zs, color=colour, lw=lw)


if __name__ == '__main__':
    npz_root = r'/media/admin1/123/EV-UAV-dataset/test/test_021.npz'
    events = np.load(npz_root)
    evs_norm = events['evs_norm']
    evs_loc = events['ev_loc']
    points = evs_loc[:, 0:3]
    label = evs_norm[:, 4]
    show_points_matplt(points, label)
