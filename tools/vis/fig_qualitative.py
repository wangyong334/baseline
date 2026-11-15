"""Qualitative comparison figure (V20, 11-15): rows = scenes, columns = Input | methods ... | GT, each panel the
events of one recording in (x, y, t) drawn with tools/vis/evuav_vis.py (the official EV-SpSegNet tool, extended):
correct green, false red, missed blue, background grey (background subsampled for display only), optional zoom box
with a zoomed inset per scene.
Usage:
    python tools/vis/fig_qualitative.py --data-root <EV-UAV-dataset> --out fig_qual.pdf \
        --scene test_022 --scene test_009:0-4000 ... \
        --column "EV-SpSegNet=prob:<dump dir>" --column "PACT=prob:<dump dir>" \
        --column "Streaming SNN=prior:<export dir>" --column "SDNet=final:<export dir>" \
        [--zoom test_022=150,230,90,170,2000,4000] [--view 18,-58] [--max-bg 60000] [--result-bg none|faint]
    kinds: prob = dump_competitor.py output (target iff prob >= 0.9); prior / final = export_preds.py output (target
    iff logit >= tau of its meta.json); host = plug-in prior dir of plugin/dump_*_z0.py (RVT / SAST / YOLOv10: target
    iff the event's box score >= the host's val threshold thr* in host_meta.json, i.e. the host's own detections).
    A column whose directory is missing is drawn as an empty placeholder.
    Scenes and time windows come from tools/vis/scene_stats.py (ranked by stated criteria); the Input column shows all
    events (background subsampled), result and GT columns only the decided / labelled targets unless --result-bg faint.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evuav_vis as V  # noqa: E402

HEADER = dict(boxstyle="square,pad=0.25", facecolor="#BDD7EE", edgecolor="#2F5597", lw=1.0)


def load_events(path):
    d = np.load(path)
    loc, norm = d["ev_loc"], d["evs_norm"]
    return loc[:, 0].astype(np.float32), loc[:, 1].astype(np.float32), loc[:, 2].astype(np.float64), norm[:, 4] > 0.5


def prediction(kind, folder, name, n, split="test"):
    """Binary per-event prediction of one method for one recording, or None if it is not available."""
    f = os.path.join(folder, "z0_" + split, name) if kind == "host" else os.path.join(folder, name)
    if not os.path.exists(f):
        return None
    d = np.load(f)
    if kind == "prob":
        p = d["prob"] >= 0.9
    elif kind == "host":                                   # plug-in host: box score >= its own val threshold thr*
        with open(os.path.join(folder, "host_meta.json")) as fh:
            p = d["score"] >= json.load(fh)["thr"]
    else:
        with open(os.path.join(folder, "meta.json")) as fh:
            tau = json.load(fh)["tau"]
        p = d["prior_logit" if kind == "prior" else "final_logit"] >= tau
    if len(p) != n:
        raise RuntimeError("%s: %d predictions for %d events" % (f, len(p), n))
    return p


def parse_scene(s):
    name, _, win = s.partition(":")
    name = name if name.endswith(".npz") else name + ".npz"
    t0, t1 = (float(v) for v in win.split("-")) if win else (None, None)
    return name, t0, t1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--scene", action="append", required=True, help="name[:t0-t1 ms]")
    ap.add_argument("--column", action="append", default=[], help="TITLE=kind:DIR, kind in prob / prior / final")
    ap.add_argument("--zoom", action="append", default=[], help="name=x0,x1,y0,y1,t0,t1 (px, ms)")
    ap.add_argument("--view", default="18,-58")
    ap.add_argument("--max-bg", type=int, default=60000)
    ap.add_argument("--result-bg", choices=("none", "faint"), default="none",
                    help="background events in the result and GT columns: none (SpTopoNet style) or faint grey")
    ap.add_argument("--panel", default="2.3,2.1", help="panel width,height in inches")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    view = tuple(float(v) for v in args.view.split(","))
    pw, ph = (float(v) for v in args.panel.split(","))
    cols = [("Input", "input", None)]
    for c in args.column:
        title, _, spec = c.partition("=")
        kind, _, folder = spec.partition(":")
        cols.append((title, kind, folder))
    cols.append(("GT", "gt", None))
    zooms = {}
    for z in args.zoom:
        n, _, box = z.partition("=")
        zooms[n if n.endswith(".npz") else n + ".npz"] = tuple(float(v) for v in box.split(","))
    scenes = [parse_scene(s) for s in args.scene]
    fig = plt.figure(figsize=(pw * len(cols), ph * len(scenes) + 0.6))
    top, bottom = 1.0 - 0.32 / (ph * len(scenes) + 0.6), 0.38 / (ph * len(scenes) + 0.6)
    H = (top - bottom) / len(scenes)
    W = 1.0 / len(cols)
    for j, (title, _, _) in enumerate(cols):
        fig.text((j + 0.5) * W, top + 0.25 * (1 - top), title, ha="center", va="center", fontsize=9, weight="bold",
                 bbox=HEADER)
    for i, (name, t0, t1) in enumerate(scenes):
        x, y, t, lab = load_events(os.path.join(args.data_root, args.split, name))
        keep = np.ones(len(t), bool) if t0 is None else (t >= t0) & (t < t1)
        t_range = None if t0 is None else (t0 / 1000.0, t1 / 1000.0)
        box = zooms.get(name)
        for j, (title, kind, folder) in enumerate(cols):
            rect = [j * W + 0.004, top - (i + 1) * H + 0.004, W - 0.008, H - 0.008]
            ax = fig.add_axes(rect, projection="3d")
            if kind == "input":
                cat = np.full(len(t), "input", dtype=object)
            elif kind == "gt":
                cat = np.where(lab, "target", "background").astype(object)
            else:
                pred = prediction(kind, folder, name, len(t), args.split) if folder else None
                if pred is None:
                    ax.set_axis_off()
                    ax.text2D(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center", color="#888888")
                    continue
                cat = V.outcome(pred, lab)
            if kind != "input" and args.result_bg == "none":
                cat[cat == "background"] = "skip"
            V.draw_events(ax, x[keep], y[keep], t[keep], cat[keep], max_background=args.max_bg, view=view,
                          t_range=t_range)
            if box is not None:
                V.draw_box(ax, box)
                inset = fig.add_axes([rect[0] + 0.02 * W, rect[1] + 0.52 * H, 0.42 * W, 0.42 * H], projection="3d")
                inb = keep & (x >= box[0]) & (x < box[1]) & (y >= box[2]) & (y < box[3]) & (t >= box[4]) & (t < box[5])
                V.draw_events(inset, x[inb], y[inb], t[inb], cat[inb], max_background=args.max_bg, view=view,
                              t_range=(box[4] / 1000.0, box[5] / 1000.0), xy_range=box[:4], labels=False)
                for axis in (inset.xaxis, inset.yaxis, inset.zaxis):
                    axis.pane.set_edgecolor("#202020")
        fig.text(0.002, top - (i + 0.5) * H, "(%s)" % "abcdefgh"[i], fontsize=8, va="center")
    legend = [Patch(color=V.COLOURS[k], label=lbl) for k, lbl in
              (("correct", "Correct"), ("false", "False alarm"), ("missed", "Missed"), ("background", "Background"))]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    root, _ = os.path.splitext(args.out)
    fig.savefig(root + ".pdf", dpi=600)
    fig.savefig(root + ".png", dpi=200)
    print("FIG QUALITATIVE FINISHED %s(.pdf/.png): %d scenes x %d columns" % (root, len(scenes), len(cols)))


if __name__ == "__main__":
    main()
