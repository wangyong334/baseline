"""Mechanism figure (V20, 11-15): what SDM does over time, for one target of one recording.
    (a) time-space view (t horizontal, x vertical) of the events around the target, decided by the streaming SNN alone
    (b) the same events decided by SDNet; events SDM corrected are ringed, events it made wrong crossed
    (c) decision variable U - tau of the events that got a decision neuron, column by column (prior, then the event's
        own slice, then each following slice), with the calibrated bound +-b that collapses toward the deadline; a dot
        marks the column where each event was decided
Inputs: the recording (dataset npz) and an export_preds.py directory (prior, final, commit column, trajectories,
bounds in meta.json). The target defaults to the one with the most events SDNet corrects.
Usage: python tools/vis/fig_mechanism.py --data-root <EV-UAV-dataset> --export <dir> --scene test_019 --out fig_mech
       [--tid 3] [--window t0-t1 ms | --span 1000] [--margin 25] [--max-lines 60]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evuav_vis as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--export", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tid", type=int, default=None)
    ap.add_argument("--window", default=None, help="t0-t1 in ms (default: around the target's corrected events)")
    ap.add_argument("--span", type=float, default=1000.0, help="length of the automatic window (ms)")
    ap.add_argument("--margin", type=float, default=25.0, help="px around the target's events")
    ap.add_argument("--max-lines", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    name = args.scene if args.scene.endswith(".npz") else args.scene + ".npz"
    d = np.load(os.path.join(args.data_root, args.split, name))
    loc, norm = d["ev_loc"], d["evs_norm"]
    x, y, t = loc[:, 0].astype(float), loc[:, 1].astype(float), loc[:, 2].astype(float)
    lab, tid = norm[:, 4] > 0.5, norm[:, 5].astype(np.int64)
    with open(os.path.join(args.export, "meta.json")) as fh:
        meta = json.load(fh)
    tau, wms = meta["tau"], meta["window_ms"]
    bounds = np.array([np.inf if b is None else b for b in meta["bounds"]])
    p = np.load(os.path.join(args.export, name))
    prior, final = p["prior_logit"] >= tau, p["final_logit"] >= tau
    commit, uidx, uZ = p["commit"].astype(int), p["unit_idx"], p["unit_Z"].astype(float)
    fixed = (prior != lab) & (final == lab)
    k = args.tid
    if k is None:
        cand = np.unique(tid[lab])
        k = int(max(cand, key=lambda c: (fixed & (tid == c)).sum()))
    mine = lab & (tid == k)
    if args.window:
        t0, t1 = (float(v) for v in args.window.split("-"))
    else:                                                  # the span where SDNet corrects most events near it
        tf = t[fixed & (lab & (tid == k) | ~lab)]
        near = np.abs(x[fixed] - np.median(x[mine])) < 80
        tf = t[fixed][near] if near.any() else t[mine]
        starts = np.arange(t[mine].min(), max(t[mine].max() - args.span, t[mine].min()) + 1, 50.0)
        best = max(starts, key=lambda s0: (((tf >= s0) & (tf < s0 + args.span)).sum(),
                                            ((t[mine] >= s0) & (t[mine] < s0 + args.span)).sum()))
        t0, t1 = best, best + args.span
    inwin = mine & (t >= t0) & (t < t1)
    x0, x1 = x[inwin].min() - args.margin, x[inwin].max() + args.margin
    y0, y1 = y[inwin].min() - args.margin, y[inwin].max() + args.margin
    roi = (t >= t0) & (t < t1) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    print("scene %s target %d: window %.0f-%.0f ms, roi x %.0f-%.0f y %.0f-%.0f, %d events (%d targets), "
          "%d corrected by SDNet" % (name, k, t0, t1, x0, x1, y0, y1, roi.sum(), (roi & lab).sum(), (roi & fixed).sum()))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 1.15], wspace=0.28))
    for ax, pred, title in ((axes[0], prior, "(a) Streaming SNN"), (axes[1], final, "(b) SDNet")):
        cat = V.outcome(pred[roi], lab[roi])
        tr, xr = t[roi] / 1000.0, x[roi]
        for c in ("background", "correct", "missed", "false"):
            m = cat == c
            ax.scatter(tr[m], xr[m], s=V.SIZES[c] * 2.5, c=V.COLOURS[c], edgecolor="none",
                       alpha=0.5 if c == "background" else 1.0, rasterized=True)
        if pred is final:                                  # what SDNet changed: corrected (ring), made wrong (cross)
            good, bad = fixed[roi], ((prior == lab) & (final != lab))[roi]
            ax.scatter(tr[good], xr[good], s=18, facecolor="none", edgecolor="#111111", lw=0.5)
            ax.scatter(tr[bad], xr[bad], s=14, marker="x", color="#111111", lw=0.6)
        if t1 - t0 <= 1500:
            for s0 in np.arange(np.ceil(t0 / wms) * wms, t1, wms):
                ax.axvline(s0 / 1000.0, color="#E6E6E6", lw=0.5, zorder=0)
        ax.set_xlim(t0 / 1000.0, t1 / 1000.0)
        ax.set_ylim(x0, x1)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("x (px)")
        ax.set_title(title, fontsize=10)

    ax = axes[2]
    cols = np.arange(len(bounds))
    pos = cols - 1.0                                       # column 0 = prior (shown at -1), 1 + k = step k
    b = np.where(np.isfinite(bounds), bounds, np.nan)
    ax.fill_between(pos, -b, b, color="#EDEDED", step=None, label="undecided band")
    ax.plot(pos, b, color="#9A9A9A", lw=0.8)
    ax.plot(pos, -b, color="#9A9A9A", lw=0.8)
    ax.axhline(0, color="#555555", lw=0.6, ls="--")
    sel = np.flatnonzero(roi[uidx])
    rng = np.random.RandomState(0)
    if len(sel) > args.max_lines:
        tgt = sel[lab[uidx[sel]]]
        bg = sel[~lab[uidx[sel]]]
        nt = min(len(tgt), args.max_lines // 2)
        sel = np.concatenate([rng.choice(tgt, nt, replace=False) if len(tgt) > nt else tgt,
                              rng.choice(bg, min(len(bg), args.max_lines - nt), replace=False)])
    for j in sel:
        i = uidx[j]
        c = commit[i]                                      # a neuron fires once: draw it until it decides
        z = uZ[j, :c + 1] - tau
        colour = V.COLOURS["correct"] if lab[i] else "#7F7F7F"
        ax.plot(pos[:c + 1], z, color=colour, lw=0.7, alpha=0.8)
        ax.scatter([pos[c]], [z[c]], s=12, color=colour, edgecolor="#111111", lw=0.4, zorder=3)
    ax.set_xticks(pos)
    ax.set_xticklabels(["prior"] + [str(int(v)) for v in pos[1:]], fontsize=7)
    ax.set_xlabel("slices waited (%d ms each)" % wms)
    ax.set_ylabel(r"$U-\tau$")
    ax.set_title("(c) Decision variable", fontsize=10)
    lim = np.nanmax(np.abs(b[1:])) * 1.6 if np.isfinite(b[1:]).any() else 8.0
    ax.set_ylim(-lim, lim)
    ax.legend(handles=[Line2D([], [], color=V.COLOURS["correct"], label="target event"),
                       Line2D([], [], color="#7F7F7F", label="background event"),
                       Line2D([], [], marker="o", ls="none", markerfacecolor="w", markeredgecolor="#111111",
                              label="decided")], fontsize=7, loc="upper right", frameon=False)
    handles = [Line2D([], [], marker="o", ls="none", color=V.COLOURS[c], label=l) for c, l in
               (("correct", "Correct"), ("false", "False alarm"), ("missed", "Missed"))]
    handles.append(Line2D([], [], marker="o", ls="none", markerfacecolor="none", markeredgecolor="#111111",
                          label="corrected by SDM"))
    handles.append(Line2D([], [], marker="x", ls="none", color="#111111", label="made wrong by SDM"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.36, -0.04))
    fig.subplots_adjust(bottom=0.24, left=0.06, right=0.99, top=0.9)
    root, _ = os.path.splitext(args.out)
    fig.savefig(root + ".pdf", dpi=600)
    fig.savefig(root + ".png", dpi=200)
    print("FIG MECHANISM FINISHED %s(.pdf/.png)" % root)


if __name__ == "__main__":
    main()
