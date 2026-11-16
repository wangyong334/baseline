"""Paper figure 4.2 (11-16): temporal continuity per target, with the two measures of PACT (ECCV'26, Sec. 4.5).
For every test target its 50 ms frames with events form a hit sequence (a frame is hit when at least one of the
target's events in it is labelled target, the Pd rule of the official evaluator); both measures are normalised by the
number of frames: mean track length = mean length of the runs of hits (higher is better), max interruption = longest
run of misses (lower is better). Targets with fewer than 3 frames are left out.
Methods: run/vis/{EV-Seg,PACT} (dump_competitor.py, prob >= 0.9), the streaming backbone (prior) and SDNet (final) of
run/vis/sdnet_v5s37_coopw_s0 (export_preds.py, >= tau).
Usage (EV-UAV code dir, local): python tools/vis/fig_continuity.py [--out Res/fig_paper/fig_continuity]
Last line: "FIG CONTINUITY FINISHED ...".
"""
import argparse
import json
import os

import numpy as np

PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "red_2": "#E9A6A1", "red_strong": "#B64342"}
SLICE_MS = 50.0


def runs(b):
    """Lengths of the runs of True in a boolean sequence (0 if there is none)."""
    if not b.any():
        return np.array([0])
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def continuity(data_root, methods, names, tau, min_frames=3):
    out = {m: {"track": [], "gap": []} for m in methods}
    for name in names:
        with np.load(os.path.join(data_root, name)) as d:
            loc, norm = d["ev_loc"], d["evs_norm"]
        lab, tid = norm[:, 4] > 0.5, norm[:, 5].astype(np.int64)
        frame = (loc[:, 2] // SLICE_MS).astype(np.int64)
        pred = {}
        for m, (kind, folder) in methods.items():
            e = np.load(os.path.join(folder, name))
            pred[m] = e["prob"] >= 0.9 if kind == "prob" else e[kind + "_logit"] >= tau
        for t in np.unique(tid[lab]):
            sel = lab & (tid == t)
            frames, idx = np.unique(frame[sel], return_inverse=True)
            if len(frames) < min_frames:
                continue
            for m in methods:
                hit = np.bincount(idx, weights=pred[m][sel].astype(float), minlength=len(frames)) > 0
                out[m]["track"].append(runs(hit).mean() / len(frames))
                out[m]["gap"].append(runs(~hit).max() / len(frames))
    return {m: {k: np.asarray(v) for k, v in r.items()} for m, r in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="D:/桌面-D/EV-UAV/dataset/EV-UAV-dataset/test")
    ap.add_argument("--vis", default="Res/vis_data/run/vis")
    ap.add_argument("--out", default="Res/fig_paper/fig_continuity")
    args = ap.parse_args()
    sd = os.path.join(args.vis, "sdnet_v5s37_coopw_s0")
    with open(os.path.join(sd, "meta.json")) as fh:
        meta = json.load(fh)
    methods = {"EV-Sp-\nSegNet": ("prob", os.path.join(args.vis, "EV-Seg")), "PACT\n": ("prob", os.path.join(args.vis, "PACT")),
               "Streaming\nSNN": ("prior", sd), "SDNet\n(ours)": ("final", sd)}
    colors = [PALETTE["red_2"], PALETTE["red_strong"], PALETTE["blue_secondary"], PALETTE["blue_main"]]
    res = continuity(args.data_root, methods, meta["recordings"], meta["tau"])
    for m, r in res.items():
        print("%-22s targets %d | mean track length median %.3f mean %.3f | max interruption mean %.3f std %.3f" % (
            m.replace("\n", " "), len(r["track"]), np.median(r["track"]), r["track"].mean(), r["gap"].mean(),
            r["gap"].std()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 7,
                         "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.55))
    for ax, key, title in zip(axes, ("track", "gap"), ("Mean track length $\\uparrow$", "Max interruption $\\downarrow$")):
        vals = [res[m][key] for m in methods]
        bp = ax.boxplot(vals, widths=0.6, patch_artist=True, showfliers=False, showmeans=True,
                        medianprops={"color": "black", "linewidth": 0.8},
                        meanprops={"marker": "o", "markersize": 2.5, "markerfacecolor": "white",
                                   "markeredgecolor": "black", "markeredgewidth": 0.5},
                        whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7}, boxprops={"linewidth": 0.7})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(list(methods), fontsize=5.6)
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(length=2, width=0.6, pad=1.5)
    axes[0].set_ylabel("fraction of target duration", fontsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (args.out, ext), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("FIG CONTINUITY FINISHED %s(.pdf/.png): %d methods" % (args.out, len(methods)))


if __name__ == "__main__":
    main()
