"""Paper figure 4.3.2 (11-16): what the decision units see while they wait. For the units of the paper model (test,
backbone seed 37, two populations, decision seed 0; run/vis/sdnet_v5s37_coopw_s0 from export_preds.py) the decision
variable per column c of the trajectory, target and background units apart (median, 25-75 % and 10-90 % bands), against
the calibrated stopping bound +-b_c (a unit commits at the first column with |Z_c - tau| >= b_c). Column 0 is the
backbone prior at arrival; column c >= 1 is the unit's membrane after step c - 1, i.e. after its own slice and c - 1
further slices (column 11 = deadline, 10 further slices = 500 ms). Below, the share of units committing per column.
Also prints, per column, the AUC between target and background units, the accuracy of sign(Z_c - tau) and the share
committed so far (numbers for the text).
Usage (EV-UAV code dir, local): python tools/vis/fig_evidence_steps.py [--out Res/fig_paper/fig_evidence_steps]
Last line: "FIG EVIDENCE STEPS FINISHED ...".
"""
import argparse
import json
import os

import numpy as np

PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "grey": "#6E6E6E"}


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1.0
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load_units(data_root, export_dir):
    with open(os.path.join(export_dir, "meta.json")) as fh:
        meta = json.load(fh)
    Z, Y, C, L = [], [], [], []
    for name in meta["recordings"]:
        with np.load(os.path.join(data_root, name)) as d:
            lab = d["evs_norm"][:, 4] > 0.5
        e = np.load(os.path.join(export_dir, name))
        u = e["unit_idx"]
        Z.append(e["unit_Z"].astype(np.float32))
        Y.append(lab[u])
        C.append(e["commit"][u])
        L.append(e["last"][u])
    return meta, np.concatenate(Z), np.concatenate(Y), np.concatenate(C), np.concatenate(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="D:/桌面-D/EV-UAV/dataset/EV-UAV-dataset/test")
    ap.add_argument("--export", default="Res/vis_data/run/vis/sdnet_v5s37_coopw_s0")
    ap.add_argument("--out", default="Res/fig_paper/fig_evidence_steps")
    args = ap.parse_args()
    meta, Z, Y, C, L = load_units(args.data_root, args.export)
    tau, bounds = meta["tau"], np.asarray(meta["bounds"])
    S = Z.shape[1]
    print("units %d (target %d, background %d), %d steps" % (len(Y), Y.sum(), (~Y).sum(), S))
    q = {"t": [], "b": []}
    for k in range(S):
        ok = L >= k
        z, y = Z[ok, k] - tau, Y[ok]
        for key, sel in (("t", y), ("b", ~y)):
            q[key].append(np.percentile(z[sel], [10, 25, 50, 75, 90]))
        print("column %2d (%s) | AUC %.3f | accuracy %.3f | median target %+.2f background %+.2f | committed %.1f%% | "
              "bound %.2f" % (k, "prior" if k == 0 else "waited %d" % (k - 1), auc(z[y], z[~y]), np.mean((z >= 0) == y),
                              np.median(z[y]), np.median(z[~y]),
                 100 * np.mean(C <= k), bounds[k]))
    q = {k: np.asarray(v) for k, v in q.items()}
    share = np.array([np.mean(C == k) for k in range(S)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 7,
                         "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "legend.frameon": False})
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.35, 2.45), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]})
    k = np.arange(S)
    for key, c, lab in (("t", PALETTE["blue_main"], "target events"), ("b", PALETTE["grey"], "background events")):
        ax.fill_between(k, q[key][:, 0], q[key][:, 4], color=c, alpha=0.12, lw=0)
        ax.fill_between(k, q[key][:, 1], q[key][:, 3], color=c, alpha=0.28, lw=0)
        ax.plot(k, q[key][:, 2], color=c, lw=1.4, marker="o", ms=2.2, label=lab)
    ax.plot(k, bounds, color="black", lw=0.9, ls="--", label="decision boundary $\\pm b^k$")
    ax.plot(k, -bounds, color="black", lw=0.9, ls="--")
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_ylabel("$u_i^k-\\tau$ (log-odds)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.8, handlelength=1.6, ncol=3,
              columnspacing=0.9, borderaxespad=0.1)
    ax.tick_params(length=2, width=0.6)
    bx.bar(k, 100 * share, color=PALETTE["blue_secondary"], width=0.7)
    for i in np.flatnonzero(share >= 0.05):
        bx.text(i, 100 * share[i] + 1.5, "%.0f%%" % (100 * share[i]), ha="center", va="bottom", fontsize=5.5)
    bx.set_ylabel("commit (%)")
    bx.set_ylim(0, 100 * share.max() * 1.3)
    bx.set_xlabel("slices waited after the slice of the event")
    bx.set_xticks(k)
    bx.set_xticklabels(["prior"] + [str(i) for i in range(S - 1)])
    bx.tick_params(length=2, width=0.6)
    fig.tight_layout(pad=0.3, h_pad=0.4)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (args.out, ext), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("FIG EVIDENCE STEPS FINISHED %s(.pdf/.png): %d units, %d steps" % (args.out, len(Y), S))


if __name__ == "__main__":
    main()
