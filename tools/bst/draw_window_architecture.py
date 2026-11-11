"""Editable figures for source-window aggregation and sequential decisions.

Generate SVG/PNG only; no claims about trained performance or risk guarantees.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bst.draw_architecture import setup, box, arrow, note, COLORS, INK, backbone, source_bank
import matplotlib.pyplot as plt


def overall():
    fig, ax = setup("BST-SNN: revisable evidence, organized by observation window",
                    "Shared SNN computation; window-grouped ANN decision head; independent event labels and stopping times.")
    box(ax, .35, 6.15, 1.7, 1.2, "Events", "x, y, t, polarity")
    box(ax, 2.65, 5.95, 2.85, 1.6, "BIR backbone", "50 ms shared update\nBounded state / event readout", "bir", 10.0)
    box(ax, 6.15, 6.15, 2.3, 1.2, "Initial belief", "$p_0=\\sigma(z_0)$", "bir")
    box(ax, 9.1, 5.95, 2.8, 1.6, "Birth decision", "Fixed band by default\nOptional cheap value MLP", "vow", 10.0)
    box(ax, 13.05, 5.95, 2.6, 1.6, "Commit", "Event label / score\nCommit time", "output", 10.0)
    for left, right in [(2.05, 2.65), (5.5, 6.15), (8.45, 9.1), (11.9, 13.05)]:
        arrow(ax, [(left, 6.75), (right, 6.75)])
    box(ax, .35, 2.0, 3.1, 1.85, "Shared source bank", "Encode each source once\nRetain source ID and window\nCache fixed relations only", "bank", 10.3)
    box(ax, 4.25, 2.0, 6.7, 1.85, "SDU: re-audit old observations", "Within-window normalized evidence\nAcross-window discounted accumulation\nPrior + recomputed total correction", "sdu", 11.0)
    box(ax, 11.75, 2.0, 3.9, 1.85, "VoW: wait or commit", "Stop risk versus\nconditional continuation cost\nDeadline commits", "vow", 10.5)
    arrow(ax, [(7.3, 6.15), (7.3, 4.65), (1.9, 4.65), (1.9, 3.85)],
          "arrived scores + polarity + source IDs", (4.1, 4.82), color=COLORS["bank"][1])
    arrow(ax, [(10.5, 5.95), (10.5, 4.35), (7.65, 4.35), (7.65, 3.85)],
          "admitted query + own state", (9.1, 4.5), color=COLORS["vow"][1])
    arrow(ax, [(3.45, 2.95), (4.25, 2.95)])
    arrow(ax, [(10.95, 2.95), (11.75, 2.95)])
    arrow(ax, [(14.3, 3.85), (14.3, 5.95)], "commit", (14.75, 4.9), color=COLORS["output"][1])
    arrow(ax, [(13.7, 2.), (13.7, .95), (7.6, .95), (7.6, 2.)],
          "wait for next window: retained queries only", (10.25, 1.1), color=COLORS["vow"][1])
    note(ax, "Old observations remain available after commitment; time grouping does not assert independent windows.\nAge-zero control: run SDU once at admission, then commit before later-window evidence.")
    return fig


def sources():
    fig = source_bank()
    ax = fig.axes[0]
    ax.texts[0].set_text("Shared evidence: cache source features, preserve window identity")
    ax.texts[1].set_text("Query-independent source projections are cached; context-dependent weights and signed values are recomputed.")
    return fig


def sdu():
    fig, ax = setup("SDU: normalize repeated observations, accumulate revisable time groups",
                    "A source window is the grouping unit. All retained old tokens remain individually addressable.")
    box(ax, .35, 6.15, 3.2, 1.45, "Current query", "Prior-anchored state + age\nWindow-balanced visible context\nChannel-wise LayerNorm", "sdu", 10.)
    box(ax, .35, 3.4, 3.2, 1.5, "Indexed history", "Source key / value / context\nRelative geometry + window ID\nCausal masks", "bank", 10.)
    box(ax, 4.3, 6.15, 3.1, 1.45, "Reweight every old token", "Query-key interaction\n$a_{ijh}=\\max[\\sigma(u_{ijh}),\\epsilon]$\nValid tokens only; invalid = 0", "sdu", 9.8)
    box(ax, 4.3, 3.4, 3.1, 1.5, "Revalue every old token", "Source + relation + query\n$v_{ijh}=s_h\\tanh(r_{ijh})$\n$s_h>0$: learned head scale", "sdu", 10.)
    box(ax, 8.15, 4.65, 3.25, 1.7, "Within each source window", "$m_{igh}=\\frac{\\sum_{j\\in g} a_{ijh}v_{ijh}}{\\sum_{j\\in g}a_{ijh}}$\nEmpty groups contribute zero", "sdu", 10.5)
    box(ax, 12.15, 4.65, 3.4, 1.7, "Across arrived windows", "$\\rho=\\sigma(\\mathrm{MLP}[q,c_g,\\Delta w])$\n$R_i=\\sum_{g,h}\\rho_{igh}m_{igh}$\nNo normalization across windows", "sdu", 10.1)
    box(ax, 8.15, 1.45, 3.25, 1.6, "Next query state", "Prior anchor + tanh residual\nWindow-balanced context\nWindow-level statistics", "sdu", 10.)
    box(ax, 12.15, 1.45, 3.4, 1.6, "Variational belief readout", "$\\ell_i=z_i+R_i$\n$p_i=\\sigma(\\ell_i)$\nReplace total correction", "output", 10.2)
    arrow(ax, [(3.55, 6.87), (4.3, 6.87)])
    arrow(ax, [(3.55, 6.35), (3.9, 6.35), (3.9, 5.4), (5.85, 5.4), (5.85, 4.9)])
    arrow(ax, [(3.55, 4.15), (4.3, 4.15)])
    arrow(ax, [(1.95, 4.9), (1.95, 5.55), (4.65, 5.55), (4.65, 6.15)])
    arrow(ax, [(7.4, 6.9), (7.75, 6.9), (7.75, 5.75), (8.15, 5.75)])
    arrow(ax, [(7.4, 4.15), (7.75, 4.15), (7.75, 5.15), (8.15, 5.15)])
    arrow(ax, [(11.4, 5.5), (12.15, 5.5)])
    arrow(ax, [(13.85, 4.65), (13.85, 3.05)])
    arrow(ax, [(9.8, 4.65), (9.8, 3.05)])
    arrow(ax, [(13.0, 4.65), (13.0, 3.7), (10.8, 3.7), (10.8, 3.05)],
          "discounted context", (11.65, 3.85), color=COLORS["sdu"][1], fontsize=9.)
    arrow(ax, [(8.15, 2.25), (.1, 2.25), (.1, 6.87), (.35, 6.87)],
          "next changed evidence prefix", (4.2, 2.38), color=COLORS["sdu"][1], dashed=True)
    note(ax, "Discounts are learned aggregation factors, not calibrated reliability probabilities. Repeated unchanged prefixes keep the applied belief fixed.")
    return fig


def vow():
    fig, ax = setup("VoW: a finite-horizon decision with two explicit costs",
                    "The network estimates a conditional future policy cost; the comparator implements the decision rule.")
    box(ax, .35, 4.95, 3.2, 2.0, "Causal SDU state", "Current belief / change\nWindow-level evidence statistics\nAge / remaining horizon\nNo future labels as input", "sdu", 10.5)
    box(ax, 4.35, 6.15, 4.3, 1.45, "Stop branch: analytic risk", "$R_s=\\min[c_{FN}p,c_{FP}(1-p)]$\nCosts match the report threshold", "vow", 10.7)
    box(ax, 4.35, 3.2, 4.3, 1.65, "Wait branch: conditional return", "Linear - SiLU - Linear - Softplus\n$f_\\theta(S_d)\\approx\\mathbb{E}[J_{d+1}^{\\pi}\\mid S_d]$\n$Q_w=c_{step}+f_\\theta(S_d)$", "vow", 10.4)
    box(ax, 9.55, 4.95, 2.4, 1.8, "Compare", "$Q_w<R_s$?\nDeadline overrides", "vow", 10.6)
    box(ax, 12.9, 6.15, 2.65, 1.25, "Commit", "Remove query state", "output", 10.)
    box(ax, 12.9, 3.6, 2.65, 1.25, "Wait", "Read next window", "sdu", 10.)
    arrow(ax, [(3.55, 6.45), (3.95, 6.45), (3.95, 6.88), (4.35, 6.88)])
    arrow(ax, [(3.55, 5.45), (3.95, 5.45), (3.95, 4.02), (4.35, 4.02)])
    arrow(ax, [(8.65, 6.88), (9.05, 6.88), (9.05, 6.05), (9.55, 6.05)])
    arrow(ax, [(8.65, 4.02), (9.05, 4.02), (9.05, 5.4), (9.55, 5.4)])
    arrow(ax, [(11.95, 6.15), (12.4, 6.15), (12.4, 6.78), (12.9, 6.78)], "no", (12.45, 6.9))
    arrow(ax, [(11.95, 5.45), (12.4, 5.45), (12.4, 4.23), (12.9, 4.23)], "yes", (12.45, 4.35))
    box(ax, .35, .95, 7.1, 1.45, "Training: frozen-policy returns", "Roll out a deployable policy, construct realized returns, fit conditional means\nRepeat policy evaluation / improvement; never choose hindsight label minima", "neutral", 10.)
    box(ax, 8.35, .95, 7.2, 1.45, "Admission happens before costly SDU work", "Default: public fixed band; optional cheap birth head\nShared grid statistics + own features predict cost after opening", "neutral", 10.)
    arrow(ax, [(5.6, 2.4), (5.6, 3.2)], "supervise", (6.55, 2.65), dashed=True, color=COLORS["vow"][1])
    note(ax, "Model-based risk is not a guaranteed error bound. Current action cost is explicit; future costs are included in the learned return.", y=.25)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT/"docs/figures/bst_window"))
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    for name, draw in (("01_overall", overall), ("02_bir", backbone), ("03_shared_evidence", sources),
                       ("04_sdu", sdu), ("05_vow", vow)):
        fig = draw()
        for extension in ("svg", "png"):
            path = output/(name+"."+extension)
            fig.savefig(str(path), dpi=args.dpi, bbox_inches="tight", pad_inches=.12)
            print(path)
        plt.close(fig)


if __name__ == "__main__":
    main()
