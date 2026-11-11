"""Render editable scientific architecture figures; no trained weights required.

Run with a Python environment containing matplotlib (not necessarily torch):
    python tools/bst/draw_architecture.py --out docs/figures/bst
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {"bir": ("#E7F1FB", "#27659A"),
          "bank": ("#E7F4ED", "#277653"),
          "sdu": ("#FFF0DC", "#A96012"),
          "vow": ("#F0EAF8", "#72509B"),
          "neutral": ("#F4F6F8", "#526272"),
          "output": ("#E8F5F4", "#247D78")}
INK = "#213547"


def setup(title, subtitle, width=16, height=8.8):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.3, 8.68, title, fontsize=17, weight="bold", color=INK, va="top")
    ax.text(0.3, 8.15, subtitle, fontsize=10.5, color="#526272", va="top")
    return fig, ax


def box(ax, x, y, w, h, title, body="", kind="neutral", fontsize=10.4):
    fill, edge = COLORS[kind]
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.09",
                          linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(patch)
    if body:
        ax.text(x + w / 2, y + h - 0.20, title, color=INK, fontsize=fontsize,
                weight="bold", ha="center", va="top", zorder=4)
        ax.text(x + w / 2, y + h / 2 - 0.17, body, color=INK,
                fontsize=fontsize - 0.6, ha="center", va="center", linespacing=1.45, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, color=INK, fontsize=fontsize,
                weight="bold", ha="center", va="center", zorder=4)
    return (x, y, w, h)


def arrow(ax, points, label=None, label_xy=None, color=INK, dashed=False, fontsize=9.6):
    # Orthogonal routing avoids edges crossing text or block interiors.
    for i in range(len(points) - 2):
        ax.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]],
                color=color, lw=1.4, linestyle="--" if dashed else "-", zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=13, lw=1.4, color=color,
                                linestyle="--" if dashed else "-", zorder=2))
    if label:
        if label_xy is None:
            label_xy = ((points[0][0] + points[-1][0]) / 2,
                        (points[0][1] + points[-1][1]) / 2 + 0.12)
        ax.text(*label_xy, label, ha="center", va="bottom", fontsize=fontsize,
                color=color, bbox=dict(facecolor="white", edgecolor="none", pad=1.1), zorder=5)


def note(ax, text, y=0.26):
    ax.text(0.3, y, text, fontsize=9.4, color="#526272", va="bottom")


def overall():
    fig, ax = setup("BST-SNN: shared evidence, event-specific sequential decisions",
                    "One streaming backbone; a shared history; independent beliefs and commit times.")
    box(ax, 0.3, 6.25, 1.7, 1.25, "Event stream", "x, y, t, polarity")
    box(ax, 2.7, 6.05, 3.1, 1.65, "BIR U-Net", "Window-wise shared features\nBounded memory / quantized reports", "bir")
    box(ax, 6.5, 6.25, 2.15, 1.25, "Initial belief", "$z_i = f_i - \\log w$\nBefore SDU", "bir")
    box(ax, 9.55, 6.25, 2.55, 1.25, "Cheap birth decision", "Fixed band (default)\nOptional shared context + MLP", "vow", 9.6)
    box(ax, 13.1, 4.5, 2.5, 1.35, "Committed output", "Event label + commit time\nReport score / posterior estimate", "output", 9.8)
    box(ax, 0.3, 2.0, 2.35, 1.55, "Pending event records", "Own prior / time / polarity\nOwn lightweight state", "neutral")
    box(ax, 3.5, 2.0, 3.1, 1.55, "Shared evidence bank", "Encode each source once\nKeep original source indices", "bank")
    box(ax, 7.5, 2.0, 3.0, 1.55, "SDU: re-audit history", "Age 0: available history\nAge 1..D: newly arrived windows\nEvent-specific posterior refinement", "sdu", 9.8)
    box(ax, 11.4, 2.0, 3.0, 1.55, "VoW: wait or commit", "Expected continuation cost\nversus stopping risk", "vow")
    arrow(ax, [(2., 6.88), (2.7, 6.88)])
    arrow(ax, [(5.8, 6.88), (6.5, 6.88)])
    arrow(ax, [(8.65, 6.88), (9.55, 6.88)])
    arrow(ax, [(12.1, 6.88), (14.35, 6.88), (14.35, 5.85)], "commit now", (13.1, 7.02))
    arrow(ax, [(10.4, 6.25), (10.4, 4.95), (1.47, 4.95), (1.47, 3.55)],
          "open SDU", (8.65, 5.08), color=COLORS["vow"][1])
    arrow(ax, [(7.4, 6.25), (7.4, 4.15), (5.05, 4.15), (5.05, 3.55)],
          "all observed source scores + polarity", (4.85, 4.27), color=COLORS["bank"][1])
    arrow(ax, [(6.6, 2.8), (7.5, 2.8)], "read", (7.03, 2.93))
    arrow(ax, [(2.65, 2.6), (3.0, 2.6), (3.0, 1.37), (8.4, 1.37), (8.4, 2.0)],
          "query metadata + state", (5.65, 1.48))
    arrow(ax, [(10.5, 2.8), (11.4, 2.8)])
    arrow(ax, [(13.2, 3.55), (13.2, 4.15), (14.35, 4.15), (14.35, 4.5)],
          "commit", (14.95, 3.89), color=COLORS["output"][1])
    arrow(ax, [(12.8, 2.0), (12.8, 0.78), (1.47, 0.78), (1.47, 2.0)],
          "wait: next available window, active queries only", (8.5, 0.92), color=COLORS["vow"][1])
    note(ax, "Shared computation does not imply shared event labels. Committed observations remain usable by other pending events.")
    return fig


def backbone():
    fig, ax = setup("BIR: streaming multi-scale backbone and bounded neuron",
                    "Existing backbone retained. Seven state groups; one update per 50 ms window.")
    box(ax, 0.35, 7.0, 3.2, 0.75, "12-channel event tensor", "", "neutral", 10.2)
    box(ax, 6.4, 7.0, 3.45, 0.75, "Event readout: feature + p + t", "", "bir", 9.7)
    coords = [(0.45, 5.65), (1.85, 4.4), (3.25, 3.15), (4.65, 1.9),
              (6.05, 3.15), (7.45, 4.4), (8.85, 5.65)]
    labels = [("enc1", "12 channels"), ("enc2", "24 channels"), ("enc3", "48 channels"),
              ("enc4", "48 channels"), ("dec3", "48 channels"), ("dec2", "24 channels"),
              ("dec1", "12 channels")]
    for (x, y), (title, body) in zip(coords, labels):
        box(ax, x, y, 1.10, 0.83, title, body, "bir", 9.2)
    for i in range(6):
        x, y = coords[i]
        nx, ny = coords[i + 1]
        arrow(ax, [(x + 1.1, y + 0.415), (nx, ny + 0.415)])
    arrow(ax, [(0.95, 7.0), (0.95, 6.48)])
    arrow(ax, [(9.4, 6.48), (9.4, 7.0)])
    for left, right in ((0, 6), (1, 5), (2, 4)):
        x, y = coords[left]
        rx, ry = coords[right]
        arrow(ax, [(x + 1.1, y + 0.65), (rx, ry + 0.65)], color=COLORS["bir"][1], dashed=True)
    ax.text(4.9, 6.49, "skip connections", color=COLORS["bir"][1], fontsize=9.3, ha="center")
    ax.text(4.95, 1.25, "spatial stride: 1 / 2 / 4 / 8", color=INK, fontsize=10, ha="center")
    ax.plot([10.4, 10.4], [0.85, 7.7], color="#C9D2DC", lw=1)
    box(ax, 11.0, 6.65, 4.35, 0.8, "Driving current", "$I_k - b$", "bir", 10.1)
    box(ax, 11.0, 4.7, 4.35, 1.35, "Bounded memory", "$M_k=\\mathrm{clip}_{[0,L\\theta]}$\n$[\\lambda M_{k-1}+(1-\\lambda)(I_k-b)]$", "bir", 10.1)
    box(ax, 11.0, 3.0, 4.35, 1.0, "Quantized report", "$q_k=\\mathrm{round}(M_k/\\theta)$", "bir", 10.1)
    box(ax, 11.0, 1.3, 4.35, 1.0, "Change in report level", "$\\Delta q_k=q_k-q_{k-1}$", "bir", 10.1)
    arrow(ax, [(13.18, 6.65), (13.18, 6.05)])
    arrow(ax, [(13.18, 4.7), (13.18, 4.0)])
    arrow(ax, [(13.18, 3.0), (13.18, 2.3)])
    arrow(ax, [(15.35, 5.37), (15.75, 5.37), (15.75, 6.35), (14.7, 6.35), (14.7, 6.05)],
          "carry", (15.1, 6.37), color=COLORS["bir"][1])
    note(ax, "The tensor has 2 count channels + 5 temporal bins x 2 polarities. Report-change savings and dense GPU runtime are distinct metrics.")
    return fig


def source_bank():
    fig, ax = setup("Shared evidence bank: source features once, relationships on read",
                    "Cross-query reuse without replacing a neighborhood by one irreversible summary.")
    box(ax, 0.45, 5.65, 2.7, 1.4, "Observed source event", "$[z_j, p_j]$\nOptional extra source features", "bank")
    box(ax, 4.05, 5.65, 3.35, 1.4, "Shared source encoder", "Linear - LayerNorm - SiLU\nLinear - SiLU", "bank")
    box(ax, 8.3, 5.45, 3.25, 1.8, "Projected source features", "key: 2 x 8\nvalue: 2 x 8\ncontext: 16", "bank")
    box(ax, 12.45, 5.65, 3.0, 1.4, "Finite history cache", "Source ID -> encoded features\nReuse across queries and steps", "bank", 10.0)
    arrow(ax, [(3.15, 6.35), (4.05, 6.35)])
    arrow(ax, [(7.4, 6.35), (8.3, 6.35)])
    arrow(ax, [(11.55, 6.35), (12.45, 6.35)])
    box(ax, 0.45, 2.1, 3.3, 1.55, "Preserved metadata", "Source ID / x / y / t / window\nOriginal event identity retained", "neutral")
    box(ax, 4.65, 2.1, 3.3, 1.55, "Causal neighbor index", "Only arrived windows\nSelf-exclusion / padded masks", "neutral")
    box(ax, 9.0, 2.1, 3.35, 1.55, "Query-specific relationship", "$\\Delta x,\\Delta y,\\Delta t,\\Delta w$\nLightweight relation encoding", "sdu")
    box(ax, 13.15, 2.1, 2.3, 1.55, "SDU read", "Separate event beliefs", "sdu")
    arrow(ax, [(1.8, 5.65), (1.8, 3.65)], "keep metadata", (2.5, 4.5))
    arrow(ax, [(3.75, 2.88), (4.65, 2.88)])
    arrow(ax, [(7.95, 2.88), (9.0, 2.88)])
    arrow(ax, [(12.35, 2.88), (13.15, 2.88)])
    arrow(ax, [(14.3, 5.65), (14.3, 3.65)], "gather by source ID", (13.65, 4.55), color=COLORS["bank"][1])
    ax.text(7.9, 1.15, "Source encoding is independent of the querying event. Relative geometry is introduced only during reading.",
            fontsize=11, color=INK, ha="center")
    note(ax, "Defaults shown: source width 32, 2 heads, relation width 8, context width 16. Training parameter updates invalidate encoded caches.")
    return fig


def sdu():
    fig, ax = setup("SDU: revisable evidence with an event-specific lightweight state",
                    "Both old evidence weights and old signed values can change when new evidence becomes available.")
    box(ax, 0.35, 5.8, 3.0, 1.55, "Current query context", "Prior-anchored state\n+ visible-source summary + age\nFeature-wise LayerNorm", "sdu", 10.1)
    box(ax, 0.35, 2.3, 3.0, 1.55, "Indexed history", "Shared keys / values / context\nRelative geometry\nCausal mask", "bank", 10.1)
    box(ax, 4.25, 5.8, 3.0, 1.35, "Reweight old evidence", "Query-key interaction\nSigmoid gates $a_{ijh}$", "sdu")
    box(ax, 4.25, 2.5, 3.0, 1.35, "Revalue old evidence", "Source + relation + query\nSiLU -> signed value $v_{ijh}$", "sdu")
    box(ax, 8.15, 4.25, 3.15, 1.65, "Aggregate retained evidence", "$r_{ih}=\\frac{\\sum_j a_{ijh}v_{ijh}}{\\sum_j a_{ijh}}$\nOne mean per head", "sdu", 10.2)
    box(ax, 12.25, 4.25, 3.2, 1.65, "Replace total correction", "$\\ell_i^{(d)}=z_i+\\sum_h r_{ih}$\nNever add the whole history twice", "output", 10.2)
    box(ax, 8.15, 1.15, 3.15, 1.5, "Next query state", "Prior context + tanh residual\nWeighted source summary\nHead corrections + log mass", "sdu", 9.8)
    arrow(ax, [(3.35, 6.55), (4.25, 6.55)])
    arrow(ax, [(3.35, 6.1), (3.8, 6.1), (3.8, 4.5), (5.75, 4.5), (5.75, 3.85)])
    arrow(ax, [(3.35, 3.1), (4.25, 3.1)])
    arrow(ax, [(1.85, 3.85), (1.85, 4.85), (5.0, 4.85), (5.0, 5.8)])
    arrow(ax, [(7.25, 6.45), (7.7, 6.45), (7.7, 5.35), (8.15, 5.35)])
    arrow(ax, [(7.25, 3.2), (7.7, 3.2), (7.7, 4.8), (8.15, 4.8)])
    arrow(ax, [(11.3, 5.08), (12.25, 5.08)])
    arrow(ax, [(9.73, 4.25), (9.73, 2.65)])
    arrow(ax, [(8.15, 1.9), (0.1, 1.9), (0.1, 6.55), (0.35, 6.55)],
          "carry to next new evidence", (4.0, 1.98), color=COLORS["sdu"][1], dashed=True)
    box(ax, 12.25, 1.15, 3.2, 1.5, "No new evidence", "Keep belief and context\nAdvance policy age only", "neutral", 10.0)
    note(ax, "Weights are aggregation weights, not calibrated association probabilities. Signed values are learned features, not proven individual log likelihood ratios.")
    return fig


def vow():
    fig, ax = setup("VoW: stop risk versus conditional continuation cost",
                    "Two information taps: a cheap pre-SDU birth decision and a post-SDU stopping decision.")
    box(ax, 0.4, 5.8, 3.2, 1.35, "Before SDU: cheap features", "Own features (3) + shared context (7)\nNo SDU neighbor encoding", "bir", 10.0)
    box(ax, 0.4, 2.5, 3.2, 1.5, "After SDU: causal state", "Belief / correction / change\nContext / evidence statistics\nAge and remaining horizon", "sdu", 10.0)
    box(ax, 4.55, 5.8, 3.0, 1.35, "Optional birth value head", "Expected policy cost after opening\n+ initialization cost", "vow", 9.8)
    box(ax, 4.55, 2.5, 3.0, 1.5, "Continuation value head", "Expected later policy cost\n+ current waiting / update costs", "vow", 9.8)
    box(ax, 8.55, 4.7, 3.2, 1.65, "Model-based stop risk", "$R_{stop}=\\min[c_{FP}(1-p),c_{FN}p]$\nCosts match the report threshold", "vow", 9.8)
    box(ax, 12.7, 4.7, 2.85, 1.65, "Decision", "Continue if expected cost is lower\nOtherwise commit\nDeadline always commits", "vow", 9.4)
    box(ax, 8.55, 1.55, 7.0, 1.45, "Learn a conditional expectation, then compare actions", "Frozen-policy rollout targets from labels\nNo per-event hindsight minimum; no future inputs at inference", "neutral", 10.4)
    arrow(ax, [(3.6, 6.48), (4.55, 6.48)])
    arrow(ax, [(3.6, 3.25), (4.55, 3.25)])
    arrow(ax, [(7.55, 6.75), (12.1, 6.75), (12.1, 5.8), (12.7, 5.8)], "open cost", (9.9, 6.86), color=COLORS["vow"][1])
    arrow(ax, [(7.55, 3.25), (12.1, 3.25), (12.1, 5.1), (12.7, 5.1)], "wait cost", (10.0, 3.37), color=COLORS["vow"][1])
    arrow(ax, [(11.75, 5.53), (12.7, 5.53)])
    ax.text(0.4, 1.65, "$p=\\sigma(\\ell)$: posterior estimate\n$\\mathrm{report}=\\sigma(\\ell+\\log w)$\n$\\tau=\\mathrm{logit}(0.9)-\\log w$",
            fontsize=11.5, color=INK, linespacing=1.8)
    note(ax, "A fixed birth band is the default; learned admission is optional. Current-step costs are external; future costs are already in the learned return.")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--out", default=str(root / "docs" / "figures" / "bst"))
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--pdf", action="store_true", help="Also export vector PDF")
    args = parser.parse_args()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, draw in (("01_overall", overall), ("02_bir", backbone),
                       ("03_shared_evidence", source_bank), ("04_sdu", sdu), ("05_vow", vow)):
        fig = draw()
        extensions = ["svg", "png"] + (["pdf"] if args.pdf else [])
        for extension in extensions:
            path = output / (name + "." + extension)
            fig.savefig(str(path), dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
            print(path)
        plt.close(fig)


if __name__ == "__main__":
    main()
