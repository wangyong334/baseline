"""Editable r3 scientific diagrams, faithful to the implemented modules.

Render with: python tools/bst/draw_final_architecture.py
SVG text stays editable; PNG files are provided for visual review. Older figure
directories are never overwritten. No trained accuracy or risk bound is implied.
"""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bst.draw_architecture import backbone


INK = "#203447"
MUTED = "#586C7D"
PALETTE = {"bir": ("#EAF2FC", "#3577AC"),
           "bank": ("#EAF5EE", "#3A8360"),
           "sdu": ("#FFF1DF", "#BD7A24"),
           "vow": ("#F2ECFA", "#8964AC"),
           "output": ("#E6F5F3", "#2D8580"),
           "neutral": ("#F3F6F9", "#8495A4")}


def canvas(title, subtitle):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(18, 10), facecolor="white")
    fig.subplots_adjust(left=.015, right=.985, top=.985, bottom=.015)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(.45, 9.65, title, fontsize=21, weight="bold", color=INK, va="top")
    ax.text(.45, 9.05, subtitle, fontsize=11.4, color=MUTED, va="top")
    ax._block_text = []
    return fig, ax


def block(ax, x, y, w, h, title, body="", kind="neutral", fontsize=11):
    fill, edge = PALETTE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.025,rounding_size=.11",
                              linewidth=1.5, edgecolor=edge, facecolor=fill, zorder=3))
    if body:
        t = ax.text(x+w/2, y+h-.22, title, color=INK, fontsize=fontsize+.4,
                    weight="bold", ha="center", va="top", zorder=4)
        b = ax.text(x+w/2, y+(h-.55)/2, body, color=INK, fontsize=fontsize-.1,
                    ha="center", va="center", linespacing=1.55, zorder=4)
        ax._block_text.extend([(t, (x, y, w, h)), (b, (x, y, w, h))])
    else:
        t = ax.text(x+w/2, y+h/2, title, color=INK, fontsize=fontsize+.4,
                    weight="bold", ha="center", va="center", zorder=4)
        ax._block_text.append((t, (x, y, w, h)))


def edge(ax, points, label=None, label_xy=None, kind="neutral", dashed=False, fontsize=10):
    color = PALETTE[kind][1] if kind != "neutral" else INK
    style = "--" if dashed else "-"
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.5, linestyle=style, zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=15, lw=1.5, color=color,
                                linestyle=style, zorder=2))
    if label:
        x, y = label_xy
        ax.text(x, y, label, color=color, ha="center", va="center", fontsize=fontsize,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.7), zorder=5)


def op(ax, x, y, symbol, kind="sdu"):
    ax.add_patch(Circle((x, y), .23, facecolor="white", edgecolor=PALETTE[kind][1], lw=1.8, zorder=4))
    ax.text(x, y, symbol, ha="center", va="center", fontsize=20, color=INK, zorder=5)


def footer(ax, text):
    ax.plot([.45, 17.55], [.69, .69], color="#D9E1E8", lw=.9)
    ax.text(.45, .46, text, fontsize=10, color=MUTED, va="top", linespacing=1.45)


def overall():
    fig, ax = canvas("BST-SNN  |  Shared encoding, revisable sequential evidence",
                     "Final r3: unchanged BIR backbone, null-competitive window evidence, birth-referenced refinement and a learned waiting decision.")
    block(ax, .5, 7.15, 2.0, 1.3, "Event stream", "x, y, t, polarity")
    block(ax, 3.15, 6.95, 3.35, 1.7, "BIR backbone", "Shared window update\nBounded state + event readout", "bir")
    block(ax, 7.15, 7.15, 2.15, 1.3, "Initial belief", "$z_0$, own event prior", "bir")
    block(ax, 10.0, 7.15, 3.15, 1.3, "Cheap admission", "Fixed band by default\nOptional learned birth head", "vow", 10.5)
    block(ax, 14.6, 7.15, 2.9, 1.3, "Committed output", "Event label / score / time", "output", 10.5)
    for a, b in ((2.5, 3.15), (6.5, 7.15), (9.3, 10.)):
        edge(ax, [(a, 7.8), (b, 7.8)])
    edge(ax, [(13.15, 7.8), (14.6, 7.8)], "commit", (13.88, 8.04), "output")
    block(ax, .5, 3.05, 3.6, 2.1, "Shared source bank", "Encode each arrived source once\nRetain IDs + window identity\nCache fixed source/relative features", "bank", 10.8)
    ax.add_patch(FancyBboxPatch((4.75, 2.8), 8.2, 2.75, boxstyle="round,pad=.03,rounding_size=.14",
                              facecolor="#FFFCF6", edgecolor=PALETTE["sdu"][1], lw=1.5, zorder=1))
    ax.text(5.0, 5.26, "SDU  /  one belief per admitted event", color=INK, fontsize=12.5, weight="bold")
    block(ax, 5.0, 3.2, 2.25, 1.55, "Re-audit history", "Reweight + revalue\nall retained old tokens", "sdu", 10.5)
    block(ax, 7.75, 3.2, 2.35, 1.55, "Window evidence", "Zero-valued null option\nDiscount across windows", "sdu", 10.2)
    block(ax, 10.6, 3.2, 2.1, 1.55, "Innovation", "$\\ell_d=z_0+F_d-F_0$\nFrozen birth reference", "sdu", 10.2)
    edge(ax, [(7.25, 3.98), (7.75, 3.98)])
    edge(ax, [(10.1, 3.98), (10.6, 3.98)])
    block(ax, 14.05, 3.05, 3.45, 2.1, "VoW: wait or commit", "Stop risk vs. conditional\ncontinuation policy cost\nDeadline commits", "vow", 11)
    edge(ax, [(4.1, 4.1), (4.75, 4.1)])
    edge(ax, [(12.95, 4.1), (14.05, 4.1)])
    edge(ax, [(8.22, 7.15), (8.22, 6.2), (2.3, 6.2), (2.3, 5.15)],
         "all observed source scores + metadata", (4.9, 6.38), "bank")
    edge(ax, [(11.58, 7.15), (11.58, 6.03), (8.85, 6.03), (8.85, 5.55)],
         "admit query + initialize reference", (11.45, 5.81), "vow", fontsize=9.5)
    edge(ax, [(15.78, 5.15), (15.78, 7.15)], "commit", (16.35, 6.1), "output")
    edge(ax, [(15.78, 3.05), (15.78, 2.03), (8.85, 2.03), (8.85, 2.8)],
         "wait: next arrived window, pending queries only", (12.1, 2.22), "vow")
    for x, w, label in ((.5, 5.1, "Shared source computation; separate event beliefs"),
                        (6.05, 5.55, "Old sources remain available for reinterpretation"),
                        (12.05, 5.45, "Unchanged evidence prefix: keep belief + context")):
        block(ax, x, .98, w, .58, label, fontsize=9.4)
    footer(ax, "Learning yields revisable evidence potentials, not proven token likelihood ratios. The hybrid SNN + ANN decision head uses only arrived evidence at inference.")
    return fig


def bir():
    fig = backbone()
    ax = fig.axes[0]
    ax.texts[0].set_text("BIR: existing streaming backbone, retained without architectural change")
    ax.texts[0].set_fontsize(15.5)
    ax.texts[1].set_text("Multi-scale shared computation and bounded temporal memory; no SDU or VoW information flows into the frozen prior backbone.")
    return fig


def sources():
    fig, ax = canvas("Shared evidence bank  |  Reuse computation, retain observations",
                     "Inference caches query-independent projections. Each changed prefix recomputes the query-dependent evidence weights and signed values.")
    ax.text(.5, 8.2, "SHARED  /  once per arrived source at fixed model parameters", color=PALETTE["bank"][1], fontsize=11, weight="bold")
    block(ax, .5, 6.2, 2.75, 1.4, "Source event", "$[z_j,p_j]$\nMetadata retained separately", "bank", 10.6)
    block(ax, 4.05, 6.05, 3.5, 1.7, "Shared source encoder", "Linear - LayerNorm - SiLU\nLinear - SiLU", "bank")
    block(ax, 8.4, 6.05, 3.65, 1.7, "Cached projections", "Keys $k_j$ / value features $v_j$\nContext features $u_j$", "bank")
    block(ax, 12.9, 6.05, 4.55, 1.7, "Finite source cache", "Original ID + source window\nReuse across queries and ages\nCommitment does not erase a source", "bank", 10.6)
    for a, b in ((3.25, 4.05), (7.55, 8.4), (12.05, 12.9)):
        edge(ax, [(a, 6.9), (b, 6.9)], kind="bank")
    ax.text(.5, 4.9, "QUERY-SPECIFIC  /  retain every selected token; gather by original ID", color=PALETTE["sdu"][1], fontsize=11, weight="bold")
    block(ax, .5, 2.6, 3.4, 1.7, "Causal spatial index", "Pixel / window lookup\nSelf-exclusion + valid mask\nOnly available windows", "neutral", 10.6)
    block(ax, 4.65, 2.6, 3.55, 1.7, "Fixed pair features", "$\\Delta x,\\Delta y,\\Delta t,\\Delta w$\nRelation MLP + source gather\nCache within a fixed-weight rollout", "sdu", 10.3)
    block(ax, 9.0, 2.6, 3.75, 1.7, "Individually addressable rows", "Old IDs + newly arrived IDs\nNo irreversible token averaging\nGroup identity is preserved", "bank", 10.5)
    block(ax, 13.55, 2.6, 3.9, 1.7, "SDU re-audit", "New query context $q_d$\nRecompute gates + signed values\nReuse static projections only", "sdu", 10.5)
    for a, b in ((3.9, 4.65), (8.2, 9.), (12.75, 13.55)):
        edge(ax, [(a, 3.45), (b, 3.45)])
    edge(ax, [(1.88, 6.2), (1.88, 5.3), (.2, 5.3), (.2, 4.6), (1.88, 4.6), (1.88, 4.3)],
         "x, y, t, ID", (2.65, 5.55), "bank")
    edge(ax, [(15.2, 6.05), (15.2, 5.15), (10.85, 5.15), (10.85, 4.3)], "read fixed features", (12.7, 5.35), "bank")
    block(ax, .5, 1.03, 7.75, .82, "Training: geometry/IDs reusable; learned features refresh after optimizer updates.", fontsize=9.7)
    block(ax, 9.0, 1.03, 8.45, .82, "Online: encode new sources once; append pairs causally; discard expired history.", fontsize=9.7)
    footer(ax, "These are implementation savings with the same selected evidence. A source window is a grouping unit, not a claim of statistical independence.")
    return fig


def sdu_window():
    fig, ax = canvas("SDU  |  Null-competitive evidence within revisable windows",
                     "A learned zero-valued competitor lets a window contribute little evidence without forcing its remaining tokens to sum to one.")
    block(ax, .5, 6.35, 2.9, 1.65, "Current query $q_d$", "Prior-anchored state\n+ visible-window context + age\nLayerNorm over channels", "sdu", 10.2)
    block(ax, .5, 3.75, 2.9, 1.65, "All retained tokens", "Source keys / values / context\nRelative geometry\nCausal masks", "bank", 10.2)
    block(ax, 4.1, 6.35, 3.15, 1.65, "Reweight old + new", "Query-key interaction\n$a_{jh}=\\sigma(u_{jh})$\nMasked tokens have zero weight", "sdu", 10.4)
    block(ax, 4.1, 3.75, 3.15, 1.65, "Revalue old + new", "SiLU(source + relation + query)\nHead readout $r_{jh}$\n$v_{jh}=s_h\\tanh(r_{jh}),\ s_h>0$", "sdu", 10.0)
    block(ax, 8.0, 3.75, 4.25, 4.25, "Within source window $g$", "$w_{jh}=\\dfrac{a_{jh}}{\\sum_{k\\in g}a_{kh}+n_g\\nu_h}$\n\n$m_{gh}=\\sum_{j\\in g}w_{jh}v_{jh}$\n$c_g=\\sum_{j\\in g}\\overline{w}_{j}u_j$\n$\\alpha_{gh}=\\sum_{j\\in g}w_{jh}\\leq 1$\n\nNull value and context are zero", "sdu", 11.5)
    block(ax, 13.0, 5.75, 4.45, 2.25, "Across arrived windows", "$\\rho_{gh}=\\sigma(\\mathrm{MLP}[q_d,c_g,\\Delta w_g])$\n$F_{dh}=\\sum_g\\rho_{gh}m_{gh}$\n$F_d=\\sum_hF_{dh}$", "sdu", 11)
    block(ax, 13.0, 3.75, 4.45, 1.25, "Effective evidence mass", "$M_{dh}=\\sum_g\\rho_{gh}\\alpha_{gh}$", "output", 11.5)
    block(ax, 8.0, 1.3, 4.25, 1.45, "Learned null competitor", "$\\nu_h=\\mathrm{softplus}(\\eta_h)+\\epsilon>0$\n$n_g$ = valid token count in window $g$", "neutral", 10.6)
    block(ax, .5, 1.3, 6.75, 1.45, "A new window changes the interpretation of old tokens", "Current $q_d$ drives every retained gate and signed value.\nThe full prefix potential is recomputed, never repeatedly added.", "neutral", 10.7)
    block(ax, 13., 1.3, 4.45, 1.45, "To birth-referenced readout", "Current potential + null-aware\ncontext / mass statistics", "output", 10.6)
    edge(ax, [(3.4, 7.18), (4.1, 7.18)])
    edge(ax, [(3.4, 4.58), (4.1, 4.58)])
    edge(ax, [(3.4, 6.55), (3.72, 6.55), (3.72, 5.86), (5.68, 5.86), (5.68, 5.4)], kind="sdu")
    edge(ax, [(1.95, 5.4), (1.95, 5.78), (4.55, 5.78), (4.55, 6.35)], kind="bank")
    edge(ax, [(7.25, 7.18), (8., 7.18)])
    edge(ax, [(7.25, 4.58), (8., 4.58)])
    edge(ax, [(10.13, 2.75), (10.13, 3.75)], "zero-valued alternative", (10.13, 3.19), kind="neutral", fontsize=9.8)
    edge(ax, [(12.25, 6.88), (13., 6.88)])
    edge(ax, [(12.25, 4.38), (13., 4.38)])
    edge(ax, [(15.23, 3.75), (15.23, 2.75)], kind="output")
    footer(ax, "At fixed query/content, replicating a complete window leaves its aggregate unchanged; all affinities zero implies zero raw potential.\nNull support and window discounts are learned aggregation coefficients, not calibrated association or reliability probabilities.")
    return fig


def innovation():
    fig, ax = canvas("SDU state and readout  |  Change relative to the birth evidence",
                     "A frozen differentiable reference prevents age-zero recalibration; new observations can change both old evidence and the current context.")
    block(ax, .5, 6.4, 4.0, 1.5, "Current prefix potential", "$F_d=\\sum_{g,h}\\rho_{gh}m_{gh}$\nAll selected arrived observations", "sdu", 11.5)
    block(ax, .5, 3.85, 4.0, 1.7, "Birth reference $F_0$", "Save first age-zero potential\nKeep its gradient path in training\nDo not replace it at later ages", "neutral", 10.8)
    op(ax, 6.1, 7.15, "−")
    op(ax, 9.35, 7.15, "+")
    block(ax, 8.15, 8.0, 2.4, .58, "Prior logit $z_0$", kind="bir", fontsize=11)
    block(ax, 11.3, 6.4, 6.15, 1.5, "Revised event belief", "$\\ell_d=z_0+F_d-F_0$\n$\\ell_0=z_0$ exactly; $\\widehat{p}_d=\\sigma(\\ell_d)$", "output", 12)
    edge(ax, [(4.5, 7.15), (5.87, 7.15)])
    edge(ax, [(4.5, 4.7), (6.1, 4.7), (6.1, 6.92)], "subtract once", (6.9, 5.8), "neutral")
    edge(ax, [(6.33, 7.15), (9.12, 7.15)], "$F_d-F_0$", (7.75, 7.45), "sdu", fontsize=12)
    edge(ax, [(9.35, 8.), (9.35, 7.38)], kind="bir")
    edge(ax, [(9.58, 7.15), (11.3, 7.15)])
    ax.text(.5, 3.15, "CONTEXT STATE  /  null support also gates the residual update", color=PALETTE["sdu"][1], fontsize=11, weight="bold")
    block(ax, .5, 1.15, 4.0, 1.55, "Null-aware residual inputs", "$[\\overline{u}_d,\\ F_{d,:},\\ \\log(1+M_{d,:})]$\nSummary retains rejected/null mass", "sdu", 10.7)
    block(ax, 5.25, 1.15, 3.3, 1.55, "Residual network", "Linear - SiLU\nLinear - tanh", "sdu", 11)
    block(ax, 8.9, 3.75, 3.65, 1.65, "Context support", "$\\gamma_d=\\dfrac{1}{GH}\\sum_{g,h}\\rho_{gh}\\alpha_{gh}$\n$0\\leq\\gamma_d\\leq1$", "sdu", 11.3)
    op(ax, 10.45, 1.93, "×")
    op(ax, 12.35, 1.93, "+")
    block(ax, 13.15, 1.15, 4.3, 1.55, "Next context", "$c_{d+1}=a_0+\\gamma_d\\,r_d$\nUsed at the next changed prefix", "sdu", 11)
    block(ax, 13.15, 4.1, 4.3, .8, "Fixed prior-conditioned anchor $a_0$", kind="bir", fontsize=10.5)
    edge(ax, [(4.5, 1.93), (5.25, 1.93)])
    edge(ax, [(8.55, 1.93), (10.22, 1.93)], "$r_d$", (9.3, 2.23), "sdu", fontsize=12)
    edge(ax, [(10.45, 3.75), (10.45, 2.16)], kind="sdu")
    edge(ax, [(10.68, 1.93), (12.12, 1.93)])
    edge(ax, [(13.15, 4.5), (12.8, 4.5), (12.8, 2.9), (12.35, 2.9), (12.35, 2.16)], kind="bir")
    edge(ax, [(12.58, 1.93), (13.15, 1.93)])
    footer(ax, "$G$ = max(1, number of nonempty source windows); $H$ counts heads. No new evidence: keep belief/context, advance policy age only.\nNull rejection zeros the current potential; the innovation still includes the frozen $-F_0$ term. This is a learned residual, not an exact Bayes factor.")
    return fig


def vow():
    fig, ax = canvas("VoW  |  Compare stopping risk with conditional continuation cost",
                     "Learn the cost of a deployable future policy first, then compare actions using only the current causal state.")
    block(ax, .5, 4.85, 3.55, 2.7, "Current causal state $S_d$", "Revised belief / change / prior\nNull-aware context + mass\nWindow count / concentration\nElapsed age + remaining horizon", "sdu", 10.8)
    block(ax, 4.95, 6.4, 5.0, 1.55, "Stop branch: model-based risk", "$R_s=\\min[c_{FN}\\widehat{p}_d,\\ c_{FP}(1-\\widehat{p}_d)]$\nCosts match the reporting threshold", "vow", 11)
    block(ax, 4.95, 3.65, 5.0, 1.8, "Wait branch: conditional return", "Linear - SiLU - Linear - Softplus\n$f_\\theta(S_d)\\approx\\mathbb{E}[J_{d+1}^{\\pi}\\mid S_d]$\n$Q_w=c_{step}+f_\\theta(S_d)$", "vow", 11.3)
    block(ax, 10.85, 4.85, 2.75, 2.1, "Compare", "$Q_w<R_s$ ?\nDeadline overrides", "vow", 12)
    block(ax, 14.6, 6.5, 2.85, 1.25, "Commit", "Finalize event + time", "output", 11)
    block(ax, 14.6, 3.85, 2.85, 1.25, "Wait", "Next window + re-audit", "sdu", 10.5)
    edge(ax, [(4.05, 6.7), (4.5, 6.7), (4.5, 7.18), (4.95, 7.18)])
    edge(ax, [(4.05, 5.55), (4.5, 5.55), (4.5, 4.55), (4.95, 4.55)])
    edge(ax, [(9.95, 7.18), (10.4, 7.18), (10.4, 6.35), (10.85, 6.35)])
    edge(ax, [(9.95, 4.55), (10.4, 4.55), (10.4, 5.45), (10.85, 5.45)])
    edge(ax, [(13.6, 6.35), (14.1, 6.35), (14.1, 7.13), (14.6, 7.13)], "no", (14.08, 7.44), "output")
    edge(ax, [(13.6, 5.45), (14.1, 5.45), (14.1, 4.48), (14.6, 4.48)], "yes", (14.06, 4.15), "sdu")
    block(ax, .5, 1.2, 9.45, 1.45, "Training: policy evaluation, then improvement", "Frozen-policy rollouts produce realized costs; regress their conditional expectation.\nLabels supervise training targets only; never choose the per-event hindsight minimum.", "neutral", 10.5)
    block(ax, 10.85, 1.2, 6.6, 1.45, "Admission precedes costly SDU work", "Default: a declared fixed band. Optional cheap birth head\npredicts opening cost from own/shared backbone features.", "neutral", 10.4)
    edge(ax, [(7.45, 2.65), (7.45, 3.65)], "supervise continuation cost", (7.45, 3.13), "vow", dashed=True, fontsize=10)
    footer(ax, "The stop branch depends on estimated belief quality; it is not a certified error bound. Current-step costs are added once; later policy costs belong inside $J^{\\pi}$.")
    return fig


def validate_boxes(fig):
    """Fail on text outside its intended block; keep diagrams reviewable."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for text, (x, y, w, h) in getattr(ax, "_block_text", []):
            extent = text.get_window_extent(renderer).transformed(ax.transData.inverted())
            if extent.x0 < x-.015 or extent.x1 > x+w+.015 or extent.y0 < y-.015 or extent.y1 > y+h+.015:
                raise RuntimeError("Text exceeds block: " + text.get_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT/"docs/figures/bst_final"))
    parser.add_argument("--dpi", type=int, default=210)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    for name, draw in (("01_overall", overall), ("02_bir", bir), ("03_shared_evidence", sources),
                       ("04_sdu_window", sdu_window), ("05_birth_innovation", innovation), ("06_vow", vow)):
        fig = draw()
        validate_boxes(fig)
        for extension in ("svg", "png"):
            path = output/(name+"."+extension)
            fig.savefig(str(path), dpi=args.dpi, bbox_inches="tight", pad_inches=.14, facecolor="white")
            print(path)
        plt.close(fig)


if __name__ == "__main__":
    main()
