"""Render an inspectable single-query diagnostic as SVG and PNG."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True, help="output path stem, no extension")
    args = ap.parse_args()
    with open(args.trace, encoding="utf-8") as fh:
        trace = json.load(fh)
    steps = trace["steps"]
    x = np.array([s["extra_wait_ms"] for s in steps])
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.3), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x, [s["logit"] for s in steps], "o-", color="#24649b", label="Updated belief logit")
    ax.axhline(trace["prior_logit"], ls="--", color="#777777", label="Initial logit")
    ax.set(title="Belief = initial logit + current full-prefix correction", ylabel="Internal logit")
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    if "R_stop" in steps[0]:
        ax.plot(x, [s["R_stop"] for s in steps], "o-", label="Model stopping risk")
        ax.plot(x, [s["Q_wait"] for s in steps], "s-", label="Predicted wait cost")
        ax.legend(fontsize=9)
    ax.set(title="VoW compares two estimated costs", ylabel="Decision cost")
    ax = axes[1, 0]
    old = {token["neighbor_id"] for token in steps[0]["tokens"]}
    chosen = sorted(steps[0]["tokens"], key=lambda t: abs(t["total_contribution"]), reverse=True)[:3]
    for token in chosen:
        source = token["neighbor_id"]
        values = []
        for step in steps:
            row = next((t for t in step["tokens"] if t["neighbor_id"] == source), None)
            values.append(row["value"][0] if row else np.nan)
        ax.plot(x, values, "o-", ms=4, label="Source %d, head 0" % source)
    ax.set(title="The same old sources receive revised values", ylabel="Signed model value")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    old_sum, later_sum = [], []
    for step in steps:
        old_sum.append(sum(t["total_contribution"] for t in step["tokens"] if t["neighbor_id"] in old))
        later_sum.append(sum(t["total_contribution"] for t in step["tokens"] if t["neighbor_id"] not in old))
    ax.plot(x, old_sum, "o-", label="Sources already visible at age 0")
    ax.plot(x, later_sum, "s-", label="Sources arriving later")
    ax.plot(x, np.array(old_sum) + later_sum, "--", color="#303030", label="Total replacement correction")
    ax.set(title="Actual normalized contributions reconstruct the correction", ylabel="Contribution to internal logit")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Extra acquisition wait (ms)")
        ax.grid(alpha=.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("%s / event %s: model explanation, not causal attribution\n"
                 "Forced admission diagnostic; later steps after a policy commit would be counterfactual."
                 % (trace["recording"], trace["event_id"]), fontsize=12)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(args.out + "." + ext, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
