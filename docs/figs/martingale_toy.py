"""Toy example for the martingale-consistency loss (V18, 11-14): per-step BCE cannot see whether a belief sequence is
coherent over time; the martingale term can, and coherence is what a sequential stopping rule relies on.

Two belief sequences over the same events (label y ~ Bernoulli(0.3), K = 6 windows):
    A  exact posteriors of i.i.d. Gaussian evidence, P_k = sigmoid(logit(0.3) + sum_{j<=k} 2 mu x_j): a martingale
    B  the same values of P_k, shuffled among the events of the same label separately at every step
Per step, (P_k, y) has exactly the same distribution in A and B, so the per-step BCE is identical; only the links
between consecutive steps differ (B is not a realisable predictor -- its shuffle uses the label -- it only isolates the
temporal coherence). Measured: per-step BCE, the martingale term (model/sdu.py), and what a stopping rule does with
each: commit at the first step whose |logit P_k - tau| >= b; reversal = the committed side differs from the side at the
last step; the smallest bound whose reversal rate stays <= ALPHA and the mean stopping step it costs.
Writes docs/figs/martingale_toy.png and prints the numbers.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from model.sdu import martingale_penalty  # noqa: E402

N, K, MU, PRIOR, TAU, BOUND, ALPHA = 20000, 6, 0.5, 0.3, 0.0, 2.0, 0.03


def sequences(seed=0):
    rng = np.random.RandomState(seed)
    y = rng.rand(N) < PRIOR
    x = rng.randn(N, K) + np.where(y, MU, -MU)[:, None]
    U = np.log(PRIOR / (1 - PRIOR)) + np.cumsum(2 * MU * x, 1)
    A = 1.0 / (1.0 + np.exp(-U))
    B = A.copy()
    for k in range(K):                                  # shuffle within each label, independently per step
        for lab in (True, False):
            i = np.nonzero(y == lab)[0]
            B[i, k] = A[rng.permutation(i), k]
    return y, A, B


def stopping(L, bound):
    stop = np.full(len(L), K - 1)
    for k in range(K - 1, -1, -1):                      # first step that crosses the bound
        stop = np.where(np.abs(L[:, k] - TAU) >= bound, k, stop)
    commit = L[np.arange(len(L)), stop] >= TAU
    return (commit != (L[:, -1] >= TAU)).mean(), stop.mean() + 1


def stats(P, y):
    eps = 1e-7
    bce = -(y[:, None] * np.log(P + eps) + (~y)[:, None] * np.log(1 - P + eps)).mean(0)
    U = torch.as_tensor(np.log(P / (1 - P)), dtype=torch.float64)
    mart = float(martingale_penalty(U, torch.ones_like(U), torch.ones(len(y), dtype=torch.float64)))
    L = np.log(P / (1 - P))
    rev, _ = stopping(L, BOUND)
    flips = (np.diff((L >= TAU).astype(int), axis=1) != 0).sum(1).mean()
    for b in np.arange(0.25, 12.01, 0.05):              # smallest bound that keeps reversals <= ALPHA
        r, wait = stopping(L, b)
        if r <= ALPHA:
            break
    return bce, mart, rev, flips, b, wait


def main():
    y, A, B = sequences()
    res = {"A: martingale (exact posterior)": stats(A, y), "B: same per-step values, shuffled": stats(B, y)}
    for name, (bce, mart, rev, flips, b, wait) in res.items():
        print("%-36s BCE per step %s | martingale term %.2e | side flips per event %.2f | stop at |U| >= %.0f: "
              "reversed later %.3f | reversals <= %.2f need bound %.2f and %.2f windows on average" % (
                  name, " ".join("%.4f" % v for v in bce), mart, flips, BOUND, rev, ALPHA, b, wait))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    rng = np.random.RandomState(3)
    pick = rng.choice(np.nonzero(y)[0], 6, replace=False)
    for i, (name, P) in enumerate((("A", A), ("B", B))):
        for e in pick:
            ax[i].plot(range(1, K + 1), P[e], marker="o", lw=1)
        ax[i].axhline(1 / (1 + np.exp(-TAU)), color="gray", ls="--", lw=0.8)
        ax[i].set_title("%s: %s" % (name, "martingale" if name == "A" else "same marginals, shuffled"), fontsize=10)
        ax[i].set_xlabel("window")
        ax[i].set_ylim(0, 1)
    ax[0].set_ylabel("belief P(target) of 6 target events")
    labels = ["BCE\n(mean of steps)", "martingale\nterm x 10", "reversed after\nstop at |U|>=2",
              "windows waited\nfor <= 3% reversals"]
    vals = {k: [res[k][0].mean(), res[k][1] * 10, res[k][2], res[k][5]] for k in res}
    xs = np.arange(4)
    for off, (k, v) in zip((-0.18, 0.18), vals.items()):
        ax[2].bar(xs + off, v, 0.36, label=k[0])
        for xx, vv in zip(xs + off, v):
            ax[2].text(xx, vv, "%.2f" % vv, ha="center", va="bottom", fontsize=7)
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(labels, fontsize=7)
    ax[2].legend(fontsize=8)
    ax[2].set_title("same BCE, different coherence", fontsize=10)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "martingale_toy.png")
    fig.savefig(out, dpi=150)
    print("figure:", out)


if __name__ == "__main__":
    main()
