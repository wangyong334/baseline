"""V17 stopping rule (Eq. 7): when a decision unit fires, calibrated on held-out recordings.

Columns of an event's trajectory: 0 = the backbone's prior z0 at arrival (no decision unit yet), 1 + k = the unit's
membrane after step k. The event commits at the first column c whose margin |Z_c - tau| reaches the bound b_c, else at
its last available column (the deadline, or the end of the recording). Risk = probability that the committed decision
differs from the deadline decision (11-10 definition of "the evidence suffices"). Per column, among the events still
open there, the change probability as a function of the margin is fitted by isotonic regression (non-increasing, no
bins); b_c is the smallest margin from which every isotonic block has a Clopper-Pearson upper bound below alpha (level
delta / number of columns, Bonferroni). Column 0 is the entry rule: an event gets a decision unit only while its prior
is not yet safe, so there is no hand-set band. Constants: alpha (declared risk) and delta (calibration confidence); the
deadline is the latency requirement. The bounds narrow towards the deadline by themselves (less left to come).
V18 (11-14): rule "window" (default). The change probability r(m) of an open event is taken non-increasing in its
margin m (the isotonic assumption of V17). Then the events just BELOW a candidate bound b change at least as often as
an event at b, so their change rate, certified below alpha, certifies r(b) < alpha and, by monotonicity, every margin
above b. Candidates are scanned from the largest margin down (fixed-sequence testing: the hypotheses r(b) >= alpha
are nested, so no multiplicity correction); each uses the n_w events at or below it (n_w = 4 x the fewest events that
can certify alpha without any change); the bound is the last candidate that passes. Why: rule "isotonic" (V17)
certifies every isotonic block from the top down, so the block above the largest late flip must certify alpha on
its own; one flip among the few hundred most confident units of a column made its bound infinite (V18 local,
v5s37: seed 0 at step 0, seed 1 at steps 0-6; born units then waited 3.1 vs 7.2 windows, energy x 2, same accuracy).
A tail rule (the change rate of everything above b) is not used: millions of easy events would average it away.
"""
import math

import numpy as np
from scipy.stats import beta as beta_dist


def pav_nonincreasing(y):
    """Isotonic (non-increasing) least-squares fit of y, pool-adjacent-violators."""
    vals, wts, lens = [], [], []
    for yi in -np.asarray(y, float):
        vals.append(yi)
        wts.append(1.0)
        lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            n = lens[-2] + lens[-1]
            vals.pop(), wts.pop(), lens.pop()
            vals[-1], wts[-1], lens[-1] = v, w, n
    return -np.repeat(vals, lens)


def ucb(k, n, level):
    return 1.0 if k >= n else float(beta_dist.ppf(1.0 - level, k + 1, n - k))


def bound_from_changes(m, change, alpha, level):
    """Smallest margin from which every isotonic block of the change rate has an upper bound below alpha."""
    if m.size == 0:
        return float("inf")
    order = np.argsort(m, kind="stable")
    ms, cs = m[order], change[order].astype(float)
    fit = pav_nonincreasing(cs)
    starts = np.flatnonzero(np.r_[True, fit[1:] != fit[:-1]])
    ends = np.r_[starts[1:], len(fit)]
    b = float("inf")
    for a, e in zip(starts[::-1], ends[::-1]):              # from the largest margins down
        if ucb(int(cs[a:e].sum()), int(e - a), level) >= alpha:
            break
        b = float(ms[a])
    return b


def window_size(alpha, level, factor=4):
    """Events per certification window: factor x the fewest events that can certify alpha without any change."""
    return factor * int(math.ceil(math.log(level) / math.log(1.0 - alpha)))


def bound_from_window(m, change, alpha, level, n_w=None):
    """Smallest margin b reached by scanning from the largest margin down while the n_w events at or below each
    candidate have a change rate with a Clopper-Pearson upper bound below alpha (monotone change probability)."""
    N = int(m.size)
    n_w = window_size(alpha, level) if n_w is None else int(n_w)
    if N == 0:
        return float("inf")
    order = np.argsort(-m, kind="stable")
    ms = m[order]
    cs = np.r_[0, np.cumsum(change[order].astype(np.int64))]
    i = np.arange(N)
    j = np.minimum(N, i + n_w)
    k, n = cs[j] - cs[i], j - i
    full = n == n_w
    ok = np.zeros(N, bool)
    if full.any():                                       # the largest change count that still certifies alpha
        k_max = -1
        while ucb(k_max + 1, n_w, level) < alpha:
            k_max += 1
        ok[full] = k[full] <= k_max
    for t in np.nonzero(~full)[0]:                       # truncated windows at the smallest margins
        ok[t] = ucb(int(k[t]), int(n[t]), level) < alpha
    fail = np.nonzero(~ok)[0]
    first = int(fail[0]) if fail.size else N
    return float("inf") if first == 0 else float(ms[first - 1])


def calibrate(Z, last, tau, alpha, delta, collapse=True, rule="window", factor=4):
    """Bounds b_0 .. b_{C-1} of a split's trajectories Z [N, C] (column 0 = prior); last [N] = last available column.
    collapse: a unit's bound never widens from one step to the next (b_c <= b_{c-1} for c >= 2): with less still to
    come, the same margin is at least as safe. It replaces an infinite bound where too few held-out events remain open
    to certify one; the realised change rate on test checks the assumption. rule: "window" (V18) or "isotonic" (V17);
    factor: window rule only, n_w = factor x the fewest events that certify alpha without any change."""
    if rule not in ("window", "isotonic"):
        raise ValueError("unknown stopping rule %r" % (rule,))
    if rule == "window":
        fit = lambda m, ch, a, lv: bound_from_window(m, ch, a, lv, window_size(a, lv, factor))  # noqa: E731
    else:
        fit = bound_from_changes
    n, C = Z.shape
    final = Z[np.arange(n), last] >= tau
    open_ = np.ones(n, bool)
    level = delta / max(C - 1, 1)
    bounds = []
    for c in range(C - 1):
        cand = open_ & (c < last)
        m = np.abs(Z[cand, c] - tau)
        change = (Z[cand, c] >= tau) != final[cand]
        b = fit(m, change, alpha, level)
        if collapse and c >= 2:
            b = min(b, bounds[-1])
        bounds.append(b)
        open_ &= ~(cand & (np.abs(Z[:, c] - tau) >= b))
    bounds.append(0.0)                                       # deadline: whatever is still open commits
    return bounds


def commit(Z, last, tau, bounds):
    """Committed column of every event: the first c < last with |Z_c - tau| >= b_c, else last."""
    L = last.copy()
    done = np.zeros(len(last), bool)
    for c in range(Z.shape[1]):
        hit = ~done & (c < last) & (np.abs(Z[:, c] - tau) >= bounds[c])
        L[hit] = c
        done |= hit
    return L


def unit_load(win, L, n_win_of, rec, n_steps, cost):
    """Decision units per window and their cost. An event committed at column L >= 1 ran steps 0 .. L - 1, step k in
    window win + k. cost = a function k -> cost of step k of one unit, an array [N, n_steps] (cost of step k of event
    i), or a list of those (a list of per-window costs is then returned). Returns per-window arrays (all windows of all
    recordings)."""
    costs = list(cost) if isinstance(cost, (list, tuple)) else [cost]
    units, loads = [], [[] for _ in costs]
    for r in np.unique(rec):
        sel = rec == r
        w, steps = win[sel], np.maximum(L[sel], 0)
        nw = int(n_win_of[r])
        u = np.zeros(nw, np.float64)
        mc = [np.zeros(nw, np.float64) for _ in costs]
        sub = [c if callable(c) else c[sel] for c in costs]
        for k in range(n_steps):
            run = steps > k
            if not run.any():
                continue
            pos = w[run] + k
            u += np.bincount(pos, minlength=nw)[:nw]
            for m, c in zip(mc, sub):
                wt = np.full(pos.size, float(c(k))) if callable(c) else c[run, k].astype(np.float64)
                m += np.bincount(pos, weights=wt, minlength=nw)[:nw]
        units.append(u.astype(np.int64))
        for out, m in zip(loads, mc):
            out.append(m)
    loads = [np.concatenate(m) for m in loads]
    return np.concatenate(units), (loads if isinstance(cost, (list, tuple)) else loads[0])
