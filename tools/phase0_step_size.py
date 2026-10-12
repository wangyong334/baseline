"""Phase 0: step-size error of fixed-node solvers for multi-fire LIF neurons and networks (synthetic inputs, no dataset).

Neuron (instantaneous synapses, soft reset):  u <- u e^{-dt/tau} + I,  n = floor(max(u, 0) / theta),  u <- u - n theta.
Input events sit on a fine grid (dt_f = tau / 384); simulating on that grid is exact for the given input (reference).

Coarse solvers with step h = alpha * tau (node updates per step in brackets):
    binary      all inputs of the step at its end, at most one spike per step (the usual discrete LIF)     [1]
    end         all inputs at the step end, multi-fire                                                  [1]
    mid         all inputs at the step midpoint, multi-fire                                             [1]
    end_k       end + expected delivery / reset factor kappa = (1 - e^-alpha) / alpha                   [1]
    end_b       end + Brownian-bridge hidden-crossing spike (uses the per-step jump variance)           [1]
    end_kb      end + kappa + bridge                                                                    [1]
    lin2        nodes at both step ends, linear weights (1 - xi, xi)                                    [2]
    lin3        nodes at 0, 1/2, 1, piecewise-linear weights                                            [3]
    end/2 end/3 end/4   end, multi-fire, at h/2, h/3, h/4                                               [2, 3, 4]

Experiments:
    E1  single neuron, four input regimes: subthreshold, drift-driven, fluctuation-driven, isolated strong events
    E2  threshold-margin law: count-flip probability under a membrane error vs the potential density at k * theta
    E3  node-update budget vs error (from E1 / E4)
    E4  3-layer random network 64-64-64: error vs depth, reduction-lemma self-check, branching gain

    python tools/phase0_step_size.py --out outputs/phase0
"""
import argparse
import json
import math
import os
import time

import numpy as np

TAU = 1.0
TAU2 = 1.0                      # downstream trace time constant for the "what the next layer sees" metric
THETA = 1.0
FINE = 384                      # fine steps per tau
T_TOTAL = 34                    # tau
WARM = 2.0                      # tau excluded from the metrics
ALPHAS = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0)
REC = 24                        # fine steps per reference record = the finest coarse step (alpha = 1/16)

SCHEMES = {
    "binary": dict(fracs=(1.0,), cap1=True),
    "end": dict(fracs=(1.0,)),
    "mid": dict(fracs=(0.5,)),
    "end_k": dict(fracs=(1.0,), kappa=True),
    "end_b": dict(fracs=(1.0,), bridge=True),
    "end_kb": dict(fracs=(1.0,), kappa=True, bridge=True),
    "lin2": dict(fracs=(0.0, 1.0)),
    "lin3": dict(fracs=(0.0, 0.5, 1.0)),
    "end/2": dict(fracs=(1.0,), sub=2),
    "end/3": dict(fracs=(1.0,), sub=3),
    "end/4": dict(fracs=(1.0,), sub=4),
}
UPDATES = {"binary": 1, "end": 1, "mid": 1, "end_k": 1, "end_b": 1, "end_kb": 1, "lin2": 2, "lin3": 3,
           "end/2": 2, "end/3": 3, "end/4": 4}


def node_weights(fracs, m):
    """Weights [K, m] that distribute an event at xi = (q + 1) / m over the nodes (hat functions; one node takes all)."""
    xi = (np.arange(m) + 1.0) / m
    if len(fracs) == 1:
        return np.ones((1, m))
    return np.stack([np.interp(xi, fracs, np.eye(len(fracs))[k]) for k in range(len(fracs))])


def simulate(provider, n_steps, h, fracs, weights, fire=True, cap1=False, kappa=False, bridge=False, group=1,
             seed=0, record_upre=None, shape=None):
    """Node solver for a feed-forward stack. provider(c, k) -> (I, S2, Nc) of layer 0 at node k of step c.
    weights: matrices of layers 1.. (fed by the previous layer's spikes). Records are taken every `group` steps.
    Returns dict cnt [R, L, *shape] (spikes summed over the group), mem / z [R, L, *shape] at the record times."""
    rng = np.random.RandomState(seed)
    nl = 1 + len(weights)
    sq = [w * w for w in weights]
    u = [np.zeros(shape) for _ in range(nl)]
    z = [np.zeros(shape) for _ in range(nl)]
    last_spike = [np.full(shape, -1e9) for _ in range(nl)]
    a_own = h / TAU
    k_own = (1 - math.exp(-a_own)) / a_own if kappa else 1.0
    k_out = (1 - math.exp(-h / TAU2)) / (h / TAU2) if kappa else 1.0
    n_rec = n_steps // group
    cnt = np.zeros((n_rec, nl) + shape)
    mem = np.zeros((n_rec, nl) + shape)
    zz = np.zeros((n_rec, nl) + shape)
    ups, act = [], []
    t_last = 0.0
    for c in range(n_steps):
        r = c // group
        for k, f in enumerate(fracs):
            t = (c + f) * h
            d, d2 = math.exp(-(t - t_last) / TAU), math.exp(-(t - t_last) / TAU2)
            t_last = t
            prev = None
            upre_step = []
            for l in range(nl):
                if l == 0:
                    cur, s2, nc = provider(c, k)
                else:
                    cur = (prev * k_out) @ weights[l - 1]
                    if bridge:
                        s2, nc = prev @ sq[l - 1], prev.sum(-1, keepdims=True)
                u0 = u[l]
                up = u0 * d + cur
                if fire:
                    n = np.floor(np.maximum(up, 0.0) / THETA)
                    if cap1:
                        n = np.minimum(n, 1.0)
                    if bridge:
                        var = s2 - cur * cur / np.maximum(nc, 1.0)
                        lev = (n + 1.0) * THETA
                        ok = (var > 1e-12) & (lev > u0) & (lev > up)
                        p = np.where(ok, np.exp(-2.0 * (lev - u0) * (lev - up) / np.where(ok, var, 1.0)), 0.0)
                        n = n + (rng.random_sample(shape) < p)
                    u[l] = up - n * THETA * k_own
                    last_spike[l] = np.where(n > 0, t, last_spike[l])
                else:
                    n = np.zeros(shape)
                    u[l] = up
                z[l] = z[l] * d2 + n * k_out
                cnt[r, l] += n
                prev = n
                upre_step.append(up)
            if record_upre is not None and (c + 1) % record_upre == 0 and k == len(fracs) - 1:
                ups.append(np.stack(upre_step))
                act.append(np.stack([last_spike[l] >= t - TAU for l in range(nl)]))
        if (c + 1) % group == 0:
            te = (c + 1) * h - t_last
            for l in range(nl):
                mem[r, l] = u[l] * math.exp(-te / TAU)
                zz[r, l] = z[l] * math.exp(-te / TAU2)
    out = {"cnt": cnt, "mem": mem, "z": zz}
    if record_upre is not None:
        out["upre"], out["active"] = np.stack(ups), np.stack(act)
    return out


def run_scheme(name, alpha, make_provider, n_fine, weights, shape, fire=True, seed=0):
    cfg = SCHEMES[name]
    sub = cfg.get("sub", 1)
    m = int(round(alpha * FINE)) // sub
    h = alpha * TAU / sub
    fracs = cfg["fracs"]
    provider = make_provider(m, node_weights(fracs, m))
    return simulate(provider, n_fine // m, h, fracs, weights, fire=fire, cap1=cfg.get("cap1", False),
                    kappa=cfg.get("kappa", False), bridge=cfg.get("bridge", False), group=sub, seed=seed, shape=shape)


def rms(x):
    return float(np.sqrt(np.mean(x * x)))


def compare(ref, rec, alpha):
    """Per-layer metrics of a coarse run against the reference at the coarse step ends."""
    ratio = int(round(alpha * FINE)) // REC
    n = rec["z"].shape[0]
    rcnt = ref["cnt"][:n * ratio].reshape((n, ratio) + ref["cnt"].shape[1:]).sum(1)
    rmem, rz = ref["mem"][ratio - 1::ratio][:n], ref["z"][ratio - 1::ratio][:n]
    w0 = int(math.ceil(WARM / alpha))
    win = int(round(4.0 / alpha))
    out = []
    for l in range(rec["z"].shape[1]):
        a, b = rec["cnt"][w0:, l], rcnt[w0:, l]
        nw = a.shape[0] // win
        la = a[:nw * win].reshape((nw, win) + a.shape[1:]).sum(1)
        lb = b[:nw * win].reshape((nw, win) + b.shape[1:]).sum(1)
        tot = max(float(b.sum()), 1e-12)
        out.append({
            "mem_err": rms(rec["mem"][w0:, l] - rmem[w0:, l]) / max(rms(rmem[w0:, l]), 1e-12),
            "z_err": rms(rec["z"][w0:, l] - rz[w0:, l]) / max(rms(rz[w0:, l]), 1e-12) if tot > 0 else float("nan"),
            "win_cnt_err": float(np.abs(a - b).sum()) / tot,
            "long_cnt_err": float(np.abs(la - lb).sum()) / max(float(lb.sum()), 1e-12),
            "rate_bias": float(a.sum()) / tot - 1.0,
        })
    return out


def consistency(rec_a, rec_2a, ref, alpha):
    """Same solver at alpha and 2 alpha, compared at the 2 alpha step ends (normalised by the reference trace)."""
    w0 = int(math.ceil(WARM / (2 * alpha)))
    za = rec_a["z"][1::2]
    n = min(za.shape[0], rec_2a["z"].shape[0])
    ratio = int(round(2 * alpha * FINE)) // REC
    rz = ref["z"][ratio - 1::ratio][:n]
    return [rms(za[w0:n, l] - rec_2a["z"][w0:n, l]) / max(rms(rz[w0:, l]), 1e-12) for l in range(za.shape[1])]


# ---------------------------------------------------------------- inputs

def single_inputs(regime, n, rng):
    j = T_TOTAL * FINE
    dtf = TAU / FINE

    def events(rate, draw):
        c = rng.poisson(rate * dtf, size=(j, n)).astype(np.int32)
        idx = np.repeat(np.arange(j * n), c.ravel())
        w = draw(idx.size)
        x = np.bincount(idx, w, j * n).reshape(j, n)
        x2 = np.bincount(idx, w * w, j * n).reshape(j, n)
        return x, x2, c.astype(np.float64)

    if regime == "subthreshold":
        return events(20, lambda k: rng.normal(0.05, 0.3, k)), False
    if regime == "drift":
        return events(20, lambda k: np.full(k, 0.06)), True
    if regime == "fluctuation":
        return events(40, lambda k: rng.normal(0.02, 0.3, k)), True
    if regime == "isolated":
        b = events(20, lambda k: rng.normal(0.0, 0.05, k))
        s = events(0.5, lambda k: rng.uniform(1.0, 1.8, k))
        return tuple(p + q for p, q in zip(b, s)), True
    raise ValueError(regime)


def single_provider(x, x2, c):
    def make(m, wk):
        if m == 1:
            return lambda i, k: (x[i], x2[i], c[i])
        nb = x.shape[0] // m
        xs = [np.einsum("q,bqn->bn", w, x.reshape(nb, m, -1)) for w in wk]
        x2s = np.einsum("bqn->bn", x2.reshape(nb, m, -1))
        cs = np.einsum("bqn->bn", c.reshape(nb, m, -1))
        return lambda i, k: (xs[k][i], x2s[i], cs[i])
    return make


def net_provider(cin, w1):
    w1sq = w1 * w1

    def make(m, wk):
        nb = cin.shape[0] // m
        if m == 1:
            agg, tot = [cin], cin
        else:
            flat = cin.reshape((nb, m) + cin.shape[1:])
            agg = [np.einsum("q,bq...->b...", w, flat) for w in wk]
            tot = flat.sum(1)

        def p(i, k):
            a = agg[k][i]
            return a @ w1, tot[i] @ w1sq, tot[i].sum(-1, keepdims=True)
        return p
    return make


def fine_projected_lin2(cin, w1, ws, m):
    """Independent fine-time engine for the reduction-lemma check: the lin2-projected input of every coarse step is
    injected at the fine steps of its two nodes (end node of step c, then start node of step c + 1 at the same time),
    and the network leaks on the fine grid in between."""
    nb = cin.shape[0] // m
    flat = cin.reshape((nb, m) + cin.shape[1:])
    xi = (np.arange(m) + 1.0) / m
    a_in = np.einsum("q,bq...->b...", 1 - xi, flat)
    b_in = np.einsum("q,bq...->b...", xi, flat)
    shape = cin.shape[1:2] + (w1.shape[1],)
    mats = [w1] + list(ws)
    u = [np.zeros(shape) for _ in mats]
    cnt = np.zeros((nb, len(mats)) + shape)
    beta = math.exp(-TAU / FINE / TAU)

    def inject(x, step):
        prev = x
        for l, w in enumerate(mats):
            up = u[l] + prev @ w
            n = np.floor(np.maximum(up, 0.0) / THETA)
            u[l] = up - n * THETA
            cnt[step, l] += n
            prev = n

    inject(a_in[0], 0)
    for j in range(1, cin.shape[0] + 1):
        for l in range(len(mats)):
            u[l] = u[l] * beta
        if j % m == 0:
            c = j // m - 1
            inject(b_in[c], c)
            if c + 1 < nb:
                inject(a_in[c + 1], c + 1)
    return cnt


# ---------------------------------------------------------------- E2 margin law

def flip_law(upre, active, eps_list, rng, delta=0.02):
    """Measured count-flip probability under Gaussian membrane noise vs prediction density(k theta) * E|eps|."""
    rows = []
    for group, mask in (("active", active), ("silent", ~active)):
        u = upre[mask]
        if u.size < 1000:
            continue
        kmax = int(max(1, np.ceil(u.max() / THETA))) + 1
        dens = sum(np.mean(np.abs(u - k * THETA) < delta) / (2 * delta) for k in range(1, kmax + 1))
        base = np.floor(np.maximum(u, 0) / THETA)
        for s in eps_list:
            e = rng.normal(0, s, u.size)
            flip = float(np.mean(np.floor(np.maximum(u + e, 0) / THETA) != base))
            pred = dens * s * math.sqrt(2 / math.pi)
            rows.append({"group": group, "eps_sd": s, "measured": flip, "predicted": pred, "density": dens,
                         "n": int(u.size)})
    return rows


def branching_gain(upre_post, w, rng, n_pre=16, delta=0.02):
    """Immediate count changes in a layer caused by one extra presynaptic spike (measured on recorded potentials)
    vs prediction density(k theta) * sum_j |w_ij|."""
    u = upre_post.reshape(-1, upre_post.shape[-1])
    kmax = int(max(1, np.ceil(u.max() / THETA))) + 1
    dens = sum(np.mean(np.abs(u - k * THETA) < delta) / (2 * delta) for k in range(1, kmax + 1))
    base = np.floor(np.maximum(u, 0) / THETA)
    pre = rng.choice(w.shape[0], n_pre, replace=False)
    meas = float(np.mean([np.abs(np.floor(np.maximum(u + w[i], 0) / THETA) - base).sum(1).mean() for i in pre]))
    pred = float(dens * np.abs(w[pre]).sum(1).mean())
    return {"measured": meas, "predicted": pred, "density": float(dens)}


# ---------------------------------------------------------------- main

def calibrate_net(rng, b, width, in_rate, mean_drive, sd_drive):
    """Weights giving each layer the same mean / sd drive per tau given the measured presynaptic rate."""
    def layer(rate, fan):
        mu = mean_drive / (fan * rate)
        sd = math.sqrt(max(sd_drive ** 2 / (fan * rate) - mu * mu, 1e-6))
        return rng.normal(mu, sd, (fan, width))
    w1 = layer(in_rate, width)
    ws = [layer(2.0, width), layer(2.0, width)]
    j = 8 * FINE
    cin = rng.poisson(in_rate / FINE, size=(j, b, width)).astype(np.float64)
    for _ in range(2):
        ref = simulate(lambda i, k: (cin[i] @ w1, None, None), j, TAU / FINE, (1.0,), ws, group=REC,
                       shape=(b, width))
        rates = ref["cnt"].sum((0, 2, 3)) / (8 * b * width)
        ws = [layer(max(float(rates[0]), 0.05), width), layer(max(float(rates[1]), 0.05), width)]
    return w1, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/phase0")
    ap.add_argument("--neurons", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rng = np.random.RandomState(0)
    n_fine = T_TOTAL * FINE
    report = {"config": {"tau": TAU, "tau2": TAU2, "theta": THETA, "fine_per_tau": FINE, "T_tau": T_TOTAL,
                         "warmup_tau": WARM, "alphas": ALPHAS, "neurons": args.neurons, "batch": args.batch}}

    # E1 single neuron
    report["E1"], report["E2"] = {}, {}
    for regime in ("subthreshold", "drift", "fluctuation", "isolated"):
        (x, x2, c), fire = single_inputs(regime, args.neurons, rng)
        make = single_provider(x, x2, c)
        ref = simulate(make(1, node_weights((1.0,), 1)), n_fine, TAU / FINE, (1.0,), [], fire=fire, group=REC,
                       record_upre=REC, shape=(args.neurons,))
        rate = float(ref["cnt"].sum() / (args.neurons * T_TOTAL))
        res, recs = {}, {}
        for name in SCHEMES:
            if not fire and name not in ("end", "mid", "lin2", "lin3", "end/2", "end/3", "end/4"):
                continue
            res[name], recs[name] = {}, {}
            for a in ALPHAS:
                rec = run_scheme(name, a, make, n_fine, [], (args.neurons,), fire=fire, seed=1)
                res[name][a] = compare(ref, rec, a)[0]
                recs[name][a] = rec
            for a in ALPHAS[:-1]:
                res[name][a]["consistency_a_2a"] = consistency(recs[name][a], recs[name][2 * a], ref, a)[0]
        report["E1"][regime] = {"rate_per_tau": rate, "results": {k: {str(a): v for a, v in d.items()}
                                                                  for k, d in res.items()}}
        if fire:
            report["E2"][regime] = flip_law(ref["upre"][:, 0], ref["active"][:, 0], (0.01, 0.03, 0.1, 0.3), rng)
        print("E1 %-12s done: reference rate %.3f spikes/tau (%.0f s)" % (regime, rate, time.time() - t0), flush=True)
        del x, x2, c, ref, recs

    # E4 network
    b, width = args.batch, 64
    w1, ws = calibrate_net(rng, b, width, 10.0, 1.0, 1.5)
    cin = rng.poisson(10.0 / FINE, size=(n_fine, b, width)).astype(np.float64)
    make = net_provider(cin, w1)
    ref = simulate(make(1, node_weights((1.0,), 1)), n_fine, TAU / FINE, (1.0,), ws, group=REC, record_upre=REC,
                   shape=(b, width))
    rates = (ref["cnt"].sum((0, 2, 3)) / (b * width * T_TOTAL)).tolist()
    net = {"rates_per_tau": rates, "results": {}}
    recs = {}
    for name in SCHEMES:
        net["results"][name], recs[name] = {}, {}
        for a in ALPHAS:
            rec = run_scheme(name, a, make, n_fine, ws, (b, width), seed=1)
            net["results"][name][str(a)] = compare(ref, rec, a)
            recs[name][a] = {"z": rec["z"]}
            if name == "lin2" and a in (0.25, 1.0):
                recs[name][a]["cnt"] = rec["cnt"]
        for a in ALPHAS[:-1]:
            cons = consistency(recs[name][a], recs[name][2 * a], ref, a)
            for l, v in enumerate(cons):
                net["results"][name][str(a)][l]["consistency_a_2a"] = v
    lemma = {}
    for a in (0.25, 1.0):
        m = int(round(a * FINE))
        node = recs["lin2"][a]["cnt"]
        fine = fine_projected_lin2(cin, w1, ws, m)
        lemma[str(a)] = {"spike_count_mismatches": int((node != fine).sum()), "entries": int(node.size),
                         "max_abs_diff": float(np.abs(node - fine).max())}
    net["lemma_check_lin2"] = lemma
    net["E2_flip"] = [flip_law(ref["upre"][:, l], ref["active"][:, l], (0.01, 0.03, 0.1, 0.3), rng)
                      for l in range(3)]
    net["branching_gain"] = [branching_gain(ref["upre"][:, l + 1], ws[l], rng) for l in range(2)]
    report["E4"] = net
    print("E4 network done: rates %s (%.0f s)" % (" ".join("%.2f" % r for r in rates), time.time() - t0), flush=True)

    with open(os.path.join(args.out, "phase0.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print_report(report)
    plot(report, args.out)
    print("PHASE0 FINISHED (%.0f s) -> %s" % (time.time() - t0, args.out), flush=True)


def print_report(rep):
    sp = ("binary", "end", "mid", "end_k", "end_b", "end_kb", "lin2", "lin3", "end/2", "end/3", "end/4")
    for regime, d in rep["E1"].items():
        key = "mem_err" if regime == "subthreshold" else "z_err"
        print("\nE1 %s (rate %.3f/tau): %s vs alpha" % (regime, d["rate_per_tau"], key))
        names = [s for s in sp if s in d["results"]]
        print("%-7s " % "alpha" + " ".join("%7s" % s for s in names))
        for a in ALPHAS:
            print("%-7g " % a + " ".join("%7.4f" % d["results"][s][str(a)][key] for s in names))
        if regime != "subthreshold":
            print("  rate bias:")
            for a in ALPHAS:
                print("%-7g " % a + " ".join("%+7.3f" % d["results"][s][str(a)]["rate_bias"] for s in names))
            print("  long-window (4 tau) count error:")
            for a in ALPHAS:
                print("%-7g " % a + " ".join("%7.3f" % d["results"][s][str(a)]["long_cnt_err"] for s in names))
    print("\nE2 flip law (single neuron): measured vs predicted density(k theta) * E|eps|")
    for regime, rows in rep["E2"].items():
        for r in rows:
            print("  %-11s %-6s eps_sd %.2f: measured %.4f predicted %.4f (density %.3f)" % (
                regime, r["group"], r["eps_sd"], r["measured"], r["predicted"], r["density"]))
    net = rep["E4"]
    print("\nE4 network rates per layer: %s" % " ".join("%.3f" % r for r in net["rates_per_tau"]))
    for l in range(3):
        print("E4 layer %d z_err vs alpha" % (l + 1))
        print("%-7s " % "alpha" + " ".join("%7s" % s for s in sp))
        for a in ALPHAS:
            print("%-7g " % a + " ".join("%7.4f" % net["results"][s][str(a)][l]["z_err"] for s in sp))
    print("E4 rate bias layer 3:")
    for a in ALPHAS:
        print("%-7g " % a + " ".join("%+7.3f" % net["results"][s][str(a)][2]["rate_bias"] for s in sp))
    print("E4 consistency (same solver at alpha vs 2 alpha), layer 3:")
    for a in ALPHAS[:-1]:
        print("%-7g " % a + " ".join("%7.4f" % net["results"][s][str(a)][2]["consistency_a_2a"] for s in sp))
    print("E4 reduction-lemma check (lin2 node solver vs independent fine engine on the projected input): %s" %
          json.dumps(net["lemma_check_lin2"]))
    for l, g in enumerate(net["branching_gain"]):
        print("E4 branching gain into layer %d: measured %.3f predicted %.3f" % (l + 2, g["measured"], g["predicted"]))
    for l, rows in enumerate(net["E2_flip"]):
        for r in rows:
            print("E4 flip layer %d %-6s eps_sd %.2f: measured %.4f predicted %.4f" % (
                l + 1, r["group"], r["eps_sd"], r["measured"], r["predicted"]))
    print("\nE3 budget at alpha = 1 (updates per step: z_err fluctuation | z_err network layer 3)")
    for s in sorted(sp, key=lambda k: UPDATES[k]):
        print("  %-7s %d : %.4f | %.4f" % (s, UPDATES[s], rep["E1"]["fluctuation"]["results"][s]["1.0"]["z_err"],
                                           net["results"][s]["1.0"][2]["z_err"]))


def plot(rep, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    show = ("binary", "end", "mid", "lin2", "end/2", "end_k", "end_b", "end_kb")
    color = {s: "C%d" % i for i, s in enumerate(show)}
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    panels = [(r, rep["E1"][r]["results"], "mem_err" if r == "subthreshold" else "z_err", None)
              for r in ("subthreshold", "drift", "fluctuation", "isolated")]
    panels.append(("network layer 3", rep["E4"]["results"], "z_err", 2))
    a = np.array(ALPHAS)
    for ax, (title, res, key, layer) in zip(axes, panels):
        for s in show:
            if s not in res:
                continue
            v = [res[s][str(x)][layer][key] if layer is not None else res[s][str(x)][key] for x in ALPHAS]
            ax.loglog(a, v, marker="o", color=color[s], label=s)
        for p, ls in ((0.5, ":"), (1, "--"), (2, "-.")):
            ax.loglog(a, 0.05 * (a / a[0]) ** p, "k" + ls, lw=0.8, label="slope %g" % p)
        ax.set_title(title)
        ax.set_xlabel("alpha = dt / tau")
        ax.set_ylabel(key)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_error_vs_alpha.png"), dpi=120)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    pts = [r for rows in rep["E2"].values() for r in rows] + [r for rows in rep["E4"]["E2_flip"] for r in rows]
    ax.loglog([r["predicted"] for r in pts], [r["measured"] for r in pts], "o")
    lim = [1e-4, 1]
    ax.loglog(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("predicted flip prob. density(k theta) E|eps|")
    ax.set_ylabel("measured flip prob.")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_flip_law.png"), dpi=120)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (title, getter) in zip(axes, (
            ("fluctuation, alpha=1", lambda s: rep["E1"]["fluctuation"]["results"][s]["1.0"]["z_err"]),
            ("network layer 3, alpha=1", lambda s: rep["E4"]["results"][s]["1.0"][2]["z_err"]))):
        for s in SCHEMES:
            ax.plot(UPDATES[s], getter(s), "o")
            ax.annotate(s, (UPDATES[s], getter(s)), fontsize=7)
        ax.set_xlabel("node updates per step")
        ax.set_ylabel("z_err")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_budget.png"), dpi=120)


if __name__ == "__main__":
    main()
