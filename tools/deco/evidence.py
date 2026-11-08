"""DeCo step 2 (final design, 11-06): pending events of a split and their neighbourhood evidence tokens. CPU only.

Pending = arrival evidence z0 inside the band tau +- logit(1 - alpha) (tau = report threshold 0.9 on the calibrated
scale). For a pending event e (window s, pixel x, y, time t), in every window s + dw (dw = -P .. D), among the events
within --radius px of (x, y) (e itself excluded):
    the K_NEAR nearest events of any kind, and the K_EVID nearest non-background events (z0 > lower bound at
    --filter-alpha) not already picked; plus the number of events and of non-background events within the radius.
P and D: whole windows inside --past-ms and --wait-ms. K_NEAR and K_EVID are given per 50 ms and scaled with the window
length (per window max(1, round(k * window_ms / 50))), so every slice length gets the same evidence budget per second.
The radius is a plain hyperparameter (val-selected 32 px, flat between 12 and 32 px; DECO_PROBE_LOG 4.9).
--radius 0 = no cap (V10 design, 11-07): the K nearest events of the whole window; the decision neurons learn their
receptive field instead (train_dnm.py --design v10), and the counts become the window totals.
Token features: dx, dy (px), dt (ms), dw (windows), polarity, neighbour z0 (clipped to +-20). No labels of neighbours.
Usage: python tools/deco/evidence.py --z0-dir <dir>/z0_<split> --split <split> --data-root <root> --alpha 0.01 --out <npz>
"""
import argparse
import glob
import math
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

THRESHOLD = 0.9


def _logit(p):
    return math.log(p / (1.0 - p))


def bounds(tau, alpha):
    half = _logit(1.0 - float(alpha))
    return tau - half, tau + half


def band_bounds(tau, alpha, band="sym"):
    """Pending band. sym: tau +- logit(1 - alpha) (V10). martingale (V11): if the calibrated posterior p is a
    martingale, Ville's inequality bounds the chance that the decision ever flips by p0 / p_tau (background side) and
    (1 - p0) / (1 - p_tau) (target side); the event needs a decision neuron unless that bound is <= alpha:
    (logit(alpha p_tau), logit(1 - alpha (1 - p_tau))), p_tau = sigmoid(tau)."""
    if band == "sym":
        return bounds(tau, alpha)
    if band != "martingale":
        raise ValueError("unknown band %s" % band)
    p_tau = 1.0 / (1.0 + math.exp(-tau))
    return _logit(float(alpha) * p_tau), _logit(1.0 - float(alpha) * (1.0 - p_tau))


def per_window(k, window_ms):
    return max(1, int(round(k * window_ms / 50.0)))


def windows_within(ms, window_ms):
    return int(math.floor(ms / float(window_ms) + 1e-9))


def knn(tree, ids, qxy, k, radius):
    kk = min(k, ids.size)
    dist, nb = tree.query(qxy, k=kk, distance_upper_bound=radius)
    if kk == 1:
        dist, nb = dist[:, None], nb[:, None]
    ok = np.isfinite(dist)
    return np.where(ok, ids[np.minimum(nb, ids.size - 1)], -1), ok


def first_k(cand, ok, k):
    order = np.argsort(~ok, axis=1, kind="stable")[:, :k]
    c, o = np.take_along_axis(cand, order, 1), np.take_along_axis(ok, order, 1)
    if c.shape[1] < k:
        pad = k - c.shape[1]
        c = np.concatenate([c, np.full((len(c), pad), -1)], 1)
        o = np.concatenate([o, np.zeros((len(o), pad), bool)], 1)
    return c, o


def candidate_picks(loc, zc, win, n_win, pend, offsets, k_near, k_evid, flo, radius):
    """Evidence candidates of the pending events `pend` (indices into the recording): for every window offset, the
    k_near nearest events of any kind and the k_evid nearest non-background events (zc > flo) not already picked,
    the pending event itself excluded. Returns pick [n, n_off, K] (event indices, -1 = none), pick_ok, nall, nevid.
    Shared by the evidence files and the nested network's evaluation (tools/deco/train_nested.py)."""
    K = k_near + k_evid
    n = pend.size
    unbounded = not np.isfinite(radius)
    pick_all = np.full((n, len(offsets), K), -1, dtype=np.int64)
    ok_all = np.zeros((n, len(offsets), K), dtype=bool)
    nall = np.zeros((n, len(offsets)), dtype=np.float32)
    nevid = np.zeros((n, len(offsets)), dtype=np.float32)
    s_all, x_all, y_all = win[pend], loc[pend, 0], loc[pend, 1]
    order = np.argsort(win, kind="stable")
    bnd = np.searchsorted(win[order], np.arange(n_win + 1))
    trees = {}
    for oi, dw in enumerate(offsets):
        target_w = s_all + dw
        for w in np.unique(target_w):
            if w < 0 or w >= n_win:
                continue
            ids = order[bnd[w]:bnd[w + 1]]
            if ids.size == 0:
                continue
            if w not in trees:
                ev = ids[zc[ids] > flo]
                trees[w] = (cKDTree(np.stack([loc[ids, 0], loc[ids, 1]], 1).astype(np.float64)), ids,
                            cKDTree(np.stack([loc[ev, 0], loc[ev, 1]], 1).astype(np.float64)) if ev.size else None, ev)
            tree_a, ids_a, tree_e, ids_e = trees[w]
            q = np.nonzero(target_w == w)[0]
            me = pend[q][:, None]
            qxy = np.stack([x_all[q], y_all[q]], 1).astype(np.float64)
            ca, oa = knn(tree_a, ids_a, qxy, k_near + 1, radius)
            self_a = oa & (ca == me)
            oa &= ~self_a
            near, near_ok = first_k(ca, oa, k_near)
            nall[q, oi] = (ids_a.size if unbounded else
                           tree_a.query_ball_point(qxy, radius, return_length=True)) - self_a.any(1)
            if tree_e is not None:
                ce, oe = knn(tree_e, ids_e, qxy, k_evid + k_near + 1, radius)
                self_e = oe & (ce == me)
                nevid[q, oi] = (ids_e.size if unbounded else
                                tree_e.query_ball_point(qxy, radius, return_length=True)) - self_e.any(1)
                oe &= ~self_e
                oe &= ~(ce[:, :, None] == np.where(near_ok, near, -2)[:, None, :]).any(2)
                evid, evid_ok = first_k(ce, oe, k_evid)
            else:
                evid, evid_ok = np.full((q.size, k_evid), -1), np.zeros((q.size, k_evid), bool)
            pick_ok = np.concatenate([near_ok, evid_ok], 1)
            pick_all[q, oi] = np.where(pick_ok, np.concatenate([near, evid], 1), -1)
            ok_all[q, oi] = pick_ok
    return pick_all, ok_all, nall, nevid


def one_recording(job):
    (path_z0, path_data, alpha, filter_alpha, past, depth, window_ms, radius, k_near, k_evid) = job[:10]
    band = job[10] if len(job) > 10 else "sym"
    with np.load(path_z0) as tr:
        z0_all = np.asarray(tr["traj"][:, 0], np.float32)
        log_w = float(tr["log_w"])
    with np.load(path_data) as d:
        loc, norm = np.asarray(d["ev_loc"]), np.asarray(d["evs_norm"])
    tau = _logit(THRESHOLD) - log_w
    lo, hi = band_bounds(tau, alpha, band)
    flo, _ = bounds(tau, filter_alpha)
    zc = np.clip(np.nan_to_num(z0_all, nan=-20.0), -20.0, 20.0)
    win = (loc[:, 2] // window_ms).astype(np.int64)
    n_win = int(win.max()) + 1
    pend = np.nonzero((z0_all > lo) & (z0_all < hi))[0]
    offsets = list(range(-past, depth + 1))
    pick, pick_ok, nall, nevid = candidate_picks(loc, zc, win, n_win, pend, offsets, k_near, k_evid, flo, radius)
    s_all, x_all, y_all, t_all = win[pend], loc[pend, 0], loc[pend, 1], loc[pend, 2]
    safe = np.where(pick_ok, pick, 0)
    dwv = np.broadcast_to(np.asarray(offsets, np.float64)[None, :, None], safe.shape)
    block = np.stack([loc[safe, 0] - x_all[:, None, None], loc[safe, 1] - y_all[:, None, None],
                      loc[safe, 2] - t_all[:, None, None], dwv, norm[safe, 3], zc[safe]], -1).astype(np.float32)
    block[~pick_ok] = 0.0
    tok = block.astype(np.float16)
    meta = {"idx": pend, "s": s_all, "x": x_all, "y": y_all, "t": t_all, "p": norm[pend, 3].astype(np.float32),
            "label": norm[pend, 4] > 0.5, "z0": z0_all[pend], "F": np.minimum(depth, n_win - 1 - s_all)}
    return meta, tok, pick_ok, nall, nevid, log_w, int(z0_all.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--z0-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="pending band risk (0.01 = the 2%% band)")
    ap.add_argument("--filter-alpha", type=float, default=0.01, help="non-background filter for evidence tokens")
    ap.add_argument("--out", required=True)
    ap.add_argument("--past-ms", type=float, default=500.0)
    ap.add_argument("--wait-ms", type=float, default=500.0)
    ap.add_argument("--radius", type=float, default=32.0, help="local radius cap, px (0 = no cap)")
    ap.add_argument("--k-near", type=int, default=8, help="nearest events per 50 ms")
    ap.add_argument("--k-evid", type=int, default=8, help="nearest non-background events per 50 ms")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--band", choices=("sym", "martingale"), default="sym",
                    help="pending band: sym = tau +- logit(1 - alpha) (V10); martingale = Ville bound (V11)")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.z0_dir, "*.npz")))
    if not files:
        raise RuntimeError("no z0 files in %s" % args.z0_dir)
    with np.load(files[0]) as tr:
        window_ms = int(tr["window_ms"])
    past = windows_within(args.past_ms, window_ms)
    depth = windows_within(args.wait_ms, window_ms)
    if depth < 1:
        raise ValueError("--wait-ms %.0f is shorter than one window (%d ms)" % (args.wait_ms, window_ms))
    k_near, k_evid = per_window(args.k_near, window_ms), per_window(args.k_evid, window_ms)
    radius = args.radius if args.radius > 0 else np.inf
    jobs = [(f, os.path.join(args.data_root, args.split, os.path.basename(f)), args.alpha, args.filter_alpha, past, depth,
             window_ms, radius, k_near, k_evid, args.band) for f in files]
    t0 = time.time()
    parts = []
    with Pool(max(1, args.workers)) as pool:
        for ri, res in enumerate(pool.imap(one_recording, jobs)):
            parts.append(res)
            print("%s %3d/%d %s: %d pending of %d events (%.0f s)" % (args.split, ri + 1, len(files),
                  os.path.basename(files[ri]), res[0]["idx"].size, res[6], time.time() - t0), flush=True)
    out = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    out["rec"] = np.concatenate([np.full(p[0]["idx"].size, i) for i, p in enumerate(parts)])
    out["tok"] = np.concatenate([p[1] for p in parts])
    out["tok_valid"] = np.concatenate([p[2] for p in parts])
    out["nall"] = np.concatenate([p[3] for p in parts])
    out["nevid"] = np.concatenate([p[4] for p in parts])
    n_events = sum(p[6] for p in parts)
    out.update(names=np.array([os.path.basename(f) for f in files]), log_w=np.float32(parts[0][5]),
               alpha=np.float64(args.alpha), filter_alpha=np.float64(args.filter_alpha), D=np.int64(depth),
               past=np.int64(past), window_ms=np.int64(window_ms), offsets=np.arange(-past, depth + 1),
               radius=np.float64(radius), k_near=np.int64(k_near), k_evid=np.int64(k_evid),
               n_events=np.int64(n_events))
    tau0 = _logit(THRESHOLD) - float(parts[0][5])
    b_lo, b_hi = band_bounds(tau0, args.alpha, args.band)
    out.update(band=np.array(args.band), band_lo=np.float64(b_lo), band_hi=np.float64(b_hi))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    print("EVIDENCE FINISHED %s: %d pending of %d events (%.2f%%), window %d ms, offsets %d..%d, radius %.0f px, "
          "K %d + %d per window, valid tokens per event-window %.2f"
          % (args.out, out["idx"].size, n_events, 100.0 * out["idx"].size / max(n_events, 1), window_ms, -past, depth,
             args.radius, k_near, k_evid, out["tok_valid"].sum(2).mean()), flush=True)


if __name__ == "__main__":
    main()
