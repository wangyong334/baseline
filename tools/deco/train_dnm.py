"""DeCo step 3 (final design, 11-06): decision neurons on a frozen backbone -- train on train, pick the epoch on val,
report val / test, export trajectories for tools/final_analysis.py.

Model (PAPER-NOTES 14.2; DECO_PROBE_LOG 4.x):
    h(0) = MLP(z0 / 5, polarity), Lambda(0) = z0 (the backbone's calibrated arrival logit).
    Step d = 0 .. D. Step 0 runs at arrival on the tokens of windows <= s (the past is re-read); step d >= 1 adds the
    tokens of window s + d. Each step: one-query multi-head cross-attention from h + age(d) to every arrived token,
    dh = attention + FFN, Lambda(d) = Lambda(d-1) + delta(h + dh), h = h + dh. A learned empty token is attended only
    when no token has arrived. No membrane taps, no attribution gate, no monotonic penalty (all removed by the probes).
Loss: mean BCE(Lambda(d), y) over the available steps of the pending events.
Options from the 11-07 local probes (DECO_PROBE_LOG section 5; defaults reproduce the 11-06 design exactly):
    --mil L     + L * target-frame "at least one" loss: for each group of pending events of the same official 50 ms
                frame and target that no non-pending event already detects, -log(1 - prod(1 - sigmoid(Lambda_i)))
                (multiple-instance noisy-OR; raises Pd / ACC, lowers IoU; 0.3 was the knee)
    --norm data token / arrival features as z-scores from training statistics instead of the hand scales
Design v10 (--design v10, 11-07; DECO_PROBE_LOG 7.5-7.9), adopted after the local pre-checks:
    evidence without a radius cap (tools/deco/evidence.py --radius 0), 6 token features (dx, dy, dt, window offset,
    polarity, neighbour z0) z-scored with training statistics, no count features or band flags; attention score of a
    token for head h = q.k / sqrt(d) + b_h - |dx, dy|^2 / (2 sigma_h^2) (learned receptive field, sigma_h starts at the
    RMS token distance of the training set; --rf-per head_offset: one per head and window offset); the empty key is
    always attendable (attention sink), and the evidence mass -log(attention weight of the empty key) per head enters
    the update through a zero-initialised linear layer. Commit timing: tools/deco/fit_head.py (change-risk head).
Commit (evaluation): a pending event commits at the first step where Lambda leaves tau +- logit(1 - commit_alpha),
else at its last available step; non-pending events keep the arrival decision.
Exports traj_<split>/<recording>.npz (column d = calibrated logit at lag d): non-pending events keep z0 at every lag,
pending ones take Lambda(d); NaN past the recording end.
Large evidence (11-07; EV-Flying has ~8.4M pending training events, ~30 GiB of tokens): when the per-token arrays of
the three splits exceed --gpu-evidence-gb, they stay memory-mapped from the evidence files in host memory and only
the rows of each batch are copied to the GPU. The values are the same, so the results are identical.
Usage: python tools/deco/train_dnm.py --evidence <prefix> --z0-root <dir with z0_<split>> --data-root <root> --out <dir>
"""
import argparse
import json
import math
import os
import struct
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THRESHOLD = 0.9
N_FEAT = 11
N_FEAT_V10 = 6


def _logit(p):
    return math.log(p / (1.0 - p))


def bounds(tau, alpha):
    half = _logit(1.0 - float(alpha))
    return tau - half, tau + half


HOST_KEYS = ("tok", "tok_valid", "nall", "nevid")              # per-token / per-window evidence arrays


def npz_memmap(path, key):
    """Read-only memmap of array `key` inside an .npz written by np.savez (members stored uncompressed); None when the
    member is compressed or has an unknown .npy header version."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + ".npy")
    if info.compress_type != zipfile.ZIP_STORED:
        return None
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        local = fh.read(30)                                      # zip local file header
        if local[:4] != b"PK\x03\x04":
            return None
        n_name, n_extra = struct.unpack("<HH", local[26:30])
        fh.seek(info.header_offset + 30 + n_name + n_extra)
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
        else:
            return None
        offset = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="F" if fortran else "C")


def evidence_bytes(path):
    """Uncompressed size of the per-token arrays of one evidence file."""
    with zipfile.ZipFile(path) as zf:
        return sum(zf.getinfo(k + ".npy").file_size for k in HOST_KEYS)


def take(a, idx):
    """Rows idx of an evidence array: a tensor (on the GPU), or a host array (memmap) whose rows are copied to the
    device of idx batch by batch."""
    if torch.is_tensor(a):
        return a[idx]
    return torch.from_numpy(np.ascontiguousarray(a[idx.cpu().numpy()])).to(idx.device)


class DecisionNeurons(nn.Module):
    def __init__(self, dim=64, heads=4, dmax=40):
        super(DecisionNeurons, self).__init__()
        self.dim, self.heads, self.hd, self.dmax = dim, heads, dim // heads, dmax
        self.h0 = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.tok = nn.Sequential(nn.Linear(N_FEAT, dim), nn.GELU(), nn.Linear(dim, dim))
        self.empty = nn.Parameter(torch.randn(dim) * 0.02)
        self.age = nn.Embedding(dmax + 1, dim)
        self.ln_q, self.ln_k, self.ln_f = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(dim, dim) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.delta = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        with torch.no_grad():                                        # Lambda starts exactly at z0
            self.delta[-1].weight.zero_()
            self.delta[-1].bias.zero_()

    def forward(self, feat, ok, dw, valid, z_in, z0, p, D):
        if D > self.dmax:
            raise ValueError("D = %d exceeds the age table (%d)" % (D, self.dmax))
        B, H, hd = z0.shape[0], self.heads, self.hd
        h = self.h0(torch.stack([z_in, p], 1))
        lam = z0.clone()
        kt = self.ln_k(self.tok(feat))
        Kt, Vt = self.wk(kt).view(B, -1, H, hd), self.wv(kt).view(B, -1, H, hd)
        ke = self.ln_k(self.empty)
        Ke, Ve = self.wk(ke).view(1, H, hd), self.wv(ke).view(1, H, hd)
        lams = []
        for d in range(D + 1):
            q = self.wq(self.ln_q(h + self.age.weight[d])).view(B, H, hd)
            arrived = ok & (dw <= d)
            empty = ~arrived.any(1, keepdim=True)
            S = torch.cat([torch.einsum("bhd,bthd->bht", q, Kt), (q * Ke).sum(-1)[..., None]], -1) / math.sqrt(hd)
            W = torch.softmax(S.masked_fill(~torch.cat([arrived, empty], 1)[:, None, :], float("-inf")), -1)
            out = torch.einsum("bht,bthd->bhd", W[:, :, :-1], Vt) + W[:, :, -1:] * Ve
            dh = self.wo(out.reshape(B, self.dim))
            dh = dh + self.ff(self.ln_f(dh))
            m = valid[:, d].float()
            lam = lam + m * self.delta(h + dh)[:, 0]
            h = h + m[:, None] * dh
            lams.append(lam)
        return torch.stack(lams, 1)


class DecisionNeuronsV10(nn.Module):
    """V10 decision neurons: learned receptive field per head (bias b_h, scale sigma_h), always-on empty key, evidence
    mass input. Same recurrence as DecisionNeurons otherwise."""

    def __init__(self, n_off, K, sigma0, dim=64, heads=4, dmax=40, per="head"):
        super(DecisionNeuronsV10, self).__init__()
        self.dim, self.heads, self.hd, self.dmax = dim, heads, dim // heads, dmax
        self.h0 = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.tok = nn.Sequential(nn.Linear(N_FEAT_V10, dim), nn.GELU(), nn.Linear(dim, dim))
        self.empty = nn.Parameter(torch.randn(dim) * 0.02)
        self.age = nn.Embedding(dmax + 1, dim)
        self.ln_q, self.ln_k, self.ln_f = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(dim, dim) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.delta = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        n_rf = n_off if per == "head_offset" else 1
        self.log_sigma = nn.Parameter(torch.full((heads, n_rf), math.log(sigma0)))
        self.rf_bias = nn.Parameter(torch.zeros(heads, n_rf))
        self.register_buffer("off_idx", torch.arange(n_off).repeat_interleave(K) if n_rf > 1
                             else torch.zeros(n_off * K, dtype=torch.long))
        self.wm = nn.Linear(heads, dim)
        with torch.no_grad():                                        # Lambda starts exactly at z0; mass starts unused
            self.delta[-1].weight.zero_()
            self.delta[-1].bias.zero_()
            self.wm.weight.zero_()
            self.wm.bias.zero_()

    def forward(self, feat, ok, dw, valid, z_in, z0, p, D, dist2):
        if D > self.dmax:
            raise ValueError("D = %d exceeds the age table (%d)" % (D, self.dmax))
        B, H, hd = z0.shape[0], self.heads, self.hd
        h = self.h0(torch.stack([z_in, p], 1))
        lam = z0.clone()
        kt = self.ln_k(self.tok(feat))
        Kt, Vt = self.wk(kt).view(B, -1, H, hd), self.wv(kt).view(B, -1, H, hd)
        ke = self.ln_k(self.empty)
        Ke, Ve = self.wk(ke).view(1, H, hd), self.wv(ke).view(1, H, hd)
        sig2 = torch.exp(2.0 * self.log_sigma)[:, self.off_idx]                     # [H, T]
        prior = self.rf_bias[:, self.off_idx][None] - dist2[:, None, :] / (2.0 * sig2[None])   # [B, H, T]
        lams = []
        for d in range(D + 1):
            q = self.wq(self.ln_q(h + self.age.weight[d])).view(B, H, hd)
            arrived = ok & (dw <= d)
            St = torch.einsum("bhd,bthd->bht", q, Kt) / math.sqrt(hd) + prior
            s0 = (q * Ke).sum(-1) / math.sqrt(hd)                                   # [B, H] empty-key score
            S = torch.cat([St.masked_fill(~arrived[:, None, :], float("-inf")), s0[..., None]], -1)
            lse = torch.logsumexp(S, -1)
            W = torch.exp(S - lse[..., None])
            out = torch.einsum("bht,bthd->bhd", W[:, :, :-1], Vt) + W[:, :, -1:] * Ve
            dh = self.wo(out.reshape(B, self.dim))
            dh = dh + self.wm(lse - s0)                                              # evidence mass per head
            dh = dh + self.ff(self.ln_f(dh))
            m = valid[:, d].float()
            lam = lam + m * self.delta(h + dh)[:, 0]
            h = h + m[:, None] * dh
            lams.append(lam)
        return torch.stack(lams, 1)


class FeaturizerV10(object):
    """V10 inputs: the 6 raw token features z-scored with the training statistics of the valid tokens, the arrival logit
    z-scored likewise, and the squared pixel distance of every token for the receptive field."""

    def fit(self, g, n, chunk=4096):
        s1 = torch.zeros(N_FEAT_V10, dtype=torch.float64, device=g["z0"].device)
        s2, cnt, d2 = torch.zeros_like(s1), 0.0, 0.0
        for i in range(0, n, chunk):
            raw, ok, _, dist2 = self.raw(g, torch.arange(i, min(i + chunk, n), device=g["z0"].device))
            x = raw[ok].double()
            s1 += x.sum(0)
            s2 += (x * x).sum(0)
            cnt += x.shape[0]
            d2 += float(dist2[ok].double().sum())
        mu = s1 / cnt
        self.mu, self.sd = mu.float(), (s2 / cnt - mu * mu).clamp(min=1e-8).sqrt().float()
        self.z_mu, self.z_sd = float(g["z0"].mean()), float(g["z0"].std())
        return math.sqrt(d2 / cnt)                                   # RMS token distance: start of every sigma

    def raw(self, g, idx):
        tok = take(g["tok"], idx).float()                            # [B, n_off, K, 6]
        B = tok.shape[0]
        tok = tok.reshape(B, -1, 6)
        dw = tok[..., 3]
        feat = torch.stack([tok[..., 0], tok[..., 1], tok[..., 2], dw * g["win_s"], tok[..., 4], tok[..., 5]], -1)
        return feat, take(g["ok"], idx).reshape(B, -1), dw, tok[..., 0] ** 2 + tok[..., 1] ** 2

    def batch(self, g, idx):
        feat, ok, dw, dist2 = self.raw(g, idx)
        z0 = g["z0"][idx]
        return ((feat - self.mu) / self.sd, ok, dw, g["valid"][idx], (z0 - self.z_mu) / self.z_sd, z0, g["p"][idx],
                g["y"][idx], dist2)


def load_evidence(path, dev, feat_lo, feat_hi, host=False):
    """Arrays of one evidence split. host=True: the per-token arrays (HOST_KEYS) stay memory-mapped in host memory and
    take() copies the rows of each batch to the GPU; everything else goes to dev. Same values either way."""
    with np.load(path) as f:
        m = {k: f[k] for k in f.files if not (host and k in HOST_KEYS)}
    if host:
        for k in HOST_KEYS:
            mm = npz_memmap(path, k)
            if mm is None:                                       # compressed member: read it into host memory
                with np.load(path) as f:
                    mm = f[k]
            m[k] = mm
    D = int(m["D"])
    valid = np.arange(D + 1)[None, :] <= m["F"][:, None]
    put = (lambda a: a) if host else (lambda a: torch.as_tensor(np.ascontiguousarray(a)).to(dev))   # noqa: E731
    g = {"tok": put(m["tok"]),
         "ok": put(m["tok_valid"]),
         "nall": put(m["nall"]),
         "nevid": put(m["nevid"]),
         "valid": torch.as_tensor(valid).to(dev),
         "z0": torch.as_tensor(m["z0"].astype(np.float32)).to(dev),
         "p": torch.as_tensor(m["p"].astype(np.float32)).to(dev),
         "y": torch.as_tensor(m["label"].astype(np.float32)).to(dev),
         "k_near": int(m["k_near"]), "win_s": float(m["window_ms"]) / 1000.0, "lo": feat_lo, "hi": feat_hi,
         "mu": None, "sd": None, "z_mu": 0.0, "z_sd": 5.0}
    return m, g, D


def batch(g, idx, raw=False):
    tok = take(g["tok"], idx).float()                            # [B, n_off, K, 6]
    B, n_off, K = tok.shape[0], tok.shape[1], tok.shape[2]
    tok = tok.reshape(B, -1, 6)
    dw = tok[..., 3]
    z = tok[..., 5]
    rep = lambda a: take(a, idx).float()[:, :, None].expand(B, n_off, K).reshape(B, -1)   # noqa: E731
    flag = (torch.arange(K, device=tok.device) >= g["k_near"]).float()[None, None].expand(B, n_off, K).reshape(B, -1)
    ok = take(g["ok"], idx).reshape(B, -1)
    if raw or g["mu"] is not None:                               # unscaled features (z-scored below)
        feat = torch.stack([tok[..., 0], tok[..., 1], tok[..., 2], dw * g["win_s"], tok[..., 4], z,
                            (z >= g["hi"]).float(), (z <= g["lo"]).float(), torch.log1p(rep(g["nall"])), flag,
                            torch.log1p(rep(g["nevid"]))], -1)
        if raw:
            return feat, ok
        feat = (feat - g["mu"]) / g["sd"]
    else:
        feat = torch.stack([tok[..., 0] / 16.0, tok[..., 1] / 16.0, tok[..., 2] / 500.0, dw * g["win_s"], tok[..., 4],
                            z / 5.0, (z >= g["hi"]).float(), (z <= g["lo"]).float(), torch.log1p(rep(g["nall"])) / 5.0,
                            flag, torch.log1p(rep(g["nevid"])) / 5.0], -1)
    z0 = g["z0"][idx]
    return feat, ok, dw, g["valid"][idx], (z0 - g["z_mu"]) / g["z_sd"], z0, g["p"][idx], g["y"][idx]


def fit_norm(g, n, chunk=4096):
    """Training-set mean / std of the unscaled token features (valid tokens) and of the arrival logit."""
    s1 = torch.zeros(N_FEAT, dtype=torch.float64, device=g["z0"].device)
    s2, cnt = torch.zeros_like(s1), 0
    for i in range(0, n, chunk):
        feat, ok = batch(g, torch.arange(i, min(i + chunk, n), device=g["z0"].device), raw=True)
        x = feat[ok].double()
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
        cnt += x.shape[0]
    mu = s1 / cnt
    return mu.float(), (s2 / cnt - mu * mu).clamp(min=1e-8).sqrt().float(), float(g["z0"].mean()), float(g["z0"].std())


def commit_lags(traj, F_, lo, hi):
    L = F_.copy()
    done = np.zeros(len(F_), bool)
    for d in range(traj.shape[1]):
        hit = ~done & (d <= F_) & ((traj[:, d] >= hi) | (traj[:, d] <= lo))
        L[hit] = d
        done |= hit
    return L


def nonpending_counts(z0_dir, names, lo, hi, tau):
    c = np.zeros(3, np.int64)
    for name in names:
        with np.load(os.path.join(z0_dir, name)) as tr:
            z, y = np.asarray(tr["traj"][:, 0]), np.asarray(tr["labels"]) > 0.5
        keep = ~((z > lo) & (z < hi))
        pred, pos = (z >= tau)[keep], y[keep]
        c += [int((pred & pos).sum()), int((pred & ~pos).sum()), int((~pred & pos).sum())]
    return c


def evaluate(net, fwd, m, g, D, c_np, tau, clo, chi):
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(m["idx"]), 1024):
            b = torch.arange(i, min(i + 1024, len(m["idx"])), device=g["z0"].device)
            outs.append(fwd(g, b)[0].cpu().numpy())
    lt = np.concatenate(outs) if outs else np.zeros((0, D + 1), np.float32)
    F_ = np.minimum(m["F"], D)
    y = m["label"].astype(bool)

    def iou(L):
        pred = lt[np.arange(len(L)), L] >= tau
        c = c_np + [(pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()]
        return float(c[0]) / float(max(c.sum(), 1)), c

    Lc = commit_lags(lt, F_, clo, chi)
    ic, cc = iou(Lc)
    i0, c0 = iou(np.zeros_like(F_))
    i_f, cf = iou(F_)
    return lt, Lc, {"crossing_iou": ic, "arrival_iou": i0, "fixed_iou": i_f, "crossing_counts": cc.tolist(),
                    "arrival_counts": c0.tolist(), "fixed_counts": cf.tolist(),
                    "mean_wait_pending": float(Lc.mean()) if len(Lc) else 0.0, "pending": int(len(Lc))}


def export(lt, m, z0_dir, out_dir, data_root, split, D):
    os.makedirs(out_dir, exist_ok=True)
    for ri, name in enumerate(list(m["names"])):
        with np.load(os.path.join(z0_dir, name)) as tr:
            z, labels, log_w = np.asarray(tr["traj"][:, 0], np.float32), tr["labels"], tr["log_w"]
            wms = int(tr["window_ms"])
        with np.load(os.path.join(data_root, split, name)) as d:
            t = np.asarray(d["ev_loc"])[:, 2]
        win = (t // wms).astype(np.int64)
        avail = np.minimum(D, int(win.max()) - win)
        traj = np.repeat(z[:, None], D + 1, axis=1)
        sel = np.nonzero(m["rec"] == ri)[0]
        traj[m["idx"][sel]] = lt[sel]
        traj[np.arange(D + 1)[None, :] > avail[:, None]] = np.nan
        np.savez(os.path.join(out_dir, name), traj=traj, log_w=log_w, labels=labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evidence", required=True, help="prefix: <prefix>_train.npz / _val.npz / _test.npz")
    ap.add_argument("--z0-root", required=True, help="dir holding z0_train / z0_val / z0_test")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--commit-alpha", type=float, default=0.01)
    ap.add_argument("--mil", type=float, default=0.0, help="weight of the target-frame 'at least one' loss (0 = off)")
    ap.add_argument("--norm", choices=("hand", "data"), default="hand", help="feature scaling (v9)")
    ap.add_argument("--design", choices=("v9", "v10"), default="v9",
                    help="v9 = 11-06 design (default, bit-exact); v10 = learned receptive field (--radius 0 evidence)")
    ap.add_argument("--rf-per", choices=("head", "head_offset"), default="head",
                    help="v10: one receptive field per head (adopted) or per head and window offset")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-evidence-gb", type=float, default=8.0,
                    help="per-token evidence of the three splits above this size (GiB) stays memory-mapped in host "
                         "memory and is copied to the GPU per batch (EV-Flying); results are identical")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    with np.load(args.evidence + "_val.npz", allow_pickle=False) as f:      # scalars only: skip the token arrays
        ref = {k: f[k] for k in f.files if k not in HOST_KEYS}
    tau = _logit(THRESHOLD) - float(ref["log_w"])
    flo, fhi = bounds(tau, float(ref["filter_alpha"]))
    plo, phi = bounds(tau, float(ref["alpha"]))
    clo, chi = bounds(tau, args.commit_alpha)
    paths = {split: args.evidence + "_%s.npz" % split for split in ("train", "val", "test")}
    tok_gb = sum(evidence_bytes(p) for p in paths.values()) / 2.0 ** 30
    host = tok_gb > args.gpu_evidence_gb
    print("evidence: %.2f GiB of token arrays -> %s" % (
        tok_gb, "memory-mapped in host memory, copied to the GPU per batch" if host else "on %s" % dev), flush=True)
    data = {}
    for split in ("train", "val", "test"):
        m, g, D = load_evidence(paths[split], dev, flo, fhi, host)
        c_np = nonpending_counts(os.path.join(args.z0_root, "z0_" + split), list(m["names"]), plo, phi, tau) \
            if split != "train" else None
        data[split] = (m, g, c_np)
        print("%s: %d pending events, D = %d, past %d, window %d ms, radius %.0f px, K %d + %d per window"
              % (split, len(m["idx"]), D, int(m["past"]), int(m["window_ms"]), float(m["radius"]), int(m["k_near"]),
                 int(m["k_evid"])), flush=True)
    m_tr, g_tr, _ = data["train"]
    n = len(m_tr["idx"])
    if args.design == "v10" and (args.mil > 0 or args.norm != "hand"):
        raise ValueError("--mil and --norm belong to the v9 design (v10 always z-scores its inputs)")
    if args.design == "v10" and np.isfinite(float(m_tr["radius"])):
        print("warning: v10 expects evidence without a radius cap (evidence.py --radius 0); got %.0f px"
              % float(m_tr["radius"]), flush=True)
    if args.norm == "data":                    # the training statistics serve all splits
        stats = fit_norm(g_tr, n)
        for split in data:
            data[split][1].update(mu=stats[0], sd=stats[1], z_mu=stats[2], z_sd=stats[3])
    mil = None
    if args.mil > 0:                           # at-risk target-frame groups of the training pending events
        from mil_groups import group_of
        gid = group_of(m_tr, os.path.join(args.z0_root, "z0_train"), args.data_root, "train")
        members = np.nonzero(gid >= 0)[0]
        members = members[np.argsort(gid[members], kind="stable")]
        n_groups = int(gid.max()) + 1
        mil = (torch.as_tensor(members, device=dev), np.searchsorted(gid[members], np.arange(n_groups + 1)), n_groups)
        print("MIL: %d at-risk target-frame groups, %d pending events in them" % (n_groups, members.size), flush=True)
    extra = {}
    if args.design == "v10":
        fz = FeaturizerV10()
        sigma0 = fz.fit(g_tr, n)
        net = DecisionNeuronsV10(int(m_tr["tok"].shape[1]), int(m_tr["tok"].shape[2]), sigma0,
                                 per=args.rf_per).to(dev)
        extra["sigma0"] = sigma0
        print("v10: receptive field starts at sigma0 = %.1f px (RMS token distance on train), one per %s"
              % (sigma0, args.rf_per.replace("_", " and ")), flush=True)

        def fwd(g, idx):
            feat, ok, dw, valid, z_in, z0, p, y, dist2 = fz.batch(g, idx)
            return net(feat, ok, dw, valid, z_in, z0, p, D, dist2), valid, y
    else:
        net = DecisionNeurons().to(dev)

        def fwd(g, idx):
            feat, ok, dw, valid, z_in, z0, p, y = batch(g, idx)
            return net(feat, ok, dw, valid, z_in, z0, p, D), valid, y
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    t0 = time.time()
    best, best_state, best_ep, hist = -1.0, None, -1, []
    for ep in range(args.epochs):
        net.train()
        perm = torch.as_tensor(np.random.permutation(n), dtype=torch.long, device=dev)
        for i in range(0, n, args.batch):
            lams, valid, y = fwd(g_tr, perm[i:i + args.batch])
            vm = valid.float()
            ell = F.binary_cross_entropy_with_logits(lams, y[:, None].expand_as(lams), reduction="none")
            loss = (ell * vm).sum() / vm.sum().clamp(min=1)
            if mil is not None:                # noisy-OR per group: P(at least one of its events is target)
                members_t, starts, n_groups = mil
                pick, parts, total = [], [], 0          # random groups until batch / 2 events; big groups -> 64 members
                for k in np.random.permutation(n_groups):
                    r = np.arange(starts[k], starts[k + 1])
                    if r.size > 64:
                        r = np.random.choice(r, 64, replace=False)
                    if pick and total + r.size > args.batch // 2:
                        break
                    pick.append(k)
                    parts.append(r)
                    total += r.size
                rows_t = torch.as_tensor(np.concatenate(parts), device=dev)
                lg, valid, _ = fwd(g_tr, members_t[rows_t])
                sizes = [len(r) for r in parts]               # groups are contiguous: deterministic segment sums
                log_neg = F.logsigmoid(-lg) * valid.float()
                s = torch.stack([x.sum(0) for x in torch.split(log_neg, sizes)])
                any_valid = torch.stack([x.sum(0) for x in torch.split(valid.float(), sizes)]) > 0
                s = s.clamp(max=-1e-6)
                log_p = torch.where(s < -0.6931, torch.log1p(-torch.exp(s)), torch.log(-torch.expm1(s)))
                loss = loss + args.mil * (-(log_p * any_valid.float()).sum() / any_valid.float().sum().clamp(min=1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        m_va, g_va, c_va = data["val"]
        _, _, r = evaluate(net, fwd, m_va, g_va, D, c_va, tau, clo, chi)
        hist.append(round(r["crossing_iou"], 4))
        print("ep %d val crossing %.4f arrival %.4f fixed %.4f wait %.2f (%.0f s)" % (
            ep, r["crossing_iou"], r["arrival_iou"], r["fixed_iou"], r["mean_wait_pending"], time.time() - t0), flush=True)
        if r["crossing_iou"] > best:
            best, best_ep = r["crossing_iou"], ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    torch.save({"model": best_state, "args": vars(args)}, os.path.join(args.out, "decision_neurons.pt"))
    evidence = {k: (ref[k].item() if np.ndim(ref[k]) == 0 else None) for k in
                ("alpha", "filter_alpha", "D", "past", "window_ms", "radius", "k_near", "k_evid")}
    summary = {"args": vars(args), "evidence": evidence, "best_epoch": best_ep, "val_history": hist,
               "seconds": round(time.time() - t0, 1), "host_evidence": bool(host)}
    if args.design == "v10":                   # the learned receptive field (px) and head biases
        extra.update(sigma=torch.exp(net.log_sigma).detach().cpu().tolist(),
                     rf_bias=net.rf_bias.detach().cpu().tolist())
        summary["v10"] = extra
    for split in ("val", "test"):
        m, g, c_np = data[split]
        lt, Lc, r = evaluate(net, fwd, m, g, D, c_np, tau, clo, chi)
        summary[split] = r
        np.savez(os.path.join(args.out, "raw_%s.npz" % split), lam_traj=lt.astype(np.float32), lags=Lc)
        export(lt, m, os.path.join(args.z0_root, "z0_" + split), os.path.join(args.out, "traj_" + split),
               args.data_root, split, D)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("DNM TRAIN FINISHED best ep %d | val crossing %.4f arrival %.4f fixed %.4f wait %.2f | test crossing %.4f "
          "arrival %.4f fixed %.4f wait %.2f" % (
              best_ep, summary["val"]["crossing_iou"], summary["val"]["arrival_iou"], summary["val"]["fixed_iou"],
              summary["val"]["mean_wait_pending"], summary["test"]["crossing_iou"], summary["test"]["arrival_iou"],
              summary["test"]["fixed_iou"], summary["test"]["mean_wait_pending"]), flush=True)


if __name__ == "__main__":
    main()
