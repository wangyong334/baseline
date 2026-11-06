"""DeCo step 3: decision neurons on a frozen backbone -- train on train, pick the epoch on val, report val / test, export.

Model (the 11-06 probe's main design, FastProbe without membrane taps):
    h(0) = MLP(z0 / 5, polarity), Lambda(0) = z0 (backbone's calibrated arrival logit)
    step d (d = 1..D, and d = 0 with --step0): one-query multi-head cross-attention from h(d-1) + age(d) to every token
    of the hourglass that has arrived (window offset <= d) plus a null token; dh = attention + FFN;
    m = 1 - null weight (1 without --null); Lambda(d) = Lambda(d-1) + m * delta(h + dh); h(d) = h(d-1) + m * dh.
Loss: mean BCE(Lambda(d), y) + lam * mean [BCE_d - BCE_(d-1)]_+ over the available steps.
Commit (evaluation): pending events commit at the first step >= 1 (>= 0 with --step0) where Lambda leaves the band
tau +- logit(1 - commit_alpha), else at their last available step; non-pending events keep the arrival decision.
Exports traj_<split>/<recording>.npz in the module-three format (column d = calibrated logit at lag d) for
tools/final_analysis.py: non-pending events keep z0 at every lag, pending ones take Lambda(d); NaN past the recording end.
Usage: python tools/deco/train_probe.py --tokens <prefix> --z0-root <dir with z0_<split>> --out <dir> [--seed 0]
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THRESHOLD = 0.9
N_FEAT = 11


def _logit(p):
    return math.log(p / (1.0 - p))


def bounds(tau, alpha):
    half = _logit(1.0 - float(alpha))
    return tau - half, tau + half


class DecisionNeurons(nn.Module):
    def __init__(self, dim=64, heads=4, dmax=40, use_null=True, step0=False):
        super(DecisionNeurons, self).__init__()
        self.dim, self.heads, self.hd = dim, heads, dim // heads
        self.use_null, self.step0 = use_null, step0
        self.h0 = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.tok = nn.Sequential(nn.Linear(N_FEAT, dim), nn.GELU(), nn.Linear(dim, dim))
        self.null = nn.Parameter(torch.randn(dim) * 0.02)
        self.age = nn.Embedding(dmax + 1, dim)
        self.ln_q, self.ln_k, self.ln_f = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(dim, dim) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.delta = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        with torch.no_grad():
            self.delta[-1].weight.zero_()
            self.delta[-1].bias.zero_()

    def forward(self, feat, ok, dw, valid, z0, p, D):
        B, H, hd = z0.shape[0], self.heads, self.hd
        h = self.h0(torch.stack([z0 / 5.0, p], 1))
        lam = z0.clone()
        kt = self.ln_k(self.tok(feat))
        Kt, Vt = self.wk(kt).view(B, -1, H, hd), self.wv(kt).view(B, -1, H, hd)
        kn = self.ln_k(self.null)
        Kn, Vn = self.wk(kn).view(1, H, hd).expand(B, H, hd), self.wv(kn).view(1, 1, H, hd).expand(B, 1, H, hd)
        lams, ms = [], []
        for d in range(0 if self.step0 else 1, D + 1):
            q = self.wq(self.ln_q(h + self.age.weight[d])).view(B, H, hd)
            mt = ok & (dw <= d)
            mn = torch.ones(B, 1, dtype=torch.bool, device=z0.device) if self.use_null else ~mt.any(1, keepdim=True)
            S = torch.cat([torch.einsum("bhd,bthd->bht", q, Kt), (q * Kn).sum(-1)[..., None]], -1) / math.sqrt(hd)
            W = torch.softmax(S.masked_fill(~torch.cat([mt, mn], 1)[:, None, :], float("-inf")), -1)
            out = torch.einsum("bht,bthd->bhd", W[:, :, :-1], Vt) + W[:, :, -1:] * Vn[:, 0]
            dh = self.wo(out.reshape(B, self.dim))
            dh = dh + self.ff(self.ln_f(dh))
            m = (1.0 - W[:, :, -1].mean(1)) if self.use_null else torch.ones(B, device=z0.device)
            m = m * valid[:, d].float()
            delta = self.delta(h + dh)[:, 0]
            h = h + m[:, None] * dh
            lam = lam + m * delta
            lams.append(lam)
            ms.append(m)
        return torch.stack(lams, 1), torch.stack(ms, 1)


def load_tokens(path, dev, feat_lo, feat_hi):
    m = dict(np.load(path))
    D = int(m["D"])
    offs = m["offsets"]
    sel = np.nonzero(offs <= D)[0]
    n, K = len(m["idx"]), m["tok"].shape[2]
    valid = np.arange(D + 1)[None, :] <= m["F"][:, None]
    g = {"tok": torch.as_tensor(np.ascontiguousarray(m["tok"][:, sel])).to(dev),
         "ok": torch.as_tensor(np.ascontiguousarray(m["tok_valid"][:, sel])).to(dev),
         "ncone": torch.as_tensor(np.ascontiguousarray(m["ncone"][:, sel])).to(dev),
         "nevid": torch.as_tensor(np.ascontiguousarray(m["nevid"][:, sel])).to(dev),
         "valid": torch.as_tensor(valid).to(dev),
         "z0": torch.as_tensor(m["z0"].astype(np.float32)).to(dev),
         "p": torch.as_tensor(m["p"].astype(np.float32)).to(dev),
         "y": torch.as_tensor(m["label"].astype(np.float32)).to(dev),
         "K": K, "k_near": int(m["k_near"]), "win_s": float(m["window_ms"]) / 1000.0, "lo": feat_lo, "hi": feat_hi}
    return m, g, D


def batch(g, idx):
    tok = g["tok"][idx].float()                                  # [B, n_off, K, 6]
    B, n_off, K = tok.shape[0], tok.shape[1], tok.shape[2]
    tok = tok.reshape(B, -1, 6)
    dw = tok[..., 3]
    z = tok[..., 5]
    rep = lambda a: a[idx].float()[:, :, None].expand(B, n_off, K).reshape(B, -1)   # noqa: E731
    flag = (torch.arange(K, device=tok.device) >= g["k_near"]).float()[None, None].expand(B, n_off, K).reshape(B, -1)
    feat = torch.stack([tok[..., 0] / 16.0, tok[..., 1] / 16.0, tok[..., 2] / 500.0, dw * g["win_s"], tok[..., 4],
                        z / 5.0, (z >= g["hi"]).float(), (z <= g["lo"]).float(), torch.log1p(rep(g["ncone"])) / 5.0,
                        flag, torch.log1p(rep(g["nevid"])) / 5.0], -1)
    ok = g["ok"][idx].reshape(B, -1)
    return feat, ok, dw, g["valid"][idx], g["z0"][idx], g["p"][idx], g["y"][idx]


def commit_lags(traj, F_, lo, hi, start):
    L = F_.copy()
    done = np.zeros(len(F_), bool)
    for d in range(start, traj.shape[1]):
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


def evaluate(net, m, g, D, args, c_np, tau, clo, chi):
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(m["idx"]), 1024):
            b = torch.arange(i, min(i + 1024, len(m["idx"])), device=g["z0"].device)
            feat, ok, dw, valid, z0, p, y = batch(g, b)
            lams, _ = net(feat, ok, dw, valid, z0, p, D)
            outs.append((lams if lams.shape[1] == D + 1 else torch.cat([z0[:, None], lams], 1)).cpu().numpy())
    lt = np.concatenate(outs)
    F_ = np.minimum(m["F"], D)
    y = m["label"].astype(bool)

    def iou(L):
        pred = lt[np.arange(len(L)), L] >= tau
        c = c_np + [(pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()]
        return float(c[0]) / float(c.sum()), c

    Lc = commit_lags(lt, F_, clo, chi, 0 if args.step0 else 1)
    ic, cc = iou(Lc)
    i_f, cf = iou(F_)
    return lt, Lc, {"crossing_iou": ic, "fixed_iou": i_f, "crossing_counts": cc.tolist(), "fixed_counts": cf.tolist(),
                    "mean_wait_pending": float(Lc.mean()), "pending": int(len(Lc))}


def export(lt, m, z0_dir, out_dir, data_root, split, D, step0):
    os.makedirs(out_dir, exist_ok=True)
    names = list(m["names"])
    for ri, name in enumerate(names):
        with np.load(os.path.join(z0_dir, name)) as tr:
            z, labels, log_w = np.asarray(tr["traj"][:, 0], np.float32), tr["labels"], tr["log_w"]
            wms = int(tr["window_ms"])
        with np.load(os.path.join(data_root, split, name)) as d:
            t = np.asarray(d["ev_loc"])[:, 2]
        win = (t // wms).astype(np.int64)
        avail = np.minimum(D, int(win.max()) - win)
        traj = np.repeat(z[:, None], D + 1, axis=1)
        sel = np.nonzero(m["rec"] == ri)[0]
        rows = m["idx"][sel]
        traj[rows, 1:] = lt[sel, 1:]
        if step0:
            traj[rows, 0] = lt[sel, 0]
        traj[np.arange(D + 1)[None, :] > avail[:, None]] = np.nan
        np.savez(os.path.join(out_dir, name), traj=traj, log_w=log_w, labels=labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokens", required=True, help="prefix: <prefix>_train.npz / _val.npz / _test.npz")
    ap.add_argument("--z0-root", required=True, help="dir holding z0_train / z0_val / z0_test")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam-tel", type=float, default=1.0)
    ap.add_argument("--null", type=int, default=1)
    ap.add_argument("--step0", type=int, default=0, help="1: also update at arrival from the past half (zero wait)")
    ap.add_argument("--commit-alpha", type=float, default=0.01)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    probe = dict(np.load(args.tokens + "_val.npz", allow_pickle=False))
    tau = _logit(THRESHOLD) - float(probe["log_w"])
    flo, fhi = bounds(tau, float(probe["filter_alpha"]))
    plo, phi = bounds(tau, float(probe["alpha"]))
    clo, chi = bounds(tau, args.commit_alpha)
    data = {}
    for split in ("train", "val", "test"):
        m, g, D = load_tokens(args.tokens + "_%s.npz" % split, dev, flo, fhi)
        c_np = nonpending_counts(os.path.join(args.z0_root, "z0_" + split), list(m["names"]), plo, phi, tau) \
            if split != "train" else None
        data[split] = (m, g, c_np)
        print("%s: %d pending events, D = %d, window %d ms" % (split, len(m["idx"]), D, int(m["window_ms"])), flush=True)
    net = DecisionNeurons(use_null=bool(args.null), step0=bool(args.step0)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    m_tr, g_tr, _ = data["train"]
    n = len(m_tr["idx"])
    t0 = time.time()
    best, best_state, best_ep, hist = -1.0, None, -1, []
    for ep in range(args.epochs):
        net.train()
        perm = torch.as_tensor(np.random.permutation(n), dtype=torch.long, device=dev)
        for i in range(0, n, args.batch):
            feat, ok, dw, valid, z0, p, y = batch(g_tr, perm[i:i + args.batch])
            lams, _ = net(feat, ok, dw, valid, z0, p, D)
            vm = valid[:, D + 1 - lams.shape[1]:].float()
            ell = F.binary_cross_entropy_with_logits(lams, y[:, None].expand_as(lams), reduction="none")
            prev = torch.cat([F.binary_cross_entropy_with_logits(z0, y, reduction="none")[:, None], ell[:, :-1]], 1)
            loss = ((ell * vm).sum() + args.lam_tel * (F.relu(ell - prev) * vm).sum()) / vm.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        m_va, g_va, c_va = data["val"]
        _, _, r = evaluate(net, m_va, g_va, D, args, c_va, tau, clo, chi)
        hist.append(round(r["crossing_iou"], 4))
        print("ep %d val crossing %.4f fixed %.4f wait %.2f (%.0f s)" % (ep, r["crossing_iou"], r["fixed_iou"],
              r["mean_wait_pending"], time.time() - t0), flush=True)
        if r["crossing_iou"] > best:
            best, best_ep = r["crossing_iou"], ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    torch.save({"model": best_state, "args": vars(args)}, os.path.join(args.out, "decision_neurons.pt"))
    summary = {"args": vars(args), "best_epoch": best_ep, "val_history": hist, "seconds": round(time.time() - t0, 1)}
    for split in ("val", "test"):
        m, g, c_np = data[split]
        lt, Lc, r = evaluate(net, m, g, D, args, c_np, tau, clo, chi)
        summary[split] = r
        np.savez(os.path.join(args.out, "raw_%s.npz" % split), lam_traj=lt.astype(np.float32), lags=Lc)
        export(lt, m, os.path.join(args.z0_root, "z0_" + split), os.path.join(args.out, "traj_" + split),
               args.data_root, split, D, bool(args.step0))
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("DECO PROBE FINISHED best ep %d | val crossing %.4f fixed %.4f wait %.2f | test crossing %.4f fixed %.4f "
          "wait %.2f" % (best_ep, summary["val"]["crossing_iou"], summary["val"]["fixed_iou"],
                         summary["val"]["mean_wait_pending"], summary["test"]["crossing_iou"],
                         summary["test"]["fixed_iou"], summary["test"]["mean_wait_pending"]), flush=True)


if __name__ == "__main__":
    main()
