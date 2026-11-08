"""Nested V11 (11-08): end-to-end joint training of the small network (BIR U-Net + report head) and the big network
(V11 spiking decision neurons), then export in the format of the staged pipeline (tools/deco/fit_range.py runs on it).

Start: the backbone checkpoint (V5) and a staged V11 run (tools/deco/train_dnm.py --design v11). With --epochs 0 the
export is the staged network itself (the report columns of the neurons start at zero): the consistency gate.
Training: the backbone's parallel-stream TBPTT (train_stream_v1.py trainer=streams) with chunks of --k windows. In every
chunk the small network runs first (logits, reports); the pending events of windows [past, k - 1 - D] whose whole
evidence span lies inside the same recording piece get a decision neuron (at most --max-pending per chunk), with
candidates found on the GPU (tools/deco/nested_v11.gpu_picks). Loss = BCE of the small network on every event (pos
weight as in its training) + its message-budget penalty + --lambda-big * stepwise BCE of the neurons. One update per
chunk, gradients clipped at the backbone's grad_clip. The epoch is chosen on val (pooled IoU of the fixed-band commit,
as train_dnm.py).
Export (best epoch): per split, z0_<split>/<recording>.npz (the new priors), evm_<split>_meta.npz (pending events,
band, names) and run/raw_<split>.npz (trajectories, mass, effective count) with candidates from
tools/deco/evidence.candidate_picks, i.e. exactly the staged evidence rule.
Usage: python tools/deco/train_nested.py --checkpoint <backbone .pt> --stage1 <staged v11 run dir>
       --stage1-evidence <evidence prefix> --data-root <root> --out <dir> [--epochs 4] [--k 32] [--streams 2]
Then:  python tools/deco/fit_range.py --run <out>/run --evidence <out>/evm --z0-root <out> --data-root <root>
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import evidence as EV  # noqa: E402
import nested_v11 as NV  # noqa: E402
import sdn_v11  # noqa: E402
import train_dnm as TD  # noqa: E402
import train_stream_v1 as T  # noqa: E402
from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from dataset.stream_source import make_window_source  # noqa: E402

THRESHOLD = 0.9


def logit(p):
    return math.log(p / (1.0 - p))


def stage1_parts(args, device):
    """Weights, options, featurizer statistics and evidence settings of the staged run."""
    ck = torch.load(os.path.join(args.stage1, "decision_neurons.pt"), map_location="cpu")
    s1 = ck["args"]
    if s1.get("design") != "v11":
        raise ValueError("the staged run must be a v11 run (got %s)" % s1.get("design"))
    with np.load(args.stage1_evidence + "_val.npz") as f:
        ev = {k: f[k] for k in f.files if k not in TD.HOST_KEYS}
    fz = ck.get("featurizer")
    if fz is None:                                   # older runs: recompute from the staged training evidence
        path = args.stage1_evidence + "_train.npz"
        tau = logit(THRESHOLD) - float(ev["log_w"])
        flo, fhi = TD.bounds(tau, float(ev["filter_alpha"]))
        m, g, _ = TD.load_evidence(path, device, flo, fhi, host=True)
        f = TD.FeaturizerV10()
        sigma0 = f.fit(g, len(m["idx"]))
        fz = {"mu": f.mu.cpu().tolist(), "sd": f.sd.cpu().tolist(), "z_mu": f.z_mu, "z_sd": f.z_sd, "sigma0": sigma0}
        del m, g
    summ = json.load(open(os.path.join(args.stage1, "summary.json"), encoding="utf-8"))
    sigma0 = float(fz.get("sigma0") or summ["v11"]["sigma0"])
    return ck["model"], s1, fz, sigma0, ev


def band_of(ev, tau):
    if "band_lo" in ev:
        return float(ev["band_lo"]), float(ev["band_hi"])
    return TD.bounds(tau, float(ev["alpha"]))


def build(args, device):
    ck = torch.load(args.checkpoint, map_location=device)
    cfg = dict(ck["config"])
    cfg["root"] = args.data_root
    cfg.update(execution="layer", input_device="gpu", tbptt_k=args.k, streams=args.streams, accumulate=1,
               stream_groups=1)
    q99 = np.asarray(ck["stats"]["q99"], dtype=np.float32)
    pos_weight = float(ck["stats"]["pos_weight"])
    small = T.build_net(cfg, device)
    small.load_state_dict(ck["model"])
    if not bool(small.gain_calibrated):
        raise RuntimeError("backbone gain not calibrated: %s" % args.checkpoint)
    state, s1, fz, sigma0, ev = stage1_parts(args, device)
    P, D = int(ev["past"]), int(ev["D"])
    K = int(ev["k_near"]) + int(ev["k_evid"])
    rho_init = tuple(s1.get("v11_rho_init", (0.5, 0.12)))
    big = sdn_v11.SpikingDecisionNeuronsV11(P + D + 1, K, sigma0, audit=s1.get("v11_audit", "reaudit"),
                                            rho=s1.get("v11_rho", "learn"), gate=s1.get("v11_gate", "sigmoid"),
                                            rho_init=rho_init, ctx_mass=s1.get("v11_ctx_mass", "on") == "on",
                                            n_extra=args.report_dim)
    big.load_stage1(state)
    report = NV.ReportHead(small, args.report_dim)
    model = NV.NestedDeCo(small, big, report, fz["mu"], fz["sd"], fz["z_mu"], fz["z_sd"], int(ev["window_ms"])).to(device)
    log_w = math.log(pos_weight)
    tau = logit(THRESHOLD) - log_w
    lo, hi = band_of(ev, tau)
    flo, _ = TD.bounds(tau, float(ev["filter_alpha"]))
    settings = {"P": P, "D": D, "k_near": int(ev["k_near"]), "k_evid": int(ev["k_evid"]), "window_ms": int(ev["window_ms"]),
                "alpha": float(ev["alpha"]), "filter_alpha": float(ev["filter_alpha"]), "band": (lo, hi), "flo": flo,
                "tau": tau, "log_w": log_w, "pos_weight": pos_weight}
    return model, cfg, q99, settings, ck


# ----------------------------------------------------------------------------------------------------------------------
# export (exact staged evidence rule)
# ----------------------------------------------------------------------------------------------------------------------


def _picks_job(job):
    loc, zc, pend, offsets, k_near, k_evid, flo, window_ms = job
    win = (loc[:, 2] // window_ms).astype(np.int64)
    n_win = int(win.max()) + 1
    pick, ok, _, _ = EV.candidate_picks(loc, zc, win, n_win, pend, offsets, k_near, k_evid, flo, np.inf)
    return pick, ok


def export_split(model, cfg, q99, st, split, device, out=None, limit=0, workers=8, batch=1024, commit_alpha=0.01,
                 z0_dir=None):
    """Priors and reports of every event, pending events, staged candidates, decision-neuron trajectories.
    Returns the summary (pooled IoU of arrival / fixed-band commit / deadline); with out, writes the export files."""
    model.eval()
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(cfg["root"], split, "*.npz")))
    if limit:
        names = names[:limit]
    P, D, wms = st["P"], st["D"], st["window_ms"]
    offsets = list(range(-P, D + 1))
    lo, hi = st["band"]
    tau, log_w = st["tau"], st["log_w"]
    dtype = next(model.small.parameters()).dtype
    recs, jobs = [], []
    for name in names:
        seq = EvUAVStream(cfg["root"], split, cfg, [name])[0]
        with np.load(os.path.join(cfg["root"], split, name)) as d:
            loc, norm = np.asarray(d["ev_loc"]), np.asarray(d["evs_norm"])
        n = loc.shape[0]
        z = np.full(n, np.nan, np.float32)
        rep = np.zeros((n, model.report.dim), np.float32)
        if z0_dir is not None:                       # consistency gate only: priors from a staged z0 dump, no reports
            with np.load(os.path.join(z0_dir, name)) as tr:
                z = np.asarray(tr["traj"][:, 0], np.float32)
        else:
            source = make_window_source(seq, cfg, q99, device, dtype)
            states = None
            with torch.no_grad():
                step = int(cfg.get("eval_chunk", 32))
                for a in range(0, seq.n_windows, step):
                    b = min(seq.n_windows, a + step)
                    x, events, _, idx = source.chunk(a, b)
                    logits, states, _ = model.small.forward_chunk(x, events, states)
                    r = model.report()
                    z[idx] = logits.float().cpu().numpy() - log_w
                    rep[idx] = r.float().cpu().numpy()
        pend = np.nonzero((z > lo) & (z < hi))[0]
        zc = np.clip(np.nan_to_num(z, nan=-20.0), -20.0, 20.0)
        recs.append({"name": name, "loc": loc, "norm": norm, "z": z, "rep": rep, "pend": pend})
        jobs.append((loc, zc, pend, offsets, st["k_near"], st["k_evid"], st["flo"], wms))
    if workers > 1:
        with Pool(min(workers, len(jobs))) as pool:
            picks = pool.map(_picks_job, jobs)
    else:
        picks = [_picks_job(j) for j in jobs]
    lam, mass, neff, meta = [], [], [], {k: [] for k in ("idx", "rec", "F", "z0", "label", "p", "s")}
    c_np = np.zeros(3, np.int64)
    c_prior = np.zeros(3, np.int64)                  # the small network alone (z0 >= tau on every event)
    for rc in recs:
        yy, pp = rc["norm"][:, 4] > 0.5, rc["z"] >= tau
        c_prior += [int((pp & yy).sum()), int((pp & ~yy).sum()), int((~pp & yy).sum())]
    all_rep = np.concatenate([rc["rep"] for rc in recs]) if recs else np.zeros((0, model.report.dim), np.float32)
    for ri, (rc, (pick, ok)) in enumerate(zip(recs, picks)):
        loc, norm, z, rep, pend = rc["loc"], rc["norm"], rc["z"], rc["rep"], rc["pend"]
        y = norm[:, 4] > 0.5
        keep = np.ones(len(z), bool)
        keep[pend] = False
        pred = z >= tau
        c_np += [int((pred & y & keep).sum()), int((pred & ~y & keep).sum()), int((~pred & y & keep).sum())]
        win = (loc[:, 2] // wms).astype(np.int64)
        F_ = np.minimum(D, int(win.max()) - win[pend])
        if pend.size:
            xt = torch.as_tensor(loc[:, 0], dtype=torch.float32, device=device)
            yt = torch.as_tensor(loc[:, 1], dtype=torch.float32, device=device)
            tt = torch.as_tensor(loc[:, 2], dtype=torch.float32, device=device)
            pt = torch.as_tensor(norm[:, 3], dtype=torch.float32, device=device)
            zt = torch.as_tensor(np.nan_to_num(z, nan=-20.0), dtype=torch.float32, device=device)
            rt = torch.as_tensor(rep, device=device)
            for a in range(0, pend.size, batch):
                sl = slice(a, min(pend.size, a + batch))
                pe = torch.as_tensor(pend[sl], device=device)
                pk = torch.as_tensor(pick[sl], device=device)
                okk = torch.as_tensor(ok[sl], device=device)
                raw, rtok, okT, dw, dist2 = NV.build_tokens(xt, yt, tt, pt, zt, rt, pe, pk, okk, offsets)
                valid = torch.as_tensor(np.arange(D + 1)[None, :] <= F_[sl][:, None], device=device)
                model.big.keep_extras = True
                with torch.no_grad():
                    U = model.decide(raw, rtok, okT, dw, dist2, zt[pe], pt[pe], valid, D)
                ex = model.big.extras
                model.big.keep_extras = False
                lam.append(U.float().cpu().numpy())
                mass.append(ex["mass"].float().cpu().numpy())
                neff.append(ex["neff"].float().cpu().numpy())
        meta["idx"].append(pend)
        meta["rec"].append(np.full(pend.size, ri))
        meta["F"].append(F_)
        meta["z0"].append(z[pend])
        meta["label"].append(y[pend])
        meta["p"].append(norm[pend, 3].astype(np.float32))
        meta["s"].append(win[pend])
    meta = {k: np.concatenate(v) for k, v in meta.items()}
    lt = np.concatenate(lam) if lam else np.zeros((0, D + 1), np.float32)
    mass = np.concatenate(mass) if mass else np.zeros((0, D + 1), np.float32)
    neff = np.concatenate(neff) if neff else np.zeros((0, D + 1), np.float32)
    y = meta["label"].astype(bool)
    clo, chi = TD.bounds(tau, commit_alpha)
    Fm = np.minimum(meta["F"], D)

    def iou(L):
        pred = lt[np.arange(len(L)), L] >= tau
        c = c_np + [(pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()]
        return float(c[0]) / float(max(c.sum(), 1))
    Lc = TD.commit_lags(lt, Fm, clo, chi)
    summary = {"split": split, "recordings": len(names), "pending": int(len(Fm)),
               "prior_iou": float(c_prior[0]) / float(max(c_prior.sum(), 1)),
               "report_std": [round(float(v), 4) for v in all_rep.std(0)] if len(all_rep) else [],
               "crossing_iou": iou(Lc), "arrival_iou": iou(np.zeros_like(Fm)), "fixed_iou": iou(Fm),
               "mean_wait_pending": float(Lc.mean()) if len(Lc) else 0.0}
    if out is not None:
        zdir = os.path.join(out, "z0_" + split)
        os.makedirs(zdir, exist_ok=True)
        for rc in recs:
            np.savez(os.path.join(zdir, rc["name"]), traj=rc["z"][:, None], labels=rc["norm"][:, 4].astype(np.float32),
                     log_w=np.float32(log_w), window_ms=np.int64(wms))
        np.savez(os.path.join(out, "evm_%s_meta.npz" % split), names=np.array(names), log_w=np.float32(log_w),
                 alpha=np.float64(st["alpha"]), filter_alpha=np.float64(st["filter_alpha"]), D=np.int64(D),
                 past=np.int64(P), window_ms=np.int64(wms), band_lo=np.float64(lo), band_hi=np.float64(hi),
                 nonpending=c_np, **meta)
        run = os.path.join(out, "run")
        os.makedirs(run, exist_ok=True)
        np.savez(os.path.join(run, "raw_%s.npz" % split), lam_traj=lt.astype(np.float32), lags=Lc, mass=mass, neff=neff)
        summary["calibration_slope"] = TD.calibration_slopes(lt, y, Fm)
    return summary


# ----------------------------------------------------------------------------------------------------------------------
# joint training
# ----------------------------------------------------------------------------------------------------------------------


def piece_map(pieces, k, device):
    """[k, B] piece id of every (window, stream) of the chunk (-1 = empty window)."""
    pm = torch.full((k, len(pieces)), -1, dtype=torch.long)
    pid = 0
    for b, stream in enumerate(pieces):
        for _, lo, hi, offset, _ in stream:
            pm[offset:offset + (hi - lo), b] = pid
            pid += 1
    return pm.to(device)


def joint_chunk(model, st, events, logits, reports, labels, k, pm, args, gen):
    """Decision-neuron stepwise BCE of the chunk's pending events (None when there is none)."""
    P, D = st["P"], st["D"]
    lo, hi = st["band"]
    z0 = logits.float() - st["log_w"]
    z0_det = z0.detach()
    tb, bb = events["t"], events["b"]
    piece = pm[tb, bb]
    born = (z0_det > lo) & (z0_det < hi) & (tb >= P) & (tb <= k - 1 - D) & (piece >= 0)
    born &= (pm[(tb - P).clamp(min=0), bb] == piece) & (pm[(tb + D).clamp(max=k - 1), bb] == piece)
    pend = torch.nonzero(born, as_tuple=True)[0]
    if pend.numel() == 0:
        return None, 0
    if pend.numel() > args.max_pending:
        pend = pend[torch.randperm(pend.numel(), generator=gen, device="cpu")[:args.max_pending].to(pend.device)]
    offsets = list(range(-P, D + 1))
    off = torch.as_tensor(offsets, device=z0.device)
    cell = torch.where(piece >= 0, piece * k + tb, torch.full_like(piece, -1))
    cq = (piece[pend] * k)[:, None] + (tb[pend][:, None] + off[None, :])
    xf, yf = events["x"].float(), events["y"].float()
    t_ms = (tb.float() + events["t_local"].float()) * float(st["window_ms"])
    pf = events["p"].float()
    pick, ok = NV.gpu_picks(xf, yf, cell, z0_det, pend, cq, st["k_near"], st["k_evid"], st["flo"])
    zp = z0 if args.prior_grad == "on" else z0_det   # off (default): the priors are trained by the prior loss only;
    raw, rtok, okT, dw, dist2 = NV.build_tokens(xf, yf, t_ms, pf, zp, reports.float(), pend, pick, ok, offsets)
    valid = torch.ones((pend.numel(), D + 1), dtype=torch.bool, device=z0.device)   # the reports carry the rest
    U = model.decide(raw, rtok, okT, dw, dist2, zp[pend], pf[pend], valid, D)
    y = labels[pend].float()[:, None].expand_as(U)
    return F.binary_cross_entropy_with_logits(U, y), int(pend.numel())


def train_epoch(model, cfg, q99, st, train_set, lengths, optimizer, comm, device, rng, args):
    k, n_streams = int(cfg["tbptt_k"]), int(cfg["streams"])
    plan = T.plan_streams(lengths, n_streams, k, rng, cfg.get("stream_layout", "greedy"))
    item_ends = {(index, first + n) for _, items in plan for index, first, n in map(T.plan_item, items)}
    n_chunks = int(math.ceil(T.stream_total_windows(plan) / float(k)))
    if args.max_chunks:
        n_chunks = min(n_chunks, args.max_chunks)
    dtype = next(model.small.parameters()).dtype
    sources = T.StreamSources(train_set, cfg, q99, device, dtype, None)
    weight = torch.tensor([st["pos_weight"]], dtype=dtype, device=device)
    gen = torch.Generator().manual_seed(int(rng.randint(2 ** 31 - 1)))
    if comm is not None:
        comm.start_epoch()
    states = None
    rec = {"small": 0.0, "big": 0.0, "chunks": 0, "big_chunks": 0, "pending": 0}
    model.train()
    for c in range(n_chunks):
        pieces = T.stream_pieces(plan, c * k, (c + 1) * k)
        x, events, labels, keep = T.assemble_streams(sources, pieces, k, cfg, device, dtype)
        valid_w = T.stream_valid(pieces, k, device, dtype)
        if labels.numel():
            logits, new_states, info = model.small.forward_chunk(x, events, states, keep=keep, collect=comm is not None)
            reports = model.report()
            small_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight, reduction="mean")
            terms = [small_loss]
            if comm is not None:
                pen = comm.chunk_loss(model.small, info["u_pre"], states, keep, 1, valid_w)
                if pen is not None:
                    terms.append(pen)
            big_loss, n_p = joint_chunk(model, st, events, logits, reports, labels, k, piece_map(pieces, k, device),
                                        args, gen)
            if big_loss is not None:
                terms.append(args.lambda_big * big_loss)
                rec["big"] += float(big_loss.detach())
                rec["big_chunks"] += 1
                rec["pending"] += n_p
            optimizer.zero_grad(set_to_none=True)
            sum(terms).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            if comm is not None:
                comm.end_update()
            rec["small"] += float(small_loss.detach())
            rec["chunks"] += 1
            states = T.detach_states(new_states)
        else:
            with torch.no_grad():
                _, new_states, _ = model.small.forward_chunk(x, events, states, keep=keep)
                model.report._h = None
            states = T.detach_states(new_states)
        for stream in pieces:
            for index, lo_, hi_, _, _ in stream:
                if (index, hi_) in item_ends:
                    sources.release(index)
    rec["small"] /= max(rec["chunks"], 1)
    rec["big"] /= max(rec["big_chunks"], 1)
    rec["pending_per_chunk"] = rec["pending"] / max(rec["big_chunks"], 1)
    if comm is not None:
        rec["comm"] = comm.summary()
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, help="backbone checkpoint (the staged run's backbone)")
    ap.add_argument("--stage1", required=True, help="staged v11 run dir (decision_neurons.pt, summary.json)")
    ap.add_argument("--stage1-evidence", required=True, help="evidence prefix of the staged run")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4, help="0 = export the staged network (consistency gate)")
    ap.add_argument("--k", type=int, default=32, help="TBPTT chunk (windows); births in [past, k - 1 - D]")
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--report-dim", type=int, default=4)
    ap.add_argument("--lambda-big", type=float, default=1.0)
    ap.add_argument("--lr-small", type=float, default=1e-4)
    ap.add_argument("--lr-big", type=float, default=5e-4)
    ap.add_argument("--max-pending", type=int, default=1024)
    ap.add_argument("--prior-grad", choices=("off", "on"), default="off",
                    help="off (default): the neurons' loss does not reach the priors (own or neighbours'), only the "
                         "reports, so the prior stays a calibrated prior; on: also through the priors (ablation)")
    ap.add_argument("--max-chunks", type=int, default=0, help="smoke tests: chunks per epoch")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: recordings per split in the evaluation")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--comm-mu", type=float, default=None, help="start of the budget multiplier (default: the "
                    "backbone run's last mu_end from metrics.jsonl next to the checkpoint, else 0)")
    ap.add_argument("--gate-z0", default=None, help="consistency gate (--epochs 0 only): read the priors of every split "
                    "from <dir>/z0_<split> instead of running the small network")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.gate_z0 and args.epochs:
        raise ValueError("--gate-z0 is only for --epochs 0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    model, cfg, q99, st, ck = build(args, device)
    print("nested V11: P %d D %d K %d+%d band (%.2f, %.2f) tau %.2f report dim %d | backbone %s | staged %s" % (
        st["P"], st["D"], st["k_near"], st["k_evid"], st["band"][0], st["band"][1], st["tau"], args.report_dim,
        args.checkpoint, args.stage1), flush=True)
    t0 = time.time()
    hist = []
    best, best_state, best_ep = -1.0, None, -1
    if args.epochs > 0:
        small_params = list(model.small.parameters())
        big_params = list(model.big.parameters()) + list(model.report.parameters())
        optimizer = torch.optim.AdamW([{"params": small_params, "lr": args.lr_small, "weight_decay": 0.0},
                                       {"params": big_params, "lr": args.lr_big, "weight_decay": 1e-4}])
        base = [args.lr_small, args.lr_big]
        comm = T.CommBudget.from_config(cfg, model.small)
        if comm is not None:
            mu = args.comm_mu
            mp = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "metrics.jsonl")
            if mu is None and os.path.exists(mp):
                rows = [json.loads(ln) for ln in open(mp, encoding="utf-8") if ln.strip()]
                rows = [r for r in rows if r.get("comm")]
                mu = float(rows[-1]["comm"]["mu_end"]) if rows else None
            comm.mu = float(mu or 0.0)
            print("message budget: %.4g per window, alpha %.2f, mu starts at %.4g" % (comm.budget, comm.alpha, comm.mu),
                  flush=True)
        train_set = EvUAVStream(cfg["root"], "train", cfg, T.train_file_names(cfg))
        lengths = [int(train_set[i].n_windows) for i in range(len(train_set))]
        for ep in range(args.epochs):
            scale = 1.0 - 0.9 * ep / max(args.epochs - 1, 1)
            for g, b in zip(optimizer.param_groups, base):
                g["lr"] = b * scale
            rng = np.random.RandomState(args.seed * 1000 + ep + 11)
            rec = train_epoch(model, cfg, q99, st, train_set, lengths, optimizer, comm, device, rng, args)
            val = export_split(model, cfg, q99, st, "val", device, limit=args.limit, workers=args.workers)
            rec.update(epoch=ep, val=val, seconds=round(time.time() - t0, 1))
            hist.append(rec)
            print("ep %d | small BCE %.4f big BCE %.4f pending/chunk %.0f | val crossing %.4f fixed %.4f arrival %.4f "
                  "prior-only %.4f pending %d report std %s | %.0f s" % (
                      ep, rec["small"], rec["big"], rec["pending_per_chunk"], val["crossing_iou"], val["fixed_iou"],
                      val["arrival_iou"], val["prior_iou"], val["pending"], val["report_std"], time.time() - t0),
                  flush=True)
            if val["crossing_iou"] > best:
                best, best_ep = val["crossing_iou"], ep
                best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(), "args": vars(args), "settings": {k_: v for k_, v in st.items()},
                "config": cfg, "stats": ck["stats"], "best_epoch": best_ep},
               os.path.join(args.out, "nested.pt"))
    torch.save({"model": model.small.state_dict(), "config": cfg, "stats": ck["stats"]},
               os.path.join(args.out, "small_backbone.pt"))
    summary = {"args": vars(args), "best_epoch": best_ep, "history": hist,
               "v11": model.big.learned_constants()}
    for split in ("val", "test"):
        summary[split] = export_split(model, cfg, q99, st, split, device, out=args.out, limit=args.limit,
                                      workers=args.workers,
                                      z0_dir=os.path.join(args.gate_z0, "z0_" + split) if args.gate_z0 else None)
    summary["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("NESTED FINISHED best ep %d | val crossing %.4f fixed %.4f arrival %.4f | test crossing %.4f fixed %.4f "
          "arrival %.4f | %.0f s" % (best_ep, summary["val"]["crossing_iou"], summary["val"]["fixed_iou"],
                                     summary["val"]["arrival_iou"], summary["test"]["crossing_iou"],
                                     summary["test"]["fixed_iou"], summary["test"]["arrival_iou"], summary["seconds"]),
          flush=True)


if __name__ == "__main__":
    main()
