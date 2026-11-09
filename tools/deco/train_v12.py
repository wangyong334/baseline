"""DeCo-SNN V12 (11-09): train the whole network end to end from scratch, export, fit the commit range, score.

Network: tools/deco/deco_v12.py (BIR U-Net small network + one spiking decision neuron per event + feedback map).
Training: the backbone's stateful TBPTT (train_stream_v1 trainer=streams: B parallel streams, chunks of k windows,
one update per chunk). Per chunk: layers enc1 .. dec2 run over the chunk at once (they get no feedback), then window by
window dec1 (+ the previous window's feedback map) -> priors z0 of the window's events -> births -> every living neuron
takes its step -> feedback map. The decision layer (event store + neurons) and the map are carried across chunks like
the network states, so training sees exactly the deployed decision layer.
Loss (one objective): every membrane value of every decision neuron is a posterior log-odds of "target": the birth
value U(-1) = z0 and U(0) .. U(D). Each (event, step) pair counts once, in the chunk of the step's window: the birth
values of all events of the chunk, and the steps 0 .. D that fall into the chunk of a sample of the neurons that lived
in it (labels x uncertain / confident strata, importance weights, so the sum estimates the sum over all pairs). The log
loss is class-balanced by the logit adjustment of the backbone (pos_weight w on U + log w: the minimiser is still the
calibrated log-odds, and the birth term is exactly the backbone's own training loss); --balance off: plain log loss.
Gradient routing (11-09 local checks): the birth terms train the small network and the feedback convolution (the
prior is the backbone's own calibrated prior, and the loop learns to improve the next priors from the decision layer's
news); the step terms train the decision neurons on top of the priors (stop-gradient at the stage boundary). With
--prior-grad on the step terms also reach the small network through the neighbours' priors and the neurons' contexts
(ablation; in the local checks it hurt the decision layer and made waiting worse instead of better);
+ the message-budget penalty of the small network (Lagrangian, as its own training).
Model selection: pooled IoU of the fixed-band commit on val (streaming export of every event).
Final (best epoch): streaming export of val / test exactly as deployed, z0_<split>/<recording>.npz,
evm_<split>_meta.npz, run/raw_<split>.npz; tools/deco/fit_range.py fits the commit thresholds on the val trajectories
(label-free) and gives the official four metrics (run/range/range.json); runtime.json: per-window latency of the
deployed network (one window at a time, CUDA-synchronised): small network, decision layer, feedback map.
Usage (code dir): python tools/deco/train_v12.py --data-root <root> --out <dir> [--seed 37] [--feedback innov|belief|none]
       [--birth none|band] [--epochs 50]
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import deco_v12 as V12  # noqa: E402
import train_dnm as TD  # noqa: E402
import train_stream_v1 as T  # noqa: E402
from dataset.ev_uav_stream import EvUAVStream  # noqa: E402
from dataset.stream_windows import num_input_channels  # noqa: E402
from model.lif2d_stream import mask_state  # noqa: E402
from utils.stream_common import linear_epoch_lr  # noqa: E402


def build(args, device):
    cfg = T.load_flat_config(args.config)
    cfg["root"] = args.data_root
    cfg.update(execution="layer", input_device="gpu", tbptt_k=args.k, streams=args.streams, accumulate=1,
               stream_groups=1, trainer="streams", seed=args.seed)
    stats, q99, pos_weight = T.load_stats(cfg)
    small = T.build_net(cfg, device)
    model = V12.DeCoV12(small, P=args.past, D=args.deadline, k_near=args.k_near, k_evid=args.k_evid,
                        window_ms=int(cfg["window_ms"]), pos_weight=pos_weight, alpha=args.alpha,
                        feedback=args.feedback, birth=args.birth, value=args.value, gate=args.gate, rho=args.rho,
                        audit=args.audit, dim=args.dim)
    return model.to(device), cfg, q99, stats


# ----------------------------------------------------------------------------------------------------------------------
# streams -> chunks of events in the engine's layout
# ----------------------------------------------------------------------------------------------------------------------


def item_table(plan):
    """(stream, recording index) -> (global window of the item's first window, first recording window, windows)."""
    out = {}
    for b, (delay, items) in enumerate(plan):
        start = delay
        for item in items:
            index, first, n = T.plan_item(item)
            out[(b, index)] = (start, first, n)
            start += n
    return out


def export_plan(lengths, n_streams):
    """Recordings in name order, dealt to the streams with the fewest windows, no delays (export / evaluation)."""
    streams = [[] for _ in range(n_streams)]
    totals = [0] * n_streams
    for i, n in enumerate(lengths):
        s = int(np.argmin(totals))
        streams[s].append((i, int(n)))
        totals[s] += int(n)
    return [(0, items) for items in streams if items]


def assemble(sources, pieces, k, g0, cfg, device, dtype, items):
    """x [k, B, C, H, W], keep [k, B] and the chunk's events as engine columns sorted by (window, stream) (None when
    the chunk has no event)."""
    B = len(pieces)
    x = torch.zeros((k, B, num_input_channels(cfg["time_bins"]), int(cfg["pad_height"]), int(cfg["pad_width"])),
                    dtype=dtype, device=device)
    keep = torch.ones((k, B), dtype=dtype, device=device)
    parts = {name: [] for name in ("g", "b", "y", "x", "pf", "tl", "lab", "rec", "fidx", "rw", "glo", "ghi")}
    for b, stream in enumerate(pieces):
        for index, lo, hi, offset, new_start in stream:
            xs, ev, lab, idx = sources.get(index).chunk(lo, hi)
            x[offset:offset + (hi - lo), b] = xs[:, 0]
            if new_start:
                keep[offset, b] = 0
            n = int(lab.numel())
            if n == 0:
                continue
            gstart, first, nwin = items[(b, index)]
            rw = ev["t"].long() + int(lo)

            def full(v):
                return torch.full((n,), int(v), dtype=torch.long, device=device)
            parts["g"].append(rw - int(first) + int(gstart))
            parts["b"].append(full(b))
            parts["y"].append(ev["y"].long())
            parts["x"].append(ev["x"].long())
            parts["pf"].append(ev["p"].float())
            parts["tl"].append(ev["t_local"].float())
            parts["lab"].append(lab.float())
            parts["rec"].append(full(index))
            parts["fidx"].append(torch.as_tensor(np.asarray(idx), dtype=torch.long, device=device))
            parts["rw"].append(rw)
            parts["glo"].append(full(gstart))
            parts["ghi"].append(full(gstart + nwin - 1))
    if not parts["g"]:
        return x, keep, None
    cols = {key: torch.cat(v) for key, v in parts.items()}
    n = int(cols["g"].numel())
    order = torch.argsort((cols["g"] * B + cols["b"]) * n + torch.arange(n, device=device))
    cols = {key: v[order] for key, v in cols.items()}
    cols["xf"], cols["yf"] = cols["x"].float(), cols["y"].float()
    cols["tm"] = (cols["rw"].float() + cols["tl"]) * float(cfg["window_ms"])
    cols["z"] = torch.full_like(cols["xf"], -20.0)
    return x, keep, cols


def run_windows(model, eng, s1, deep1, keep, states, fbmap, g0, zs, timer=None):
    """Window by window: dec1 (+ feedback) -> priors -> births -> steps -> feedback map -> retirements. Appends the
    priors (with gradients when enabled) to zs in store order; returns (dec1 state, feedback map)."""
    st = eng.st
    state7 = states[6] if states is not None else None
    sync = (lambda: torch.cuda.synchronize(eng.dev)) if (timer is not None and eng.dev.type == "cuda") else (lambda: None)
    for w in range(int(s1.shape[0])):
        g = g0 + w
        t0 = time.perf_counter()
        if keep is not None:
            state7 = mask_state(state7, keep, w)
        fb = fbmap
        if fb is not None and keep is not None:
            fb = fb * keep[w].to(fb.dtype).view(-1, 1, 1, 1)
        u7, state7 = model.dec1_window(s1[w], deep1[w], state7, fb)
        a, e = st.window(g)
        if e > a:
            ev = {"b": st.b[a:e], "y": st.y[a:e], "x": st.x[a:e], "p": st.pf[a:e], "t_local": st.tl[a:e]}
            z = model.readout_window(u7, deep1[w], ev).float()
            zs.append(z)
            st.z[a:e] = z.detach()
        sync()
        t1 = time.perf_counter()
        eng.birth(g, a, e)
        eng.advance(g)
        sync()
        t2 = time.perf_counter()
        fbmap = eng.feedback(g)
        sync()
        t3 = time.perf_counter()
        if timer is not None:
            timer["dec1_ms"].append(1000.0 * (t1 - t0))
            timer["neurons_ms"].append(1000.0 * (t2 - t1))
            timer["map_ms"].append(1000.0 * (t3 - t2))
            timer["events"].append(e - a)
            timer["living"].append(eng.n)
        eng.retire(g)
    return state7, fbmap


# ----------------------------------------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------------------------------------


def allocate(sizes, n):
    """Equal shares of n samples over the strata, what a small stratum cannot take goes to the others."""
    q = [0] * len(sizes)
    left = int(n)
    open_ = [i for i, s in enumerate(sizes) if s > 0]
    while left > 0 and open_:
        share = max(1, left // len(open_))
        nxt = []
        for i in open_:
            take = min(share, sizes[i] - q[i], left)
            q[i] += take
            left -= take
            if q[i] < sizes[i]:
                nxt.append(i)
            if left == 0:
                break
        open_ = nxt
    return q


def stratified(cand, strata, n, gen, n_strata=6):
    """Sample of the candidate rows with equal shares per stratum; importance weights (stratum size / sample size)."""
    groups = [cand[strata == s] for s in range(n_strata)]
    quota = allocate([int(g.numel()) for g in groups], n)
    rows, wts = [], []
    for g, m in zip(groups, quota):
        if m > 0:
            sel = torch.randperm(int(g.numel()), generator=gen)[:m].to(g.device)
            rows.append(g[sel])
            wts.append(torch.full((m,), float(g.numel()) / float(m), device=g.device))
    if not rows:
        return cand[:0], torch.zeros(0, device=cand.device)
    return torch.cat(rows), torch.cat(wts)


def log_loss(model, u, y, balance):
    if balance == "on":
        pw = torch.tensor([model.pos_weight], dtype=u.dtype, device=u.device)
        return F.binary_cross_entropy_with_logits(u + model.log_w, y, pos_weight=pw, reduction="none")
    return F.binary_cross_entropy_with_logits(u, y, reduction="none")


def decision_loss(model, eng, zs, g0, g1, base0, args, gen):
    """The decision objective of one chunk (see the module docstring) and a small record."""
    st, D = eng.st, model.D
    z_chunk = torch.cat(zs)
    lab = st.lab[base0:]
    if int(z_chunk.numel()) != int(lab.numel()):
        raise RuntimeError("priors (%d) do not match the chunk's events (%d)" % (z_chunk.numel(), lab.numel()))
    prior = log_loss(model, z_chunk, lab, args.balance)
    nrn = eng.chunk_neurons()
    gs, ghi = nrn["gs"], nrn["ghi"]
    d_lo = (g0 - gs).clamp(min=0)
    d_hi = torch.min(torch.min(ghi - gs, g1 - 1 - gs), torch.full_like(gs, D))
    n_i = (d_hi - d_lo + 1).clamp(min=0)
    pairs = float(lab.numel()) + float(n_i.sum())
    cand = torch.nonzero(n_i > 0, as_tuple=True)[0]
    rec = {"events": int(lab.numel()), "neurons": int(cand.numel()), "prior": float(prior.detach().mean())}
    num = prior.sum()
    if cand.numel():
        u_, pos = nrn["U"], nrn["lab"] > 0.5             # strata: label x {uncertain, confident right, confident wrong}
        hard = (u_ > model.band[0]) & (u_ < model.band[1])
        if args.strata == 6:
            wrong = ~hard & ((u_ >= model.tau) != pos)
            strata = 3 * pos.long() + torch.where(hard, torch.zeros_like(pos.long()), 1 + wrong.long())
        else:                                            # 11-09 first version: label x {uncertain, confident}
            strata = 2 * pos.long() + hard.long()
        rows, wts = stratified(cand, strata[cand], args.samples, gen, n_strata=args.strata)
        z_full = torch.cat([st.z[:base0], z_chunk])
        if args.prior_grad == "off":
            z_full = z_full.detach()
        steps = torch.arange(D + 1, device=z_full.device)[None, :]
        run = steps <= d_hi[rows][:, None]
        lmask = (run & (steps >= d_lo[rows][:, None])).float()
        U = model.trajectories(st, z_full, nrn["eid"][rows], nrn["pick"][rows].long(), nrn["ok"][rows], run,
                               own_prior_grad=args.own_prior_grad == "on")
        y = nrn["lab"][rows][:, None].expand_as(U)
        per = (log_loss(model, U, y, args.balance) * lmask).sum(1)
        num = num + (per * wts).sum()
        rec.update(sampled=int(rows.numel()), hard=int(hard[cand].sum()),
                   step=float(((per * wts).sum() / (wts * lmask.sum(1)).sum().clamp(min=1e-9)).detach()))
    return num / max(pairs, 1.0), rec


def train_epoch(model, cfg, q99, train_set, lengths, optimizer, comm, device, rng, args):
    k, B = int(cfg["tbptt_k"]), int(cfg["streams"])
    H, W = int(cfg["pad_height"]), int(cfg["pad_width"])
    plan = T.plan_streams(lengths, B, k, rng, cfg.get("stream_layout", "greedy"))
    items = item_table(plan)
    item_ends = {(index, first + n) for _, its in plan for index, first, n in map(T.plan_item, its)}
    n_chunks = int(math.ceil(T.stream_total_windows(plan) / float(k)))
    if args.max_chunks:
        n_chunks = min(n_chunks, args.max_chunks)
    dtype = next(model.small.parameters()).dtype
    sources = T.StreamSources(train_set, cfg, q99, device, dtype, None)
    gen = torch.Generator().manual_seed(int(rng.randint(2 ** 31 - 1)))
    eng = V12.Engine(model, B, H, W, device, sub_batch=args.sub_batch, pick_block=args.pick_block)
    if comm is not None:
        comm.start_epoch()
    states, fbmap = None, None
    agg = {"loss": 0.0, "prior": 0.0, "step": 0.0, "updates": 0, "neurons": 0, "sampled": 0, "events": 0}
    prof = {key: 0.0 for key in ("data", "layers", "windows", "loss", "backward")}
    peak, reserved = 0.0, 0.0
    t0 = time.time()
    model.train()
    cuda = device.type == "cuda"

    def mark():
        if args.profile and cuda:
            torch.cuda.synchronize(device)
        return time.perf_counter()
    for c in range(n_chunks):
        g0, g1 = c * k, (c + 1) * k
        if args.profile and cuda:
            torch.cuda.reset_peak_memory_stats(device)
        ta = mark()
        pieces = T.stream_pieces(plan, g0, g1)
        x, keep, cols = assemble(sources, pieces, k, g0, cfg, device, dtype, items)
        valid_w = T.stream_valid(pieces, k, device, dtype)
        eng.begin_chunk(g0 - model.P - model.D)
        base0 = eng.st.n
        if cols is not None:
            eng.st.append(cols)
            tb = mark()
            s1, deep1, st6, u6 = model.pre_dec1(x, states, keep, collect=comm is not None)
            tc = mark()
            zs = []
            state7, fbmap = run_windows(model, eng, s1, deep1, keep, states, fbmap, g0, zs)
            td = mark()
            loss, rec = decision_loss(model, eng, zs, g0, g1, base0, args, gen)
            te = mark()
            prof["data"] += tb - ta
            prof["layers"] += tc - tb
            prof["windows"] += td - tc
            prof["loss"] += te - td
            terms = [loss]
            if comm is not None:
                pen = comm.chunk_loss(model.small, model.message_layers_keep(u6), states, keep, 1, valid_w)
                if pen is not None:
                    terms.append(pen)
            total = sum(terms)
            if not math.isfinite(float(total.detach())):
                raise RuntimeError("non-finite loss in chunk %d" % c)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            for group in optimizer.param_groups:              # each network clipped as in its own training
                torch.nn.utils.clip_grad_norm_(group["params"], float(cfg["grad_clip"]))
            optimizer.step()
            if comm is not None:
                comm.end_update()
            prof["backward"] += mark() - te
            if args.profile and cuda:
                peak = max(peak, torch.cuda.max_memory_allocated(device) / 2.0 ** 30)
                reserved = max(reserved, torch.cuda.memory_reserved(device) / 2.0 ** 30)
            agg["loss"] += float(loss.detach())
            agg["prior"] += rec["prior"]
            agg["step"] += rec.get("step", 0.0)
            agg["updates"] += 1
            agg["neurons"] += rec["neurons"]
            agg["sampled"] += rec.get("sampled", 0)
            agg["events"] += rec["events"]
            new_states = list(st6) + [state7]
        else:
            with torch.no_grad():
                s1, deep1, st6, _ = model.pre_dec1(x, states, keep)
                state7, fbmap = run_windows(model, eng, s1, deep1, keep, states, fbmap, g0, [])
            new_states = list(st6) + [state7]
        states = T.detach_states(new_states)
        for stream in pieces:
            for index, lo_, hi_, _, _ in stream:
                if (index, hi_) in item_ends:
                    sources.release(index)
        if args.log_every and (c + 1) % args.log_every == 0:
            u = max(agg["updates"], 1)
            print("  chunk %d/%d | loss %.4f prior %.4f step %.4f | neurons/chunk %.0f events/chunk %.0f | %.0f s%s" % (
                c + 1, n_chunks, agg["loss"] / u, agg["prior"] / u, agg["step"] / u, agg["neurons"] / u,
                agg["events"] / u, time.time() - t0, (" | s/chunk " + " ".join("%s %.2f" % (key, v / u) for key, v in
                                                       prof.items()) + " | peak %.1f GB reserved %.1f GB" % (peak, reserved)) if args.profile else ""),
                flush=True)
    u = max(agg["updates"], 1)
    out = {"loss": agg["loss"] / u, "prior": agg["prior"] / u, "step": agg["step"] / u, "updates": agg["updates"],
           "neurons_per_chunk": agg["neurons"] / u, "sampled_per_chunk": agg["sampled"] / u,
           "events_per_chunk": agg["events"] / u, "train_seconds": round(time.time() - t0, 1)}
    if args.profile:
        out["profile_s_per_chunk"] = {key: round(v / u, 3) for key, v in prof.items()}
        out["peak_gb"] = round(peak, 2)
        out["reserved_gb"] = round(reserved, 2)
    if comm is not None:
        out["comm"] = comm.summary()
    return out


# ----------------------------------------------------------------------------------------------------------------------
# streaming export (as deployed)
# ----------------------------------------------------------------------------------------------------------------------


def stream_split(model, cfg, q99, split, device, names, n_streams, chunk, sub_batch, pick_block, timer=None, gate=None):
    """Streams every recording of names through the deployed network (gate: the fire-and-stop band of the gated
    deployment, None = every neuron computes all its steps). Returns per recording the priors of all events (file
    order), the neurons (recording, file index, birth window in the recording, trajectory, mass, effective count;
    trajectories NaN after a gated neuron fired) and the decision layer's work (neuron steps, tokens, windows)."""
    model.eval()
    ds = EvUAVStream(cfg["root"], split, cfg, names)
    seqs_n = []
    for i in range(len(ds)):
        seq = ds[i]
        seqs_n.append((int(seq.n_windows), int(seq.n_events)))
    plan = export_plan([n for n, _ in seqs_n], n_streams)
    items = item_table(plan)
    item_ends = {(index, first + n) for _, its in plan for index, first, n in map(T.plan_item, its)}
    B = len(plan)
    H, W = int(cfg["pad_height"]), int(cfg["pad_width"])
    dtype = next(model.small.parameters()).dtype
    sources = T.StreamSources(ds, cfg, q99, device, dtype, None)
    eng = V12.Engine(model, B, H, W, device, keep_traj=True, sub_batch=sub_batch, pick_block=pick_block, gate=gate)
    z_rec = [np.full(n_ev, np.nan, np.float32) for _, n_ev in seqs_n]
    outs = []
    total = T.stream_total_windows(plan)
    states, fbmap = None, None
    sync = (lambda: torch.cuda.synchronize(device)) if (timer is not None and device.type == "cuda") else (lambda: None)
    with torch.no_grad():
        for g0 in range(0, total, chunk):
            g1 = min(total, g0 + chunk)
            k = g1 - g0
            pieces = T.stream_pieces(plan, g0, g1)
            x, keep, cols = assemble(sources, pieces, k, g0, cfg, device, dtype, items)
            eng.begin_chunk(g0 - model.P - model.D)
            base0 = eng.st.n
            if cols is not None:
                eng.st.append(cols)
            sync()
            t0 = time.perf_counter()
            s1, deep1, st6, _ = model.pre_dec1(x, states, keep)
            sync()
            if timer is not None:
                timer["layers_ms"].extend([1000.0 * (time.perf_counter() - t0) / k] * k)
            state7, fbmap = run_windows(model, eng, s1, deep1, keep, states, fbmap, g0, [], timer=timer)
            states = list(st6) + [state7]
            st = eng.st
            if st.n > base0:
                rec_c, fidx_c = st.rec[base0:].cpu().numpy(), st.fidx[base0:].cpu().numpy()
                z_c = st.z[base0:].cpu().numpy()
                for r in np.unique(rec_c):
                    m_ = rec_c == r
                    z_rec[r][fidx_c[m_]] = z_c[m_]
            for q in eng.retired:
                outs.append({"rec": st.rec[q["eid"]].cpu().numpy(), "fidx": st.fidx[q["eid"]].cpu().numpy(),
                             "rw": st.rw[q["eid"]].cpu().numpy(), "traj": q["traj"].cpu().numpy(),
                             "mass": q["mass"].cpu().numpy(), "neff": q["neff"].cpu().numpy()})
            eng.retired = []
            for stream in pieces:
                for index, lo_, hi_, _, _ in stream:
                    if (index, hi_) in item_ends:
                        sources.release(index)
    if eng.n:
        raise RuntimeError("%d neurons still living after the last window" % eng.n)
    cat = {}
    for key in ("rec", "fidx", "rw", "traj", "mass", "neff"):
        if outs:
            cat[key] = np.concatenate([o[key] for o in outs])
        elif key in ("traj", "mass", "neff"):
            cat[key] = np.zeros((0, model.D + 1), np.float32)
        else:
            cat[key] = np.zeros(0, np.int64)
    cat["work"] = dict(eng.stats, windows=int(sum(n for n, _ in seqs_n)))
    return z_rec, cat


def export_split(model, cfg, q99, split, device, args, out=None, limit=0, commit_alpha=0.01, gate=None):
    """Streaming export of a split; summary (pooled IoU of the priors, arrival, fixed-band commit, deadline, the
    decision layer's work per window); with out, writes the files tools/deco/fit_range.py reads. gate: see
    stream_split (a fired neuron keeps its committed membrane in the exported trajectory)."""
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(cfg["root"], split, "*.npz")))
    if limit:
        names = names[:limit]
    tau, D, wms = model.tau, model.D, model.window_ms
    z_rec, nr = stream_split(model, cfg, q99, split, device, names, args.eval_streams, args.eval_chunk,
                             args.sub_batch, args.pick_block, gate=gate)
    work = nr.pop("work")
    o = np.lexsort((nr["fidx"], nr["rec"]))
    nr = {key: v[o] for key, v in nr.items()}
    labels, pol, maxwin = [], [], []
    for name in names:
        with np.load(os.path.join(cfg["root"], split, name)) as d:
            norm = np.asarray(d["evs_norm"])
            t = np.asarray(d["ev_loc"])[:, 2]
        labels.append(norm[:, 4] > 0.5)
        pol.append(norm[:, 3].astype(np.float32))
        maxwin.append(int(t.max()) // wms if t.size else 0)
    rec = nr["rec"].astype(np.int64)
    F_ = np.minimum(D, np.asarray(maxwin, np.int64)[rec] - nr["rw"]).clip(min=0)
    lt = nr["traj"].astype(np.float64)
    for d in range(1, D + 1):                          # steps not taken (fired, or the recording ended): held
        m_ = np.isnan(lt[:, d])
        lt[m_, d] = lt[m_, d - 1]
    lt = lt.astype(np.float32)
    per_rec = [nr["fidx"][rec == r] for r in range(len(names))]
    yb = np.concatenate([labels[r][per_rec[r]] for r in range(len(names))]) if len(F_) else np.zeros(0, bool)
    c_np, c_prior = np.zeros(3, np.int64), np.zeros(3, np.int64)
    for r in range(len(names)):
        z, y = z_rec[r], labels[r]
        if np.isnan(z).any():
            raise RuntimeError("%s: %d events without a prior" % (names[r], int(np.isnan(z).sum())))
        pred = z >= tau
        c_prior += [int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())]
        keep = np.ones(len(z), bool)
        keep[per_rec[r]] = False
        c_np += [int((pred & y & keep).sum()), int((pred & ~y & keep).sum()), int((~pred & y & keep).sum())]
    clo, chi = TD.bounds(tau, commit_alpha)

    def iou(L):
        pr = lt[np.arange(len(L)), L] >= tau
        c = c_np + [(pr & yb).sum(), (pr & ~yb).sum(), (~pr & yb).sum()]
        return float(c[0]) / float(max(c.sum(), 1))
    Lc = TD.commit_lags(lt, F_, clo, chi)
    summary = {"split": split, "recordings": len(names), "neurons": int(len(F_)),
               "prior_iou": float(c_prior[0]) / float(max(c_prior.sum(), 1)),
               "crossing_iou": iou(Lc), "arrival_iou": iou(np.zeros_like(F_)), "fixed_iou": iou(F_),
               "mean_wait": float(Lc.mean()) if len(Lc) else 0.0,
               "work_per_window": {key: round(work[key] / float(max(work["windows"], 1)), 1)
                                   for key in ("steps", "tokens", "new_tokens", "births")}}
    if out is not None:
        lo, hi = (-np.inf, np.inf) if model.birth == "none" else model.band
        zdir = os.path.join(out, "z0_" + split)
        os.makedirs(zdir, exist_ok=True)
        for r, name in enumerate(names):
            np.savez(os.path.join(zdir, name), traj=z_rec[r][:, None], labels=labels[r].astype(np.float32),
                     log_w=np.float32(model.log_w), window_ms=np.int64(wms))
        z0 = np.concatenate([z_rec[r][per_rec[r]] for r in range(len(names))]) if len(F_) else np.zeros(0)
        p_ = np.concatenate([pol[r][per_rec[r]] for r in range(len(names))]) if len(F_) else np.zeros(0)
        np.savez(os.path.join(out, "evm_%s_meta.npz" % split), names=np.array(names), log_w=np.float32(model.log_w),
                 alpha=np.float64(commit_alpha), filter_alpha=np.float64(0.01), D=np.int64(D), past=np.int64(model.P),
                 window_ms=np.int64(wms), band_lo=np.float64(lo), band_hi=np.float64(hi), nonpending=c_np,
                 idx=nr["fidx"].astype(np.int64), rec=rec, F=F_.astype(np.int64), z0=z0.astype(np.float32),
                 label=yb, p=p_.astype(np.float32), s=nr["rw"].astype(np.int64))
        run = os.path.join(out, "run")
        os.makedirs(run, exist_ok=True)
        np.savez(os.path.join(run, "raw_%s.npz" % split), lam_traj=lt, lags=Lc, mass=nr["mass"].astype(np.float32),
                 neff=nr["neff"].astype(np.float32))
        summary["calibration_slope"] = TD.calibration_slopes(lt, yb, F_)
    return summary


def decision_macs(model):
    """MACs of the decision layer: embedding one token (token MLP + key / summary / value projections), auditing one
    token in one step (value output, gate, pooling), one neuron step (query, value context, summary, mass and
    feed-forward layers)."""
    d, H = model.big.dim, model.big.heads
    return 6 * d + d * d + 3 * d * d, d * H + 2 * d, 7 * d * d + 2 * H * d


def backbone_macs(model, cfg, device):
    """Dense MACs of the small network for one window of one stream (every convolution counted in full)."""
    total = [0]

    def hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1]
        cin = mod.in_channels // mod.groups
        total[0] += int(out.numel()) // int(out.shape[0]) * cin * k
    hs = [mod.register_forward_hook(hook) for mod in model.small.modules()
          if isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
    x = torch.zeros((1, 1, num_input_channels(cfg["time_bins"]), int(cfg["pad_height"]), int(cfg["pad_width"])),
                    device=device)
    with torch.no_grad():
        s1, deep1, _, _ = model.pre_dec1(x, None)
        model.dec1_window(s1[0], deep1[0], None, None)
    for h in hs:
        h.remove()
    return total[0]


def runtime(model, cfg, q99, device, args, out):
    """Per-window latency of the deployed network on test recordings, one recording at a time, one window at a time:
    full (every neuron computes its D + 1 steps) and gated (a neuron stops once it has fired, fixed band)."""
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(cfg["root"], "test", "*.npz")))[:args.runtime]
    embed, audit, per_step = decision_macs(model)
    res = {"recordings": len(names), "window_ms": model.window_ms,
           "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
           "backbone_dense_gmac_per_window": round(backbone_macs(model, cfg, device) / 1e9, 3)}
    for mode, gate in (("full", None), ("gated", model.band)):
        timer = {key: [] for key in ("layers_ms", "dec1_ms", "neurons_ms", "map_ms", "events", "living")}
        work = {"steps": 0, "tokens": 0, "births": 0, "new_tokens": 0, "windows": 0}
        for name in names:
            _, nr = stream_split(model, cfg, q99, "test", device, [name], 1, 1, args.sub_batch, args.pick_block,
                                 timer=timer, gate=gate)
            for key in work:
                work[key] += nr["work"][key]
        arr = {key: np.asarray(v, np.float64) for key, v in timer.items()}
        small = arr["layers_ms"] + arr["dec1_ms"]
        total = small + arr["neurons_ms"] + arr["map_ms"]
        r = {"windows": int(len(total))}
        for key, a in (("small_network_ms", small), ("decision_layer_ms", arr["neurons_ms"]),
                       ("feedback_map_ms", arr["map_ms"]), ("total_ms", total)):
            r[key] = {"mean": round(float(a.mean()), 3), "p50": round(float(np.percentile(a, 50)), 3),
                      "p95": round(float(np.percentile(a, 95)), 3), "max": round(float(a.max()), 3)}
        nw = float(max(work["windows"], 1))
        r["events_per_window_mean"] = round(float(arr["events"].mean()), 1)
        r["living_neurons_per_window_mean"] = round(float(arr["living"].mean()), 1)
        r["neuron_steps_per_window"] = round(work["steps"] / nw, 1)
        r["decision_gmac_per_window"] = round((work["tokens"] * (embed + audit) + work["steps"] * per_step) / nw / 1e9, 3)
        r["decision_gmac_per_window_embed_once"] = round(
            (work["new_tokens"] * embed + work["tokens"] * audit + work["steps"] * per_step) / nw / 1e9, 3)
        r["real_time_factor_mean"] = round(float(total.mean()) / float(model.window_ms), 4)
        r["windows_over_budget"] = round(float((total > model.window_ms).mean()), 4)
        res[mode] = r
    with open(os.path.join(out, "runtime.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def fit_range_run(args, run_root, log_name):
    """tools/deco/fit_range.py on an export (official four metrics); returns its last line."""
    cmd = [sys.executable, os.path.join(HERE, "fit_range.py"), "--run", os.path.join(run_root, "run"),
           "--evidence", os.path.join(run_root, "evm"), "--z0-root", run_root, "--data-root", args.data_root]
    path = os.path.join(args.out, log_name)
    with open(path, "w") as fh:
        code = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    lines = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]
    return lines[-1] if (code == 0 and lines) else "fit_range FAILED (see %s)" % log_name


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/evisseg_stream_tif_comm_peak.yaml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--k", type=int, default=16, help="TBPTT chunk (windows), as the backbone")
    ap.add_argument("--streams", type=int, default=4, help="parallel training streams, as the backbone")
    ap.add_argument("--past", type=int, default=10)
    ap.add_argument("--deadline", type=int, default=10)
    ap.add_argument("--k-near", type=int, default=8)
    ap.add_argument("--k-evid", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.01, help="risk level of the pending band (strata, birth band, "
                    "fixed-band commit of the model selection)")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--samples", type=int, default=1024, help="neurons per chunk in the step terms of the loss")
    ap.add_argument("--strata", type=int, choices=(4, 6), default=6, help="sampling strata: label x {uncertain, "
                    "confident right, confident wrong} (6) or label x {uncertain, confident} (4)")
    ap.add_argument("--lr-small", type=float, default=None, help="default: the config's lr (to its lr_end)")
    ap.add_argument("--lr-big", type=float, default=2e-3, help="decision neurons and feedback (to a tenth)")
    ap.add_argument("--feedback", choices=("innov", "belief", "none"), default="innov")
    ap.add_argument("--birth", choices=("none", "band"), default="none")
    ap.add_argument("--value", choices=("signed", "ei"), default="signed")
    ap.add_argument("--gate", choices=("sigmoid", "softmax", "compete"), default="sigmoid")
    ap.add_argument("--rho", choices=("mean", "learn", "sum"), default="mean")
    ap.add_argument("--audit", choices=("reaudit", "ledger"), default="reaudit")
    ap.add_argument("--balance", choices=("on", "off"), default="on", help="class-balanced log loss (logit adjusted)")
    ap.add_argument("--prior-grad", choices=("on", "off"), default="off",
                    help="off (default, 11-09 local checks): the step terms train the decision neurons only; the small "
                    "network (and the feedback convolution) is trained by the birth terms = its own prior loss, so the "
                    "two stages stay coupled through the forward path and the feedback loop but not through gradients; "
                    "on (ablation): the step terms also reach the small network through the neighbours' priors and the "
                    "neurons' contexts")
    ap.add_argument("--own-prior-grad", choices=("on", "off"), default="off",
                    help="on: the step terms also move the neuron's own prior additively (U = z0 + evidence); off "
                    "(default): the prior is the birth term's, the evidence is fitted on top of it, the decision loss "
                    "reaches the small network through the neighbours' priors and the neurons' contexts")
    ap.add_argument("--init-small", default=None, help="ablation: start the small network from a backbone checkpoint")
    ap.add_argument("--init-model", default=None, help="start the whole network from a V12 checkpoint (last.pt / best.pt)")
    ap.add_argument("--freeze-small", action="store_true", help="diagnostic: train the decision neurons only (small "
                    "network and feedback convolution fixed)")
    ap.add_argument("--sub-batch", type=int, default=2048, help="neurons per step call (memory only)")
    ap.add_argument("--pick-block", type=int, default=2048, help="queries per candidate block (memory only)")
    ap.add_argument("--eval-streams", type=int, default=4, help="recordings streamed in parallel by the export")
    ap.add_argument("--eval-chunk", type=int, default=32)
    ap.add_argument("--max-chunks", type=int, default=0, help="smoke tests: chunks per epoch")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: recordings per split in the evaluation")
    ap.add_argument("--runtime", type=int, default=24, help="test recordings for the runtime measurement (0 = skip)")
    ap.add_argument("--fit-range", choices=("on", "off"), default="on", help="run tools/deco/fit_range.py at the end")
    ap.add_argument("--gated", choices=("on", "off"), default="on", help="also export / score the gated deployment "
                    "(a neuron stops computing once its membrane leaves the alpha band)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--profile", action="store_true", help="synchronised per-phase timing and peak memory per chunk")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    T.seed_everything(args.seed, False)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    model, cfg, q99, stats = build(args, device)
    lr_small = float(args.lr_small or cfg["lr"])
    lr_small_end = lr_small * float(cfg["lr_end"]) / float(cfg["lr"])
    last_path = os.path.join(args.out, "last.pt")
    resuming = args.resume and os.path.exists(last_path)
    if args.init_model:
        model.load_state_dict(torch.load(args.init_model, map_location=device)["model"])
    elif args.init_small:
        ck = torch.load(args.init_small, map_location=device)
        model.small.load_state_dict(ck["model"])
    elif not resuming:
        T.run_calibration(model.small, cfg, q99, device, args.out)
    if args.freeze_small:
        for p_ in list(model.small.parameters()) + (list(model.fb.parameters()) if model.fb is not None else []):
            p_.requires_grad_(False)
    groups = [{"params": list(model.small.parameters()), "lr": lr_small, "weight_decay": 0.0},
              {"params": list(model.big.parameters()) + (list(model.fb.parameters()) if model.fb is not None else []),
               "lr": args.lr_big, "weight_decay": 1e-4}]
    optimizer = torch.optim.AdamW(groups)
    comm = None if args.freeze_small else T.CommBudget.from_config(cfg, model.small)
    hist, best, best_ep, start = [], -1.0, -1, 0
    if resuming:
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        hist, best, best_ep, start = ck["history"], ck["best"], ck["best_epoch"], ck["epoch"] + 1
        if comm is not None and ck.get("mu") is not None:
            comm.mu = float(ck["mu"])
        print("resumed after epoch %d (best %.4f at %d)" % (start - 1, best, best_ep), flush=True)
    if not bool(model.small.gain_calibrated):
        raise RuntimeError("small network gain not calibrated")
    n_small = sum(p.numel() for p in model.small.parameters())
    n_big = sum(p.numel() for p in model.big.parameters()) + (sum(p.numel() for p in model.fb.parameters())
                                                              if model.fb is not None else 0)
    print("DeCo-SNN V12 | feedback %s birth %s | neuron %s/%s/%s/%s dim %d | P %d D %d K %d+%d | tau %.3f band (%.2f, "
          "%.2f) | balance %s prior-grad %s | k %d streams %d samples %d | params small %d big %d | seed %d" % (
              args.feedback, args.birth, args.value, args.gate, args.rho, args.audit, args.dim, model.P, model.D,
              model.k_near, model.k_evid, model.tau, model.band[0], model.band[1], args.balance, args.prior_grad,
              args.k, args.streams, args.samples, n_small, n_big, args.seed), flush=True)
    train_set = EvUAVStream(cfg["root"], "train", cfg, T.train_file_names(cfg))
    lengths = [int(train_set[i].n_windows) for i in range(len(train_set))]
    t0 = time.time()
    for ep in range(start, args.epochs):
        optimizer.param_groups[0]["lr"] = linear_epoch_lr(ep, args.epochs, lr_small, lr_small_end)
        optimizer.param_groups[1]["lr"] = linear_epoch_lr(ep, args.epochs, args.lr_big, args.lr_big * 0.1)
        rng = np.random.RandomState(args.seed * 1000 + ep + 11)
        rec = train_epoch(model, cfg, q99, train_set, lengths, optimizer, comm, device, rng, args)
        val = export_split(model, cfg, q99, "val", device, args, limit=args.limit)
        rec.update(epoch=ep, val=val, seconds=round(time.time() - t0, 1))
        hist.append(rec)
        cm = rec.get("comm") or {}
        print("ep %d | loss %.4f (prior %.4f step %.4f) | neurons/chunk %.0f | mu %s | val crossing %.4f fixed %.4f "
              "arrival %.4f prior %.4f wait %.2f | %.0f s" % (
                  ep, rec["loss"], rec["prior"], rec["step"], rec["neurons_per_chunk"], cm.get("mu_end", "-"),
                  val["crossing_iou"], val["fixed_iou"], val["arrival_iou"], val["prior_iou"], val["mean_wait"],
                  time.time() - t0), flush=True)
        if val["crossing_iou"] > best:
            best, best_ep = val["crossing_iou"], ep
            torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(args.out, "best.pt"))
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": ep, "history": hist,
                    "best": best, "best_epoch": best_ep, "mu": comm.mu if comm is not None else None}, last_path)
    if os.path.exists(os.path.join(args.out, "best.pt")):
        model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=device)["model"])
    torch.save({"model": model.state_dict(), "args": vars(args), "config": cfg, "stats": stats, "best_epoch": best_ep},
               os.path.join(args.out, "deco_v12.pt"))
    summary = {"args": vars(args), "best_epoch": best_ep, "history": hist, "v11": model.big.learned_constants()}
    for split in ("val", "test"):
        summary[split] = export_split(model, cfg, q99, split, device, args, out=args.out, limit=args.limit)
        print("export %s: crossing %.4f fixed %.4f arrival %.4f prior %.4f wait %.2f" % (
            split, summary[split]["crossing_iou"], summary[split]["fixed_iou"], summary[split]["arrival_iou"],
            summary[split]["prior_iou"], summary[split]["mean_wait"]), flush=True)
    if args.gated == "on":                          # deployment mode: a neuron stops once it has fired
        gdir = os.path.join(args.out, "gated")
        for split in ("val", "test"):
            summary["gated_" + split] = export_split(model, cfg, q99, split, device, args, out=gdir, limit=args.limit,
                                                     gate=model.band)
            print("gated export %s: crossing %.4f fixed %.4f | work/window %s" % (
                split, summary["gated_" + split]["crossing_iou"], summary["gated_" + split]["fixed_iou"],
                summary["gated_" + split]["work_per_window"]), flush=True)
    if args.runtime:
        summary["runtime"] = runtime(model, cfg, q99, device, args, args.out)
        print("runtime: %s" % json.dumps(summary["runtime"]), flush=True)
    summary["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    rng_line, gated_line = "-", "-"
    if args.fit_range == "on":
        rng_line = fit_range_run(args, args.out, "range.log")
        if args.gated == "on":
            gated_line = fit_range_run(args, os.path.join(args.out, "gated"), "range_gated.log")
    rt = summary.get("runtime", {})
    print("V12 FINISHED best ep %d | val crossing %.4f fixed %.4f | test crossing %.4f fixed %.4f arrival %.4f prior %.4f "
          "| runtime full %s gated %s ms/window | %.0f s" % (
              best_ep, summary["val"]["crossing_iou"], summary["val"]["fixed_iou"], summary["test"]["crossing_iou"],
              summary["test"]["fixed_iou"], summary["test"]["arrival_iou"], summary["test"]["prior_iou"],
              rt.get("full", {}).get("total_ms", {}).get("mean", "-"), rt.get("gated", {}).get("total_ms", {}).get(
                  "mean", "-"), summary["seconds"]), flush=True)
    print("GATED (fixed row = the gated deployment): %s" % gated_line, flush=True)
    print(rng_line, flush=True)


if __name__ == "__main__":
    main()
