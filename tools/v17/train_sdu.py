"""V17 step 1: train the spiking decision units (model/sdu.py) on a frozen backbone's priors.

Every training step draws events by margin-stratified importance sampling (tools/v17/common.sampling_probs: all target
events, background capped per margin bin, weight 1 / p), so the units see every margin and the loss equals in
expectation the loss on all events -- no band. Loss: BCE of the membrane at every existing step (a strictly proper
score: each U(k) is trained towards the posterior given what has arrived by step k). The epoch is picked on val by
the deadline IoU of all events (independent of the stopping rule, which is calibrated afterwards). Exports the
membrane trajectories of every val / test event (traj_<split>.npz) for tools/v17/eval_sdu.py.
Defaults = the main model (11-13 evening, l_glm): integrate-to-bound units (audit ledger) that read every box with a
context-shifted linear model of its statistics (tokens glm), distance attribution, context without feed-forward, d 32.
--cascade-from <run>: cascade fine-tuning -- start from that run's units and train only on the events its calibrated
entry rule sends to a unit (|z0 - tau| < b0 of <run>/eval/eval.json), the events the units receive at inference; the
epoch (or the start, epoch -1) is picked by the val IoU of entry + deadline (outside the entry the prior commits).
Usage: python tools/v17/train_sdu.py --z0-root <dir with z0_train / z0_val / z0_test> --data-root <root> --out <dir>
Last line: "SDU TRAIN FINISHED best ep ... | val IoU arrival ... step0 ... deadline ... | test ...".
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from model.sdu import N_FEAT, SpikingDecisionUnits, box_layout  # noqa: E402


def build_net(cfg, device):
    rect, r, _ = box_layout(cfg["levels"])
    net = SpikingDecisionUnits(cfg["n_steps"], r, dim=cfg["dim"], heads=cfg["heads"], value=cfg["value"],
                               gate=cfg["gate"], pool=cfg["pool"], evidence=cfg.get("evidence", "boxes"),
                               context=cfg["context"] == "on", audit=cfg["audit"], tokens=cfg.get("tokens", "mlp"),
                               ctx_ff=cfg.get("context_ff", "on") == "on").to(device)
    return net, rect


def normalization(recs, probs, rect, n_steps, device, seed, per_rec=400):
    """Training statistics of the token features (valid windows) and of the prior, from a fixed thinned sample."""
    rng = np.random.RandomState(seed + 7)
    s1, s2, cnt, zs = np.zeros(N_FEAT), np.zeros(N_FEAT), 0.0, []
    for rec, p in zip(recs, probs):
        idx = np.nonzero(rng.rand(len(p)) < p)[0]
        if idx.size > per_rec:
            idx = rng.choice(idx, per_rec, replace=False)
        if not idx.size:
            continue
        maps = C.rec_maps(rec, device)
        feat, ok = C.event_features(maps, rec, idx, rect, n_steps)
        x = feat[ok].reshape(-1, N_FEAT).double().cpu().numpy()
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
        cnt += x.shape[0]
        zs.append(rec["z0"][idx])
        del maps
    mu = s1 / cnt
    z = np.concatenate(zs)
    return mu, np.sqrt(np.maximum(s2 / cnt - mu * mu, 1e-8)), float(z.mean()), float(z.std())


def train_epoch(net, opt, recs, probs, rect, n_steps, device, rng, batch, pool_events, max_batches=0):
    net.train()
    order = rng.permutation(len(recs))
    losses, n_batches, n_events = [], 0, 0
    i = 0
    while i < len(order):
        parts = []
        total = 0
        while i < len(order) and total < pool_events:              # a pool of recordings, at least pool_events events
            rec, p = recs[order[i]], probs[order[i]]
            i += 1
            idx = np.nonzero(rng.rand(len(p)) < p)[0]
            if not idx.size:
                continue
            maps = C.rec_maps(rec, device)
            for a in range(0, idx.size, 8192):
                sub = idx[a:a + 8192]
                feat, ok = C.event_features(maps, rec, sub, rect, n_steps)
                parts.append((feat.half(), ok, torch.as_tensor(rec["z0"][sub], device=device),
                              torch.as_tensor(rec["pol"][sub].astype(np.int64), device=device),
                              torch.as_tensor(rec["lab"][sub].astype(np.float32), device=device),
                              torch.as_tensor(1.0 / p[sub], dtype=torch.float32, device=device)))
            total += idx.size
            del maps
        if not parts:
            continue
        feat, ok, z0, pol, y, w = (torch.cat([q[j] for q in parts]) for j in range(6))
        del parts
        C.release(device)
        perm = torch.as_tensor(rng.permutation(feat.shape[0]), dtype=torch.long, device=device)
        for b in range(0, feat.shape[0], batch):
            sel = perm[b:b + batch]
            U = net(feat[sel].float(), ok[sel], z0[sel], pol[sel])
            vm = ok[sel].float()
            ell = F.binary_cross_entropy_with_logits(U, y[sel][:, None].expand_as(U), reduction="none") * vm
            loss = (ell.sum(1) * w[sel]).sum() / (vm.sum(1) * w[sel]).sum().clamp(min=1e-6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            n_batches += 1
            if max_batches and n_batches >= max_batches:
                return float(np.mean(losses)), n_batches, n_events + int(sel.numel())
        n_events += int(feat.shape[0])
    return float(np.mean(losses)) if losses else float("nan"), n_batches, n_events


def split_scores(recs, U, tau, b0=None):
    """Micro IoU of the arrival prior, of step 0 for every event and of the deadline for every event; with an entry
    bound b0 also of entry + deadline (the prior commits where |z0 - tau| >= b0, the unit's deadline elsewhere)."""
    last = [U[i][np.arange(len(r["F"])), r["F"].astype(np.int64)] for i, r in enumerate(recs)]
    out = {"arrival": C.micro_iou(recs, [r["z0"] for r in recs], tau),
           "step0": C.micro_iou(recs, [u[:, 0] for u in U], tau),
           "deadline": C.micro_iou(recs, last, tau)}
    if b0 is not None:
        out["entry_deadline"] = C.micro_iou(recs, [np.where(np.abs(r["z0"] - tau) < b0, d, r["z0"])
                                                   for r, d in zip(recs, last)], tau)
    return out


def save_traj(path, recs, U, nz=None, ne=None):
    extra = {} if nz is None else {"nz": np.concatenate(nz), "ne": np.concatenate(ne)}
    np.savez(path, U=np.concatenate(U).astype(np.float32), F=np.concatenate([r["F"] for r in recs]),
             start=np.cumsum([0] + [len(r["F"]) for r in recs]), names=np.array([r["name"] for r in recs]), **extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--z0-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--deadline-ms", type=float, default=500.0, help="latency requirement: last decision step")
    ap.add_argument("--levels", type=int, default=4, help="foveated box levels (side 3^(levels+1) px)")
    ap.add_argument("--dim", type=int, default=32, help="width of the context (rounds 1-3: 64)")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--value", choices=("ei", "signed"), default="ei")
    ap.add_argument("--gate", choices=("attribution", "distance"), default="distance",
                    help="attribution of a box: distance only (main) or also its content (rounds 1-3)")
    ap.add_argument("--pool", choices=("window", "mean", "sum"), default="window",
                    help="window: attributed mean inside a window, windows add up (default); mean: one mean over all "
                         "windows (11-13 round 1); sum: every token independent")
    ap.add_argument("--evidence", choices=("boxes", "events"), default="events",
                    help="observation unit inside a window: the event (count-weighted + pseudo-observation, main model "
                         "since 11-13 round 2) or the box")
    ap.add_argument("--context", choices=("on", "off"), default="on")
    ap.add_argument("--context-ff", choices=("on", "off"), default="off",
                    help="feed-forward block of the context update (off: one linear update; rounds 1-3: on)")
    ap.add_argument("--audit", choices=("reaudit", "ledger"), default="ledger",
                    help="ledger: each window read once, its evidence added (integrate-to-bound, main); reaudit: every "
                         "step re-reads all windows so far (rounds 1-4)")
    ap.add_argument("--tokens", choices=("mlp", "linear", "glm"), default="glm",
                    help="how a unit reads a box: glm = context-shifted linear model of the box statistics (main), "
                         "mlp = LayerNorm embedding (rounds 1-3), linear = embedding linear in the features")
    ap.add_argument("--cascade-from", default="",
                    help="cascade fine-tuning from this run dir (needs its eval/eval.json): only the events its entry "
                         "rule sends to a unit; the model options are taken from that run")
    ap.add_argument("--sample-cap", type=int, default=50000,
                    help="compute budget: background events per margin bin and epoch (unbiased through 1 / p)")
    ap.add_argument("--bin-width", type=float, default=1.0, help="margin bin width of the sampling (logit units)")
    ap.add_argument("--pool-events", type=int, default=262144,
                    help="events gathered (from about 37 recordings) before they are shuffled into batches")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="first N recordings of every split (smoke tests)")
    ap.add_argument("--max-batches", type=int, default=0, help="stop every epoch after N batches (smoke tests)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    probe = C.load_split(args.z0_root, args.data_root, "val", 1, limit=1)[0]
    wms = probe["window_ms"]
    n_steps = int(round(args.deadline_ms / wms)) + 1
    t0 = time.time()
    data = {s: C.load_split(args.z0_root, args.data_root, s, n_steps, args.limit) for s in ("train", "val", "test")}
    tau = C.tau_of(probe["log_w"])
    if any(abs(C.tau_of(r["log_w"]) - tau) > 1e-6 for s in data for r in data[s]):
        raise RuntimeError("recordings with different pos_weight")
    b0 = None
    if args.cascade_from:                                      # model options and weights of the source run
        src = torch.load(os.path.join(args.cascade_from, "sdu.pt"), map_location="cpu")
        with open(os.path.join(args.cascade_from, "eval", "eval.json")) as fh:
            b0 = json.load(fh)["bounds"][0]
        b0 = float("inf") if b0 is None else float(b0)
        cfg = dict(src["config"])
        if cfg["n_steps"] != n_steps or abs(cfg["tau"] - tau) > 1e-6:
            raise RuntimeError("the source run has other steps or another threshold")
        cfg.update(cascade_from=os.path.abspath(args.cascade_from), cascade_b0=b0)
    else:
        cfg = {k: getattr(args, k) for k in ("levels", "dim", "heads", "value", "gate", "pool", "evidence", "context",
                                             "context_ff", "audit", "tokens")}
        cfg.update(n_steps=n_steps, window_ms=wms, tau=tau)
    cfg.update(z0_root=os.path.abspath(args.z0_root), data_root=os.path.abspath(args.data_root), seed=args.seed)
    net, rect = build_net(cfg, dev)
    if b0 is None:
        probs, bins = C.sampling_probs(data["train"], tau, args.sample_cap, args.bin_width)
    else:                                                      # the events the entry rule sends to a unit
        probs = [(np.abs(r["z0"] - tau) < b0).astype(np.float64) for r in data["train"]]
        bins = {"cascade_entry_bound": b0}
    n_train = sum(len(p) for p in probs)
    exp_sample = sum(float(p.sum()) for p in probs)
    print("events: train %d (expected %.0f sampled per epoch), val %d, test %d; %d steps (window %d ms, deadline %.0f "
          "ms); %d boxes; %d parameters (%.0f s)" % (
              n_train, exp_sample, sum(len(r["F"]) for r in data["val"]), sum(len(r["F"]) for r in data["test"]),
              n_steps, wms, args.deadline_ms, len(rect), sum(p.numel() for p in net.parameters()), time.time() - t0),
          flush=True)
    if b0 is None:
        mu, sd, z_mu, z_sd = normalization(data["train"], probs, rect, n_steps, dev, args.seed)
        net.set_normalization(mu, sd, z_mu, z_sd)
        print("feature mean %s std %s; prior mean %.3f std %.3f" % (np.round(mu, 3), np.round(sd, 3), z_mu, z_sd),
              flush=True)
    else:
        net.load_state_dict(src["model"])                      # weights and normalisation of the source run
        print("cascade from %s: entry bound %.4f, %d training events" % (args.cascade_from, b0, int(exp_sample)),
              flush=True)
    key = "deadline" if b0 is None else "entry_deadline"       # model selection on val
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = np.random.RandomState(args.seed)
    best, best_state, best_ep, hist = -1.0, None, -1, []
    if b0 is not None:                                         # the source units compete as epoch -1
        sc = split_scores(data["val"], C.trajectories(net, data["val"], rect, dev), tau, b0)
        best = sc[key]
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        hist.append(dict(epoch=-1, **{k: round(v, 5) for k, v in sc.items()}))
        print("ep -1 (source) | val IoU arrival %.4f deadline %.4f entry+deadline %.4f" % (
            sc["arrival"], sc["deadline"], sc["entry_deadline"]), flush=True)
    for ep in range(args.epochs):
        te = time.time()
        loss, nb, ne = train_epoch(net, opt, data["train"], probs, rect, n_steps, dev, rng, args.batch,
                                   args.pool_events, args.max_batches)
        sched.step()
        tt = time.time() - te
        U_val = C.trajectories(net, data["val"], rect, dev)
        sc = split_scores(data["val"], U_val, tau, b0)
        hist.append(dict(epoch=ep, loss=loss, batches=nb, events=ne, train_s=round(tt, 1),
                         epoch_s=round(time.time() - te, 1), **{k: round(v, 5) for k, v in sc.items()}))
        print("ep %d loss %.4f (%d batches, %d events, %.0f s) | val IoU arrival %.4f step0 %.4f deadline %.4f%s (%.0f s)"
              % (ep, loss, nb, ne, tt, sc["arrival"], sc["step0"], sc["deadline"],
                 "" if b0 is None else " entry+deadline %.4f" % sc["entry_deadline"], time.time() - te), flush=True)
        if sc[key] > best:
            best, best_ep = sc[key], ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    torch.save({"model": best_state, "config": cfg, "args": vars(args), "best_epoch": best_ep},
               os.path.join(args.out, "sdu.pt"))
    summary = {"config": cfg, "args": vars(args), "best_epoch": best_ep, "history": hist, "sampling_bins": bins,
               "learned": net.learned_constants(), "parameters": int(sum(p.numel() for p in net.parameters())),
               "seconds": round(time.time() - t0, 1)}
    for split in ("val", "test"):
        U, nz, ne = C.trajectories(net, data[split], rect, dev, counts=True)
        save_traj(os.path.join(args.out, "traj_%s.npz" % split), data[split], U, nz, ne)
        summary[split] = split_scores(data[split], U, tau, b0)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    v, t = summary["val"], summary["test"]
    print("SDU TRAIN FINISHED best ep %d | val IoU arrival %.4f step0 %.4f deadline %.4f | test IoU arrival %.4f step0 "
          "%.4f deadline %.4f | %.0f s" % (best_ep, v["arrival"], v["step0"], v["deadline"], t["arrival"], t["step0"],
                                            t["deadline"], summary["seconds"]), flush=True)


if __name__ == "__main__":
    main()
