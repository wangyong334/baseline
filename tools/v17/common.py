"""V17 shared helpers: recordings with the frozen backbone's priors, margin-stratified sampling, evidence features of
chosen events, membrane trajectories of whole splits, and the official IoU / ACC / Pd / Fa at the report threshold."""
import glob
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.sdu import EvidenceMaps, box_layout  # noqa: E402,F401

# Plain fp32 matrix products: torch < 1.12 uses TF32 on Ampere / Ada GPUs by default, which made the stream replay
# differ from the offline trajectories by up to 2e-2 on the server (11-13).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

THRESHOLD = 0.9
SENSOR = (260, 346)                       # EV-UAV (height, width)


def logit(p):
    return math.log(p / (1.0 - p))


def tau_of(log_w):
    """Report threshold 0.9 on the calibrated scale (z = backbone logit - log pos_weight)."""
    return logit(THRESHOLD) - float(log_w)


def load_split(z0_root, data_root, split, n_steps, limit=0):
    """Recordings of a split: positions, times, polarity, labels, target ids and the calibrated prior z0 of every event.
    F = last decision step that exists (k <= n_steps - 1 and the window k after the event's own window exists)."""
    z_dir = os.path.join(z0_root, "z0_" + split)
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(z_dir, "*.npz")))
    if limit:
        names = names[:int(limit)]
    if not names:
        raise RuntimeError("no prior files in %s" % z_dir)
    recs = []
    for name in names:
        with np.load(os.path.join(z_dir, name)) as tr:
            z0 = np.asarray(tr["traj"][:, 0], np.float32)
            log_w, wms = float(tr["log_w"]), int(tr["window_ms"])
        with np.load(os.path.join(data_root, split, name)) as d:
            loc, norm = np.asarray(d["ev_loc"]), np.asarray(d["evs_norm"])
        if loc.shape[0] != z0.shape[0]:
            raise RuntimeError("%s: %d priors for %d events" % (name, z0.shape[0], loc.shape[0]))
        t = loc[:, 2].astype(np.int64)
        win = t // wms
        n_win = int(win.max()) + 1
        recs.append({"name": name, "x": loc[:, 0].astype(np.int32), "y": loc[:, 1].astype(np.int32),
                     "t": t.astype(np.int32), "win": win.astype(np.int32), "pol": (norm[:, 3] > 0.5).astype(np.int8),
                     "lab": norm[:, 4] > 0.5, "tid": norm[:, 5].astype(np.float32),
                     "z0": np.clip(np.nan_to_num(z0, nan=-20.0), -20.0, 20.0), "n_win": n_win, "log_w": log_w,
                     "window_ms": wms, "F": np.minimum(n_steps - 1, n_win - 1 - win).astype(np.int16)})
    return recs


def local_paths(cfg, z0_root=None, data_root=None):
    """Run config with the prior / data dirs of this machine (a run trained on the server, read locally)."""
    cfg = dict(cfg)
    if z0_root:
        cfg["z0_root"] = z0_root
    if data_root:
        cfg["data_root"] = data_root
    return cfg


def margin_bin(z0, tau, width):
    return np.floor((np.asarray(z0, np.float64) - tau) / float(width)).astype(np.int64)


def sampling_probs(recs, tau, cap, width):
    """Margin-stratified importance sampling (11-10 plan): target events always, background events of every margin bin
    (width logit units, aligned to tau) with probability min(1, cap / count of the bin). Training with weight 1 / p is
    in expectation the same as training on every event; cap only sets the compute."""
    allb = np.concatenate([margin_bin(r["z0"], tau, width)[~r["lab"]] for r in recs])
    keys, counts = np.unique(allb, return_counts=True)
    p_keys = np.minimum(1.0, float(cap) / np.maximum(counts, 1))
    probs = []
    for r in recs:
        b = margin_bin(r["z0"], tau, width)
        pos = np.minimum(np.searchsorted(keys, b), max(len(keys) - 1, 0))
        pr = np.where(keys[pos] == b, p_keys[pos], 1.0) if len(keys) else np.ones(b.shape)
        probs.append(np.where(r["lab"], 1.0, pr))
    return probs, {int(k): [int(c), float(p)] for k, c, p in zip(keys, counts, p_keys)}


def rec_maps(rec, device, sensor=SENSOR):
    return EvidenceMaps(rec["x"], rec["y"], rec["win"], rec["pol"], rec["z0"], rec["n_win"], sensor[0], sensor[1],
                        device)


def event_features(maps, rec, idx, rect, n_steps):
    """Features [n, S, K, N_FEAT] and step mask [n, S] of the events idx of one recording."""
    idx = np.asarray(idx, np.int64)
    return maps.features(rec["x"][idx].astype(np.int64), rec["y"][idx].astype(np.int64),
                         rec["win"][idx].astype(np.int64), rec["pol"][idx].astype(np.int64), rec["z0"][idx], rect,
                         n_steps)


def release(device):
    """Return cached GPU blocks: the per-recording maps and chunks differ in size, so the cache fragments otherwise
    (a laptop GPU then spills into shared memory and slows down several times)."""
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def box_counts(feat):
    """Per event and step: boxes that hold events (nz) and the events inside the unit's square (ne), from features
    [n, S, K, N_FEAT] (log1p counts; zero past the recording end)."""
    n_box = torch.round(torch.expm1(feat[..., 0].float()) + torch.expm1(feat[..., 1].float()))
    return (n_box > 0).sum(-1).to(torch.int16), n_box.sum(-1).to(torch.int32)


def trajectories(net, recs, rect, device, chunk=4096, sensor=SENSOR, counts=False):
    """Membrane U [n_events, S] of every event of every recording (decision units run for all events; the stopping
    rule later decides which ones a deployment would actually run). counts=True also returns box_counts (lists of
    nz, ne [n_events, S]) for the event-driven cost."""
    net.eval()
    out, nzs, nes = [], [], []
    with torch.no_grad():
        for rec in recs:
            maps = rec_maps(rec, device, sensor)
            n = rec["z0"].shape[0]
            U = np.zeros((n, net.n_steps), np.float32)
            nz, ne = np.zeros((n, net.n_steps), np.int16), np.zeros((n, net.n_steps), np.int32)
            for a in range(0, n, chunk):
                idx = np.arange(a, min(a + chunk, n))
                feat, ok = event_features(maps, rec, idx, rect, net.n_steps)
                z0 = torch.as_tensor(rec["z0"][idx], device=maps.device)
                pol = torch.as_tensor(rec["pol"][idx].astype(np.int64), device=maps.device)
                U[idx] = net(feat, ok, z0, pol).float().cpu().numpy()
                if counts:
                    nz[idx], ne[idx] = (t.cpu().numpy() for t in box_counts(feat))
            out.append(U)
            nzs.append(nz)
            nes.append(ne)
            del maps
            release(device)
    return (out, nzs, nes) if counts else out


def split_counts(recs, rect, n_steps, device, chunk=8192, sensor=SENSOR):
    """box_counts of every event of a split without a network (runs whose trajectories have no counts)."""
    nzs, nes = [], []
    with torch.no_grad():
        for rec in recs:
            maps = rec_maps(rec, device, sensor)
            n = rec["z0"].shape[0]
            nz, ne = np.zeros((n, n_steps), np.int16), np.zeros((n, n_steps), np.int32)
            for a in range(0, n, chunk):
                idx = np.arange(a, min(a + chunk, n))
                feat, _ = event_features(maps, rec, idx, rect, n_steps)
                nz[idx], ne[idx] = (t.cpu().numpy() for t in box_counts(feat))
            nzs.append(nz)
            nes.append(ne)
            del maps
            release(device)
    return np.concatenate(nzs), np.concatenate(nes)


def report_prob(z, log_w):
    """Calibrated logit -> report-scale probability in float32 (as the official evaluation)."""
    z32 = np.asarray(z, dtype=np.float32) + np.float32(log_w)
    return torch.sigmoid(torch.from_numpy(np.ascontiguousarray(z32))).numpy()


def official(recs, values, sensor=SENSOR):
    """IoU / ACC (target-event recall) / Pd / Fa at the report threshold 0.9: micro counts over all events and the
    FastRoc port of utils/eval.py (50 ms frames, correct_thresh 1e-4)."""
    from utils.stream_metrics import FastRoc
    roc = FastRoc(50, 1e-4, int(sensor[0]), int(sensor[1]))
    c = np.zeros(3, np.int64)
    for rec, z in zip(recs, values):
        prob = report_prob(z, rec["log_w"])
        pred = prob >= np.float32(THRESHOLD)
        pos = rec["lab"]
        c += [int((pred & pos).sum()), int((pred & ~pos).sum()), int((~pred & pos).sum())]
        roc.update(rec["t"].astype(np.int64), prob, rec["tid"].astype(np.float64), pos.astype(np.float32),
                   rec["x"].astype(np.int64), rec["y"].astype(np.int64), THRESHOLD)
    pd, fa = roc.result()
    tp, fp, fn = (int(v) for v in c)
    return {"iou": tp / float(max(tp + fp + fn, 1)), "acc": tp / float(max(tp + fn, 1)), "pd": float(pd),
            "fa": float(fa), "tp": tp, "fp": fp, "fn": fn}


def micro_iou(recs, values, tau):
    c = np.zeros(3, np.int64)
    for rec, z in zip(recs, values):
        pred, pos = np.asarray(z) >= tau, rec["lab"]
        c += [int((pred & pos).sum()), int((pred & ~pos).sum()), int((~pred & pos).sum())]
    return float(c[0]) / float(max(c.sum(), 1))
