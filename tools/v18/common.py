"""V18 shared helpers: recordings with the frozen backbone's priors, margin-stratified sampling, evidence features of
chosen events, membrane trajectories of whole splits, the official IoU / ACC / Pd / Fa at the report threshold, and
the decision spikes of a trained first population of units (V18: the second population reads them)."""
import glob
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.sdu import N_FEAT, N_MSG, TRACK_BOXES, EvidenceMaps, SpikingDecisionUnits, box_layout  # noqa: E402,F401
import stopping as P  # noqa: E402

MSG_KEYS = ("own_step", "own_type", "mx", "my", "mwin", "mtype")

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


def build_net(cfg, device):
    rect, r, _ = box_layout(cfg["levels"])
    net = SpikingDecisionUnits(cfg["n_steps"], r, dim=cfg["dim"], heads=cfg["heads"], value=cfg["value"],
                               gate=cfg["gate"], pool=cfg["pool"], evidence=cfg.get("evidence", "boxes"),
                               context=cfg["context"] == "on", audit=cfg["audit"], tokens=cfg.get("tokens", "mlp"),
                               ctx_ff=cfg.get("context_ff", "on") == "on",
                               n_feat=cfg.get("n_feat", N_FEAT), n_msg=cfg.get("n_msg", 0),
                               track={"off": False, "on": "recentre"}.get(cfg.get("track", "off"), cfg.get("track")),
                               memb=cfg.get("membership", "off") == "on",
                               pseudo=cfg.get("pseudo", "on") == "on", dendrite=cfg.get("dendrite", "mean"),
                               ring=cfg.get("ring", "off") == "on", clutter=cfg.get("clutter", "off") == "on",
                               content=cfg.get("content", "box")).to(device)
    return net, rect


def rec_maps(rec, device, sensor=SENSOR, msg=None, track=False):
    """Evidence maps of one recording; msg = its first-population decision spikes (V18) or None; track = tracked
    fields (V18: the boxes follow the source)."""
    messages = None if msg is None else (msg["mx"], msg["my"], msg["mwin"], msg["mtype"])
    return EvidenceMaps(rec["x"], rec["y"], rec["win"], rec["pol"], rec["z0"], rec["n_win"], sensor[0], sensor[1],
                        device, messages=messages, track=track)


def event_features(maps, rec, idx, rect, n_steps, msg=None, net=None):
    """Features [n, S, K, n_feat] and step mask [n, S] of the events idx of one recording (with msg: the own decision
    spike of each event is left out). With net: the fields the units read (fixed or tracked) and a third output, the
    membership cue [n, 2] (None for units without it)."""
    idx = np.asarray(idx, np.int64)
    own = None if msg is None else (msg["own_step"][idx], msg["own_type"][idx])
    args = (rec["x"][idx].astype(np.int64), rec["y"][idx].astype(np.int64), rec["win"][idx].astype(np.int64),
            rec["pol"][idx].astype(np.int64), rec["z0"][idx], rect, n_steps)
    if net is None:
        return maps.features(*args, own_msg=own)
    if not (net.track or net.memb):
        feat, ok = maps.features(*args, own_msg=own)
        return feat, ok, None
    feat, ok, cue = maps.features(*args, own_msg=own, follow=net.track_mode if net.track else False,
                                  return_memb=True)
    return feat, ok, (cue if net.memb else None)


def needs_moments(net):
    """Maps of units with tracked fields or the membership cue hold the prior-weighted moments."""
    return bool(getattr(net, "track", False) or getattr(net, "memb", False))


def load_run(run, z0_root=None, data_root=None, device="cpu"):
    """Trained units of a run: net (eval mode), rect, config (with this machine's paths) and calibrated bounds."""
    ck = torch.load(os.path.join(run, "sdu.pt"), map_location="cpu")
    cfg = local_paths(ck["config"], z0_root, data_root)
    net, rect = build_net(cfg, device)
    net.load_state_dict(ck["model"])
    net.eval()
    with open(os.path.join(run, "eval", "eval.json")) as fh:
        bounds = [float("inf") if b is None else float(b) for b in json.load(fh)["bounds"]]
    return net, rect, cfg, bounds


def first_population_messages(run1, recs, device, z0_root=None, data_root=None):
    """Decision spikes of a trained first population (V17 units that read the evidence only) on recordings recs, as a
    deployment produces them: an event with a unit commits at column L >= 1 (calibrated bounds of run1) and emits one
    spike in window win + L - 1 at its pixel, type 1 = target. Per recording: own_step (-1: no unit), own_type and the
    spikes (mx, my, mwin, mtype)."""
    net1, rect1, cfg1, bounds1 = load_run(run1, z0_root, data_root, device)
    if cfg1.get("n_feat", N_FEAT) != N_FEAT or cfg1.get("n_msg", 0):
        raise ValueError("the first population must read the evidence only (%s)" % run1)
    U = trajectories(net1, recs, rect1, device)
    tau = cfg1["tau"]
    out = []
    for rec, u in zip(recs, U):
        if u.shape[1] != cfg1["n_steps"]:
            raise ValueError("first population with %d steps" % u.shape[1])
        Z = np.concatenate([rec["z0"][:, None].astype(np.float64), u.astype(np.float64)], 1)
        L = P.commit(Z, 1 + rec["F"].astype(np.int64), tau, bounds1)
        typ = (Z[np.arange(len(L)), L] >= tau).astype(np.int8)
        step = np.where(L >= 1, L - 1, -1).astype(np.int16)
        sel = np.nonzero(L >= 1)[0]
        out.append({"own_step": step, "own_type": typ, "mx": rec["x"][sel].astype(np.int32),
                    "my": rec["y"][sel].astype(np.int32),
                    "mwin": (rec["win"][sel].astype(np.int64) + step[sel]).astype(np.int32), "mtype": typ[sel]})
    del net1
    release(device)
    return out, {"run": os.path.abspath(run1), "bounds": bounds1, "tau": tau}


def save_messages(path, msgs):
    ev = np.cumsum([0] + [len(m["own_step"]) for m in msgs])
    sp = np.cumsum([0] + [len(m["mx"]) for m in msgs])
    np.savez(path, ev_start=ev, sp_start=sp, **{k: np.concatenate([m[k] for m in msgs]) for k in MSG_KEYS})


def load_messages(path):
    d = np.load(path)
    out = []
    for i in range(len(d["ev_start"]) - 1):
        a, b, c, e = d["ev_start"][i], d["ev_start"][i + 1], d["sp_start"][i], d["sp_start"][i + 1]
        out.append({k: (d[k][a:b] if k.startswith("own") else d[k][c:e]) for k in MSG_KEYS})
    return out


def split_messages(cfg, run, split, recs, device):
    """First-population spikes of a split for a V18 run (cached as <run>/messages_<split>.npz); None for V17 runs."""
    if not cfg.get("stage1"):
        return None
    path = os.path.join(run, "messages_%s.npz" % split)
    if os.path.exists(path):
        msgs = load_messages(path)
        if len(msgs) == len(recs) and all(len(m["own_step"]) == len(r["z0"]) for m, r in zip(msgs, recs)):
            return msgs
    msgs, _ = first_population_messages(cfg["stage1"], recs, device, cfg.get("z0_root"), cfg.get("data_root"))
    save_messages(path, msgs)
    return msgs


def release(device):
    """Return cached GPU blocks: the per-recording maps and chunks differ in size, so the cache fragments otherwise
    (a laptop GPU then spills into shared memory and slows down several times)."""
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def box_counts(feat):
    """Per event and step: boxes that hold events (nz), the events inside the unit's square (ne) and inside its
    level-0/1 boxes (ne1, what a tracked field's centroid reads), from features [n, S, K, N_FEAT] (log1p counts; zero
    past the recording end)."""
    n_box = torch.round(torch.expm1(feat[..., 0].float()) + torch.expm1(feat[..., 1].float()))
    return ((n_box > 0).sum(-1).to(torch.int16), n_box.sum(-1).to(torch.int32),
            n_box[..., :TRACK_BOXES].sum(-1).to(torch.int32))


def trajectories(net, recs, rect, device, chunk=4096, sensor=SENSOR, counts=False, msgs=None):
    """Membrane U [n_events, S] of every event of every recording (decision units run for all events; the stopping
    rule later decides which ones a deployment would actually run). counts=True also returns box_counts (lists of
    nz, ne, ne1 [n_events, S]) for the event-driven cost. msgs: first-population spikes per recording (V18)."""
    net.eval()
    out, cnt = [], ([], [], [])
    with torch.no_grad():
        for ri, rec in enumerate(recs):
            msg = None if msgs is None else msgs[ri]
            maps = rec_maps(rec, device, sensor, msg, track=needs_moments(net))
            n = rec["z0"].shape[0]
            U = np.zeros((n, net.n_steps), np.float32)
            c = (np.zeros((n, net.n_steps), np.int16), np.zeros((n, net.n_steps), np.int32),
                 np.zeros((n, net.n_steps), np.int32))
            for a in range(0, n, chunk):
                idx = np.arange(a, min(a + chunk, n))
                feat, ok, cue = event_features(maps, rec, idx, rect, net.n_steps, msg, net)
                z0 = torch.as_tensor(rec["z0"][idx], device=maps.device)
                pol = torch.as_tensor(rec["pol"][idx].astype(np.int64), device=maps.device)
                U[idx] = net(feat, ok, z0, pol, memb=cue).float().cpu().numpy()
                if counts:
                    for arr, t in zip(c, box_counts(feat)):
                        arr[idx] = t.cpu().numpy()
            out.append(U)
            for lst, arr in zip(cnt, c):
                lst.append(arr)
            del maps
            release(device)
    return (out,) + cnt if counts else out


def split_counts(recs, rect, n_steps, device, chunk=8192, sensor=SENSOR, track=False):
    """box_counts (nz, ne, ne1) of every event of a split without a network (runs whose trajectories have none)."""
    cnt = ([], [], [])
    with torch.no_grad():
        for rec in recs:
            maps = rec_maps(rec, device, sensor, track=track)
            n = rec["z0"].shape[0]
            c = (np.zeros((n, n_steps), np.int16), np.zeros((n, n_steps), np.int32), np.zeros((n, n_steps), np.int32))
            for a in range(0, n, chunk):
                idx = np.arange(a, min(a + chunk, n))
                feat, _ = event_features(maps, rec, idx, rect, n_steps)
                for arr, t in zip(c, box_counts(feat)):
                    arr[idx] = t.cpu().numpy()
            for lst, arr in zip(cnt, c):
                lst.append(arr)
            del maps
            release(device)
    return tuple(np.concatenate(lst) for lst in cnt)


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
