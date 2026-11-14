"""V18 analysis A1: how much of the backbone's error can waiting remove?

The backbone commits every event when its window closes. Its errors at arrival (misses: target events below the
threshold; false alarms: background events above it) are split into
    unreachable   the prior is so far from the threshold that the entry rule commits it (no decision unit)
    corrected     the event gets a unit and its committed label is right
    not corrected the event gets a unit but its committed label is still wrong
and the units' new errors (backbone right, committed label wrong) are counted as well. For the reachable errors the
share that is right after k windows (every unit waiting to step k, no stopping) traces how evidence accumulates.
Model-free view (labels only): for every error, are there target events within R px in the following windows? A
missed target that continues nearby can be recovered by waiting; a false alarm with no target nearby later is noise
that does not persist. Border zone: events whose own window holds labelled target AND background events within +-2 px
-- the border of an annotated box, where waiting brings no deciding evidence; its share among the errors that remain
shows how much of the residual error is spatial rather than temporal. By event type (target onset, later target,
background near a target, other background).
By target speed (labelled centroid of the target id, window w -> w + 1, px per window): target events with a unit,
backbone and committed miss rate, share of the backbone's misses corrected (fixed boxes lose fast targets).
By kind: a backbone miss is existence-type when the backbone finds < 30% of that target's events in the window (the
target as a whole is missed), membership-type at >= 70% (a found target, a missed event); a false alarm near a target
by the share of the nearest target found. Waiting can confirm that a target exists, not which border events are
its own.
Usage: python tools/v18/reducible.py --run <dir> [--split test] [--radius 13]
Writes <run>/eval/reducible_<split>.json (+ .png). Last line: "REDUCIBLE FINISHED ...".
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import stopping as P  # noqa: E402
from analyze_sdu import TYPES, event_types  # noqa: E402

LOOK = (1, 2, 5)                                       # following windows looked at in the model-free view
SPEED = (0.0, 1.0, 2.0, 4.0, 8.0, np.inf)              # target speed bins (px per window)
KIND = (0.3, 0.7)                                      # existence < 0.3 <= mixed < 0.7 <= membership (share found)
KINDS = ("existence", "mixed", "membership")


def future_targets(rec, idx, radius, look):
    """Labelled target events within radius px of events idx in windows win + 1 .. win + look (per look)."""
    H, W = C.SENSOR
    lab, win = rec["lab"], rec["win"].astype(np.int64)
    grid = np.zeros((rec["n_win"] + 1, H + 1, W + 1), np.int32)
    np.add.at(grid, (win[lab] + 1, rec["y"][lab] + 1, rec["x"][lab] + 1), 1)
    grid = grid.cumsum(0).cumsum(1).cumsum(2)          # prefix over windows and space
    x, y, w = rec["x"][idx].astype(np.int64), rec["y"][idx].astype(np.int64), win[idx]
    x0, x1 = np.clip(x - radius, 0, W), np.clip(x + radius + 1, 0, W)
    y0, y1 = np.clip(y - radius, 0, H), np.clip(y + radius + 1, 0, H)

    def box(t):
        return grid[t, y1, x1] - grid[t, y0, x1] - grid[t, y1, x0] + grid[t, y0, x0]

    out = {}
    for k in look:
        t_hi = np.minimum(w + 1 + k, rec["n_win"])
        out[k] = box(t_hi) - box(np.minimum(w + 1, rec["n_win"]))
    return out


def border_zone(rec, half=2):
    """Events whose own window holds, within +-half px, both labelled target and labelled background events (the
    event itself left out): the border of an annotated target box, where the label is decided by where the box was
    drawn rather than by evidence that waiting could bring."""
    H, W = C.SENSOR
    lab, win = rec["lab"], rec["win"].astype(np.int64)
    x, y = rec["x"].astype(np.int64), rec["y"].astype(np.int64)
    tg = np.zeros((rec["n_win"], H + 1, W + 1), np.int32)
    al = np.zeros_like(tg)
    np.add.at(tg, (win[lab], y[lab] + 1, x[lab] + 1), 1)
    np.add.at(al, (win, y + 1, x + 1), 1)
    tg, al = tg.cumsum(1).cumsum(2), al.cumsum(1).cumsum(2)
    x0, x1 = np.clip(x - half, 0, W), np.clip(x + half + 1, 0, W)
    y0, y1 = np.clip(y - half, 0, H), np.clip(y + half + 1, 0, H)

    def box(g):
        return g[win, y1, x1] - g[win, y0, x1] - g[win, y1, x0] + g[win, y0, x0]

    n_t = box(tg) - lab.astype(np.int64)
    n_b = box(al) - box(tg) - (~lab).astype(np.int64)
    return (n_t > 0) & (n_b > 0)


def target_frames(rec, found):
    """Per (target id, window): labelled centroid and the share of its events the backbone finds."""
    lab = rec["lab"]
    key = np.round(rec["tid"][lab]).astype(np.int64) * 100000 + rec["win"][lab].astype(np.int64)
    u, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    cx = np.bincount(inv, rec["x"][lab].astype(np.float64)) / n
    cy = np.bincount(inv, rec["y"][lab].astype(np.float64)) / n
    fr = np.bincount(inv, found[lab].astype(np.float64)) / n
    return {int(k): (a, b, f) for k, a, b, f in zip(u, cx, cy, fr)}


def kind_of(share):
    return 0 if share < KIND[0] else (2 if share >= KIND[1] else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--radius", type=int, default=13, help="model-free view: px around the event (level-1 boxes)")
    ap.add_argument("--z0-root", default=None)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    ck = torch.load(os.path.join(args.run, "sdu.pt"), map_location="cpu")
    cfg = C.local_paths(ck["config"], args.z0_root, args.data_root)
    with open(os.path.join(args.run, "eval", "eval.json")) as fh:
        bounds = [float("inf") if b is None else float(b) for b in json.load(fh)["bounds"]]
    recs = C.load_split(cfg["z0_root"], cfg["data_root"], args.split, cfg["n_steps"], int(ck["args"].get("limit", 0)))
    tr = np.load(os.path.join(args.run, "traj_%s.npz" % args.split))
    tau, S = cfg["tau"], cfg["n_steps"]
    groups = ("all",) + TYPES
    cnt = {g: {k: 0 for k in ("events", "errors", "misses", "false_alarms", "unreachable", "corrected",
                              "not_corrected", "new_errors")} for g in groups}
    step_ok = np.zeros(S)
    n_reach = 0
    free = {kind: {k: 0 for k in LOOK} for kind in ("misses", "false_alarms")}
    n_kind = {"misses": 0, "false_alarms": 0}
    zones = ("all_events", "backbone_errors", "unreachable", "corrected", "not_corrected", "new_errors")
    bz = {z: [0, 0] for z in zones}                        # [events, of them in the border zone]
    nsp = len(SPEED) - 1
    spd = np.zeros((nsp, 4))                               # target units: events, backbone misses, committed, corrected
    kinds = {k: np.zeros((3, 3)) for k in ("misses", "false_alarms_near_target")}   # [kind] corrected, not, unreach
    for ri, rec in enumerate(recs):
        a, b = int(tr["start"][ri]), int(tr["start"][ri + 1])
        U = tr["U"][a:b].astype(np.float64)
        F = tr["F"][a:b].astype(np.int64)
        Z = np.concatenate([rec["z0"][:, None].astype(np.float64), U], 1)
        L = P.commit(Z, 1 + F, tau, bounds)
        lab = rec["lab"]
        base = Z[:, 0] >= tau
        final = Z[np.arange(len(L)), L] >= tau
        err = base != lab
        reach = np.abs(Z[:, 0] - tau) < bounds[0]
        typ = event_types(rec)
        for gi, g in enumerate(groups):
            s = np.ones(len(lab), bool) if g == "all" else typ == gi - 1
            c = cnt[g]
            c["events"] += int(s.sum())
            c["errors"] += int((err & s).sum())
            c["misses"] += int((err & lab & s).sum())
            c["false_alarms"] += int((err & ~lab & s).sum())
            c["unreachable"] += int((err & ~reach & s).sum())
            c["corrected"] += int((err & reach & (final == lab) & s).sum())
            c["not_corrected"] += int((err & reach & (final != lab) & s).sum())
            c["new_errors"] += int((~err & (final != lab) & s).sum())
        if lab.any():                                      # by target speed and by kind of error
            fr = target_frames(rec, base)
            key = np.round(rec["tid"]).astype(np.int64) * 100000 + rec["win"].astype(np.int64)
            for i in np.nonzero(lab & reach)[0]:
                a_, b_ = fr.get(int(key[i])), fr.get(int(key[i]) + 1)
                if a_ is None or b_ is None:
                    continue
                j = int(np.searchsorted(SPEED, np.hypot(b_[0] - a_[0], b_[1] - a_[1]), side="right") - 1)
                spd[j] += [1, not base[i], not final[i], (not base[i]) and final[i]]
            outcome = np.where(reach, np.where(final == lab, 0, 1), 2)
            for i in np.nonzero(err & lab)[0]:
                kinds["misses"][kind_of(fr[int(key[i])][2]), outcome[i]] += 1
            ti = np.nonzero(lab)[0]
            for i in np.nonzero(err & ~lab & (typ == 2))[0]:
                m = ti[rec["win"][ti] == rec["win"][i]]
                if m.size:
                    k_ = m[int(np.argmin(np.hypot(rec["x"][m] - rec["x"][i], rec["y"][m] - rec["y"][i])))]
                    kinds["false_alarms_near_target"][kind_of(fr[int(key[k_])][2]), outcome[i]] += 1
        bord = border_zone(rec)
        for z, m in (("all_events", np.ones(len(lab), bool)), ("backbone_errors", err), ("unreachable", err & ~reach),
                     ("corrected", err & reach & (final == lab)), ("not_corrected", err & reach & (final != lab)),
                     ("new_errors", ~err & (final != lab))):
            bz[z][0] += int(m.sum())
            bz[z][1] += int((m & bord).sum())
        e = np.nonzero(err & reach)[0]
        if e.size:
            kk = np.minimum(np.arange(S)[None, :], F[e, None])
            step_ok += ((U[e[:, None], kk] >= tau) == lab[e, None]).sum(0)
            n_reach += e.size
        for kind, sel in (("misses", err & lab), ("false_alarms", err & ~lab)):
            idx = np.nonzero(sel)[0]
            if not idx.size:
                continue
            ft = future_targets(rec, idx, args.radius, LOOK)
            n_kind[kind] += idx.size
            for k in LOOK:
                free[kind][k] += int((ft[k] > 0).sum())
    res = {"split": args.split, "radius_px": args.radius, "bounds": bounds, "by_type": cnt,
           "reachable_errors": n_reach,
           "reachable_errors_right_by_step": (step_ok / max(n_reach, 1)).round(4).tolist(),
           "model_free": {kind: {"events": n_kind[kind],
                                 "target_nearby_in_next_windows": {str(k): free[kind][k] / float(max(n_kind[kind], 1))
                                                                   for k in LOOK}} for kind in free},
           "border_zone_share": {z: {"events": v[0], "share": v[1] / float(max(v[0], 1))} for z, v in bz.items()},
           "target_units_by_speed": [{"speed_px_per_window": [SPEED[j], None if np.isinf(SPEED[j + 1]) else
                                                              SPEED[j + 1]],
                                      "events": int(spd[j, 0]), "backbone_miss": spd[j, 1] / max(spd[j, 0], 1),
                                      "committed_miss": spd[j, 2] / max(spd[j, 0], 1),
                                      "misses_corrected": spd[j, 3] / max(spd[j, 1], 1)} for j in range(nsp)],
           "errors_by_kind": {g: {KINDS[j]: {"events": int(M[j].sum()), "corrected": M[j, 0] / max(M[j].sum(), 1),
                                             "not_corrected": M[j, 1] / max(M[j].sum(), 1),
                                             "unreachable": M[j, 2] / max(M[j].sum(), 1)} for j in range(3)}
                              for g, M in kinds.items()}}
    out = os.path.join(args.run, "eval")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "reducible_%s.json" % args.split), "w") as fh:
        json.dump(res, fh, indent=1)
    try:
        figure(res, os.path.join(out, "reducible_%s.png" % args.split))
    except Exception as exc:                            # the figure is optional
        print("figure skipped: %s" % exc)
    a = cnt["all"]
    for g in groups:
        c = cnt[g]
        print("%-24s events %8d errors %6d (miss %5d, FA %5d) | unreachable %5d corrected %5d not corrected %5d | "
              "new errors %5d" % (g, c["events"], c["errors"], c["misses"], c["false_alarms"], c["unreachable"],
                                  c["corrected"], c["not_corrected"], c["new_errors"]), flush=True)
    print("reachable errors right after k windows: %s" % " ".join("%.3f" % v for v in res["reachable_errors_right_by_step"]))
    print("border zone (labelled target and background within +-2 px in the own window), share of: " + ", ".join(
        "%s %.3f" % (z, res["border_zone_share"][z]["share"]) for z in zones))
    for r in res["target_units_by_speed"]:
        if r["events"]:
            print("target units, speed [%g, %s) px/window: %6d events | backbone miss %.4f -> committed %.4f | misses "
                  "corrected %.3f" % (r["speed_px_per_window"][0], r["speed_px_per_window"][1] or "inf", r["events"],
                                      r["backbone_miss"], r["committed_miss"], r["misses_corrected"]))
    for g, d in res["errors_by_kind"].items():
        print("%s by kind: %s" % (g, " | ".join("%s %d (corrected %.3f, not %.3f, unreachable %.3f)" % (
            k, v["events"], v["corrected"], v["not_corrected"], v["unreachable"]) for k, v in d.items())))
    for kind in free:
        print("model-free %-12s target within %d px in the next 1 / 2 / 5 windows: %s" % (
            kind, args.radius, " / ".join("%.3f" % res["model_free"][kind]["target_nearby_in_next_windows"][str(k)]
                                          for k in LOOK)))
    print("REDUCIBLE FINISHED %s: backbone errors %d = unreachable %.3f + corrected %.3f + not corrected %.3f; new "
          "errors %d" % (args.split, a["errors"], a["unreachable"] / max(a["errors"], 1), a["corrected"] /
                         max(a["errors"], 1), a["not_corrected"] / max(a["errors"], 1), a["new_errors"]), flush=True)


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    names = list(res["by_type"].keys())[1:]
    parts = ("corrected", "not_corrected", "unreachable")
    bottom = np.zeros(len(names))
    for p in parts:
        v = np.array([res["by_type"][n][p] for n in names], float)
        ax[0].bar(range(len(names)), v, bottom=bottom, label=p.replace("_", " "))
        bottom += v
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax[0].set_ylabel("backbone errors (events)")
    ax[0].legend(fontsize=8)
    r = res["reachable_errors_right_by_step"]
    ax[1].plot(range(len(r)), r, marker="o")
    ax[1].set_xlabel("windows waited")
    ax[1].set_ylabel("reachable errors right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
