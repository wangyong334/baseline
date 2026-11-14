"""V17 step 4: does the decision layer behave the way its principle says? (analysis only, no training)

From a trained run (sdu.pt, traj_<split>.npz, eval/eval.json) on one split (default test), under the calibrated
stopping rule:
  1. receptive fields: learned sigma (px) and bias of every head (target / background channel)
  2. who gets a unit and how long it waits, by event type -- target onset (the first 100 ms of its track), later target,
     background within 10 px of a target event of the same window, other background -- and by recording group (dense
     shot-noise recordings vs the others, utils/recording_groups.py)
  3. what the units change against the backbone, by event type: rescued / suppressed targets, removed / added false
     alarms
  4. evidence over the steps: mean signed evidence of the target and background channels and the attributed mass, for
     units on targets and on background
  5. where the attribution goes at the commit step: mean gate per box level and window offset, per channel
Writes analysis_<split>.json and figures under <run>/eval.
Usage: python tools/v17/analyze_sdu.py --run <dir> [--split test] [--max-units 20000]
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
from model.sdu import box_layout  # noqa: E402
from train_sdu import build_net  # noqa: E402
from utils.recording_groups import dense_names  # noqa: E402

TYPES = ("target_onset", "target_later", "background_near_target", "background_other")


def event_types(rec, near_px=10, onset_windows=2):
    """0 target onset, 1 later target, 2 background near a target event of the same window, 3 other background."""
    lab, win, tid = rec["lab"], rec["win"].astype(np.int64), rec["tid"]
    typ = np.full(len(lab), 3, np.int64)
    if lab.any():
        first = {}
        for g in np.unique(tid[lab]):
            first[g] = int(win[lab & (tid == g)].min())
        onset = np.array([win[i] - first.get(tid[i], 0) < onset_windows for i in np.nonzero(lab)[0]])
        typ[np.nonzero(lab)[0]] = np.where(onset, 0, 1)
        H, W = C.SENSOR
        grid = np.zeros((rec["n_win"], H + 1, W + 1), np.int32)
        np.add.at(grid, (win[lab], rec["y"][lab] + 1, rec["x"][lab] + 1), 1)
        grid = grid.cumsum(1).cumsum(2)
        bg = np.nonzero(~lab)[0]
        x, y, w = rec["x"][bg].astype(np.int64), rec["y"][bg].astype(np.int64), win[bg]
        x0, x1 = np.clip(x - near_px, 0, W), np.clip(x + near_px + 1, 0, W)
        y0, y1 = np.clip(y - near_px, 0, H), np.clip(y + near_px + 1, 0, H)
        n = grid[w, y1, x1] - grid[w, y0, x1] - grid[w, y1, x0] + grid[w, y0, x0]
        typ[bg[n > 0]] = 2
    return typ


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-units", type=int, default=20000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--z0-root", default=None, help="override the prior dir stored in the run (another machine)")
    ap.add_argument("--data-root", default=None, help="override the data root stored in the run")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(os.path.join(args.run, "sdu.pt"), map_location="cpu")
    cfg = C.local_paths(ck["config"], args.z0_root, args.data_root)
    net, rect = build_net(cfg, dev)
    net.load_state_dict(ck["model"])
    net.eval()
    with open(os.path.join(args.run, "eval", "eval.json")) as fh:
        bounds = [float("inf") if b is None else float(b) for b in json.load(fh)["bounds"]]
    recs = C.load_split(cfg["z0_root"], cfg["data_root"], args.split, cfg["n_steps"], int(ck["args"].get("limit", 0)))
    tr = np.load(os.path.join(args.run, "traj_%s.npz" % args.split))
    tau, S = cfg["tau"], cfg["n_steps"]
    msgs = C.split_messages(cfg, args.run, args.split, recs, dev)      # V18: decision spikes of the first population
    dense = set(dense_names(cfg["data_root"], (args.split,))[args.split])
    res = {"learned": net.learned_constants(), "bounds": bounds, "dense_recordings": sorted(dense)}
    stats = {g: {t: np.zeros(8) for t in TYPES} for g in ("normal", "dense")}
    picks = []
    for ri, rec in enumerate(recs):
        a, b = int(tr["start"][ri]), int(tr["start"][ri + 1])
        Z = np.concatenate([rec["z0"][:, None].astype(np.float64), tr["U"][a:b].astype(np.float64)], 1)
        last = 1 + tr["F"][a:b].astype(np.int64)
        L = P.commit(Z, last, tau, bounds)
        typ = event_types(rec)
        base = Z[:, 0] >= tau
        final = Z[np.arange(len(L)), L] >= tau
        born = L >= 1
        wait = np.maximum(L - 1, 0)
        lab = rec["lab"]
        g = "dense" if rec["name"] in dense else "normal"
        for ti, tname in enumerate(TYPES):
            s = typ == ti
            stats[g][tname] += [s.sum(), (born & s).sum(), wait[born & s].sum(), (lab & ~base & final & s).sum(),
                                (lab & base & ~final & s).sum(), (~lab & base & ~final & s).sum(),
                                (~lab & ~base & final & s).sum(), (base != final)[s].sum()]
        for e in np.nonzero(born)[0]:
            picks.append((ri, e, L[e], typ[e]))
    out = {}
    for g in stats:
        out[g] = {}
        for tname, v in stats[g].items():
            n, nb = int(v[0]), int(v[1])
            out[g][tname] = {"events": n, "unit_frac": nb / float(max(n, 1)), "wait_windows_of_units": v[2] / max(nb, 1),
                             "rescued": int(v[3]), "suppressed": int(v[4]), "fa_removed": int(v[5]),
                             "fa_added": int(v[6]), "changed_vs_backbone": int(v[7])}
    res["by_type"] = out
    rng = np.random.RandomState(0)
    if len(picks) > args.max_units:
        picks = [picks[i] for i in np.sort(rng.choice(len(picks), args.max_units, replace=False))]
    _, _, lev = box_layout(cfg["levels"])
    n_lev, K = int(lev.max()) + 1, len(lev)
    sign = net.head_sign.cpu().numpy()
    ev = {k: np.zeros((S, 2)) for k in ("target_units", "background_units")}
    mass = {k: np.zeros(S) for k in ev}
    cnt = {k: np.zeros(S) for k in ev}
    gate = {k: np.zeros((2, n_lev, S)) for k in ev}
    gcnt = {k: np.zeros((n_lev, S)) for k in ev}
    by_rec = {}
    for ri, e, L, ty in picks:
        by_rec.setdefault(ri, []).append((e, L, ty))
    with torch.no_grad():
        for ri, items in by_rec.items():
            rec = recs[ri]
            msg = None if msgs is None else msgs[ri]
            idx = np.array([e for e, _, _ in items])
            for maps, rs, mg, local, sel in C.segmented(rec, idx, S, dev, msg, C.needs_moments(net)):
                feat, ok, cue = C.event_features(maps, rs, local, rect, S, mg, net)
                net(feat, ok, torch.as_tensor(rs["z0"][local], device=dev),
                    torch.as_tensor(rs["pol"][local].astype(np.int64), device=dev), keep=True, memb=cue)
                lam = net.extras["lam"].cpu().numpy()                 # [n, S, H] signed
                ms = net.extras["mass"].cpu().numpy()                 # [n, S, H]
                gt = net.extras["gates"].cpu().numpy().reshape(len(local), S, -1, S, K)   # [n, step, H, window, box]
                okn = ok.cpu().numpy()
                for j, (e, L, ty) in enumerate(items[i] for i in sel):
                    key = "target_units" if ty < 2 else "background_units"
                    v = okn[j]
                    ev[key][v, 0] += lam[j, v][:, sign > 0].sum(1)
                    ev[key][v, 1] += lam[j, v][:, sign < 0].sum(1)
                    mass[key][v] += ms[j, v].sum(1)
                    cnt[key][v] += 1
                    k = L - 1                                          # commit step
                    for ch, hs in enumerate((sign > 0, sign < 0)):
                        gk = gt[j, k][hs].mean(0)                      # [window, box]
                        for lv in range(n_lev):
                            gate[key][ch, lv, :k + 1] += gk[:k + 1, lev == lv].mean(1)
                    gcnt[key][:, :k + 1] += 1
                del maps
    res["evidence_by_step"] = {k: {"target_channel": (ev[k][:, 0] / np.maximum(cnt[k], 1)).round(4).tolist(),
                                   "background_channel": (ev[k][:, 1] / np.maximum(cnt[k], 1)).round(4).tolist(),
                                   "attributed_mass": (mass[k] / np.maximum(cnt[k], 1)).round(3).tolist(),
                                   "units": cnt[k].astype(int).tolist()} for k in ev}
    res["gate_at_commit"] = {k: {ch: (gate[k][i] / np.maximum(gcnt[k], 1)).round(4).tolist()
                                 for i, ch in enumerate(("target_channel", "background_channel"))} for k in gate}
    os.makedirs(os.path.join(args.run, "eval"), exist_ok=True)
    with open(os.path.join(args.run, "eval", "analysis_%s.json" % args.split), "w") as fh:
        json.dump(res, fh, indent=1)
    try:
        figures(res, os.path.join(args.run, "eval"), args.split, cfg["window_ms"])
    except Exception as exc:                                           # figures are optional
        print("figures skipped: %s" % exc)
    print("learned receptive fields (px) %s, biases %s" % (res["learned"]["sigma_px"], res["learned"]["rf_bias"]))
    for g in out:
        for tname, r in out[g].items():
            print("%-6s %-24s events %8d  units %6.2f%%  wait %.2f  rescued %5d suppressed %5d  FA removed %5d added %5d"
                  % (g, tname, r["events"], 100 * r["unit_frac"], r["wait_windows_of_units"], r["rescued"],
                     r["suppressed"], r["fa_removed"], r["fa_added"]), flush=True)
    print("SDU ANALYSIS FINISHED %s: %d units analysed" % (args.split, len(picks)), flush=True)


def figures(res, out_dir, split, wms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, key, title in zip(axes, ("target_units", "background_units"), ("units on targets", "units on background")):
        r = res["evidence_by_step"][key]
        steps = np.arange(len(r["units"])) * wms
        ax.plot(steps, r["target_channel"], label="target channel (+)")
        ax.plot(steps, r["background_channel"], label="background channel (-)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(title)
        ax.set_xlabel("time after the event's window (ms)")
    axes[0].set_ylabel("mean evidence (log-odds)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "evidence_%s.png" % split), dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5))
    for i, key in enumerate(("target_units", "background_units")):
        for j, ch in enumerate(("target_channel", "background_channel")):
            g = np.asarray(res["gate_at_commit"][key][ch])
            im = axes[i, j].imshow(g, aspect="auto", cmap="viridis", vmin=0, vmax=max(g.max(), 1e-3))
            axes[i, j].set_title("%s, %s" % (key.replace("_", " "), ch.replace("_", " ")), fontsize=9)
            axes[i, j].set_ylabel("box level")
            axes[i, j].set_xlabel("window offset")
            fig.colorbar(im, ax=axes[i, j])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gates_%s.png" % split), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
