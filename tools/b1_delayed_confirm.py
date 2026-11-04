"""B1 control for module three: non-learned delayed confirmation on V5's outputs (FINAL_PLAN_20261104.md §4.4).

The classic M-of-N confirmation (radar track confirmation / temporal hysteresis), per event, same cells (4x4 px) and
the same future budget as RCL, nothing learned:
    Pos_t(c)  = 1 if some event of window t in cell c has V5 report probability >= 0.9
    N_e(d)    = number of windows s+1..s+d whose Pos, dilated by r cells (motion), is 1 at the event's cell
    decision at lag d (fixed delay):
        V5 positive -> keep if N_e(d) >= m_keep, else 0
        V5 negative -> 1 if p_e >= p_low and N_e(d) >= m_rescue, else 0
r, m_keep, m_rescue, p_low and the lag d are chosen on val (best IoU per lag); test is reported once with the val
choice. Reads the lag-0 column of the RCL trajectories (= V5's calibrated logit) and the dataset NPZ files.
Usage:
    python tools/b1_delayed_confirm.py --val-traj <dir> --test-traj <dir> --data-root <root> --window-ms 50 --out <json>
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import final_analysis as FA  # noqa: E402
from utils.recording_groups import dense_names  # noqa: E402

CELL = 4
RADII = (1, 2, 4)
LAGS = (1, 3, 5, 10)
M_KEEP = (0, 1, 2, 3)
M_RESCUE = (1, 2, 3, 5, 7, 9, 10 ** 6)
P_LOW = (0.3, 0.5, 0.7)


def persistence(rec, prob, window_ms, radius, lags):
    """{lag d: N_e(d)} for every event (int16)."""
    win = rec["t"] // int(window_ms)
    T = int(win.max()) + 1
    cy, cx = rec["y"] // CELL, rec["x"] // CELL
    hc, wc = int(cy.max()) + 1, int(cx.max()) + 1
    pos = np.zeros((T, hc, wc), dtype=bool)
    hit = prob >= np.float32(FA.THRESHOLD)
    pos[win[hit], cy[hit], cx[hit]] = True
    if radius > 0:                                   # box dilation by `radius` cells (integral image)
        c = np.zeros((T, hc + 1, wc + 1), dtype=np.int32)
        c[:, 1:, 1:] = pos.astype(np.int32).cumsum(1).cumsum(2)
        y0 = np.clip(np.arange(hc) - radius, 0, hc)
        y1 = np.clip(np.arange(hc) + radius + 1, 0, hc)
        x0 = np.clip(np.arange(wc) - radius, 0, wc)
        x1 = np.clip(np.arange(wc) + radius + 1, 0, wc)
        s = (c[:, y1][:, :, x1] - c[:, y0][:, :, x1] - c[:, y1][:, :, x0] + c[:, y0][:, :, x0])
        pos = s > 0
    cum = np.concatenate([np.zeros((1, hc, wc), dtype=np.int32), pos.astype(np.int32).cumsum(0)])  # cum[t] = sum_{<t}
    out = {}
    for d in lags:
        hi = np.minimum(win + 1 + int(d), T)         # windows s+1 .. s+d (clipped at the end of the recording)
        out[d] = (cum[hi, cy, cx] - cum[win + 1, cy, cx]).astype(np.int16)
    return out


def decide(prob, n, m_keep, m_rescue, p_low):
    base = prob >= np.float32(FA.THRESHOLD)
    return np.where(base, n >= m_keep, (prob >= np.float32(p_low)) & (n >= m_rescue))


def grid_counts(recs, window_ms):
    """{(r, d, m_keep, m_rescue, p_low): [TP, FP, FN]} over a split."""
    table = {}
    for rec in recs:
        prob = FA.report_prob(rec["traj"][:, 0], rec["log_w"])
        pos = rec["pos"]
        for r in RADII:
            per = persistence(rec, prob, window_ms, r, LAGS)
            for d in LAGS:
                n = per[d]
                for mk, mr, pl in itertools.product(M_KEEP, M_RESCUE, P_LOW):
                    if mk > d or (mr > d and mr < 10 ** 6):
                        continue
                    pred = decide(prob, n, mk, mr, pl)
                    key = (r, d, mk, mr, pl)
                    table[key] = table.get(key, np.zeros(3, dtype=np.int64)) + FA.counts(pred, pos)
    return table


def evaluate(recs, dense, window_ms, cfg, name):
    r, d, mk, mr, pl = cfg
    stats = FA.PolicyStats(name, window_ms)
    tau_cal = FA._logit(FA.THRESHOLD) - recs[0]["log_w"]
    for rec in recs:
        prob = FA.report_prob(rec["traj"][:, 0], rec["log_w"])
        ref = FA.references(rec, tau_cal, window_ms)
        n = persistence(rec, prob, window_ms, r, (d,))[d]
        out = decide(prob, n, mk, mr, pl).astype(np.float32)
        lag = np.minimum(int(d), int(ref["win"].max()) - ref["win"])
        stats.add(rec, out, lag, ref, None, rec["name"] in dense)
    return stats.result()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--val-traj", required=True)
    parser.add_argument("--test-traj", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--window-ms", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--width", type=int, default=346)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    dense = dense_names(args.data_root, ("val", "test"), args.height, args.width)
    val = FA.load_split(args.val_traj, args.data_root, "val")
    table = grid_counts(val, args.window_ms)
    best = {}
    for key, c in table.items():
        d = key[1]
        if d not in best or FA.iou_of(c) > FA.iou_of(table[best[d]]):
            best[d] = key
    report = {"grid": {"radii": RADII, "lags": LAGS, "m_keep": M_KEEP, "m_rescue": M_RESCUE, "p_low": P_LOW},
              "chosen_on_val": {}}
    test = FA.load_split(args.test_traj, args.data_root, "test")
    for d in sorted(best):
        cfg = best[d]
        cfg_d = dict(zip(("radius_cells", "lag", "m_keep", "m_rescue", "p_low"), cfg))
        report["chosen_on_val"][str(d)] = {"config": cfg_d,
                                           "val": evaluate(val, set(dense["val"]), args.window_ms, cfg, "b1_val"),
                                           "test": evaluate(test, set(dense["test"]), args.window_ms, cfg, "b1_test")}
        v, t = report["chosen_on_val"][str(d)]["val"], report["chosen_on_val"][str(d)]["test"]
        print("B1 等 %d 窗（val 选的 %s）：val IoU %.4f（普通 %.4f / 极密 %s）| test IoU %.4f ACC %.4f Pd %.4f Fa %.3e"
              "（普通 %.4f / 极密 %s）| 第一拍 %.3f / %.3f" % (
                  d, cfg_d, v["iou"], v["groups"]["normal"], "%.4f" % v["groups"]["dense"],
                  t["iou"], t["acc"], t["pd"], t["fa"], t["groups"]["normal"], "%.4f" % t["groups"]["dense"],
                  v["first_beat_recall"], t["first_beat_recall"]), flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False, default=float)
    print("B1 FINISHED:", args.out, flush=True)


if __name__ == "__main__":
    main()
