"""Read-only probe (11-08): are arrival errors caused by where an event sits inside its window, or by the window being
too short? Uses only right / wrong at the official threshold (no uncertainty band needed). CPU, numpy only.
  A. per backbone (one per window length): arrival IoU / recall / false-positive rate per fifth of the window
     (phase = position of the event inside its own window: 0-20 % = window start ... 80-100 % = window end);
  B. events the reference backbone (e.g. 50 ms) gets wrong at arrival: share that each other backbone gets right,
     split into missed targets and false positives, and by the phase inside the reference window.
Event order is the dataset order in every z0 dump, so events align across backbones.
Usage: python tools/deco/phase_window_probe.py --data-root <EV-UAV root> --split test --ref run/deco/v5s37 \
           --tags run/deco/v5w25s37 run/deco/v5s37 run/deco/v5w100s37 run/deco/v5w200s37 run/deco/v5w400s37 --out <txt>"""
import argparse
import glob
import math
import os

import numpy as np

PH = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]


def load_tag(tag, split):
    out = {}
    for f in sorted(glob.glob(os.path.join(tag, "z0_" + split, "*.npz"))):
        with np.load(f) as t:
            z = np.asarray(t["traj"][:, 0], np.float64)
            tau = math.log(9.0) - float(t["log_w"])
            w = int(t["window_ms"]) if "window_ms" in t.files else None
            y = np.asarray(t["labels"]) > 0.5
        out[os.path.basename(f)] = (z >= tau, y, w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    times = {}
    L = []
    tags = {t: load_tag(t, a.split) for t in a.tags}
    if a.ref not in tags:
        tags[a.ref] = load_tag(a.ref, a.split)
    names = sorted(set.intersection(*[set(d) for d in tags.values()]))
    for n in names:
        times[n] = np.asarray(np.load(os.path.join(a.data_root, a.split, n))["ev_loc"], np.float64)[:, 2]

    L.append("== A. arrival result by position inside the window (split %s, %d recordings)" % (a.split, len(names)))
    L.append("backbone                  window  phase    events%%   IoU     recall   FP per 1000 background")
    for t, d in tags.items():
        acc = np.zeros((5, 4))
        w = None
        for n in names:
            p, y, w = d[n]
            ph = np.mod(times[n], w) / w
            b = np.clip(np.digitize(ph, PH) - 1, 0, 4)
            for k in range(5):
                m = b == k
                acc[k] += [m.sum(), (m & p & y).sum(), (m & p & ~y).sum(), (m & ~p & y).sum()]
        nbg = None
        for k in range(5):
            n_, tp, fp, fn = acc[k]
            tgt = tp + fn
            L.append("%-25s %4s ms  %3d-%3d%%  %6.1f  %.4f  %.4f   %7.3f" % (
                os.path.basename(t.rstrip("/")), w, int(PH[k] * 100), int(round(PH[k + 1] * 100)),
                100 * n_ / acc[:, 0].sum(), tp / max(tp + fp + fn, 1), tp / max(tgt, 1), 1e3 * fp / max(n_ - tgt, 1)))
        tp, fp, fn = acc[:, 1].sum(), acc[:, 2].sum(), acc[:, 3].sum()
        L.append("%-25s %4s ms  all       100.0  %.4f  %.4f" % (os.path.basename(t.rstrip("/")), w, tp / (tp + fp + fn), tp / (tp + fn)))

    L.append("\n== B. events the reference (%s) gets wrong at arrival: share that each backbone gets right" % os.path.basename(a.ref.rstrip("/")))
    ref = tags[a.ref]
    L.append("backbone                  window   missed targets fixed   false positives fixed   | by phase in the reference window (all wrong events)")
    for t, d in tags.items():
        if t == a.ref:
            continue
        fixed = np.zeros(4)
        byph = np.zeros((5, 2))
        w = None
        for n in names:
            pr, y, wr = ref[n]
            p, _, w = d[n]
            miss, fpos = y & ~pr, ~y & pr
            fixed += [miss.sum(), (miss & p).sum(), fpos.sum(), (fpos & ~p).sum()]
            ph = np.clip(np.digitize(np.mod(times[n], wr) / wr, PH) - 1, 0, 4)
            wrong = miss | fpos
            ok = (p == y)
            for k in range(5):
                m = wrong & (ph == k)
                byph[k] += [m.sum(), (m & ok).sum()]
        L.append("%-25s %4s ms   %5.3f (of %6d)          %5.3f (of %6d)          | %s" % (
            os.path.basename(t.rstrip("/")), w, fixed[1] / max(fixed[0], 1), fixed[0], fixed[3] / max(fixed[2], 1), fixed[2],
            "  ".join("%d-%d%%: %.3f" % (int(PH[k] * 100), int(round(PH[k + 1] * 100)), byph[k, 1] / max(byph[k, 0], 1)) for k in range(5))))
    text = "\n".join(L) + "\n"
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
