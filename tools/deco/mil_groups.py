"""Target-frame groups of pending events for the multiple-instance ("at least one") loss (11-07 probe).

Group = (recording, official 50 ms frame (t // 50, t % 50 != 0), target id != 0) among the PENDING target events of a
split; a group is "at risk" when no non-pending event of the same frame and target is already positive at arrival
(z0 >= tau), i.e. the decision neurons alone decide whether this target is detected in this frame (official Pd).
group_of(evidence npz, z0 dir, data root, split) -> int array over the pending events (-1 = not in an at-risk group).
"""
import math
import os

import numpy as np


def group_of(ev, z0_dir, data_root, split, frame_ms=50):
    tau = math.log(0.9 / 0.1) - float(ev["log_w"])
    out = np.full(len(ev["idx"]), -1, np.int64)
    next_id = 0
    for ri, name in enumerate(ev["names"]):
        name = str(name)
        sel = np.nonzero(ev["rec"] == ri)[0]
        if sel.size == 0:
            continue
        with np.load(os.path.join(data_root, split, name)) as d:
            t = np.asarray(d["ev_loc"])[:, 2].astype(np.int64)
            tid = np.asarray(d["evs_norm"])[:, 5]
        with np.load(os.path.join(z0_dir, name)) as tr:
            z = np.asarray(tr["traj"][:, 0], np.float64)
        pend = np.zeros(len(t), bool)
        pend[ev["idx"][sel]] = True
        ok = (t % frame_ms != 0) & (tid != 0)
        frame = t // frame_ms
        # (frame, target) pairs already detected by a non-pending positive event
        done = ok & ~pend & (z >= tau)
        done_keys = set(zip(frame[done].tolist(), tid[done].tolist()))
        rows = ev["idx"][sel]
        keys = list(zip(frame[rows].tolist(), tid[rows].tolist()))
        local = {}
        for j, (row, key) in enumerate(zip(rows, keys)):
            if not ok[row] or key in done_keys:
                continue
            if key not in local:
                local[key] = next_id
                next_id += 1
            out[sel[j]] = local[key]
    return out


if __name__ == "__main__":
    import sys
    ev_prefix, z0_root, data_root = sys.argv[1:4]
    for split in ("train", "val", "test"):
        ev = np.load("%s_%s.npz" % (ev_prefix, split))
        g = group_of(ev, os.path.join(z0_root, "z0_" + split), data_root, split)
        n = g.max() + 1
        sizes = np.bincount(g[g >= 0], minlength=n) if n > 0 else np.zeros(0)
        print("%s: %d at-risk target-frame groups, %d pending events in them (of %d pending), size median %d, "
              "groups with 1 event %d" % (split, n, int((g >= 0).sum()), len(g), int(np.median(sizes)) if n else 0,
                                         int((sizes == 1).sum())))
