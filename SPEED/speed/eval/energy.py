"""Theoretical energy accounting (45 nm estimates, Horowitz 2014): MAC 4.6 pJ, accumulate / elementwise 0.9 pJ.

Each part reports operations per processing window: mac (real-valued multiply-accumulate), ac (spike-driven
accumulate or elementwise op) and transcendental (exp / log / softplus, charged as `transcendental_macs` MACs;
results are given for 1, 10 and 20 because there is no accepted unit cost).
"""
MAC_PJ = 4.6
AC_PJ = 0.9
TRANSCENDENTAL_MACS = (1.0, 10.0, 20.0)


def part(mac=0.0, ac=0.0, transcendental=0.0, note=""):
    return {"mac": float(mac), "ac": float(ac), "transcendental": float(transcendental), "note": note}


def energy_mj(ops, transcendental_macs=10.0):
    pj = ops["mac"] * MAC_PJ + ops["ac"] * AC_PJ + ops["transcendental"] * transcendental_macs * MAC_PJ
    return pj * 1e-9


def pixel_head(c_in, hidden, n_out, positions, transcendental_per_position=0.0, note=""):
    """1x1-conv head c_in -> hidden -> ReLU -> n_out on real-valued input, evaluated at `positions` pixels."""
    p = float(positions)
    return part(mac=p * (c_in * hidden + hidden * n_out), ac=p * hidden,
                transcendental=p * transcendental_per_position, note=note)


def system_energy(parts, window_us, clip_us=None, transcendental_macs=TRANSCENDENTAL_MACS):
    """Per-part and total energy per window, per second and (optionally) per clip, for each transcendental cost."""
    windows_per_s = 1e6 / float(window_us)
    windows_per_clip = None if clip_us is None else float(clip_us) / float(window_us)
    out = {"window_us": int(window_us), "parts": {}, "total": {}}
    for scale in transcendental_macs:
        key = "x%g" % scale
        total = 0.0
        for name, ops in parts.items():
            mj = energy_mj(ops, scale)
            total += mj
            row = out["parts"].setdefault(name, dict(ops))
            row[key] = {"mj_per_window": mj, "mj_per_s": mj * windows_per_s}
            if windows_per_clip is not None:
                row[key]["mj_per_clip"] = mj * windows_per_clip
        out["total"][key] = {"mj_per_window": total, "mj_per_s": total * windows_per_s}
        if windows_per_clip is not None:
            out["total"][key]["mj_per_clip"] = total * windows_per_clip
    return out
