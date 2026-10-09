"""Physically consistent perturbations of a stream (V4-62; EventDrop, IJCAI 2021, for random dropping).

    thinning      every event is kept with probability p: a thinned Poisson process is Poisson, i.e. weaker targets /
                  a less sensitive sensor, same geometry
    time scaling  timestamps are multiplied by s <= 1: every motion becomes 1/s times faster and every event rate 1/s
                  times higher; labels unchanged (label velocities are recomputed from them)
augment_stream (training) draws p ~ U[keep_min, 1] and s ~ log-U[time_scale_min, 1] per call; perturb_stream applies
fixed values (robustness tests at inference). Both act before the clock partitions the stream.
"""
import math

import numpy as np

from speed.data.events import EventStream


def perturb_stream(stream, rng, keep=1.0, time_scale=1.0):
    keep, time_scale = float(keep), float(time_scale)
    if not (0.0 < keep <= 1.0 and 0.0 < time_scale <= 1.0):
        raise ValueError("keep and time_scale must lie in (0, 1]")
    if keep == 1.0 and time_scale == 1.0:
        return stream
    kept = rng.random_sample(stream.n_events) < keep if keep < 1.0 else np.ones(stream.n_events, bool)
    t = np.floor(stream.t[kept] * time_scale).astype(np.int64)
    span = max(int(math.ceil(stream.span_us * time_scale)), int(t.max()) + 1 if t.size else 1)
    return EventStream(stream.name, stream.x[kept], stream.y[kept], t, stream.p[kept], stream.label[kept],
                       stream.target_id[kept], stream.cls[kept], stream.width, stream.height, span_us=span)


def augment_stream(stream, rng, keep_min=1.0, time_scale_min=1.0):
    keep_min, time_scale_min = float(keep_min), float(time_scale_min)
    if not (0.0 < keep_min <= 1.0 and 0.0 < time_scale_min <= 1.0):
        raise ValueError("keep_min and time_scale_min must lie in (0, 1]")
    p = rng.uniform(keep_min, 1.0) if keep_min < 1.0 else 1.0
    s = math.exp(rng.uniform(math.log(time_scale_min), 0.0)) if time_scale_min < 1.0 else 1.0
    return perturb_stream(stream, rng, p, s)
