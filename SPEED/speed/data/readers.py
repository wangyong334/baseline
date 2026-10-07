import os

import numpy as np

from speed.data.events import EventStream


def _as_int(column, what, source):
    values = np.asarray(column)
    rounded = np.round(values)
    if not np.array_equal(values, rounded):
        raise ValueError("%s: %s contains non-integer values" % (source, what))
    return rounded.astype(np.int64)


def read_evuav_npz(path, card, name=None):
    """EV-UAV layout (also used by tools/edds_convert.py): ev_loc = integer (x, y, t), evs_norm[:, 3:6] = (p, label, id)."""
    source = name or os.path.basename(str(path))
    with np.load(path) as data:
        loc = np.asarray(data["ev_loc"])
        norm = np.asarray(data["evs_norm"])
    if loc.ndim != 2 or loc.shape[1] < 3 or norm.ndim != 2 or norm.shape[1] < 6 or loc.shape[0] != norm.shape[0]:
        raise ValueError("%s: unexpected array shapes %s / %s" % (source, loc.shape, norm.shape))
    x = _as_int(loc[:, 0], "x", source)
    y = _as_int(loc[:, 1], "y", source)
    t = _as_int(loc[:, 2], "t", source) * int(card["time_scale_us"])
    p = _as_int(norm[:, 3], "p", source)
    label = _as_int(norm[:, 4], "label", source)
    target_id = _as_int(norm[:, 5], "target_id", source)
    cls = (label == 1).astype(np.int16) * int(card.get("single_class", 1))
    return EventStream(source, x, y, t, p, label, target_id, cls, card["sensor"]["width"], card["sensor"]["height"],
                       span_us=card.get("clip_duration_us")).validate()


def read_evflying_npy(path, card, name=None):
    """EV-Flying layout: float64 columns (x, y, p, t_us, id, class); id -1 and class 0 mark background."""
    source = name or os.path.basename(str(path))
    raw = np.load(path, mmap_mode="r")
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError("%s: expected 6 columns, got shape %s" % (source, raw.shape))
    x = _as_int(raw[:, 0], "x", source)
    y = _as_int(raw[:, 1], "y", source)
    p = _as_int(raw[:, 2], "p", source)
    t = _as_int(raw[:, 3], "t", source) * int(card["time_scale_us"])
    ident = _as_int(raw[:, 4], "id", source)
    cls = _as_int(raw[:, 5], "class", source)
    if ident.size and ident[ident >= 0].size and ident[ident >= 0].min() < 1:
        raise ValueError("%s: target ids must start at 1 so that 0 can mean background" % source)
    target_id = np.where(ident >= 0, ident, 0)
    label = (target_id > 0).astype(np.uint8)
    return EventStream(source, x, y, t, p, label, target_id, cls, card["sensor"]["width"], card["sensor"]["height"],
                       span_us=card.get("clip_duration_us")).validate()


READERS = {"evuav_npz": read_evuav_npz, "evflying_npy": read_evflying_npy}


def read_recording(card, path, name=None):
    return READERS[card["format"]](path, card, name)
