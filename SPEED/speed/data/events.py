import numpy as np


class EventStream(object):
    """Events of one recording in file order, timestamps in integer microseconds.

    target_id 0 marks background; label == 1 exactly when target_id > 0; cls > 0 exactly when label == 1.
    span_us is the nominal extent [0, span_us): the clip length for fixed-length clips, otherwise t.max() + 1.
    """

    def __init__(self, name, x, y, t, p, label, target_id, cls, width, height, span_us=None):
        self.name = str(name)
        self.x = np.ascontiguousarray(x, dtype=np.int64)
        self.y = np.ascontiguousarray(y, dtype=np.int64)
        self.t = np.ascontiguousarray(t, dtype=np.int64)
        self.p = np.ascontiguousarray(p, dtype=np.int8)
        self.label = np.ascontiguousarray(label, dtype=np.uint8)
        self.target_id = np.ascontiguousarray(target_id, dtype=np.int64)
        self.cls = np.ascontiguousarray(cls, dtype=np.int16)
        self.width, self.height = int(width), int(height)
        if span_us is None:
            span_us = int(self.t.max()) + 1 if self.t.size else 0
        self.span_us = int(span_us)

    @property
    def n_events(self):
        return int(self.t.shape[0])

    def validate(self):
        n = self.n_events
        for name in ("x", "y", "p", "label", "target_id", "cls"):
            if getattr(self, name).shape != (n,):
                raise ValueError("%s: field %s has shape %s, expected (%d,)"
                                 % (self.name, name, getattr(self, name).shape, n))
        checks = (
            ("t<0", self.t < 0), ("t>=span", self.t >= self.span_us),
            ("x out of range", (self.x < 0) | (self.x >= self.width)),
            ("y out of range", (self.y < 0) | (self.y >= self.height)),
            ("p not in {0,1}", (self.p != 0) & (self.p != 1)),
            ("label not in {0,1}", self.label > 1),
            ("target_id<0", self.target_id < 0),
            ("label/target_id mismatch", (self.label == 1) != (self.target_id > 0)),
            ("label/cls mismatch", (self.label == 1) != (self.cls > 0)),
        )
        problems = ["%s: %d" % (desc, int(np.count_nonzero(mask))) for desc, mask in checks if mask.any()]
        if problems:
            raise ValueError("%s: invalid events -> %s" % (self.name, "; ".join(problems)))
        return self

    def n_windows(self, window_us, origin_us=0):
        span = self.span_us - int(origin_us)
        return max(0, -(-span // int(window_us)))

    def window_partition(self, window_us, n_windows=None, origin_us=0):
        """(order, bounds): order is the stable time sort of file indices; window k holds order[bounds[k]:bounds[k+1]].

        Windows are [origin + k*w, origin + (k+1)*w); events before origin are excluded.
        """
        w, origin = int(window_us), int(origin_us)
        if n_windows is None:
            n_windows = self.n_windows(w, origin)
        order = np.argsort(self.t, kind="stable").astype(np.int64)
        rel = self.t[order] - origin
        start = int(np.searchsorted(rel, 0, side="left"))
        order = order[start:]
        window_of_sorted = rel[start:] // w
        bounds = np.searchsorted(window_of_sorted, np.arange(int(n_windows) + 1), side="left")
        return order, bounds.astype(np.int64)

    def time_slice(self, t0_us, t1_us, rebase=True):
        """Events with t in [t0, t1) as a new stream (file order kept) and their indices in this stream."""
        index = np.flatnonzero((self.t >= int(t0_us)) & (self.t < int(t1_us))).astype(np.int64)
        shift = int(t0_us) if rebase else 0
        span = int(t1_us) - shift
        sub = EventStream("%s[%d:%d]" % (self.name, int(t0_us), int(t1_us)), self.x[index], self.y[index],
                          self.t[index] - shift, self.p[index], self.label[index], self.target_id[index],
                          self.cls[index], self.width, self.height, span_us=span)
        return sub, index

    def summary(self):
        target = self.label == 1
        return {"name": self.name, "n_events": self.n_events, "n_target_events": int(np.count_nonzero(target)),
                "n_targets": int(np.unique(self.target_id[target]).size), "span_us": self.span_us,
                "width": self.width, "height": self.height}
