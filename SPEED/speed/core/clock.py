"""Slot 1 (clock) and the device-resident event blocks every other slot reads.

A clock partitions a stream into steps. Steps carry their end times and lengths explicitly, so slots never assume a
fixed step length; the base variant uses fixed steps.
EventBlocks keeps one stream's events on the device in step order and hands out chunks of consecutive steps.
"""
import numpy as np
import torch


class Steps(object):
    def __init__(self, order, bounds, t_end_us, dt_ms):
        self.order = order                       # file indices in step order (stable time sort)
        self.bounds = bounds                     # step k holds order[bounds[k]:bounds[k+1]]
        self.t_end_us = t_end_us                 # int64 [n_steps]
        self.dt_ms = dt_ms                       # float64 [n_steps]

    @property
    def n_steps(self):
        return int(self.bounds.shape[0] - 1)

    def step_of_events(self):
        return np.repeat(np.arange(self.n_steps, dtype=np.int64), np.diff(self.bounds))


class FixedClock(object):
    kind = "fixed"

    def __init__(self, step_ms):
        self.step_us = int(round(float(step_ms) * 1000.0))
        if self.step_us <= 0:
            raise ValueError("step_ms must be positive")
        self.step_ms = self.step_us / 1000.0

    def partition(self, stream, origin_us=0):
        n = stream.n_windows(self.step_us, origin_us)
        order, bounds = stream.window_partition(self.step_us, n, origin_us)
        t_end = int(origin_us) + self.step_us * np.arange(1, n + 1, dtype=np.int64)
        return Steps(order, bounds, t_end, np.full(n, self.step_ms))


CLOCKS = {"fixed": lambda cfg: FixedClock(cfg["step_ms"])}


def build_clock(cfg):
    return CLOCKS[cfg.get("kind", "fixed")](cfg)


def canvas_size(height, width, multiple):
    """Sensor size padded up (bottom / right only) to a multiple of the backbone's total downsampling."""
    m = int(multiple)
    return -(-int(height) // m) * m, -(-int(width) // m) * m


class EventBlocks(object):
    """Events of one stream on the device, in step order.

    block(start, end) -> {"events": {t, b, y, x, pixel, negative, age_ms}, "labels", "idx", "n_steps", "start"}
        t is the step index within the block, age_ms the time from the event to the end of its step.
    accumulate(keys, values, size) sums values into a flat buffer with index_put_ (deterministic on CUDA).
    extras: optional per-event arrays in file order (e.g. {"vel": [N, 2]}), returned in each block's events.
    """

    def __init__(self, stream, steps, height, width, device, dtype=torch.float32, extras=None):
        self.steps, self.device, self.dtype = steps, device, dtype
        self.height, self.width = int(height), int(width)
        self.plane = self.height * self.width
        order = steps.order
        step = steps.step_of_events()
        x, y = stream.x[order].astype(np.int64), stream.y[order].astype(np.int64)
        age = (steps.t_end_us[step] - stream.t[order]).astype(np.float64) / 1000.0

        def upload(array, tensor_dtype):
            return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=tensor_dtype)

        self.step_of = upload(step, torch.long)
        self.pixel = upload(y * self.width + x, torch.long)
        self.negative = upload((stream.p[order] == 0).astype(np.int64), torch.long)
        self.x, self.y = upload(x, torch.long), upload(y, torch.long)
        self.label = upload(stream.label[order].astype(np.float32), torch.float32)
        self.age = upload(age, dtype)
        self.order = order
        self.extras = {name: upload(np.asarray(value)[order], torch.float64 if np.asarray(value).dtype.kind == "f"
                                    else torch.long) for name, value in (extras or {}).items()}

    def accumulate(self, keys, values, size):
        out = torch.zeros(size, dtype=self.dtype, device=self.device)
        if keys.numel():
            out.index_put_((keys,), values.to(self.dtype), accumulate=True)
        return out

    def block(self, start, end):
        start, end = int(start), int(end)
        if end <= start:
            raise ValueError("a block needs at least one step")
        a, b = int(self.steps.bounds[start]), int(self.steps.bounds[end])
        events = {"t": self.step_of[a:b] - start, "b": torch.zeros(b - a, dtype=torch.long, device=self.device),
                  "y": self.y[a:b], "x": self.x[a:b], "pixel": self.pixel[a:b], "negative": self.negative[a:b],
                  "age_ms": self.age[a:b]}
        for name, value in self.extras.items():
            events[name] = value[a:b]
        return {"events": events, "labels": self.label[a:b].to(self.dtype), "idx": self.order[a:b],
                "n_steps": end - start, "start": start, "source": self}
