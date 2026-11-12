"""V14 overlapping observations with immutable original event IDs.

An observation is 100 ms and advances by 50 ms.  An 8 s recording therefore
has 159 complete observations.  Events in the first/last 50 ms are observed
once; all others twice.  Local time and time bins belong to an *appearance*,
not to an event ID: the same event has different local coordinates in its two
observations.  No events are resampled and no future padding is introduced.
"""
import numpy as np
import torch

from dataset import stream_windows as sw


class OverlapSequence(object):
    """Original arrays plus a window-major index of event appearances.

    ``x/y/t/p/label/target_id`` and coverage metadata have length N in original
    file order. ``order`` contains original IDs, possibly twice. ``bounds``
    indexes that expanded array. ``expanded_inner_bin`` and
    ``expanded_t_local`` have the same length as ``order``.
    """

    def __init__(self, name, x, y, t, p, label, target_id,
                 window_ms=100, stride_ms=50, total_ms=8000, time_bins=5):
        self.name = str(name)
        self.window_ms, self.stride_ms = int(window_ms), int(stride_ms)
        self.total_ms, self.time_bins = int(total_ms), int(time_bins)
        if self.stride_ms <= 0 or self.window_ms != 2 * self.stride_ms:
            raise ValueError("V14 requires window_ms == 2 * stride_ms > 0")
        if (self.total_ms < self.window_ms or
                (self.total_ms - self.window_ms) % self.stride_ms):
            raise ValueError("total_ms must end on a complete overlapping window")
        if self.time_bins <= 0:
            raise ValueError("time_bins must be positive")
        self.x, self.y, self.t = [np.asarray(a, dtype=np.int64) for a in (x, y, t)]
        self.p = np.asarray(p, dtype=np.int8)
        self.label = np.asarray(label, dtype=np.float32)
        self.target_id = np.asarray(target_id, dtype=np.float64)
        arrays = (self.x, self.y, self.t, self.p, self.label, self.target_id)
        if any(a.ndim != 1 or a.size != self.t.size for a in arrays):
            raise ValueError("all original event arrays must have shape [N]")
        if np.any(self.t < 0) or np.any(self.t >= self.total_ms):
            raise ValueError("event timestamp outside [0,total_ms)")
        self.window_starts = np.arange(
            0, self.total_ms - self.window_ms + 1, self.stride_ms, dtype=np.int64)
        self.window_ends = self.window_starts + self.window_ms
        sorted_id = np.argsort(self.t, kind="stable").astype(np.int64)
        sorted_t = self.t[sorted_id]
        starts = np.searchsorted(sorted_t, self.window_starts, side="left")
        ends = np.searchsorted(sorted_t, self.window_ends, side="left")
        self.bounds = np.concatenate(([0], np.cumsum(ends - starts))).astype(np.int64)
        self.order = np.concatenate([sorted_id[a:b] for a, b in zip(starts, ends)])
        self.window_of = np.repeat(np.arange(self.n_windows, dtype=np.int64), ends - starts)
        local_ms = self.t[self.order] - self.window_starts[self.window_of]
        self.expanded_inner_bin = (local_ms * self.time_bins // self.window_ms).astype(np.int64)
        self.expanded_t_local = (local_ms.astype(np.float64) / self.window_ms).astype(np.float32)
        self.coverage_count = np.bincount(self.order, minlength=self.n_events).astype(np.int64)
        self.first_window = np.full(self.n_events, self.n_windows, dtype=np.int64)
        self.last_window = np.full(self.n_events, -1, dtype=np.int64)
        np.minimum.at(self.first_window, self.order, self.window_of)
        np.maximum.at(self.last_window, self.order, self.window_of)
        position = np.arange(self.order.size, dtype=np.int64)
        self.first_position = np.full(self.n_events, self.order.size, dtype=np.int64)
        self.last_position = np.full(self.n_events, -1, dtype=np.int64)
        np.minimum.at(self.first_position, self.order, position)
        np.maximum.at(self.last_position, self.order, position)
        if np.any((self.coverage_count < 1) | (self.coverage_count > 2)):
            raise AssertionError("each event must have exactly one or two appearances")
        self.output_time_ms = self.window_ends[self.last_window]
        self.first_output_time_ms = self.window_ends[self.first_window]

    @property
    def n_events(self):
        return int(self.t.size)

    @property
    def n_windows(self):
        return int(self.window_starts.size)

    def window_index(self, k):
        if not 0 <= int(k) < self.n_windows:
            raise IndexError("window index out of range")
        return self.order[self.bounds[k]:self.bounds[k + 1]]

    def window_metadata(self, k):
        """Numpy metadata aligned with ``window_index(k)``; final is unique."""
        idx = self.window_index(k)
        count = self.coverage_count[idx]
        return {
            "id": idx, "coverage_count": count,
            "first_mask": self.first_window[idx] == k,
            "second_mask": (count == 2) & (self.last_window[idx] == k),
            "single_mask": count == 1,
            "final_mask": self.last_window[idx] == k,
            "output_time_ms": self.output_time_ms[idx],
            "event_time_ms": self.t[idx],
        }


def load_overlap_npz(path, height, width, window_ms=100, stride_ms=50,
                     total_ms=8000, time_bins=5):
    """Load the repository's original NPZ layout without changing event IDs."""
    if int(stride_ms) <= 0 or int(total_ms) % int(stride_ms):
        raise ValueError("total_ms must be divisible by positive stride_ms")
    seq = sw.load_npz_events(path, height, width, int(stride_ms),
                             int(total_ms) // int(stride_ms), time_bins)
    return OverlapSequence(seq.name, seq.x, seq.y, seq.t, seq.p, seq.label,
                           seq.target_id, window_ms, stride_ms, total_ms, time_bins)


def build_overlap_window_input(seq, k, q99, pad_height, pad_width, clip_max=3.0):
    """Reference numpy representation; the V5 2+2*time_bins layout is intact."""
    idx = seq.window_index(k)
    bins = seq.expanded_inner_bin[seq.bounds[k]:seq.bounds[k + 1]]
    counts = sw.count_channels(seq.x[idx], seq.y[idx], seq.p[idx], bins,
                               seq.time_bins, int(pad_height), int(pad_width))
    return sw.normalize_counts(counts, q99, clip_max), idx


class OverlapWindowSource(object):
    """Batched count construction on ``device``, compatible with forward_chunk.

    ``chunk(start, end, batch_index=0)`` uses an exclusive end, exactly like
    TorchWindowSource. The output is [T,1,C,H,W]; batch_index labels events
    when a caller concatenates independent stream slots along the batch axis.
    ``window(k)`` returns [1,C,H,W] and omits the relative step key ``t``.
    Additional metadata is ignored by the unmodified V5 readout.
    """

    def __init__(self, seq, cfg, q99, device, dtype=torch.float32):
        self.seq, self.device, self.dtype = seq, torch.device(device), dtype
        self.height, self.width = int(cfg["pad_height"]), int(cfg["pad_width"])
        if int(cfg.get("time_bins", seq.time_bins)) != seq.time_bins:
            raise ValueError("sequence and input time_bins disagree")
        if np.any(seq.x >= self.width) or np.any(seq.y >= self.height) or np.any(seq.x < 0) or np.any(seq.y < 0):
            raise ValueError("event coordinate outside padded input")
        self.channels = sw.num_input_channels(seq.time_bins)
        self.plane = self.height * self.width
        self.order, self.bounds = seq.order, seq.bounds

        def upload(array, tensor_dtype):
            return torch.from_numpy(np.ascontiguousarray(array)).to(device=self.device, dtype=tensor_dtype)

        ids = seq.order
        negative = (seq.p[ids] == 0).astype(np.int64)
        pixel = seq.y[ids] * self.width + seq.x[ids]
        self.window_of = upload(seq.window_of, torch.long)
        self.whole_key = upload(negative * self.plane + pixel, torch.long)
        self.bin_key = upload((2 + 2 * seq.expanded_inner_bin + negative) * self.plane + pixel, torch.long)
        count = seq.coverage_count[ids]
        self.data = {
            "id": upload(ids, torch.long),
            "y": upload(seq.y[ids], torch.long), "x": upload(seq.x[ids], torch.long),
            "p": upload(seq.p[ids], dtype),
            "t_local": upload(seq.expanded_t_local, dtype),
            "coverage_count": upload(count, torch.long),
            "first_mask": upload(seq.window_of == seq.first_window[ids], torch.bool),
            "second_mask": upload((count == 2) & (seq.window_of == seq.last_window[ids]), torch.bool),
            "single_mask": upload(count == 1, torch.bool),
            "final_mask": upload(seq.window_of == seq.last_window[ids], torch.bool),
            "output_time_ms": upload(seq.output_time_ms[ids], torch.long),
            "event_time_ms": upload(seq.t[ids], torch.long),
        }
        self.label = upload(seq.label[ids], dtype)
        table, self.n_sat = sw.normalization_table(q99, cfg.get("input_clip", 3.0))
        if table.shape[0] != self.channels:
            raise ValueError("q99 length does not match input channels")
        self.table = upload(table.reshape(-1), dtype)
        self.table_base = upload(np.arange(self.channels) * (self.n_sat + 1), torch.long).view(
            1, 1, self.channels, 1, 1)

    def chunk(self, start, end, batch_index=0):
        start, end = int(start), int(end)
        if not 0 <= start < end <= self.seq.n_windows:
            raise ValueError("chunk requires 0 <= start < end <= n_windows")
        if int(batch_index) < 0:
            raise ValueError("batch_index must be nonnegative")
        a, b = int(self.bounds[start]), int(self.bounds[end])
        steps, volume = end - start, self.channels * self.plane
        counts = torch.zeros(steps * volume, dtype=torch.float32, device=self.device)
        if b > a:
            base = (self.window_of[a:b] - start) * volume
            index = torch.cat((base + self.whole_key[a:b], base + self.bin_key[a:b]))
            counts.index_put_((index,), torch.ones(index.numel(), dtype=torch.float32, device=self.device),
                              accumulate=True)
        lookup = counts.to(torch.long).clamp_(max=self.n_sat).view(
            steps, 1, self.channels, self.height, self.width)
        x = self.table[lookup + self.table_base]
        events = {key: value[a:b] for key, value in self.data.items()}
        events["t"] = self.window_of[a:b] - start
        events["b"] = torch.full((b - a,), int(batch_index), dtype=torch.long, device=self.device)
        events["batch_idx"] = events["b"]
        return x, events, self.label[a:b], self.order[a:b]

    def window(self, k, batch_index=0):
        x, events, label, ids = self.chunk(k, k + 1, batch_index=batch_index)
        del events["t"]
        return x[0], events, label, ids


def update_overlap_histogram(hist, seq, k, pad_height, pad_width):
    """Exactly the dense nonzero-count histogram, without allocating C*H*W."""
    ids = seq.window_index(k)
    if not ids.size:
        return hist
    bins = seq.expanded_inner_bin[seq.bounds[k]:seq.bounds[k + 1]]
    plane = int(pad_height) * int(pad_width)
    pixel = seq.y[ids] * int(pad_width) + seq.x[ids]
    negative = (seq.p[ids] == 0).astype(np.int64)
    keys = np.concatenate((negative * plane + pixel, (2 + 2 * bins + negative) * plane + pixel))
    unique, counts = np.unique(keys, return_counts=True)
    channels = unique // plane
    n_ch = sw.num_input_channels(seq.time_bins)
    table = np.bincount(counts * n_ch + channels,
                        minlength=(int(counts.max()) + 1) * n_ch).reshape(-1, n_ch)
    for c in range(n_ch):
        if hist[c].size < table.shape[0]:
            hist[c] = np.pad(hist[c], (0, table.shape[0] - hist[c].size), mode="constant")
        hist[c][:table.shape[0]] += table[:, c]
    return hist


def compute_overlap_stats(paths, cfg, split="train"):
    """Fit separate 100 ms normalization on explicitly supplied training files.

    Label counts count each original event once, not each appearance.  Callers
    must supply their training manifest; validation/test fitting is rejected.
    """
    if split != "train":
        raise ValueError("normalization may only be fitted on the training split")
    paths = list(paths)
    if not paths:
        raise ValueError("empty training manifest")
    hist = sw.new_count_histogram(sw.num_input_channels(cfg["time_bins"]))
    n_pos = n_neg = n_windows = appearances = 0
    total_ms = int(cfg.get("total_ms", cfg.get("duration_ms", 8000)))
    for path in paths:
        seq = load_overlap_npz(path, cfg["height"], cfg["width_px"],
                               cfg.get("window_ms", 100), cfg.get("stride_ms", 50),
                               total_ms, cfg["time_bins"])
        for k in range(seq.n_windows):
            update_overlap_histogram(hist, seq, k, cfg["pad_height"], cfg["pad_width"])
        pos = int(np.count_nonzero(seq.label == 1))
        n_pos += pos
        n_neg += seq.n_events - pos
        n_windows += seq.n_windows
        appearances += int(seq.order.size)
    cap = float(cfg.get("pos_weight_cap", 20.0))
    return {
        "q99": [sw.quantile_from_histogram(h, 0.99) for h in hist],
        "channel_names": sw.channel_names(cfg["time_bins"]),
        "nonzero_pixels_per_channel": [int(h[1:].sum()) for h in hist],
        "train_pos": n_pos, "train_neg": n_neg,
        "train_neg_pos_ratio": float(n_neg) / float(n_pos) if n_pos else None,
        "pos_weight_cap": cap, "pos_weight": sw.pos_weight_from_counts(n_pos, n_neg, cap),
        "split": "train", "source_files": [str(p) for p in paths],
        "files": len(paths), "windows": n_windows, "appearances": appearances,
        "config_used": {"window_ms": int(cfg.get("window_ms", 100)),
                        "stride_ms": int(cfg.get("stride_ms", 50)), "total_ms": total_ms,
                        "time_bins": int(cfg["time_bins"]), "height": int(cfg["height"]),
                        "width_px": int(cfg["width_px"]), "pad_height": int(cfg["pad_height"]),
                        "pad_width": int(cfg["pad_width"])},
    }
