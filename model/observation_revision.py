"""One-step delayed feature revision using a current-observation side channel.

The persistent SNN supplies the query/estimate. In the observation arm, the
value source is the first convolution's driving current before its membrane
recurrence; the caller must supply the plain V5 input without a history-based
normalizer. The explicit deep control instead receives stateful features.
Local attention supplies a learned association, not a Bayesian independence
guarantee or a calibrated correspondence probability. The gate scales a signed
feature correction; it is not a calibrated confidence/reliability estimate.
Python 3.8 / PyTorch 1.9 compatible.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


def _linear_resize_adjoint(gradient, input_length, scale):
    """Gather each input's finite interpolation support, then reduce in order.

    For align_corners=False the source coordinate is (j + .5)/scale - .5,
    clamped at the image boundary. Integer enlargement has at most 2*scale
    contributing output positions per input. This is the transpose of that
    interpolation, without the atomic scatter used by old CUDA upsampling.
    Temporary storage is at most twice the incoming gradient, not a dense
    input_length x output_length interpolation matrix.
    """
    position = torch.arange(input_length, device=gradient.device)
    offset = torch.arange(2 * scale, device=gradient.device)
    indices = position[:, None] * scale - scale // 2 + offset[None, :]
    valid = (indices >= 0) & (indices < gradient.shape[-1])
    safe = indices.clamp(0, gradient.shape[-1] - 1)
    coordinate = ((safe.to(gradient.dtype) + 0.5) / scale - 0.5).clamp(0, input_length - 1)
    weight = (1 - (coordinate - position[:, None].to(gradient.dtype)).abs()).clamp_min(0)
    weight = weight * valid.to(gradient.dtype)
    gathered = gradient.index_select(-1, safe.reshape(-1))
    gathered = gathered.reshape(*gradient.shape[:-1], input_length, 2 * scale)
    return (gathered * weight).sum(-1)


class _BilinearResize(torch.autograd.Function):
    """Native bilinear forward with a deterministic integer-scale adjoint.

    PyTorch 1.9 CUDA rejects native bilinear backward in strict deterministic
    mode. Keeping its forward preserves inference exactly; only the order of
    gradient accumulation changes. This operator supports first-order training
    and integer enlargement (the patch-to-canvas path), not arbitrary resizing.
    """
    @staticmethod
    def forward(ctx, value, size):
        height, width = value.shape[-2:]
        if (len(size) != 2 or size[0] < height or size[1] < width
                or size[0] % height or size[1] % width):
            raise ValueError("Deterministic bilinear resize requires integer enlargement")
        ctx.input_hw = (height, width)
        ctx.scales = (size[0] // height, size[1] // width)
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        height, width = ctx.input_hw
        sy, sx = ctx.scales
        gradient = _linear_resize_adjoint(gradient, width, sx)
        gradient = _linear_resize_adjoint(gradient.transpose(-2, -1), height, sy)
        return gradient.transpose(-2, -1), None


def bilinear_resize(value, size):
    # Use the same path on CPU/GPU so old-version CPU tests exercise the
    # backward implementation used on the server. Inference remains native.
    if torch.is_grad_enabled() and value.requires_grad:
        return _BilinearResize.apply(value, tuple(size))
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)


class ObservationRevision(nn.Module):
    def __init__(self, channels=12, width=12, heads=3, patch=4, radius=2,
                 source="observation", pair_batch=4, sensor_hw=None):
        super().__init__()
        if min(channels, width, heads, patch, pair_batch) < 1 or radius < 0:
            raise ValueError("Positive dimensions/batch size and nonnegative radius required")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if source not in ("observation", "deep", "causal"):
            raise ValueError("Unknown observation source")
        self.channels, self.width, self.heads = int(channels), int(width), int(heads)
        self.patch, self.radius, self.pair_batch = int(patch), int(radius), int(pair_batch)
        self.source = source
        if sensor_hw is not None:
            if (not isinstance(sensor_hw, (tuple, list)) or len(sensor_hw) != 2
                    or any(isinstance(v, bool) or int(v) != v or int(v) < 1 for v in sensor_hw)):
                raise ValueError("sensor_hw must contain two positive integer dimensions")
            sensor_hw = tuple(int(v) for v in sensor_hw)
        self.sensor_hw = sensor_hw
        self.encoder = nn.Sequential(nn.Conv2d(channels * patch * patch, width, 1), nn.SiLU())
        self.query = nn.Conv2d(channels + width, width, 1)
        self.key = nn.Conv2d(width, width, 1)
        # Bias permits an observed empty input to have a learned meaning.
        # A missing future observation is handled explicitly by availability.
        self.value = nn.Conv2d(width, width, 1)
        self.output = nn.Conv2d(width, channels, 1, bias=False)
        self.gate = nn.Conv2d(2 * channels + 1, channels, 1)
        kernel = 2 * radius + 1
        self.relative_bias = nn.Parameter(torch.zeros(heads, kernel * kernel))
        self.null_key = nn.Parameter(torch.zeros(heads, width // heads))
        self.null_bias = nn.Parameter(torch.zeros(heads))
        # Start from exactly the original V5 function. The gate is not zeroed,
        # so the output projection receives gradients on the first update.
        nn.init.zeros_(self.output.weight)
        # Only a geometric mask is cached; never a feature or temporal state.
        self._mask_cache = {}
        self._geometry_cache = {}

    def _geometry(self, height, width, ref):
        """Sensor masks on the padded canvas; a partial patch stays usable.

        None means the whole canvas is valid, preserving the original numeric
        path for callers without padding. No data-dependent mask is cached.
        """
        if self.sensor_hw is None or self.sensor_hw == (height, width):
            return None
        sh, sw = self.sensor_hw
        if sh > height or sw > width:
            raise ValueError("sensor_hw cannot exceed the padded canvas")
        key = (height, width, str(ref.device), ref.dtype, self.sensor_hw)
        if key not in self._geometry_cache:
            if len(self._geometry_cache) >= 4:
                self._geometry_cache.clear()
            rows = torch.arange(height, device=ref.device) < sh
            columns = torch.arange(width, device=ref.device) < sw
            pixel = (rows[:, None] & columns[None, :]).to(ref.dtype)[None, None]
            fraction = F.avg_pool2d(pixel, self.patch, self.patch)
            compact = fraction.gt(0).to(ref.dtype)
            self._geometry_cache[key] = (pixel, compact, fraction)
        return self._geometry_cache[key]

    def encode_observation(self, observation):
        if observation.dim() != 4 or observation.shape[1] != self.channels:
            raise ValueError("observation must be [B,channels,H,W]")
        if observation.shape[-2] % self.patch or observation.shape[-1] % self.patch:
            raise ValueError("Canvas dimensions must be divisible by patch")
        geometry = self._geometry(*observation.shape[-2:], observation)
        if geometry is not None:
            # A convolution can produce nonzero values outside the physical
            # sensor. These values must not masquerade as new observations.
            observation = observation * geometry[0]
        # Space-to-depth retains each subpixel before learned compression.
        # The subsequent projection is deliberately compact, not lossless.
        encoded = self.encoder(F.pixel_unshuffle(observation, self.patch))
        # Bias is meaningful for an observed empty sensor patch, but not for a
        # patch made entirely of canvas padding.
        return encoded if geometry is None else encoded * geometry[1]

    def _mask(self, height, width, ref):
        key = (height, width, str(ref.device), ref.dtype, self.sensor_hw)
        if key not in self._mask_cache:
            # Bounded cache: normal training uses one canvas, test probes may
            # use others. The mask does not contain data or autograd history.
            if len(self._mask_cache) >= 4:
                self._mask_cache.clear()
            geometry = self._geometry(height * self.patch, width * self.patch, ref)
            ones = ref.new_ones(1, 1, height, width) if geometry is None else geometry[1]
            self._mask_cache[key] = F.unfold(
                ones, 2 * self.radius + 1, padding=self.radius).bool().unsqueeze(1)
            if geometry is not None:
                # Invalid query patches only have the null option. Their
                # zero messages are excluded during interpolation below.
                self._mask_cache[key] &= ones.flatten(2).bool().unsqueeze(1)
        return self._mask_cache[key]

    def _revise_batch(self, old_features, old_observation, new_observation, available, return_info):
        batch, _, height, width = old_observation.shape
        positions, dim = height * width, self.width // self.heads
        kernel, neighbors = 2 * self.radius + 1, (2 * self.radius + 1) ** 2
        geometry = self._geometry(*old_features.shape[-2:], old_features)
        pooled_source = old_features if geometry is None else old_features * geometry[0]
        pooled = F.avg_pool2d(pooled_source, self.patch, self.patch)
        if geometry is not None:
            pooled = pooled / geometry[2].clamp_min(1.0 / self.patch ** 2)
            old_observation = old_observation * geometry[1]
            new_observation = new_observation * geometry[1]
        q = self.query(torch.cat((pooled, old_observation), 1)).reshape(batch, self.heads, dim, positions)
        keys = F.unfold(self.key(new_observation), kernel, padding=self.radius)
        values = F.unfold(self.value(new_observation), kernel, padding=self.radius)
        keys = keys.reshape(batch, self.heads, dim, neighbors, positions)
        values = values.reshape(batch, self.heads, dim, neighbors, positions)
        scores = (q.unsqueeze(3) * keys).sum(2) / math.sqrt(dim)
        scores = scores + self.relative_bias[None, :, :, None]
        scores = scores.masked_fill(~self._mask(height, width, scores), -float("inf"))
        null_score = (q * self.null_key[None, :, :, None]).sum(2) / math.sqrt(dim)
        null_score = null_score + self.null_bias[None, :, None]
        attention = torch.softmax(torch.cat((scores, null_score.unsqueeze(2)), 2), dim=2)
        message = (attention[:, :, None, :neighbors, :] * values).sum(3)
        message = message.reshape(batch, self.width, height, width)
        reject = attention[:, :, -1].mean(1).reshape(batch, 1, height, width)
        projected = self.output(message)
        if geometry is None:
            correction = bilinear_resize(projected, old_features.shape[-2:])
            reject_full = bilinear_resize(reject, old_features.shape[-2:])
        else:
            # Do not attenuate a valid boundary pixel by averaging with a
            # nonexistent neighboring patch. Normalize interpolation over
            # valid query patches; the full-resolution mask is applied below.
            scale = bilinear_resize(geometry[1], old_features.shape[-2:]).clamp_min(1e-12)
            correction = bilinear_resize(projected * geometry[1], old_features.shape[-2:]) / scale
            reject_full = bilinear_resize(reject * geometry[1], old_features.shape[-2:]) / scale
        gate = torch.sigmoid(self.gate(torch.cat((old_features, correction, reject_full), 1)))
        delta = gate * correction
        if geometry is not None:
            delta = delta * geometry[0]
        revised = old_features + delta
        if available is not None:
            # where provides an exact fallback even if unavailable placeholders
            # would otherwise carry a nonzero value-encoder bias.
            mask = available.to(device=old_features.device, dtype=torch.bool).reshape(batch, 1, 1, 1)
            revised = torch.where(mask, revised, old_features)
            delta = torch.where(mask, delta, torch.zeros_like(delta))
        if return_info:
            if geometry is None:
                null_mass_mean = reject.mean((1, 2, 3))
                gate_mean = gate.mean((1, 2, 3))
                correction_rms = delta.square().mean((1, 2, 3)).sqrt()
            else:
                valid_pixels = geometry[0].sum() * self.channels
                null_mass_mean = (reject * geometry[1]).sum((1, 2, 3)) / geometry[1].sum()
                gate_mean = (gate * geometry[0]).sum((1, 2, 3)) / valid_pixels
                correction_rms = ((delta.square() * geometry[0]).sum((1, 2, 3)) / valid_pixels).sqrt()
            return revised, {"null_probability": reject, "null_mass_mean": null_mass_mean,
                             "gate_mean": gate_mean, "correction_rms": correction_rms}
        return revised

    def revise(self, old_features, old_observation, new_observation, available=None, return_info=False):
        if old_features.dim() != 4 or old_features.shape[1] != self.channels:
            raise ValueError("old_features must be [B,channels,H,W]")
        if old_features.shape[-2] % self.patch or old_features.shape[-1] % self.patch:
            raise ValueError("Canvas dimensions must be divisible by patch")
        self._geometry(*old_features.shape[-2:], old_features)
        expected = (old_features.shape[0], self.width, old_features.shape[2] // self.patch,
                    old_features.shape[3] // self.patch)
        if tuple(old_observation.shape) != expected or tuple(new_observation.shape) != expected:
            raise ValueError("Both compact observations must align with the old canvas")
        if available is not None and available.numel() != old_features.shape[0]:
            raise ValueError("available requires one flag per pair")
        if self.source == "causal":
            if return_info:
                return old_features, {"null_probability": old_features.new_ones(expected[0], 1, *expected[2:]),
                                      "null_mass_mean": old_features.new_ones(expected[0]),
                                      "gate_mean": old_features.new_zeros(expected[0]),
                                      "correction_rms": old_features.new_zeros(expected[0])}
            return old_features
        if old_features.shape[0] <= self.pair_batch:
            return self._revise_batch(old_features, old_observation, new_observation, available, return_info)
        results, diagnostics = [], []
        for start in range(0, old_features.shape[0], self.pair_batch):
            end = start + self.pair_batch
            result = self._revise_batch(old_features[start:end], old_observation[start:end],
                                        new_observation[start:end],
                                        None if available is None else available[start:end], return_info)
            if return_info:
                result, info = result
                diagnostics.append(info)
            results.append(result)
        result = torch.cat(results)
        if return_info:
            return result, {key: torch.cat([d[key] for d in diagnostics]) for key in diagnostics[0]}
        return result

    def forward(self, old_features, old_observation, new_observation, available=None, return_info=False):
        return self.revise(old_features, old_observation, new_observation, available, return_info)

    def dense_macs(self, height, width):
        """One new observation plus one delayed revision, excluding V5/readout.

        Counts conv/linear and QK/AV multiply-accumulates; excludes nonlinear,
        softmax, pooling, interpolation and memory traffic. Not an energy model.
        """
        p = (height // self.patch) * (width // self.patch)
        c, d, k = self.channels, self.width, (2 * self.radius + 1) ** 2
        return int(p * (c * self.patch ** 2 * d + (c + d) * d + 2 * d * d
                        + 2 * k * d + d + d * c) + height * width * (2 * c + 1) * c)
