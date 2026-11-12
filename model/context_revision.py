"""Shared-grid context contrast and conditional two-view feature revision.

Both views describe the same output fragment but can contain overlapping input
and recurrent history.  Local contrast is not a decomposition of independent
evidence, and the signed revision is not an exact Bayesian posterior.  The
module does not create per-event hidden states or modify the SNN state chain.
Python 3.8 / PyTorch 1.9 compatible.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .deterministic_resize import bilinear_resize


class ContextRevision(nn.Module):
    def __init__(self, channels=12, dim=12, hidden=24, heads=3, radius=2,
                 downsample=4, height=260, width=346, variant="full"):
        super().__init__()
        positive = (channels, dim, hidden, heads, downsample, height, width)
        if any(isinstance(x, bool) or int(x) != x or x < 1 for x in positive):
            raise ValueError("Dimensions, heads, downsample and sensor size must be positive integers")
        if isinstance(radius, bool) or int(radius) != radius or radius < 0:
            raise ValueError("radius must be a nonnegative integer")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if variant not in ("full", "no_context", "no_revision"):
            raise ValueError("variant must be full, no_context, or no_revision")
        self.channels, self.dim, self.hidden = int(channels), int(dim), int(hidden)
        self.heads, self.radius, self.downsample = int(heads), int(radius), int(downsample)
        self.height, self.width, self.variant = int(height), int(width), variant
        # With the default 12-channel V5 decoder, U is exactly masked pooling.
        self.projection = (nn.Identity() if channels == dim
                           else nn.Conv2d(channels, dim, 1, bias=False))
        if variant != "no_context":
            self.query = nn.Conv2d(dim, dim, 1)
            self.key = nn.Conv2d(dim, dim, 1)
            # A value bias cancels identically under local centering.
            self.value = nn.Conv2d(dim, dim, 1, bias=False)
            neighbors = (2 * radius + 1) ** 2
            self.relative_bias = nn.Parameter(torch.zeros(heads, neighbors))
            self.null_key = nn.Parameter(torch.zeros(heads, dim // heads))
            self.null_bias = nn.Parameter(torch.zeros(heads))
        if variant == "no_revision":
            self.concat_hidden = max(1, 3 * hidden // 2)
            self.revision = nn.Sequential(nn.Conv2d(4 * dim, self.concat_hidden, 1),
                                          nn.SiLU(), nn.Conv2d(self.concat_hidden, channels, 1))
        else:
            self.revision = nn.Sequential(nn.Conv2d(6 * dim, hidden, 1),
                                          nn.SiLU(), nn.Conv2d(hidden, 2 * channels, 1))
        # Full/no_context start from the second view; no_revision from the
        # mean.  Only this last layer learns on the first backward pass.
        nn.init.zeros_(self.revision[-1].weight)
        nn.init.zeros_(self.revision[-1].bias)
        self._geometry_cache = {}

    def _validate_features(self, features):
        if features.dim() != 4 or features.shape[1] != self.channels:
            raise ValueError("features must have shape [B,channels,H,W]")
        height, width = features.shape[-2:]
        if height % self.downsample or width % self.downsample:
            raise ValueError("Canvas dimensions must be divisible by downsample")
        if self.height > height or self.width > width:
            raise ValueError("Physical sensor dimensions exceed the feature canvas")
        if not features.is_floating_point():
            raise ValueError("Feature tensors must be floating point")

    def _geometry(self, height, width, ref):
        key = (height, width, str(ref.device), ref.dtype)
        if key not in self._geometry_cache:
            if len(self._geometry_cache) >= 4:
                self._geometry_cache.clear()
            rows = torch.arange(height, device=ref.device) < self.height
            cols = torch.arange(width, device=ref.device) < self.width
            pixels = (rows[:, None] & cols[None, :])[None, None]
            fraction = F.avg_pool2d(pixels.to(ref.dtype), self.downsample, self.downsample)
            compact = fraction.gt(0)
            patch = F.unfold(compact.to(ref.dtype), 2 * self.radius + 1,
                             padding=self.radius).bool().unsqueeze(1)
            patch = patch & compact.flatten(2).unsqueeze(1)
            self._geometry_cache[key] = (pixels, fraction, compact, patch)
        return self._geometry_cache[key]

    def encode(self, features):
        """Encode one view once; cache the returned U/R for its next pair.

        U and R are [B,dim,H/downsample,W/downsample].  Padding-only queries
        produce zero descriptors, partial boundary patches retain all valid
        pixels, and null_mass is diagnostic rather than calibrated confidence.
        """
        self._validate_features(features)
        pixels, fraction, compact, patch_mask = self._geometry(*features.shape[-2:], features)
        clean = features.masked_fill(~pixels, 0)
        pooled = F.avg_pool2d(clean, self.downsample, self.downsample)
        pooled = pooled / fraction.clamp_min(1.0 / self.downsample ** 2)
        u = self.projection(pooled).masked_fill(~compact, 0)
        batch, _, height, width = u.shape
        positions, neighbors = height * width, (2 * self.radius + 1) ** 2
        kernel = 2 * self.radius + 1
        if self.variant == "no_context":
            values = F.unfold(u, kernel, padding=self.radius).reshape(
                batch, self.dim, neighbors, positions)
            mask = patch_mask[:, 0].to(u.dtype)
            mean = (values * mask[:, None]).sum(2) / mask.sum(1).clamp_min(1)[:, None]
            r = mean.reshape(batch, self.dim, height, width).masked_fill(~compact, 0)
            return {"u": u, "r": r}

        per_head = self.dim // self.heads
        q = self.query(u).reshape(batch, self.heads, per_head, positions)
        keys = F.unfold(self.key(u), kernel, padding=self.radius).reshape(
            batch, self.heads, per_head, neighbors, positions)
        values = F.unfold(self.value(u), kernel, padding=self.radius).reshape(
            batch, self.heads, per_head, neighbors, positions)
        logits = (q.unsqueeze(3) * keys).sum(2) / math.sqrt(per_head)
        logits = logits + self.relative_bias[None, :, :, None]
        logits = logits.masked_fill(~patch_mask, -float("inf"))
        null = (q * self.null_key[None, :, :, None]).sum(2) / math.sqrt(per_head)
        null = null + self.null_bias[None, :, None]
        weights = torch.softmax(torch.cat((logits, null.unsqueeze(2)), 2), dim=2)
        mask = patch_mask.to(values.dtype)
        # The unweighted valid-neighbor reference removes a constant local
        # value response.  It does not assert that those neighbors are all
        # background or that the common B information has been subtracted.
        mean = (values * mask.unsqueeze(2)).sum(3) / mask.sum(2).clamp_min(1).unsqueeze(2)
        local_weights = weights[:, :, :neighbors]
        # Equivalent to sum(alpha * (V - mean)), without materializing another
        # [B,heads,head_dim,K,H*W] centered-value activation (425 MiB at B=64).
        r = (local_weights.unsqueeze(2) * values).sum(3)
        r = r - local_weights.sum(2).unsqueeze(2) * mean
        r = r.reshape(batch, self.dim, height, width).masked_fill(~compact, 0)
        null_mass = weights[:, :, -1].mean(1).reshape(batch, 1, height, width)
        return {"u": u, "r": r, "null_mass": null_mass}

    def _validate_encoded(self, encoded, reference):
        expected = (reference.shape[0], self.dim,
                    reference.shape[2] // self.downsample, reference.shape[3] // self.downsample)
        if not isinstance(encoded, dict):
            raise ValueError("Encoded view must be a dictionary with u and r")
        for name in ("u", "r"):
            value = encoded.get(name)
            if not torch.is_tensor(value) or tuple(value.shape) != expected:
                raise ValueError("Encoded %s must have shape %s" % (name, expected))
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("Encoded views must match feature device and dtype")

    def _upsample(self, value, compact, size):
        valid = compact.to(value.dtype)
        numerator = bilinear_resize(value * valid, size)
        denominator = bilinear_resize(valid, size).clamp_min(1e-12)
        return numerator / denominator

    def fuse(self, firstF, secondF, first_encoded=None, second_encoded=None, return_info=False):
        """Revise a pair, retaining high-resolution F1/F2 paths.

        Both input features must refer to the same padded coordinate system.
        The caller aligns the output events by original B IDs.  Revised
        features are never written back into the backbone's recurrent state.
        """
        self._validate_features(firstF)
        self._validate_features(secondF)
        if (firstF.shape != secondF.shape or firstF.device != secondF.device
                or firstF.dtype != secondF.dtype):
            raise ValueError("The two feature views must have identical shape, device and dtype")
        first_encoded = self.encode(firstF) if first_encoded is None else first_encoded
        second_encoded = self.encode(secondF) if second_encoded is None else second_encoded
        self._validate_encoded(first_encoded, firstF)
        self._validate_encoded(second_encoded, secondF)
        u1, r1 = first_encoded["u"], first_encoded["r"]
        u2, r2 = second_encoded["u"], second_encoded["r"]
        pixels, _, compact, _ = self._geometry(*secondF.shape[-2:], secondF)
        size = secondF.shape[-2:]
        if self.variant == "no_revision":
            residual = self.revision(torch.cat((u1, u2, r1, r2), 1))
            gate_full = secondF.new_full((secondF.shape[0], self.channels, *size), 0.5)
        else:
            conditional = torch.cat((u2, r2, u1 - u2, r1 - r2, u1 * u2, r1 * r2), 1)
            gate, residual = self.revision(conditional).split(self.channels, dim=1)
            gate_full = self._upsample(torch.tanh(gate), compact, size)
        residual_full = self._upsample(residual, compact, size)
        # Padding must not modify valid results or receive a learned update.
        delta = (gate_full * (firstF - secondF) + residual_full).masked_fill(~pixels, 0)
        result = secondF + delta
        if not return_info:
            return result
        count = pixels.sum().to(secondF.dtype) * self.channels
        info = {
            "gate_abs_mean": (gate_full.abs() * pixels).sum((1, 2, 3)) / count,
            "residual_rms": ((residual_full.square() * pixels).sum((1, 2, 3)) / count).sqrt(),
            "correction_rms": (delta.square().sum((1, 2, 3)) / count).sqrt(),
        }
        for label, encoded in (("first", first_encoded), ("second", second_encoded)):
            if "null_mass" in encoded:
                info[label + "_null_mean"] = (encoded["null_mass"] * compact).sum((1, 2, 3)) / compact.sum()
        return result, info

    def forward(self, firstF, secondF, first_encoded=None, second_encoded=None, return_info=False):
        return self.fuse(firstF, secondF, first_encoded, second_encoded, return_info)

    def dense_macs(self, height, width):
        """Steady state: one new encode plus one cached pair's fusion.

        Includes linear projections, QK/AV and null-key products.  Excludes
        pooling, subtraction, multiplication gates, nonlinearities, softmax,
        interpolation and memory traffic.  This is not an energy estimate.
        A cold pair without cached encodings additionally needs one encode.
        """
        if height % self.downsample or width % self.downsample:
            raise ValueError("Canvas dimensions must be divisible by downsample")
        positions = (height // self.downsample) * (width // self.downsample)
        d, c, k = self.dim, self.channels, (2 * self.radius + 1) ** 2
        encode = 0 if c == d else c * d
        if self.variant != "no_context":
            encode += 3 * d * d + 2 * k * d + d
        if self.variant == "no_revision":
            fusion = 4 * d * self.concat_hidden + self.concat_hidden * c
        else:
            fusion = 6 * d * self.hidden + self.hidden * 2 * c
        return int(positions * (encode + fusion))
