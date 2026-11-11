"""Window-level evidence potentials with within-window density normalization.

The grouping is an acquisition unit, NOT an independence assertion. Replicating
an entire selected window's fixed tokens leaves normalized summaries unchanged;
adding a distinct window may add evidence. Across-window discount coefficients
are learned, not calibrated reliability probabilities. No token is discarded.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class WindowEvidence(nn.Module):
    def __init__(self, state_dim, heads, past_windows, max_wait, gate=True, bounded=True):
        super().__init__()
        self.groups = int(past_windows + max_wait + 1)
        self.past_windows = int(past_windows)
        self.gated, self.bounded = bool(gate), bool(bounded)
        if gate:
            self.discount = nn.Sequential(nn.Linear(2 * state_dim + 1, state_dim), nn.SiLU(),
                                          nn.Linear(state_dim, heads))
        if bounded:
            # Unit initial potential scale, learned jointly by the prefix loss.
            self.raw_scale = nn.Parameter(torch.full((heads,), math.log(math.expm1(1.))))
        self.register_buffer("window_age", (torch.arange(self.groups).float() - past_windows)
                             / max(max_wait, 1), persistent=False)

    def bound_values(self, values):
        return F.softplus(self.raw_scale)[None, None] * torch.tanh(values) if self.bounded else values

    def group_sum(self, values, index):
        """O(B*T*C) scatter; never materialize a token-by-token attention map."""
        out = values.new_zeros((values.shape[0], self.groups) + values.shape[2:])
        expanded = index.reshape(index.shape + (1,) * (values.ndim - 2)).expand_as(values)
        return out.scatter_add(1, expanded, values)

    @staticmethod
    def token_read(values, index):
        expanded = index.reshape(index.shape + (1,) * (values.ndim - 2))
        return values.gather(1, expanded.expand(index.shape + values.shape[2:]))

    def summarize_visible(self, source_context, mask, dw):
        # prepare_pairs validates the finite integer range of present windows.
        index = (dw + self.past_windows).long().clamp(0, self.groups - 1)
        counts = self.group_sum(mask.to(source_context.dtype)[..., None], index)
        active = counts[..., 0] > 0
        pooled = self.group_sum(source_context, index) / counts.clamp(min=1.)
        units = active.sum(1).to(source_context.dtype)
        balanced = pooled.sum(1) / units[:, None].clamp(min=1.)
        return {"index": index, "active": active, "units": units, "balanced": balanced, "mask": mask}

    def forward(self, weights, values, source_context, query_context, grouping):
        index, active = grouping["index"], grouping["active"]
        # Stabilize each valid affinity BEFORE normalization. A floor on the
        # group denominator would break complete-group replication invariance.
        # Extreme underflow falls back to uniform within-window affinities;
        # invalid tokens remain exactly zero and never become fake evidence.
        epsilon = max(1e-12, torch.finfo(weights.dtype).tiny)
        weights = torch.where(grouping["mask"][..., None], weights.clamp(min=epsilon), torch.zeros_like(weights))
        mass = self.group_sum(weights, index)
        divisor = self.token_read(mass, index)
        within = weights / torch.where(divisor > 0, divisor, torch.ones_like(divisor))
        window_value = self.group_sum(within * values, index)
        window_context = self.group_sum(within.mean(-1)[..., None] * source_context, index)
        if self.gated:
            query = query_context[:, None].expand(-1, self.groups, -1)
            age = self.window_age.to(query).reshape(1, self.groups, 1).expand(query.shape[0], -1, -1)
            discount = torch.sigmoid(self.discount(torch.cat([query, window_context, age], -1)))
        else:
            discount = torch.ones_like(window_value)
        discount = discount * active[..., None].to(discount.dtype)
        effective = within * self.token_read(discount, index)
        correction = (discount * window_value).sum(1)
        mass_h = discount.sum(1)
        summary_weight = discount.mean(-1)
        summary = (window_context * summary_weight[..., None]).sum(1)
        summary = summary / summary_weight.sum(1)[:, None].clamp(min=1e-8)
        neff = (mass_h.square() / discount.square().sum(1).clamp(min=1e-8)).mean(-1)
        return {"gate": weights, "head_correction": correction, "mass_h": mass_h, "summary": summary,
                "neff": neff, "effective_weight": effective, "window_discount": discount,
                "window_value": window_value, "window_active": active, "units": grouping["units"]}
