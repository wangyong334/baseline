"""Window-level evidence potentials with within-window density normalization.

The grouping is an acquisition unit, NOT an independence assertion. Replicating
an entire selected window's fixed tokens leaves normalized summaries unchanged;
adding a distinct window may add evidence. Across-window discount coefficients
are learned, not calibrated reliability probabilities. No token is discarded.

With null_evidence enabled, each observed window also has a zero-valued null
competitor: w_jh = a_jh / (sum_j a_jh + n_window * nu_h), nu_h > 0.
The accepted mass can fall to zero, whereas ordinary normalized attention
cancels uniform attenuation of every affinity. Null normalization keeps the
complete-group replication property by scaling null mass with valid n_window.
Context is a subprobability mixture with the same zero-valued null branch;
neither value nor context is renormalized by accepted mass afterward.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class WindowEvidence(nn.Module):
    def __init__(self, state_dim, heads, past_windows, max_wait, gate=True, bounded=True,
                 null_evidence=False):
        super().__init__()
        self.groups = int(past_windows + max_wait + 1)
        self.past_windows = int(past_windows)
        self.gated, self.bounded = bool(gate), bool(bounded)
        self.null_evidence = bool(null_evidence)
        if gate:
            self.discount = nn.Sequential(nn.Linear(2 * state_dim + 1, state_dim), nn.SiLU(),
                                          nn.Linear(state_dim, heads))
        if bounded:
            # Unit initial potential scale, learned jointly by the prefix loss.
            self.raw_scale = nn.Parameter(torch.full((heads,), math.log(math.expm1(1.))))
        if self.null_evidence:
            # A learned zero-valued competitor per head. Multiplying its mass
            # by the valid token count preserves complete-group replication.
            self.raw_null = nn.Parameter(torch.full((heads,), math.log(math.expm1(1.))))
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
        # Legacy normalization stabilizes affinities before group division;
        # flooring the group denominator would break replication invariance.
        # The null path instead keeps a zero affinity exactly zero.
        epsilon = max(1e-12, torch.finfo(weights.dtype).tiny)
        # The null denominator is already positive for an observed group:
        # zero affinities must stay zero rather than create uniform evidence.
        affinities = weights if self.null_evidence else weights.clamp(min=epsilon)
        weights = torch.where(grouping["mask"][..., None], affinities, torch.zeros_like(weights))
        mass = self.group_sum(weights, index)
        if self.null_evidence:
            counts = self.group_sum(grouping["mask"].to(weights.dtype)[..., None], index)
            null = F.softplus(self.raw_null).to(weights) + epsilon
            denominator = mass + counts * null[None, None]
        else:
            denominator = mass
        divisor = self.token_read(denominator, index)
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
        if self.null_evidence:
            # window_value and window_context already include non-null
            # support. Apply rho once; do not multiply their values by support
            # a second time, or renormalize away the missing/null mass.
            support = mass / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
            effective_mass = discount * support
            mass_h = effective_mass.sum(1)
            units = grouping["units"].clamp(min=1.)
            summary = (effective.mean(-1)[..., None] * source_context).sum(1) / units[:, None]
            neff = (mass_h.square() / effective_mass.square().sum(1).clamp(min=1e-8)).mean(-1)
            null_mass = (1. - support) * active[..., None].to(support.dtype)
            return {"gate": weights, "head_correction": correction, "mass_h": mass_h, "summary": summary,
                    "neff": neff, "effective_weight": effective, "window_discount": discount,
                    "window_value": window_value, "window_active": active, "units": grouping["units"],
                    "window_support": support, "null_mass": null_mass,
                    "effective_window_mass": effective_mass,
                    "context_support": mass_h.mean(-1) / units}
        mass_h = discount.sum(1)
        summary_weight = discount.mean(-1)
        summary = (window_context * summary_weight[..., None]).sum(1)
        summary = summary / summary_weight.sum(1)[:, None].clamp(min=1e-8)
        neff = (mass_h.square() / discount.square().sum(1).clamp(min=1e-8)).mean(-1)
        return {"gate": weights, "head_correction": correction, "mass_h": mass_h, "summary": summary,
                "neff": neff, "effective_weight": effective, "window_discount": discount,
                "window_value": window_value, "window_active": active, "units": grouping["units"]}
