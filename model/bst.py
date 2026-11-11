"""Shared evidence processing for learned Bayesian sequential testing.

The probability interpretation is an approximation learned by a proper prefix
classification loss, not a guarantee conferred by a sigmoid.  In particular,
attention weights are not calibrated source-association probabilities.

Local equation S1: e_j = Encoder(z_j, p_j, optional_source_features_j), once per source.
Local equation S2: a_ij = sigmoid(A(e_j, relative_ij, c_i));
              v_ij = V(e_j, relative_ij, c_i), a signed, context-dependent value.
Local equation S3: L_i = L_i^prior + sum_h sum_j(a_ijh v_ijh) / sum_j(a_ijh).
Local equation S4: c_i = c_i^0 + Context(weighted_source_summary, evidence_statistics).
See docs/BST_METHOD.md, equations (4)-(6), for the paper-level notation.

S3 above describes the retained legacy mean baseline. normalization='window'
instead normalizes within each source window, then SUMS discounted window
potentials; bounded signed values and window-balanced context/VoW statistics
make this an architectural constraint, not merely a reinterpretation of S3.
See model/bst_window.py and docs/BST_WINDOW_METHOD.md (W1-W7).

With innovation=True, the reported logit is instead z0 + F_d - F_birth,
where F_birth is the differentiable potential of the age-zero prefix and is
frozen for the lifetime of that query. Age zero therefore preserves the BIR
prior exactly; later observations can still revise old tokens in F_d. This
birth reference does not establish calibrated Bayesian likelihood ratios.
With null_evidence=True, a zero-valued competitor permits the raw potential
to vanish with its affinities. It does not force the final innovation logit
back to z0: rejecting all later evidence leaves z0 - F_birth. Visible source
context can influence the re-audit before learned affinity gates are applied.

Old evidence is retained and both its weight and its value can be revised when
new evidence arrives.  Equation S3 REPLACES the total correction; it does not add
the old evidence again.  Its residual is a learned total evidence correction,
not an independently justified sum of log likelihood ratios.  Shared source
features deliberately exclude query-relative geometry.  Normalization acts only
within one source/query feature vector, never across events or future windows.

This module has no event queue or recording I/O.  Callers keep a causal source
cache, append neighbors monotonically, and remove committed queries from calls.
"""
from dataclasses import asdict, dataclass
import math
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F
from model.bst_window import WindowEvidence


@dataclass
class BSTConfig:
    source_features: int = 2
    source_dim: int = 32
    state_dim: int = 16
    relation_dim: int = 8
    heads: int = 2
    window_ms: float = 50.0
    spatial_scale: float = 16.0
    max_wait_steps: int = 10
    normalization: str = "mean"
    revalue: bool = True
    reweight: bool = True
    past_windows: int = 10
    window_gate: bool = True
    bounded_values: bool = True
    null_evidence: bool = False
    innovation: bool = False

    def __post_init__(self):
        for name in ("source_features", "source_dim", "state_dim", "relation_dim", "heads"):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        if self.source_features < 2:
            raise ValueError("source_features must include [z0, polarity]")
        if not math.isfinite(self.window_ms) or self.window_ms <= 0:
            raise ValueError("window_ms must be positive and finite")
        if not math.isfinite(self.spatial_scale) or self.spatial_scale <= 0:
            raise ValueError("spatial_scale must be positive and finite")
        if self.max_wait_steps < 0 or int(self.max_wait_steps) != self.max_wait_steps:
            raise ValueError("max_wait_steps must be a nonnegative integer")
        if self.normalization not in ("mean", "sum", "softmax", "window"):
            raise ValueError("normalization must be mean, sum, softmax or window")
        if int(self.past_windows) != self.past_windows or self.past_windows < 0:
            raise ValueError("past_windows must be a nonnegative integer")
        if self.null_evidence and self.normalization != "window":
            raise ValueError("null evidence requires window aggregation")

    def to_dict(self):
        return asdict(self)


def take_state(state: Dict[str, torch.Tensor], indices: torch.Tensor):
    """Select pending queries without sharing their mutable queue membership."""
    return {key: value[indices] for key, value in state.items()}


def detach_state(state: Dict[str, torch.Tensor]):
    return {key: value.detach() for key, value in state.items()}


def concat_states(states):
    if not states:
        raise ValueError("concat_states needs at least one state")
    keys = set(states[0])
    if any(set(s) != keys for s in states):
        raise ValueError("states have different fields")
    return {key: torch.cat([s[key] for s in states], dim=0) for key in states[0]}


def _costs(fp_cost, fn_cost):
    fp, fn = float(fp_cost), float(fn_cost)
    if not math.isfinite(fp) or not math.isfinite(fn) or fp <= 0 or fn <= 0:
        raise ValueError("false-positive and false-negative costs must be positive and finite")
    return fp, fn


def posterior_decision(logits, fp_cost=1.0, fn_cost=1.0):
    """Return (target_action, model_based_stop_risk).

    Risk(background) = C_FN p and risk(target) = C_FP (1-p).
    The implied threshold is C_FP / (C_FP + C_FN), not necessarily 0.5.
    The probabilities must be calibrated separately before interpreting this as
    real error risk.  This is neither a frequentist error bound nor a guarantee
    against reversal by a later model prediction.
    """
    fp, fn = _costs(fp_cost, fn_cost)
    p = torch.sigmoid(logits)
    risk_bg, risk_tg = fn * p, fp * (1.0 - p)
    action = logits >= math.log(fp / fn)
    return action, torch.where(action, risk_tg, risk_bg)


def stop_risk(logits, fp_cost=1.0, fn_cost=1.0):
    return posterior_decision(logits, fp_cost, fn_cost)[1]


posterior_risk = stop_risk


class ContinuationValueHead(nn.Module):
    """Nonnegative conditional downstream policy cost.

    Targets may include classification loss and later policy waiting costs.
    Add the declared CURRENT one-step waiting/computation costs outside this network.
    Fit conditional expectation BEFORE comparing with stopping risk.  Taking a
    per-sample hindsight minimum of future label errors is an oracle target.
    """
    def __init__(self, in_features, width=32):
        super().__init__()
        if int(in_features) < 1 or int(width) < 1:
            raise ValueError("head dimensions must be positive")
        self.in_features = int(in_features)
        self.net = nn.Sequential(nn.Linear(self.in_features, int(width)), nn.SiLU(),
                                 nn.Linear(int(width), 1))

    def forward(self, features):
        if features.shape[-1] != self.in_features:
            raise ValueError("wrong continuation feature dimension")
        return F.softplus(self.net(features).squeeze(-1))


class BirthValueHead(ContinuationValueHead):
    """Pre-SDU continuation cost head: accepts only already available features.

    Default input is [asinh(arrival_logit), polarity].  Optional cheap backbone
    context may be concatenated by an explicit configuration.  SDU-derived
    features must not be used to claim a pre-SDU computational saving.
    """
    def __init__(self, in_features=2, width=16):
        super().__init__(in_features=in_features, width=width)


class SharedSequentialModel(nn.Module):
    """Per-source expensive encoding and per-query lightweight re-audit.

    The default source input is [arrival_logit, polarity].  `relative` has units
    [dx pixels, dy pixels, dt milliseconds, dw windows].  `age` is the number of
    windows since the query's arrival.  Positive dw beyond age is always masked.
    Neighbors must be an append-only prefix for each query; the count-based no-op
    guard assumes old indices/features are not replaced between calls.
    """
    def __init__(self, config: Optional[BSTConfig] = None, **kwargs):
        super().__init__()
        if config is not None and kwargs:
            raise ValueError("pass config or keyword arguments, not both")
        if isinstance(config, dict):
            config = BSTConfig(**config)
        self.config = config if config is not None else BSTConfig(**kwargs)
        c = self.config
        h, r = c.heads, c.relation_dim
        self.source_encoder = nn.Sequential(
            nn.Linear(c.source_features, c.source_dim), nn.LayerNorm(c.source_dim), nn.SiLU(),
            nn.Linear(c.source_dim, c.source_dim), nn.SiLU())
        self.source_key = nn.Linear(c.source_dim, h * r)
        self.source_value = nn.Linear(c.source_dim, h * r)
        self.source_context = nn.Linear(c.source_dim, c.state_dim)
        self.prior_context = nn.Sequential(nn.Linear(2, c.state_dim), nn.SiLU(),
                                           nn.Linear(c.state_dim, c.state_dim))
        self.query_norm = nn.LayerNorm(c.state_dim)
        self.evidence_query = nn.Linear(c.state_dim, c.state_dim, bias=False)
        self.query_key = nn.Linear(c.state_dim, h * r)
        self.query_value = nn.Linear(c.state_dim, h * r)
        self.relation = nn.Sequential(nn.Linear(4, r), nn.SiLU())
        self.relation_key = nn.Linear(r, h * r, bias=False)
        self.relation_value = nn.Linear(r, h * r, bias=False)
        self.relation_bias = nn.Linear(r, h, bias=False)
        # Avoid constructing CPU literals and copying them to CUDA every step.
        self.register_buffer("relation_scale", torch.tensor([c.spatial_scale, c.spatial_scale,
                                                             c.window_ms, max(c.max_wait_steps, 1)]),
                             persistent=False)
        # No learned time-only drift: posterior updates require new evidence.
        self.age_query = nn.Linear(1, c.state_dim, bias=False)
        self.value_weight = nn.Parameter(torch.zeros(h, r))
        self.value_bias = nn.Parameter(torch.zeros(h))
        self.context_update = nn.Sequential(
            nn.Linear(c.state_dim + 2 * h, c.state_dim), nn.SiLU(),
            nn.Linear(c.state_dim, c.state_dim), nn.Tanh())
        if c.normalization == "window":
            self.window_evidence = WindowEvidence(c.state_dim, h, c.past_windows,
                                                   c.max_wait_steps, c.window_gate, c.bounded_values,
                                                   null_evidence=c.null_evidence)
        # Runtime-only grouping of parallel affine projections. State-dict names,
        # training gradients and learned architecture remain unchanged.
        self.fused_inference = True
        self._inference_projection_cache = {}

    def train(self, mode=True):
        self._inference_projection_cache.clear()
        return super().train(mode)

    def _project_many(self, x, names):
        layers = [getattr(self, name) for name in names]
        # Diagnostic Linear hooks need their original call sites. Their MAC sum
        # is identical to the grouped GEMM; benchmark timing never enables hooks.
        if self.training or torch.is_grad_enabled() or not self.fused_inference or any(layer._forward_hooks for layer in layers):
            return [layer(x) for layer in layers]
        parameters = [p for layer in layers for p in (layer.weight, layer.bias) if p is not None]
        stamp = tuple((id(p), p.data_ptr(), p._version, p.device, p.dtype) for p in parameters)
        entry = self._inference_projection_cache.get(names)
        if entry is None or entry[0] != stamp:
            weight = torch.cat([layer.weight for layer in layers], dim=0)
            if all(layer.bias is None for layer in layers):
                bias = None
            else:
                bias = torch.cat([layer.bias if layer.bias is not None else
                                  layer.weight.new_zeros(layer.out_features) for layer in layers])
            entry = (stamp, weight, bias)
            self._inference_projection_cache[names] = entry
        joined = F.linear(x, entry[1], entry[2])
        return joined.split([layer.out_features for layer in layers], dim=-1)

    @property
    def value_feature_dim(self):
        return self.config.state_dim + 10

    @property
    def value_features_dim(self):
        return self.value_feature_dim

    def encode_sources(self, source_features):
        if source_features.ndim != 2 or source_features.shape[1] != self.config.source_features:
            raise ValueError("source features must have shape [N, source_features]")
        # asinh is invertible, keeps signed log-odds, and avoids data-picked clipping.
        x = torch.cat([torch.asinh(source_features[:, :1]), source_features[:, 1:]], dim=1)
        z = self.source_encoder(x)
        h, r = self.config.heads, self.config.relation_dim
        key, value, context = self._project_many(z, ("source_key", "source_value", "source_context"))
        return {"key": key.reshape(-1, h, r), "value": value.reshape(-1, h, r), "context": context}

    def make_state(self, prior_logit, p):
        if prior_logit.ndim != 1 or p.shape != prior_logit.shape:
            raise ValueError("prior_logit and p must have matching shape [B]")
        context = self.prior_context(torch.stack([torch.asinh(prior_logit), p], dim=-1))
        zero = torch.zeros_like(prior_logit)
        # The streaming executor scatters updates into fields in place.  Each
        # field therefore needs independent storage, including initial zeros.
        # clone preserves the differentiable connection to the arrival score.
        state = {"context": context, "anchor": context.clone(),
                "score": prior_logit.clone(), "previous_score": prior_logit.clone(),
                "prior": prior_logit.clone(), "polarity": p.clone(), "correction": zero.clone(),
                "count": torch.full_like(prior_logit, -1.0), "mass": zero.clone(),
                "neff": zero.clone(), "age": zero.clone()}
        if self.config.normalization == "window":
            state["units"] = zero.clone()
        if self.config.innovation:
            # Birth evidence is a differentiable reference, never a detached
            # learned-feature cache. Keep it even after old sources are revalued.
            state["birth_potential"] = zero.clone()
            state["birth_initialized"] = torch.zeros_like(prior_logit, dtype=torch.bool)
        return state

    @staticmethod
    def _ages(age, ref):
        if isinstance(age, (int, float)):
            return torch.full_like(ref, float(age))
        age = torch.as_tensor(age, dtype=ref.dtype, device=ref.device)
        if age.ndim == 0:
            age = age.expand_as(ref)
        if age.shape != ref.shape:
            raise ValueError("age must be scalar or shape [B]")
        return age

    def prepare_pairs(self, encoded_source, neighbor_indices, relative):
        """Cache query-state-independent source and relative projections.

        This retains gradients and is invalid after an optimizer update.  Its
        memory scales with query-neighbor pairs, so it is NOT an unbounded
        persistent online cache. The online executor retains only arrived pairs
        for live queries and releases their slots on commit. The causal mask is
        still applied at EVERY re-audit; no future token enters a prefix summary
        or decision, and no context-dependent weight/value is cached here.
        """
        c = self.config
        if neighbor_indices.ndim != 2 or relative.shape != tuple(neighbor_indices.shape) + (4,):
            raise ValueError("expected indices [B,T] and relative [B,T,4]")
        b, t = neighbor_indices.shape
        n = encoded_source["key"].shape[0]
        indices = neighbor_indices.long()
        present = (indices >= 0) & (indices < n)
        safe_relative = torch.where(present[..., None], relative, torch.zeros_like(relative))
        if c.normalization == "window":
            dw = safe_relative[..., 3]
            valid_window = torch.isfinite(dw) & (dw == dw.round()) & (dw >= -c.past_windows) & (dw <= c.max_wait_steps)
            if not bool(valid_window.all()):
                raise ValueError("window aggregation requires integer offsets in the declared past/wait horizon")
        rel = self.relation(torch.asinh(safe_relative / self.relation_scale.to(relative.dtype)))
        rk, rv, rb = self._project_many(rel, ("relation_key", "relation_value", "relation_bias"))
        rk = rk.reshape(b, t, c.heads, c.relation_dim)
        rv = rv.reshape(b, t, c.heads, c.relation_dim)
        gather = indices.clamp(min=0, max=max(n - 1, 0))
        def read(name):
            x = encoded_source[name]
            if n == 0:
                return x.new_zeros((b, t) + tuple(x.shape[1:]))
            return x[gather]
        return {"key": read("key") + rk, "value": read("value") + rv,
                "context": read("context"), "bias": rb,
                "present": present, "dw": safe_relative[..., 3]}

    def reaudit(self, encoded_source, neighbor_indices, relative, state, prior_logit, p,
                age, valid=None, return_tokens=False, force_update=False, pair_cache=None,
                validate_birth=True):
        """Return (logits[B], new_state, diagnostics).

        `valid[B]` suppresses changes for padded queries.  Invalid neighbor IDs
        are -1; future neighbors are additionally suppressed by their dw.
        `force_update` is only for controlled ablations or replaced source caches.
        Default repeated calls on the same prefix are exact no-ops for beliefs.
        `pair_cache` can be prepared once per offline training minibatch or
        appended causally by the bounded online executor with prepare_pairs;
        it does not cache any context-dependent gate or value.
        Innovation queries must observe their age-zero prefix first. A late
        first call is allowed only when that birth prefix contains no selected
        evidence, as in the streaming executor's skipped empty births.
        ``validate_birth=False`` is reserved for internal executors that prove
        this ordering themselves; the public forward wrapper always validates.
        Validation of tensor ages can require a device-to-host synchronization.
        """
        c = self.config
        if neighbor_indices.ndim != 2 or relative.shape != tuple(neighbor_indices.shape) + (4,):
            raise ValueError("expected indices [B,T] and relative [B,T,4]")
        b, t = neighbor_indices.shape
        if prior_logit.shape != (b,) or p.shape != (b,):
            raise ValueError("query metadata must have shape [B]")
        if state is None:
            state = self.make_state(prior_logit, p)
        scalar_birth = isinstance(age, (int, float)) and age == 0
        age = self._ages(age, prior_logit)
        if valid is None:
            valid = torch.ones_like(prior_logit, dtype=torch.bool)
        else:
            valid = valid.to(dtype=torch.bool, device=prior_logit.device)
            if valid.shape != (b,):
                raise ValueError("valid must have shape [B]")
        pairs = self.prepare_pairs(encoded_source, neighbor_indices, relative) if pair_cache is None else pair_cache
        if pairs["present"].shape != neighbor_indices.shape:
            raise ValueError("pair cache shape does not match this query batch")
        if c.innovation and validate_birth and not scalar_birth:
            has_birth_tokens = (pairs["present"] & (pairs["dw"] <= 0)).any(-1)
            missing_birth = valid & ~state["birth_initialized"] & (age > 0) & has_birth_tokens
            if bool(missing_birth.any()):
                raise ValueError("innovation requires an age-zero birth reference before a later prefix with birth evidence")
        mask = pairs["present"] & (pairs["dw"] <= age[:, None]) & valid[:, None]
        key = torch.where(mask[..., None, None], pairs["key"], torch.zeros_like(pairs["key"]))
        value = torch.where(mask[..., None, None], pairs["value"], torch.zeros_like(pairs["value"]))
        source_context = torch.where(mask[..., None], pairs["context"], torch.zeros_like(pairs["context"]))
        rb = torch.where(mask[..., None], pairs["bias"], torch.zeros_like(pairs["bias"]))
        # The birth-conditioned anchor is invariant throughout this unroll.
        # Cache it once, without detaching its training graph.
        a0 = state["anchor"]
        context = state["context"]
        # New observations must affect the reinterpretation of old neighbors in
        # this call, not only in a later call that might never receive evidence.
        visible_count = mask.sum(-1).to(prior_logit.dtype)
        grouping = None
        if c.normalization == "window":
            grouping = self.window_evidence.summarize_visible(source_context, mask, pairs["dw"])
            visible_summary = grouping["balanced"]
        else:
            visible_summary = source_context.sum(1)
            visible_summary = visible_summary / visible_count[:, None].clamp(min=1.0)
        qctx = self.query_norm(context + self.evidence_query(visible_summary)
                               + self.age_query((age / max(c.max_wait_steps, 1))[:, None]))
        if not c.reweight:
            qkey_ctx = self.query_norm(a0)
        else:
            qkey_ctx = qctx
        vctx = qctx if c.revalue else self.query_norm(a0)
        if c.reweight == c.revalue:
            qkey, qvalue = self._project_many(qkey_ctx, ("query_key", "query_value"))
        else:
            qkey, qvalue = self.query_key(qkey_ctx), self.query_value(vctx)
        qkey = qkey.reshape(b, c.heads, c.relation_dim)
        gate_logit = (key * qkey[:, None]).sum(-1) / math.sqrt(c.relation_dim) + rb
        if c.normalization == "softmax":
            if t:
                masked = gate_logit.masked_fill(~mask[..., None], float("-inf"))
                maximum = masked.max(dim=1, keepdim=True)[0]
                maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
                weights = torch.exp(masked - maximum)
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            else:
                weights = gate_logit
        else:
            weights = torch.sigmoid(gate_logit) * mask[..., None].to(gate_logit.dtype)
        qvalue = qvalue.reshape(b, c.heads, c.relation_dim)
        hidden = F.silu(value + qvalue[:, None])
        values = (hidden * self.value_weight[None, None]).sum(-1) + self.value_bias[None, None]
        values = torch.where(mask[..., None], values, torch.zeros_like(values))
        window_result = None
        if c.normalization == "window":
            values = self.window_evidence.bound_values(values)
            window_result = self.window_evidence(weights, values, source_context, qkey_ctx, grouping)
            weights = window_result["gate"]
            head_correction = window_result["head_correction"]
            mass_h, summary = window_result["mass_h"], window_result["summary"]
            neff = window_result["neff"]
        else:
            mass_h = weights.sum(1)
            numerator = (weights * values).sum(1)
            head_correction = numerator if c.normalization == "sum" else numerator / mass_h.clamp(min=1e-8)
            summary_weight = weights.mean(-1)
            summary_mass = summary_weight.sum(-1)
            summary = torch.bmm(summary_weight[:, None], source_context).squeeze(1)
            summary = summary / summary_mass[:, None].clamp(min=1e-8)
            neff = (mass_h.square() / weights.square().sum(1).clamp(min=1e-8)).mean(-1)
        potential = head_correction.sum(-1)
        context_delta = self.context_update(torch.cat([summary, head_correction, torch.log1p(mass_h)], -1))
        if window_result is not None and "context_support" in window_result:
            context_delta = context_delta * window_result["context_support"][:, None]
        new_context = a0 + context_delta
        count = visible_count
        has_evidence = count > 0
        changed = (count > state["count"]) | bool(force_update)
        update = valid & has_evidence & changed
        birth_potential = None
        correction = potential
        if c.innovation:
            # S1: ell_d = z0 + F(E_<=d, q_d) - F(E_<=0, q_0).
            # Empty births may be skipped online and have zero potential.
            # Otherwise the caller must initialize age zero explicitly.
            birth_potential = torch.where(update & (age == 0) & ~state["birth_initialized"],
                                          potential, state["birth_potential"])
            correction = potential - birth_potential
        candidate_score = prior_logit + correction
        score = torch.where(update, candidate_score, state["score"])
        mass = mass_h.sum(-1)
        # Kish size uses tokens in legacy modes and discounted windows in window
        # mode. Neither measures statistical independence.
        def choose(candidate, old):
            sel = update.reshape((b,) + (1,) * (candidate.ndim - 1))
            return torch.where(sel, candidate, old)
        result = {"context": choose(new_context, state["context"]),
                  "anchor": state["anchor"],
                  "score": score,
                  "previous_score": torch.where(valid, state["score"], state["previous_score"]),
                  "prior": state["prior"], "polarity": state["polarity"],
                  "correction": choose(correction, state["correction"]),
                  "count": torch.where(valid, count, state["count"]),
                  "mass": choose(mass, state["mass"]), "neff": choose(neff, state["neff"]),
                  "age": torch.where(valid, age, state["age"])}
        if window_result is not None:
            result["units"] = choose(window_result["units"], state["units"])
        if birth_potential is not None:
            result["birth_potential"] = birth_potential
            result["birth_initialized"] = state["birth_initialized"] | valid
        diag = {"correction": result["correction"], "delta": score - state["score"],
                "mass": result["mass"], "neff": result["neff"], "count": result["count"],
                "updated": update, "head_correction": head_correction}
        if birth_potential is not None:
            diag["birth_potential"] = birth_potential
        if return_tokens:
            diag.update({"gate": weights, "value": values, "mask": mask})
            if window_result is not None:
                diag.update({key: window_result[key] for key in ("effective_weight", "window_discount",
                              "window_value", "window_active", "units")})
                diag.update({key: window_result[key] for key in ("window_support", "null_mass", "effective_window_mass")
                             if key in window_result})
        return score, result, diag

    def value_features(self, state, prior_logit=None, age=None):
        """Causal, cheap VoW input; no future or labels are accepted."""
        score = state["score"]
        prior = state["prior"] if prior_logit is None else prior_logit
        elapsed = state["age"] if age is None else self._ages(age, score)
        horizon = max(self.config.max_wait_steps, 1)
        scalars = [torch.asinh(score), torch.asinh(prior), state["polarity"],
                   torch.asinh(state["correction"]), torch.asinh(score - state["previous_score"]),
                   torch.log1p(state.get("units", state["count"]).clamp(min=0)), torch.log1p(state["mass"]),
                   torch.log1p(state["neff"]), elapsed / horizon,
                   (self.config.max_wait_steps - elapsed).clamp(min=0) / horizon]
        return torch.cat([state["context"], torch.stack(scalars, dim=-1)], dim=-1)

    def forward(self, source_features, neighbor_indices, relative, prior_logit, p, age,
                state=None, valid=None, return_tokens=False):
        """One-step convenience wrapper; use encode_sources + reaudit for caching."""
        return self.reaudit(self.encode_sources(source_features), neighbor_indices, relative,
                            state, prior_logit, p, age, valid=valid, return_tokens=return_tokens)
