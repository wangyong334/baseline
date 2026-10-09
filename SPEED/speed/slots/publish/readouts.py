"""Slot 8: readouts decide, for every event, a final score and the step at which it is published.

Every readout follows the same per-stream protocol, driven by the system once per step k:
    begin(n_events)
    step(ctx)            ctx: k, idx (file indices of step-k events), b/y/x, logits (mark), prob = sigmoid(logits)
                         as float32 numpy (computed once per chunk), total (event counts),
                         verifier state after step k (None without a verifier)
    flush()              end of the stream
    results(threshold) -> {name: (prob float32 [N], publish_step int64 [N])}
Base variants:
    net          sigmoid(mark), published at the end of the event's own step (zero wait)
    fixed_delay  sigmoid(mark + w F(d)) for each d in delays, published d steps later (V2-1 fused_dD;
                 truncated at the last step)
    publish      V3 wait-safe publishing units (two-sided sequential test, optional anchored evidence chain)
V4-2 variants (speed.slots.publish.learned, with the learned_evidence verifier): learned_delay, learned_publish.
"""
import math

import numpy as np
import torch

from speed.slots.verify.drift_cusum import TubeReadout, anchored_accumulate, mixture_log_weights

UPPER, LOWER, DEADLINE, EOS = 0, 1, 2, 3
REASONS = ("upper", "lower", "deadline", "eos")
COLLAPSES = ("linear", "step")


def refill_by_index(n_events, index_parts, value_parts, dtype=np.float32):
    out = np.zeros(n_events, dtype=dtype)
    hits = np.zeros(n_events, dtype=np.int64)
    for idx, val in zip(index_parts, value_parts):
        val = np.asarray(val)
        if idx.shape[0] != val.shape[0]:
            raise AssertionError("index / value length mismatch")
        out[idx] = val
        np.add.at(hits, idx, 1)
    if not np.all(hits == 1):
        raise AssertionError("refill failed: %d events unset, %d set twice"
                             % (int(np.sum(hits == 0)), int(np.sum(hits > 1))))
    return out


class NetReadout(object):
    needs_verifier = False

    def __init__(self, name="net"):
        self.name = name

    def begin(self, n_events):
        self.n_events, self.idx, self.prob, self.when = n_events, [], [], []

    def step(self, ctx):
        self.idx.append(ctx["idx"])
        self.prob.append(ctx["prob"])
        self.when.append(np.full(ctx["idx"].shape[0], ctx["k"], np.int64))

    def flush(self):
        pass

    def results(self, threshold):
        return {self.name: (refill_by_index(self.n_events, self.idx, self.prob),
                            refill_by_index(self.n_events, self.idx, self.when, np.int64))}


class FixedDelayReadout(object):
    needs_verifier = True

    def __init__(self, verifier, delays, weight=1.0, prefix="fused_d"):
        self.verifier, self.delays, self.weight, self.prefix = verifier, sorted(int(d) for d in delays), float(weight), prefix

    def begin(self, n_events):
        self.n_events = n_events
        self.tube = TubeReadout(self.verifier, self.delays)
        self.info = {}
        self.parts = {d: ([], [], []) for d in self.delays}

    def _collect(self, results):
        for key, delay, scores, published in results:
            idx, logit = self.info[key]
            part = self.parts[delay]
            part[0].append(idx)
            part[1].append(torch.sigmoid(logit + self.weight * scores).float().cpu().numpy())
            part[2].append(np.full(idx.shape[0], published, np.int64))

    def step(self, ctx):
        self.info[ctx["k"]] = (ctx["idx"], ctx["logits"])
        self._collect(self.tube.step(ctx["verifier"], ctx["k"], ctx["b"], ctx["y"], ctx["x"], ctx["k"]))

    def flush(self):
        self._collect(self.tube.flush())

    def results(self, threshold):
        out = {}
        for d, (idx, prob, when) in self.parts.items():
            out["%s%d" % (self.prefix, d)] = (refill_by_index(self.n_events, idx, prob),
                                              refill_by_index(self.n_events, idx, when, np.int64))
        return out


class PublishRule(object):
    """z = m + w F(d) (gate: m < g keeps only the negative part of F); before the deadline publish target if
    z >= theta + a s(d), background if z <= theta - b s(d); at the deadline publish 1[z >= theta]."""

    def __init__(self, theta, upper, lower, deadline, collapse="linear", gate=None, weight=1.0):
        if int(deadline) != float(deadline) or int(deadline) < 0:
            raise ValueError("deadline must be a non-negative integer number of steps")
        if float(upper) < 0 or float(lower) <= 0:
            raise ValueError("need upper >= 0 and lower > 0")
        if collapse not in COLLAPSES:
            raise ValueError("collapse must be one of %s" % (COLLAPSES,))
        self.theta, self.upper, self.lower = float(theta), float(upper), float(lower)
        self.deadline, self.collapse = int(deadline), collapse
        self.gate = None if gate is None else float(gate)
        self.weight = float(weight)

    def margins(self, age):
        s = 1.0 if self.collapse == "step" else 1.0 - float(age) / float(self.deadline)
        return self.upper * s, self.lower * s

    def score(self, logit, evidence):
        if self.gate is None:
            return logit + self.weight * evidence
        positive = (evidence + abs(evidence)) * 0.5
        return logit + self.weight * (evidence - positive * ((logit < self.gate) * 1.0))

    def decide(self, z, age):
        if age >= self.deadline:
            up = z >= self.theta
            return up, ~up
        a, b = self.margins(age)
        up = z >= self.theta + a
        return up, (z <= self.theta - b) & ~up


class PublishUnits(object):
    """Online publishing units of one stream (one pending unit per not-yet-published event)."""

    def __init__(self, verifier, rule, anchor=False):
        self.verifier, self.rule, self.anchor = verifier, rule, bool(anchor)
        self.pending = []
        self.last_k = -1

    def _score(self, entry, fresh=False):
        logit = entry["logit"]
        if fresh:
            evidence = torch.zeros_like(logit)
        else:
            logw = mixture_log_weights(self.verifier, entry)
            if logw is None:
                evidence = torch.logsumexp(entry["run"], dim=0) - math.log(self.verifier.n_hypotheses)
            else:
                evidence = torch.logsumexp(entry["run"].to(logw.dtype) + logw, dim=0).to(logit.dtype)
        return self.rule.score(logit, evidence)

    def _decide(self, entry, z, age, k):
        up, down = self.rule.decide(z, age)
        forced = age >= self.rule.deadline
        records = []
        for mask, why in ((up, DEADLINE if forced else UPPER), (down, DEADLINE if forced else LOWER)):
            if bool(mask.any()):
                records.append((entry["key"], entry["pos"][mask], up[mask], z[mask], int(k), why, int(age)))
        rest = ~(up | down)
        if forced or not bool(rest.any()):
            return records, None
        kept = {name: entry[name][rest] for name in ("b", "y", "x", "pos", "logit")}
        kept.update(k=entry["k"], key=entry["key"], run=entry["run"][:, rest])
        if self.anchor:
            kept["alive"] = entry["alive"][:, rest]
        return records, kept

    def step(self, state, k, b, y, x, key, logit, support=None):
        out = []
        k = int(k)
        if self.anchor and support is None:
            raise ValueError("anchored publishing needs the support field of every step")
        if self.pending:
            cat = lambda name: (torch.cat([e[name] for e in self.pending])  # noqa: E731
                                if len(self.pending) > 1 else self.pending[0][name])
            k_from = torch.cat([torch.full_like(e["y"], e["k"]) for e in self.pending])
            where = (cat("b"), cat("y"), cat("x"))
            values = self.verifier.gather_along_many(state["ell"], k, *where, k_from)
            if self.anchor:
                V = self.verifier.n_hypotheses
                held = self.verifier.gather_along_many(support.expand(-1, V, -1, -1), k, *where, k_from) > 0
            start, kept = 0, []
            for entry in self.pending:
                n = int(entry["y"].shape[0])
                if self.anchor:
                    entry["run"], entry["alive"] = anchored_accumulate(entry["run"], entry["alive"],
                                                                       values[:, start:start + n],
                                                                       held[:, start:start + n])
                else:
                    entry["run"] = entry["run"] + values[:, start:start + n]
                start += n
                records, rest = self._decide(entry, self._score(entry), k - entry["k"], k)
                out += records
                if rest is not None:
                    kept.append(rest)
            self.pending = kept
        n = int(y.shape[0])
        if n:
            entry = {"k": k, "key": key, "b": b, "y": y, "x": x, "logit": logit,
                     "pos": torch.arange(n, device=y.device),
                     "run": logit.new_zeros(self.verifier.n_hypotheses, n)}
            if self.anchor:
                entry["alive"] = torch.ones(self.verifier.n_hypotheses, n, dtype=torch.bool, device=y.device)
            records, rest = self._decide(entry, self._score(entry, fresh=True), 0, k)
            out += records
            if rest is not None:
                self.pending.append(rest)
        self.last_k = k
        return out

    def flush(self):
        out = []
        for entry in self.pending:
            z = self._score(entry, fresh=entry["k"] == self.last_k)
            out.append((entry["key"], entry["pos"], z >= self.rule.theta, z, int(self.last_k), EOS,
                        int(self.last_k - entry["k"])))
        self.pending = []
        return out


class MotionProbe(object):
    """Evaluation only: records the network's motion output (vy, vx px/ms, log sigma) at every event of its step."""
    needs_verifier = False
    name = "motion_probe"

    def begin(self, n_events):
        self.n_events, self.idx, self.values = n_events, [], []

    def step(self, ctx):
        m = ctx.get("motion")
        if m is None:
            raise ValueError("motion_probe needs a network with a motion head")
        self.idx.append(ctx["idx"])
        self.values.append(m[ctx["b"], :, ctx["y"], ctx["x"]].float().cpu().numpy())

    def flush(self):
        pass

    def results(self, threshold):
        out = np.zeros((self.n_events, 3), np.float32)
        for idx, val in zip(self.idx, self.values):
            out[idx] = val
        self.extra = {"motion": out}
        return {}


def published_probability(z, label, theta, threshold):
    """sigmoid(z - theta + logit(threshold)), clamped to the side of the published label (so prob >= threshold
    reproduces the label exactly in float32)."""
    thr = np.float32(threshold)
    shift = math.log(float(threshold) / (1.0 - float(threshold))) - float(theta)
    prob = (1.0 / (1.0 + np.exp(-(np.asarray(z, np.float64) + shift)))).astype(np.float32)
    below = np.nextafter(thr, np.float32(0.0))
    return np.where(label, np.maximum(prob, thr), np.minimum(prob, below)).astype(np.float32)


class PublishReadout(object):
    needs_verifier = True

    def __init__(self, verifier, rule, anchor=True, name="pub"):
        self.verifier, self.rule, self.anchor, self.name = verifier, rule, bool(anchor), name

    def begin(self, n_events):
        self.n_events = n_events
        self.units = PublishUnits(self.verifier, self.rule, anchor=self.anchor)
        self.info = {}
        self.parts = {name: ([], []) for name in ("z", "label", "window", "reason", "age")}

    def _collect(self, records):
        for key, pos, label, z, published, reason, age in records:
            idx = self.info[key][pos.cpu().numpy()]
            n = idx.shape[0]
            values = {"z": z.detach().to("cpu", torch.float64).numpy(), "label": label.cpu().numpy(),
                      "window": np.full(n, published, np.int64), "reason": np.full(n, reason, np.int64),
                      "age": np.full(n, age, np.int64)}
            for name, value in values.items():
                self.parts[name][0].append(idx)
                self.parts[name][1].append(value)

    def step(self, ctx):
        self.info[ctx["k"]] = ctx["idx"]
        support = self.verifier.support_field(ctx["total"]) if self.anchor else None
        self._collect(self.units.step(ctx["verifier"], ctx["k"], ctx["b"], ctx["y"], ctx["x"], ctx["k"],
                                      ctx["logits"], support))

    def flush(self):
        self._collect(self.units.flush())

    def results(self, threshold):
        z = refill_by_index(self.n_events, *self.parts["z"], dtype=np.float64)
        label = refill_by_index(self.n_events, *self.parts["label"], dtype=np.uint8).astype(bool)
        when = refill_by_index(self.n_events, *self.parts["window"], dtype=np.int64)
        self.extra = {"z": z, "age": refill_by_index(self.n_events, *self.parts["age"], dtype=np.int64),
                      "reason": refill_by_index(self.n_events, *self.parts["reason"], dtype=np.int64)}
        return {self.name: (published_probability(z, label, self.rule.theta, threshold), when)}


def build_readouts(cfgs, verifier, threshold):
    out = []
    for cfg in cfgs:
        kind = cfg["kind"]
        if kind == "net":
            out.append(NetReadout(cfg.get("name", "net")))
        elif kind == "fixed_delay":
            out.append(FixedDelayReadout(verifier, cfg["delays_steps"], cfg.get("fusion_weight", 1.0)))
        elif kind == "publish":
            theta = cfg.get("theta")
            if theta is None:
                theta = math.log(float(threshold) / (1.0 - float(threshold)))
            rule = PublishRule(theta, cfg.get("upper", 1.0), cfg.get("lower", 2.0), cfg.get("deadline_steps", 5),
                               cfg.get("collapse", "linear"), cfg.get("gate"), cfg.get("fusion_weight", 1.0))
            out.append(PublishReadout(verifier, rule, cfg.get("anchor", True), cfg.get("name", "pub")))
        elif kind in ("learned_delay", "learned_publish"):
            from speed.slots.publish.learned import LearnedDelayReadout, LearnedPublishReadout
            if not hasattr(verifier, "event_cloud"):
                raise ValueError("readout %s needs the learned_evidence verifier" % kind)
            if kind == "learned_delay":
                out.append(LearnedDelayReadout(verifier, cfg["delays_steps"], cfg.get("prefix", "fused_d")))
            else:
                theta = cfg.get("theta")
                if theta is None:
                    theta = math.log(float(threshold) / (1.0 - float(threshold)))
                out.append(LearnedPublishReadout(verifier, theta, cfg["epsilon"], cfg.get("name", "pub")))
        else:
            raise ValueError("unknown readout kind %s" % kind)
        if kind != "net" and verifier is None:
            raise ValueError("readout %s needs a verifier" % kind)
    return out
