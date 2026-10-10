"""Slot 8, V4-3: readouts on the tube evidence (the functions of slot 7 that the loss of slot 9 also trains).

    learned_delay    z_d = m + w_d F_d after d steps, probability sigmoid(z_d)        (V2-1 fused_dD with learned w_d)
    learned_publish  learned wait-safe publishing (V3 idea, learned rule): at age d (z_0 = m, F_0 = 0) the stability
                     head gives q_d = P(1[z_d >= theta] equals the decision at the deadline D | z_d - theta, F_d, d / D);
                     an event is published with label 1[z_d >= theta] once q_d >= 1 - epsilon, at the latest at d = D
    motion_probe     evaluation only: the most probable motion and the anchors kept for every event
"""
import numpy as np
import torch

from speed.slots.heads.readout import log_epsilon_bound
from speed.slots.publish.readouts import DEADLINE, EOS, LOWER, UPPER, published_probability, refill_by_index
from speed.slots.verify.tube_evidence import score, tube_step

PER_EVENT = ("b", "y", "x", "logit", "pos", "F")
PER_HYPOTHESIS = ("run", "alive", "logw")


class EvidenceChains(object):
    """Pending events of one stream, one entry per birth step: hypotheses fixed at birth, anchored evidence so far."""

    def __init__(self, verifier):
        self.verifier = verifier
        self.pending = []

    def add(self, k, key, b, y, x, logit, disp, logw, pos=None):
        n = int(y.shape[0])
        if n == 0:
            return
        K = int(disp.shape[0])
        self.pending.append({"k": int(k), "key": key, "b": b, "y": y, "x": x, "logit": logit, "disp": disp,
                             "logw": logw, "pos": torch.arange(n, device=y.device) if pos is None else pos,
                             "run": logit.new_zeros(K, n), "F": torch.zeros_like(logit),
                             "alive": torch.ones(K, n, dtype=torch.bool, device=y.device)})

    def advance(self, state, k):
        """Adds the evidence of step k to every pending event."""
        log_pi = self.verifier.position_log_weights()
        for e in self.pending:
            b = e["b"]
            e["run"], e["alive"] = tube_step(state["N"], b, state["mu"], b, state["g_prev"], b, state["support"],
                                             e["y"], e["x"], e["disp"], int(k) - e["k"], e["run"], e["alive"], log_pi)
            e["F"] = score(e["run"], e["logw"])

    @staticmethod
    def subset(entry, mask):
        out = {"k": entry["k"], "key": entry["key"], "disp": entry["disp"][:, mask]}
        out.update({name: entry[name][mask] for name in PER_EVENT})
        out.update({name: entry[name][:, mask] for name in PER_HYPOTHESIS})
        return out


def step_hypotheses(ctx):
    hyp = ctx["verifier"]["hyp"]
    if hyp is None:
        raise ValueError("the tube_evidence verifier gave no hypotheses for a step with events")
    return hyp["disp"], hyp["logw"]


class LearnedDelayReadout(object):
    """fusion_weight None: the learned w_d; a number: that fixed weight for every delay (e.g. 1.0 = V2)."""
    needs_verifier = True

    def __init__(self, verifier, delays, prefix="fused_d", fusion_weight=None):
        self.verifier, self.head, self.prefix = verifier, verifier.head, prefix
        self.fixed_weight = None if fusion_weight is None else float(fusion_weight)
        self.delays = sorted(set(int(d) for d in delays))
        if not self.delays or self.delays[0] < 1 or self.delays[-1] > self.head.max_delay:
            raise ValueError("delays must lie in 1..%d (the readout head's max_delay)" % self.head.max_delay)
        self.max_delay = self.delays[-1]

    def weight(self, d):
        return self.head.fusion_weight(d) if self.fixed_weight is None else torch.tensor(self.fixed_weight)

    def begin(self, n_events):
        self.n_events, self.chains, self.last_k = n_events, EvidenceChains(self.verifier), -1
        self.parts = {d: ([], [], []) for d in self.delays}

    def _collect(self, entry, d, published):
        with torch.no_grad():
            logit = entry["logit"]
            z = logit + self.weight(d).to(logit.device, logit.dtype) * entry["F"].to(logit.dtype)
        idx, part = entry["key"], self.parts[d]
        part[0].append(idx)
        part[1].append(torch.sigmoid(z).float().cpu().numpy())
        part[2].append(np.full(idx.shape[0], published, np.int64))

    def step(self, ctx):
        k = int(ctx["k"])
        with torch.no_grad():
            self.chains.advance(ctx["verifier"], k)
        for entry in self.chains.pending:
            if k - entry["k"] in self.delays:
                self._collect(entry, k - entry["k"], k)
        self.chains.pending = [e for e in self.chains.pending if k - e["k"] < self.max_delay]
        if int(ctx["y"].shape[0]):
            disp, logw = step_hypotheses(ctx)
            self.chains.add(k, ctx["idx"], ctx["b"], ctx["y"], ctx["x"], ctx["logits"], disp, logw)
        self.last_k = k

    def flush(self):
        for entry in self.chains.pending:
            for d in self.delays:
                if d > self.last_k - entry["k"]:
                    self._collect(entry, d, self.last_k)
        self.chains.pending = []

    def results(self, threshold):
        return {"%s%d" % (self.prefix, d): (refill_by_index(self.n_events, idx, prob),
                                             refill_by_index(self.n_events, idx, when, np.int64))
                for d, (idx, prob, when) in self.parts.items()}


class LearnedPublishReadout(object):
    needs_verifier = True

    def __init__(self, verifier, theta, epsilon, name="pub"):
        self.verifier, self.head, self.name = verifier, verifier.head, name
        self.theta, self.epsilon = float(theta), float(epsilon)
        self.bound = log_epsilon_bound(epsilon)
        self.deadline = self.head.max_delay

    def begin(self, n_events):
        self.n_events, self.chains, self.last_k = n_events, EvidenceChains(self.verifier), -1
        self.parts = {name: ([], []) for name in ("z", "label", "window", "reason", "age")}

    def fused(self, logit, F, age):
        if age == 0:
            return logit
        return logit + self.head.fusion_weight(age).to(logit.dtype) * F.to(logit.dtype)

    def _record(self, key, pos, label, z, k, reason, age):
        idx = key[pos.cpu().numpy()]
        n = idx.shape[0]
        lab = label.cpu().numpy()
        values = {"z": z.detach().to("cpu", torch.float64).numpy(), "label": lab,
                  "window": np.full(n, k, np.int64), "age": np.full(n, age, np.int64),
                  "reason": np.full(n, reason, np.int64) if reason is not None else np.where(lab, UPPER, LOWER)}
        for name, value in values.items():
            self.parts[name][0].append(idx)
            self.parts[name][1].append(value)

    def _decide(self, logit, F, age):
        """-> (z, label, publish mask) at this age."""
        with torch.no_grad():
            z = self.fused(logit, F, age)
            label = z >= self.theta
            if age >= self.deadline:
                return z, label, torch.ones_like(label)
            q = self.head.stability_logit(z - self.theta, F, torch.full_like(z, float(age) / self.deadline))
            return z, label, q >= self.bound

    def step(self, ctx):
        k = int(ctx["k"])
        with torch.no_grad():
            self.chains.advance(ctx["verifier"], k)
        kept = []
        for entry in self.chains.pending:
            age = k - entry["k"]
            z, label, go = self._decide(entry["logit"], entry["F"], age)
            if bool(go.any()):
                self._record(entry["key"], entry["pos"][go], label[go], z[go], k,
                             DEADLINE if age >= self.deadline else None, age)
            if not bool(go.all()):
                kept.append(EvidenceChains.subset(entry, ~go) if bool(go.any()) else entry)
        self.chains.pending = kept
        logit = ctx["logits"]
        n = int(logit.shape[0])
        if n:
            z, label, go = self._decide(logit, torch.zeros_like(logit), 0)
            pos = torch.arange(n, device=logit.device)
            if bool(go.any()):
                self._record(ctx["idx"], pos[go], label[go], z[go], k, None, 0)
            rest = ~go
            if bool(rest.any()):
                disp, logw = step_hypotheses(ctx)
                self.chains.add(k, ctx["idx"], ctx["b"][rest], ctx["y"][rest], ctx["x"][rest], logit[rest],
                                disp[:, rest], logw[:, rest], pos[rest])
        self.last_k = k

    def flush(self):
        for entry in self.chains.pending:
            age = self.last_k - entry["k"]
            with torch.no_grad():
                z = self.fused(entry["logit"], entry["F"], age)
            self._record(entry["key"], entry["pos"], z >= self.theta, z, self.last_k, EOS, age)
        self.chains.pending = []

    def results(self, threshold):
        z = refill_by_index(self.n_events, *self.parts["z"], dtype=np.float64)
        label = refill_by_index(self.n_events, *self.parts["label"], dtype=np.uint8).astype(bool)
        when = refill_by_index(self.n_events, *self.parts["window"], dtype=np.int64)
        self.extra = {"z": z, "age": refill_by_index(self.n_events, *self.parts["age"], dtype=np.int64),
                      "reason": refill_by_index(self.n_events, *self.parts["reason"], dtype=np.int64)}
        return {self.name: (published_probability(z, label, self.theta, threshold), when)}


class MotionProbe(object):
    """Evaluation only: per event the most probable motion (vy, vx px/ms) and the indices of the anchors kept."""
    needs_verifier = True
    name = "motion_probe"

    def __init__(self, dt_ms, hypotheses):
        self.dt, self.K = float(dt_ms), int(hypotheses)

    def begin(self, n_events):
        self.n_events, self.idx, self.best, self.anchors = n_events, [], [], []

    def step(self, ctx):
        if not int(ctx["y"].shape[0]):
            return
        hyp = ctx["verifier"]["hyp"]
        self.idx.append(ctx["idx"])
        self.best.append((hyp["best"] / self.dt).float().cpu().numpy())
        self.anchors.append(hyp["idx"].cpu().numpy())

    def flush(self):
        pass

    def results(self, threshold):
        best = np.zeros((self.n_events, 2), np.float32)
        anchors = np.full((self.n_events, self.K), -1, np.int64)
        for idx, b, a in zip(self.idx, self.best, self.anchors):
            best[idx] = b
            anchors[idx] = a
        self.extra = {"velocity": best, "anchors": anchors}
        return {}
