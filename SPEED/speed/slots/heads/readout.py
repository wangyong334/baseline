"""V4-2: the learned parameters of the evidence readouts (registered in the network: trained and saved with it).

    fusion      w_d per delay d = 1..D: fused logit z_d = mark + w_d * F_d (V2 used w = 1 for every d; V4-38)
    mixture     log weights of the position offsets o in a (2r+1)^2 window around a path (V2: uniform 3x3 log-mean-exp;
                initialised to that, the outer ring starts at a negligible weight and can grow)
    stability   q_d = P(the decision at delay d equals the decision at the deadline D | z_d - theta, F_d, d / D):
                the learned "wait-safe" test that replaces V3's hand-chosen margins (V3 idea kept; PABEE-like
                early exit, but learned); a unit publishes once q_d >= 1 - epsilon (epsilon = tolerated flip rate)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

OUTER_INIT = -8.0


class EvidenceReadoutHead(nn.Module):
    def __init__(self, max_delay=5, radius=2, stability_hidden=8):
        super(EvidenceReadoutHead, self).__init__()
        self.max_delay, self.radius = int(max_delay), int(radius)
        if self.max_delay < 1 or self.radius < 1:
            raise ValueError("need max_delay >= 1 and radius >= 1")
        self.fusion = nn.Parameter(torch.ones(self.max_delay))
        r = self.radius
        offsets = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.float32), persistent=False)
        logits = torch.tensor([0.0 if max(abs(dy), abs(dx)) <= 1 else OUTER_INIT for dy, dx in offsets])
        self.mix_logits = nn.Parameter(logits)
        last = nn.Linear(int(stability_hidden), 1)
        self.stability = nn.Sequential(nn.Linear(3, int(stability_hidden)), nn.ReLU(), last)
        self.stability_hidden = int(stability_hidden)

    def fusion_weight(self, delay):
        d = int(delay)
        if not 1 <= d <= self.max_delay:
            raise ValueError("delay %d outside 1..%d" % (d, self.max_delay))
        return self.fusion[d - 1]

    def mix_log_weights(self):
        return F.log_softmax(self.mix_logits, 0)

    def stability_logit(self, margin, evidence, fraction):
        """margin = z_d - theta, evidence = F_d, fraction = d / D (tensors of one shape) -> logit of q_d."""
        x = torch.stack([margin, evidence, fraction], -1).to(self.fusion.dtype)
        return self.stability(x).squeeze(-1)

    def operations_per_unit(self):
        h = self.stability_hidden
        return {"mac": 3 * h + h, "ac": h + 1, "transcendental": 1.0}


def log_epsilon_bound(epsilon):
    """Publish when q_d >= 1 - epsilon, i.e. when the stability logit >= logit(1 - epsilon)."""
    e = float(epsilon)
    if not 0.0 < e < 1.0:
        raise ValueError("epsilon must lie in (0, 1)")
    return math.log((1.0 - e) / e)
