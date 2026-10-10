"""V4-3: the learned parameters of the evidence readouts (registered in the network: trained and saved with it).

    fusion      w_d per delay d = 1..D: fused logit z_d = mark + w_d * F_d (V2 used w = 1 for every d; v4-2 learned
                0.70 ... 0.48, i.e. evidence must be discounted with the wait)
    position    log weights of the 3 x 3 offsets around a tube position (V2: uniform log-mean-exp; initialised to that;
                v4-2's learned 5 x 5 kept 99.96% of the weight inside 3 x 3)
    stability   q_d = P(the decision at delay d equals the decision at the deadline D | z_d - theta, F_d, d / D):
                the learned wait-safe rule (V3 idea); a unit publishes once q_d >= 1 - epsilon
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceReadoutHead(nn.Module):
    def __init__(self, max_delay=5, stability_hidden=8):
        super(EvidenceReadoutHead, self).__init__()
        self.max_delay = int(max_delay)
        if self.max_delay < 1:
            raise ValueError("need max_delay >= 1")
        self.fusion = nn.Parameter(torch.ones(self.max_delay))
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.position_logits = nn.Parameter(torch.zeros(len(offsets)))
        self.stability_hidden = int(stability_hidden)
        self.stability = nn.Sequential(nn.Linear(3, self.stability_hidden), nn.ReLU(), nn.Linear(self.stability_hidden, 1))

    def fusion_weight(self, delay):
        d = int(delay)
        if not 1 <= d <= self.max_delay:
            raise ValueError("delay %d outside 1..%d" % (d, self.max_delay))
        return self.fusion[d - 1]

    def position_log_weights(self):
        return F.log_softmax(self.position_logits, 0)

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
