"""V17 spiking decision units (SDU) on shared evidence maps (11-13; equations: docs/V17_METHOD.md).

A clock-driven streaming backbone commits every event when its 50 ms window closes, so events whose evidence is still
arriving (target onsets, object borders) are decided too early. Sequential analysis treats each uncertain event as a
two-hypothesis test (target / background) that starts from the backbone's prior and stops once the evidence suffices;
its neural form is a decision unit that integrates evidence towards a bound (drift-diffusion). The classical test
assumes observations that belong to the test, are independent and are not cut by a deadline. On an event stream none
of this holds, so a decision unit
    reads evidence from shared per-window maps through a fixed foveated set of boxes around its event   (Eq. 1-2)
    weighs every box by the probability that it holds the event's own source (attribution)            (Eq. 3)
    treats the attributed evidence of one window as one observation (attributed mean: the events of one source in
    one window are strongly correlated) and lets every new window add a new observation                (Eq. 4)
    recomputes its membrane from all windows so far: target channels excite, background ones inhibit   (Eq. 5)
    forms its queries from a context updated by its last audit (what it has seen conditions its reading) (Eq. 6)
When it fires (the stopping rule, Eq. 7) is calibrated on held-out recordings: tools/v17/stopping.py.
No neighbour search: the maps are built once per window and shared by every unit; a unit reads a fixed number of boxes.
Cost (11-13 evening): with events as observations a box without events has weight 0, so only boxes with events are
computed (event-driven); the token form "glm" reads each box with a context-modulated linear model of its count
statistics instead of a d-dimensional embedding (about 1/50 of the multiply-accumulates of the form "mlp").
V18 (11-14): the unit is a two-compartment decision neuron. Dendrite: in every window it settles at the
conductance-weighted mean of the box evidence (conductance = attribution x events of the box, the pseudo-observation
is the leak conductance; Eq. 4), which bounds what one window can contribute. Soma: it integrates the dendritic
currents of the windows (perfect integrator, Eq. 5) and fires once at a threshold that collapses with the unit's age
(the calibrated bounds, Eq. 7). Decision spikes as messages (n_msg): units of a first population emit one spike when
they commit (target or background, at their event's pixel); a unit of the second population receives them through
lateral synapses on its soma -- every spike of another unit in one of its boxes adds the learned weight of its type,
box and the unit's age (log(1 + spikes) per box), whether or not the box holds events in that window
(EvidenceMaps(messages=...), CooperativeStreamingLayer). Training adds the martingale term (martingale_penalty):
a posterior's next revision must not be predictable from its current value.
Moving fields (V18, 11-14 night; EvidenceMaps(track=True)): a unit's boxes move with its source instead of staying
on its event's pixel (track-before-detect: the evidence of a moving source is integrated along its path). The unit
reads its first window at its event's pixel; after every window it measures the prior-weighted centroid of the events
in its level-0/1 square and predicts where its source is in the next window (constant velocity). "transport" (main):
the field sits on the event's pixel moved by the source's predicted displacement since the event's window, so the
event keeps its place in the source; "recentre" (ablation): the field sits on the predicted source centre. Measured
without training (V17 test, units of uncertain events): after 3 windows 21% of a target's events are still inside a
fixed +-4 px square, 73% transported, 88% re-centred; but re-centring also puts 84% of the background units next to a
target onto it (fixed 19%, transported 42%), and units of targets faster than 4 px per window end with more misses
than the backbone when the boxes stay fixed.
Existence dendrite (V19, 11-14 evening; SpikingDecisionUnits(dendrite="existence", ring=True, clutter=True,
content="ring"); docs/V19_DESIGN.md): the question a unit answers is whether a target source exists at its event, so
its dendrite computes the integrated-PDA existence likelihood ratio of every window -- miss + sum over the boxes of
events x exp(association prior + content log-likelihood ratio - clutter normaliser) -- instead of the attributed mean
(a state-update form). The association prior is a ring that grows with the window offset (a source of unknown
direction moves away from the event), the content coefficients are shared around each ring (no preferred direction),
and the clutter normaliser divides a box's events by 1 + the clutter expected there (the background-like events of the
unit's square per px). An empty window gives log miss: evidence of absence.
Membership cue (V18, 11-14 night; SpikingDecisionUnits(memb=True)): the same centroid measured in the event's own
window, without the event itself, gives the event's distance from its source; it enters the unit's initial state.
Waiting can confirm THAT a target is there (existence), not WHICH border events belong to it (membership): on V17
val, target events lie a median 1.9 px from that centroid, the background events next to a target 8.2 px (the
backbone's false alarms among them 5.2 px).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEAT = 5                      # log1p(n_same), log1p(n_opp), log1p(q_same), log1p(q_opp), in-frame area fraction
N_MSG = 2                       # V18 decision spikes of the first population in the box: log1p(target), log1p(background)
CENTRE = 4                      # index of the level-0 centre box (it holds the event itself)
Q_SCALE = 4096                  # fixed-point scale of q in the integer maps (exact, order-independent sums)
EPS = 1e-6
TOKENS = ("mlp", "linear", "glm")  # how a unit reads a box (see SpikingDecisionUnits)
TRACK_HALF = 13                 # tracked fields: the centroid is measured in the level-0/1 square (+-13 px)
TRACK_MIN_Q = 4                 # ... when it holds this much prior mass (fixed point: 0.001 expected target events)
TRACK_BOXES = 17                # boxes of levels 0 and 1 in box_layout order (they tile that square)
TRACK_OPS = 25                  # per unit-step: square mass from 17 box sums, two divisions, the update


def box_layout(levels=4):
    """Foveated boxes around an event (Eq. 1): level 0 = the 3 x 3 boxes of 3 px around it, level l > 0 = the ring of
    8 boxes of 3^(l+1) px around level l - 1, so the boxes tile the square of side 3^(levels+1) px centred on the event
    without overlap (fine near the event, coarse far away). Returns rect [K, 4] = (x0, x1, y0, y1) inclusive pixel
    offsets, the centre distance r [K] (px) and the level [K]; K = 1 + 8 * levels."""
    rects, lev = [], []
    for level in range(int(levels)):
        s = 3 ** (level + 1)
        h = s // 2
        for j in (-1, 0, 1):
            for i in (-1, 0, 1):
                if level and not i and not j:
                    continue
                rects.append((i * s - h, i * s + h, j * s - h, j * s + h))
                lev.append(level)
    rect = np.asarray(rects, np.int64)
    r = np.hypot((rect[:, 0] + rect[:, 1]) / 2.0, (rect[:, 2] + rect[:, 3]) / 2.0)
    return rect, r, np.asarray(lev, np.int64)


def martingale_penalty(U, vm, w, centres=tuple(range(-5, 6)), min_eff=8.0):
    """V18 loss term (G1): a posterior sequence is a martingale -- its next revision cannot be predicted from the
    current belief, E[P(k+1) - P(k) | P(k)] = 0 with P = sigmoid(U) on the calibrated scale. For every soft bin of
    the current log-odds (triangular kernels of width 1 centred at -5 .. 5, clipped U) the squared mean drift is
    estimated without the noise term (U-statistic over distinct pairs: ((sum a d)^2 - sum a^2 d^2) / ((sum a)^2 -
    sum a^2), a = kernel x importance weight), and the bins with an effective sample of at least min_eff count
    EQUALLY: rare uncertain beliefs matter as much as the many certain ones (weighting by frequency would let the
    near-zero background drown the penalty). Measured on V17 val / test (11-14): uncertain beliefs drift down by
    0.01-0.03 per window while their true target rate is above the belief."""
    P = torch.sigmoid(U)
    both = vm[:, :-1] * vm[:, 1:]
    d = (P[:, 1:] - P[:, :-1]).reshape(-1)
    cur = U[:, :-1].clamp(min=centres[0], max=centres[-1]).reshape(-1)
    c = torch.as_tensor(centres, dtype=U.dtype, device=U.device)
    a = (1.0 - (cur[:, None] - c[None, :]).abs()).clamp(min=0) * (both * w[:, None]).reshape(-1)[:, None]
    s1, s2 = a.sum(0), (a * a).sum(0)
    num = (a * d[:, None]).sum(0) ** 2 - (a * a * (d * d)[:, None]).sum(0)
    den = s1 * s1 - s2
    eff = s1 * s1 / s2.clamp(min=1e-12)                                  # effective number of samples per bin
    ok = (eff >= min_eff) & (den > 0)
    if not bool(ok.any()):
        return U.sum() * 0.0
    return (num[ok] / den[ok]).clamp(min=0).mean()


def quantised_q(z0):
    """sigmoid(z0) in the fixed-point unit of the maps (int64 tensor)."""
    return torch.round(torch.sigmoid(z0.float()) * Q_SCALE).long()


class EvidenceMaps(object):
    """Integral images of one recording (Eq. 2): for every window and polarity, the event count n and the expected
    number of target events q = sum sigmoid(z0) of each pixel (z0 = calibrated prior of the frozen backbone). Integer
    arithmetic, so every box sum is exact and independent of the summation order; a box sum costs four lookups.
    messages = (x, y, window, type) of the decision spikes of a first population of units (V18): a unit that commits
    at the end of window w emits one spike at its event's pixel, type 1 = target, 0 = background; two more channels
    count them per window (both polarities). track=True (V18 tracked fields) adds the prior-weighted first moments
    sum q x, sum q y (int64: they exceed the int32 range) for the centroid of a square."""

    def __init__(self, x, y, win, pol, z0, n_win, height, width, device, messages=None, track=False):
        dev = torch.device(device)
        self.n_win, self.H, self.W = int(n_win), int(height), int(width)
        self.n_ch = 4 if messages is None else 6
        P = (self.H + 1) * (self.W + 1)
        x, y, win, pol = (torch.as_tensor(np.asarray(a), dtype=torch.long, device=dev) for a in (x, y, win, pol))
        z0 = torch.as_tensor(np.asarray(z0), dtype=torch.float32, device=dev)
        if x.numel() and (int(x.min()) < 0 or int(x.max()) >= self.W or int(y.min()) < 0 or int(y.max()) >= self.H):
            raise ValueError("events outside the %d x %d sensor" % (self.W, self.H))
        C_ = self.n_ch
        flat = torch.zeros(self.n_win * C_ * P, dtype=torch.int32, device=dev)
        base = win * (C_ * P) + (y + 1) * (self.W + 1) + (x + 1)
        flat.index_add_(0, base + pol * P, torch.ones_like(base, dtype=torch.int32))
        flat.index_add_(0, base + (2 + pol) * P, quantised_q(z0).int())
        if messages is not None:
            mx, my, mw, mt = (torch.as_tensor(np.asarray(a), dtype=torch.long, device=dev) for a in messages)
            if mx.numel():
                mb = mw * (C_ * P) + (my + 1) * (self.W + 1) + (mx + 1)
                flat.index_add_(0, mb + (4 + (1 - mt)) * P, torch.ones_like(mb, dtype=torch.int32))
        maps = flat.view(self.n_win, C_, self.H + 1, self.W + 1)
        maps = torch.cumsum(torch.cumsum(maps, 2, dtype=torch.int32), 3, dtype=torch.int32)
        self.maps = maps.permute(0, 2, 3, 1).reshape(self.n_win, P, C_).contiguous()   # [window, pixel, channel]
        self.mom = None
        if track:
            qi = quantised_q(z0)
            mom = torch.zeros(self.n_win * 2 * P, dtype=torch.int64, device=dev)
            b2 = win * (2 * P) + (y + 1) * (self.W + 1) + (x + 1)
            mom.index_add_(0, b2, qi * x)
            mom.index_add_(0, b2 + P, qi * y)
            mom = torch.cumsum(torch.cumsum(mom.view(self.n_win, 2, self.H + 1, self.W + 1), 2), 3)
            self.mom = mom.permute(0, 2, 3, 1).reshape(self.n_win, P, 2).contiguous()
        self.device = dev

    def window_boxes(self, x, y, rect):
        """Corner indices and in-frame area fraction of every box of every event: idx [4, B, K], area [B, K]."""
        rect = torch.as_tensor(rect, dtype=torch.long, device=self.device)
        x0 = (x[:, None] + rect[None, :, 0]).clamp(0, self.W)
        x1 = (x[:, None] + rect[None, :, 1] + 1).clamp(0, self.W)
        y0 = (y[:, None] + rect[None, :, 2]).clamp(0, self.H)
        y1 = (y[:, None] + rect[None, :, 3] + 1).clamp(0, self.H)
        full = ((rect[:, 1] - rect[:, 0] + 1) * (rect[:, 3] - rect[:, 2] + 1)).float()
        area = ((x1 - x0) * (y1 - y0)).float() / full[None, :]
        w1 = self.W + 1
        idx = torch.stack([y1 * w1 + x1, y0 * w1 + x1, y1 * w1 + x0, y0 * w1 + x0])
        return idx, area

    def _corner_sum(self, table, w, idx, sign=(1, -1, -1, 1)):
        """Box sums from the four corner indices idx (each [B] or [B, K]) of windows w ([B]) of an integral table."""
        wi = w if idx[0].dim() == 1 else w[:, None]
        return sum(s * table[wi, i].long() for s, i in zip(sign, idx))

    def track_step(self, state, age, w, cx, cy, ok, own=None):
        """One step of the tracked fields read at (cx, cy) in windows w: measure the prior-weighted centroid of the
        events in the square +-TRACK_HALF px (when it holds at least TRACK_MIN_Q of prior mass and the window exists;
        own = (q, x, y) of an event left out: a unit's own event in its own window) and update state = (px, py, vx, vy,
        ox, oy) [B, 6] (float64) as an alpha-beta tracker with alpha = beta = 1: position = the measurement (else the
        prediction p + v), velocity = measurement - last position (not after the first window, which has no previous
        position; kept without a measurement); in the first window (own given) also the event's offset from its
        source o = event - centroid (0 without a measurement). The next centre is p + v (re-centred) or p + v + o
        (transported). Returns the state and the measurement (has [B], centroid x, y [B] float64)."""
        h, w1 = TRACK_HALF, self.W + 1
        x0, x1 = (cx - h).clamp(0, self.W), (cx + h + 1).clamp(0, self.W)
        y0, y1 = (cy - h).clamp(0, self.H), (cy + h + 1).clamp(0, self.H)
        idx = (y1 * w1 + x1, y0 * w1 + x1, y1 * w1 + x0, y0 * w1 + x0)
        q = self._corner_sum(self.maps[..., 2], w, idx) + self._corner_sum(self.maps[..., 3], w, idx)
        mx, my = self._corner_sum(self.mom[..., 0], w, idx), self._corner_sum(self.mom[..., 1], w, idx)
        if own is not None:
            oq, ox, oy = own
            q, mx, my = q - oq, mx - oq * ox, my - oq * oy
        has = ok & (q >= TRACK_MIN_Q)
        qd = q.clamp(min=1).double()
        mx, my = mx.double() / qd, my.double() / qd
        px, py, vx, vy, ox, oy = state.unbind(1)
        moving = has & (age > 0)
        if own is not None:                                  # the event's place in its source
            ox = torch.where(has, own[1].double() - mx, torch.zeros_like(mx))
            oy = torch.where(has, own[2].double() - my, torch.zeros_like(my))
        return torch.stack([torch.where(has, mx, px + vx), torch.where(has, my, py + vy),
                            torch.where(moving, mx - px, vx), torch.where(moving, my - py, vy), ox, oy], 1), \
            (has, mx, my)

    @staticmethod
    def membership(meas, x, y):
        """Membership cue of units in their own window [B, 2]: log(1 + distance of the event from the prior-weighted
        centroid of the other events of its +-TRACK_HALF px square) and whether that square holds prior mass at all
        (distance 0 then)."""
        has, mx, my = meas
        d = torch.where(has, torch.hypot(mx - x.double(), my - y.double()), torch.zeros_like(mx))
        return torch.stack([torch.log1p(d), has.double()], 1).float()

    def features(self, x, y, win, pol, z0, rect, n_steps, first=0, own_window=True, own_msg=None, trk=None,
                 trk_age=None, return_track=False, follow=None, return_memb=False):
        """Box features of events for the windows win + first .. win + first + n_steps - 1 (Eq. 2):
        feat [B, n_steps, K, N_FEAT (+ N_MSG with messages)] and window mask [B, n_steps] (False past the recording
        end). Polarity is relative to the event (same / opposite); the event itself is removed from its centre box in
        its own window (the first window read when first == 0 and own_window: a streaming caller that reads a later
        window passes False). own_msg = (step, type) of each event's own first-population spike (step -1: none, counted
        from the event's window): it is removed from the box that holds the event's pixel, so a unit hears only the
        decisions of others.
        Moving fields (maps with track=True): follow = "transport" (the event's pixel moved by its source's predicted
        displacement), "recentre" (the predicted source centre; default with track=True) or False (fixed boxes all the
        same); track_step after every window, the first window at the event's pixel, measured without the event
        itself. trk = tracker states [B, 6] after trk_age windows read (a streaming caller with units that read a
        later window), None = units that start at their events; return_track=True also returns the states after these
        steps (None with fixed boxes); return_memb=True also returns the membership cue [B, 2] of units that start
        here (None otherwise)."""
        dev = self.device
        x, y, win, pol = (torch.as_tensor(a, dtype=torch.long, device=dev) for a in (x, y, win, pol))
        z0 = torch.as_tensor(z0, dtype=torch.float32, device=dev)
        B, S = x.shape[0], int(n_steps)
        rect_t = torch.as_tensor(rect, dtype=torch.long, device=dev)
        K = rect_t.shape[0]
        steps = torch.arange(first, first + S, device=dev)
        w = win[:, None] + steps[None, :]
        ok = w < self.n_win
        wc = w.clamp(max=self.n_win - 1)
        if follow is None or follow is True:
            follow = "recentre" if (self.mom is not None or follow) else False
        if follow not in (False, "recentre", "transport"):
            raise ValueError("unknown field %r" % (follow,))
        if follow and self.mom is None:
            raise ValueError("moving fields need maps built with track=True")
        shift = 1.0 if follow == "transport" else 0.0
        fresh = trk is None and first == 0 and own_window   # units that start with their own window
        own = (quantised_q(z0), x, y) if fresh else None
        state, meas = None, None
        zero = torch.zeros(B, dtype=torch.float64, device=dev)
        if not follow:                                       # fixed boxes around the event's pixel
            if trk is not None:
                raise ValueError("tracker states for fixed boxes")
            idx, area = self.window_boxes(x, y, rect)
            tot = torch.zeros(B, S, K, self.n_ch, dtype=torch.long, device=dev)
            for c, s in enumerate((1, -1, -1, 1)):
                tot += s * self.maps[wc[:, :, None], idx[c][:, None, :]].long()
            area = area[:, None, :].expand(B, S, K)
            cx, cy = x[:, None].expand(B, S), y[:, None].expand(B, S)
            if return_memb and fresh and self.mom is not None:
                _, meas = self.track_step(torch.stack([x.double(), y.double(), zero, zero, zero, zero], 1),
                                          torch.zeros_like(x), wc[:, 0], x, y, ok[:, 0], own)
        else:                                                # tracked fields: one window after the other
            if trk is None:
                if first:
                    raise ValueError("a tracked field needs its tracker state to read a later window")
                state, age = torch.stack([x.double(), y.double(), zero, zero, zero, zero], 1), torch.zeros_like(x)
            else:
                state = torch.as_tensor(trk, dtype=torch.float64, device=dev)
                age = torch.as_tensor(trk_age, dtype=torch.long, device=dev)
            tots, areas, cxs, cys = [], [], [], []
            for k in range(S):                               # read where the field is, measure at the source
                pcx, pcy = state[:, 0] + state[:, 2], state[:, 1] + state[:, 3]
                cxk = torch.round(pcx + shift * state[:, 4]).clamp(0, self.W - 1).long()
                cyk = torch.round(pcy + shift * state[:, 5]).clamp(0, self.H - 1).long()
                idx, ar = self.window_boxes(cxk, cyk, rect)
                tots.append(self._corner_sum(self.maps, wc[:, k], tuple(idx)))
                areas.append(ar)
                cxs.append(cxk)
                cys.append(cyk)
                scx = torch.round(pcx).clamp(0, self.W - 1).long()
                scy = torch.round(pcy).clamp(0, self.H - 1).long()
                state, m_k = self.track_step(state, age + k, wc[:, k], scx, scy, ok[:, k], own if k == 0 else None)
                if k == 0:
                    meas = m_k
            tot, area = torch.stack(tots, 1), torch.stack(areas, 1)
            cx, cy = torch.stack(cxs, 1), torch.stack(cys, 1)
        same = pol.view(B, 1, 1, 1).expand(B, S, K, 1)
        n_same = tot[..., 0:2].gather(3, same).squeeze(3)
        n_opp = tot[..., 0:2].gather(3, 1 - same).squeeze(3)
        q_same = tot[..., 2:4].gather(3, same).squeeze(3)
        q_opp = tot[..., 2:4].gather(3, 1 - same).squeeze(3)
        if first == 0 and own_window and trk is None:        # remove the event itself from its own centre box
            n_same[:, 0, CENTRE] -= 1
            q_same[:, 0, CENTRE] -= quantised_q(z0)
        cols = [torch.log1p(n_same.clamp(min=0).float()), torch.log1p(n_opp.float()),
                torch.log1p(q_same.clamp(min=0).float() / Q_SCALE), torch.log1p(q_opp.float() / Q_SCALE), area]
        if self.n_ch == 6:                                    # decision spikes of others: target, background
            msg = tot[..., 4:6].clone()
            if own_msg is not None:
                o_step = torch.as_tensor(np.asarray(own_msg[0]), dtype=torch.long, device=dev) - first
                o_type = torch.as_tensor(np.asarray(own_msg[1]), dtype=torch.long, device=dev)
                hit = (o_step >= 0) & (o_step < S)
                if hit.any():                                 # the box of the field that holds the event's pixel
                    b = torch.nonzero(hit).squeeze(1)
                    s_, t_ = o_step[b], o_type[b]
                    dx, dy = (x[b] - cx[b, s_])[:, None], (y[b] - cy[b, s_])[:, None]
                    inside = ((rect_t[None, :, 0] <= dx) & (dx <= rect_t[None, :, 1]) & (rect_t[None, :, 2] <= dy) &
                              (dy <= rect_t[None, :, 3]))
                    j, there = inside.long().argmax(1), inside.any(1)
                    msg[b[there], s_[there], j[there], 1 - t_[there]] -= 1
            msg = msg.clamp(min=0).float()
            cols += [torch.log1p(msg[..., 0]), torch.log1p(msg[..., 1])]
        feat = torch.stack(cols, -1) * ok[:, :, None, None].float()
        out = (feat, ok)
        if return_track:
            out += (state,)
        if return_memb:                                      # None without moments or for units not starting here
            out += (self.membership(meas, x, y) if (fresh and meas is not None) else None,)
        return out


class SpikingDecisionUnits(nn.Module):
    """One decision unit per uncertain event; membrane U = posterior log-odds that the event belongs to a target.

    Step k (k = 0 .. n_steps - 1) = after the event's window and the k following ones have arrived. Tokens are the boxes
    of those windows (Eq. 1-2), embedded once: content (linear) + slot (window offset, box) -> keys and values. At every
    step each head h re-audits all windows so far with the unit's current context c (Eq. 3-5):
        a_hj = sigmoid(q_h(c, k) . key_j / sqrt(d_h) + b_h - r_j^2 / (2 sigma_h^2))           attribution of token j
        v_hj = W_out(GELU(value_j + W_c c))_h                                                    its evidence
        e_hw = (sum_{j in w} a_hj m_j v_hj + a0_h v0_h) / (sum_{j in w} a_hj m_j + a0_h)       window w: one observation
        U(k) = z0 + sum_h s_h sum_{w <= k} e_hw                                                  new windows accumulate
    Within one window the events of one source are strongly correlated, so they form one observation (attributed
    mean); every new window is a new observation, so the windows add up (the sequential test). m_j = 1 for every box
    (evidence="boxes") or the number of events in the box (evidence="events": the event is the observation unit, and
    a learned pseudo-observation a0, v0 = "nothing of mine here" smooths windows without attributed events, a Bayesian
    pseudo-count; boxes use no pseudo-observation). s_h = +1 for target heads, -1 for background heads (value="ei":
    every value is softplus(.) >= 0, half of the heads excite, half inhibit). Context for the next step:
    c = c0 + Delta(attributed mean value, log attributed mass, evidence) (Eq. 6).
    Ablations: pool "mean" (one attributed mean over all windows: waiting cannot accumulate evidence) or "sum" (every
    token independent), value signed, gate distance (receptive field only), gate none (no attribution: every box of the
    square weighs the same), context off, audit ledger (each window read
    once, evidence added), pseudo=False (no pseudo-observation: a window without attributed events adds nothing; V18
    ablation of the dendrite's leak conductance). Starts exactly at the prior: U(k) = z0 until the output weights move.
    Token form (how a box is read; rounds 1-3 used "mlp"):
        mlp     content + slot -> LayerNorm -> keys / values (d-dim), value head on GELU(value + W_c c)
        linear  the same without LayerNorm: keys / values linear in the features (folded into one 5 x d map + a slot
                table at deployment)
        glm     no embedding: v_hj = (theta_h[age, box] + dtheta_h(c)) . x_j + b_h[age, box] + db_h(c), a linear model
                of the box's normalised count statistics x_j with fixed per-slot coefficients that the context shifts;
                the gate "attribution" reads x_j the same way (gamma_h(c) . x_j); the context reads the attributed mean
                of x per head. A box costs a few multiply-accumulates per head instead of 2 d^2 + d H.
    ctx_ff=False drops the feed-forward block of the context update. In the forms linear / glm the receptive fields
    start at sigma0 * 2^(-1/2) and sigma0 * 2^(1/2) inside every channel (a near and a far reader; target and background
    channels alike, so U = z0 still holds), otherwise the heads of a channel would stay identical under the distance
    gate. Ledger units with pool "mean" (ablation): the membrane is the prior plus the MEAN of the window currents read
    so far (windows do not accumulate). n_msg > 0 (V18): lateral synapses, U(k) += sum over the boxes of the window
    read at k of lat[k, box, type] * log(1 + decision spikes of others of that type); lat starts at 0. track =
    "transport" / "recentre" (True = "recentre"; V18): the unit's boxes move with its source (EvidenceMaps moving
    fields); the unit itself is unchanged, the flag tells its callers how to build its maps and adds the tracker to
    its cost (track_macs). memb=True (V18): the initial state also reads the membership cue (EvidenceMaps.membership:
    the event's distance from its source in its own window), c0 = MLP(prior, polarity, cue)."""

    def __init__(self, n_steps, r, dim=64, heads=4, value="ei", gate="attribution", pool="window", evidence="boxes",
                 context=True, audit="reaudit", tokens="mlp", ctx_ff=True, n_feat=N_FEAT, n_msg=0, track=False,
                 memb=False, pseudo=True, dendrite="mean", ring=False, clutter=False, content="box"):
        super(SpikingDecisionUnits, self).__init__()
        self.n_feat = Fn = int(n_feat)                                      # box statistics read by the dendrite
        self.n_msg = int(n_msg)                                             # V18: decision-spike types (lateral)
        self.track = bool(track)                                            # V18: boxes move with the source
        self.track_mode = "transport" if track == "transport" else "recentre"   # ... how (EvidenceMaps.features)
        self.memb = bool(memb)                                              # V18: membership cue in the start state
        if (value not in ("ei", "signed") or gate not in ("attribution", "distance", "none")
                or pool not in ("window", "mean", "sum") or evidence not in ("boxes", "events")
                or audit not in ("reaudit", "ledger") or tokens not in TOKENS or dendrite not in ("mean", "existence")
                or content not in ("box", "ring")):
            raise ValueError("unknown option value=%s gate=%s pool=%s evidence=%s audit=%s tokens=%s dendrite=%s "
                             "content=%s" % (value, gate, pool, evidence, audit, tokens, dendrite, content))
        if dim % heads or (value == "ei" and heads % 2):
            raise ValueError("dim must be a multiple of heads; value=ei needs an even number of heads")
        if dendrite == "existence" and (evidence != "events" or value != "ei" or pool != "window" or tokens != "glm"
                                        or gate == "attribution"):
            raise ValueError("the existence dendrite needs evidence=events, value=ei, pool=window, tokens=glm and a "
                             "distance (or no) association prior")
        if (content == "ring" and tokens != "glm") or ((ring or clutter) and evidence != "events"):
            raise ValueError("content=ring needs tokens=glm; ring / clutter need evidence=events")
        if clutter and dendrite == "mean" and not pseudo:
            raise ValueError("clutter with the mean dendrite scales the pseudo-observation: pseudo must be on")
        r = np.asarray(r, np.float64)
        self.n_steps, self.n_box, self.dim, self.heads, self.hd = int(n_steps), len(r), int(dim), int(heads), \
            int(dim) // int(heads)
        self.value, self.gate, self.pool, self.evidence = value, gate, pool, evidence
        self.dendrite, self.ring, self.clutter, self.content_share = dendrite, bool(ring), bool(clutter), content
        self.pseudo = evidence == "events" and bool(pseudo) and dendrite == "mean"   # leak conductance (mean form)
        self.context, self.audit, self.tokens, self.ctx_ff = bool(context), audit, tokens, bool(ctx_ff)
        self.register_buffer("r2", torch.as_tensor(r ** 2, dtype=torch.float32))
        # V19 layout buffers: derived from the box layout, not saved, so V18 checkpoints load unchanged
        self.register_buffer("r", torch.as_tensor(r, dtype=torch.float32), persistent=False)
        j = np.arange(len(r))                                              # box_layout order: 9 + 8 per level
        lev = np.where(j < 9, 0, (j - 9) // 8 + 1)
        self.register_buffer("box_area", torch.as_tensor(9.0 ** (lev + 1), dtype=torch.float32),   # px, full box
                             persistent=False)
        ring_of = np.where(j == CENTRE, 0, lev + 1)                        # centre, then one ring per level
        self.register_buffer("ring_idx", torch.as_tensor(ring_of, dtype=torch.long), persistent=False)
        self.n_ring = int(ring_of.max()) + 1
        sign = [1.0] * heads if value == "signed" else [1.0] * (heads // 2) + [-1.0] * (heads // 2)
        self.register_buffer("head_sign", torch.tensor(sign))
        self.register_buffer("feat_mu", torch.zeros(Fn))
        self.register_buffer("feat_sd", torch.ones(Fn))
        self.register_buffer("z_ref", torch.tensor([0.0, 5.0]))           # prior mean / std of the input scaling
        S, K, attr = self.n_steps, self.n_box, gate == "attribution"
        self.c0 = nn.Sequential(nn.Linear(4 if self.memb else 2, dim), nn.GELU(), nn.Linear(dim, dim))
        if tokens == "mlp":                                                # module set of the round 1-3 checkpoints
            self.content = nn.Linear(Fn, dim)
            self.slot = nn.Parameter(torch.randn(S, K, dim) * 0.02)
            self.ln_k, self.ln_q, self.ln_f = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.age = nn.Embedding(S, dim)
            self.wq, self.wk, self.wv = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
            self.v_ctx, self.v_out = nn.Linear(dim, dim), nn.Linear(dim, heads)
            self.wo, self.wm = nn.Linear(dim, dim), nn.Linear(2 * heads, dim)
            self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        else:
            if tokens == "linear":
                self.content = nn.Linear(Fn, dim)
                self.slot = nn.Parameter(torch.randn(S, K, dim) * 0.02)
                if attr:
                    self.ln_q, self.age = nn.LayerNorm(dim), nn.Embedding(S, dim)
                    self.wq, self.wk = nn.Linear(dim, dim), nn.Linear(dim, dim)
                self.v_ctx, self.v_out = nn.Linear(dim, dim), nn.Linear(dim, heads)
                self.wo = nn.Linear(dim, dim)
            else:
                if content == "ring":                                       # isotropic: one per window offset and ring
                    self.coef = nn.Parameter(torch.zeros(S, self.n_ring, heads, Fn))
                    self.bias = nn.Parameter(torch.zeros(S, 1, heads))     # no place bias: geometry is the prior's
                else:
                    self.coef = nn.Parameter(torch.zeros(S, K, heads, Fn))   # per window offset, box and head
                    self.bias = nn.Parameter(torch.zeros(S, K, heads))
                self.dcoef = nn.Linear(dim, heads * (Fn + 1))               # the context shifts them
                if attr:
                    self.gq = nn.Linear(dim, heads * Fn)
                self.wo = nn.Linear(heads * Fn, dim)
            self.wm = nn.Linear(2 * heads, dim)
            if self.ctx_ff:
                self.ln_f = nn.LayerNorm(dim)
                self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        sigma0 = float(math.sqrt(max((r ** 2).mean(), 1.0)))              # RMS box distance of the layout
        log_sigma = torch.full((heads,), math.log(sigma0))
        if tokens != "mlp":                                                # near and far reader in every channel
            groups = [heads] if value == "signed" else [heads // 2, heads // 2]
            log_sigma = log_sigma + torch.cat([torch.linspace(-0.5, 0.5, g) * math.log(2.0) if g > 1 else
                                               torch.zeros(1) for g in groups])
        self.log_sigma = nn.Parameter(log_sigma)
        self.rf_bias = nn.Parameter(torch.zeros(heads))
        if self.pseudo:
            self.sink_logit = nn.Parameter(torch.zeros(heads))             # weight of the pseudo-observation
            self.sink_val = nn.Linear(dim, heads)                          # its evidence, given the context
        if self.n_msg:                                                      # lateral synapses: one weight per age,
            self.lat = nn.Parameter(torch.zeros(S, K, self.n_msg))         # box and spike type (0 = silent start)
        if self.ring:                   # expanding association ring: radius v k, spread^2 sigma^2 + (w k)^2 (px, k =
            self.ring_v = nn.Parameter(torch.full((heads,), -2.0))         # window offset); softplus, starts ~0.13
            self.ring_w = nn.Parameter(torch.full((heads,), -2.0))
        if self.clutter:                # divisive clutter normalisation n / (1 + expected clutter)^kappa, kappa = 1
            self.clutter_k = nn.Parameter(torch.full((heads,), math.log(math.e - 1.0)))
        if dendrite == "existence":     # detection-or-miss: miss probability 1 - P_D of each hypothesis, given c
            self.miss_logit = nn.Parameter(torch.zeros(2))                 # target, background
            self.miss_ctx = nn.Linear(dim, 2)
        zero = (self.dcoef,) if tokens == "glm" else (self.v_out,)
        if dendrite == "existence":
            zero = zero + (self.miss_ctx,)
        with torch.no_grad():                                              # U = z0 and an unused audit at the start
            for lin in zero + (self.wm,) + ((self.sink_val,) if self.pseudo else ()):
                lin.weight.zero_()
                lin.bias.zero_()

    # ------------------------------------------------------------------ building blocks
    def set_normalization(self, feat_mu, feat_sd, z_mu, z_sd):
        with torch.no_grad():
            self.feat_mu.copy_(torch.as_tensor(feat_mu, dtype=torch.float32))
            self.feat_sd.copy_(torch.as_tensor(feat_sd, dtype=torch.float32).clamp(min=1e-4))
            self.z_ref.copy_(torch.tensor([float(z_mu), max(float(z_sd), 1e-4)]))

    def sigma(self):
        return torch.exp(self.log_sigma)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in ("r", "box_area", "ring_idx"):         # saved by the V19 checkpoints of 11-14; rebuilt from the layout
            state_dict.pop(prefix + name, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def start(self, z0, pol, memb=None):
        """Initial context c0 from the event's own prior and polarity (and the membership cue [B, 2] with memb)."""
        z_in = (z0 - self.z_ref[0]) / self.z_ref[1]
        x = torch.stack([z_in, pol.float()], 1)
        if self.memb:
            if memb is None:
                raise ValueError("units with the membership cue need it as input")
            x = torch.cat([x, memb.to(x.dtype)], 1)
        return self.c0(x)

    def embed(self, feat, first=0, slot_idx=None):
        """Tokens of windows first .. first + S - 1 (or, for one window, the per-unit window offset slot_idx [B]):
        feat [B, S, K, N_FEAT] -> dict of [B, S*K, ...]: the event counts "cnt" and, by token form, keys [.., H, hd]
        and values [.., d] (mlp; linear: keys only with the attribution gate), or the normalised features "x" [.., 5]
        and their fixed evidence logit "s" [.., H] (glm)."""
        B, S, K = feat.shape[0], feat.shape[1], feat.shape[2]
        T = S * K
        x = (feat[..., :self.n_feat] - self.feat_mu) / self.feat_sd
        tok = {"cnt": (torch.expm1(feat[..., 0]) + torch.expm1(feat[..., 1])).reshape(B, T)}
        if self.clutter:                                   # background-like events (count - prior mass) and the
            q = torch.expm1(feat[..., 2]) + torch.expm1(feat[..., 3])          # in-frame area (px) of every box
            tok["bg"] = (tok["cnt"].view(B, S, K) - q).clamp(min=0).reshape(B, T)
            tok["px"] = (feat[..., 4] * self.box_area).reshape(B, T)
        if self.n_msg:                                     # lateral current of every box: weight x log(1 + spikes)
            lat = self.lat[first:first + S][None] if slot_idx is None else self.lat[slot_idx][:, None]
            spikes = feat[..., self.n_feat:self.n_feat + self.n_msg]
            tok["lat"] = (lat * spikes).sum(-1).reshape(B, T)
        if self.tokens == "glm":
            cf, bs = self.coef, self.bias
            if self.content_share == "ring":               # one coefficient set per ring, the same around it
                cf, bs = cf[:, self.ring_idx], bs.expand(-1, K, -1)
            coef = cf[first:first + S][None] if slot_idx is None else cf[slot_idx][:, None]
            bias = bs[first:first + S][None] if slot_idx is None else bs[slot_idx][:, None]
            tok["x"] = x.reshape(B, T, self.n_feat)
            tok["s"] = ((coef * x[..., None, :]).sum(-1) + bias).reshape(B, T, self.heads)
            return tok
        slot = self.slot[first:first + S][None] if slot_idx is None else self.slot[slot_idx][:, None]
        kt = self.content(x) + slot
        if self.tokens == "mlp":
            kt = self.ln_k(kt).reshape(B, T, self.dim)
            tok["keys"], tok["vals"] = self.wk(kt).view(B, T, self.heads, self.hd), self.wv(kt)
        else:
            tok["vals"] = kt.reshape(B, T, self.dim)
            if self.gate == "attribution":
                tok["keys"] = self.wk(tok["vals"]).view(B, T, self.heads, self.hd)
        return tok

    def audit_tokens(self, c, k, tok, n_win, win_mask=None):
        """One audit over the tokens tok (embed) of n_win windows (Eq. 3-4); k = step (int or per unit), win_mask
        [B, n_win] marks the windows that have arrived (None = all). Returns the evidence per head lambda [B, H] (summed
        over windows), the attributed mass [B, H], the attributed mean value [B, d] (glm: mean features [B, 5 H]) and
        the gates [B, H, T]."""
        cnt = tok["cnt"]
        B, T = cnt.shape
        K, H = self.n_box, self.heads
        prior = self.association_prior(k, n_win)                           # [H, T] or [B, H, T] (ring, per-unit age)
        pr = prior if prior.dim() == 3 else prior[None]
        if self.dendrite == "existence":
            return self._existence(c, tok, n_win, pr, win_mask)
        if self.gate == "attribution" and self.tokens == "glm":
            q = self.gq(c).view(B, H, self.n_feat)
            logit = torch.einsum("bhf,btf->bht", q, tok["x"]) / math.sqrt(self.n_feat) + pr
        elif self.gate == "attribution":
            q = self.wq(self.ln_q(c + self.age.weight[k])).view(B, H, self.hd)
            logit = torch.einsum("bhd,bthd->bht", q, tok["keys"]) / math.sqrt(self.hd) + pr
        else:
            logit = pr.expand(B, H, T)
        wm = torch.ones(B, n_win, device=cnt.device) if win_mask is None else win_mask.float()
        a = torch.sigmoid(logit) * wm.repeat_interleave(K, 1)[:, None, :]
        if self.tokens == "glm":
            Fn = self.n_feat
            dc = self.dcoef(c).view(B, H, Fn + 1)
            v = tok["s"].permute(0, 2, 1) + torch.einsum("bhf,btf->bht", dc[..., :Fn], tok["x"]) + dc[..., Fn:]
        else:
            v = self.v_out(F.gelu(tok["vals"] + self.v_ctx(c)[:, None, :])).permute(0, 2, 1)   # [B, H, T]
        if self.value == "ei":
            v = F.softplus(v)
        w = a * cnt[:, None, :] if self.evidence == "events" else a
        num = (w * v).view(B, H, n_win, K).sum(-1)
        den = w.view(B, H, n_win, K).sum(-1)
        if self.pseudo:                                                    # pseudo-observation of every window
            a0 = torch.sigmoid(self.sink_logit)[None, :, None] * wm[:, None, :]
            if self.clutter:                    # PDA: "none of these is mine" grows with the clutter of the window
                bg = tok["bg"].view(B, n_win, K).sum(-1)                       # background-like events [B, n_win]
                a0 = a0 * torch.exp(F.softplus(self.clutter_k)[None, :, None] * torch.log1p(bg)[:, None, :])
            v0 = self.sink_val(c)
            if self.value == "ei":
                v0 = F.softplus(v0)
            num = num + a0 * v0[:, :, None]
            den = den + a0
        if self.pool == "window":
            lam = (num / den.clamp(min=EPS) * wm[:, None, :]).sum(-1)
        elif self.pool == "mean":
            lam = num.sum(-1) / den.sum(-1).clamp(min=EPS)
        else:
            lam = num.sum(-1)
        mass = w.sum(-1)
        if self.tokens == "glm":
            ubar = torch.einsum("bht,btf->bhf", w, tok["x"]) / mass[..., None].clamp(min=EPS)
        else:
            ubar = torch.einsum("bht,bthd->bhd", w, tok["vals"].view(B, T, H, self.hd)) / mass[..., None].clamp(min=EPS)
        return lam, mass, ubar.reshape(B, -1), a

    def association_prior(self, k, n_win):
        """Log association prior of the tokens of n_win windows: [H, T], or [B, H, T] when the expanding ring needs a
        per-unit age (k [B], one window). Fixed field: b_h - r^2 / (2 sigma_h^2). Expanding ring (V19): a source of
        unknown direction that moves about v_h px per window is, a windows after the event, on a ring of radius v_h a
        that widens: b_h - (r - v_h a)^2 / (2 s_h(a)^2) - log(s_h(a) (v_h a + s_h(a)) / sigma_h^2), s_h(a)^2 =
        sigma_h^2 + (w_h a)^2 (the last term keeps the mass: the area of that ring, ~ s (R + s), relative to the disc
        of the event's own window); a = the window offset of the token (the step k for one window, 0 .. n_win - 1 for a
        re-audit). b_h is the log weight of reader h among the readers of its hypothesis (existence dendrite). gate
        none: b_h only (no attribution)."""
        H, K = self.heads, self.n_box
        if self.gate == "none":
            return self.rf_bias[:, None].expand(H, n_win * K)
        s2 = self.sigma()[:, None] ** 2                                    # [H, 1]
        if not self.ring:
            return self.rf_bias[:, None] - self.r2.repeat(n_win)[None, :] / (2.0 * s2)
        v, w = F.softplus(self.ring_v), F.softplus(self.ring_w)
        if n_win > 1:                                                      # re-audit: windows 0 .. n_win - 1
            a = torch.arange(n_win, dtype=s2.dtype, device=s2.device).repeat_interleave(K)
        else:
            age = torch.as_tensor(k, device=s2.device)
            if age.dim() == 1:                                             # stream: one age per unit
                return self.association_prior_units(age)
            a = age.to(s2.dtype).expand(K)
        return self._ring(self.r.repeat(n_win)[None], a[None], s2, v[:, None], w[:, None], self.rf_bias[:, None])

    def association_prior_units(self, age):
        """Expanding-ring prior for units of different ages age [B] (one window each): [B, H, K]."""
        s2 = self.sigma()[None, :, None] ** 2
        v, w = F.softplus(self.ring_v)[None, :, None], F.softplus(self.ring_w)[None, :, None]
        return self._ring(self.r[None, None, :], age.to(s2.dtype)[:, None, None], s2, v, w, self.rf_bias[None, :, None])

    @staticmethod
    def _ring(r, a, s2, v, w, b):
        """Log density of the expanding ring at distance r after a windows (broadcast): radius R = v a, spread
        s^2 = sigma^2 + (w a)^2, normalised by the ring's area s (R + s) relative to sigma^2 (= 1 at a = 0)."""
        sa2 = s2 + (w * a) ** 2
        sa = torch.sqrt(sa2)
        R = v * a
        return b - (r - R) ** 2 / (2.0 * sa2) - torch.log(sa * (R + sa) / s2)

    def _existence(self, c, tok, n_win, prior, win_mask):
        """Existence dendrite (V19), Eq. E1-E3 of docs/V19_DESIGN.md. One window is one chance for the source of
        hypothesis g (target, background) to be detected: either it was missed (probability 1 - P_D(c)) or one of the
        window's events is its detection. The readers h of a hypothesis (heads, near and far) are components of its
        association prior (b_h = log weight), so they share ONE detection: with the events of a box sharing a score,
        the likelihood ratio of the window against clutter alone (integrated PDA) is
            miss_g + sum_{h in g} sum_b n_b exp(score_hb),
        score = association prior (geometry, time) + content log-likelihood ratio (the box statistics, context-
        shifted) - kappa_h log(1 + expected clutter events of the box) (divisive normalisation; the clutter density is
        the background-like events of the unit's square per px). The hypothesis current log(miss + D) = log miss
        (evidence of absence, <= 0) + softplus(log D - log miss) (detection, >= 0); the soma adds target - background
        (each reader carries 1 / readers of its hypothesis' current, so the head-signed sum is exactly that). The
        posterior association of a reader and box (PDA beta) n_b exp(score_hb) / (miss + D) is the gate it reports;
        the attributed mass of a reader is its share of the detection probability 1 - beta_0, summed over windows.
        prior: [1 or B, H, T]."""
        cnt = tok["cnt"]
        B, T = cnt.shape
        K, H, Fn = self.n_box, self.heads, self.n_feat
        dc = self.dcoef(c).view(B, H, Fn + 1)
        content = tok["s"].permute(0, 2, 1) + torch.einsum("bhf,btf->bht", dc[..., :Fn], tok["x"]) + dc[..., Fn:]
        score = content + prior
        if self.clutter:
            px = tok["px"].view(B, n_win, K)
            dens = tok["bg"].view(B, n_win, K).sum(-1) / px.sum(-1).clamp(min=1.0)   # events per px [B, n_win]
            expect = (dens[:, :, None] * px).reshape(B, T)                           # expected clutter per box
            score = score - F.softplus(self.clutter_k)[None, :, None] * torch.log1p(expect)[:, None, :]
        wm = torch.ones(B, n_win, device=cnt.device) if win_mask is None else win_mask.float()
        live = (cnt > 0.5) & (wm.repeat_interleave(K, 1) > 0)
        z = torch.where(live[:, None, :], torch.log(cnt.clamp(min=0.5))[:, None, :] + score,
                        torch.full_like(score, -1e4))
        hg = H // 2                                                        # readers per hypothesis
        zg = z.view(B, 2, hg, n_win, K).permute(0, 1, 3, 2, 4).reshape(B, 2, n_win, hg * K)
        logd = torch.logsumexp(zg, -1)                                     # log D per hypothesis [B, 2, n_win]
        log_miss = F.logsigmoid(self.miss_logit[None] + self.miss_ctx(c))[:, :, None]   # [B, 2, 1]
        total = log_miss + F.softplus(logd - log_miss)                    # log(miss + D)
        cur = (total * wm[:, None, :]).sum(-1)                             # hypothesis currents [B, 2]
        lam = cur.repeat_interleave(hg, 1) / hg                            # per reader: sum_h sign_h lam_h = T - B
        resp = torch.exp(z - total.repeat_interleave(hg, 1).repeat_interleave(K, 2))   # posterior association
        mass = resp.sum(-1)
        ubar = torch.einsum("bht,btf->bhf", resp, tok["x"]) / mass[..., None].clamp(min=EPS)
        return lam, mass, ubar.reshape(B, -1), resp

    def context_delta(self, ubar, mass, lam):
        """Eq. 6: what the last audit found -> change of the context."""
        dc = self.wo(ubar) + self.wm(torch.cat([torch.log1p(mass), lam], -1))
        return dc + self.ff(self.ln_f(dc)) if self.ctx_ff else dc

    def start_sums(self, B, device):
        """Ledger units: running sums of the windows read so far -- attributed mass [B, H], mass-weighted value (glm:
        features) [B, H, hd or 5], evidence [B, H] and lateral current [B] (V18 decision spikes)."""
        width = self.n_feat if self.tokens == "glm" else self.hd
        return (torch.zeros(B, self.heads, device=device), torch.zeros(B, self.heads, width, device=device),
                torch.zeros(B, self.heads, device=device), torch.zeros(B, device=device))

    def lateral(self, tok, win_mask=None):
        """V18 lateral current [B] of the windows in tok: every decision spike of another unit in a box adds the
        learned weight of its type, box and the unit's age (log(1 + spikes) per box); 0 without decision spikes."""
        if "lat" not in tok:
            return None
        lt = tok["lat"]
        if win_mask is not None:
            lt = lt * win_mask.float().repeat_interleave(self.n_box, 1)
        return lt.sum(-1)

    def step(self, c, c0, U, z0, k, tok, n_win, m, win_mask=None, sums=None):
        """Step k for units whose tokens so far are tok (n_win windows): new membrane and context (Eq. 3-6); units
        with m = False keep their state. Re-audit: U = z0 + evidence of all windows read now, c = c0 + Delta(this
        audit). Ledger (tok = the new window only, sums = start_sums at the first step): U = U + evidence of the new
        window, and c = c0 + Delta(running sums) -- the same context as re-audit computes from one audit of the same
        windows (attributed mean, total mass, total evidence), only the old windows are not read again. Returns U, c,
        lambda, mass, gates, sums (None for re-audit)."""
        lam, mass, ubar, a = self.audit_tokens(c, k, tok, n_win, win_mask)
        ev = (lam * self.head_sign).sum(-1)
        lat = self.lateral(tok, win_mask)                 # V18: decision spikes of others (somatic synapses)
        if self.audit == "reaudit":
            U = torch.where(m, z0 + ev + (0.0 if lat is None else lat), U)
            if self.context:
                c = torch.where(m[:, None], c0 + self.context_delta(ubar, mass, lam), c)
            return U, c, lam, mass, a, None
        B, H = mass.shape
        new = (sums[0] + mass, sums[1] + ubar.view(B, H, -1) * mass[..., None], sums[2] + lam,
               sums[3] + (0.0 if lat is None else lat))
        sums = tuple(torch.where(m.view((B,) + (1,) * (n.dim() - 1)), n, o) for n, o in zip(new, sums))
        if self.pool == "mean":           # ablation: the mean of the windows read so far, windows do not accumulate
            n_read = torch.as_tensor(k, dtype=U.dtype, device=U.device) + 1.0
            U = torch.where(m, z0 + (sums[2] * self.head_sign).sum(-1) / n_read + sums[3], U)
        else:
            U = torch.where(m, U + ev + (0.0 if lat is None else lat), U)
        if self.context:
            M, X, L = sums[:3]
            mean = (X / M[..., None].clamp(min=EPS)).reshape(B, -1)
            c = torch.where(m[:, None], c0 + self.context_delta(mean, M, L), c)
        return U, c, lam, mass, a, sums

    # ------------------------------------------------------------------ whole trajectories
    def forward(self, feat, valid, z0, pol, keep=False, memb=None):
        """feat [B, S, K, N_FEAT] (windows 0 .. S-1 after the event), valid [B, S] (steps that exist), z0 [B] (calibrated
        prior), pol [B] -> membrane U [B, S]; U(k) is frozen after the last valid step. keep=True also stores the
        signed evidence per head and step, the attributed mass and the gates of every step in self.extras."""
        B, S = feat.shape[0], feat.shape[1]
        K = self.n_box
        tok = self.embed(feat)
        c0 = self.start(z0, pol, memb)
        c, U = c0, z0
        sums = self.start_sums(B, feat.device) if self.audit == "ledger" else None
        outs, lam_all, mass_all, gates, lat_all = [], [], [], [], []
        for k in range(S):
            if self.audit == "reaudit":
                sl, n_win = slice(0, (k + 1) * K), k + 1
            else:
                sl, n_win = slice(k * K, (k + 1) * K), 1
            sub = {n: t[:, sl] for n, t in tok.items()}
            U, c, lam, mass, a, sums = self.step(c, c0, U, z0, k, sub, n_win, valid[:, k], sums=sums)
            outs.append(U)
            if keep:
                lam_all.append(lam * self.head_sign)
                mass_all.append(mass)
                gates.append(F.pad(a, (sl.start, S * K - sl.stop)))
                if self.n_msg:
                    lat_all.append(self.lateral({"lat": sub["lat"][:, -K:]}) * valid[:, k].float())
        if keep:
            self.extras = {"lam": torch.stack(lam_all, 1), "mass": torch.stack(mass_all, 1),
                           "gates": torch.stack(gates, 1)}
            if self.n_msg:
                self.extras["lateral"] = torch.stack(lat_all, 1)               # current of the window read at k
        return torch.stack(outs, 1)

    # ------------------------------------------------------------------ cost and interpretation
    def step_macs(self, k, nz_new=None, nz_aud=None):
        """Multiply-accumulates of step k of one unit: the boxes of the new window read once (embedding or fixed logit),
        the audit of the boxes of every window so far (re-audit; ledger: the new window), the context. nz_new / nz_aud =
        boxes with events among them (arrays broadcast with k); None = every box. With evidence="events" a box without
        events has weight 0 everywhere (membrane, mass, context), so it is skipped exactly; with evidence="boxes" every
        box counts. Folded linear maps count once (linear: W_k W_c is one 5 x d map)."""
        d, H, K, Fn = self.dim, self.heads, self.n_box, self.n_feat
        attr = self.gate == "attribution"
        k = np.asarray(k)
        dense_aud = (k + 1) * K if self.audit == "reaudit" else np.full(k.shape, K)
        if nz_new is None or self.evidence == "boxes":
            nz_new, nz_aud = np.full(k.shape, K), dense_aud
        if self.tokens == "glm" and self.dendrite == "existence":
            new = Fn + Fn * H                                             # normalise + fixed logit
            # shift, score = prior (a table of age, box, head) + content - clutter + log n, exp and sum, the
            # posterior association and its mean feature
            aud = Fn * H + 3 * H + 2 * H + H + Fn * H
            fixed = d * H * (Fn + 1) + d * H + 4 * H                      # coefficient shift, miss(c), log-sum per head
            if self.clutter:
                fixed += 3 * K + K * H                                    # density (2 sums), log1p, kappa per box
            ctx = H * Fn * d + 2 * H * d
        elif self.tokens == "glm":
            new = Fn + Fn * H                                             # normalise + fixed logit
            aud = Fn * H + H + Fn * H + 3 * H + (Fn * H if attr else 0)   # shift, mean feature, sums (+ gate)
            fixed = d * H * (Fn + 1) + (d * H * Fn if attr else 0)        # coefficient shift (+ gate query)
            if self.clutter:
                fixed += K + 2 * H                                        # background count, its power per head
            ctx = H * Fn * d + 2 * H * d
        else:
            new = Fn * d + (d * d if self.tokens == "mlp" else 0)         # values
            if attr:
                new += d * d if self.tokens == "mlp" else Fn * d          # keys
            aud = (d if attr else 0) + d * H + d + 3 * H                  # q.k, value head, mean value, sums
            fixed = d * d * (2 if attr else 1)                            # query + context term of the values
            ctx = d * d + 2 * H * d
        if self.context:
            fixed += ctx + (4 * d * d if self.ctx_ff else 0)              # context update
            if self.audit == "ledger":
                fixed += 2 * H * (Fn if self.tokens == "glm" else self.hd)   # running sums and their mean
        if self.pseudo:
            fixed += d * H                                                # pseudo-observation value
        fixed += K * self.n_msg                                           # lateral synapses (every box, an upper bound)
        c0 = (4 if self.memb else 2) * d + d * d
        total = nz_new * new + nz_aud * aud + fixed + (k == 0) * c0              # c0 at the first step
        return int(total) if np.ndim(total) == 0 else np.asarray(total, np.float64)

    @staticmethod
    def step_acs(ne):
        """Accumulates of the box sums of one unit-step in an event-driven implementation: every event inside the
        unit's square adds 1 to the count and its score to q of its box (ne = events in the square, the unit's own
        event excluded). The GPU code gets the same integer sums from integral images."""
        return 2.0 * np.asarray(ne, np.float64)

    def track_macs(self, ne1):
        """Multiply-accumulates of the centroid measurement per unit-step ([N, S] like ne1 = events of the level-0/1
        boxes): every event of that square adds q x and q y, plus the update (TRACK_OPS); at every step with tracked
        fields, at the first step only for the membership cue alone, 0 otherwise."""
        ne1 = np.asarray(ne1, np.float64)
        cost = 2.0 * ne1 + TRACK_OPS
        if self.track:
            return cost
        if self.memb:
            out = np.zeros_like(cost)
            out[..., 0] = cost[..., 0]
            return out
        return np.zeros_like(cost)

    def learned_constants(self):
        out = {"sigma_px": [round(v, 2) for v in self.sigma().detach().cpu().tolist()],
               "rf_bias": [round(v, 3) for v in self.rf_bias.detach().cpu().tolist()],
               "head_sign": self.head_sign.cpu().tolist()}
        if self.pseudo:
            out["pseudo_observation_weight"] = [round(v, 3) for v in
                                                torch.sigmoid(self.sink_logit).detach().cpu().tolist()]
        if self.ring:                          # V19: ring speed (px per window) and spread growth of every head
            out["ring_speed_px_per_window"] = [round(v, 3) for v in F.softplus(self.ring_v).detach().cpu().tolist()]
            out["ring_spread_px_per_window"] = [round(v, 3) for v in F.softplus(self.ring_w).detach().cpu().tolist()]
        if self.clutter:
            out["clutter_exponent"] = [round(v, 3) for v in F.softplus(self.clutter_k).detach().cpu().tolist()]
        if self.dendrite == "existence":     # per hypothesis (target, background), and each reader's mixture weight
            out["miss_probability"] = [round(v, 3) for v in torch.sigmoid(self.miss_logit).detach().cpu().tolist()]
            hg = self.heads // 2
            wts = torch.softmax(self.rf_bias.detach().cpu().view(2, hg), 1).reshape(-1)
            out["reader_weight"] = [round(v, 3) for v in wts.tolist()]
        if self.n_msg:                         # lateral synaptic weight per box level (mean over ages), by spike type
            j = torch.arange(self.n_box)
            lev = torch.where(j < 9, torch.zeros_like(j), (j - 9) // 8 + 1)    # box_layout order: 9 + 8 per level
            w = self.lat.detach().cpu().mean(0)                          # [K, types]
            out["lateral_by_level"] = {t: [round(float(w[lev == l, i].mean()), 4) for l in range(int(lev.max()) + 1)]
                                       for i, t in enumerate(("target_spike", "background_spike")[:self.n_msg])}
        return out


class StreamingDecisionLayer(object):
    """Deployment form of the decision layer, one window at a time.

    Window t: its events' evidence maps are built once; the new events pass the entry rule (column 0 of Eq. 7: commit
    the prior when it is already safe) and the others join the open units; every open unit reads its boxes of window t
    and all units take ONE batched step; a unit fires when its membrane reaches the bound of its step or at its last
    step (deadline or end of the recording). Re-audit units keep the tokens of all their windows (n_steps windows,
    windows still to come are masked) and re-read them; ledger units keep only their membrane and context and read the
    new window once. Returns the committed events with their column (0 = entry, 1 + k = fired at step k) and value."""

    def __init__(self, net, rect, bounds, tau, height, width, device):
        self.net, self.rect = net.eval(), rect
        self.device = torch.device(device)
        self.bounds = torch.tensor([float(b) for b in bounds], device=self.device)
        self.tau, self.H, self.W = float(tau), int(height), int(width)
        self.reset()

    def reset(self):
        self.st, self.meta = None, None

    def open_units(self):
        return 0 if self.st is None else int(self.st["U"].shape[0])

    def _write(self, st, rows, tok):
        """Tokens of one window (embed with S = 1) into the buffers tok_<name> at the units' window offsets."""
        K = self.net.n_box
        pos = st["age"][rows][:, None] * K + torch.arange(K, device=self.device)[None]
        r = rows[:, None]
        for name, t in tok.items():
            st["tok_" + name][r, pos] = t.reshape((len(rows), K) + tuple(t.shape[2:]))

    @staticmethod
    def _tokens(st):
        return {n[4:]: v for n, v in st.items() if n.startswith("tok_")}

    @torch.no_grad()
    def window(self, t, x, y, pol, z0, ids, last_step, messages=None, own_now=None):
        """messages = (x, y, type) of the decision spikes emitted at the end of window t by a first population (units
        with decision spikes as input, V18); own_now = {event id: type} of those spikes, removed from the own unit.
        Units with tracked fields keep their tracker state (trk) and update it with every window they read."""
        net, dev = self.net, self.device
        S, K = net.n_steps, net.n_box
        ledger = net.audit == "ledger"
        track = bool(getattr(net, "track", False))
        memb = bool(getattr(net, "memb", False))
        x, y, pol, ids = (np.asarray(a, np.int64) for a in (x, y, pol, ids))
        z0 = np.asarray(z0, np.float32)
        last_step = np.asarray(last_step, np.int64)
        msg = None
        if net.n_msg or net.n_feat > N_FEAT:                       # the unit reads decision spikes
            mx, my, mt = (np.zeros(0, np.int64),) * 3 if messages is None else \
                (np.asarray(a, np.int64) for a in messages)
            msg = (mx, my, np.zeros(len(mx), np.int64), mt)
        own_now = own_now or {}
        own = lambda e: None if msg is None else (                   # noqa: E731
            np.array([0 if i in own_now else -1 for i in e], np.int64),
            np.array([own_now.get(i, 0) for i in e], np.int64))
        maps = EvidenceMaps(x, y, np.zeros(len(x), np.int64), pol, z0, 1, self.H, self.W, dev, messages=msg,
                            track=track or memb)
        out = ([], [], [])
        z0_t = torch.as_tensor(z0, device=dev)
        entry = (torch.abs(z0_t - self.tau) >= self.bounds[0]).cpu().numpy()
        if entry.any():                                            # safe priors commit at arrival
            out[0].append(ids[entry])
            out[1].append(np.zeros(int(entry.sum()), np.int64))
            out[2].append(z0[entry])
        tok_now = None                                             # ledger: tokens of window t, open units first
        if self.st is not None:                                    # open units read window t
            M = self.open_units()
            f, _, trk = maps.features(self.meta["x"], self.meta["y"], np.zeros(M, np.int64), self.meta["pol"],
                                      np.zeros(M, np.float32), self.rect, 1, own_window=False,  # z0 unused here
                                      own_msg=own(self.meta["ids"]), trk=self.st["trk"] if track else None,
                                      trk_age=self.st["age"] if track else None, return_track=True,
                                      follow=net.track_mode if track else False)
            if track:
                self.st["trk"] = trk                               # the state after window t
            tok_now = net.embed(f, slot_idx=self.st["age"])
            if not ledger:
                self._write(self.st, torch.arange(M, device=dev), tok_now)
        new = ~entry
        if new.any():                                              # the others start a unit (step 0 at once)
            n = int(new.sum())
            nt = torch.as_tensor(new, device=dev)
            zt, pt = z0_t[nt], torch.as_tensor(pol[new], device=dev)
            f, _, trk, cue = maps.features(x[new], y[new], np.zeros(n, np.int64), pol[new], z0[new], self.rect, 1,
                                           own_msg=own(ids[new]), return_track=True,
                                           follow=net.track_mode if track else False, return_memb=True)
            c0 = net.start(zt, pt, cue)
            nb = {"age": torch.zeros(n, dtype=torch.long, device=dev), "z0": zt, "c0": c0, "c": c0, "U": zt,
                  "last": torch.as_tensor(last_step[new], device=dev)}
            if track:
                nb["trk"] = trk
            tok = net.embed(f, slot_idx=nb["age"])
            if ledger:
                tok_now = tok if tok_now is None else {n_: torch.cat([tok_now[n_], tok[n_]]) for n_ in tok}
                nb["sum_mass"], nb["sum_value"], nb["sum_ev"], nb["sum_lat"] = net.start_sums(n, dev)
            else:
                for name, t_ in tok.items():                       # token buffers of all n_steps windows
                    nb["tok_" + name] = torch.zeros((n, S * K) + tuple(t_.shape[2:]), device=dev)
                self._write(nb, torch.arange(n, device=dev), tok)
            meta = {"ids": ids[new], "x": x[new], "y": y[new], "pol": pol[new]}
            if self.st is None:
                self.st, self.meta = nb, meta
            else:
                self.st = {k: torch.cat([self.st[k], nb[k]]) for k in self.st}
                self.meta = {k: np.concatenate([self.meta[k], meta[k]]) for k in self.meta}
        if self.st is not None:                                    # one batched step for every open unit
            st = self.st
            M = self.open_units()
            on = torch.ones(M, dtype=torch.bool, device=dev)
            if ledger:
                U, c, _, _, _, sums = net.step(st["c"], st["c0"], st["U"], st["z0"], st["age"], tok_now, 1, on,
                                               sums=(st["sum_mass"], st["sum_value"], st["sum_ev"],
                                                     st["sum_lat"]))
                st["sum_mass"], st["sum_value"], st["sum_ev"], st["sum_lat"] = sums
            else:
                win_mask = torch.arange(S, device=dev)[None, :] <= st["age"][:, None]
                U, c, _, _, _, _ = net.step(st["c"], st["c0"], st["U"], st["z0"], st["age"], self._tokens(st), S,
                                            on, win_mask)
            fire = (torch.abs(U - self.tau) >= self.bounds[1 + st["age"]]) | (st["last"] == st["age"])
            f = fire.cpu().numpy()
            if f.any():
                out[0].append(self.meta["ids"][f])
                out[1].append((1 + st["age"][fire]).cpu().numpy())
                out[2].append(U[fire].cpu().numpy())
            if f.all():
                self.reset()
            else:
                keep = ~fire
                st["U"], st["c"] = U, c
                self.st = {k: v[keep] for k, v in st.items()}
                self.st["age"] = self.st["age"] + 1
                self.meta = {k: v[~f] for k, v in self.meta.items()}
        cat = lambda parts, dt: np.concatenate(parts) if parts else np.zeros(0, dt)    # noqa: E731
        return cat(out[0], np.int64), cat(out[1], np.int64), cat(out[2], np.float32)


class CooperativeStreamingLayer(object):
    """V18 deployment: two populations of decision units in one stream. Window t: the first population (units that
    read only the evidence, V17) takes its step; every one of its units that commits at the end of window t emits one
    decision spike at its event's pixel, target or background by its decision; the second population then takes its
    step reading the evidence of window t together with these spikes (its own event's spike removed), and its
    decisions are the output. Both populations run their own entry rule and calibrated bounds."""

    def __init__(self, first, second):
        self.first, self.second = first, second

    def reset(self, x_all, y_all):
        """x_all / y_all: positions of every event of the recording, indexed by the event ids passed to window()."""
        self.first.reset()
        self.second.reset()
        self.x_all, self.y_all = np.asarray(x_all, np.int64), np.asarray(y_all, np.int64)

    def open_units(self):
        return self.first.open_units() + self.second.open_units()

    def window(self, t, x, y, pol, z0, ids, last_step):
        e1, c1, v1 = self.first.window(t, x, y, pol, z0, ids, last_step)
        unit = c1 >= 1                                         # entry commits carry only the prior: no spike
        typ = (v1[unit] >= self.first.tau).astype(np.int64)
        mid = e1[unit]
        own_now = dict(zip(mid.tolist(), typ.tolist()))
        out = self.second.window(t, x, y, pol, z0, ids, last_step,
                                 messages=(self.x_all[mid], self.y_all[mid], typ), own_now=own_now)
        self.last_first = (e1, c1, v1)
        return out
