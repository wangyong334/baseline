"""Causal birth context and frozen-policy iteration targets for VoW.

These are estimated utilities, not confidence bounds. A label can enter a
regression target, never the action used to construct a future policy return.
"""
import math
import numpy as np
import torch
from scipy.ndimage import maximum_filter, uniform_filter

from model.bst import BirthValueHead, ContinuationValueHead, stop_risk


class BirthContext:
    """Shared spatial summaries computed once per observed window.

    Only z0/p/positions are used; no SDU state, labels or future windows. Grid
    cells share context, not predictions. An event retains its own z0,p,time.
    Dimensions are declared architecture choices, not data-selected thresholds.
    """
    feature_dim = 10

    def __init__(self, recording, cell=4, history=10, height=260, width=346, retain_all=False):
        self.rec, self.cell, self.history = recording, int(cell), int(history)
        if self.cell < 1 or self.history < 0:
            raise ValueError("invalid birth context geometry")
        if len(recording) and ((recording.x >= width).any() or (recording.y >= height).any()):
            raise ValueError("birth context sensor shape excludes events")
        self.shape = (int(math.ceil(height / self.cell)), int(math.ceil(width / self.cell)))
        self.current = -1
        self.maps, self.raw = {}, {}
        self.retain_all = bool(retain_all)

    def _advance(self, window):
        r = self.rec
        for w in range(self.current + 1, int(window) + 1):
            ids = r.window_ids(w)
            count = np.zeros(self.shape, np.float32)
            total, square, pol = [np.zeros_like(count) for _ in range(3)]
            maximum = np.full(self.shape, -20.0, np.float32)
            y, x = r.y[ids] // self.cell, r.x[ids] // self.cell
            z = np.arcsinh(r.z0[ids])
            np.add.at(count, (y, x), 1)
            np.add.at(total, (y, x), z)
            np.add.at(square, (y, x), z * z)
            np.add.at(pol, (y, x), r.p[ids])
            np.maximum.at(maximum, (y, x), z)
            # Shared local 3x3 sum, never a predicted region label.
            count, total, square, pol = [uniform_filter(a, 3, mode="constant") * 9
                                          for a in (count, total, square, pol)]
            maximum = maximum_filter(maximum, 3, mode="constant", cval=-20)
            mean = total / np.maximum(count, 1)
            std = np.sqrt(np.maximum(square / np.maximum(count, 1) - mean * mean, 0))
            self.raw[w] = (count, total)
            past = [self.raw[k] for k in range(max(0, w - self.history), w)]
            pc = sum((a[0] for a in past), np.zeros_like(count))
            pt = sum((a[1] for a in past), np.zeros_like(count))
            self.maps[w] = np.stack([mean, std, np.log1p(count),
                                    np.where(count > 0, maximum, 0),
                                    pol / np.maximum(count, 1), pt / np.maximum(pc, 1), np.log1p(pc)], -1)
            if not self.retain_all:
                self.maps = {w: self.maps[w]}
            self.raw = {k: value for k, value in self.raw.items() if k >= w - self.history}
            self.current = w

    def features(self, ids, current_window=None):
        ids = np.asarray(ids, np.int64)
        if not len(ids):
            return np.zeros((0, self.feature_dim), np.float32)
        windows = self.rec.window[ids]
        if current_window is not None and (windows != int(current_window)).any():
            raise ValueError("birth features require arrival-window queries")
        self._advance(int(windows.max()))
        ctx = np.empty((len(ids), 7), np.float32)
        for w in np.unique(windows):
            if int(w) not in self.maps:
                raise ValueError("birth context was evicted; use retain_all=True for offline training")
            take = windows == w
            subset = ids[take]
            ctx[take] = self.maps[int(w)][self.rec.y[subset] // self.cell, self.rec.x[subset] // self.cell]
        r = self.rec
        own = np.stack([np.arcsinh(r.z0[ids]), r.p[ids], (r.t[ids] % r.window_ms) / r.window_ms], -1)
        return np.concatenate([own, ctx], -1).astype(np.float32)


def policy_returns(logits, labels, valid, wait_actions, tau, fp_cost, fn_cost, wait_cost):
    """Targets E[J_(d+1)|S_d], and realized J_d for ONE FROZEN policy.

    wait_actions must have been computed from causal features without labels.
    We do NOT use min(realized stop loss, realized future loss).
    Immediate wait cost is kept outside the continuation head.
    """
    logits, labels = np.asarray(logits), np.asarray(labels)
    valid, wait_actions = np.asarray(valid, bool), np.asarray(wait_actions, bool)
    if logits.ndim != 2 or logits.shape[1] < 1 or logits.shape != valid.shape or logits.shape != wait_actions.shape or labels.shape != logits.shape[:1]:
        raise ValueError("incompatible trajectory shapes")
    if not all(math.isfinite(float(value)) for value in (tau, fp_cost, fn_cost, wait_cost)) or wait_cost < 0 or fp_cost <= 0 or fn_cost <= 0:
        raise ValueError("costs must be nonnegative with positive classification costs")
    if not np.isfinite(logits).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("trajectories must be finite with binary labels")
    if not valid[:, 0].all() or ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError("valid trajectory ages must form a nonempty causal prefix")
    action = logits >= tau
    loss = np.where(action, fp_cost * (1 - labels[:, None]), fn_cost * labels[:, None])
    target = np.zeros_like(logits, dtype=np.float32)
    returns = np.zeros_like(logits, dtype=np.float32)
    eligible = valid & np.concatenate([valid[:, 1:], np.zeros((len(valid), 1), bool)], axis=1)
    for age in range(logits.shape[1] - 1, -1, -1):
        future = returns[:, age + 1] if age + 1 < logits.shape[1] else np.zeros(len(logits))
        target[:, age] = np.where(eligible[:, age], future, 0)
        wait = eligible[:, age] & wait_actions[:, age]
        returns[:, age] = np.where(valid[:, age], np.where(wait, wait_cost + future, loss[:, age]), 0)
    return target, returns, eligible


def make_birth_callback(head, model, rec, threshold, log_w, init_cost, context_config, device):
    from tools.bst.pipeline import threshold_costs
    _, fp, fn = threshold_costs(threshold, log_w)
    context = [None]
    def admit(ids, recording, window):
        if context[0] is None:
            context[0] = BirthContext(recording, **context_config)
        with torch.no_grad():
            features = torch.as_tensor(context[0].features(ids, window), device=device)
            prior = torch.as_tensor(recording.z0[ids], device=device)
            risk = stop_risk(prior, fp, fn)
            return (risk > float(init_cost) + head(features)).cpu().numpy()
    return admit
