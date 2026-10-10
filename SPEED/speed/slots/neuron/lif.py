"""Slot 3 base variant: LIF neuron with learnable per-channel time constant and subtractive (soft) reset.

    u_pre = beta * u_prev + I          (u_pre = I without history)
    s     = H(u_pre - v_th)            (surrogate gradient 1 / (1 + |v|)^2)
    u     = u_pre - v_th * s.detach()  (then clamped to [u_floor, u_ceil] * v_th when carried)
    tau   = tau_min + (tau_max - tau_min) * sigmoid(a),  beta = exp(-dt / tau)
States are passed in and returned explicitly; nothing is stored in the module.
"""
import math

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v >= 0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        return grad_output / (1.0 + v.abs()).pow(2)


class LIF2d(nn.Module):
    def __init__(self, channels, dt_ms, tau_init_ms, tau_min_ms, tau_max_ms, v_threshold, u_floor=None, u_ceil=None):
        super(LIF2d, self).__init__()
        if not tau_min_ms < tau_init_ms < tau_max_ms:
            raise ValueError("need tau_min < tau_init < tau_max")
        frac = (tau_init_ms - tau_min_ms) / (tau_max_ms - tau_min_ms)
        self.a = nn.Parameter(torch.full((channels,), math.log(frac / (1.0 - frac)), dtype=torch.float32))
        self.dt_ms = float(dt_ms)
        self.tau_min_ms, self.tau_max_ms = float(tau_min_ms), float(tau_max_ms)
        self.v_threshold = float(v_threshold)
        if u_floor is not None and float(u_floor) > 0:
            raise ValueError("u_floor must be <= 0 (units of v_th)")
        if u_ceil is not None and float(u_ceil) <= 0:
            raise ValueError("u_ceil must be > 0 (units of v_th)")
        self.u_floor = None if u_floor is None else float(u_floor) * self.v_threshold
        self.u_ceil = None if u_ceil is None else float(u_ceil) * self.v_threshold

    def bound_state(self, state):
        # Only the carried state is bounded; the firing decision uses the unbounded u_pre.
        if self.u_floor is None and self.u_ceil is None:
            return state
        return torch.clamp(state, min=self.u_floor, max=self.u_ceil)

    def tau(self):
        return self.tau_min_ms + (self.tau_max_ms - self.tau_min_ms) * torch.sigmoid(self.a)

    def beta(self):
        return torch.exp(-self.dt_ms / self.tau())

    def forward(self, current, state):
        if state is None:
            u_pre = current
        else:
            u_pre = self.beta().to(current.dtype).view(1, -1, 1, 1) * state + current
        spikes = SurrogateSpike.apply(u_pre - self.v_threshold)
        return spikes, self.bound_state(u_pre - self.v_threshold * spikes.detach()), u_pre

    def forward_steps(self, current, state, carry=True, keep_u_pre=True):
        """current [T,B,C,H,W]; same result as calling forward T times."""
        steps = int(current.shape[0])
        if steps == 0:
            raise ValueError("a chunk needs at least one step")
        beta = self.beta().to(current.dtype).view(1, -1, 1, 1)
        spikes, u_pres = [], []
        for t in range(steps):
            prev = state if carry else None
            u_pre = current[t] if prev is None else beta * prev + current[t]
            out = SurrogateSpike.apply(u_pre - self.v_threshold)
            state = self.bound_state(u_pre - self.v_threshold * out.detach())
            spikes.append(out)
            if keep_u_pre:
                u_pres.append(u_pre)
        return torch.stack(spikes), state, (torch.stack(u_pres) if keep_u_pre else None)


class ChannelGain(nn.Module):
    """Positive per-channel gain exp(log_gain); calibrated once before training, learnable afterwards."""

    def __init__(self, channels):
        super(ChannelGain, self).__init__()
        self.log_gain = nn.Parameter(torch.zeros(channels, dtype=torch.float32))

    def gain(self):
        return torch.exp(self.log_gain)

    def set_gain(self, gains):
        gains = torch.as_tensor(gains, dtype=self.log_gain.dtype, device=self.log_gain.device)
        if gains.shape != self.log_gain.shape or bool((gains <= 0).any()):
            raise ValueError("gain shape mismatch or non-positive gain")
        with torch.no_grad():
            self.log_gain.copy_(torch.log(gains))

    def forward(self, x):
        return x * self.gain().to(x.dtype).view(1, -1, 1, 1)


def detach_states(states):
    if states is None:
        return None
    return [None if s is None else s.detach() for s in states]


NEURONS = {"lif": LIF2d}


def build_neuron_factory(cfg, dt_ms):
    """cfg: {kind, v_threshold, tau_init_ms, tau_min_ms, tau_max_ms, u_floor, u_ceil} -> channels -> module."""
    kind = cfg.get("kind", "lif")
    if kind not in NEURONS:
        raise ValueError("unknown neuron kind %s" % kind)
    kwargs = dict(dt_ms=dt_ms, tau_init_ms=float(cfg["tau_init_ms"]), tau_min_ms=float(cfg["tau_min_ms"]),
                  tau_max_ms=float(cfg["tau_max_ms"]), v_threshold=float(cfg["v_threshold"]),
                  u_floor=cfg.get("u_floor"), u_ceil=cfg.get("u_ceil"))
    return lambda channels: NEURONS[kind](channels, **kwargs)
