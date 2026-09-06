"""Local constant-current LIF rate coding, NOT causal event-stream inference."""
import math
import torch
from torch import nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad):
        (voltage,) = ctx.saved_tensors
        return grad / (1.0 + voltage.abs()).pow(2)


class LIFRate(nn.Module):
    """ReLU current -> T LIF steps -> threshold-scaled mean binary firing.

    State is local to forward: no row identity or graph survives a sample.
    Reset is subtractive and its spike derivative is detached.
    All parameters are fixed in v0; zero-current emits no spikes.
    """
    def __init__(self, steps=4, beta=0.9, threshold=1.0):
        super().__init__()
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("beta must be in [0,1]")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be positive and finite")
        self.steps, self.beta, self.threshold = steps, beta, threshold
        self.collect = False
        self.stats = {}

    def forward(self, x):
        current = torch.relu(x)
        membrane = torch.zeros_like(current)
        count = torch.zeros_like(current)
        for _ in range(self.steps):
            membrane = self.beta * membrane + current
            spike = SurrogateSpike.apply(membrane - self.threshold)
            count = count + spike
            membrane = membrane - self.threshold * spike.detach()
        if self.collect:
            with torch.no_grad():
                self.stats = {
                    "elements": current.numel(), "steps": self.steps,
                    "spike_count": count.sum().item(),
                    "firing_rate": count.mean().item() / self.steps,
                    "silent_element_fraction": (count == 0).float().mean().item(),
                    "silent_channel_fraction": (count.sum(0) == 0).float().mean().item(),
                    "saturated_element_fraction": (count == self.steps).float().mean().item(),
                    "current_mean": current.mean().item(),
                    "current_max": current.max().item(),
                    "membrane_abs_max": membrane.abs().max().item(),
                }
        return count * (self.threshold / self.steps)
