"""Static five-level activation control; no membrane, time or sparse dispatch."""
import torch
from torch import nn


class _QuantizeFive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # Nearest level, ties upward; do not use round's ties-to-even rule.
        return torch.floor(torch.clamp(x, 0., 1.) * 4. + .5) / 4.

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Same interior/boundary mask as Hardtanh(0,1), straight-through inside.
        return grad_output * ((x > 0.) & (x < 1.)).to(grad_output.dtype)


class QuantRate5(nn.Module):
    def forward(self, x):
        return _QuantizeFive.apply(x)


def install_quantized_activations(net, stages):
    stages = list(stages)
    if not stages or len(set(stages)) != len(stages) or any(s not in (1, 2, 3, 4) for s in stages):
        raise ValueError('Expected unique stages in 1..4')
    sites = []
    for stage in stages:
        blocks = list(getattr(net, 'conv%d' % stage).children())
        block = blocks[-1]
        if not isinstance(block._modules.get('2'), nn.ReLU):
            raise RuntimeError('Unexpected terminal activation layout')
        block._modules['2'] = QuantRate5()
        sites.append('conv%d.%d.2' % (stage, len(blocks) - 1))
    return sites
