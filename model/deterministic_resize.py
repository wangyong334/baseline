"""Integer bilinear enlargement with a deterministic first-order adjoint.

The native forward is unchanged.  The backward gathers each input pixel's
finite support instead of scattering CUDA atomic additions, which old PyTorch
versions reject in strict deterministic mode.  This is the same mathematical
adjoint, not a promise of bitwise equality to native floating-point reductions.
"""
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


def _linear_resize_adjoint(gradient, input_length, scale):
    position = torch.arange(input_length, device=gradient.device)
    offset = torch.arange(2 * scale, device=gradient.device)
    indices = position[:, None] * scale - scale // 2 + offset[None, :]
    valid = (indices >= 0) & (indices < gradient.shape[-1])
    safe = indices.clamp(0, gradient.shape[-1] - 1)
    coordinate = ((safe.to(gradient.dtype) + 0.5) / scale - 0.5).clamp(0, input_length - 1)
    weight = (1 - (coordinate - position[:, None].to(gradient.dtype)).abs()).clamp_min(0)
    weight = weight * valid.to(gradient.dtype)
    gathered = gradient.index_select(-1, safe.reshape(-1))
    gathered = gathered.reshape(*gradient.shape[:-1], input_length, 2 * scale)
    return (gathered * weight).sum(-1)


class _BilinearResize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, size):
        height, width = value.shape[-2:]
        if (len(size) != 2 or size[0] < height or size[1] < width
                or size[0] % height or size[1] % width):
            raise ValueError("Deterministic bilinear resize requires integer enlargement")
        ctx.input_hw = (height, width)
        ctx.scales = (size[0] // height, size[1] // width)
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        height, width = ctx.input_hw
        sy, sx = ctx.scales
        gradient = _linear_resize_adjoint(gradient, width, sx)
        gradient = _linear_resize_adjoint(gradient.transpose(-2, -1), height, sy)
        return gradient.transpose(-2, -1), None


def bilinear_resize(value, size):
    """Enlarge by integer factors; supports first-order gradients only."""
    if torch.is_grad_enabled() and value.requires_grad:
        return _BilinearResize.apply(value, tuple(size))
    return F.interpolate(value, size=size, mode="bilinear", align_corners=False)
