"""Forward (push) warping by bilinear splatting.

Every source pixel s sends values[..., s] to the continuous position s + d(s) and shares it among the 4 nearest pixels
with bilinear weights (sums on collisions; whatever lands outside the canvas is dropped). Pushing uses the displacement
known at the source, which is where a moving target is observed; pulling would need it at the destination.
Differentiable w.r.t. the values and the displacement; index_put_ with accumulate is deterministic on CUDA.
A zero displacement returns the input exactly.
"""
import torch


def bilinear_splat(values, disp_y, disp_x):
    """values [B,C,H,W]; disp_y, disp_x [B,1,H,W] in pixels -> pushed values [B,C,H,W]."""
    B, C, H, W = (int(n) for n in values.shape)
    dev = values.device
    ys = torch.arange(H, device=dev, dtype=disp_y.dtype).view(1, 1, H, 1)
    xs = torch.arange(W, device=dev, dtype=disp_x.dtype).view(1, 1, 1, W)
    ty, tx = ys + disp_y, xs + disp_x
    y0, x0 = torch.floor(ty), torch.floor(tx)
    fy, fx = (ty - y0).to(values.dtype), (tx - x0).to(values.dtype)
    y0, x0 = y0.long(), x0.long()
    plane = (torch.arange(B, device=dev).view(B, 1, 1, 1) * C + torch.arange(C, device=dev).view(1, C, 1, 1)) * H
    out = torch.zeros(B * C * H * W, device=dev, dtype=values.dtype)
    for dy, dx, w in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        yy, xx = y0 + dy, x0 + dx
        inside = ((yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)).expand(B, C, H, W)
        index = ((plane + yy.clamp(0, H - 1)) * W + xx.clamp(0, W - 1)).expand(B, C, H, W)
        contrib = values * w
        out = out.index_put((index[inside],), contrib[inside], accumulate=True)
    return out.view(B, C, H, W)


def splat_points(values, b, y, x, disp_y, disp_x, shape):
    """Sparse push: values [n] at integer pixels (b, y, x) [n] moved by (disp_y, disp_x) [n] and shared bilinearly
    -> dense [B,1,H,W] of the given shape (B, H, W). Differentiable w.r.t. values and displacement."""
    B, H, W = (int(n) for n in shape)
    out = torch.zeros(B * H * W, device=values.device, dtype=values.dtype)
    if int(values.numel()) == 0:
        return out.view(B, 1, H, W)
    ty, tx = y.to(disp_y.dtype) + disp_y, x.to(disp_x.dtype) + disp_x
    y0, x0 = torch.floor(ty), torch.floor(tx)
    fy, fx = (ty - y0).to(values.dtype), (tx - x0).to(values.dtype)
    y0, x0 = y0.long(), x0.long()
    for dy, dx, w in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        yy, xx = y0 + dy, x0 + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        index = (b * H + yy.clamp(0, H - 1)) * W + xx.clamp(0, W - 1)
        out = out.index_put((index[inside],), (values * w)[inside], accumulate=True)
    return out.view(B, 1, H, W)


def sample_bilinear(maps, b, c, y, x):
    """maps [B,C,H,W]; integer b, c and float y, x of any common shape -> bilinear values (0 outside the canvas).
    Differentiable w.r.t. the maps and the positions."""
    B, C, H, W = (int(n) for n in maps.shape)
    flat = maps.reshape(-1)
    y0, x0 = torch.floor(y), torch.floor(x)
    fy, fx = (y - y0).to(maps.dtype), (x - x0).to(maps.dtype)
    y0, x0 = y0.long(), x0.long()
    out = torch.zeros(y.shape, device=maps.device, dtype=maps.dtype)
    for dy, dx, w in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx), (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        yy, xx = y0 + dy, x0 + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        index = ((b * C + c) * H + yy.clamp(0, H - 1)) * W + xx.clamp(0, W - 1)
        out = out + torch.where(inside, flat[index], torch.zeros_like(out)) * w
    return out
