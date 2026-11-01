"""Safe dense L2 gradient clipping for the object-head trainers (torch 1.9)."""
import math

import torch


@torch.no_grad()
def clip_grad_norm_checked_(named_parameters, max_norm, context=""):
    """Compute the norm in float64; reject invalid gradients before mutating any.

    Never call optimizer.step after a failed check: Adam's moments would be poisoned.
    Finite float32 gradients can have an overflowing float32 squared norm, so that
    case must be clipped in higher precision rather than treated as a bad update.
    """
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    grads = [(name, p.grad) for name, p in named_parameters if p.grad is not None]
    if not grads:
        return 0.0
    norms = torch.stack([g.detach().double().norm(2) for _, g in grads])
    finite = torch.isfinite(norms)
    if not bool(finite.all()):
        bad = [name for (name, _), ok in zip(grads, finite.cpu().tolist()) if not ok]
        raise RuntimeError("Non-finite gradients before optimizer.step (%s): %s" % (context, ", ".join(bad)))
    total = float(norms.norm(2))
    if not math.isfinite(total):
        raise RuntimeError("Non-finite float64 gradient norm before optimizer.step (%s)" % context)
    scale = min(1.0, max_norm / (total + 1e-6))
    for _, grad in grads:
        grad.mul_(scale)
    return total
