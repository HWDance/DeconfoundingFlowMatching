"""Entropic optimal-transport utilities used by DeconfoundingFM."""

from __future__ import annotations

import torch


def _validate_points(x_src: torch.Tensor, x_tgt: torch.Tensor, eps: float, n_iters: int):
    if x_src.ndim != 2 or x_tgt.ndim != 2:
        raise ValueError("OT utilities expect flattened 2D tensors (N,d).")
    if x_src.shape[1] != x_tgt.shape[1]:
        raise ValueError("Source and target points must have the same feature dimension.")
    if x_src.shape[0] < 1 or x_tgt.shape[0] < 1:
        raise ValueError("Source and target point sets must be non-empty.")
    if eps <= 0 or not torch.isfinite(torch.tensor(float(eps))):
        raise ValueError("eps must be finite and strictly positive.")
    if int(n_iters) < 1:
        raise ValueError("n_iters must be >= 1.")
    if not torch.isfinite(x_src).all() or not torch.isfinite(x_tgt).all():
        raise ValueError("OT inputs must be finite.")


@torch.no_grad()
def entropic_coupling_plan(
    x_src: torch.Tensor,
    x_tgt: torch.Tensor,
    *,
    eps: float = 0.1,
    n_iters: int = 50,
) -> torch.Tensor:
    """Return a log-domain Sinkhorn plan with uniform marginals."""
    _validate_points(x_src, x_tgt, eps, n_iters)
    ns, nt = x_src.shape[0], x_tgt.shape[0]
    log_a = -torch.log(torch.tensor(float(ns), device=x_src.device, dtype=x_src.dtype))
    log_b = -torch.log(torch.tensor(float(nt), device=x_src.device, dtype=x_src.dtype))
    cost = torch.cdist(x_src, x_tgt, p=2).square()
    log_kernel = -cost / float(eps)
    u = torch.zeros(ns, device=x_src.device, dtype=x_src.dtype)
    v = torch.zeros(nt, device=x_src.device, dtype=x_src.dtype)
    for _ in range(int(n_iters)):
        u = log_a - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_kernel + u[:, None], dim=0)
    plan = torch.exp(log_kernel + u[:, None] + v[None, :])
    if not torch.isfinite(plan).all():
        raise FloatingPointError("Sinkhorn produced a non-finite transport plan.")
    return plan


@torch.no_grad()
def sinkhorn_target_dual(
    x_src: torch.Tensor,
    x_tgt: torch.Tensor,
    eps: float = 0.1,
    n_iters: int = 50,
) -> torch.Tensor:
    """Return target log-duals for the uniform-marginal entropic coupling.

    These are the duals used by the paper implementation's conditional
    out-of-sample extension

    ``pi(j | y) proportional exp(v_j - ||y-x_tgt[j]||^2 / eps)``.
    """
    _validate_points(x_src, x_tgt, eps, n_iters)
    ns, nt = x_src.shape[0], x_tgt.shape[0]
    log_a = -torch.log(torch.tensor(float(ns), device=x_src.device, dtype=x_src.dtype))
    log_b = -torch.log(torch.tensor(float(nt), device=x_src.device, dtype=x_src.dtype))
    cost = torch.cdist(x_src, x_tgt, p=2).square()
    log_kernel = -cost / float(eps)
    u = torch.zeros(ns, device=x_src.device, dtype=x_src.dtype)
    v = torch.zeros(nt, device=x_src.device, dtype=x_src.dtype)
    for _ in range(int(n_iters)):
        u = log_a - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_kernel + u[:, None], dim=0)
    if not torch.isfinite(v).all():
        raise FloatingPointError("Sinkhorn produced non-finite target duals.")
    return v


@torch.no_grad()
def sinkhorn_target_dual_weighted(
    x_src: torch.Tensor,
    x_tgt: torch.Tensor,
    source_weights: torch.Tensor,
    eps: float = 0.1,
    n_iters: int = 50,
) -> torch.Tensor:
    """Weighted-source variant retained for optional moment tilting experiments."""
    _validate_points(x_src, x_tgt, eps, n_iters)
    source_weights = source_weights.to(device=x_src.device, dtype=x_src.dtype).reshape(-1)
    if source_weights.shape[0] != x_src.shape[0]:
        raise ValueError("source_weights must have one entry per source point.")
    if (source_weights < 0).any() or float(source_weights.sum()) <= 0:
        raise ValueError("source_weights must be non-negative with positive total mass.")
    source_weights = source_weights / source_weights.sum()
    nt = x_tgt.shape[0]
    log_a = torch.log(source_weights.clamp_min(1e-12))
    log_b = -torch.log(torch.tensor(float(nt), device=x_src.device, dtype=x_src.dtype))
    log_kernel = -torch.cdist(x_src, x_tgt, p=2).square() / float(eps)
    u = torch.zeros(x_src.shape[0], device=x_src.device, dtype=x_src.dtype)
    v = torch.zeros(nt, device=x_src.device, dtype=x_src.dtype)
    for _ in range(int(n_iters)):
        u = log_a - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_kernel + u[:, None], dim=0)
    return v


@torch.no_grad()
def ot_conditional_probabilities(
    y: torch.Tensor,
    x_tgt: torch.Tensor,
    v: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Evaluate the discrete EOT conditional over the target support."""
    if y.ndim != 2 or x_tgt.ndim != 2:
        raise ValueError("y and x_tgt must be 2D flattened tensors.")
    if y.shape[1] != x_tgt.shape[1] or v.reshape(-1).shape[0] != x_tgt.shape[0]:
        raise ValueError("Incompatible OT conditional shapes.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    cost = torch.cdist(y, x_tgt, p=2).square()
    logits = v.reshape(1, -1) - cost / float(eps)
    probs = torch.softmax(logits, dim=1)
    if not torch.isfinite(probs).all():
        raise FloatingPointError("OT conditional contains non-finite probabilities.")
    return probs


@torch.no_grad()
def sample_from_ot_conditional(
    y: torch.Tensor,
    x_tgt: torch.Tensor,
    v: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Sample one target-support point from the EOT conditional per query."""
    probs = ot_conditional_probabilities(y, x_tgt, v, eps)
    idx = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(1)
    return x_tgt[idx]
