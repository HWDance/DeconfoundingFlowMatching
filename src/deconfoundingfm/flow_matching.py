"""Shape-agnostic flow-matching primitives."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def flow_matching_loss(
    velocity: nn.Module,
    y0: torch.Tensor,
    y1: torch.Tensor,
    t: torch.Tensor,
    context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Straight-line conditional flow-matching loss for vectors or images."""
    if y0.shape != y1.shape:
        raise ValueError(f"y0 and y1 must have the same shape; got {y0.shape} and {y1.shape}.")
    batch = y0.shape[0]
    if t.numel() != batch:
        raise ValueError("t must contain one interpolation time per sample.")
    t_view = t.reshape((batch,) + (1,) * (y0.ndim - 1))
    y_t = (1.0 - t_view) * y0 + t_view * y1
    target_velocity = y1 - y0
    prediction = velocity(y_t, t, context)
    return (prediction - target_velocity).reshape(batch, -1).square().sum(dim=1).mean()
