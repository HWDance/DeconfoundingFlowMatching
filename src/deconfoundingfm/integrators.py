"""Small fixed-step ODE integrators used by DeconfoundingFM."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _validate_steps(steps: int) -> int:
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    return steps


def integrate_euler(
    velocity: nn.Module,
    y0: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    steps: int = 50,
) -> torch.Tensor:
    """Fixed-step forward Euler integration on ``t in [0,1]``."""
    steps = _validate_steps(steps)
    y = y0.clone()
    batch = y0.shape[0]
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((batch,), k * dt, device=y.device, dtype=y.dtype)
        y = y + dt * velocity(y, t, context)
    return y


def integrate_midpoint(
    velocity: nn.Module,
    y0: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    steps: int = 50,
) -> torch.Tensor:
    """Fixed-step explicit midpoint (RK2) integration on ``t in [0,1]``."""
    steps = _validate_steps(steps)
    y = y0.clone()
    batch = y0.shape[0]
    dt = 1.0 / steps
    for k in range(steps):
        t0 = torch.full((batch,), k * dt, device=y.device, dtype=y.dtype)
        tm = torch.full((batch,), (k + 0.5) * dt, device=y.device, dtype=y.dtype)
        k1 = velocity(y, t0, context)
        y_mid = y + 0.5 * dt * k1
        k2 = velocity(y_mid, tm, context)
        y = y + dt * k2
    return y


@torch.no_grad()
def path_energy_midpoint(
    velocity: nn.Module,
    y0: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    steps: int = 50,
    reduce: str = "mean",
):
    """Approximate ``integral ||v(Y_t,t)||^2 dt`` along midpoint trajectories."""
    steps = _validate_steps(steps)
    if reduce not in {"none", "mean", "sum"}:
        raise ValueError("reduce must be one of {'none','mean','sum'}.")
    y = y0.clone()
    batch = y0.shape[0]
    dt = 1.0 / steps
    energy = torch.zeros(batch, device=y.device, dtype=y.dtype)
    for k in range(steps):
        t0 = torch.full((batch,), k * dt, device=y.device, dtype=y.dtype)
        tm = torch.full((batch,), (k + 0.5) * dt, device=y.device, dtype=y.dtype)
        k1 = velocity(y, t0, context)
        y_mid = y + 0.5 * dt * k1
        v_mid = velocity(y_mid, tm, context)
        energy = energy + v_mid.reshape(batch, -1).square().sum(dim=1) * dt
        y = y + dt * v_mid
    if reduce == "none":
        return energy
    if reduce == "sum":
        return energy.sum()
    return energy.mean()
