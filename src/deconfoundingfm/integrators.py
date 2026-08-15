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


@torch.no_grad()
def integrate_midpoint_trajectory(
    velocity: nn.Module,
    y0: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    steps: int = 50,
):
    """Return midpoint (RK2) trajectories and midpoint velocities on ``t in [0,1]``.

    Returns
    -------
    trajectory:
        Tensor of shape ``(steps + 1, batch, ...)`` containing ``Y_0, ..., Y_1``.
    midpoint_velocities:
        Tensor of shape ``(steps, batch, ...)`` containing the midpoint velocity
        used on each integration interval.
    """
    steps = _validate_steps(steps)
    y = y0.clone()
    batch = y0.shape[0]
    dt = 1.0 / steps
    traj = [y.clone()]
    vmids = []
    for k in range(steps):
        t0 = torch.full((batch,), k * dt, device=y.device, dtype=y.dtype)
        tm = torch.full((batch,), (k + 0.5) * dt, device=y.device, dtype=y.dtype)
        k1 = velocity(y, t0, context)
        y_mid = y + 0.5 * dt * k1
        v_mid = velocity(y_mid, tm, context)
        y = y + dt * v_mid
        vmids.append(v_mid.clone())
        traj.append(y.clone())
    return torch.stack(traj, dim=0), torch.stack(vmids, dim=0)


@torch.no_grad()
def normalized_path_energy_midpoint(
    velocity: nn.Module,
    y0: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    steps: int = 50,
    reduce: str = "mean",
):
    """Approximate per-dimension path energy ``(1/d) ∫ ||v(Y_t,t)||^2 dt``."""
    e = path_energy_midpoint(velocity, y0, context=context, steps=steps, reduce="none")
    d = y0[0].numel()
    e = e / float(d)
    if reduce == "none":
        return e
    if reduce == "sum":
        return e.sum()
    if reduce == "mean":
        return e.mean()
    raise ValueError("reduce must be one of {'none','mean','sum'}.")


@torch.no_grad()
def velocity_derivative_energy_midpoint(
    velocity: nn.Module,
    y0: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    steps: int = 50,
    reduce: str = "mean",
    normalize_by_dim: bool = True,
):
    """Approximate velocity-derivative energy along midpoint trajectories.

    This estimates

    ``∫ || d/dt v_t(Y_t) ||^2 dt``

    using finite differences of the midpoint velocities actually experienced along
    the RK2 trajectory.  When ``normalize_by_dim`` is true, the result is divided
    by the flattened outcome dimension.
    """
    if reduce not in {"none", "mean", "sum"}:
        raise ValueError("reduce must be one of {'none','mean','sum'}.")
    steps = _validate_steps(steps)
    _, vmids = integrate_midpoint_trajectory(velocity, y0, context=context, steps=steps)
    batch = y0.shape[0]
    if steps == 1:
        out = torch.zeros(batch, device=y0.device, dtype=y0.dtype)
    else:
        dt = 1.0 / steps
        dv = (vmids[1:] - vmids[:-1]) / dt
        out = dv.reshape(steps - 1, batch, -1).square().sum(dim=2).sum(dim=0) * dt
    if normalize_by_dim:
        out = out / float(y0[0].numel())
    if reduce == "none":
        return out
    if reduce == "sum":
        return out.sum()
    return out.mean()
