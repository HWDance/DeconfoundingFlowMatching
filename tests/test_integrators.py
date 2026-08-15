import torch
import torch.nn as nn

from deconfoundingfm.integrators import (
    integrate_euler,
    integrate_midpoint,
    integrate_midpoint_trajectory,
    normalized_path_energy_midpoint,
    path_energy_midpoint,
    velocity_derivative_energy_midpoint,
)


class ConstantVelocity(nn.Module):
    def forward(self, y, t, context=None):
        return torch.ones_like(y) * 2.0


def test_integrators_constant_velocity_exact():
    y0 = torch.zeros(4, 3)
    expected = torch.full_like(y0, 2.0)
    assert torch.allclose(integrate_euler(ConstantVelocity(), y0, steps=5), expected)
    assert torch.allclose(integrate_midpoint(ConstantVelocity(), y0, steps=5), expected)


def test_path_energy_vector_and_image():
    v = ConstantVelocity()
    e_vec = path_energy_midpoint(v, torch.zeros(2, 3), steps=4, reduce="none")
    e_img = path_energy_midpoint(v, torch.zeros(2, 1, 2, 2), steps=4, reduce="none")
    assert torch.allclose(e_vec, torch.full((2,), 12.0))
    assert torch.allclose(e_img, torch.full((2,), 16.0))


def test_midpoint_trajectory_and_normalized_energies():
    v = ConstantVelocity()
    y0 = torch.zeros(2, 1, 2, 2)
    traj, vmids = integrate_midpoint_trajectory(v, y0, steps=4)
    assert traj.shape == (5, 2, 1, 2, 2)
    assert vmids.shape == (4, 2, 1, 2, 2)
    # Constant velocity 2 on four coordinates gives raw energy 16 and normalized 4.
    ebar = normalized_path_energy_midpoint(v, y0, steps=4, reduce="none")
    assert torch.allclose(ebar, torch.full((2,), 4.0))
    # Constant velocity has zero derivative energy.
    dv = velocity_derivative_energy_midpoint(v, y0, steps=4, reduce="none")
    assert torch.allclose(dv, torch.zeros(2))
