import torch
import torch.nn as nn

from deconfoundingfm.flow_matching import flow_matching_loss


class ExactStraightVelocity(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, y, t, context=None):
        return self.delta.expand_as(y)


def test_zero_loss_for_exact_constant_straight_velocity_vector():
    y0 = torch.zeros(4, 2)
    y1 = torch.ones(4, 2) * 3
    t = torch.rand(4)
    assert flow_matching_loss(ExactStraightVelocity(torch.tensor([3.0, 3.0])), y0, y1, t) == 0


def test_zero_loss_for_exact_constant_straight_velocity_image():
    y0 = torch.zeros(3, 1, 2, 2)
    y1 = torch.ones_like(y0)
    t = torch.rand(3)
    assert flow_matching_loss(ExactStraightVelocity(torch.ones(1, 1, 2, 2)), y0, y1, t) == 0
