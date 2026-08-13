import torch
import torch.nn as nn


class DummyPropensity(nn.Module):
    def forward(self, X):
        return torch.sigmoid(0.2 * X[:, 0]).clamp(0.1, 0.9)


class DummyOutcome(nn.Module):
    def __init__(self, outcome_shape):
        super().__init__()
        self.outcome_shape = tuple(outcome_shape)

    @torch.no_grad()
    def sample_conditional(self, x, a, n_per_context=1):
        n = len(x)
        view = (-1,) + (1,) * len(self.outcome_shape)
        loc = a.reshape(view).float() + 0.1 * x[:, 0].reshape(view)
        if n_per_context == 1:
            loc = loc.expand((n,) + self.outcome_shape)
            return loc + 0.01 * torch.randn_like(loc)
        loc = loc[:, None].expand((n, n_per_context) + self.outcome_shape)
        return loc + 0.01 * torch.randn_like(loc)


def vector_data(n=24, dim_y=2):
    torch.manual_seed(123)
    X = torch.randn(n, 2)
    A = (torch.arange(n) % 2).float()
    Y = torch.randn(n, dim_y) + A[:, None]
    return X, A, Y


def image_data(n=8):
    torch.manual_seed(321)
    X = torch.randn(n, 1)
    A = (torch.arange(n) % 2).float()
    Y = torch.randn(n, 1, 8, 8) + A[:, None, None, None]
    return X, A, Y
