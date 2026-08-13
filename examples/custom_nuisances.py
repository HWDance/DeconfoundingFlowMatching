"""Supplying already-fitted custom nuisance objects."""

import torch
import torch.nn as nn

from deconfoundingfm import DeconfoundingFM, DeconfoundingFMConfig


class KnownPropensity(nn.Module):
    def forward(self, X):
        return torch.sigmoid(4.0 * (X[:, 0] - 0.5))


class KnownConditionalOutcome(nn.Module):
    """Sampler for the toy law Y|X,A ~ Normal(2X+A, 0.2^2)."""

    @torch.no_grad()
    def sample_conditional(self, x, a, n_per_context=1):
        mean = 2.0 * x[:, :1] + a.reshape(-1, 1).float()
        if n_per_context == 1:
            return mean + 0.2 * torch.randn_like(mean)
        mean = mean[:, None, :].expand(-1, n_per_context, -1)
        return mean + 0.2 * torch.randn_like(mean)


torch.manual_seed(0)
X = torch.rand(256, 1)
A = torch.bernoulli(torch.sigmoid(4.0 * (X[:, 0] - 0.5)))
Y = 2.0 * X[:, 0] + A + 0.2 * torch.randn(256)

model = DeconfoundingFM(
    DeconfoundingFMConfig(
        epochs=10,
        plugin_reservoir=4,
        plugin_batch=2,
        device="cpu",
    ),
    propensity_model=KnownPropensity(),
    outcome_model=KnownConditionalOutcome(),
)
model.fit(X, A, Y, verbose=False)
print(model.sample(a=1, n=10))
