"""A slightly fuller one-dimensional confounding example."""

import torch

from deconfoundingfm import DeconfoundingFM, DeconfoundingFMConfig


torch.manual_seed(7)
n = 1000
X = torch.rand(n, 1)
pi = torch.sigmoid(5.0 * (X[:, 0] - 0.5))
A = torch.bernoulli(pi)
Y = 10.0 * X[:, 0] + A + torch.randn(n)

cfg = DeconfoundingFMConfig(
    coupling="independent",
    nuisance_epochs=100,
    epochs=100,
    plugin_reservoir=16,
    plugin_batch=4,
)
model = DeconfoundingFM(cfg).fit(X, A, Y)

for arm in (0, 1):
    draws = model.sample(arm, 2000)
    print(f"arm={arm}: mean={draws.mean().item():.3f}, sd={draws.std().item():.3f}")
