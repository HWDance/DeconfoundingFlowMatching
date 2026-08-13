"""Fast end-to-end DeconfoundingFM example on a scalar synthetic outcome."""

import torch

from deconfoundingfm import DeconfoundingFM, DeconfoundingFMConfig


def make_data(n: int = 256, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 1, generator=g)
    p = torch.sigmoid(4.0 * (X[:, 0] - 0.5))
    A = torch.bernoulli(p, generator=g)
    Y = 2.0 * X[:, 0] + A + 0.25 * torch.randn(n, generator=g)
    return X, A, Y


X, A, Y = make_data()

# Small epoch counts keep the example fast; increase them for real applications.
model = DeconfoundingFM(
    DeconfoundingFMConfig(
        device="cpu",
        coupling="independent",
        nuisance_epochs=5,
        epochs=5,
        nuisance_ode_steps=5,
        ode_steps=5,
        propensity_trees=20,
        plugin_reservoir=4,
        plugin_batch=2,
        batch_size=128,
        nuisance_batch_size=128,
    )
)
model.fit(X, A, Y, verbose=False)

print("counterfactual samples:", model.sample(a=1, n=8).flatten())
print("deconfounded observed samples:", model.transform(Y[A == 1][:8], a=1).flatten())
print("diagnostics:", model.diagnostics())
