"""Tiny image-shape API demonstration (not a meaningful image benchmark)."""

import torch
import torch.nn as nn

from deconfoundingfm import DeconfoundingFM, DeconfoundingFMConfig


class ToyPropensity(nn.Module):
    def forward(self, X):
        return torch.full((len(X),), 0.5, device=X.device)


class ToyImageConditional(nn.Module):
    @torch.no_grad()
    def sample_conditional(self, x, a, n_per_context=1):
        # One-channel 8x8 images; treatment shifts global intensity.
        loc = a.reshape(-1, 1, 1, 1).float().expand(-1, 1, 8, 8)
        if n_per_context == 1:
            return loc + 0.1 * torch.randn_like(loc)
        loc = loc[:, None].expand(-1, n_per_context, -1, -1, -1)
        return loc + 0.1 * torch.randn_like(loc)


torch.manual_seed(1)
N = 16
X = torch.randn(N, 1)
A = (torch.arange(N) % 2).float()
Y = A[:, None, None, None].expand(-1, 1, 8, 8) + 0.1 * torch.randn(N, 1, 8, 8)

model = DeconfoundingFM(
    DeconfoundingFMConfig(
        architecture="auto",
        coupling="eot",
        device="cpu",
        epochs=1,
        ode_steps=1,
        unet_channels=2,
        batch_size=8,
        plugin_reservoir=1,
        plugin_batch=1,
        eot_iterations=3,
        eot_source_batch=8,
    ),
    propensity_model=ToyPropensity(),
    outcome_model=ToyImageConditional(),
).fit(X, A, Y, verbose=False)

print("sample shape:", tuple(model.sample(a=1, n=2).shape))
print("transformed shape:", tuple(model.transform(Y[:2], a=1).shape))
