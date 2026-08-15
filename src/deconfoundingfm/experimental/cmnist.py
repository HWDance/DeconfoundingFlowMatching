from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from ..datasets.mnist.mnist_colour import (
    BG_BLACK,
    generate_color_mnist_confounding,
    generate_two_color_observational_population,
    load_mnist_idx,
    recolor_foreground_background_batch,
    sample_x_given_arm,
    true_propensity,
)


def _maybe_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class CMNISTConfig:
    """The ColorMNIST configuration used by the original paper experiments."""

    digits: tuple[int, int] = (1, 6)
    x_low: float = 0.0
    x_high: float = 1.0
    propensity_w: float = 5.0
    tau: float = 0.08
    k: float = 10.0
    fg_alpha: float = 0.0
    n_bw_shapes: int = 10_000
    color_draws_per_shape: int = 2


class ColorMNISTDGP:
    """Exact original CMNIST DGP backed by the packaged UByte t10k files."""

    def __init__(self, config: CMNISTConfig | None = None, *, device=None):
        self.config = config or CMNISTConfig()
        self.device = _maybe_device(device)
        self.pools = load_mnist_idx(
            package="deconfoundingfm.datasets.mnist", device=self.device
        )

    def propensity(self, x: torch.Tensor):
        return true_propensity(x, w=self.config.propensity_w)

    @torch.no_grad()
    def make_observational_population(self, *, seed: int = 0):
        """Fixed 20k population: 10k grayscale shape draws x two X|A colors each."""
        return generate_two_color_observational_population(
            n_bw_shapes=self.config.n_bw_shapes,
            color_draws_per_shape=self.config.color_draws_per_shape,
            digits=self.config.digits,
            x_low=self.config.x_low,
            x_high=self.config.x_high,
            w=self.config.propensity_w,
            device=self.device,
            seed=seed,
            tau=self.config.tau,
            k=self.config.k,
            fg_alpha=self.config.fg_alpha,
        )

    @torch.no_grad()
    def sample_observational(self, n: int, *, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(int(seed))
        x = torch.empty(n, device=self.device).uniform_(self.config.x_low, self.config.x_high)
        p = self.propensity(x)
        a = torch.bernoulli(p).long()
        S = torch.empty((n, 1, 28, 28), device=self.device)
        for arm, digit in ((0, self.config.digits[0]), (1, self.config.digits[1])):
            pos = (a == arm).nonzero(as_tuple=True)[0]
            pool = self.pools[int(digit)]
            j = torch.randint(len(pool), (len(pos),), device=self.device)
            S[pos] = pool[j]
        y = recolor_foreground_background_batch(
            S,
            x,
            fg_alpha=self.config.fg_alpha,
            bg_rgb=BG_BLACK,
            tau=self.config.tau,
            k=self.config.k,
        )
        return x.view(-1, 1).float(), a.view(-1, 1).float(), y.float()

    @torch.no_grad()
    def sample_interventional(self, a: int, n: int):
        a = int(a)
        x = torch.empty(n, device=self.device).uniform_(self.config.x_low, self.config.x_high)
        digit = int(self.config.digits[a])
        pool = self.pools[digit]
        j = torch.randint(len(pool), (n,), device=self.device)
        S = pool[j]
        y = recolor_foreground_background_batch(
            S,
            x,
            fg_alpha=self.config.fg_alpha,
            bg_rgb=BG_BLACK,
            tau=self.config.tau,
            k=self.config.k,
        )
        av = torch.full((n, 1), float(a), device=self.device)
        return x.view(-1, 1).float(), av, y.float()


class ExactColorMNISTSourceGenerator:
    """Exact stand-in for a pretrained biased generator of P(Y | A=a).

    Each call draws fresh X|A=a and a fresh grayscale shape from the same UByte
    digit pool used in the original experiment, then applies the exact original
    foreground recoloring map. This is the only oracle component of the demo.
    Propensity and P(Y|X,A) are still estimated from the fixed labeled dataset.
    """

    def __init__(self, dgp: ColorMNISTDGP):
        self.dgp = dgp

    @torch.no_grad()
    def sample(self, a: int, n: int, device=None):
        a = int(a)
        target_device = _maybe_device(device or self.dgp.device)
        cfg = self.dgp.config
        x = sample_x_given_arm(
            a,
            int(n),
            w=cfg.propensity_w,
            x_low=cfg.x_low,
            x_high=cfg.x_high,
            device=self.dgp.device,
        )
        digit = int(cfg.digits[a])
        pool = self.dgp.pools[digit]
        j = torch.randint(len(pool), (int(n),), device=self.dgp.device)
        S = pool[j]
        y = recolor_foreground_background_batch(
            S,
            x,
            fg_alpha=cfg.fg_alpha,
            bg_rgb=BG_BLACK,
            tau=cfg.tau,
            k=cfg.k,
        )
        return y.to(target_device)


# Backward-compatible name for one release; prefer ExactColorMNISTSourceGenerator.
OracleArmConditionalGenerator = ExactColorMNISTSourceGenerator


@torch.no_grad()
def sliced_wasserstein_images(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_projections: int = 256,
    p: int = 2,
    seed: int = 123,
) -> float:
    """Sliced Wasserstein on flattened images, matching the research metric form."""
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1).to(x.device)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    generator = torch.Generator(device=x.device).manual_seed(int(seed))
    theta = torch.randn(n_projections, x.shape[1], device=x.device, generator=generator)
    theta = theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-12)
    px = (x @ theta.T).sort(dim=0).values
    py = (y @ theta.T).sort(dim=0).values
    if p == 1:
        out = (px - py).abs().mean()
    elif p == 2:
        out = ((px - py) ** 2).mean().sqrt()
    else:
        raise ValueError("Only p=1 or p=2 supported.")
    return float(out.cpu())


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))


def load_result_bundle(path):
    path = Path(path)
    metrics = json.loads((path / "metrics.json").read_text())
    config = json.loads((path / "config.json").read_text())
    convergence = json.loads((path / "convergence.json").read_text())
    samples = torch.load(path / "samples.pt", map_location="cpu")
    manifest = json.loads((path / "run_manifest.json").read_text())
    return {
        "metrics": metrics,
        "config": config,
        "convergence": convergence,
        "samples": samples,
        "manifest": manifest,
        "path": path,
    }
