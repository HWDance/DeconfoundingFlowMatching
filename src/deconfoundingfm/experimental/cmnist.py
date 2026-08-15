from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

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
    bundle = {
        "metrics": metrics,
        "config": config,
        "convergence": convergence,
        "samples": samples,
        "manifest": manifest,
        "path": path,
    }
    if (path / "trajectory_summary.json").exists():
        bundle["trajectory_summary"] = json.loads((path / "trajectory_summary.json").read_text())
    if (path / "color_diagnostics.json").exists():
        bundle["color_diagnostics"] = json.loads((path / "color_diagnostics.json").read_text())
    if (path / "trajectories.pt").exists():
        bundle["trajectories"] = torch.load(path / "trajectories.pt", map_location="cpu")
    if (path / "model_manifest.json").exists():
        bundle["model_manifest"] = json.loads((path / "model_manifest.json").read_text())
    return bundle



@torch.no_grad()
def recover_color_values(images: torch.Tensor, *, foreground_threshold: float = 0.05) -> torch.Tensor:
    """Recover scalar color values ``x`` from RGB CMNIST images.

    The original CMNIST map uses foreground colors ``(x, alpha, 1-x)`` on a black
    background.  We therefore estimate ``x`` on foreground pixels by
    ``R / (R + B)`` and aggregate across all foreground pixels.
    """
    x = images.detach().cpu().float().clamp(0.0, 1.0)
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError('Expected images with shape (N,3,H,W).')
    r = x[:, 0]
    b = x[:, 2]
    denom = (r + b).clamp_min(1e-8)
    fg = (x.sum(dim=1) > foreground_threshold)
    vals = (r / denom)[fg]
    return vals.clamp(0.0, 1.0)


def empirical_w1_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().cpu().flatten().sort().values
    y = y.detach().cpu().flatten().sort().values
    n = min(len(x), len(y))
    if n == 0:
        return float('nan')
    return float((x[:n] - y[:n]).abs().mean())


def ks_uniform_statistic(x: torch.Tensor) -> float:
    x = x.detach().cpu().flatten().sort().values
    n = len(x)
    if n == 0:
        return float('nan')
    u = torch.arange(1, n + 1, dtype=x.dtype) / n
    u0 = torch.arange(0, n, dtype=x.dtype) / n
    d_plus = (u - x).abs().max()
    d_minus = (x - u0).abs().max()
    return float(torch.maximum(d_plus, d_minus))


def color_distribution_diagnostics(samples_by_arm: dict[int, torch.Tensor]) -> dict:
    vals0 = recover_color_values(samples_by_arm[0])
    vals1 = recover_color_values(samples_by_arm[1])
    uniform_ref = torch.linspace(0.0, 1.0, steps=min(len(vals0), len(vals1), 2048))
    out = {
        'px0_vs_px1_w1': empirical_w1_1d(vals0, vals1),
        'px0_vs_uniform_w1': empirical_w1_1d(vals0, uniform_ref),
        'px1_vs_uniform_w1': empirical_w1_1d(vals1, uniform_ref),
        'px0_vs_uniform_ks': ks_uniform_statistic(vals0),
        'px1_vs_uniform_ks': ks_uniform_statistic(vals1),
        'n_recovered_px0': int(len(vals0)),
        'n_recovered_px1': int(len(vals1)),
    }
    return out


def make_inception_feature_extractor(*, device='cpu'):
    from torchvision.models import Inception_V3_Weights, inception_v3
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


@torch.no_grad()
def inception_features(images: torch.Tensor, model, *, batch_size: int = 64, device='cpu') -> torch.Tensor:
    images = images.detach().float()
    feats = []
    device = torch.device(device)
    for start in range(0, len(images), batch_size):
        x = images[start:start + batch_size].to(device)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = x.clamp(0.0, 1.0)
        f = model(x)
        if isinstance(f, tuple):
            f = f[0]
        feats.append(f.detach().cpu())
    return torch.cat(feats, dim=0)


def _cov(feats: torch.Tensor) -> torch.Tensor:
    x = feats.float()
    x = x - x.mean(dim=0, keepdim=True)
    denom = max(len(x) - 1, 1)
    return (x.T @ x) / denom


def _trace_sqrt_product(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    # sqrtm(c1^{1/2} c2 c1^{1/2}) using eigen decompositions on CPU.
    evals1, evecs1 = torch.linalg.eigh(c1)
    evals1 = evals1.clamp_min(0)
    c1_half = (evecs1 * evals1.sqrt().unsqueeze(0)) @ evecs1.T
    prod = c1_half @ c2 @ c1_half
    evals_prod = torch.linalg.eigvalsh(prod).clamp_min(0)
    return evals_prod.sqrt().sum()


def fid_from_features(real_feats: torch.Tensor, fake_feats: torch.Tensor) -> float:
    mu1 = real_feats.float().mean(dim=0)
    mu2 = fake_feats.float().mean(dim=0)
    c1 = _cov(real_feats)
    c2 = _cov(fake_feats)
    mean_term = (mu1 - mu2).square().sum()
    trace_term = torch.trace(c1) + torch.trace(c2) - 2.0 * _trace_sqrt_product(c1, c2)
    return float((mean_term + trace_term).cpu())
