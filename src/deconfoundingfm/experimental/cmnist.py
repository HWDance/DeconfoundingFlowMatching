from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..datasets.mnist.mnist_colour import (
    BG_BLACK,
    generate_two_color_observational_population,
    load_mnist_idx,
    recolor_foreground_background_batch,
    sample_x_given_arm,
    true_propensity,
)
from ..integrators import integrate_midpoint, integrate_midpoint_trajectory
from ..nn.velocity import UNet


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
        self.pools = load_mnist_idx(package="deconfoundingfm.datasets.mnist", device=self.device)

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
    def sample_observational(self, n: int, *, seed: int | None = None):
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
    def make_iid_observational_population(self, n: int, *, seed: int = 0):
        """Fixed iid observational dataset with independently generated rows."""
        x, a, y = self.sample_observational(int(n), seed=int(seed))
        return {
            "X": x,
            "A": a,
            "Y": y,
            "meta": {
                "construction": "iid_observational",
                "n_observations": int(n),
                "seed": int(seed),
            },
        }

    @torch.no_grad()
    def recreate_observational_population(
        self,
        *,
        design: str,
        seed: int,
        n: int | None = None,
    ):
        if design == "paired_two_color":
            return self.make_observational_population(seed=int(seed))
        if design == "iid_observational":
            if n is None or int(n) < 1:
                raise ValueError("iid_observational reconstruction requires n >= 1.")
            return self.make_iid_observational_population(int(n), seed=int(seed))
        raise ValueError(f"Unknown CMNIST observational design: {design!r}.")

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


class FixedObservationalCMNISTBaseSampler:
    """Arm-stratified empirical base reconstructed from one fixed population.

    The image tensors are deliberately reconstructed from the DGP configuration
    and observational seed instead of being serialized in every correction
    checkpoint. Sampling is with replacement from observed Y values in the
    requested treatment arm.
    """

    def __init__(
        self,
        dgp: ColorMNISTDGP,
        *,
        observational_seed: int,
        observational_design: str = "paired_two_color",
        observational_n: int | None = None,
    ):
        self.dgp = dgp
        self.observational_seed = int(observational_seed)
        self.observational_design = str(observational_design)
        self.observational_n = None if observational_n is None else int(observational_n)
        self.population = dgp.recreate_observational_population(
            design=self.observational_design,
            seed=self.observational_seed,
            n=self.observational_n,
        )
        treatment = self.population["A"].reshape(-1).long()
        outcomes = self.population["Y"]
        self._bases = {arm: outcomes[treatment == arm] for arm in (0, 1)}
        if any(len(base) == 0 for base in self._bases.values()):
            raise RuntimeError("Fixed CMNIST population is missing a treatment arm.")

    @torch.no_grad()
    def sample(self, a: int, n: int, device=None):
        a = int(a)
        n = int(n)
        if a not in (0, 1):
            raise ValueError("a must be 0 or 1.")
        if n < 1:
            raise ValueError("n must be >= 1.")
        base = self._bases[a]
        indices = torch.randint(len(base), (n,), device=base.device)
        target_device = _maybe_device(device or base.device)
        return base[indices].to(target_device)

    def recreate_observational_population(self, *, device=None):
        target_device = self.dgp.device if device is None else torch.device(device)
        if target_device == self.dgp.device:
            return self.population
        dgp = ColorMNISTDGP(self.dgp.config, device=target_device)
        return dgp.recreate_observational_population(
            design=self.observational_design,
            seed=self.observational_seed,
            n=self.observational_n,
        )



# Backward-compatible name for one release; prefer ExactColorMNISTSourceGenerator.
OracleArmConditionalGenerator = ExactColorMNISTSourceGenerator


CHECKPOINT_FORMAT_VERSION = 1


class CMNISTCorrectionSampler:
    """Inference-only CMNIST correction flow reconstructed from a saved checkpoint.

    The fitted correction velocity and reconstructable base recipe are sufficient
    for post-training sampling. Training nuisances, propensity estimates, plug-in
    reservoirs, optimizer state, and cached base samples are deliberately omitted.
    """

    def __init__(
        self,
        velocity: UNet,
        base_sampler,
        *,
        variant: str,
        step: int,
        ode_steps: int,
        base_noise_std: float,
        observational_seed: int,
        observational_design: str,
        observational_n: int | None,
        checkpoint_metadata: dict,
    ):
        self.velocity = velocity.eval()
        self.base_sampler = base_sampler
        # Compatibility for callers that used this attribute to reach the DGP.
        self.source_generator = base_sampler
        self.variant = str(variant)
        self.step = int(step)
        self.ode_steps = int(ode_steps)
        self.base_noise_std = float(base_noise_std)
        self.observational_seed = int(observational_seed)
        self.observational_design = str(observational_design)
        self.observational_n = None if observational_n is None else int(observational_n)
        self.checkpoint_metadata = checkpoint_metadata

    @property
    def device(self) -> torch.device:
        return next(self.velocity.parameters()).device

    def _context(self, arm: int, n: int) -> torch.Tensor:
        arm = int(arm)
        if arm not in (0, 1):
            raise ValueError("arm must be 0 or 1.")
        context = torch.zeros(int(n), 2, device=self.device)
        context[:, arm] = 1.0
        return context

    @torch.no_grad()
    def sample_base(self, arm: int, n: int) -> torch.Tensor:
        base = self.base_sampler.sample(int(arm), int(n), device=self.device)
        if self.base_noise_std > 0:
            base = base + self.base_noise_std * torch.randn_like(base)
        return base

    @torch.no_grad()
    def transform(
        self,
        arm: int,
        y0: torch.Tensor,
        *,
        ode_steps: int | None = None,
    ) -> torch.Tensor:
        y0 = y0.to(device=self.device, dtype=next(self.velocity.parameters()).dtype)
        steps = self.ode_steps if ode_steps is None else int(ode_steps)
        return integrate_midpoint(
            self.velocity,
            y0,
            context=self._context(int(arm), len(y0)),
            steps=steps,
        )

    @torch.no_grad()
    def sample(
        self,
        arm: int,
        n: int,
        *,
        ode_steps: int | None = None,
        return_base: bool = False,
    ):
        y0 = self.sample_base(int(arm), int(n))
        y1 = self.transform(int(arm), y0, ode_steps=ode_steps)
        if return_base:
            return y0, y1
        return y1

    @torch.no_grad()
    def trajectory(
        self,
        arm: int,
        n: int | None = None,
        *,
        y0: torch.Tensor | None = None,
        ode_steps: int | None = None,
    ):
        if y0 is None:
            if n is None:
                raise ValueError("Provide n or y0.")
            y0 = self.sample_base(int(arm), int(n))
        else:
            y0 = y0.to(device=self.device, dtype=next(self.velocity.parameters()).dtype)
        steps = self.ode_steps if ode_steps is None else int(ode_steps)
        return integrate_midpoint_trajectory(
            self.velocity,
            y0,
            context=self._context(int(arm), len(y0)),
            steps=steps,
        )

    def recreate_observational_population(self, *, device=None):
        target_device = self.device if device is None else torch.device(device)
        if hasattr(self.base_sampler, "recreate_observational_population"):
            return self.base_sampler.recreate_observational_population(device=target_device)
        cfg = self.base_sampler.dgp.config
        dgp = ColorMNISTDGP(cfg, device=target_device)
        return dgp.recreate_observational_population(
            design=self.observational_design,
            seed=self.observational_seed,
            n=self.observational_n,
        )


def save_cmnist_correction_checkpoint(
    path: str | Path,
    *,
    state_dict: dict[str, torch.Tensor],
    variant: str,
    step: int,
    ode_steps: int,
    unet_c: int,
    target_config: dict,
    dgp_config: CMNISTConfig,
    observational_seed: int,
    base_mode: str = "source_generator",
    observational_design: str = "paired_two_color",
    observational_n: int | None = None,
):
    """Save a portable, sample-free correction checkpoint."""
    if base_mode not in {"source_generator", "observational_empirical"}:
        raise ValueError(
            "base_mode must be 'source_generator' or 'observational_empirical'."
        )
    if observational_design not in {"paired_two_color", "iid_observational"}:
        raise ValueError(
            "observational_design must be 'paired_two_color' or 'iid_observational'."
        )
    if observational_design == "iid_observational" and (
        observational_n is None or int(observational_n) < 1
    ):
        raise ValueError("iid_observational checkpoints require observational_n >= 1.")
    dgp_payload = asdict(dgp_config)
    if observational_design == "iid_observational":
        dgp_payload.pop("n_bw_shapes", None)
        dgp_payload.pop("color_draws_per_shape", None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": (
            "cmnist_generator_correction"
            if base_mode == "source_generator"
            else "cmnist_correction"
        ),
        "variant": str(variant),
        "step": int(step),
        "state_dict": {key: value.detach().cpu().clone() for key, value in state_dict.items()},
        "velocity_config": {
            "kind": "unet",
            "in_channels": 3,
            "out_channels": 3,
            "num_classes": 2,
            "c": int(unet_c),
        },
        "target_config": dict(target_config),
        "dgp_config": dgp_payload,
        "observational_seed": int(observational_seed),
        "observational_design": str(observational_design),
        "observational_n": (
            None if observational_n is None else int(observational_n)
        ),
        "base_mode": base_mode,
        "source_generator": (
            "ExactColorMNISTSourceGenerator"
            if base_mode == "source_generator"
            else None
        ),
        "base_reconstruction": (
            "Reconstruct the configured observational population from "
            "observational_design, observational_n, and observational_seed; "
            "use Y[A == arm]"
            if base_mode == "observational_empirical"
            else "ExactColorMNISTSourceGenerator"
        ),
        "omitted_state": [
            "nuisance_outcome",
            "nuisance_propensity",
            "plugin_reservoir",
            "optimizer",
            "cached_base_samples",
        ],
    }
    torch.save(payload, path)
    return path


def load_cmnist_correction_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> CMNISTCorrectionSampler:
    """Load a correction checkpoint and reconstruct its CMNIST base sampler."""
    path = Path(path)
    target_device = _maybe_device(device)
    payload = torch.load(path, map_location=target_device, weights_only=True)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format in {path}.")
    if payload.get("kind") not in {
        "cmnist_generator_correction",
        "cmnist_correction",
    }:
        raise ValueError(f"Not a CMNIST correction checkpoint: {path}.")

    velocity_cfg = payload["velocity_config"]
    if velocity_cfg.get("kind") != "unet":
        raise ValueError(f"Unsupported CMNIST velocity kind: {velocity_cfg.get('kind')!r}.")
    velocity = UNet(
        in_channels=int(velocity_cfg["in_channels"]),
        out_channels=int(velocity_cfg["out_channels"]),
        num_classes=int(velocity_cfg["num_classes"]),
        c=int(velocity_cfg["c"]),
    ).to(target_device)
    velocity.load_state_dict(payload["state_dict"], strict=True)

    dgp_values = dict(payload["dgp_config"])
    dgp_values["digits"] = tuple(dgp_values["digits"])
    dgp = ColorMNISTDGP(CMNISTConfig(**dgp_values), device=target_device)
    base_mode = payload.get("base_mode", "source_generator")
    observational_design = payload.get("observational_design", "paired_two_color")
    observational_n = payload.get("observational_n")
    if base_mode == "source_generator":
        base_sampler = ExactColorMNISTSourceGenerator(dgp)
    elif base_mode == "observational_empirical":
        base_sampler = FixedObservationalCMNISTBaseSampler(
            dgp,
            observational_seed=payload["observational_seed"],
            observational_design=observational_design,
            observational_n=observational_n,
        )
    else:
        raise ValueError(f"Unsupported CMNIST checkpoint base mode: {base_mode!r}.")
    target_cfg = payload.get("target_config", {})
    return CMNISTCorrectionSampler(
        velocity,
        base_sampler,
        variant=payload["variant"],
        step=payload["step"],
        ode_steps=int(target_cfg.get("ode_steps", payload.get("ode_steps", 50))),
        base_noise_std=float(target_cfg.get("base_noise_std", 0.0)),
        observational_seed=payload["observational_seed"],
        observational_design=observational_design,
        observational_n=observational_n,
        checkpoint_metadata=payload,
    )


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


def _pack_display_tensor(images: torch.Tensor) -> torch.Tensor:
    """Store display-only RGB tensors compactly without changing their shape."""
    if not torch.is_floating_point(images):
        return images.detach().cpu()
    return (
        images.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    )


def _unpack_display_tensor(images: torch.Tensor) -> torch.Tensor:
    if images.dtype == torch.uint8:
        return images.float().div(255.0)
    return images


def pack_display_samples(samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Quantize plotting-grid images; metrics are never computed from this artifact."""
    return {key: _pack_display_tensor(value) for key, value in samples.items()}


def _map_trajectory_images(value, *, pack: bool, parent_key: str | None = None):
    if isinstance(value, dict):
        return {
            key: _map_trajectory_images(item, pack=pack, parent_key=key)
            for key, item in value.items()
        }
    if parent_key == "trajectory" and torch.is_tensor(value):
        return _pack_display_tensor(value) if pack else _unpack_display_tensor(value)
    return value


def pack_display_trajectories(trajectories: dict) -> dict:
    """Quantize only trajectory image tensors, retaining exact scalar diagnostics."""
    return _map_trajectory_images(trajectories, pack=True)


def unpack_display_trajectories(trajectories: dict) -> dict:
    return _map_trajectory_images(trajectories, pack=False)


def load_result_bundle(path):
    path = Path(path)
    metrics = json.loads((path / "metrics.json").read_text())
    config = json.loads((path / "config.json").read_text())
    convergence = json.loads((path / "convergence.json").read_text())
    stored_samples = torch.load(
        path / "samples.pt", map_location="cpu", weights_only=True
    )
    samples = {
        key: _unpack_display_tensor(value) for key, value in stored_samples.items()
    }
    manifest = json.loads((path / "run_manifest.json").read_text())
    bundle = {
        "metrics": metrics,
        "config": config,
        "convergence": convergence,
        "samples": samples,
        "manifest": manifest,
        "path": path,
    }
    optional_json = {
        "trajectory_summary": "trajectory_summary.json",
        "color_diagnostics": "color_diagnostics.json",
        "model_manifest": "model_manifest.json",
        "data_manifest": "data_manifest.json",
    }
    for key, filename in optional_json.items():
        if (path / filename).exists():
            bundle[key] = json.loads((path / filename).read_text())
    optional_tensors = {
        "trajectories": "trajectories.pt",
        "color_values": "color_values.pt",
    }
    for key, filename in optional_tensors.items():
        if (path / filename).exists():
            value = torch.load(path / filename, map_location="cpu", weights_only=True)
            bundle[key] = (
                unpack_display_trajectories(value) if key == "trajectories" else value
            )
    return bundle


@torch.no_grad()
def recover_color_values(
    images: torch.Tensor, *, foreground_threshold: float = 0.05
) -> torch.Tensor:
    """Recover one scalar color value per CMNIST image using ``R / (R + B)``.

    Foreground pixels are selected from total RGB intensity. Aggregating red and
    blue mass before taking the ratio prevents larger digits from receiving more
    weight in downstream distributional comparisons.
    """
    images = images.detach().cpu().float().clamp(0.0, 1.0)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected images with shape (N,3,H,W).")
    r = images[:, 0]
    b = images[:, 2]
    foreground = images.sum(dim=1) > float(foreground_threshold)
    r_mass = (r * foreground).sum(dim=(1, 2))
    b_mass = (b * foreground).sum(dim=(1, 2))
    denom = r_mass + b_mass
    values = r_mass / denom.clamp_min(1e-8)
    values[denom <= 1e-8] = float("nan")
    return values


def empirical_w1_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    """Empirical one-dimensional W1 using a shared quantile grid."""
    x = x.detach().cpu().flatten()
    y = y.detach().cpu().flatten()
    x = x[torch.isfinite(x)]
    y = y[torch.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    n_quantiles = max(len(x), len(y), 2)
    q = torch.linspace(0.0, 1.0, n_quantiles)
    return float((torch.quantile(x, q) - torch.quantile(y, q)).abs().mean())


def ks_uniform_statistic(x: torch.Tensor) -> float:
    x = x.detach().cpu().flatten()
    x = x[torch.isfinite(x)].clamp(0.0, 1.0).sort().values
    n = len(x)
    if n == 0:
        return float("nan")
    upper = torch.arange(1, n + 1, dtype=x.dtype) / n
    lower = torch.arange(0, n, dtype=x.dtype) / n
    return float(torch.maximum((upper - x).max(), (x - lower).max()))


def color_distribution_diagnostics(samples_by_arm: dict[int, torch.Tensor]) -> dict:
    values0 = recover_color_values(samples_by_arm[0])
    values1 = recover_color_values(samples_by_arm[1])
    values0 = values0[torch.isfinite(values0)]
    values1 = values1[torch.isfinite(values1)]
    n_uniform = max(min(len(values0), len(values1)), 2)
    uniform_ref = (torch.arange(n_uniform, dtype=torch.float32) + 0.5) / n_uniform
    return {
        "arm0_vs_arm1_w1": empirical_w1_1d(values0, values1),
        "arm0_vs_uniform_w1": empirical_w1_1d(values0, uniform_ref),
        "arm1_vs_uniform_w1": empirical_w1_1d(values1, uniform_ref),
        "arm0_vs_uniform_ks": ks_uniform_statistic(values0),
        "arm1_vs_uniform_ks": ks_uniform_statistic(values1),
        "n_images_arm0": len(values0),
        "n_images_arm1": len(values1),
    }
