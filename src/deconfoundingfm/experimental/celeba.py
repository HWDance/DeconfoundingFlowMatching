"""Portable inference utilities for the CelebA gender/hair experiment.

The image files are deliberately not packaged. A checkpoint stores the exact
data-generating configuration and sampled record indices, and this module
reconstructs the observational empirical base from a standard aligned CelebA
directory before sampling.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..integrators import integrate_midpoint, integrate_midpoint_trajectory
from ..nn.velocity import UNet

CELEBA_CHECKPOINT_FORMAT_VERSION = 1
_SPLIT_IDS = {"train": 0, "valid": 1, "val": 1, "test": 2}


def _maybe_device(device=None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass(frozen=True)
class CelebAGenderHairConfig:
    """Configuration of the legacy controlled CelebA experiment."""

    image_size: int = 64
    n_obs: int = 20_000
    n_ref: int = 2_000
    train_split: str = "train"
    ref_split: str = "valid"
    split_mode: str = "custom"
    source_splits: tuple[str, ...] = ("train", "valid")
    a_name: str = "Male"
    x_name: str = "Blond_Hair"
    clean_hair: bool = True
    hair_pair: tuple[str, str] = ("Blond_Hair", "Brown_Hair")
    drop_attrs: tuple[str, ...] = ("Black_Hair", "Gray_Hair", "Bald", "Wearing_Hat")
    target_px1: float = 0.3
    p_a1_x0: float = 0.6
    p_a1_x1: float = 0.1
    seed: int = 2


@dataclass(frozen=True)
class CelebARecord:
    filename: str
    A: int
    X: int


def _normalise_splits(source_splits: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(source_splits, str):
        values = source_splits.replace("+", ",").split(",")
    else:
        values = source_splits
    out = []
    for value in values:
        split = str(value).strip().lower()
        if not split:
            continue
        if split not in _SPLIT_IDS:
            raise ValueError(f"Unknown CelebA split {split!r}.")
        if split not in out:
            out.append(split)
    if not out:
        raise ValueError("source_splits must not be empty.")
    return tuple(out)


def _sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts_from_probs(probs: dict[tuple[int, int], float], n: int):
    raw = {key: int(n) * float(value) for key, value in probs.items()}
    counts = {key: int(np.floor(value)) for key, value in raw.items()}
    remainder = int(n) - sum(counts.values())
    order = sorted(
        raw,
        key=lambda key: (raw[key] - counts[key], str(key)),
        reverse=True,
    )
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def _observational_counts(config: CelebAGenderHairConfig):
    q = float(config.target_px1)
    p0 = float(config.p_a1_x0)
    p1 = float(config.p_a1_x1)
    return _counts_from_probs(
        {
            (0, 0): (1.0 - q) * (1.0 - p0),
            (1, 0): (1.0 - q) * p0,
            (0, 1): q * (1.0 - p1),
            (1, 1): q * p1,
        },
        config.n_obs,
    )


def _reference_counts(config: CelebAGenderHairConfig, arm: int):
    q = float(config.target_px1)
    return _counts_from_probs(
        {(int(arm), 0): 1.0 - q, (int(arm), 1): q},
        config.n_ref,
    )


class CelebAGenderHairPool:
    """Minimal aligned-CelebA reader matching the legacy record ordering."""

    def __init__(self, root: str | Path, config: CelebAGenderHairConfig):
        self.root = Path(root).expanduser().resolve()
        self.config = config
        self.img_dir = self.root / "img_align_celeba"
        self.attr_file = self.root / "list_attr_celeba.txt"
        self.split_file = self.root / "list_eval_partition.txt"
        for path in (self.img_dir, self.attr_file, self.split_file):
            if not path.exists():
                raise FileNotFoundError(f"Missing CelebA input: {path}")
        if config.split_mode != "custom":
            raise ValueError(
                "Portable legacy CelebA checkpoints currently require split_mode='custom'."
            )
        self.records = self._load_combined_records()

    def _load_combined_records(self) -> list[CelebARecord]:
        partition = {}
        with self.split_file.open() as handle:
            for line in handle:
                filename, split_id = line.split()
                partition[filename] = int(split_id)

        with self.attr_file.open() as handle:
            _ = int(handle.readline())
            names = handle.readline().split()
            attr_index = {name: index for index, name in enumerate(names)}
            required = [
                self.config.a_name,
                self.config.x_name,
                *self.config.hair_pair,
                *self.config.drop_attrs,
            ]
            missing = [name for name in required if name not in attr_index]
            if missing:
                raise ValueError(f"Missing CelebA attributes: {missing}")
            rows = []
            for line in handle:
                parts = line.split()
                filename = parts[0]
                values = tuple(1 if int(value) == 1 else 0 for value in parts[1:])
                rows.append((filename, values))

        hair_indices = [attr_index[name] for name in self.config.hair_pair]
        drop_indices = [attr_index[name] for name in self.config.drop_attrs]
        a_index = attr_index[self.config.a_name]
        x_index = attr_index[self.config.x_name]
        records = []
        # The legacy custom loader concatenated separately filtered official
        # splits, rather than filtering their union in global filename order.
        for split in _normalise_splits(self.config.source_splits):
            split_id = _SPLIT_IDS[split]
            for filename, values in rows:
                if partition.get(filename) != split_id:
                    continue
                if self.config.clean_hair:
                    if sum(values[index] for index in hair_indices) != 1:
                        continue
                    if any(values[index] == 1 for index in drop_indices):
                        continue
                records.append(
                    CelebARecord(
                        filename=filename,
                        A=int(values[a_index]),
                        X=int(values[x_index]),
                    )
                )
        return records

    def cell_indices(self):
        treatment = torch.tensor([record.A for record in self.records])
        covariate = torch.tensor([record.X for record in self.records])
        return {
            (arm, x): torch.where((treatment == arm) & (covariate == x))[0]
            for arm in (0, 1)
            for x in (0, 1)
        }

    def labels(self, indices: torch.Tensor):
        positions = indices.detach().cpu().tolist()
        treatment = torch.tensor([self.records[index].A for index in positions], dtype=torch.long)
        covariate = torch.tensor([self.records[index].X for index in positions], dtype=torch.float32)
        return treatment.view(-1, 1), covariate.view(-1, 1)

    def _load_image(self, index: int) -> torch.Tensor:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - exercised only without demo extras
            raise ImportError("CelebA reconstruction requires Pillow.") from exc
        image = Image.open(self.img_dir / self.records[int(index)].filename).convert("RGB")
        width, height = image.size
        crop = min(width, height, 178)
        left = (width - crop) // 2
        top = (height - crop) // 2
        image = image.crop((left, top, left + crop, top + crop))
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize((self.config.image_size, self.config.image_size), resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = 2.0 * array - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def load_images(
        self,
        indices: torch.Tensor,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        indices = indices.detach().cpu().long()
        shape = (len(indices), 3, self.config.image_size, self.config.image_size)
        images = torch.empty(shape, dtype=dtype)
        for position, index in enumerate(indices.tolist()):
            images[position].copy_(self._load_image(index).to(dtype=dtype))
        return images.to(device)


def generate_celeba_indices(
    pool: CelebAGenderHairPool,
    config: CelebAGenderHairConfig,
):
    """Regenerate the legacy disjoint observed/reference index sets."""
    generator = torch.Generator().manual_seed(int(config.seed))
    cells = pool.cell_indices()
    obs_counts = _observational_counts(config)
    ref_counts = {arm: _reference_counts(config, arm) for arm in (0, 1)}
    observed_parts = []
    reference_parts = {0: [], 1: []}

    for cell in ((0, 0), (1, 0), (0, 1), (1, 1)):
        n_observed = int(obs_counts.get(cell, 0))
        n_ref0 = int(ref_counts[0].get(cell, 0))
        n_ref1 = int(ref_counts[1].get(cell, 0))
        total = n_observed + n_ref0 + n_ref1
        available = int(cells[cell].numel())
        if total > available:
            raise ValueError(
                f"CelebA cell {cell} needs {total} records but only {available} are available."
            )
        shuffled = cells[cell][torch.randperm(available, generator=generator)]
        observed_parts.append(shuffled[:n_observed])
        cursor = n_observed
        if n_ref0:
            reference_parts[0].append(shuffled[cursor : cursor + n_ref0])
            cursor += n_ref0
        if n_ref1:
            reference_parts[1].append(shuffled[cursor : cursor + n_ref1])

    observed = torch.cat(observed_parts)
    observed = observed[torch.randperm(len(observed), generator=generator)]
    references = {}
    for arm in (0, 1):
        values = torch.cat(reference_parts[arm])
        references[arm] = values[torch.randperm(len(values), generator=generator)]
    return {
        "train_indices": observed,
        "ref_indices_a0": references[0],
        "ref_indices_a1": references[1],
    }


def _coerce_config(values: dict) -> CelebAGenderHairConfig:
    values = dict(values)
    for key in ("source_splits", "hair_pair", "drop_attrs"):
        if key in values:
            values[key] = tuple(values[key])
    allowed = set(CelebAGenderHairConfig.__dataclass_fields__)
    return CelebAGenderHairConfig(**{key: value for key, value in values.items() if key in allowed})


def reconstruct_celeba_data(
    config: CelebAGenderHairConfig | dict,
    *,
    root: str | Path,
    expected_indices: dict[str, torch.Tensor] | None = None,
    load_observational: bool = True,
    load_references: bool = True,
    device: str | torch.device = "cpu",
):
    """Recreate exact legacy data and verify checkpointed indices when supplied."""
    if isinstance(config, dict):
        config = _coerce_config(config)
    pool = CelebAGenderHairPool(root, config)
    indices = generate_celeba_indices(pool, config)
    if expected_indices is not None:
        for name, generated in indices.items():
            expected = expected_indices[name].detach().cpu().long()
            if not torch.equal(generated, expected):
                raise RuntimeError(
                    f"Reconstructed {name} differs from the checkpoint; refusing to refill bases."
                )

    treatment, covariate = pool.labels(indices["train_indices"])
    result = {
        "A": treatment.to(device),
        "X": covariate.to(device),
        "indices": {key: value.clone() for key, value in indices.items()},
        "pool_size": len(pool.records),
        "metadata_sha256": {
            "list_attr_celeba.txt": _sha256_file(pool.attr_file),
            "list_eval_partition.txt": _sha256_file(pool.split_file),
        },
    }
    if load_observational:
        result["Y"] = pool.load_images(indices["train_indices"], device=device)
    if load_references:
        for arm in (0, 1):
            key = f"ref_indices_a{arm}"
            ref_a, ref_x = pool.labels(indices[key])
            result[f"Y{arm}_ref"] = pool.load_images(indices[key], device=device)
            result[f"A{arm}_ref"] = ref_a.to(device)
            result[f"X{arm}_ref"] = ref_x.to(device)
    return result


class FixedCelebABaseSampler:
    """Arm-stratified empirical base reconstructed from the legacy population."""

    def __init__(self, Y: torch.Tensor, A: torch.Tensor, *, device=None):
        self.device = _maybe_device(device)
        treatment = A.detach().reshape(-1).long()
        images = Y.detach()
        self.base_by_arm = {
            arm: images[treatment == arm].to(self.device) for arm in (0, 1)
        }
        if any(len(values) == 0 for values in self.base_by_arm.values()):
            raise ValueError("CelebA empirical base requires observations in both arms.")

    def sample(self, arm: int, n: int, *, device=None):
        arm = int(arm)
        if arm not in (0, 1):
            raise ValueError("arm must be 0 or 1.")
        base = self.base_by_arm[arm]
        indices = torch.randint(len(base), (int(n),), device=base.device)
        values = base[indices]
        return values.to(self.device if device is None else device)


class CelebACorrectionSampler:
    """Inference-only target flow with its reconstructed empirical base."""

    def __init__(
        self,
        velocity: UNet,
        base_sampler: FixedCelebABaseSampler | None,
        *,
        variant: str,
        epoch: int,
        ode_steps: int,
        base_noise_std: float,
        checkpoint_metadata: dict,
    ):
        self.velocity = velocity.eval()
        self.base_sampler = base_sampler
        self.variant = str(variant)
        self.epoch = int(epoch)
        self.ode_steps = int(ode_steps)
        self.base_noise_std = float(base_noise_std)
        self.checkpoint_metadata = checkpoint_metadata

    @property
    def device(self):
        return next(self.velocity.parameters()).device

    def attach_population(self, Y: torch.Tensor, A: torch.Tensor):
        self.base_sampler = FixedCelebABaseSampler(Y, A, device=self.device)
        return self

    def _context(self, arm: int, n: int):
        if int(arm) not in (0, 1):
            raise ValueError("arm must be 0 or 1.")
        context = torch.zeros(int(n), 2, device=self.device)
        context[:, int(arm)] = 1.0
        return context

    @torch.no_grad()
    def sample_base(self, arm: int, n: int):
        if self.base_sampler is None:
            raise RuntimeError("Attach or reconstruct the CelebA observational population first.")
        values = self.base_sampler.sample(int(arm), int(n), device=self.device)
        if self.base_noise_std > 0:
            values = values + self.base_noise_std * torch.randn_like(values)
        return values

    @torch.no_grad()
    def transform(self, arm: int, y0: torch.Tensor, *, ode_steps: int | None = None):
        y0 = y0.to(self.device, dtype=next(self.velocity.parameters()).dtype)
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
        return (y0, y1) if return_base else y1

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
            y0 = y0.to(self.device, dtype=next(self.velocity.parameters()).dtype)
        steps = self.ode_steps if ode_steps is None else int(ode_steps)
        return integrate_midpoint_trajectory(
            self.velocity,
            y0,
            context=self._context(int(arm), len(y0)),
            steps=steps,
        )


def save_celeba_correction_checkpoint(
    path: str | Path,
    *,
    state_dict: dict[str, torch.Tensor],
    variant: str,
    epoch: int,
    ode_steps: int,
    unet_c: int,
    target_config: dict,
    plugin_config: dict,
    propensity: dict,
    data_config: CelebAGenderHairConfig | dict,
    data_indices: dict[str, torch.Tensor],
    data_root_hint: str | Path | None,
    legacy_provenance: dict,
    legacy_evaluation: dict,
):
    """Save a compact, sample-free CelebA correction checkpoint."""
    if isinstance(data_config, CelebAGenderHairConfig):
        data_values = asdict(data_config)
    else:
        data_values = dict(data_config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = {
        key: value.detach().cpu().long().clone() for key, value in data_indices.items()
    }
    payload = {
        "format_version": CELEBA_CHECKPOINT_FORMAT_VERSION,
        "kind": "celeba_correction",
        "variant": str(variant),
        "seed": int(data_values["seed"]),
        "epoch": int(epoch),
        "step": int(epoch),
        "step_unit": "epoch",
        "state_dict": {
            key: value.detach().cpu().clone() for key, value in state_dict.items()
        },
        "velocity_config": {
            "kind": "unet",
            "in_channels": 3,
            "out_channels": 3,
            "num_classes": 2,
            "c": int(unet_c),
        },
        "target_config": dict(target_config),
        "plugin_config": dict(plugin_config),
        "propensity": dict(propensity),
        "dgp_config": data_values,
        "observational_seed": int(data_values["seed"]),
        "observational_design": "custom_disjoint_attribute_cells",
        "observational_n": int(data_values["n_obs"]),
        "reference_n_per_arm": int(data_values["n_ref"]),
        "data_indices": indices,
        "data_index_sha256": {key: _sha256_tensor(value) for key, value in indices.items()},
        "data_root_hint": None if data_root_hint is None else str(data_root_hint),
        "base_mode": "observational_empirical",
        "base_reconstruction": (
            "Regenerate the seed-2 custom train+valid split; verify the stored exact indices; "
            "load Y and stratify by A."
        ),
        "legacy_provenance": dict(legacy_provenance),
        "legacy_evaluation": dict(legacy_evaluation),
        "omitted_state": [
            "nuisance_outcome",
            "nuisance_propensity",
            "plugin_reservoir",
            "optimizer",
            "cached_base_samples",
            "preview_samples",
        ],
    }
    torch.save(payload, path)
    return path


def _resolve_data_root(payload: dict, data_root: str | Path | None):
    if data_root is not None:
        return Path(data_root)
    if os.environ.get("CELEBA_ROOT"):
        return Path(os.environ["CELEBA_ROOT"])
    if payload.get("data_root_hint"):
        return Path(payload["data_root_hint"])
    raise ValueError("Provide data_root=... or set CELEBA_ROOT to reconstruct CelebA bases.")


def load_celeba_correction_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    data_root: str | Path | None = None,
    population: dict | None = None,
    reconstruct_base: bool = True,
):
    """Load target weights and, by default, reconstruct the exact empirical base."""
    target_device = _maybe_device(device)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != CELEBA_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported CelebA checkpoint format in {path}.")
    if payload.get("kind") != "celeba_correction":
        raise ValueError(f"Not a CelebA correction checkpoint: {path}")
    velocity_config = payload["velocity_config"]
    velocity = UNet(
        in_channels=int(velocity_config["in_channels"]),
        out_channels=int(velocity_config["out_channels"]),
        num_classes=int(velocity_config["num_classes"]),
        c=int(velocity_config["c"]),
    ).to(target_device)
    velocity.load_state_dict(payload["state_dict"], strict=True)

    base_sampler = None
    if reconstruct_base:
        if population is None:
            population = reconstruct_celeba_data(
                payload["dgp_config"],
                root=_resolve_data_root(payload, data_root),
                expected_indices=payload["data_indices"],
                load_references=False,
                device="cpu",
            )
        base_sampler = FixedCelebABaseSampler(
            population["Y"], population["A"], device=target_device
        )

    target_config = payload.get("target_config", {})
    return CelebACorrectionSampler(
        velocity,
        base_sampler,
        variant=payload["variant"],
        epoch=payload["epoch"],
        ode_steps=int(target_config.get("ode_steps", 50)),
        base_noise_std=float(target_config.get("base_noise_std", 0.0)),
        checkpoint_metadata=payload,
    )


def validate_celeba_checkpoint_indices(payload: dict):
    for key, expected_hash in payload["data_index_sha256"].items():
        actual_hash = _sha256_tensor(payload["data_indices"][key])
        if actual_hash != expected_hash:
            raise RuntimeError(f"Checkpoint index hash failed for {key}.")
    return True


def arm_counts(A: torch.Tensor):
    treatment = A.detach().cpu().reshape(-1).long()
    return {f"arm{arm}": int((treatment == arm).sum()) for arm in (0, 1)}


def select_trajectory_frames(trajectory: torch.Tensor, fractions: Iterable[float]):
    steps = len(trajectory) - 1
    return torch.stack(
        [trajectory[round(float(fraction) * steps)] for fraction in fractions], dim=0
    )
