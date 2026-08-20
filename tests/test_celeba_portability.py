from __future__ import annotations

import torch

from deconfoundingfm.experimental import (
    CelebAGenderHairConfig,
    generate_celeba_indices,
    load_celeba_correction_checkpoint,
    save_celeba_correction_checkpoint,
    validate_celeba_checkpoint_indices,
)
from deconfoundingfm.nn.velocity import UNet


class _FakePool:
    def cell_indices(self):
        return {
            (0, 0): torch.arange(0, 40),
            (1, 0): torch.arange(40, 80),
            (0, 1): torch.arange(80, 120),
            (1, 1): torch.arange(120, 160),
        }


def test_celeba_index_generation_is_seeded_and_disjoint():
    config = CelebAGenderHairConfig(
        n_obs=20,
        n_ref=4,
        target_px1=0.5,
        p_a1_x0=0.5,
        p_a1_x1=0.5,
        seed=17,
    )
    first = generate_celeba_indices(_FakePool(), config)
    second = generate_celeba_indices(_FakePool(), config)
    for key in first:
        assert torch.equal(first[key], second[key])
    assert len(first["train_indices"]) == 20
    assert len(first["ref_indices_a0"]) == len(first["ref_indices_a1"]) == 4
    observed = set(first["train_indices"].tolist())
    ref0 = set(first["ref_indices_a0"].tolist())
    ref1 = set(first["ref_indices_a1"].tolist())
    assert observed.isdisjoint(ref0)
    assert observed.isdisjoint(ref1)
    assert ref0.isdisjoint(ref1)


def test_celeba_checkpoint_roundtrip_and_empirical_base_refill(tmp_path):
    velocity = UNet(in_channels=3, out_channels=3, num_classes=2, c=2)
    config = CelebAGenderHairConfig(n_obs=4, n_ref=2, seed=9)
    indices = {
        "train_indices": torch.tensor([3, 1, 8, 5]),
        "ref_indices_a0": torch.tensor([2, 7]),
        "ref_indices_a1": torch.tensor([4, 6]),
    }
    path = tmp_path / "epoch_500.pt"
    save_celeba_correction_checkpoint(
        path,
        state_dict=velocity.state_dict(),
        variant="ot",
        epoch=500,
        ode_steps=1,
        unet_c=2,
        target_config={"ode_steps": 1, "base_noise_std": 0.1, "use_ot": True},
        plugin_config={"base_kind": "empirical"},
        propensity={"kind": "empirical", "smoothing": 10.0},
        data_config=config,
        data_indices=indices,
        data_root_hint=None,
        legacy_provenance={"source": "unit-test"},
        legacy_evaluation={"sw2_mean": 0.1},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert validate_celeba_checkpoint_indices(payload)
    assert payload["kind"] == "celeba_correction"
    assert payload["base_mode"] == "observational_empirical"
    assert payload["step"] == payload["epoch"] == 500
    assert "cached_base_samples" not in payload

    sampler = load_celeba_correction_checkpoint(
        path,
        device="cpu",
        reconstruct_base=False,
    )
    images = torch.linspace(-1, 1, 4 * 3 * 8 * 8).reshape(4, 3, 8, 8)
    treatment = torch.tensor([0, 0, 1, 1])
    sampler.attach_population(images, treatment)
    torch.manual_seed(1)
    y0, y1 = sampler.sample(1, 2, return_base=True)
    trajectory, velocities = sampler.trajectory(1, y0=y0)
    assert y0.shape == y1.shape == (2, 3, 8, 8)
    assert trajectory.shape == (2, 2, 3, 8, 8)
    assert velocities.shape == (1, 2, 3, 8, 8)
    assert torch.isfinite(trajectory).all()
