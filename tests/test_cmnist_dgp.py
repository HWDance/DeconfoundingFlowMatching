from pathlib import Path

import torch

from deconfoundingfm.datasets.mnist.mnist_colour import (
    generate_two_color_observational_population,
    load_mnist_idx,
    recolor_foreground_background,
    recolor_foreground_background_batch,
)
from deconfoundingfm.experimental import (
    CMNISTConfig,
    ColorMNISTDGP,
    ExactColorMNISTSourceGenerator,
    load_cmnist_correction_checkpoint,
    recover_color_values,
    save_cmnist_correction_checkpoint,
)
from deconfoundingfm.experimental.cmnist import (
    pack_display_samples,
    pack_display_trajectories,
    unpack_display_trajectories,
)
from deconfoundingfm.nn.velocity import UNet


def test_display_artifact_compaction_preserves_shapes_and_diagnostics():
    images = torch.tensor([-0.1, 0.5, 1.1]).view(1, 3, 1, 1)
    packed_samples = pack_display_samples({"grid": images})
    assert packed_samples["grid"].dtype == torch.uint8
    assert tuple(packed_samples["grid"].shape) == tuple(images.shape)

    score = torch.tensor([0.25])
    packed = pack_display_trajectories(
        {"decfm": {"arm0": {"top": {"trajectory": images, "score": score}}}}
    )
    assert packed["decfm"]["arm0"]["top"]["trajectory"].dtype == torch.uint8
    assert packed["decfm"]["arm0"]["top"]["score"] is score

    restored = unpack_display_trajectories(packed)
    trajectory = restored["decfm"]["arm0"]["top"]["trajectory"]
    assert trajectory.dtype == torch.float32
    assert torch.allclose(trajectory, images.clamp(0, 1), atol=1.0 / 255.0)
    assert restored["decfm"]["arm0"]["top"]["score"] is score


def test_packaged_original_t10k_idx_files_are_used():
    pools = load_mnist_idx(package="deconfoundingfm.datasets.mnist", device="cpu")
    assert sum(len(v) for v in pools.values()) == 10_000
    assert len(pools[1]) == 1135
    assert len(pools[6]) == 958
    assert pools[1].shape[1:] == (1, 28, 28)


def test_vectorized_recolor_is_exactly_scalar_recolor():
    pools = load_mnist_idx(package="deconfoundingfm.datasets.mnist", device="cpu")
    s = pools[1][0]
    x = torch.tensor([0.37])
    scalar = recolor_foreground_background(
        s,
        fg_rgb=(0.37, 0.0, 0.63),
        bg_rgb=(0.0, 0.0, 0.0),
        tau=0.08,
        k=10.0,
    )
    batch = recolor_foreground_background_batch(s.unsqueeze(0), x, fg_alpha=0.0, tau=0.08, k=10.0)[
        0
    ]
    assert torch.equal(scalar, batch)


def test_two_color_population_has_requested_pair_structure():
    pop = generate_two_color_observational_population(
        n_bw_shapes=12,
        color_draws_per_shape=2,
        digits=(1, 6),
        w=5.0,
        device="cpu",
        seed=7,
    )
    assert pop["X"].shape == (24, 1)
    assert pop["A"].shape == (24, 1)
    assert pop["Y"].shape == (24, 3, 28, 28)
    assert pop["meta"]["n_observations"] == 24
    assert torch.equal(pop["shape_id"].view(-1, 2)[:, 0], pop["shape_id"].view(-1, 2)[:, 1])
    assert torch.equal(pop["A"].view(-1, 2)[:, 0], pop["A"].view(-1, 2)[:, 1])
    assert torch.equal(pop["color_draw"].view(-1, 2), torch.tensor([[0, 1]]).expand(12, 2))
    assert ((0.0 <= pop["X"]) & (pop["X"] <= 1.0)).all()


def test_iid_observational_population_is_deterministic_and_rowwise():
    dgp = ColorMNISTDGP(CMNISTConfig(), device="cpu")
    first = dgp.make_iid_observational_population(37, seed=19)
    second = dgp.recreate_observational_population(
        design="iid_observational", n=37, seed=19
    )
    assert first["meta"]["construction"] == "iid_observational"
    assert first["X"].shape == (37, 1)
    assert first["A"].shape == (37, 1)
    assert first["Y"].shape == (37, 3, 28, 28)
    for key in ("X", "A", "Y"):
        assert torch.equal(first[key], second[key])


def test_exact_source_generator_draws_correct_arm_conditioned_colors():
    dgp = ColorMNISTDGP(CMNISTConfig(), device="cpu")
    source = ExactColorMNISTSourceGenerator(dgp)
    torch.manual_seed(11)
    y0 = source.sample(0, 64, device="cpu")
    torch.manual_seed(12)
    y1 = source.sample(1, 64, device="cpu")
    assert y0.shape == y1.shape == (64, 3, 28, 28)
    assert torch.isfinite(y0).all() and torch.isfinite(y1).all()
    # Under positive confounding, X|A=1 is redder on average than X|A=0.
    assert y1[:, 0].mean() > y0[:, 0].mean()


def test_cmnist_runner_estimates_propensity_not_oracle():
    runner = Path(__file__).resolve().parents[1] / "examples" / "cmnist" / "run.py"
    text = runner.read_text()
    assert "RandomForestPropensityEstimator" in text
    assert "OracleSigmoidPropensity" not in text


def test_cmnist_runners_use_requested_full_iid_noise_refresh_defaults():
    root = Path(__file__).resolve().parents[1] / "examples" / "cmnist"
    for runner_name in ("run.py", "run_offline.py"):
        text = (root / runner_name).read_text()
        assert "save_model_checkpoints" in text
        assert '"path_kind": "relative_to_result_directory"' in text
        assert 'default=20_000' in text
        assert 'default=100_000' in text
        assert 'default=0.1' in text
        assert 'default=10_000' in text
        assert "make_iid_observational_population" in text
        assert "update_plugin_reservoir=True" in text
        assert "default=5_000" in text
        assert "default=512" in text




def test_recover_color_values_returns_one_ratio_per_image():
    images = torch.zeros(2, 3, 4, 4)
    images[0, 0, :2, :2] = 0.25
    images[0, 2, :2, :2] = 0.75
    images[1, 0, :, :] = 0.8
    images[1, 2, :, :] = 0.2
    values = recover_color_values(images)
    assert values.shape == (2,)
    assert torch.allclose(values, torch.tensor([0.25, 0.8]))


def test_portable_correction_checkpoint_reloads_and_recreates_data(tmp_path):
    dgp_cfg = CMNISTConfig(n_bw_shapes=4, color_draws_per_shape=2)
    velocity = UNet(in_channels=3, out_channels=3, num_classes=2, c=2)
    path = tmp_path / "step_000001.pt"
    save_cmnist_correction_checkpoint(
        path,
        state_dict=velocity.state_dict(),
        variant="decfm",
        step=1,
        ode_steps=1,
        unet_c=2,
        target_config={"ode_steps": 1, "use_ot": False},
        dgp_config=dgp_cfg,
        observational_seed=7,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["omitted_state"] == [
        "nuisance_outcome",
        "nuisance_propensity",
        "plugin_reservoir",
        "optimizer",
        "cached_base_samples",
    ]
    sampler = load_cmnist_correction_checkpoint(path, device="cpu")
    y0, y1 = sampler.sample(1, 2, return_base=True)
    trajectory, velocities = sampler.trajectory(1, y0=y0)
    assert y0.shape == y1.shape == (2, 3, 28, 28)
    assert trajectory.shape == (2, 2, 3, 28, 28)
    assert velocities.shape == (1, 2, 3, 28, 28)
    assert torch.isfinite(trajectory).all()

    recreated = sampler.recreate_observational_population(device="cpu")
    expected = ColorMNISTDGP(dgp_cfg, device="cpu").make_observational_population(seed=7)
    for key in ("X", "A", "Y", "shape_id", "color_draw"):
        assert torch.equal(recreated[key], expected[key])



def test_offline_runner_uses_fixed_observational_bases_only():
    runner = Path(__file__).resolve().parents[1] / "examples" / "cmnist" / "run_offline.py"
    text = runner.read_text()
    assert "ConditionalFlowFM(offline_nuisance_cfg" in text
    assert "return DeconfoundingFlow(" in text
    assert "GeneratorConditionalFlowFM" not in text
    assert "GeneratorDeconfoundingFlow" not in text
    assert 'base_mode="observational_empirical"' in text
    assert (
        '"nuisance_base": '
        '"fixed_observational_Y_stratified_by_A_plus_gaussian_noise"'
    ) in text
    assert (
        '"target_base": '
        '"fixed_observational_Y_stratified_by_A_plus_gaussian_noise"'
    ) in text
    assert 'make_iid_observational_population' in text


def test_offline_checkpoint_reconstructs_empirical_arm_bases(tmp_path):
    dgp_cfg = CMNISTConfig(n_bw_shapes=8, color_draws_per_shape=2)
    velocity = UNet(in_channels=3, out_channels=3, num_classes=2, c=2)
    path = tmp_path / "offline_step_000001.pt"
    save_cmnist_correction_checkpoint(
        path,
        state_dict=velocity.state_dict(),
        variant="ot",
        step=1,
        ode_steps=1,
        unet_c=2,
        target_config={
            "ode_steps": 1,
            "use_ot": True,
            "base_noise_std": 0.1,
            "update_plugin_reservoir": True,
            "plugin_reservoir_update_frequency": 10_000,
        },
        dgp_config=dgp_cfg,
        observational_seed=13,
        base_mode="observational_empirical",
        observational_design="iid_observational",
        observational_n=16,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["kind"] == "cmnist_correction"
    assert payload["base_mode"] == "observational_empirical"
    assert payload["source_generator"] is None
    assert payload["observational_design"] == "iid_observational"
    assert payload["observational_n"] == 16
    assert "cached_base_samples" not in payload

    sampler = load_cmnist_correction_checkpoint(path, device="cpu")
    assert sampler.base_noise_std == 0.1
    population = sampler.recreate_observational_population(device="cpu")
    expected = ColorMNISTDGP(dgp_cfg, device="cpu").make_iid_observational_population(
        16, seed=13
    )
    for key in ("X", "A", "Y"):
        assert torch.equal(population[key], expected[key])
    treatment = population["A"].reshape(-1).long()
    for arm in (0, 1):
        exact_samples = sampler.base_sampler.sample(arm, 3, device="cpu")
        arm_base = population["Y"][treatment == arm]
        for sample in exact_samples:
            assert (arm_base == sample).flatten(1).all(dim=1).any()
        noised_samples = sampler.sample_base(arm, 3)
        assert not torch.equal(noised_samples, exact_samples)
