from pathlib import Path

import torch

from deconfoundingfm.datasets.mnist.mnist_colour import (
    load_mnist_idx,
    recolor_foreground_background,
    recolor_foreground_background_batch,
    generate_two_color_observational_population,
)
from deconfoundingfm.experimental import (
    CMNISTConfig,
    ColorMNISTDGP,
    ExactColorMNISTSourceGenerator,
)


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
    batch = recolor_foreground_background_batch(
        s.unsqueeze(0), x, fg_alpha=0.0, tau=0.08, k=10.0
    )[0]
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


def test_cmnist_gaussian_baseline_uses_generator_free_nuisance():
    runner = Path(__file__).resolve().parents[1] / "examples" / "cmnist" / "run.py"
    text = runner.read_text()
    assert "gaussian_nuisance = ConditionalFlowFM(" in text
    assert "nuisance_model=gaussian_nuisance" in text
    assert 'gaussian_nuisance_cfg.base_kind = "gaussian"' in text
    assert 'hasattr(gaussian_nuisance, "source_generator")' in text
