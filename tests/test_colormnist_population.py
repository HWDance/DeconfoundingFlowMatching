import torch

from deconfoundingfm.datasets import ColorMNISTConfig, ColorMNISTPopulation


def _pools(size=8):
    # Two simple grayscale shape pools; enough to exercise vectorized sampling.
    one = torch.zeros(4, 1, size, size)
    one[:, :, 1:-1, size // 2] = 1.0
    six = torch.zeros(5, 1, size, size)
    six[:, :, 1:-1, 1] = 1.0
    six[:, :, -2, 1:-2] = 1.0
    return {1: one, 6: six}


def test_colormnist_population_shapes_and_fresh_sampling():
    source = ColorMNISTPopulation(
        _pools(),
        config=ColorMNISTConfig(digits=(1, 6), confounding_w=5.0),
        device="cpu",
    )
    torch.manual_seed(0)
    batch = source.sample_observational(16)
    assert batch["X"].shape == (16, 1)
    assert batch["A"].shape == (16, 1)
    assert batch["Y"].shape == (16, 3, 8, 8)
    assert torch.all((batch["Y"] >= 0) & (batch["Y"] <= 1))

    src = source.sample_source(1, 7)
    ref = source.sample_interventional(0, 9)
    assert src["Y"].shape == (7, 3, 8, 8)
    assert ref["Y"].shape == (9, 3, 8, 8)

    p = source.propensity(torch.tensor([[0.0], [0.5], [1.0]]))
    assert p[0] < p[1] < p[2]
