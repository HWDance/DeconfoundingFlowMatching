import torch

from deconfoundingfm.nuisance import RandomForestConfig, RandomForestPropensityEstimator


def test_random_forest_accepts_column_treatment_and_returns_on_input_device():
    torch.manual_seed(0)
    X = torch.randn(30, 2)
    A = (torch.arange(30) % 2).float().view(-1, 1)
    model = RandomForestPropensityEstimator(
        RandomForestConfig(in_dim=2, n_estimators=5, max_depth=2, random_state=0)
    ).fit(X, A)
    p = model(X)
    assert p.shape == (30,)
    assert p.device == X.device
    assert ((0 <= p) & (p <= 1)).all()
