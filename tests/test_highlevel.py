import torch

from deconfoundingfm import DeconfoundingFM, DeconfoundingFMConfig
from conftest import DummyOutcome, DummyPropensity, image_data, vector_data


def test_highlevel_default_vector_fit_sample_transform_and_diagnostics():
    X, A, Y = vector_data(n=32, dim_y=1)
    cfg = DeconfoundingFMConfig(
        device="cpu",
        epochs=1,
        nuisance_epochs=1,
        ode_steps=1,
        nuisance_ode_steps=1,
        hidden=4,
        layers=1,
        nuisance_hidden=4,
        nuisance_layers=1,
        batch_size=16,
        nuisance_batch_size=16,
        plugin_reservoir=1,
        plugin_batch=1,
        propensity_trees=5,
    )
    model = DeconfoundingFM(cfg).fit(X, A, Y, verbose=False)
    assert model.architecture_ == "mlp"
    assert model.sample(1, 2).shape == (2, 1)
    assert model.transform(Y[:3, 0], 1).shape == (3, 1)
    diag = model.diagnostics()
    assert diag["automatic_cross_fitting"] is False
    assert diag["arm_counts"] == {0: 16, 1: 16}
    assert 0 <= diag["propensity"]["fraction_clipped"] <= 1


def test_highlevel_image_eot_with_custom_nuisances():
    X, A, Y = image_data()
    cfg = DeconfoundingFMConfig(
        coupling="eot",
        device="cpu",
        epochs=1,
        ode_steps=1,
        batch_size=4,
        plugin_reservoir=1,
        plugin_batch=1,
        unet_channels=2,
        eot_iterations=2,
        eot_source_batch=4,
    )
    model = DeconfoundingFM(
        cfg,
        outcome_model=DummyOutcome((1, 8, 8)),
        propensity_model=DummyPropensity(),
    ).fit(X, A, Y, verbose=False)
    assert model.architecture_ == "unet"
    assert model.sample(1, 1).shape == (1, 1, 8, 8)
    assert model.transform(Y[0], 1).shape == (1, 1, 8, 8)
    assert "eot_epsilon_last" in model.diagnostics()
