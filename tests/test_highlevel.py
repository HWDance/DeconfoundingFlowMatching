import torch

from deconfoundingfm import DeconfoundingFM, DeconfoundingFMConfig
from conftest import DummyOutcome, DummyPropensity, image_data, vector_data


def test_highlevel_default_vector_fit_sample_transform_and_diagnostics():
    X, A, Y = vector_data(n=32, dim_y=1)
    cfg = DeconfoundingFMConfig(
        device="cpu",
        epochs=1,
        iterations=None,
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
        iterations=None,
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


def test_highlevel_exact_iteration_budget_and_ot_alias():
    X, A, Y = vector_data(n=32, dim_y=1)
    cfg = DeconfoundingFMConfig(
        coupling="ot",
        device="cpu",
        iterations=3,
        epochs=99,  # ignored when iterations is supplied
        ode_steps=1,
        batch_size=16,
        plugin_reservoir=1,
        plugin_batch=1,
        eot_iterations=2,
        eot_source_batch=8,
    )
    model = DeconfoundingFM(
        cfg,
        outcome_model=DummyOutcome((1,)),
        propensity_model=DummyPropensity(),
    ).fit(X, A, Y, verbose=False)
    assert model.model_.training_steps_ == 3
    assert len(model.model_.training_loss_) == 3
    assert model.diagnostics()["training_iterations"] == 3
    assert model.diagnostics()["coupling"] == "ot"
    assert "eot_epsilon_last" in model.diagnostics()


def test_highlevel_checkpoint_snapshots_are_recorded():
    X, A, Y = vector_data(n=32, dim_y=1)
    cfg = DeconfoundingFMConfig(
        device="cpu",
        iterations=4,
        ode_steps=1,
        batch_size=16,
        plugin_reservoir=1,
        plugin_batch=1,
    )
    model = DeconfoundingFM(
        cfg,
        outcome_model=DummyOutcome((1,)),
        propensity_model=DummyPropensity(),
    ).fit(X, A, Y, verbose=False, checkpoint_steps=[1, 3, 4])

    assert sorted(model.model_.checkpoint_state_dicts_) == [1, 3, 4]
    final_state = model.velocity_.state_dict()
    for key, value in model.model_.checkpoint_state_dicts_[4].items():
        assert torch.equal(value, final_state[key].detach().cpu())
