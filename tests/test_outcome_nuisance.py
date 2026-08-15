import torch

from deconfoundingfm.nuisance import ConditionalFlowFM, ConditionalFlowFMConfig
from conftest import image_data, vector_data


def test_vector_conditional_nuisance():
    X, A, Y = vector_data(n=12, dim_y=2)
    cfg = ConditionalFlowFMConfig(
        dim_y=2, dim_x=2, hidden=4, layers=1, epochs=1, batch_size=6, ode_steps=1
    )
    model = ConditionalFlowFM(cfg, device="cpu").fit(X, A, Y, verbose=False)
    assert model.sample_conditional(X[:2], A[:2], 1).shape == (2, 2)
    assert model.sample_conditional(X[:2], A[:2], 2).shape == (2, 2, 2)


def test_image_conditional_nuisance():
    X, A, Y = image_data()
    cfg = ConditionalFlowFMConfig(
        dim_y=1,
        dim_x=1,
        epochs=1,
        batch_size=4,
        ode_steps=1,
        velocity_kind="unetx",
        y_is_image=True,
        y_channels=1,
        y_height=8,
        y_width=8,
        num_classes=2,
        x_dim=1,
        unet_c=2,
        film_hidden=2,
    )
    model = ConditionalFlowFM(cfg, device="cpu").fit(X, A, Y, verbose=False)
    assert model.sample_conditional(X[:2], A[:2], 1).shape == (2, 1, 8, 8)
    assert model.sample_conditional(X[:2], A[:2], 2).shape == (2, 2, 1, 8, 8)


def test_gaussian_conditional_nuisance_exact_steps_are_generator_free():
    X, A, Y = vector_data(n=12, dim_y=2)
    cfg = ConditionalFlowFMConfig(
        dim_y=2,
        dim_x=2,
        hidden=4,
        layers=1,
        batch_size=6,
        ode_steps=1,
        base_kind="gaussian",
    )
    model = ConditionalFlowFM(cfg, device="cpu").fit_iterations(
        X, A, Y, iterations=2, verbose=False
    )
    assert model.training_steps_ == 2
    assert len(model.training_loss_) == 2
    assert model._base0.numel() == model._base1.numel() == 0
    assert not hasattr(model, "source_generator")
    assert torch.isfinite(model.sample_conditional(X[:2], A[:2], 1)).all()
