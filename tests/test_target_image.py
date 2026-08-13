import pytest
import torch

from deconfoundingfm.core import DeconfoundingFlow, DeconfoundingFlowConfig
from deconfoundingfm.nn import UNet
from conftest import DummyOutcome, DummyPropensity, image_data


@pytest.mark.parametrize("use_ot", [False, True])
def test_image_target_independent_and_eot(use_ot):
    X, A, Y = image_data()
    cfg = DeconfoundingFlowConfig(
        dim_y=1,
        epochs=1,
        batch_size=4,
        plugin_reservoir=1,
        plugin_batch=1,
        ode_steps=1,
        use_ot=use_ot,
        ot_iters=2,
        ot_src_batch=4,
    )
    model = DeconfoundingFlow(
        cfg,
        DummyOutcome((1, 8, 8)),
        DummyPropensity(),
        device="cpu",
        velocity=UNet(1, 1, 2, c=2),
    )
    model.fit(X, A, Y)
    assert model._Yhat0_store.shape == (len(X), 1, 1, 8, 8)
    assert model.sample(0, 2).shape == (2, 1, 8, 8)
    assert model.transform(Y[:1], 1).shape == (1, 1, 8, 8)
