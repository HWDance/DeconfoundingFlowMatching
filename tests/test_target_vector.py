import pytest
import torch

from deconfoundingfm.core import DeconfoundingFlow, DeconfoundingFlowConfig
from conftest import DummyOutcome, DummyPropensity, vector_data


@pytest.mark.parametrize("use_ot", [False, True])
def test_vector_target_independent_and_eot(use_ot):
    X, A, Y = vector_data()
    cfg = DeconfoundingFlowConfig(
        dim_y=2,
        epochs=1,
        iterations=None,
        batch_size=12,
        plugin_reservoir=1,
        plugin_batch=1,
        ode_steps=2,
        use_ot=use_ot,
        ot_iters=3,
        ot_src_batch=8,
    )
    model = DeconfoundingFlow(cfg, DummyOutcome((2,)), DummyPropensity(), device="cpu")
    model.fit(X, A, Y)
    assert model._Yhat0_store.shape == (len(X), 1, 2)  # regression test: M=1 keeps MC axis
    assert model._Yhat1_store.shape == (len(X), 1, 2)
    assert model.sample(0, 3).shape == (3, 2)
    assert model.transform(Y[:2], 1).shape == (2, 2)
    assert torch.isfinite(torch.tensor(model.training_loss_)).all()
