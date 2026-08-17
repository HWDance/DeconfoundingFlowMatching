import pytest
import torch

from deconfoundingfm.core import DeconfoundingFlow, DeconfoundingFlowConfig
from deconfoundingfm.experimental import GeneratorDeconfoundingFlow
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



@pytest.mark.parametrize("generator_backed", [False, True])
def test_plugin_reservoir_refreshes_after_exact_completed_iterations(generator_backed):
    X, A, Y = vector_data(n=12, dim_y=2)
    cfg = DeconfoundingFlowConfig(
        dim_y=2,
        iterations=5,
        batch_size=12,
        plugin_reservoir=1,
        plugin_batch=1,
        ode_steps=1,
        update_plugin_reservoir=True,
        plugin_reservoir_update_frequency=2,
    )
    if generator_backed:
        class ZeroSource:
            def sample(self, a, n, device=None):
                return torch.zeros(int(n), 2, device=device)

        model = GeneratorDeconfoundingFlow(
            cfg,
            DummyOutcome((2,)),
            DummyPropensity(),
            ZeroSource(),
            device="cpu",
        )
    else:
        model = DeconfoundingFlow(
            cfg,
            DummyOutcome((2,)),
            DummyPropensity(),
            device="cpu",
        )

    calls = []
    original = model.set_plugin_samples

    def record_refresh(*args, **kwargs):
        calls.append(len(calls))
        return original(*args, **kwargs)

    model.set_plugin_samples = record_refresh
    model.fit(X, A, Y, verbose=False)
    # Initial reservoir, then refreshes after completed updates 2 and 4.
    assert len(calls) == 3
