import torch

from deconfoundingfm.couplings import (
    entropic_coupling_plan,
    ot_conditional_probabilities,
    sample_from_ot_conditional,
    sinkhorn_target_dual,
)


def test_entropic_plan_has_uniform_marginals():
    torch.manual_seed(0)
    x = torch.randn(7, 2)
    y = torch.randn(5, 2)
    plan = entropic_coupling_plan(x, y, eps=0.7, n_iters=200)
    assert plan.shape == (7, 5)
    assert torch.allclose(plan.sum(1), torch.full((7,), 1 / 7), atol=2e-4)
    assert torch.allclose(plan.sum(0), torch.full((5,), 1 / 5), atol=2e-4)
    assert torch.allclose(plan.sum(), torch.tensor(1.0), atol=2e-4)


def test_ot_conditional_is_finite_and_normalized():
    torch.manual_seed(1)
    src = torch.randn(6, 3)
    tgt = torch.randn(4, 3)
    dual = sinkhorn_target_dual(src, tgt, eps=0.5, n_iters=50)
    probs = ot_conditional_probabilities(src[:3], tgt, dual, eps=0.5)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(1), torch.ones(3))
    draw = sample_from_ot_conditional(src[:3], tgt, dual, eps=0.5)
    assert draw.shape == (3, 3)
