# DeconfoundingFM

**DeconfoundingFM** learns a flow from the observational treatment-conditional outcome distribution
$P(Y\mid A=a)$ to the counterfactual/interventional distribution $P(Y(a))$, using flow matching
and a doubly robust estimating objective.

This repository is the **applied implementation** of:

> Hugh Dance, Johnny Xi, Peter Orbanz, and Benjamin Bloem-Reddy.  
> *Debiased Counterfactual Generation via Flow Matching from Observations*.  
> arXiv:2605.07665, 2026.

<!-- The package uses **one canonical target
implementation**:

- `coupling="independent"` gives the standard doubly robust flow-matching estimator;
- `coupling="ot"` (or the legacy alias `"eot"`) activates the minibatch entropic-OT conditional;
- vector outcomes use an MLP velocity;
- image outcomes use a U-Net velocity.

Likewise, there is **one conditional outcome nuisance implementation** for \(P(Y\mid X,A)\): an MLP
for vector outcomes and a covariate-conditioned U-Net for images. -->

See [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md) for the exact mapping from the
research source and the verification performed during the refactor.

## Status and scope

This repo is still in **beta mode**. The high-level API currently supports:

- binary treatment `A in {0, 1}`;
- vector/tabular covariates `X`;
- scalar/vector outcomes `Y` with an MLP velocity;
- image outcomes `Y` with a U-Net velocity;
- independent and minibatch entropic-OT target couplings;
- built-in or user-supplied propensity and conditional-outcome nuisance models.

The package does **not yet automatically cross-fit nuisance models**. The default high-level fit is
therefore intended as a practical point-estimation/training pipeline. If the formal semiparametric
efficiency guarantees are required, nuisance estimation must respect the sample-splitting/cross-fitting
conditions in the paper. This limitation is surfaced in `model.diagnostics()` rather than hidden.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

For the executable demo notebook:

```bash
pip install -e ".[demo]"
jupyter notebook examples/demo.ipynb
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

The core package depends only on PyTorch, NumPy, and scikit-learn. The demo extra adds Matplotlib and Jupyter.

## Quick start: vector outcome

```python
from deconfoundingfm import DeconfoundingFM, DeconfoundingFMConfig

model = DeconfoundingFM(
    DeconfoundingFMConfig(
        coupling="independent",
        device="cuda",  # or "cpu"
    )
)

model.fit(X, A, Y)

# Samples from the estimated counterfactual distribution P(Y(1))
y1 = model.sample(a=1, n=5000)

# Apply the learned deconfounding map to observed arm-1 outcomes
y1_deconfounded = model.transform(Y[A == 1], a=1)

print(model.diagnostics())
```

With `architecture="auto"` (the default), `(N,d_y)` outcomes select an MLP automatically.

For vector outcomes, the package defaults intentionally use a small **one-hidden-layer MLP of width 64**, a target learning rate of `1e-4`, `10_000` target optimizer updates, batch size `256`, and a 64-draw cached plug-in reservoir (`plugin_batch=4`). The executed public demo notebook intentionally overrides these with a target learning rate of `3e-4` and `20_000` target updates.

A single runnable walkthrough is in [`examples/demo.ipynb`](examples/demo.ipynb). The demo keeps the 1x64 MLP, 10,000-update budget, batch size 256, and 64-draw plug-in reservoir, while using a target learning rate of `3e-4`. It compares DeconfoundingFM, OT-DeconfoundingFM, and a matched Gaussian-base FM baseline on a structured three-mode problem where the Gaussian base overlaps the central target mode. The notebook includes the source/target geometry, final generated distributions, **SW2 convergence at 250/500/1k/2k/5k/10k updates**, and learned trajectories.


### Exact optimizer-step budgets

The default vector configuration uses an exact 10,000-update target budget. To change it, set `iterations`:

```python
DeconfoundingFMConfig(iterations=10_000)
```

This runs exactly 10,000 target-flow optimizer updates regardless of dataset or minibatch size. If `iterations=None`, training falls back to the epoch budget in `epochs`. The demo uses the default 10,000 target iterations for DeconfoundingFM, OT-DeconfoundingFM, and the Gaussian-base comparison.

## Image outcomes

For image tensors of shape `(N, C, H, W)`, DeconfoundingFM automatically switches to the U-Net
implementation:

```python
model = DeconfoundingFM(
    DeconfoundingFMConfig(
        architecture="auto",
        coupling="ot",
        unet_channels=32,
        nuisance_unet_channels=32,
        device="cuda",
    )
)
model.fit(X, A, images)
images_a1 = model.sample(a=1, n=64)
```

The default nuisance for images is a U-Net conditioned on both `X` and `A`; the target U-Net is conditioned only on the treatment arm.

## Independent versus EOT coupling

The default

```python
DeconfoundingFMConfig(coupling="independent")
```

uses the estimator for which the paper develops the clean doubly robust/efficiency theory.

For higher-dimensional outcomes, one can use

```python
DeconfoundingFMConfig(coupling="ot")
```

to construct minibatch entropic-OT conditionals between plug-in counterfactual draws and the
observational base. This often gives lower-energy paths which can be easier fit.

## Default nuisance models

If no nuisance models are supplied, `fit` estimates:

1. `P(A=1 | X)` with a random forest; and
2. `P(Y | X, A)` with conditional flow matching.

The conditional outcome model itself uses the empirical `P(Y | A=a)` base by default, mirroring the
main implementation strategy in the research code.

### Supplying your own nuisances

User-supplied nuisance objects are treated as **already fitted**:

```python
model = DeconfoundingFM(
    DeconfoundingFMConfig(coupling="independent"),
    propensity_model=my_propensity,
    outcome_model=my_conditional_sampler,
)
model.fit(X, A, Y)
```

The propensity object must be callable on a torch tensor `X` and return one value
`P(A=1 | X=x_i)` per row. The outcome object must implement:

```python
sample_conditional(x, a, n_per_context=1)
```

and return `(N, ...)` when one draw is requested, or `(N, M, ...)` for `M` draws.

## Diagnostics

```python
model.diagnostics()
```

reports:

- treatment-arm counts;
- raw and clipped propensity ranges;
- the fraction of propensity predictions clipped for stability;
- inverse-propensity effective sample size in each arm;
- the selected architecture and coupling;
- the last adaptive EOT epsilon when applicable;
- whether automatic cross-fitting was used (currently `False`).

These diagnostics are not a substitute for validating overlap or nuisance-model quality, but they make
common applied failure modes visible by default.

## Repository structure

```text
deconfoundingfm/
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── docs/
│   └── IMPLEMENTATION_NOTES.md
├── src/
│   └── deconfoundingfm/
│       ├── __init__.py
│       ├── estimator.py
│       ├── flow_matching.py
│       ├── integrators.py
│       ├── couplings.py
│       ├── core/
│       │   ├── data.py
│       │   └── target.py
│       ├── nuisance/
│       │   ├── outcome.py
│       │   └── propensity.py
│       └── nn/
│           ├── velocity.py
│           ├── mlp.py
│           └── unet.py
├── examples/
│   └── demo.ipynb
├── tests/
└── .github/workflows/
```


## Lower-level API

The main building blocks remain importable for researchers who need finer control:

```python
from deconfoundingfm.core import DeconfoundingFlow, DeconfoundingFlowConfig
from deconfoundingfm.nuisance import ConditionalFlowFM, ConditionalFlowFMConfig
from deconfoundingfm.nn import MLPVelocityField, UNet, UNetX
from deconfoundingfm.couplings import entropic_coupling_plan
from deconfoundingfm.integrators import integrate_midpoint
```

## Causal interpretation

For the standard causal interpretation of $P(Y(a))$, the usual identifying conditions are required:
consistency, conditional exchangeability given $X$, and positivity/overlap. In generative
rebalancing applications without that causal interpretation, the same target can instead be read as
replacing as the distribution of outcomes with attribute $A=a$ in the regime where $A \perp X$.

## Tests

- vector target + independent coupling;
- vector target + EOT;
- image/U-Net target + independent coupling;
- image/U-Net target + EOT;
- vector and image conditional outcome nuisances;
- the `plugin_reservoir=1` edge case;
- propensity clipping and nuisance interfaces;
- ODE integration and EOT marginal checks;
- the high-level fit/sample/transform API;
- package imports and absence of legacy `doflow` imports.

GitHub Actions runs the suite on supported Python versions.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
