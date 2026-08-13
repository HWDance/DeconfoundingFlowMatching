# DeconfoundingFM

**DeconfoundingFM** learns a flow from an observational treatment-conditional outcome distribution
\(P(Y\mid A=a)\) to the counterfactual/interventional distribution \(P(Y(a))\), using flow matching
and a doubly robust estimating objective.

This repository is the **applied implementation** of:

> Hugh Dance, Johnny Xi, Peter Orbanz, and Benjamin Bloem-Reddy.  
> *Debiased Counterfactual Generation via Flow Matching from Observations*.  
> arXiv:2605.07665, 2026.

The paper-reproduction repository can remain broad and experiment-heavy. This repository deliberately
contains only the reusable method, small examples, tests, and packaging needed to apply it to new data.

## What is consolidated here?

The research code evolved several parallel target-flow files for independent coupling, OT coupling,
vector outcomes, and image outcomes. The applied package now uses **one canonical target
implementation**:

- `coupling="independent"` gives the standard doubly robust flow-matching estimator;
- `coupling="eot"` activates the minibatch entropic-OT conditional;
- vector outcomes use an MLP velocity;
- image outcomes use a U-Net velocity.

Likewise, there is **one conditional outcome nuisance implementation** for \(P(Y\mid X,A)\): an MLP
for vector outcomes and a covariate-conditioned U-Net for images.

See [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md) for the exact mapping from the
research source and the verification performed during the refactor.

## Status and scope

This is an **alpha research release**. The high-level API currently supports:

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

For development:

```bash
pip install -e ".[dev]"
pytest
```

The core package depends only on PyTorch, NumPy, and scikit-learn. Plotting dependencies are optional.

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

A fast runnable version is in [`examples/quickstart.py`](examples/quickstart.py).

## Image outcomes

For image tensors of shape `(N, C, H, W)`, DeconfoundingFM automatically switches to the U-Net
implementation:

```python
model = DeconfoundingFM(
    DeconfoundingFMConfig(
        architecture="auto",
        coupling="eot",
        unet_channels=32,
        nuisance_unet_channels=32,
        device="cuda",
    )
)
model.fit(X, A, images)
images_a1 = model.sample(a=1, n=64)
```

The default nuisance for images is a U-Net conditioned on both `X` and `A`; the target U-Net is
conditioned only on the treatment arm. See [`examples/image_api.py`](examples/image_api.py) for a
small API demonstration.

## Independent versus EOT coupling

The default

```python
DeconfoundingFMConfig(coupling="independent")
```

uses the estimator for which the paper develops the clean doubly robust/efficiency theory.

For higher-dimensional outcomes, one can use

```python
DeconfoundingFMConfig(coupling="eot")
```

to construct minibatch entropic-OT conditionals between plug-in counterfactual draws and the
observational base. This often gives lower-energy paths, but the estimated coupling introduces an
additional source of statistical error that is not first-order corrected by the fixed-coupling EIF.
Accordingly, EOT is exposed as an **experimental** option.

The refactor preserves the research code's adaptive EOT regularization convention. A discrepancy
between that convention and one sentence in the paper implementation appendix is documented in
[`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md).

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

and return `(N, ...)` when one draw is requested, or `(N, M, ...)` for `M` draws. See
[`examples/custom_nuisances.py`](examples/custom_nuisances.py).

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
│   ├── quickstart.py
│   ├── demo.ipynb
│   ├── synthetic_1d.py
│   ├── custom_nuisances.py
│   └── image_api.py
├── tests/
└── .github/workflows/
```

There is no second `doflow`/`backends` package: the selected research implementation has been
consolidated into the actual DeconfoundingFM package.

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

For the standard causal interpretation of `P(Y(a))`, the usual identifying conditions are required:
consistency, conditional exchangeability given `X`, and positivity/overlap. In generative
rebalancing applications without that causal interpretation, the same target can instead be read as
replacing the empirical `(A,X)` regime by `A=a` and the marginal distribution of `X`, while retaining
the conditional outcome law.

## Tests

The test suite exercises the actual configuration matrix rather than historical module names:

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
