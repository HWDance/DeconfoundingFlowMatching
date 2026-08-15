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

### Conda / Mamba (recommended for a fresh clone)

The repository ships an [`environment.yml`](environment.yml) that creates a reproducible Python environment and installs the package in editable mode with both demo and development dependencies:

```bash
conda env create -f environment.yml
conda activate deconfoundingfm
```

`mamba env create -f environment.yml` can be used equivalently. Run the command from the repository root because the environment installs the local checkout with `-e .[demo,dev]`.

A quick installation/GPU check is:

```bash
python -c "import deconfoundingfm, torch; print(deconfoundingfm.__version__); print('CUDA:', torch.cuda.is_available())"
pytest
```

The Conda environment is portable and does not bundle a host GPU driver. On a GPU machine, `torch.cuda.is_available()` should be `True` before launching the CMNIST run; if it is not, install the PyTorch build appropriate for that machine and then rerun `pip install -e ".[demo,dev]"`.

To refresh an existing environment after pulling changes:

```bash
conda env update -f environment.yml --prune
```

### `venv` / pip

A plain Python virtual environment remains supported:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[demo,dev]"
```

For package-only use without notebooks or development tools, use `pip install -e .` instead.

The core package depends on PyTorch, NumPy, and scikit-learn. The `demo` extra adds Matplotlib and Jupyter; `dev` adds the test/build tooling.

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

A single runnable walkthrough is in [`examples/demo.ipynb`](examples/demo.ipynb). The demo keeps the 1x64 MLP, batch size 256, and 64-draw plug-in reservoir, while using a target learning rate of `3e-4` and a 20,000-update budget. It compares DeconfoundingFM, OT-DeconfoundingFM, and a matched Gaussian-base FM baseline on a structured three-mode problem where the Gaussian base overlaps the central target mode. The notebook includes the source/target geometry, final generated distributions, **SW2 convergence at 250/500/1k/2k/5k/10k/15k/20k updates**, and learned trajectories.


### Exact optimizer-step budgets

The default vector configuration uses an exact 10,000-update target budget. To change it, set `iterations`:

```python
DeconfoundingFMConfig(iterations=10_000)
```

This runs exactly 10,000 target-flow optimizer updates regardless of dataset or minibatch size. If `iterations=None`, training falls back to the epoch budget in `epochs`. The package default is 10,000 target iterations; the executed demo overrides this to 20,000 for DeconfoundingFM, OT-DeconfoundingFM, and the Gaussian-base comparison.

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

## CMNIST backend demo

The repository also includes a heavier **ColorMNIST generator-correction demo** in `examples/cmnist/`. It bundles the exact original MNIST `t10k` UByte files and the same foreground-color DGP used by the research experiments (digits `(1,6)`, `tau=0.08`, `k=10`, black background, and the original red/blue map). No torchvision download or sklearn substitute is used.

The backend constructs a fixed labeled dataset with **10,000 grayscale shape draws × two independent `X|A` color draws = 20,000 observations**. `pi_hat(A|X)` and both conditional nuisances are estimated from those observations. DeconfoundingFM and OT-DeconfoundingFM use a nuisance and target flow based on fresh draws from the frozen biased source-generator stand-in. The Gaussian comparison instead trains its own architecture-matched nuisance from Gaussian noise and uses a Gaussian target base; neither Gaussian flow has access to the source generator.

The intended workflow is:

```bash
python examples/cmnist/run.py --device cuda --output examples/cmnist/results/default
jupyter notebook examples/cmnist/demo.ipynb
```

`run.py` performs the heavy GPU training and writes a self-contained results bundle; `demo.ipynb` only loads that bundle and visualizes it. See `examples/cmnist/RUN.md` for an unattended Codex/cluster recipe. The committed default bundle is an exact-DGP preview only, with no fabricated trained-model metrics.


For the CMNIST backend demo, the runner writes SW2, optional FID, color-distribution diagnostics, and trajectory diagnostics. The reported path energies are normalized per outcome dimension: `bar_E_v = (1/d) int ||v_t(Y_t)||^2 dt` and `bar_E_vdot = (1/d) int ||d/dt v_t(Y_t)||^2 dt`.
