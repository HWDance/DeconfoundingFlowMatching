# Refactor and demo verification

The applied repository was checked at four levels: equivalence against the research source, automated tests, exact optimizer-step runs, and notebook execution.

## 1. Research-source equivalence

Before deleting the parallel target implementations, the generalized OT-capable target was compared to the separate independent target in the original repository. With identical model weights, empirical bases, cached nuisance samples, cached propensities, minibatches, and RNG seeds:

- vector target, arm 0/1: exact equality;
- image/U-Net target, arm 0/1: exact equality.

The new canonical target was then compared with the original generalized target for vector/image outcomes with independent/EOT coupling, again obtaining exact equality away from the explicitly documented robustness fixes. The generalized conditional-outcome nuisance was likewise checked against the older MLP-only implementation and the refactored MLP/UNetX implementation.

## 2. Automated tests

The current suite contains **18 tests**. In addition to the original vector/image, nuisance, propensity, coupling, integration, and public-API tests, it now checks that:

- `iterations=k` produces exactly `k` optimizer updates regardless of the epoch setting;
- `coupling="ot"` activates the entropic-OT target path;
- the exact iteration count is surfaced in diagnostics.

Build-environment result:

```text
18 passed
```

## 3. Exact 10,000-iteration demo runs

The committed demo outputs were generated from the same deterministic observational dataset and the same fitted nuisance estimators. Each target was run separately for **exactly 10,000 optimizer updates**:

```text
DeconfoundingFM       10,000 updates
OT-DeconfoundingFM    10,000 updates
Gaussian-base FM      10,000 updates
```

The demo uses a three-component Gaussian-mixture outcome structure, moderate covariate effects, and treatment selection through `sigmoid(3.2 X)`. With target learning rate `1e-4`, the committed arm-1 evaluation gives:

```text
SW2 observed source       -> target: 0.432
SW2 DeconfoundingFM       -> target: 0.393
SW2 OT-DeconfoundingFM    -> target: 0.262
SW2 Gaussian-base FM      -> target: 0.588

Mean path energy, DeconfoundingFM:    6.623
Mean path energy, OT-DeconfoundingFM: 0.189
```

Thus the demo exhibits both intended effects: the observational base avoids reconstructing the multimodal outcome geometry from Gaussian noise, while the OT coupling yields a much lower-energy deconfounding transport.

## 4. Notebook validation

`examples/demo.ipynb` is the only example and is committed with the three figures and numerical outputs embedded for direct viewing on GitHub.

To validate the executable code path without conflating correctness with a long benchmark runtime, a temporary copy of the notebook was run end-to-end with the target and nuisance iteration budgets reduced; every cell, including the independent, OT, Gaussian, metrics, energy, and trajectory cells, executed successfully. Separately, the three target configurations used in the committed outputs were each run for the full 10,000 updates as documented above.

The notebook automatically uses CUDA when available and otherwise uses CPU. For this small CPU example it sets PyTorch to one CPU thread, which avoids excessive thread-pool overhead for the many small operations in minibatch OT.

## 5. Wheel installation

A `deconfoundingfm-0.2.2-py3-none-any.whl` wheel was built locally with build isolation disabled, installed into a clean target directory with no dependency installation, and the full 18-test suite was rerun against the installed wheel:

```text
18 passed
```

## 6. Static/package checks

- package source compiles successfully;
- package source contains no imports from the old `doflow` namespace;
- the public package has one target implementation and one conditional-outcome implementation;
- `examples/` contains only `demo.ipynb`;
- the notebook contains no error outputs.

The verification environment exposes CPU-only PyTorch, so a CUDA runtime smoke test remains appropriate before declaring a release candidate.
