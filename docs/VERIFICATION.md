# Refactor and demo verification

The applied repository was checked against the retained research implementation, with automated tests, a full executed demonstration, and packaging/static checks.

## 1. Research-source equivalence

Before deleting the parallel research variants, the generalized OT-capable target was compared against the separate independent implementation with identical weights, empirical bases, nuisance caches, propensity caches, minibatches, and RNG seeds. Vector and image/U-Net target losses agreed exactly in the independent branch. The generalized conditional-outcome nuisance was likewise checked against the older MLP-only implementation. The applied package therefore keeps only the generalized implementations.

The explicitly documented robustness changes in `IMPLEMENTATION_NOTES.md` are not claimed to be bit-for-bit identical to the old repository; they preserve the intended estimator while fixing shape/clipping issues and making the applied API safer.

## 2. Automated tests

The current suite contains **21 tests**, covering package/public imports, vector and image target flows, independent and OT coupling paths, propensity and conditional-outcome nuisances, ODE integration, exact `iterations=k` semantics, checkpoint snapshots, and the public `fit -> sample -> transform` interface.

Current result:

```text
21 passed
```

## 3. Executed public demo

The repository contains one demonstration: `examples/demo.ipynb`. Its target settings are:

```text
target MLP:           1 hidden layer, width 64
target learning rate: 3e-4
target updates:       10,000
batch size:           256
plugin reservoir:     64
plugin batch:          4
OT Sinkhorn iters:    20
OT source batch:      128
```

The propensity and conditional-outcome nuisances are fitted once and reused across all three target methods. The outcome design has three well-separated Gaussian-mixture components; the middle counterfactual mode is centered at the origin, giving the standard Gaussian base direct overlap with one target mode while leaving the other two modes distant.

The notebook was executed end-to-end on CPU with `nbconvert`. DeconfoundingFM, OT-DeconfoundingFM, and Gaussian-base FM each completed the full **10,000 optimizer updates** with no cell errors. The committed final diagnostics are:

```text
SW2 observed source       -> target: 0.432
SW2 DeconfoundingFM       -> target: 0.344
SW2 OT-DeconfoundingFM    -> target: 0.276
SW2 Gaussian-base FM      -> target: 0.589

Mean path energy, DeconfoundingFM:    6.832
Mean path energy, OT-DeconfoundingFM: 0.124
```

The final SW2 values use fixed source samples so that they agree exactly with the 10,000-update convergence checkpoint.

## 4. SW2 convergence checkpoints

The target velocity parameters are snapshotted during the same continuous optimization run. Snapshotting copies parameters to CPU and does not consume RNG state or alter optimization. Every checkpoint is evaluated on the same fixed source and target samples.

```text
updates      DeconfoundingFM   OT-DeconfoundingFM   Gaussian-base FM
   250         0.348              0.389              1.609
   500         0.306              0.339              0.963
 1,000         0.303              0.299              0.963
 2,000         0.309              0.292              0.970
 5,000         0.305              0.294              0.842
10,000         0.344              0.276              0.589
```

The convergence curve is embedded as the fourth figure in the notebook, alongside the geometry, final generated distributions, and trajectory plots.

## 5. Packaging/static checks

- full source test suite: **21 passed**;
- package source compiles successfully;
- package source contains no imports from the old `doflow` namespace;
- the public package has one target implementation and one conditional-outcome implementation;
- `examples/` contains only the executed `demo.ipynb`;
- the notebook contains four embedded figures and no error outputs;
- the demonstration environment is CPU-only, so a CUDA smoke test remains appropriate before declaring a release candidate.

## 6. Wheel installation

The package was built and installed without network dependencies, then imported successfully as version `0.2.8`.
