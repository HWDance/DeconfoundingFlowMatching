# Refactor and demo verification

The applied repository was checked against the retained research implementation, with automated tests, a full executed demonstration, and packaging/static checks.

## 1. Research-source equivalence

Before deleting the parallel research variants, the generalized OT-capable target was compared against the separate independent implementation with identical weights, empirical bases, nuisance caches, propensity caches, minibatches, and RNG seeds. Vector and image/U-Net target losses agreed exactly in the independent branch. The generalized conditional-outcome nuisance was likewise checked against the older MLP-only implementation. The applied package therefore keeps only the generalized implementations.

The explicitly documented robustness changes in `IMPLEMENTATION_NOTES.md` are not claimed to be bit-for-bit identical to the old repository; they preserve the intended estimator while fixing shape/clipping issues and making the applied API safer.

## 2. Automated tests

The current suite contains **23 tests**, covering package/public imports, vector and image target flows, independent and OT coupling paths, propensity and conditional-outcome nuisances, ODE integration, exact `iterations=k` semantics, checkpoint snapshots, and the public `fit -> sample -> transform` interface.

Current result:

```text
23 passed
```

## 3. Executed public demo

The repository contains an executed vector demonstration (`examples/demo.ipynb`) and a GPU-oriented population ColorMNIST demonstration (`examples/cmnist_population_demo.ipynb`). Its target settings are:

```text
target MLP:           1 hidden layer, width 64
target learning rate: 3e-4
target updates:       20,000
batch size:           256
plugin reservoir:     64
plugin batch:          4
OT Sinkhorn iters:    20
OT source batch:      128
```

The propensity and conditional-outcome nuisances are fitted once and reused across all three target methods. The outcome design has three well-separated Gaussian-mixture components; the middle counterfactual mode is centered at the origin, giving the standard Gaussian base direct overlap with one target mode while leaving the other two modes distant.

The notebook was executed end-to-end on CPU with `nbconvert`. DeconfoundingFM, OT-DeconfoundingFM, and Gaussian-base FM each completed the full **20,000 optimizer updates** with no cell errors. The committed final diagnostics are for the 20,000-update demo notebook:

```text
SW2 observed source       -> target: 0.432
SW2 DeconfoundingFM       -> target: 0.354
SW2 OT-DeconfoundingFM    -> target: 0.280
SW2 Gaussian-base FM      -> target: 0.478

Mean path energy, DeconfoundingFM:    8.152
Mean path energy, OT-DeconfoundingFM: 0.116
```

The final SW2 values use fixed evaluation samples and agree with the 20,000-update convergence checkpoint up to printed rounding.

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
15,000         0.336              0.277              0.511
20,000         0.354              0.280              0.478
```

The convergence curve is embedded as the fourth figure in the notebook, alongside the geometry, final generated distributions, and trajectory plots.

## 5. Packaging/static checks

- full source test suite: **23 passed**;
- package source compiles successfully;
- package source contains no imports from the old `doflow` namespace;
- the public package has one target implementation and one conditional-outcome implementation;
- `examples/` contains the executed vector `demo.ipynb` and the population `cmnist_population_demo.ipynb`;
- the notebook contains four embedded figures and no error outputs;
- the demonstration environment is CPU-only, so a CUDA smoke test remains appropriate before declaring a release candidate.

## 6. Wheel installation

The package was built and installed without network dependencies, then imported successfully as version `0.2.10`.


## Population ColorMNIST path

The population sampler and trainer have CPU smoke tests using synthetic MNIST-like digit pools. These tests exercise fresh observational sampling, exact arm-conditional source sampling, one population nuisance update, renewable plugin-reservoir construction, one image target update, and post-fit target sampling. The ColorMNIST recolouring map and oracle propensity were also compared numerically against the retained research DGP on the original MNIST `t10k` files: both agreed exactly (`max abs diff = 0`).

A smoke-mode copy of the complete `cmnist_population_demo.ipynb` was executed end-to-end against the original local IDX files with no cell errors, including independent, OT, and Gaussian target branches. The committed notebook retains the full CUDA-oriented settings rather than the smoke outputs; the release environment itself has CPU-only PyTorch, so the full 20k image run was not executed here.
