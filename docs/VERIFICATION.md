# Refactor verification

The refactor was checked at three levels: equivalence against the research source, package-level tests,
and built-wheel installation.

## 1. Research-source equivalence

Before deleting the parallel target implementations, the generalized OT-capable target was compared to
the separate independent target in the original repository. With identical model weights, empirical
bases, cached nuisance samples, cached propensities, minibatches, and RNG seeds:

- vector target, arm 0: exact equality;
- vector target, arm 1: exact equality;
- image/U-Net target, arm 0: exact equality;
- image/U-Net target, arm 1: exact equality.

The same comparison was then run between the **new canonical target** and the original generalized
research target for the full configuration matrix:

- vector + independent: exact equality in both arms;
- vector + EOT: exact equality in both arms;
- image/U-Net + independent: exact equality in both arms;
- image/U-Net + EOT: exact equality in both arms.

The generalized conditional outcome nuisance was also checked against both its older MLP-only source
and the new canonical nuisance:

- vector/MLP flow-matching loss: exact equality;
- vector/MLP conditional samples: exact equality;
- image/UNetX flow-matching loss: exact equality against the generalized research source;
- image/UNetX conditional samples: exact equality against the generalized research source.

These checks used settings away from the explicitly fixed edge cases (e.g. propensity clipping and the
single-draw reservoir axis), so they test that the refactor itself did not alter the existing algorithms.

## 2. Automated tests

The repository test suite currently contains 17 tests and covers:

- imports/version;
- no legacy `doflow` or `backends` import paths in installed source;
- Euler and midpoint integration;
- vector and image path energy;
- flow-matching loss sanity;
- Sinkhorn uniform marginals;
- EOT conditional normalization/sampling;
- vector target with independent coupling;
- vector target with EOT;
- image/U-Net target with independent coupling;
- image/U-Net target with EOT;
- the `plugin_reservoir=1` regression for vectors and images;
- vector/MLP conditional nuisance;
- image/UNetX conditional nuisance;
- random-forest propensity input/output contracts;
- high-level built-in vector fit → sample → transform → diagnostics;
- high-level image + EOT fit → sample → transform using custom nuisances.

Result in the build environment:

```text
17 passed
```

## 3. Packaging checks

The package was built into a wheel using the local build toolchain with build isolation disabled (the
verification environment has no network access):

```text
deconfoundingfm-0.2.0-py3-none-any.whl
```

That wheel was then installed into a clean target directory, imported from the installed wheel rather
than the source tree, and the full 17-test suite was run against the installed package:

```text
17 passed
```

All 14 installable submodules were also enumerated with `pkgutil` and imported successfully.

## 4. Runnable examples

The following examples were executed successfully from the source checkout:

- `examples/quickstart.py`;
- `examples/custom_nuisances.py`;
- `examples/image_api.py`.

## 5. Static checks

- `python -Werror -m compileall` succeeds on package source and examples;
- package source contains no imports from the old `doflow` namespace;
- the public package has a single implementation tree rather than a mirrored backend package.

## Environment limitation

The verification container exposes CPU-only PyTorch (`torch.cuda.is_available() == False`), so the
CUDA runtime path could not be executed here. The code uses standard PyTorch device placement and the
same U-Net/MLP kernels as the research implementation, but a GPU smoke test should still be run on the
cluster before declaring a release candidate.
