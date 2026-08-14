# Implementation and refactor notes

This applied repository was produced by selecting the most general implementations from the research
repository and making the architecture/coupling choices orthogonal configuration options.

## Canonical implementation mapping

The target flow is consolidated from the generalized research implementation that supported both
vector and image outcomes and included a `use_ot` switch. The separate independent-coupling target
files are therefore not copied into this repository.

The conditional outcome nuisance is consolidated from the generalized implementation that supported
both an MLP (`velocity_kind="mlp"`) and a covariate-conditioned image U-Net
(`velocity_kind="unetx"`). The older MLP-only conditional-flow file is therefore not copied either.

Paper baselines, oracle methods, IPW-only variants, one-dimensional special cases, multi-arm forks,
benchmark datasets, runners, notebooks, and SLURM code remain responsibilities of the paper
reproduction repository rather than this applied package.

## Equivalence checks performed before consolidation

The following checks were run against the original research source before selecting the canonical
files:

1. **Independent target path:** the generalized OT-capable target with `use_ot=False` was compared
   numerically against the separate independent-coupling image-compatible target. With identical
   velocity weights, empirical bases, nuisance caches, propensity caches, batches, and RNG seeds,
   the per-arm training losses were **bit-for-bit identical** for both vector outcomes and U-Net image
   outcomes.
2. **Vector conditional nuisance:** the MLP branch of the generalized vector/image nuisance was
   compared against the older MLP-only nuisance. With matched velocity weights, empirical bases,
   batches, and RNG seeds, both the flow-matching loss and conditional samples were **bit-for-bit
   identical**.

These checks are why the applied package keeps the generalized implementations rather than parallel
copies for each variant.

## Explicit robustness fixes in the applied package

A small number of changes are intentional rather than purely cosmetic:

- **Reservoir size one:** the research target assumed nuisance draws always had an explicit Monte
  Carlo axis. The nuisance sampler intentionally drops that axis when `n_per_context=1`, which makes
  `plugin_reservoir=1` ambiguous (especially for images). The applied target normalizes nuisance
  output to `(N, M, ...)` before caching. This is covered by tests for vector and image outcomes.
- **Propensity clipping:** the generalized target carried an epsilon/minimum-propensity setting but did
  not apply it in the cached propensity path. The applied target clips to
  `[min_propensity, 1-min_propensity]` before inverse weighting and exposes the amount of clipping in
  diagnostics.
- **Optimization scope:** the target optimizer is constructed from the target velocity parameters only;
  nuisance models are fixed during target training.
- **CPU-stable OT primitives:** pairwise squared Euclidean costs use the algebraically equivalent
  matrix-multiplication formula rather than `torch.cdist(...).square()`, and row-wise samples from the
  discrete OT conditional use inverse-CDF categorical sampling rather than `torch.multinomial`.
  These changes preserve the same entropic-OT objective/conditional distribution while avoiding
  severe CPU slowdowns observed for long runs with sharply peaked categorical conditionals.
- **Validation:** shapes, finite values, treatment support, OT epsilon, and nuisance-output contracts are
  checked explicitly so failures occur near their source.

These changes should be viewed as correctness/robustness fixes for an applied interface, not as new
methodological variants.

## EOT epsilon convention

The current generalized research code uses, when adaptive scaling is enabled,

```text
epsilon = ot_eps_scale * mean(pairwise squared minibatch cost)
```

The paper implementation appendix describes the image setting in words as scaling by the empirical
**standard deviation** of the minibatch cost matrix. The applied package does **not silently reconcile
this discrepancy**: it preserves the behavior of the research code, and `eot_epsilon_scale=None` can
be used to request a fixed epsilon instead. This should be resolved explicitly in a future paper/code
synchronization pass.

## Cross-fitting

The paper's semiparametric efficiency statements rely on nuisance fold independence/sample splitting
(or cross-fitting). The initial applied high-level API does not automatically train fold-specific
nuisance models. It therefore reports `automatic_cross_fitting=False` in diagnostics. The underlying
training objective is the intended DeconfoundingFM objective, but users requiring the formal
sample-splitting guarantees should use a cross-fitted workflow until first-class cross-fitting is added.
