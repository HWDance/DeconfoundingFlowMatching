# CelebA pretrained correction study

This example migrates the selected seed-2 independent and OT CelebA correction
flows from the legacy doFlow implementation and evaluates them through the
current API. Both are 64-channel U-Nets trained for 500 epochs with the same
20,000-example observational population and a training base with Gaussian
noise standard deviation 0.2.

CelebA is not redistributed. Set `CELEBA_ROOT` to a standard aligned CelebA
directory containing:

- `img_align_celeba/*.jpg`
- `list_attr_celeba.txt`
- `list_eval_partition.txt`

The portable checkpoints contain exact record indices and their hashes, but no
source images, nuisance model, optimizer, plug-in reservoir, or cached samples.
Loading a checkpoint regenerates the seed-2 split, verifies all indices, loads
the original observations, and refills its arm-stratified empirical base.

To reproduce the migration when the old checkout is available:

```bash
python examples/celeba/migrate_seed2.py
```

To recover the exact identities used in the legacy paper figures and evaluate
those identities with the migrated models:

```bash
python examples/celeba/recover_paper_samples.py --device cuda
```

The recovery script verifies the cached images against the reconstructed seed-1
CelebA records and records the original candidate-pool provenance in a compact
audit manifest.

To rebuild the quantitative demo artifacts and execute the notebook:

```bash
CELEBA_ROOT=/path/to/celeba python examples/celeba/evaluate_pretrained.py --device cuda
python examples/celeba/recover_paper_samples.py --device cuda
python examples/celeba/make_notebook.py
python -m nbconvert --to notebook --execute --inplace examples/demo_celeba.ipynb
```

The evaluation uses 2,000 generated and reference images per arm, 128 SW2
projections, 50 midpoint integration steps, and shared source draws and
projection seeds across methods, with no noise added at test time. The executed notebook uses the fresh paired
metrics together with compact, provenance-audited paper-sample artifacts.
