# CelebA pretrained correction study

This example migrates the selected seed-2 independent and OT CelebA correction
flows from the legacy doFlow implementation and evaluates them through the
current API. Both are 64-channel U-Nets trained for 500 epochs with the same
20,000-example observational population and an empirical base with Gaussian
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

To rebuild the lightweight demo artifacts and execute the notebook:

```bash
CELEBA_ROOT=/path/to/celeba python examples/celeba/evaluate_pretrained.py --device cuda
python -m nbconvert --to notebook --execute --inplace examples/demo_celeba.ipynb
```

The evaluation uses 2,000 generated and reference images per arm, 128 SW2
projections, 50 midpoint integration steps, and shared source draws and
projection seeds across methods. Change extremes are selected by whole-image
RMS pixel displacement from a shared 512-image candidate batch per arm.
