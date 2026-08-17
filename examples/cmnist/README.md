# CMNIST generator-correction study

This study reruns the two generator-correction methods used here: independent-coupling DeconfoundingFM (`decfm`) and OT-DeconfoundingFM (`ot`). It uses the packaged original MNIST `t10k` IDX files and the original red/blue ColorMNIST construction, so the CMNIST data itself requires no download.

Create or refresh the repository environment from the repository root:

```bash
conda env create -f environment.yml
conda activate deconfoundingfm
# For an existing environment:
conda env update -f environment.yml --prune
```

Run the backend and then open or execute the notebook:

```bash
python examples/cmnist/run.py --device cuda --output examples/cmnist/results/default
python -m nbconvert --to notebook --execute --inplace examples/cmnist/demo.ipynb
```

A separate offline empirical-base variant is available without modifying the
online result bundle:

```bash
python examples/cmnist/run_offline.py --device cuda --output examples/cmnist/results/offline
python -m nbconvert --to notebook --execute --inplace examples/cmnist/demo_offline.ipynb
```

In this variant, the nuisance and both target flows use the same fixed
observational outcomes, stratified by arm, as their empirical bases. Sampling
is with replacement; the exact 20,000-example population is reconstructed from
checkpoint metadata rather than embedded in each checkpoint. Independent
Gaussian pixel noise with standard deviation `0.1` is added to every base draw.


## Data and fitted components

The binary design uses digits `(1, 6)`, `X ~ Uniform(0,1)`, and
`P(A=1|X=x) = sigmoid(5(x-0.5))`, with `tau=0.08`, `k=10`, and a black background.

The fixed labeled population consists of 20,000 independently generated
observational rows. For every row, `X` and then `A|X` are sampled, a digit shape
is sampled with replacement from the arm-specific t10k pool, and the shape is
colored by that row's `X`. There is no paired two-colors-per-shape construction.
The propensity is estimated by a cross-validated random forest. A
generator-base conditional flow estimates `P(Y|X,A)` from those observations,
drawing a fresh biased-generator base plus independent Gaussian pixel noise
with standard deviation `0.1` on every nuisance update. Both online target
methods use the same noised source-generator recipe.

The observational population is not duplicated in the result bundle. Its complete DGP configuration and seed are saved in `data_manifest.json`, which deterministically reconstructs it. The packaged digit data and generator recipe also let any loaded correction checkpoint draw arbitrarily many fresh base images.

## Saved outputs

The default run uses batch size 128, 100,000 nuisance updates, and 100,000
target updates per method. The target plug-in reservoir is built before
training and refreshed after every 10,000 completed target updates (through
90,000). Target velocity checkpoints are saved at 1k, 2.5k, 5k, 10k, 20k,
50k, 75k, and 100k updates under `models/{decfm,ot}/`. Each checkpoint contains
only correction-flow weights and reconstruction metadata. It deliberately
omits nuisance models, propensity fits, optimizer state, plug-in reservoirs,
and cached base samples.

Full local runs retain every scheduled checkpoint. To keep the repository
pull lightweight, the committed demo bundles retain only the final Independent
and OT checkpoints; the complete checkpoint convergence is preserved in
`convergence.json`.

The result directory contains:

- `metrics.json`: final per-arm and averaged SW2 values, using 5,000 fresh samples per arm.
- `convergence.json`: checkpoint SW2 on one deterministic 512-sample truth/base batch per arm, including shared Gaussian base noise, across every method and checkpoint.
- `color_values.pt` and `color_diagnostics.json`: one per-image foreground color value, `R/(R+B)`, and arm/uniform comparisons.
- `trajectories.pt` and `trajectory_summary.json`: full trajectories for the top and bottom eight foreground-changing examples selected from one shared batch of 512 source images per arm. Display tensors use compact 8-bit RGB storage and are restored to floating point by the bundle loader.
- `samples.pt`: compact 8-bit plotting grids only.
- `model_manifest.json`: portable result-relative paths for the checkpoints included in the bundle.
- `data_manifest.json`: deterministic observational-data reconstruction metadata.
- `config.json` and `run_manifest.json`: exact run and environment metadata.

SW2 uses the same 256 projection directions for source, independent, and OT comparisons.

The notebook plots the final metrics, sample grids, checkpoint convergence, learned color densities, flow-change extremes, and selected trajectories. It also reloads a saved checkpoint and draws genuinely fresh endpoints and trajectories without loading any training-time sample store.
