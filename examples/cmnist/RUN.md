# CMNIST cluster run

From the repository root, create or refresh the checked-in environment:

```bash
conda env create -f environment.yml
conda activate deconfoundingfm
# Existing environment:
conda env update -f environment.yml --prune
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

Run a GPU smoke test first:

```bash
python examples/cmnist/run.py --smoke --device cuda --output examples/cmnist/results/smoke
```

Then run the full backend and execute the notebook:

```bash
python examples/cmnist/run.py --device cuda --output examples/cmnist/results/default
python -m nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 \
  examples/cmnist/demo.ipynb
```

For the separate offline empirical-base study:

```bash
python examples/cmnist/run_offline.py --smoke --device cuda \
  --output examples/cmnist/results/offline_smoke
python examples/cmnist/run_offline.py --device cuda \
  --output examples/cmnist/results/offline
python examples/cmnist/audit_results.py examples/cmnist/results/offline \
  --device cuda --require-full-defaults \
  --expected-study-mode offline_empirical
python -m nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 \
  examples/cmnist/demo_offline.ipynb
```

The offline runner changes only the base source: the outcome nuisance and both
target variants sample with replacement from the arm-stratified observed
outcomes in the same fixed 20,000-example population. No fresh generator bases
are used. Checkpoints omit these tensors and reconstruct the population from
the saved DGP configuration and seed before recovering Y[A == arm].

The intended full defaults are:

- exact packaged original `t10k` IDX source and digits `(1,6)`;
- `w=5`, `tau=0.08`, `k=10`, and `fg_alpha=0`;
- 10,000 grayscale shapes x two colors = 20,000 fixed observations;
- 1,000-tree propensity random forest with five-fold depth cross-validation;
- one generator-base outcome nuisance trained for 20,000 optimizer updates;
- independent-coupling and OT target flows, each trained for 20,000 updates;
- batch size 128, U-Net width 32, 50 midpoint ODE steps;
- plug-in reservoir 2 and plug-in batch 1;
- target checkpoints at 250, 500, 1k, 2k, 5k, 10k, 15k, and 20k;
- final SW2/FID evaluation on 5,000 fresh samples per arm;
- checkpoint SW2 on one deterministic 512-sample truth/base batch per arm, shared across methods and steps;
- trajectory selection from a shared batch of 512 per arm, saving top/bottom eight.

A successful result directory has `config.json`, `metrics.json`, `convergence.json`, `run_manifest.json`, `data_manifest.json`, `model_manifest.json`, `samples.pt`, `color_values.pt`, `color_diagnostics.json`, `trajectories.pt`, `trajectory_summary.json`, and both checkpoint directories. Checkpoint payloads must have no nuisance, propensity, optimizer, plug-in-reservoir, or cached-base tensors.

The supplied `hpc/cmnist_full.sbatch` requests one L40S, 32 GiB host memory, and four hours. It runs the backend, audits the result bundle, and executes the notebook.
