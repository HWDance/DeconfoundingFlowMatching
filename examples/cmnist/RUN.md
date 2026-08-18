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
are used. Independent Gaussian pixel noise with standard deviation `0.1` is
added to each empirical base draw. Checkpoints omit these tensors and
reconstruct the population from the saved DGP configuration and seed before
recovering Y[A == arm].


The intended full defaults are:

- exact packaged original `t10k` IDX source and digits `(1,6)`;
- `w=5`, `tau=0.08`, `k=10`, and `fg_alpha=0`;
- 20,000 independently generated observational rows, with a fresh arm-specific shape sampled with replacement for every row;
- 1,000-tree propensity random forest with five-fold depth cross-validation;
- one outcome nuisance trained for 100,000 optimizer updates;
- independent-coupling and OT target flows, each trained for 200,000 updates;
- batch size 128, U-Net width 32, 50 midpoint ODE steps;
- base Gaussian pixel-noise standard deviation `0.1` for nuisance and target flows;
- plug-in reservoir 2 and plug-in batch 1, rebuilt after every 10,000 completed target updates;
- validation snapshots at 1k, 2.5k, 5k, 10k, 20k, 50k, 75k, 100k, 125k, 150k, 175k, and 200k;
- validation-selected best versus final SW2 comparison on a separate shared 5,000-sample test draw per arm;
- checkpoint SW2 on one deterministic 512-sample truth/base batch per arm, shared across methods and steps;
- trajectory selection from a shared batch of 512 per arm, saving top/bottom eight.

A successful result directory has `config.json`, `metrics.json`, `convergence.json`, `run_manifest.json`, `data_manifest.json`, `model_manifest.json`, `samples.pt`, `color_values.pt`, `color_diagnostics.json`, `trajectories.pt`, `trajectory_summary.json`, and both checkpoint directories. Checkpoint payloads must have no nuisance, propensity, optimizer, plug-in-reservoir, or cached-base tensors. Only each method's validation-selected best and final 200k checkpoints are retained, while `convergence.json` preserves every scheduled validation score. Plotting grids and saved trajectory images use compact 8-bit storage.

The supplied online and offline Slurm scripts each request one L40S, 32 GiB
host memory, and 12 hours. Each trains into a staging directory, audits the
complete result bundle, atomically replaces its prior result directory only
after the audit passes, and then executes the corresponding notebook.
