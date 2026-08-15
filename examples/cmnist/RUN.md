# Run instructions (Codex / cluster)

From a fresh clone, create the checked-in environment first:

```bash
conda env create -f environment.yml
conda activate deconfoundingfm
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count())"
```

Then, from the repository root, run the smoke test followed by the full backend and refresh the viewing notebook:

```bash
python examples/cmnist/run.py --smoke --device cuda --output examples/cmnist/results/smoke
python examples/cmnist/run.py --device cuda --output examples/cmnist/results/default
jupyter nbconvert --to notebook --execute --inplace examples/cmnist/demo.ipynb
```

If the environment already exists after a pull, refresh it with `conda env update -f environment.yml --prune`.

For the full run, do not change the defaults unless explicitly requested. The intended defaults are:

- exact packaged original `t10k` UByte CMNIST source;
- digits `(1,6)`;
- `w=5`, `tau=0.08`, `k=10`, `fg_alpha=0`;
- 10,000 grayscale shape draws × 2 colors = 20,000 fixed labeled observations;
- propensity estimated by 1000-tree RF with 5-fold depth CV;
- 20,000 optimizer updates for each of two matched nuisances: generator-base and Gaussian-base;
- 20,000 target optimizer updates;
- batch 128, U-Net width 32, ODE steps 50;
- plugin reservoir 2;
- checkpoints 250, 500, 1k, 2k, 5k, 10k, 15k, 20k.

After the backend finishes, verify that `config.json`, `metrics.json`, `convergence.json`, `run_manifest.json`, and `samples.pt` all exist in the requested result directory. Then execute the notebook so its committed outputs match the saved bundle.
