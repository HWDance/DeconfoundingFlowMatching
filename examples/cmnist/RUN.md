# Run instructions (Codex / cluster)

From the repository root:

```bash
python -m pip install -e ".[demo]"
python examples/cmnist/run.py --smoke --device cuda --output examples/cmnist/results/smoke
python examples/cmnist/run.py --device cuda --output examples/cmnist/results/default
jupyter nbconvert --to notebook --execute --inplace examples/cmnist/demo.ipynb
```

For the full run, do not change the defaults unless explicitly requested. The intended defaults are:

- exact packaged original `t10k` UByte CMNIST source;
- digits `(1,6)`;
- `w=5`, `tau=0.08`, `k=10`, `fg_alpha=0`;
- 10,000 grayscale shape draws × 2 colors = 20,000 fixed labeled observations;
- propensity estimated by 1000-tree RF with 5-fold depth CV;
- 20,000 nuisance optimizer updates;
- 20,000 target optimizer updates;
- batch 128, U-Net width 32, ODE steps 50;
- plugin reservoir 2;
- checkpoints 250, 500, 1k, 2k, 5k, 10k, 15k, 20k.

After the backend finishes, verify that `config.json`, `metrics.json`, `convergence.json`, `run_manifest.json`, and `samples.pt` all exist in the requested result directory. Then execute the notebook so its committed outputs match the saved bundle.
