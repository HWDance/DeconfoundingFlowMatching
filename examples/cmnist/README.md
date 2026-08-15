# CMNIST generator-correction demo

This example uses the **same ColorMNIST generator as the paper experiments**. The exact original MNIST `t10k` UByte files are bundled in the Python package, so no torchvision download or sklearn fallback is used.

Create/activate the repository environment from the repository root first:

```bash
conda env create -f environment.yml
conda activate deconfoundingfm
```

The heavy experiment and the notebook are deliberately separated:

```bash
python examples/cmnist/run.py --device cuda --output examples/cmnist/results/default
jupyter notebook examples/cmnist/demo.ipynb
```

`run.py` trains and saves the results bundle. `demo.ipynb` only loads that bundle and plots it.

## Data construction

The binary causal design uses digits `(1, 6)`, `X ~ Uniform(0,1)`, and
`P(A=1|X=x) = sigmoid(5(x-0.5))` with the same clipping, recoloring map, `tau=0.08`, `k=10`, and black background as the original research code.

Because the raw t10k file contains all ten digits while the causal experiment uses only two treatment digits, the backend first draws **10,000 grayscale shapes from the original arm-specific digit pools**, then gives each selected grayscale shape **two independent draws from the correct `X|A` color distribution**. This yields the fixed 20,000 labeled observational examples used to estimate the nuisances.

The exact source generator remains separate from that fixed causal dataset: each base request draws a fresh grayscale shape and fresh `X|A=a` color, giving fresh samples from `P(Y|A=a)`.

## What is estimated versus oracle

- `pi_hat(A|X)`: estimated from the fixed 20k `(X,A)` dataset by random forest with cross-validation.
- Generator-base `P_hat(Y|X,A)`: estimated from the fixed 20k labeled observations, with its flow-matching base sampled fresh from the exact pretrained-generator stand-in on every optimizer update.
- Gaussian-base `P_hat_G(Y|X,A)`: a separate, architecture-matched nuisance trained on the same fixed observations from standard Gaussian noise. It has no access to the source generator.
- DeconfoundingFM / OT-DeconfoundingFM use the generator-base nuisance and fresh source-generator target bases. The Gaussian comparison uses only the Gaussian nuisance and a Gaussian target base.
- Exact/oracle access is used **only by the generator-based variants** and for held-out evaluation references.


The backend run now also saves: correction-only model checkpoints, optional FID metrics, CMNIST color-distribution diagnostics, and trajectory summaries including per-dimension $\bar E_v$ and $\bar E_{\dot v}$.
