# CMNIST online/offline study

This study compares independent-coupling DeconfoundingFM (`decfm`) and OT-DeconfoundingFM (`ot`) on the CMNIST dataset from the paper. It uses the packaged original MNIST `t10k` IDX files and the original red/blue ColorMNIST construction, so the CMNIST data itself requires no download.

**Precomputed online and offline result bundles are included with the repository.** After cloning, the demo notebooks can be opened immediately to inspect saved metrics, samples, color diagnostics, convergence curves, and trajectories; **the expensive backend training runs do not need to be rerun** unless you want to reproduce or refresh the experiments from scratch. The backend commands below are provided for that purpose.

Create or refresh the repository environment from the repository root:

```bash
conda env create -f environment.yml
conda activate deconfoundingfm

# For an existing environment:
conda env update -f environment.yml --prune
```

## Online generator-base study

The online study assumes access to a "pretrained" generator \(\widehat P(Y\mid A=a)\) for $P(Y|A=a)$ (here set as the true conditional distribution itself), which will be used as the observational base sampler, as well as a fixed observational dataset \((X_i,A_i,Y_i)_{i=1}^{20000}\) with color labels \(X_i\). Both the conditional-outcome nuisance flow and the final correction flows therefore receive **fresh draws from \(\widehat P(Y\mid A=a)\)** whenever base samples are required; they are not restricted to the finite outcomes in the labeled observational dataset. Independent Gaussian pixel noise with standard deviation `0.1` is added to each draw.

To reproduce the backend run from scratch:

```bash
python examples/cmnist/run.py \
    --device cuda \
    --output examples/cmnist/results/default
```

The repository already includes `examples/cmnist/results/default`, so the saved online results can be inspected without rerunning the backend:

```bash
jupyter notebook examples/demo_cmnist_online.ipynb
```

## Offline empirical-base study

The offline study removes access to fresh generator samples. Instead, the nuisance and both correction flows use the **fixed observed outcomes in the 20,000-row labeled dataset**, stratified by treatment arm, as their empirical base distributions and sample those outcomes with replacement. Independent Gaussian pixel noise with standard deviation `0.1` is added to each empirical base draw.

Thus the central distinction is

\[
\begin{aligned}
\textbf{online:}\quad&
Y_0 \sim G_a \approx P(Y\mid A=a)
\quad\text{freshly at each use},\\
\textbf{offline:}\quad&
Y_0 \sim \widehat P_{\mathrm{emp}}(Y\mid A=a)
\quad\text{from the fixed observed sample}.
\end{aligned}
\]

The online experiment therefore mimics how deconfounding flows can be applied **post hoc to correct a pretrained observational generator**, whereas the offline experiment corresponds more closely to the original empirical-base formulation in which only the finite observational dataset is available.

To reproduce the offline backend from scratch:

```bash
python examples/cmnist/run_offline.py \
    --device cuda \
    --output examples/cmnist/results/offline
```

The repository already includes `examples/cmnist/results/offline`, so the saved offline results can likewise be inspected without rerunning training:

```bash
jupyter notebook examples/demo_cmnist_offline.ipynb
```

## Data and fitted components

The binary design uses digits `(1, 6)`,

\[
X\sim\operatorname{Uniform}(0,1),
\qquad
P(A=1\mid X=x)
=
\operatorname{sigmoid}\!\left(5(x-0.5)\right),
\]

with `tau=0.08`, `k=10`, and a black background.

The fixed labeled observational population contains **20,000 independently generated rows**. For each row, \(X\) and then \(A\mid X\) are sampled, a digit shape is sampled with replacement from the corresponding arm-specific `t10k` pool, and the shape is colored according to that row's \(X\).

The propensity is estimated from this population using a cross-validated random forest. A conditional flow estimates \(P(Y\mid X,A)\) from the same labeled observations. In the **online study**, its FM base is a fresh draw from the biased observational generator \(P(Y\mid A=a)\), with independent Gaussian pixel noise of standard deviation `0.1`; the two target correction flows use the same fresh-generator base mechanism. The labeled 20,000-row dataset therefore supplies the **causal information**, while the pretrained-generator stand-in supplies arbitrarily many observational base samples. In the **offline study**, the conditional nuisance and target flows instead draw their bases from the observed \(Y_i\)'s within each treatment arm, again with Gaussian base noise of standard deviation `0.1`. No fresh observational images are available beyond resampling the fixed dataset.

To reconstruct the observational dataset, the complete DGP configuration and random seed are stored in `data_manifest.json`. In the online study, the packaged digit data and generator recipe additionally allow a loaded correction checkpoint to generate arbitrarily many fresh observational base images.

## Training and saved outputs

The default run uses batch size 128, **100,000 nuisance updates**, and **200,000 target updates per correction method**. For computational efficiency during training the deconfounding flow, a reservoir of samples from $\hat P(Y|A,X)$ are generated up-front which are used to Monte Carlo estimate the deconfounding flow-matching gradietns. The reservoir which is refreshed after every 10,000 completed target updates through 190,000 updates.

Validation snapshots are evaluated at 1k, 2.5k, 5k, 10k, 20k, 50k, 75k, 100k, 125k, 150k, 175k, and 200k updates. Each method retains only its validation-selected best checkpoint and its final 200k checkpoint, with a single file retained when these coincide. Best and final states are then compared on a separate shared **5,000-sample test draw per treatment arm**.

The result directory contains:

- `metrics.json`: best-versus-final per-arm and averaged SW2 values on the separate 5,000-per-arm test set.
- `convergence.json`: checkpoint SW2 on one deterministic 512-sample truth/base batch per arm, using shared projection directions and base randomness across comparisons.
- `color_values.pt` and `color_diagnostics.json`: recovered per-image foreground color \(R/(R+B)\) and comparisons between \(P_X(0)\), \(P_X(1)\), and the uniform interventional target.
- `trajectories.pt` and `trajectory_summary.json`: saved correction paths and trajectory statistics, including the dimension-normalized path energies \(\bar E_v\) and \(\bar E_{\dot v}\).
- `samples.pt`: compact 8-bit plotting grids only.
- `model_manifest.json`: validation-selected best and final steps with portable result-relative checkpoint paths.
- `data_manifest.json`: deterministic observational-data reconstruction metadata.
- `config.json` and `run_manifest.json`: exact run and environment metadata.

The saved result bundles included in the repository are sufficient for the main demo visualizations and comparisons. The portable correction checkpoints used for **fresh post-training inference** are produced by the backend runs and are not included in the lightweight repository bundle; reproducing fresh checkpoint inference therefore requires rerunning the corresponding backend (or otherwise supplying those checkpoint files). These checkpoints contain correction-flow weights and reconstruction metadata while omitting nuisance models, propensity fits, optimizer state, plug-in reservoirs, and cached base samples.

The demo notebooks are therefore **lightweight post-run viewers**. They compare validation-selected best and final metrics, source/target/corrected image samples, checkpoint convergence, paper-style logit color densities, flow-change extremes, trajectory visualizations, and path diagnostics. They do not retrain the nuisance or correction models.
