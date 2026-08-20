#!/usr/bin/env python
"""Build the CelebA demo notebook from concise, reviewable source cells."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[2]
DESTINATION = REPO_ROOT / "examples" / "demo_celeba.ipynb"


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def main():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook["cells"] = [
        markdown(
            r"""
# CelebA pretrained offline empirical-base study

This notebook evaluates the migrated **seed-2** independent and OT correction
flows from the original CelebA experiment. Both target flows use the same exact
20,000-image observational population, reconstructed from saved record indices,
and an empirical base with Gaussian noise standard deviation (0.2).

Here (A=0) denotes women, (A=1) denotes men, and (X=1) denotes blond hair
within the cleaned blond/brown subset. The true interventional reference has
(P(X=1mid do(A=a))=0.3) in both arms.
"""
        ),
        code(
            r"""
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.utils import make_grid

_here = Path.cwd().resolve()
REPO_ROOT = next(p for p in [_here, *_here.parents] if (p / "src" / "deconfoundingfm").exists())
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deconfoundingfm.experimental import load_result_bundle

RESULT_DIR = Path(os.environ.get(
    "CELEBA_RESULT_DIR",
    REPO_ROOT / "examples" / "celeba" / "results" / "pretrained",
)).resolve()
bundle = load_result_bundle(RESULT_DIR)
metrics = bundle["metrics"]
config = bundle["config"]
data_manifest = bundle["data_manifest"]
model_manifest = bundle["model_manifest"]

print("Result directory:", RESULT_DIR)
print("Evaluation:", config["eval_n"], "samples/arm |", config["sw2_projections"], "SW2 projections")
print("Models: seed 2 | epoch 500 | U-Net c=64 | base noise std", config["base_noise_std"])
print("Population:", data_manifest["observational_n"], "observations from", data_manifest["pool_n"], "clean records")
"""
        ),
        markdown(
            """
## Distributional accuracy

The fresh new-API comparison is paired: independent and OT receive the same
noised empirical-base images and use the same projection directions. Lower SW2
and moment RMSE are better.
"""
        ),
        code(
            r"""
labels = {
    "source": "Uncorrected empirical base",
    "decfm": "Independent",
    "ot": "OT",
}
rows = []
for key in ("source", "decfm", "ot"):
    result = metrics[key]
    rows.append({
        "method": labels[key],
        "SW2 arm 0": result["sw2_by_arm"][0],
        "SW2 arm 1": result["sw2_by_arm"][1],
        "mean SW2": result["sw2"],
        "mean RMSE arm 0": result["mean_rmse_by_arm"][0],
        "mean RMSE arm 1": result["mean_rmse_by_arm"][1],
        "std RMSE arm 0": result["std_rmse_by_arm"][0],
        "std RMSE arm 1": result["std_rmse_by_arm"][1],
    })
pd.DataFrame(rows).style.format(precision=4)
"""
        ),
        code(
            r"""
legacy_rows = []
for key in ("decfm", "ot"):
    legacy = metrics[key]["legacy_saved_evaluation"]
    legacy_rows.append({
        "method": labels[key],
        "legacy SW2": legacy["sw2_mean"],
        "fresh paired SW2": metrics[key]["sw2"],
        "difference": metrics[key]["sw2"] - legacy["sw2_mean"],
    })
pd.DataFrame(legacy_rows).style.format(precision=5)
"""
        ),
        markdown(
            """
The fresh values need not equal the old saved values exactly because they use a
new deterministic set of generated base draws. Their close agreement checks
that checkpoint conversion, preprocessing, base refill, and integration all
preserve the trained flows.
"""
        ),
        markdown("## Compact sample comparison"),
        code(
            r"""
samples = bundle["samples"]
sample_rows = (
    ("Uncorrected source", "source"),
    ("True interventional", "true"),
    ("Independent", "decfm"),
    ("OT", "ot"),
)
fig, axes = plt.subplots(len(sample_rows), 2, figsize=(7, 10))
for row, (label, prefix) in enumerate(sample_rows):
    for arm in (0, 1):
        images = samples[f"{prefix}_a{arm}"][:8]
        grid = make_grid(images, nrow=4, padding=1).permute(1, 2, 0)
        axes[row, arm].imshow(grid)
        axes[row, arm].axis("off")
        if row == 0:
            axes[row, arm].set_title("Women ($A=0$)" if arm == 0 else "Men ($A=1$)")
    axes[row, 0].text(-0.08, 0.5, label, rotation=90, va="center", ha="right",
                      transform=axes[row, 0].transAxes, fontsize=11)
plt.tight_layout()
"""
        ),
        markdown(
            """
## Exact observational design

These bars describe the known hair labels of the reconstructed source
population and true references. They do not apply an unvalidated hair classifier
to generated images.
"""
        ),
        code(
            r"""
summary = data_manifest["observational_summary"]
observed = [summary["p_blond_by_arm"]["arm0"], summary["p_blond_by_arm"]["arm1"]]
target = [data_manifest["dgp_config"]["target_px1"]] * 2

fig, ax = plt.subplots(figsize=(6, 3.5))
x = torch.arange(2).numpy()
ax.bar(x - 0.18, observed, width=0.36, label="Observational empirical base")
ax.bar(x + 0.18, target, width=0.36, label="True interventional target")
ax.set_xticks(x, ["Women ($A=0$)", "Men ($A=1$)"])
ax.set_ylabel("Proportion blond")
ax.set_ylim(0, 0.55)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

checks = []
for method in ("decfm", "ot"):
    check = metrics[method]["empirical_base_validation"]
    checks.append({
        "method": labels[method],
        "arm 0 count": check["arm0_count"],
        "arm 1 count": check["arm1_count"],
        "arm 0 exact": check["arm0_exact"],
        "arm 1 exact": check["arm1_exact"],
        "max difference": max(check["arm0_max_abs_diff"], check["arm1_max_abs_diff"]),
    })
pd.DataFrame(checks)
"""
        ),
        markdown(
            """
## Largest and smallest learned changes

For each method and arm, 512 shared noised source images were ranked by
whole-image RMS pixel displacement. The current selection is intentionally
pixel-based; it does not claim that the ranking isolates hair edits.
"""
        ),
        code(
            r"""
trajectory_summary = bundle["trajectory_summary"]
change_rows = []
for method in ("decfm", "ot"):
    for arm in (0, 1):
        values = trajectory_summary[method][f"arm{arm}"]
        change_rows.append({
            "method": labels[method],
            "arm": arm,
            "mean RMS change": values["mean"],
            "median": values["quantiles"]["0.5"],
            "top selected mean": sum(values["top_scores"]) / len(values["top_scores"]),
            "bottom selected mean": sum(values["bottom_scores"]) / len(values["bottom_scores"]),
        })
pd.DataFrame(change_rows).style.format(precision=4)
"""
        ),
        code(
            r"""
trajectories = bundle["trajectories"]

def plot_extremes(method, n_show=3):
    rows = [(0, "top"), (0, "bottom"), (1, "top"), (1, "bottom")]
    fig, axes = plt.subplots(len(rows), 2 * n_show, figsize=(10, 7))
    for row, (arm, extreme) in enumerate(rows):
        values = trajectories[method][f"arm{arm}"][extreme]["trajectory"]
        for rank in range(n_show):
            axes[row, 2 * rank].imshow(values[0, rank].permute(1, 2, 0))
            axes[row, 2 * rank + 1].imshow(values[-1, rank].permute(1, 2, 0))
            axes[row, 2 * rank].axis("off")
            axes[row, 2 * rank + 1].axis("off")
            if row == 0:
                axes[row, 2 * rank].set_title(f"start {rank + 1}")
                axes[row, 2 * rank + 1].set_title(f"end {rank + 1}")
        axes[row, 0].text(
            -0.12, 0.5, f"A={arm}, {extreme}", rotation=90, va="center", ha="right",
            transform=axes[row, 0].transAxes,
        )
    fig.suptitle(f"{labels[method]}: pixel-change extremes", y=1.01)
    plt.tight_layout()

plot_extremes("decfm")
plot_extremes("ot")
"""
        ),
        markdown("## Representative high-change trajectories"),
        code(
            r"""
rows = [(method, arm) for method in ("decfm", "ot") for arm in (0, 1)]
times = trajectory_summary["decfm"]["arm0"]["times"]
fig, axes = plt.subplots(len(rows), len(times), figsize=(10, 8))
for row, (method, arm) in enumerate(rows):
    values = trajectories[method][f"arm{arm}"]["top"]["trajectory"][:, 0]
    for column, time in enumerate(times):
        axes[row, column].imshow(values[column].permute(1, 2, 0))
        axes[row, column].axis("off")
        if row == 0:
            axes[row, column].set_title(f"$t={time:g}$")
    axes[row, 0].text(
        -0.12, 0.5, f"{labels[method]}, A={arm}", rotation=90, va="center", ha="right",
        transform=axes[row, 0].transAxes,
    )
plt.tight_layout()
"""
        ),
        markdown(
            """
The migrated checkpoints contain only the learned target velocity, inference
configuration, exact record indices, and provenance metadata. The nuisance
flows, optimizers, plug-in reservoirs, cached empirical bases, and preview
samples remain omitted.
"""
        ),
    ]
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, DESTINATION)
    print(DESTINATION)


if __name__ == "__main__":
    main()
