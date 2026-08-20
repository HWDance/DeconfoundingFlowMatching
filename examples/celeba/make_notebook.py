#!/usr/bin/env python
"""Build the compact CelebA demo notebook."""

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
# CelebA pretrained correction study

Seed-2 independent and OT correction flows trained on the same fixed 20,000-image
observational population. Training used base noise standard deviation 0.2;
evaluation uses clean empirical bases with **no test-time noise**.
"""
        ),
        code(
            r"""
from pathlib import Path
import json
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
samples = bundle["samples"]
paper_samples = torch.load(RESULT_DIR / "paper_samples.pt", map_location="cpu", weights_only=True)
paper_manifest = json.loads((RESULT_DIR / "paper_samples_manifest.json").read_text())

assert config["test_base_noise_std"] == 0.0
print("Evaluation:", config["eval_n"], "samples/arm |", config["sw2_projections"], "SW2 projections")
print("Training noise:", config["training_base_noise_std"], "| test noise:", config["test_base_noise_std"])
"""
        ),
        markdown("## Design"),
        code(
            r"""
summary = data_manifest["observational_summary"]
observed = [summary["p_blond_by_arm"]["arm0"], summary["p_blond_by_arm"]["arm1"]]
target = [data_manifest["dgp_config"]["target_px1"]] * 2

fig, ax = plt.subplots(figsize=(6, 3.5))
x = torch.arange(2).numpy()
ax.bar(x - 0.18, observed, width=0.36, label="Observational")
ax.bar(x + 0.18, target, width=0.36, label="Interventional target")
ax.set_xticks(x, ["Women ($A=0$)", "Men ($A=1$)"])
ax.set_ylabel("Proportion blond")
ax.set_ylim(0, 0.55)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

pd.DataFrame([{
    "observations": data_manifest["observational_n"],
    "reference samples / arm": data_manifest["reference_n_per_arm"],
    "training base noise": config["training_base_noise_std"],
    "test base noise": config["test_base_noise_std"],
}])
"""
        ),
        markdown("## Trained-model performance"),
        code(
            r"""
rows = []
for key, label in (("decfm", "Independent"), ("ot", "OT")):
    rows.append({
        "method": label,
        "SW2 before": metrics["source"]["sw2"],
        "SW2 after": metrics[key]["sw2"],
        "SW2 reduction": metrics["source"]["sw2"] - metrics[key]["sw2"],
        "after: women": metrics[key]["sw2_by_arm"][0],
        "after: men": metrics[key]["sw2_by_arm"][1],
        "mean RMSE: women": metrics[key]["mean_rmse_by_arm"][0],
        "mean RMSE: men": metrics[key]["mean_rmse_by_arm"][1],
    })
pd.DataFrame(rows).style.format(precision=4)
"""
        ),
        markdown("## Generated samples and true interventional samples"),
        code(
            r"""
sample_rows = (
    ("True interventional", "true"),
    ("Independent", "decfm"),
    ("OT", "ot"),
)
fig, axes = plt.subplots(len(sample_rows), 2, figsize=(7, 7.5))
for row, (label, prefix) in enumerate(sample_rows):
    for arm in (0, 1):
        grid = make_grid(samples[f"{prefix}_a{arm}"][:8], nrow=4, padding=1).permute(1, 2, 0)
        axes[row, arm].imshow(grid)
        axes[row, arm].axis("off")
        if row == 0:
            axes[row, arm].set_title("Women ($A=0$)" if arm == 0 else "Men ($A=1$)")
    axes[row, 0].text(-0.08, 0.5, label, rotation=90, va="center", ha="right",
                      transform=axes[row, 0].transAxes)
plt.tight_layout()
"""
        ),
        markdown(
            """
## Selected trajectories

Exact identities from the saved paper figures. Eligible high-change pools are
shown first; selected identities have red borders.
"""
        ),
        code(
            r"""
def plot_candidate_pool(arm, columns):
    values = paper_samples["representative"][f"arm{arm}"]["candidate_starts"]
    selected = paper_manifest["representative_trajectories"][f"arm{arm}"]["drawn_ranks_zero_based"][0]
    rows = (len(values) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(1.15 * columns, 1.35 * rows), squeeze=False)
    for rank, ax in enumerate(axes.flat):
        if rank >= len(values):
            ax.axis("off")
            continue
        ax.imshow(values[rank].permute(1, 2, 0))
        ax.set_title(f"rank {rank}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2.5 if rank == selected else 0.5)
            spine.set_edgecolor("crimson" if rank == selected else "0.75")
    fig.suptitle("Women: independent top 8" if arm == 0 else "Men: OT top 20", y=1.02)
    plt.tight_layout()

plot_candidate_pool(0, 8)
plot_candidate_pool(1, 10)
"""
        ),
        code(
            r"""
def plot_selected_trajectory(arm):
    item = paper_samples["representative"][f"arm{arm}"]
    rows = (
        ("Saved legacy", item["legacy_trajectory"]),
        ("Independent", item["decfm_trajectory"]),
        ("OT", item["ot_trajectory"]),
    )
    fig, axes = plt.subplots(3, len(paper_manifest["times"]), figsize=(10, 5.8))
    for row, (label, values) in enumerate(rows):
        for column, time in enumerate(paper_manifest["times"]):
            axes[row, column].imshow(values[column].permute(1, 2, 0))
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(f"$t={time:g}$")
        axes[row, 0].text(-0.12, 0.5, label, rotation=90, va="center", ha="right",
                          transform=axes[row, 0].transAxes)
    title = "Women ($A=0$)" if arm == 0 else "Men ($A=1$)"
    fig.suptitle(title, y=1.01)
    plt.tight_layout()

plot_selected_trajectory(0)
plot_selected_trajectory(1)
"""
        ),
        markdown("## Selected before/after gallery"),
        code(
            r"""
gallery = paper_samples["gallery"]
gallery_rows = (
    ("Start", gallery["start"]),
    ("Saved legacy OT", gallery["legacy_ot_end"]),
    ("Independent", gallery["decfm_end"]),
    ("OT", gallery["ot_end"]),
)
fig, axes = plt.subplots(4, len(gallery["start"]), figsize=(13, 6))
for row, (label, values) in enumerate(gallery_rows):
    for column, image in enumerate(values):
        axes[row, column].imshow(image.permute(1, 2, 0))
        axes[row, column].axis("off")
    axes[row, 0].text(-0.12, 0.5, label, rotation=90, va="center", ha="right",
                      transform=axes[row, 0].transAxes)
plt.tight_layout()
"""
        ),
    ]
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, DESTINATION)
    print(DESTINATION)


if __name__ == "__main__":
    main()
