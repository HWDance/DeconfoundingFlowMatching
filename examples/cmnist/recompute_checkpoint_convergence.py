#!/usr/bin/env python
"""Recompute CMNIST checkpoint SW2 from portable target-flow checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deconfoundingfm.experimental import (
    load_cmnist_correction_checkpoint,
    sliced_wasserstein_images,
)


@torch.no_grad()
def transform_chunked(sampler, arm: int, bases: torch.Tensor, chunk: int) -> torch.Tensor:
    outputs = []
    for start in range(0, len(bases), int(chunk)):
        outputs.append(
            sampler.transform(
                arm,
                bases[start : start + int(chunk)],
            )
            .detach()
            .cpu()
        )
    return torch.cat(outputs)


def mean_arm_sw2(samples, truth, *, projections: int, seed: int) -> float:
    values = [
        sliced_wasserstein_images(
            samples[arm],
            truth[arm],
            n_projections=projections,
            seed=seed + arm,
        )
        for arm in (0, 1)
    ]
    return float(sum(values) / 2)


def _checkpoint_path(result_dir: Path, relative: str) -> Path:
    path = (result_dir / relative).resolve()
    if not path.is_relative_to(result_dir.resolve()):
        raise ValueError(f"Checkpoint escapes result directory: {relative}")
    return path


def recompute(result_dir: Path, *, device: str) -> dict:
    result_dir = result_dir.resolve()
    config = json.loads((result_dir / "config.json").read_text())
    manifest = json.loads((result_dir / "model_manifest.json").read_text())
    if manifest.get("artifact_policy", "all_checkpoints") != "all_checkpoints":
        raise ValueError(
            "Full convergence recomputation requires every scheduled checkpoint; "
            "best/final-only bundles preserve the original curve in convergence.json."
        )
    methods = ("decfm", "ot")
    if set(manifest["models"]) != set(methods):
        raise ValueError("Expected exactly the independent and OT checkpoint families.")

    evaluation_n = int(config["checkpoint_eval_n"])
    projections = int(config["sw2_projections"])
    chunk = int(config["sample_chunk"])
    sample_seed = int(config["seed"]) + 7000
    projection_seed = int(config["seed"]) + 5000

    first_entry = manifest["models"]["decfm"]
    first_step = min(map(int, first_entry["checkpoints"]))
    recipe_sampler = load_cmnist_correction_checkpoint(
        _checkpoint_path(
            result_dir,
            first_entry["checkpoints"][str(first_step)],
        ),
        device=device,
    )

    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)
    truth = {}
    bases = {}
    for arm in (0, 1):
        _, _, truth_arm = recipe_sampler.source_generator.dgp.sample_interventional(
            arm,
            evaluation_n,
        )
        truth[arm] = truth_arm.detach().cpu()
        bases[arm] = recipe_sampler.sample_base(arm, evaluation_n).detach().cpu()
    del recipe_sampler

    curves = {}
    expected_steps = None
    for method in methods:
        entry = manifest["models"][method]
        steps = sorted(map(int, entry["checkpoints"]))
        if expected_steps is None:
            expected_steps = steps
        elif steps != expected_steps:
            raise ValueError("Independent and OT checkpoint schedules differ.")
        values = []
        for step in steps:
            sampler = load_cmnist_correction_checkpoint(
                _checkpoint_path(
                    result_dir,
                    entry["checkpoints"][str(step)],
                ),
                device=device,
            )
            samples = {arm: transform_chunked(sampler, arm, bases[arm], chunk) for arm in (0, 1)}
            value = mean_arm_sw2(
                samples,
                truth,
                projections=projections,
                seed=projection_seed,
            )
            values.append(value)
            print(f"{method} step {step}: SW2={value:.8f}", flush=True)
            del sampler, samples
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        curves[method] = values

    convergence = {
        "steps": expected_steps,
        **curves,
        "evaluation_n_per_arm": evaluation_n,
        "sw2_projections": projections,
        "projection_seed": projection_seed,
        "sample_seed": sample_seed,
        "shared_truth_across_methods_and_steps": True,
        "shared_source_bases_across_methods_and_steps": True,
    }
    (result_dir / "convergence.json").write_text(json.dumps(convergence, indent=2))
    return convergence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    convergence = recompute(args.result_dir, device=args.device)
    print(json.dumps(convergence, indent=2))


if __name__ == "__main__":
    main()
