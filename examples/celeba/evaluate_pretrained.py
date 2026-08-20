#!/usr/bin/env python
"""Evaluate the migrated seed-2 CelebA correction flows and build demo artifacts."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deconfoundingfm.experimental import (
    load_celeba_correction_checkpoint,
    reconstruct_celeba_data,
    validate_celeba_checkpoint_indices,
)
from deconfoundingfm.experimental.celeba import select_trajectory_frames


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "examples" / "celeba" / "results" / "pretrained",
    )
    p.add_argument("--models", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-n", type=int, default=2_000)
    p.add_argument("--n-projections", type=int, default=128)
    p.add_argument("--projection-batch-size", type=int, default=64)
    p.add_argument("--sample-chunk", type=int, default=32)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--eval-seed", type=int, default=123)
    p.add_argument("--test-base-noise-std", type=float, default=0.0)
    p.add_argument("--trajectory-n", type=int, default=512)
    p.add_argument("--trajectory-keep", type=int, default=6)
    p.add_argument("--display-n", type=int, default=16)
    p.add_argument("--smoke", action="store_true")
    return p


def save_json(value, path: Path):
    path.write_text(json.dumps(value, indent=2, allow_nan=True))


def pack_images(images: torch.Tensor):
    return (
        images.detach()
        .cpu()
        .float()
        .add(1.0)
        .div(2.0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
    )


def draw_empirical_base(population, arm: int, n: int, *, noise_std: float, seed: int):
    treatment = population["A"].reshape(-1).long()
    base = population["Y"][treatment == int(arm)]
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randint(len(base), (int(n),), generator=generator)
    values = base[indices].clone()
    if float(noise_std) > 0:
        noise = torch.randn(values.shape, generator=generator, dtype=values.dtype)
        values.add_(noise, alpha=float(noise_std))
    return values, indices


@torch.no_grad()
def transform_chunked(model, arm: int, values: torch.Tensor, *, chunk: int, ode_steps: int):
    outputs = []
    for start in range(0, len(values), int(chunk)):
        stop = min(start + int(chunk), len(values))
        outputs.append(
            model.transform(int(arm), values[start:stop], ode_steps=ode_steps)
            .detach()
            .cpu()
        )
        print(f"    arm {arm}: {stop}/{len(values)}", flush=True)
    return torch.cat(outputs)


def sliced_wasserstein_legacy(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_projections: int,
    projection_batch_size: int,
    seed: int,
    device: str,
):
    """Legacy SW2 directions are generated on CPU and moved to the metric device."""
    n = min(len(x), len(y))
    x = x[:n].float().reshape(n, -1).to(device)
    y = y[:n].float().reshape(n, -1).to(device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = []
    for start in range(0, int(n_projections), int(projection_batch_size)):
        count = min(int(projection_batch_size), int(n_projections) - start)
        theta = torch.randn(count, x.shape[1], generator=generator, dtype=x.dtype)
        theta = theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-12)
        theta = theta.to(device)
        projected_x = (x @ theta.T).sort(dim=0).values
        projected_y = (y @ theta.T).sort(dim=0).values
        values.append((projected_x - projected_y).square().mean(dim=0).cpu())
    return float(math.sqrt(max(torch.cat(values).mean().item(), 0.0)))


def image_mean_rmse(x: torch.Tensor, y: torch.Tensor):
    return float((x.float().mean(0) - y.float().mean(0)).square().mean().sqrt())


def image_std_rmse(x: torch.Tensor, y: torch.Tensor):
    return float((x.float().std(0) - y.float().std(0)).square().mean().sqrt())


def method_metrics(
    samples,
    references,
    *,
    n_projections: int,
    projection_batch_size: int,
    projection_seed: int,
    device: str,
):
    sw2 = [
        sliced_wasserstein_legacy(
            samples[arm],
            references[arm],
            n_projections=n_projections,
            projection_batch_size=projection_batch_size,
            seed=projection_seed + arm,
            device=device,
        )
        for arm in (0, 1)
    ]
    return {
        "sw2": float(sum(sw2) / 2),
        "sw2_by_arm": sw2,
        "mean_rmse_by_arm": [
            image_mean_rmse(samples[arm], references[arm]) for arm in (0, 1)
        ],
        "std_rmse_by_arm": [
            image_std_rmse(samples[arm], references[arm]) for arm in (0, 1)
        ],
    }


def population_summary(population):
    treatment = population["A"].reshape(-1).long()
    covariate = population["X"].reshape(-1)
    cells = {
        f"a{arm}_x{x}": int(((treatment == arm) & (covariate == x)).sum())
        for arm in (0, 1)
        for x in (0, 1)
    }
    return {
        "n": len(treatment),
        "arm_counts": {f"arm{arm}": int((treatment == arm).sum()) for arm in (0, 1)},
        "p_blond": float(covariate.mean()),
        "p_blond_by_arm": {
            f"arm{arm}": float(covariate[treatment == arm].mean()) for arm in (0, 1)
        },
        "cells": cells,
    }


def base_validation(model, population):
    treatment = population["A"].reshape(-1).long()
    out = {}
    for arm in (0, 1):
        expected = population["Y"][treatment == arm]
        actual = model.base_sampler.base_by_arm[arm].detach().cpu()
        out[f"arm{arm}_count"] = len(actual)
        out[f"arm{arm}_exact"] = bool(torch.equal(expected, actual))
        out[f"arm{arm}_max_abs_diff"] = float((expected - actual).abs().max())
    return out


def trajectory_artifacts_for_arm(
    model,
    arm: int,
    source,
    endpoint,
    source_positions,
    *,
    args,
):
    # Keep the ranking logic separate so trajectory() receives the correct arm.
    count = min(int(args.trajectory_n), len(source))
    keep = min(int(args.trajectory_keep), count)
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    scores = (endpoint[:count] - source[:count]).float().square().flatten(1).mean(1).sqrt()
    selections = {
        "top": torch.topk(scores, keep).indices,
        "bottom": torch.topk(scores, keep, largest=False).indices,
    }
    saved = {}
    summary = {
        "selection_metric": "whole-image RMS pixel displacement in normalized [-1,1] space",
        "candidate_n": count,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "quantiles": {
            str(q): float(torch.quantile(scores, q))
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
        "times": list(fractions),
    }
    for label, indices in selections.items():
        trajectory, _ = model.trajectory(
            arm,
            y0=source[indices],
            ode_steps=args.ode_steps,
        )
        saved[label] = {
            "candidate_indices": indices.cpu(),
            "empirical_arm_positions": source_positions[indices].cpu(),
            "scores": scores[indices].cpu(),
            "trajectory": pack_images(select_trajectory_frames(trajectory, fractions)),
        }
        summary[f"{label}_scores"] = [float(value) for value in scores[indices]]
    return saved, summary


def main():
    args = build_parser().parse_args()
    args.output = args.output.resolve()
    args.models = (args.models or (args.output / "models")).resolve()
    if args.data_root is None:
        import os

        root = os.environ.get("CELEBA_ROOT")
        args.data_root = None if root is None else Path(root)
    if args.data_root is None:
        raise ValueError("Pass --data-root or set CELEBA_ROOT.")
    args.data_root = args.data_root.expanduser().resolve()
    if args.smoke:
        args.eval_n = min(args.eval_n, 8)
        args.n_projections = min(args.n_projections, 8)
        args.sample_chunk = min(args.sample_chunk, 4)
        args.ode_steps = min(args.ode_steps, 2)
        args.trajectory_n = min(args.trajectory_n, 8)
        args.trajectory_keep = min(args.trajectory_keep, 2)
        args.display_n = min(args.display_n, 4)
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = {
        variant: args.models / variant / "epoch_500.pt" for variant in ("decfm", "ot")
    }
    payloads = {
        variant: torch.load(path, map_location="cpu", weights_only=True)
        for variant, path in checkpoint_paths.items()
    }
    for payload in payloads.values():
        validate_celeba_checkpoint_indices(payload)
    for key in ("train_indices", "ref_indices_a0", "ref_indices_a1"):
        if not torch.equal(
            payloads["decfm"]["data_indices"][key], payloads["ot"]["data_indices"][key]
        ):
            raise RuntimeError(f"Selected checkpoints disagree on {key}.")

    print("Reconstructing exact seed-2 observational and reference data...", flush=True)
    population = reconstruct_celeba_data(
        payloads["decfm"]["dgp_config"],
        root=args.data_root,
        expected_indices=payloads["decfm"]["data_indices"],
        device="cpu",
    )
    references = {arm: population[f"Y{arm}_ref"][: args.eval_n] for arm in (0, 1)}
    training_noise_std = float(payloads["decfm"]["target_config"]["base_noise_std"])
    if float(payloads["ot"]["target_config"]["base_noise_std"]) != training_noise_std:
        raise RuntimeError("Selected checkpoints disagree on training base noise.")
    test_noise_std = float(args.test_base_noise_std)
    source = {}
    source_positions = {}
    for arm in (0, 1):
        source[arm], source_positions[arm] = draw_empirical_base(
            population,
            arm,
            args.eval_n,
            noise_std=test_noise_std,
            seed=args.eval_seed + 1000 + arm,
        )

    generated = {}
    trajectory_tensors = {}
    trajectory_summary = {
        "selection_batch_n_per_arm": min(args.trajectory_n, args.eval_n),
        "selection_keep_per_extreme": min(args.trajectory_keep, args.trajectory_n),
        "selection_metric": "whole-image RMS pixel displacement in normalized [-1,1] space",
        "shared_source_batch_across_methods": True,
    }
    base_checks = {}
    for variant in ("decfm", "ot"):
        print(f"Loading and evaluating {variant}...", flush=True)
        model = load_celeba_correction_checkpoint(
            checkpoint_paths[variant],
            device=args.device,
            population=population,
        )
        base_checks[variant] = base_validation(model, population)
        generated[variant] = {}
        trajectory_tensors[variant] = {}
        trajectory_summary[variant] = {}
        for arm in (0, 1):
            generated[variant][arm] = transform_chunked(
                model,
                arm,
                source[arm],
                chunk=args.sample_chunk,
                ode_steps=args.ode_steps,
            )
            saved, summary = trajectory_artifacts_for_arm(
                model,
                arm,
                source[arm],
                generated[variant][arm],
                source_positions[arm],
                args=args,
            )
            trajectory_tensors[variant][f"arm{arm}"] = saved
            trajectory_summary[variant][f"arm{arm}"] = summary
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Computing shared-projection SW2 and moment diagnostics...", flush=True)
    metric_device = args.device
    metrics = {
        "source": method_metrics(
            source,
            references,
            n_projections=args.n_projections,
            projection_batch_size=args.projection_batch_size,
            projection_seed=args.eval_seed + 10,
            device=metric_device,
        )
    }
    for variant in ("decfm", "ot"):
        metrics[variant] = method_metrics(
            generated[variant],
            references,
            n_projections=args.n_projections,
            projection_batch_size=args.projection_batch_size,
            projection_seed=args.eval_seed,
            device=metric_device,
        )
        metrics[variant]["legacy_saved_evaluation"] = payloads[variant]["legacy_evaluation"]
        metrics[variant]["empirical_base_validation"] = base_checks[variant]
    metrics["evaluation"] = {
        "n_per_arm": args.eval_n,
        "n_projections": args.n_projections,
        "projection_batch_size": args.projection_batch_size,
        "projection_seed": args.eval_seed,
        "sample_seed": args.eval_seed,
        "ode_steps": args.ode_steps,
        "training_base_noise_std": training_noise_std,
        "test_base_noise_std": test_noise_std,
        "shared_source_bases_across_methods": True,
        "shared_projection_directions_across_methods": True,
        "reference_indices_are_exact_legacy_seed2": True,
        "metric_note": (
            "SW2 on flattened normalized RGB images. Legacy-compatible projection "
            "directions are generated on CPU. Fresh new-API values use deterministic "
            "shared clean empirical-base draws with no test-time noise; legacy values are retained as provenance."
        ),
    }

    display_n = min(args.display_n, args.eval_n)
    sample_tensors = {
        f"{name}_a{arm}": pack_images(values[arm][:display_n])
        for name, values in {
            "source": source,
            "true": references,
            "decfm": generated["decfm"],
            "ot": generated["ot"],
        }.items()
        for arm in (0, 1)
    }
    torch.save(sample_tensors, args.output / "samples.pt")
    torch.save(trajectory_tensors, args.output / "trajectories.pt")

    model_manifest = {
        "format_version": 1,
        "path_kind": "relative_to_result_directory",
        "artifact_policy": "selected_legacy_seed",
        "selection_split": "legacy_2000_per_arm_evaluation",
        "selection_metric": "sw2_mean",
        "models": {
            variant: {
                "variant": variant,
                "seed": 2,
                "epoch": 500,
                "step_unit": "epoch",
                "selection_metric": "sw2_mean",
                "legacy_selection_value": payloads[variant]["legacy_evaluation"]["sw2_mean"],
                "checkpoints": {"500": f"models/{variant}/epoch_500.pt"},
            }
            for variant in ("decfm", "ot")
        },
    }
    dgp_config = payloads["decfm"]["dgp_config"]
    pop_summary = population_summary(population)
    data_manifest = {
        "reconstruction": (
            "reconstruct_celeba_data(dgp_config, root=CELEBA_ROOT, "
            "expected_indices=checkpoint['data_indices'])"
        ),
        "observational_design": "custom_disjoint_attribute_cells",
        "observational_seed": 2,
        "dgp_config": dgp_config,
        "observational_n": dgp_config["n_obs"],
        "reference_n_per_arm": dgp_config["n_ref"],
        "pool_n": population["pool_size"],
        "observational_summary": pop_summary,
        "index_sha256": payloads["decfm"]["data_index_sha256"],
        "metadata_sha256": population["metadata_sha256"],
        "direct_observational_tensor_saved": False,
        "display_artifact_encoding": "uint8_rgb_after_clipping_and_mapping_from_-1_1",
        "base_source": "fixed_observational_Y_stratified_by_A",
        "base_noise_std": test_noise_std,
        "training_base_noise_std": training_noise_std,
        "test_base_noise_std": test_noise_std,
        "empirical_base_reconstructable": True,
        "data_root_env_var": "CELEBA_ROOT",
    }
    config = {
        "study_mode": "migrated_pretrained_offline_empirical",
        "device": args.device,
        "seed": 2,
        "eval_seed": args.eval_seed,
        "eval_n": args.eval_n,
        "sw2_projections": args.n_projections,
        "sample_chunk": args.sample_chunk,
        "ode_steps": args.ode_steps,
        "trajectory_n": args.trajectory_n,
        "trajectory_keep": args.trajectory_keep,
        "base_noise_std": test_noise_std,
        "training_base_noise_std": training_noise_std,
        "test_base_noise_std": test_noise_std,
        "observational_n": dgp_config["n_obs"],
        "reference_n_per_arm": dgp_config["n_ref"],
        "reported_variants": ["decfm", "ot"],
        "target_configs": {
            variant: payloads[variant]["target_config"] for variant in ("decfm", "ot")
        },
        "plugin_configs": {
            variant: payloads[variant]["plugin_config"] for variant in ("decfm", "ot")
        },
        "propensity": payloads["decfm"]["propensity"],
        "smoke": args.smoke,
    }
    convergence = {
        "epochs": [500],
        "decfm": [payloads["decfm"]["legacy_evaluation"]["sw2_mean"]],
        "ot": [payloads["ot"]["legacy_evaluation"]["sw2_mean"]],
        "role": "legacy_selected_seed2_checkpoint",
        "evaluation_n_per_arm": 2000,
        "sw2_projections": 128,
        "note": "Only the selected final legacy checkpoint was retained; this is not a curve.",
    }
    run_manifest = {
        "completed": True,
        "source": "migrated_seed2_legacy_pretrained",
        "methods": ["decfm", "ot"],
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    save_json(metrics, args.output / "metrics.json")
    save_json(model_manifest, args.output / "model_manifest.json")
    save_json(data_manifest, args.output / "data_manifest.json")
    save_json(config, args.output / "config.json")
    save_json(convergence, args.output / "convergence.json")
    save_json(trajectory_summary, args.output / "trajectory_summary.json")
    save_json(run_manifest, args.output / "run_manifest.json")
    print(
        "Done:",
        {
            key: round(metrics[key]["sw2"], 6) for key in ("source", "decfm", "ot")
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
