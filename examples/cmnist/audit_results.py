#!/usr/bin/env python
"""Validate a completed CMNIST generator-correction result bundle."""

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
    load_result_bundle,
)

EXPECTED_METHODS = {"decfm", "ot"}
EXPECTED_STEPS = {250, 500, 1000, 2000, 5000, 10000, 15000, 20000}
REQUIRED_FILES = {
    "config.json",
    "metrics.json",
    "convergence.json",
    "run_manifest.json",
    "data_manifest.json",
    "model_manifest.json",
    "samples.pt",
    "color_values.pt",
    "color_diagnostics.json",
    "trajectories.pt",
    "trajectory_summary.json",
}


def _resolved_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise AssertionError(f"Checkpoint escapes result directory: {relative}")
    return candidate


def audit_result_directory(
    result_dir: Path,
    *,
    device: str = "cpu",
    require_full_defaults: bool = False,
) -> dict:
    result_dir = result_dir.resolve()
    missing = sorted(name for name in REQUIRED_FILES if not (result_dir / name).is_file())
    if missing:
        raise AssertionError(f"Missing result artifacts: {missing}")

    bundle = load_result_bundle(result_dir)
    config = bundle["config"]
    metrics = bundle["metrics"]
    manifest = bundle["model_manifest"]
    data_manifest = bundle["data_manifest"]

    if set(config["reported_variants"]) != EXPECTED_METHODS:
        raise AssertionError(f"Unexpected variants: {config['reported_variants']}")
    if manifest.get("path_kind") != "relative_to_result_directory":
        raise AssertionError("Checkpoint paths are not portable result-relative paths.")
    if set(manifest["models"]) != EXPECTED_METHODS:
        raise AssertionError(f"Unexpected checkpoint methods: {sorted(manifest['models'])}")
    if data_manifest.get("direct_observational_tensor_saved") is not False:
        raise AssertionError("Observational tensors should be reconstructed, not bundled.")
    if data_manifest.get("fresh_base_samples_available") is not True:
        raise AssertionError("Fresh generator bases are not declared reconstructable.")

    expected_eval_n = int(config["eval_n"])
    if int(metrics["evaluation_n_per_arm"]) != expected_eval_n:
        raise AssertionError("Final metric sample count disagrees with the run config.")
    if require_full_defaults and expected_eval_n != 5000:
        raise AssertionError(f"Expected 5,000 final samples per arm, got {expected_eval_n}.")
    if require_full_defaults and int(config["trajectory_n"]) != 512:
        raise AssertionError("Expected 512 trajectory candidates per arm.")
    if require_full_defaults and set(map(int, config["checkpoints"])) != EXPECTED_STEPS:
        raise AssertionError("The full checkpoint schedule is incomplete.")
    convergence = bundle["convergence"]
    if require_full_defaults:
        if int(convergence.get("evaluation_n_per_arm", 0)) != 512:
            raise AssertionError("Checkpoint convergence must use 512 samples per arm.")
        if not convergence.get("shared_truth_across_methods_and_steps"):
            raise AssertionError("Checkpoint convergence does not share its truth set.")
        if not convergence.get("shared_source_bases_across_methods_and_steps"):
            raise AssertionError("Checkpoint convergence does not share generator bases.")
    if not bool(config["skip_fid"]):
        for key in ("source_fid", "decfm_fid", "ot_fid"):
            if key not in metrics or not torch.isfinite(torch.tensor(metrics[key])):
                raise AssertionError(f"Missing or non-finite metric: {key}")

    checkpoint_count = 0
    checkpoint_bytes = 0
    for method, entry in manifest["models"].items():
        checkpoint_steps = {int(step) for step in entry["checkpoints"]}
        if int(entry["final_step"]) not in checkpoint_steps:
            raise AssertionError(f"{method} final checkpoint is absent from its series.")
        if require_full_defaults and checkpoint_steps != EXPECTED_STEPS:
            raise AssertionError(f"{method} checkpoint schedule is incomplete: {checkpoint_steps}")
        for step_text, relative in entry["checkpoints"].items():
            if Path(relative).is_absolute():
                raise AssertionError(f"Absolute checkpoint path in manifest: {relative}")
            checkpoint_path = _resolved_child(result_dir, relative)
            if not checkpoint_path.is_file():
                raise AssertionError(f"Missing checkpoint: {checkpoint_path}")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if payload["kind"] != "cmnist_generator_correction":
                raise AssertionError(f"Wrong checkpoint type: {checkpoint_path}")
            if payload["variant"] != method or payload["step"] != int(step_text):
                raise AssertionError(f"Checkpoint metadata mismatch: {checkpoint_path}")
            if set(payload) & {
                "nuisance_outcome",
                "nuisance_propensity",
                "plugin_reservoir",
                "optimizer",
                "cached_base_samples",
            }:
                raise AssertionError(f"Training/sample state leaked into {checkpoint_path}")
            if not payload["state_dict"]:
                raise AssertionError(f"Empty velocity state in {checkpoint_path}")
            checkpoint_count += 1
            checkpoint_bytes += checkpoint_path.stat().st_size

        final_relative = entry["checkpoints"][str(entry["final_step"])]
        sampler = load_cmnist_correction_checkpoint(
            _resolved_child(result_dir, final_relative),
            device=device,
        )
        for arm in (0, 1):
            base, corrected = sampler.sample(arm, 1, ode_steps=1, return_base=True)
            if base.shape != (1, 3, 28, 28) or corrected.shape != base.shape:
                raise AssertionError(f"Fresh {method}/arm{arm} inference has wrong shape.")
            if not torch.isfinite(corrected).all():
                raise AssertionError(f"Fresh {method}/arm{arm} inference is non-finite.")

    color_values = bundle["color_values"]
    for method in ("source", "truth", "decfm", "ot"):
        for arm in (0, 1):
            values = color_values[method][f"arm{arm}"]
            if len(values) != expected_eval_n or not torch.isfinite(values).all():
                raise AssertionError(f"Invalid color values for {method}/arm{arm}.")

    trajectory_n = int(config["trajectory_n"])
    keep = int(config["trajectory_keep"])
    ode_steps = int(config["ode_steps"])
    trajectories = bundle["trajectories"]
    for method in EXPECTED_METHODS:
        for arm in (0, 1):
            record = trajectories[method][f"arm{arm}"]
            for extreme in ("top", "bottom"):
                selected = record[extreme]
                if selected["trajectory"].shape != (
                    ode_steps + 1,
                    min(keep, trajectory_n),
                    3,
                    28,
                    28,
                ):
                    raise AssertionError(
                        f"Invalid saved trajectory shape for {method}/arm{arm}/{extreme}."
                    )
                if selected["indices"].min() < 0 or selected["indices"].max() >= trajectory_n:
                    raise AssertionError("Saved trajectory index is outside selection batch.")
            top_min = record["top"]["foreground_pixel_change"].min()
            bottom_max = record["bottom"]["foreground_pixel_change"].max()
            if trajectory_n >= 2 * keep and top_min < bottom_max:
                raise AssertionError(f"Top/bottom ranking failed for {method}/arm{arm}.")

    return {
        "result_dir": str(result_dir),
        "methods": sorted(EXPECTED_METHODS),
        "evaluation_n_per_arm": expected_eval_n,
        "checkpoint_count": checkpoint_count,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_mib": checkpoint_bytes / (1024**2),
        "fresh_inference": "passed",
        "artifact_audit": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-full-defaults", action="store_true")
    args = parser.parse_args()
    summary = audit_result_directory(
        args.result_dir,
        device=args.device,
        require_full_defaults=args.require_full_defaults,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
