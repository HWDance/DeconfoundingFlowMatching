#!/usr/bin/env python
"""Audit migrated CelebA checkpoints, reconstruction metadata, and demo artifacts."""

from __future__ import annotations

import argparse
import json
import math
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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result",
        type=Path,
        default=REPO_ROOT / "examples" / "celeba" / "results" / "pretrained",
    )
    p.add_argument("--data-root", type=Path)
    return p


def main():
    args = parser().parse_args()
    result = args.result.resolve()
    required = {
        "config.json",
        "convergence.json",
        "data_manifest.json",
        "metrics.json",
        "model_manifest.json",
        "run_manifest.json",
        "samples.pt",
        "trajectories.pt",
        "trajectory_summary.json",
    }
    missing = sorted(name for name in required if not (result / name).is_file())
    if missing:
        raise RuntimeError(f"Missing result artifacts: {missing}")

    metrics = json.loads((result / "metrics.json").read_text())
    manifest = json.loads((result / "model_manifest.json").read_text())
    payloads = {}
    for variant in ("decfm", "ot"):
        relative = manifest["models"][variant]["checkpoints"]["500"]
        path = result / relative
        if not path.is_file() or path.stat().st_size >= 40 * 1024**2:
            raise RuntimeError(f"Unexpected checkpoint size/path: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_celeba_checkpoint_indices(payload)
        if any(key.startswith("nuisance_outcome") for key in payload["state_dict"]):
            raise RuntimeError(f"{variant} checkpoint contains nuisance weights.")
        if any(key.startswith("_base") for key in payload["state_dict"]):
            raise RuntimeError(f"{variant} checkpoint contains cached empirical bases.")
        model = load_celeba_correction_checkpoint(
            path,
            device="cpu",
            reconstruct_base=False,
        )
        if model.variant != variant or model.epoch != 500:
            raise RuntimeError(f"Incorrect reconstructed {variant} metadata.")
        payloads[variant] = payload

    for key in ("train_indices", "ref_indices_a0", "ref_indices_a1"):
        if not torch.equal(
            payloads["decfm"]["data_indices"][key],
            payloads["ot"]["data_indices"][key],
        ):
            raise RuntimeError(f"Checkpoint data mismatch: {key}")

    for method in ("source", "decfm", "ot"):
        values = [metrics[method]["sw2"], *metrics[method]["sw2_by_arm"]]
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"Non-finite metrics for {method}.")
    if not metrics["decfm"]["sw2"] < metrics["source"]["sw2"]:
        raise RuntimeError("Independent correction did not improve over the source.")
    if not metrics["ot"]["sw2"] < metrics["source"]["sw2"]:
        raise RuntimeError("OT correction did not improve over the source.")
    for variant in ("decfm", "ot"):
        check = metrics[variant]["empirical_base_validation"]
        if not check["arm0_exact"] or not check["arm1_exact"]:
            raise RuntimeError(f"{variant} empirical base validation failed.")

    samples = torch.load(result / "samples.pt", map_location="cpu", weights_only=True)
    trajectories = torch.load(
        result / "trajectories.pt", map_location="cpu", weights_only=True
    )
    if any(value.dtype != torch.uint8 for value in samples.values()):
        raise RuntimeError("Display samples are not compact uint8 tensors.")
    for variant in ("decfm", "ot"):
        for arm in ("arm0", "arm1"):
            for extreme in ("top", "bottom"):
                value = trajectories[variant][arm][extreme]["trajectory"]
                if value.dtype != torch.uint8 or value.shape[0] != 5:
                    raise RuntimeError(f"Invalid trajectory artifact: {variant}/{arm}/{extreme}")

    notebook = json.loads((REPO_ROOT / "examples" / "demo_celeba.ipynb").read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if any(cell.get("execution_count") is None for cell in code_cells):
        raise RuntimeError("CelebA demo notebook is not fully executed.")
    if any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    ):
        raise RuntimeError("CelebA demo notebook contains an execution error.")

    if args.data_root is not None:
        reconstructed = reconstruct_celeba_data(
            payloads["decfm"]["dgp_config"],
            root=args.data_root,
            expected_indices=payloads["decfm"]["data_indices"],
            load_observational=False,
            load_references=False,
        )
        if reconstructed["pool_size"] != 61_037:
            raise RuntimeError("Unexpected cleaned CelebA pool size.")

    print(
        "CelebA audit passed:",
        {
            method: round(metrics[method]["sw2"], 6)
            for method in ("source", "decfm", "ot")
        },
    )


if __name__ == "__main__":
    main()
