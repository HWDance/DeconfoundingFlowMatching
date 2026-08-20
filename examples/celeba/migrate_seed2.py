#!/usr/bin/env python
"""Convert the selected legacy seed-2 CelebA target flows to the new API."""

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
    CelebAGenderHairConfig,
    save_celeba_correction_checkpoint,
)
from deconfoundingfm.nn.velocity import UNet


def parser():
    default_legacy = REPO_ROOT.parent / "doFlow" / "doflow_clean_push"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-repo", type=Path, default=default_legacy)
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "examples" / "celeba" / "results" / "pretrained" / "models",
    )
    return p


def metric_row(path: Path, tag: str):
    rows = json.loads(path.read_text())
    matches = [row for row in rows if row["tag"] == tag]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one metric row for {tag!r} in {path}.")
    return matches[0]


def velocity_state(payload):
    state = {
        key.removeprefix("velocity."): value
        for key, value in payload["dr_state_dict"].items()
        if key.startswith("velocity.")
    }
    expected = UNet(
        in_channels=3,
        out_channels=3,
        num_classes=2,
        c=int(payload["dr_unet_c"]),
    )
    expected.load_state_dict(state, strict=True)
    return state


def data_config(meta):
    values = {
        key: meta[key]
        for key in (
            "image_size",
            "n_obs",
            "n_ref",
            "train_split",
            "ref_split",
            "split_mode",
            "source_splits",
            "a_name",
            "x_name",
            "clean_hair",
            "hair_pair",
            "target_px1",
            "p_a1_x0",
            "p_a1_x1",
            "seed",
        )
    }
    return CelebAGenderHairConfig(**values)


def main():
    args = parser().parse_args()
    legacy = args.legacy_repo.resolve()
    celeba = legacy / "experiments" / "CelebA"

    nonot_container = torch.load(
        celeba / "dr_celeba_seed2_3264_500epochs_ot=False.pt",
        map_location="cpu",
        weights_only=True,
    )
    nonot = nonot_container[1]
    ot = torch.load(
        celeba
        / "results"
        / "celeba_gender_hair_custom_empirical_noise0.2_seed2_unet64_res2_pb1.pt",
        map_location="cpu",
        weights_only=True,
    )
    selected = {
        "decfm": {
            "payload": nonot,
            "source": "experiments/CelebA/dr_celeba_seed2_3264_500epochs_ot=False.pt",
            "container_index": 1,
            "metric": metric_row(
                celeba / "eval_dr_celeba_seed1_3264" / "metrics.json",
                "run01_seed2_empirical_noise0.2_unet64_epochs500",
            ),
        },
        "ot": {
            "payload": ot,
            "source": (
                "experiments/CelebA/results/"
                "celeba_gender_hair_custom_empirical_noise0.2_seed2_unet64_res2_pb1.pt"
            ),
            "container_index": None,
            "metric": metric_row(
                celeba / "eval_drot_celeba_seed1_3264" / "metrics.json",
                "run01_drot_seed2_empirical_noise0.2_unet64_epochs500",
            ),
        },
    }

    reference_indices = nonot["data_meta"]
    for variant, item in selected.items():
        payload = item["payload"]
        if payload["seed"] != 2 or payload["dr_unet_c"] != 64:
            raise RuntimeError(f"Unexpected selected {variant} payload.")
        for key in ("train_indices", "ref_indices_a0", "ref_indices_a1"):
            if not torch.equal(payload["data_meta"][key], reference_indices[key]):
                raise RuntimeError(f"{variant} does not share the selected seed-2 {key}.")

        meta = payload["data_meta"]
        target_config = dict(payload["dr_config"])
        target_config["use_ot"] = bool(variant == "ot")
        provenance = {
            "legacy_repo": "HWDance/doFlow (local doflow_clean_push checkout)",
            "legacy_checkpoint": item["source"],
            "legacy_container_index": item["container_index"],
            "migration": "Extract dr_state_dict entries under velocity.* and strict-load new UNet.",
        }
        destination = args.output / variant / "epoch_500.pt"
        save_celeba_correction_checkpoint(
            destination,
            state_dict=velocity_state(payload),
            variant=variant,
            epoch=int(payload["dr_config"]["epochs"]),
            ode_steps=int(payload["dr_config"]["ode_steps"]),
            unet_c=int(payload["dr_unet_c"]),
            target_config=target_config,
            plugin_config=payload["plugin_config"],
            propensity=payload["propensity"],
            data_config=data_config(meta),
            data_indices={
                key: meta[key]
                for key in ("train_indices", "ref_indices_a0", "ref_indices_a1")
            },
            data_root_hint=meta["root"],
            legacy_provenance=provenance,
            legacy_evaluation=item["metric"],
        )
        print(f"{variant}: {destination.relative_to(REPO_ROOT)} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
