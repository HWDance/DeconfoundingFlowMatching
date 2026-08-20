#!/usr/bin/env python
"""Recover the exact legacy paper identities and run them through migrated flows.

The legacy figures were selected from ranked trajectory caches.  This script
keeps the exact selected identities, verifies them against reconstructed
seed-1 CelebA records, and stores compact display-only tensors.  Compatibility
seeds reproduce each fixed selection as a draw without replacement from its
documented candidate stratum; they were found after the fact and are not
evidence that the original selection was randomized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deconfoundingfm.experimental import (
    CelebAGenderHairConfig,
    load_celeba_correction_checkpoint,
)
from deconfoundingfm.experimental.celeba import (
    CelebAGenderHairPool,
    generate_celeba_indices,
    select_trajectory_frames,
)

TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)
GALLERY_DISPLAY_INDICES = (2, 3, 11, 4, 5, 7, 38, 25, 13)
GALLERY_TOP_RANKS = frozenset({2, 3, 4, 5, 7, 11, 13})
GALLERY_BOTTOM_RANKS = frozenset({5, 18})


def parser():
    default_legacy = REPO_ROOT.parent / "doFlow" / "doflow_clean_push"
    default_output = REPO_ROOT / "examples" / "celeba" / "results" / "pretrained"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-repo", type=Path, default=default_legacy)
    p.add_argument("--output", type=Path, default=default_output)
    p.add_argument("--models", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ode-steps", type=int, default=50)
    return p


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def seeded_draw(population_size: int, count: int, seed: int):
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(int(population_size), generator=generator)[: int(count)].tolist()


def data_config(cache):
    meta = cache["meta"]["data_meta"]
    allowed = {field.name for field in fields(CelebAGenderHairConfig)}
    values = {key: value for key, value in meta.items() if key in allowed}
    for key in ("source_splits", "hair_pair", "drop_attrs"):
        if key in values:
            values[key] = tuple(values[key])
    return CelebAGenderHairConfig(**values)


def identity_table(cache, pool, generated_indices, arm: int, kind: str):
    entry = cache["arms"][str(arm)][kind]
    treatment, _covariate = pool.labels(generated_indices["train_indices"])
    arm_rows = torch.where(treatment.reshape(-1).long() == int(arm))[0]
    table = []
    for rank, arm_position in enumerate(entry["indices"].long().tolist()):
        observation_row = int(arm_rows[arm_position])
        pool_index = int(generated_indices["train_indices"][observation_row])
        record = pool.records[pool_index]
        table.append(
            {
                "rank_zero_based": rank,
                "rank_one_based": rank + 1,
                "arm_position": arm_position,
                "observation_row": observation_row,
                "pool_index": pool_index,
                "filename": record.filename,
                "A": int(record.A),
                "X": int(record.X),
                "legacy_rms_change": float(entry["scores"][rank]),
            }
        )
    return table


def verify_cached_starts(cache, pool, identities, arm: int, kind: str):
    entry = cache["arms"][str(arm)][kind]
    pool_indices = torch.tensor([row["pool_index"] for row in identities])
    reconstructed = pool.load_images(pool_indices)
    maximum = float((reconstructed - entry["starts"]).abs().max())
    if maximum > 1e-6:
        raise RuntimeError(
            f"Reconstructed arm={arm} {kind} starts differ from cache (max={maximum})."
        )
    return maximum


@torch.no_grad()
def selected_trajectory(model, arm: int, starts: torch.Tensor, ode_steps: int):
    trajectory, _ = model.trajectory(arm, y0=starts, ode_steps=ode_steps)
    frames = select_trajectory_frames(trajectory, TIMES)
    return pack_images(frames[:, 0])


def main():
    args = parser().parse_args()
    legacy = args.legacy_repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    models = (args.models or output / "models").expanduser().resolve()

    cache_paths = {
        "legacy_independent_c32": legacy
        / "experiments/CelebA/figure_cache/dr_seed1_unet32/"
        "run01_dr_seed1_empirical_noise0.2_unet32_epochs500_trajectory_cache.pt",
        "legacy_ot_c64": legacy
        / "experiments/CelebA/figure_cache/drot_seed1_unet64/"
        "run01_drot_seed1_empirical_noise0.2_unet64_epochs500_trajectory_cache.pt",
    }
    caches = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in cache_paths.items()
    }
    independent = caches["legacy_independent_c32"]
    ot = caches["legacy_ot_c64"]
    if independent["tag"] != "run01_dr_seed1_empirical_noise0.2_unet32_epochs500":
        raise RuntimeError("Unexpected independent legacy cache.")
    if ot["tag"] != "run01_drot_seed1_empirical_noise0.2_unet64_epochs500":
        raise RuntimeError("Unexpected OT legacy cache.")

    # Reconstruct seed-1 record identities without loading the 20k-image tensor.
    config = data_config(ot)
    data_root = args.data_root or Path(ot["meta"]["data_meta"]["root"])
    pool = CelebAGenderHairPool(data_root, config)
    generated_indices = generate_celeba_indices(pool, config)

    independent_a0 = identity_table(independent, pool, generated_indices, 0, "top")
    ot_a1 = identity_table(ot, pool, generated_indices, 1, "top")
    ot_a0_top = identity_table(ot, pool, generated_indices, 0, "top")
    ot_a0_bottom = identity_table(ot, pool, generated_indices, 0, "bottom")
    verification = {
        "independent_a0_top_max_abs_diff": verify_cached_starts(
            independent, pool, independent_a0, 0, "top"
        ),
        "ot_a1_top_max_abs_diff": verify_cached_starts(ot, pool, ot_a1, 1, "top"),
        "ot_a0_top_max_abs_diff": verify_cached_starts(ot, pool, ot_a0_top, 0, "top"),
        "ot_a0_bottom_max_abs_diff": verify_cached_starts(
            ot, pool, ot_a0_bottom, 0, "bottom"
        ),
    }

    # These exact fixed selections are reproducible as seeded draws from the
    # original candidate strata.  The seeds are compatibility seeds found after
    # the paper figures existed, not prospective randomization.
    representative_draws = {
        "arm0": seeded_draw(8, 1, 7),
        "arm1": seeded_draw(20, 1, 4),
    }
    gallery_draws = {
        "top": seeded_draw(20, 7, 11009),
        "bottom": seeded_draw(20, 2, 83),
    }
    if representative_draws != {"arm0": [7], "arm1": [10]}:
        raise RuntimeError("Representative compatibility draws changed.")
    if set(gallery_draws["top"]) != GALLERY_TOP_RANKS:
        raise RuntimeError("Top-stratum gallery compatibility draw changed.")
    if set(gallery_draws["bottom"]) != GALLERY_BOTTOM_RANKS:
        raise RuntimeError("Bottom-stratum gallery compatibility draw changed.")

    representative = {
        "arm0": {
            "cache": independent,
            "entry": independent["arms"]["0"]["top"],
            "rank": 7,
            "identities": independent_a0,
        },
        "arm1": {
            "cache": ot,
            "entry": ot["arms"]["1"]["top"],
            "rank": 10,
            "identities": ot_a1,
        },
    }
    gallery_entry = torch.cat(
        [ot["arms"]["0"]["top"]["frames"], ot["arms"]["0"]["bottom"]["frames"]]
    )
    gallery_indices = torch.tensor(GALLERY_DISPLAY_INDICES)
    gallery_frames = gallery_entry[gallery_indices]

    artifact = {
        "gallery": {
            "start": pack_images(gallery_frames[:, 0]),
            "legacy_ot_end": pack_images(gallery_frames[:, -1]),
        },
        "representative": {},
    }
    for arm in (0, 1):
        item = representative[f"arm{arm}"]
        artifact["representative"][f"arm{arm}"] = {
            "candidate_starts": pack_images(item["entry"]["starts"]),
            "candidate_scores": item["entry"]["scores"].detach().cpu(),
            "legacy_trajectory": pack_images(item["entry"]["frames"][item["rank"]]),
        }

    selected_starts = {
        arm: representative[f"arm{arm}"]["entry"]["starts"][
            representative[f"arm{arm}"]["rank"] : representative[f"arm{arm}"]["rank"] + 1
        ]
        for arm in (0, 1)
    }
    gallery_starts = gallery_frames[:, 0]
    for variant in ("decfm", "ot"):
        print(f"Applying migrated {variant} model to exact paper identities...", flush=True)
        model = load_celeba_correction_checkpoint(
            models / variant / "epoch_500.pt",
            device=args.device,
            reconstruct_base=False,
        )
        artifact["gallery"][f"{variant}_end"] = pack_images(
            model.transform(0, gallery_starts, ode_steps=args.ode_steps)
        )
        for arm in (0, 1):
            artifact["representative"][f"arm{arm}"][f"{variant}_trajectory"] = (
                selected_trajectory(model, arm, selected_starts[arm], args.ode_steps)
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gallery_identity_rows = []
    for combined_index in GALLERY_DISPLAY_INDICES:
        if combined_index < 20:
            row = dict(ot_a0_top[combined_index])
            row["stratum"] = "top20"
            row["stratum_rank_zero_based"] = combined_index
        else:
            rank = combined_index - 20
            row = dict(ot_a0_bottom[rank])
            row["stratum"] = "bottom20"
            row["stratum_rank_zero_based"] = rank
        row["combined_index"] = combined_index
        gallery_identity_rows.append(row)

    figures = legacy / "experiments" / "CelebA" / "figures"
    manifest = {
        "format_version": 1,
        "purpose": "Exact legacy paper identities evaluated by migrated seed-2 models.",
        "times": list(TIMES),
        "selection_disclosure": (
            "Selections are the fixed identities used in saved legacy figures. The listed "
            "compatibility seeds were found after the fact to reproduce those subsets as "
            "draws without replacement; they do not imply prospective random selection."
        ),
        "candidate_ranking": (
            "Whole-image RMS displacement from t=0 to t=1, using clean empirical-base "
            "images and scanning every arm-specific seed-1 observational base image."
        ),
        "representative_trajectories": {
            "arm0": {
                "label": "Women (A=0)",
                "legacy_source": "independent, seed 1, U-Net c=32, epoch 500",
                "candidate_stratum": "top 8 of 11,000",
                "draw_without_replacement": {"population_size": 8, "count": 1, "seed": 7},
                "drawn_ranks_zero_based": representative_draws["arm0"],
                "selected": independent_a0[7],
                "legacy_figure": "celeba_a0_top_rank7_trajectory.pdf",
                "legacy_figure_sha256": sha256_file(
                    figures / "celeba_a0_top_rank7_trajectory.pdf"
                ),
            },
            "arm1": {
                "label": "Men (A=1)",
                "legacy_source": "OT, seed 1, U-Net c=64, epoch 500",
                "candidate_stratum": "top 20 of 9,000",
                "draw_without_replacement": {"population_size": 20, "count": 1, "seed": 4},
                "drawn_ranks_zero_based": representative_draws["arm1"],
                "selected": ot_a1[10],
                "legacy_figure": "celeba_a1_top_rank10_trajectory.pdf",
                "legacy_figure_sha256": sha256_file(
                    figures / "celeba_a1_top_rank10_trajectory.pdf"
                ),
            },
        },
        "women_gallery": {
            "legacy_source": "OT, seed 1, U-Net c=64, epoch 500",
            "candidate_strata": "top 20 and bottom 20 of 11,000",
            "draws_without_replacement": {
                "top20": {"population_size": 20, "count": 7, "seed": 11009},
                "bottom20": {"population_size": 20, "count": 2, "seed": 83},
            },
            "drawn_ranks_zero_based": gallery_draws,
            "display_order": "Exact order in the saved paper gallery.",
            "selected": gallery_identity_rows,
            "legacy_figure": "celeba_a0_before_after_selected8.pdf",
            "legacy_figure_sha256": sha256_file(
                figures / "celeba_a0_before_after_selected8.pdf"
            ),
        },
        "cache_provenance": {
            name: {
                "relative_path_in_legacy_repo": str(path.relative_to(legacy)),
                "sha256": sha256_file(path),
                "tag": caches[name]["tag"],
            }
            for name, path in cache_paths.items()
        },
        "identity_verification": {
            **verification,
            "seed": config.seed,
            "data_root_not_stored_in_artifact": True,
        },
        "migrated_evaluation": {
            "models": ["decfm", "ot"],
            "model_seed": 2,
            "ode_steps": args.ode_steps,
            "input": "Exact clean legacy paper start image; no new Gaussian noise added.",
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output / "paper_samples.pt")
    (output / "paper_samples_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(output / "paper_samples.pt")
    print(output / "paper_samples_manifest.json")


if __name__ == "__main__":
    main()
