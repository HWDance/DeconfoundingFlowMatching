#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

# Make the runner work from a fresh clone even before editable installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from deconfoundingfm.core.target import DeconfoundingFlowConfig
from deconfoundingfm.experimental import (
    CMNISTConfig,
    ColorMNISTDGP,
    ExactColorMNISTSourceGenerator,
    GeneratorConditionalFlowFM,
    GeneratorDeconfoundingFlow,
    recover_color_values,
    save_cmnist_correction_checkpoint,
    sliced_wasserstein_images,
)
from deconfoundingfm.experimental.cmnist import (
    color_distribution_diagnostics,
    fid_from_reference,
    inception_features,
    make_inception_feature_extractor,
    prepare_fid_reference,
    save_json,
)
from deconfoundingfm.integrators import integrate_midpoint_trajectory
from deconfoundingfm.nn.velocity import UNet
from deconfoundingfm.nuisance.outcome import ConditionalFlowFMConfig
from deconfoundingfm.nuisance.propensity import RandomForestConfig, RandomForestPropensityEstimator


def build_parser():
    p = argparse.ArgumentParser(
        description="Run the CMNIST pretrained-generator correction backend experiment."
    )
    p.add_argument("--output", type=str, default="examples/cmnist/results/default")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    # Fixed labeled observational population.
    p.add_argument("--bw-shapes", type=int, default=10_000)
    p.add_argument("--colors-per-shape", type=int, default=2)
    p.add_argument("--confounding-w", type=float, default=5.0)

    # Estimated propensity nuisance.
    p.add_argument("--propensity-trees", type=int, default=1000)
    p.add_argument("--propensity-cv-folds", type=int, default=5)

    # Conditional outcome nuisance: exact update budget, fresh generator base each update.
    p.add_argument("--nuisance-steps", type=int, default=20_000)
    p.add_argument("--nuisance-lr", type=float, default=1e-4)

    # Target flows.
    p.add_argument("--target-steps", type=int, default=20_000)
    p.add_argument("--target-lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--unet-c", type=int, default=32)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--plugin-reservoir", type=int, default=2)
    p.add_argument("--plugin-batch", type=int, default=1)
    p.add_argument("--base-noise-std", type=float, default=0.0)
    p.add_argument(
        "--checkpoints",
        type=int,
        nargs="*",
        default=[250, 500, 1000, 2000, 5000, 10000, 15000, 20000],
    )

    # Evaluation.
    p.add_argument("--eval-n", type=int, default=5_000)
    p.add_argument("--checkpoint-eval-n", type=int, default=512)
    p.add_argument("--sw2-projections", type=int, default=256)
    p.add_argument("--sample-chunk", type=int, default=64)
    p.add_argument("--skip-fid", action="store_true")
    p.add_argument("--fid-batch-size", type=int, default=64)
    p.add_argument("--trajectory-n", type=int, default=512)
    p.add_argument("--trajectory-keep", type=int, default=8)
    p.add_argument("--trajectory-chunk", type=int, default=64)

    p.add_argument("--smoke", action="store_true")
    return p


@torch.no_grad()
def sample_model_chunked(model, arm: int, n: int, chunk: int):
    parts = []
    for start in range(0, int(n), int(chunk)):
        parts.append(model.sample(int(arm), min(int(chunk), int(n) - start)).detach().cpu())
    return torch.cat(parts, dim=0)


@torch.no_grad()
def sample_source_chunked(source, arm: int, n: int, chunk: int):
    parts = []
    for start in range(0, int(n), int(chunk)):
        parts.append(
            source.sample(int(arm), min(int(chunk), int(n) - start), device="cpu").detach().cpu()
        )
    return torch.cat(parts, dim=0)


@torch.no_grad()
def transform_model_chunked(model, arm: int, bases: torch.Tensor, chunk: int):
    parts = []
    for start in range(0, len(bases), int(chunk)):
        parts.append(model.transform(bases[start : start + int(chunk)], int(arm)).detach().cpu())
    return torch.cat(parts, dim=0)


def mean_arm_sw2(samples_by_arm, truth_by_arm, *, projections, seed):
    vals = []
    for arm in (0, 1):
        vals.append(
            sliced_wasserstein_images(
                samples_by_arm[arm],
                truth_by_arm[arm],
                n_projections=projections,
                seed=seed + arm,
            )
        )
    return float(sum(vals) / len(vals)), vals


@torch.no_grad()
def foreground_pixel_change(
    y0: torch.Tensor,
    y1: torch.Tensor,
    *,
    foreground_threshold: float = 0.05,
) -> torch.Tensor:
    """Mean absolute RGB change restricted to source-digit foreground pixels."""
    if y0.shape != y1.shape or y0.ndim != 4:
        raise ValueError("y0 and y1 must have matching image batches.")
    foreground = y0.sum(dim=1) > float(foreground_threshold)
    per_pixel = (y1 - y0).abs().mean(dim=1)
    numerator = (per_pixel * foreground).sum(dim=(1, 2))
    denominator = foreground.sum(dim=(1, 2)).clamp_min(1)
    return numerator / denominator


@torch.no_grad()
def trajectory_diagnostics(
    model,
    base_samples_by_arm,
    *,
    arms=(0, 1),
    steps=50,
    keep=8,
    chunk=64,
):
    """Summarize trajectories and retain top/bottom foreground-changing examples."""
    device = next(model.velocity.parameters()).device
    summary = {}
    saved = {}
    dim = None
    arms = tuple(int(arm) for arm in arms)
    for arm in arms:
        base = base_samples_by_arm[arm]
        n = len(base)
        trajectories = []
        values = {
            "foreground_pixel_change": [],
            "path_length": [],
            "endpoint_displacement": [],
            "straightness_ratio": [],
            "bar_E_v": [],
            "bar_E_vdot": [],
        }
        for start in range(0, n, int(chunk)):
            y0 = base[start : start + int(chunk)].to(device)
            context = model._make_context(int(arm), len(y0), y0.device)
            traj, vmids = integrate_midpoint_trajectory(
                model.velocity, y0, context=context, steps=steps
            )
            dim = int(y0[0].numel())
            dt = 1.0 / int(steps)
            displacement = (traj[-1] - traj[0]).reshape(len(y0), -1).norm(dim=1) / (dim**0.5)
            speed = vmids.reshape(steps, len(y0), -1).norm(dim=2)
            path_length = speed.sum(dim=0) * dt / (dim**0.5)
            straightness = path_length / displacement.clamp_min(1e-8)
            path_energy = (
                vmids.reshape(steps, len(y0), -1).square().sum(dim=2).sum(dim=0) * dt / dim
            )
            if steps > 1:
                dv = (vmids[1:] - vmids[:-1]) / dt
                derivative_energy = (
                    dv.reshape(steps - 1, len(y0), -1).square().sum(dim=2).sum(dim=0) * dt / dim
                )
            else:
                derivative_energy = torch.zeros(len(y0), device=y0.device, dtype=y0.dtype)
            change = foreground_pixel_change(traj[0], traj[-1])
            trajectories.append(traj.detach().cpu())
            for key, tensor in (
                ("foreground_pixel_change", change),
                ("path_length", path_length),
                ("endpoint_displacement", displacement),
                ("straightness_ratio", straightness),
                ("bar_E_v", path_energy),
                ("bar_E_vdot", derivative_energy),
            ):
                values[key].append(tensor.detach().cpu())

        trajectories = torch.cat(trajectories, dim=1)
        values = {key: torch.cat(parts) for key, parts in values.items()}
        topk = min(int(keep), int(n))
        top_indices = torch.topk(values["foreground_pixel_change"], k=topk).indices
        bottom_indices = torch.topk(
            values["foreground_pixel_change"], k=topk, largest=False
        ).indices
        key = f"arm{arm}"
        summary[key] = {
            "mean_foreground_pixel_change": float(values["foreground_pixel_change"].mean()),
            "mean_path_length": float(values["path_length"].mean()),
            "mean_endpoint_displacement": float(values["endpoint_displacement"].mean()),
            "mean_straightness_ratio": float(values["straightness_ratio"].mean()),
            "mean_bar_E_v": float(values["bar_E_v"].mean()),
            "mean_bar_E_vdot": float(values["bar_E_vdot"].mean()),
            "n_trajectories": int(n),
            "ode_steps": int(steps),
            "dimension": int(dim),
            "selection": "mean absolute RGB change over source foreground pixels",
        }

        def selected(indices, trajectories=trajectories, values=values):
            return {
                "indices": indices.clone(),
                "trajectory": trajectories[:, indices].clone(),
                **{name: tensor[indices].clone() for name, tensor in values.items()},
            }

        saved[key] = {
            "selection_metric": "foreground_pixel_change",
            "top": selected(top_indices),
            "bottom": selected(bottom_indices),
        }

    averaged_keys = (
        "mean_foreground_pixel_change",
        "mean_path_length",
        "mean_endpoint_displacement",
        "mean_straightness_ratio",
        "mean_bar_E_v",
        "mean_bar_E_vdot",
    )
    summary["average_over_arms"] = {
        key: float(sum(summary[f"arm{arm}"][key] for arm in arms) / len(arms))
        for key in averaged_keys
    }
    summary["average_over_arms"].update(
        {
            "dimension": int(dim if dim is not None else 0),
            "n_trajectories_per_arm": len(base_samples_by_arm[arms[0]]),
        }
    )
    return summary, saved


def main():
    args = build_parser().parse_args()
    if args.smoke:
        args.bw_shapes = 16
        args.colors_per_shape = 2
        args.propensity_trees = 10
        args.propensity_cv_folds = 2
        args.nuisance_steps = 1
        args.target_steps = 1
        args.batch_size = 8
        args.unet_c = 2
        args.ode_steps = 1
        args.plugin_reservoir = 1
        args.eval_n = 4
        args.checkpoint_eval_n = 4
        args.sw2_projections = 4
        args.sample_chunk = 4
        args.checkpoints = [1]
        args.skip_fid = True
        args.trajectory_n = 4
        args.trajectory_keep = 2
        args.trajectory_chunk = 2

    out = Path(args.output)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------
    # 1. Exact original CMNIST DGP + fixed 20k labeled population.
    # ------------------------------------------------------------------
    cmnist_cfg = CMNISTConfig(
        digits=(1, 6),
        x_low=0.0,
        x_high=1.0,
        propensity_w=args.confounding_w,
        tau=0.08,
        k=10.0,
        fg_alpha=0.0,
        n_bw_shapes=args.bw_shapes,
        color_draws_per_shape=args.colors_per_shape,
    )
    dgp = ColorMNISTDGP(cmnist_cfg, device=device)
    population = dgp.make_observational_population(seed=args.seed)
    X, A, Y = population["X"], population["A"], population["Y"]
    expected_n = args.bw_shapes * args.colors_per_shape
    if len(X) != expected_n:
        raise RuntimeError(f"Expected {expected_n} observational examples, got {len(X)}.")
    print(
        f"CMNIST observational population: {len(X):,} images "
        f"({args.bw_shapes:,} grayscale shape draws x {args.colors_per_shape} colors)"
    )

    # ------------------------------------------------------------------
    # 2. Pretrained biased generator stand-in: exact fresh P(Y|A=a) draws.
    # ------------------------------------------------------------------
    source_generator = ExactColorMNISTSourceGenerator(dgp)

    # ------------------------------------------------------------------
    # 3. Estimate propensity from the fixed labeled dataset (never oracle).
    # ------------------------------------------------------------------
    propensity_cfg = RandomForestConfig(
        in_dim=1,
        n_estimators=args.propensity_trees,
        max_depth=5,
        min_samples_leaf=1,
        random_state=args.seed,
        n_jobs=-1,
    )
    propensity = RandomForestPropensityEstimator(propensity_cfg, device=device)
    propensity.cross_validate(X.detach().cpu(), A.detach().cpu(), n_splits=args.propensity_cv_folds)
    print("Estimated propensity CV AUC:", propensity.history.get("val_auc"))
    print("Estimated propensity RF max_depth:", propensity.history.get("max_depth"))

    # ------------------------------------------------------------------
    # 4. Estimate P(Y|X,A) for DeconfoundingFM and OT-DeconfoundingFM.
    #    This nuisance draws fresh generator samples as its FM base.
    # ------------------------------------------------------------------
    generator_nuisance_cfg = ConditionalFlowFMConfig(
        dim_y=1,
        dim_x=1,
        lr=args.nuisance_lr,
        batch_size=args.batch_size,
        ode_steps=args.ode_steps,
        base_kind="empirical",  # intercepted by GeneratorConditionalFlowFM
        base_noise_std=0.0,
        velocity_kind="unetx",
        y_is_image=True,
        y_channels=3,
        y_height=28,
        y_width=28,
        num_classes=2,
        x_dim=1,
        unet_c=args.unet_c,
    )
    generator_nuisance = GeneratorConditionalFlowFM(
        generator_nuisance_cfg, source_generator, device=device
    )
    generator_nuisance.fit_iterations(
        X,
        A,
        Y,
        iterations=args.nuisance_steps,
        batch_size=args.batch_size,
        lr=args.nuisance_lr,
        verbose=True,
    )

    init_velocity = UNet(in_channels=3, out_channels=3, num_classes=2, c=args.unet_c).to(device)
    init_state = deepcopy(init_velocity.state_dict())

    def make_target(*, use_ot=False):
        cfg = DeconfoundingFlowConfig(
            dim_y=1,
            hidden=64,
            layers=1,
            base_kind="empirical",
            batch_size=args.batch_size,
            lr=args.target_lr,
            iterations=args.target_steps,
            ode_steps=args.ode_steps,
            plugin_reservoir=args.plugin_reservoir,
            plugin_batch=args.plugin_batch,
            update_plugin_reservoir=False,
            base_noise_std=args.base_noise_std,
            use_ot=use_ot,
            ot_src_batch=args.batch_size,
            ot_plugin_batch=1,
            ot_iters=20,
            ot_eps_scale=0.1,
        )
        velocity = UNet(in_channels=3, out_channels=3, num_classes=2, c=args.unet_c).to(device)
        velocity.load_state_dict(init_state)
        return GeneratorDeconfoundingFlow(
            cfg,
            generator_nuisance,
            propensity,
            source_generator,
            device=device,
            velocity=velocity,
        )

    # Fresh source-generator base samples are drawn inside the DeCFM target steps.
    decfm = make_target(use_ot=False)
    decfm.fit(X, A, Y, verbose=True, checkpoint_steps=args.checkpoints)

    ot = make_target(use_ot=True)
    ot.fit(X, A, Y, verbose=True, checkpoint_steps=args.checkpoints)

    # Save every requested correction state plus the final state. Payloads contain
    # only velocity weights and reconstruction metadata--never plug-in/base samples.
    def save_model_checkpoints(model, variant):
        final_step = int(model.training_steps_)
        states = dict(model.checkpoint_state_dicts_)
        states[final_step] = {
            key: value.detach().cpu().clone() for key, value in model.velocity.state_dict().items()
        }
        entries = {}
        for step in sorted(states):
            relative_path = Path("models") / variant / f"step_{int(step):06d}.pt"
            save_cmnist_correction_checkpoint(
                out / relative_path,
                state_dict=states[step],
                variant=variant,
                step=step,
                ode_steps=model.cfg.ode_steps,
                unet_c=args.unet_c,
                target_config=asdict(model.cfg),
                dgp_config=cmnist_cfg,
                observational_seed=args.seed,
            )
            entries[str(int(step))] = relative_path.as_posix()
        return {
            "variant": variant,
            "final_step": final_step,
            "checkpoints": entries,
        }

    model_manifest = {
        "format_version": 1,
        "path_kind": "relative_to_result_directory",
        "models": {
            "decfm": save_model_checkpoints(decfm, "decfm"),
            "ot": save_model_checkpoints(ot, "ot"),
        },
    }
    decfm.drain_plugin_store()
    ot.drain_plugin_store()

    # ------------------------------------------------------------------
    # 5. Fresh reference evaluation from the exact DGP.
    # ------------------------------------------------------------------
    truth = {}
    source = {}
    model_samples = {"decfm": {}, "ot": {}}
    for arm in (0, 1):
        _, _, truth_arm = dgp.sample_interventional(arm, args.eval_n)
        truth[arm] = truth_arm.detach().cpu()
        source[arm] = sample_source_chunked(source_generator, arm, args.eval_n, args.sample_chunk)
        model_samples["decfm"][arm] = sample_model_chunked(
            decfm, arm, args.eval_n, args.sample_chunk
        )
        model_samples["ot"][arm] = sample_model_chunked(ot, arm, args.eval_n, args.sample_chunk)

    final_projection_seed = args.seed + 1000
    source_sw2, source_arms = mean_arm_sw2(
        source, truth, projections=args.sw2_projections, seed=final_projection_seed
    )
    decfm_sw2, decfm_arms = mean_arm_sw2(
        model_samples["decfm"],
        truth,
        projections=args.sw2_projections,
        seed=final_projection_seed,
    )
    ot_sw2, ot_arms = mean_arm_sw2(
        model_samples["ot"],
        truth,
        projections=args.sw2_projections,
        seed=final_projection_seed,
    )
    metrics = {
        "source_sw2": source_sw2,
        "decfm_sw2": decfm_sw2,
        "ot_sw2": ot_sw2,
        "source_sw2_by_arm": source_arms,
        "decfm_sw2_by_arm": decfm_arms,
        "ot_sw2_by_arm": ot_arms,
        "evaluation_n_per_arm": int(args.eval_n),
        "sw2_projections": int(args.sw2_projections),
        "sw2_projection_seed": int(final_projection_seed),
        "metric_note": "Sliced Wasserstein-2 on flattened RGB images; final value averages arms 0 and 1. Every method uses identical projection directions.",
    }
    inception = None
    if not args.skip_fid:
        print("Loading ImageNet-pretrained Inception-v3 for FID ...")
        inception = make_inception_feature_extractor(device=device)
    if inception is not None:
        fid_source = []
        fid_decfm = []
        fid_ot = []
        for arm in (0, 1):
            real_feats = inception_features(
                truth[arm],
                inception,
                batch_size=args.fid_batch_size,
                device=device,
            )
            reference = prepare_fid_reference(real_feats)
            source_feats = inception_features(
                source[arm],
                inception,
                batch_size=args.fid_batch_size,
                device=device,
            )
            fid_source.append(fid_from_reference(reference, source_feats))
            del source_feats
            decfm_feats = inception_features(
                model_samples["decfm"][arm],
                inception,
                batch_size=args.fid_batch_size,
                device=device,
            )
            fid_decfm.append(fid_from_reference(reference, decfm_feats))
            del decfm_feats
            ot_feats = inception_features(
                model_samples["ot"][arm],
                inception,
                batch_size=args.fid_batch_size,
                device=device,
            )
            fid_ot.append(fid_from_reference(reference, ot_feats))
            del real_feats, reference, ot_feats
        metrics.update(
            {
                "source_fid": float(sum(fid_source) / len(fid_source)),
                "decfm_fid": float(sum(fid_decfm) / len(fid_decfm)),
                "ot_fid": float(sum(fid_ot) / len(fid_ot)),
                "source_fid_by_arm": fid_source,
                "decfm_fid_by_arm": fid_decfm,
                "ot_fid_by_arm": fid_ot,
                "fid_evaluation_n_per_arm": int(args.eval_n),
                "fid_note": (
                    "FID computed from official ImageNet-normalized, pretrained "
                    "Inception-v3 penultimate features; final value averages arms "
                    "0 and 1. This feature definition is recorded for reproducibility."
                ),
            }
        )
    print(json.dumps(metrics, indent=2))

    # Checkpoint convergence uses one deterministic truth set and one shared source
    # batch per arm. Reusing these bases across every method/step removes base-draw
    # Monte Carlo noise from comparisons of the learned correction weights.
    checkpoint_sample_seed = args.seed + 7000
    torch.manual_seed(checkpoint_sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(checkpoint_sample_seed)
    checkpoint_truth = {}
    checkpoint_bases = {}
    for arm in (0, 1):
        _, _, y = dgp.sample_interventional(arm, args.checkpoint_eval_n)
        checkpoint_truth[arm] = y.detach().cpu()
        checkpoint_bases[arm] = sample_source_chunked(
            source_generator,
            arm,
            args.checkpoint_eval_n,
            args.sample_chunk,
        )

    checkpoint_projection_seed = args.seed + 5000

    def checkpoint_curve(model):
        steps = sorted(model.checkpoint_state_dicts_.keys())
        original = deepcopy(model.velocity.state_dict())
        vals = []
        try:
            for step in steps:
                model.velocity.load_state_dict(model.checkpoint_state_dicts_[step])
                samples = {
                    arm: transform_model_chunked(
                        model,
                        arm,
                        checkpoint_bases[arm],
                        args.sample_chunk,
                    )
                    for arm in (0, 1)
                }
                value, _ = mean_arm_sw2(
                    samples,
                    checkpoint_truth,
                    projections=args.sw2_projections,
                    seed=checkpoint_projection_seed,
                )
                vals.append(value)
        finally:
            model.velocity.load_state_dict(original)
        return steps, vals

    steps_d, vals_d = checkpoint_curve(decfm)
    steps_o, vals_o = checkpoint_curve(ot)
    if steps_o != steps_d:
        raise RuntimeError("Target checkpoint steps do not match.")
    convergence = {
        "steps": steps_d,
        "decfm": vals_d,
        "ot": vals_o,
        "evaluation_n_per_arm": int(args.checkpoint_eval_n),
        "sw2_projections": int(args.sw2_projections),
        "projection_seed": int(checkpoint_projection_seed),
        "sample_seed": int(checkpoint_sample_seed),
        "shared_truth_across_methods_and_steps": True,
        "shared_source_bases_across_methods_and_steps": True,
    }

    # Per-image R/(R+B) values approximate the paper's learned P(X(a)).
    color_sample_sets = {
        "source": source,
        "truth": truth,
        "decfm": model_samples["decfm"],
        "ot": model_samples["ot"],
    }
    color_values = {
        method: {f"arm{arm}": recover_color_values(samples_by_arm[arm]) for arm in (0, 1)}
        for method, samples_by_arm in color_sample_sets.items()
    }
    color_diag = {
        method: color_distribution_diagnostics(samples_by_arm)
        for method, samples_by_arm in color_sample_sets.items()
    }

    # Draw one shared source batch per arm, then rank each method by foreground-only
    # endpoint change. Both the top-K and bottom-K full trajectories are persisted.
    trajectory_bases = {
        arm: source_generator.sample(arm, args.trajectory_n, device="cpu") for arm in (0, 1)
    }
    traj_summary_decfm, traj_saved_decfm = trajectory_diagnostics(
        decfm,
        trajectory_bases,
        steps=args.ode_steps,
        keep=args.trajectory_keep,
        chunk=args.trajectory_chunk,
    )
    traj_summary_ot, traj_saved_ot = trajectory_diagnostics(
        ot,
        trajectory_bases,
        steps=args.ode_steps,
        keep=args.trajectory_keep,
        chunk=args.trajectory_chunk,
    )
    trajectory_summary = {
        "selection_batch_n_per_arm": int(args.trajectory_n),
        "selection_keep_per_extreme": int(args.trajectory_keep),
        "selection_metric": "mean absolute RGB change over source foreground pixels",
        "shared_source_batch_across_methods": True,
        "decfm": traj_summary_decfm,
        "ot": traj_summary_ot,
    }

    # Compact result bundle used by demo.ipynb.
    saved_samples = {
        "observed_a0": source[0][:64].clone(),
        "observed_a1": source[1][:64].clone(),
        "true_a0": truth[0][:64].clone(),
        "true_a1": truth[1][:64].clone(),
        "decfm_a0": model_samples["decfm"][0][:64].clone(),
        "decfm_a1": model_samples["decfm"][1][:64].clone(),
        "ot_a0": model_samples["ot"][0][:64].clone(),
        "ot_a1": model_samples["ot"][1][:64].clone(),
    }

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "observational_n": len(X),
            "digits": list(cmnist_cfg.digits),
            "tau": cmnist_cfg.tau,
            "k": cmnist_cfg.k,
            "fg_alpha": cmnist_cfg.fg_alpha,
            "ubyte_source": "packaged original t10k-images.idx3-ubyte / t10k-labels.idx1-ubyte",
            "generator_nuisance_base": "source_generator",
            "reported_variants": ["decfm", "ot"],
        }
    )
    save_json(config_payload, out / "config.json")
    save_json(metrics, out / "metrics.json")
    save_json(convergence, out / "convergence.json")
    save_json(color_diag, out / "color_diagnostics.json")
    save_json(trajectory_summary, out / "trajectory_summary.json")
    save_json(model_manifest, out / "model_manifest.json")
    save_json(
        {
            "reconstruction": "ColorMNISTDGP(CMNISTConfig(**dgp_config)).make_observational_population(seed=observational_seed)",
            "observational_seed": int(args.seed),
            "dgp_config": asdict(cmnist_cfg),
            "observational_n": len(X),
            "direct_observational_tensor_saved": False,
            "source_generator": "ExactColorMNISTSourceGenerator",
            "fresh_base_samples_available": True,
        },
        out / "data_manifest.json",
    )
    save_json(
        {
            "device": str(device),
            "seed": args.seed,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "skip_fid": bool(args.skip_fid),
        },
        out / "run_manifest.json",
    )
    torch.save(saved_samples, out / "samples.pt")
    torch.save(color_values, out / "color_values.pt")
    torch.save({"decfm": traj_saved_decfm, "ot": traj_saved_ot}, out / "trajectories.pt")
    print(f"Saved CMNIST result bundle to {out}")


if __name__ == "__main__":
    main()
