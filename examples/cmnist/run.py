#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

# Make the runner work from a fresh clone even before editable installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from deconfoundingfm.experimental import (
    CMNISTConfig,
    ColorMNISTDGP,
    ExactColorMNISTSourceGenerator,
    GeneratorConditionalFlowFM,
    GeneratorDeconfoundingFlow,
    sliced_wasserstein_images,
)
from deconfoundingfm.experimental.cmnist import (
    color_distribution_diagnostics,
    fid_from_features,
    inception_features,
    make_inception_feature_extractor,
)
from deconfoundingfm.integrators import integrate_midpoint_trajectory
from deconfoundingfm.nuisance.outcome import ConditionalFlowFMConfig
from deconfoundingfm.nuisance.propensity import RandomForestConfig, RandomForestPropensityEstimator
from deconfoundingfm.core.target import DeconfoundingFlowConfig
from deconfoundingfm.nn.velocity import UNet
from deconfoundingfm.experimental.cmnist import save_json


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
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--checkpoint-eval-n", type=int, default=128)
    p.add_argument("--sw2-projections", type=int, default=256)
    p.add_argument("--sample-chunk", type=int, default=64)
    p.add_argument("--skip-fid", action="store_true")
    p.add_argument("--fid-batch-size", type=int, default=64)
    p.add_argument("--trajectory-n", type=int, default=64)
    p.add_argument("--trajectory-keep", type=int, default=8)

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
        parts.append(source.sample(int(arm), min(int(chunk), int(n) - start), device="cpu").detach().cpu())
    return torch.cat(parts, dim=0)


def mean_arm_sw2(samples_by_arm, truth_by_arm, *, projections, seed):
    vals = []
    for arm in (0, 1):
        vals.append(
            sliced_wasserstein_images(
                samples_by_arm[arm], truth_by_arm[arm],
                n_projections=projections, seed=seed + arm,
            )
        )
    return float(sum(vals) / len(vals)), vals


@torch.no_grad()
def trajectory_diagnostics(model, source_generator, *, arms=(0, 1), n=64, steps=50, keep=8):
    device = next(model.velocity.parameters()).device
    summary = {}
    saved = {}
    dim = None
    for arm in arms:
        y0 = source_generator.sample(int(arm), int(n), device=device)
        ctx = model._make_context(int(arm), y0.shape[0], y0.device)
        traj, vmids = integrate_midpoint_trajectory(model.velocity, y0, context=ctx, steps=steps)
        dim = int(y0[0].numel())
        dt = 1.0 / int(steps)
        disp = (traj[-1] - traj[0]).reshape(n, -1).norm(dim=1) / (dim ** 0.5)
        speed = vmids.reshape(steps, n, -1).norm(dim=2)
        path_length = speed.sum(dim=0) * dt / (dim ** 0.5)
        straightness = path_length / disp.clamp_min(1e-8)
        path_energy_bar = vmids.reshape(steps, n, -1).square().sum(dim=2).sum(dim=0) * dt / dim
        if steps > 1:
            dv = (vmids[1:] - vmids[:-1]) / dt
            vdot_energy_bar = dv.reshape(steps - 1, n, -1).square().sum(dim=2).sum(dim=0) * dt / dim
        else:
            vdot_energy_bar = torch.zeros(n, device=y0.device, dtype=y0.dtype)
        key = f'arm{arm}'
        summary[key] = {
            'mean_path_length': float(path_length.mean().detach().cpu()),
            'mean_endpoint_displacement': float(disp.mean().detach().cpu()),
            'mean_straightness_ratio': float(straightness.mean().detach().cpu()),
            'mean_bar_E_v': float(path_energy_bar.mean().detach().cpu()),
            'mean_bar_E_vdot': float(vdot_energy_bar.mean().detach().cpu()),
            'n_trajectories': int(n),
            'ode_steps': int(steps),
            'dimension': int(dim),
        }
        topk = min(int(keep), int(n))
        idx = torch.topk(disp.detach().cpu(), k=topk).indices
        saved[key] = {
            'y0': traj[0, idx].detach().cpu(),
            'trajectory': traj[:, idx].detach().cpu(),
            'displacement': disp[idx].detach().cpu(),
            'path_length': path_length[idx].detach().cpu(),
            'bar_E_v': path_energy_bar[idx].detach().cpu(),
            'bar_E_vdot': vdot_energy_bar[idx].detach().cpu(),
        }
    summary['average_over_arms'] = {
        'mean_path_length': float(sum(summary[f'arm{a}']['mean_path_length'] for a in arms) / len(tuple(arms))),
        'mean_endpoint_displacement': float(sum(summary[f'arm{a}']['mean_endpoint_displacement'] for a in arms) / len(tuple(arms))),
        'mean_straightness_ratio': float(sum(summary[f'arm{a}']['mean_straightness_ratio'] for a in arms) / len(tuple(arms))),
        'mean_bar_E_v': float(sum(summary[f'arm{a}']['mean_bar_E_v'] for a in arms) / len(tuple(arms))),
        'mean_bar_E_vdot': float(sum(summary[f'arm{a}']['mean_bar_E_vdot'] for a in arms) / len(tuple(arms))),
        'dimension': int(dim if dim is not None else 0),
    }
    return summary, saved


def save_correction_checkpoint(model, path: Path, *, variant: str, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'variant': variant,
        'state_dict': {k: v.detach().cpu() for k, v in model.velocity.state_dict().items()},
        'ode_steps': int(model.cfg.ode_steps),
        'unet_c': int(args.unet_c),
        'base_kind': str(model.cfg.base_kind),
        'note': 'Correction-only checkpoint; no plugin reservoir, nuisance model, or saved base samples.',
    }
    torch.save(payload, path)


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

    inception = None
    if not args.skip_fid:
        print("Loading ImageNet-pretrained Inception-v3 for FID ...")
        inception = make_inception_feature_extractor(device=device)

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
    propensity.cross_validate(
        X.detach().cpu(), A.detach().cpu(), n_splits=args.propensity_cv_folds
    )
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
        X, A, Y,
        iterations=args.nuisance_steps,
        batch_size=args.batch_size,
        lr=args.nuisance_lr,
        verbose=True,
    )

    init_velocity = UNet(
        in_channels=3, out_channels=3, num_classes=2, c=args.unet_c
    ).to(device)
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
        velocity = UNet(
            in_channels=3, out_channels=3, num_classes=2, c=args.unet_c
        ).to(device)
        velocity.load_state_dict(init_state)
        return GeneratorDeconfoundingFlow(
            cfg, generator_nuisance, propensity, source_generator,
            device=device, velocity=velocity,
        )

    # Fresh source-generator base samples are drawn inside the DeCFM target steps.
    decfm = make_target(use_ot=False)
    decfm.fit(X, A, Y, verbose=True, checkpoint_steps=args.checkpoints)

    ot = make_target(use_ot=True)
    ot.fit(X, A, Y, verbose=True, checkpoint_steps=args.checkpoints)

    # Retain correction-only checkpoints for test-time use, drained of plugin stores.
    decfm.drain_plugin_store()
    ot.drain_plugin_store()
    models_dir = out / "models"
    save_correction_checkpoint(decfm, models_dir / "decfm_correction.pt", variant="decfm", args=args)
    save_correction_checkpoint(ot, models_dir / "ot_deconfoundingfm_correction.pt", variant="ot", args=args)

    # ------------------------------------------------------------------
    # 5. Fresh reference evaluation from the exact DGP.
    # ------------------------------------------------------------------
    truth = {}
    source = {}
    model_samples = {"decfm": {}, "ot": {}}
    for arm in (0, 1):
        _, _, truth_arm = dgp.sample_interventional(arm, args.eval_n)
        truth[arm] = truth_arm.detach().cpu()
        source[arm] = sample_source_chunked(
            source_generator, arm, args.eval_n, args.sample_chunk
        )
        model_samples["decfm"][arm] = sample_model_chunked(
            decfm, arm, args.eval_n, args.sample_chunk
        )
        model_samples["ot"][arm] = sample_model_chunked(
            ot, arm, args.eval_n, args.sample_chunk
        )

    source_sw2, source_arms = mean_arm_sw2(
        source, truth, projections=args.sw2_projections, seed=args.seed + 1000
    )
    decfm_sw2, decfm_arms = mean_arm_sw2(
        model_samples["decfm"], truth,
        projections=args.sw2_projections, seed=args.seed + 2000,
    )
    ot_sw2, ot_arms = mean_arm_sw2(
        model_samples["ot"], truth,
        projections=args.sw2_projections, seed=args.seed + 3000,
    )
    metrics = {
        "source_sw2": source_sw2,
        "decfm_sw2": decfm_sw2,
        "ot_sw2": ot_sw2,
        "source_sw2_by_arm": source_arms,
        "decfm_sw2_by_arm": decfm_arms,
        "ot_sw2_by_arm": ot_arms,
        "metric_note": "Sliced Wasserstein-2 on flattened RGB images; final value averages arms 0 and 1.",
    }
    if inception is not None:
        fid_source = []
        fid_decfm = []
        fid_ot = []
        for arm in (0, 1):
            real_feats = inception_features(truth[arm], inception, batch_size=args.fid_batch_size, device=device)
            source_feats = inception_features(source[arm], inception, batch_size=args.fid_batch_size, device=device)
            decfm_feats = inception_features(model_samples["decfm"][arm], inception, batch_size=args.fid_batch_size, device=device)
            ot_feats = inception_features(model_samples["ot"][arm], inception, batch_size=args.fid_batch_size, device=device)
            fid_source.append(fid_from_features(real_feats, source_feats))
            fid_decfm.append(fid_from_features(real_feats, decfm_feats))
            fid_ot.append(fid_from_features(real_feats, ot_feats))
        metrics.update({
            'source_fid': float(sum(fid_source) / len(fid_source)),
            'decfm_fid': float(sum(fid_decfm) / len(fid_decfm)),
            'ot_fid': float(sum(fid_ot) / len(fid_ot)),
            'source_fid_by_arm': fid_source,
            'decfm_fid_by_arm': fid_decfm,
            'ot_fid_by_arm': fid_ot,
            'fid_note': 'FID computed from ImageNet-pretrained Inception-v3 features; final value averages arms 0 and 1.',
        })
    print(json.dumps(metrics, indent=2))

    # Checkpoint convergence uses a smaller fixed fresh reference for tractability.
    checkpoint_truth = {}
    for arm in (0, 1):
        _, _, y = dgp.sample_interventional(arm, args.checkpoint_eval_n)
        checkpoint_truth[arm] = y.detach().cpu()

    def checkpoint_curve(model, seed_offset):
        steps = sorted(model.checkpoint_state_dicts_.keys())
        original = deepcopy(model.velocity.state_dict())
        vals = []
        try:
            for step in steps:
                model.velocity.load_state_dict(model.checkpoint_state_dicts_[step])
                s = {
                    arm: sample_model_chunked(
                        model, arm, args.checkpoint_eval_n, args.sample_chunk
                    )
                    for arm in (0, 1)
                }
                value, _ = mean_arm_sw2(
                    s,
                    checkpoint_truth,
                    projections=args.sw2_projections,
                    seed=args.seed + seed_offset,
                )
                vals.append(value)
        finally:
            model.velocity.load_state_dict(original)
        return steps, vals

    steps_d, vals_d = checkpoint_curve(decfm, 5000)
    steps_o, vals_o = checkpoint_curve(ot, 6000)
    if steps_o != steps_d:
        raise RuntimeError("Target checkpoint steps do not match.")
    convergence = {
        "steps": steps_d,
        "decfm": vals_d,
        "ot": vals_o,
    }

    # CMNIST-specific color diagnostics and path-shape diagnostics.
    color_diag = {
        'source': color_distribution_diagnostics(source),
        'decfm': color_distribution_diagnostics(model_samples['decfm']),
        'ot': color_distribution_diagnostics(model_samples['ot']),
    }
    traj_summary_decfm, traj_saved_decfm = trajectory_diagnostics(
        decfm, source_generator, n=args.trajectory_n, steps=args.ode_steps, keep=args.trajectory_keep
    )
    traj_summary_ot, traj_saved_ot = trajectory_diagnostics(
        ot, source_generator, n=args.trajectory_n, steps=args.ode_steps, keep=args.trajectory_keep
    )
    trajectory_summary = {'decfm': traj_summary_decfm, 'ot': traj_summary_ot}

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
    save_json(
        {
            'decfm': {'path': str(models_dir / 'decfm_correction.pt'), 'variant': 'decfm'},
            'ot': {'path': str(models_dir / 'ot_deconfoundingfm_correction.pt'), 'variant': 'ot'},
        },
        out / 'model_manifest.json',
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
    torch.save({'decfm': traj_saved_decfm, 'ot': traj_saved_ot}, out / 'trajectories.pt')
    print(f"Saved CMNIST result bundle to {out}")


if __name__ == "__main__":
    main()
