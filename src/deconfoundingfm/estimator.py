"""High-level applied interface for DeconfoundingFM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch

from .core.target import DeconfoundingFlow, DeconfoundingFlowConfig
from .nn.velocity import UNet
from .nuisance.outcome import ConditionalFlowFM, ConditionalFlowFMConfig
from .nuisance.propensity import RandomForestConfig, RandomForestPropensityEstimator

Coupling = Literal["independent", "ot", "eot"]
Architecture = Literal["auto", "mlp", "unet"]


@dataclass
class DeconfoundingFMConfig:
    """Configuration for the binary-treatment applied estimator.

    Defaults preserve the method's empirical observational base.  Architecture-
    specific batch sizes and nuisance-reservoir sizes are selected automatically
    when the corresponding option is left as ``None``.
    """

    coupling: Coupling = "independent"
    architecture: Architecture = "auto"
    device: Optional[str] = None
    seed: int = 0

    # Target deconfounding flow.
    hidden: int = 64
    layers: int = 2
    unet_channels: int = 32
    batch_size: Optional[int] = None
    lr: Optional[float] = None
    epochs: int = 1000
    iterations: Optional[int] = None
    ode_steps: int = 100
    min_propensity: float = 0.01
    base_kind: Literal["empirical", "gaussian"] = "empirical"
    base_noise_std: float = 0.0

    # Cached conditional-outcome draws used by the DR objective.
    plugin_reservoir: Optional[int] = None
    plugin_batch: Optional[int] = None
    update_plugin_reservoir: Optional[bool] = None
    plugin_reservoir_update_frequency: int = 50

    # Conditional outcome nuisance P(Y | X, A).
    nuisance_hidden: int = 64
    nuisance_layers: int = 2
    nuisance_unet_channels: int = 32
    nuisance_film_encoder: bool = False
    nuisance_film_hidden: int = 64
    nuisance_batch_size: Optional[int] = None
    nuisance_lr: Optional[float] = None
    nuisance_epochs: int = 1000
    nuisance_ode_steps: int = 50
    nuisance_weight_decay: float = 0.0
    nuisance_base_kind: Literal["empirical", "gaussian"] = "empirical"
    nuisance_base_noise_std: float = 0.0

    # Default random-forest propensity nuisance.
    propensity_trees: int = 500
    propensity_max_depth: Optional[int] = 5
    propensity_min_samples_leaf: int = 1
    propensity_cross_validate: bool = False
    propensity_cv_folds: int = 5

    # Minibatch EOT coupling.  If eot_epsilon_scale is not None, the research
    # code convention eps = scale * mean(squared minibatch cost) is used.
    eot_epsilon: float = 0.1
    eot_epsilon_scale: Optional[float] = 0.1
    eot_iterations: int = 20
    eot_plugin_batch: int = 1
    eot_source_batch: Optional[int] = 128


class DeconfoundingFM:
    """Fit deconfounding flows from observational ``P(Y|A=a)`` to ``P(Y(a))``.

    The public estimator supports binary treatment, vector or image outcomes,
    independent or minibatch-EOT target couplings, and either default or custom
    nuisance estimators.

    User-supplied nuisance objects are treated as already fitted.  A propensity
    object must be callable on a torch ``X`` tensor and return one probability
    ``P(A=1|X)`` per row.  An outcome nuisance must implement
    ``sample_conditional(x, a, n_per_context=...)``.
    """

    def __init__(
        self,
        config: Optional[DeconfoundingFMConfig] = None,
        *,
        outcome_model: Optional[Any] = None,
        propensity_model: Optional[Any] = None,
    ):
        self.config = config or DeconfoundingFMConfig()
        if self.config.coupling not in {"independent", "ot", "eot"}:
            raise ValueError("coupling must be 'independent', 'ot', or 'eot'.")
        if self.config.architecture not in {"auto", "mlp", "unet"}:
            raise ValueError("architecture must be 'auto', 'mlp', or 'unet'.")
        if self.config.base_kind not in {"empirical", "gaussian"}:
            raise ValueError("base_kind must be 'empirical' or 'gaussian'.")
        if self.config.nuisance_base_kind not in {"empirical", "gaussian"}:
            raise ValueError("nuisance_base_kind must be 'empirical' or 'gaussian'.")
        self.device = torch.device(
            self.config.device
            if self.config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.model_: Optional[DeconfoundingFlow] = None
        self.architecture_: Optional[str] = None
        self.dim_x_: Optional[int] = None
        self.outcome_shape_: Optional[tuple[int, ...]] = None
        self.diagnostics_: dict[str, Any] = {}
        self._fitted = False

    @staticmethod
    def _tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().clone().to(dtype=dtype)
        return torch.as_tensor(value, dtype=dtype)

    def _prepare_data(self, X: Any, A: Any, Y: Any):
        X = self._tensor(X)
        A = self._tensor(A).reshape(-1)
        Y = self._tensor(Y)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("X must have shape (N,d_x).")
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.ndim not in {2, 4}:
            raise ValueError("Y must be vector-valued (N,d_y) or image-valued (N,C,H,W).")
        if not (len(X) == len(A) == len(Y)):
            raise ValueError("X, A, and Y must have the same number of observations.")
        if not torch.isfinite(X).all() or not torch.isfinite(Y).all():
            raise ValueError("X and Y must be finite.")
        unique = set(torch.unique(A).cpu().tolist())
        if unique != {0.0, 1.0}:
            raise ValueError(f"A must contain both binary treatment arms {{0,1}}; got {sorted(unique)}.")
        return X.to(self.device), A.to(self.device), Y.to(self.device)

    def _resolve_architecture(self, Y: torch.Tensor) -> str:
        inferred = "unet" if Y.ndim == 4 else "mlp"
        if self.config.architecture == "auto":
            return inferred
        if self.config.architecture != inferred:
            raise ValueError(
                f"architecture='{self.config.architecture}' is incompatible with Y shape {tuple(Y.shape)}."
            )
        return inferred

    def _resolved_training_options(self, architecture: str) -> dict[str, Any]:
        image = architecture == "unet"
        return {
            "batch_size": self.config.batch_size or (64 if image else 512),
            "lr": self.config.lr or (1e-4 if image else 1e-3),
            "plugin_reservoir": self.config.plugin_reservoir or (1 if image else 32),
            "plugin_batch": self.config.plugin_batch or (1 if image else 4),
            "update_plugin_reservoir": (
                self.config.update_plugin_reservoir
                if self.config.update_plugin_reservoir is not None
                else image
            ),
            "nuisance_batch_size": self.config.nuisance_batch_size or (64 if image else 512),
            "nuisance_lr": self.config.nuisance_lr or (1e-4 if image else 1e-3),
        }

    def _fit_default_propensity(self, X: torch.Tensor, A: torch.Tensor):
        cfg = RandomForestConfig(
            in_dim=X.shape[1],
            n_estimators=self.config.propensity_trees,
            max_depth=self.config.propensity_max_depth,
            min_samples_leaf=self.config.propensity_min_samples_leaf,
            random_state=self.config.seed,
        )
        model = RandomForestPropensityEstimator(cfg, device=self.device)
        if self.config.propensity_cross_validate:
            model.cross_validate(X, A, n_splits=self.config.propensity_cv_folds)
        else:
            model.fit(X, A)
        return model

    def _fit_default_outcome(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        Y: torch.Tensor,
        architecture: str,
        opts: dict[str, Any],
        verbose: bool,
    ):
        if architecture == "mlp":
            cfg = ConditionalFlowFMConfig(
                dim_y=Y.shape[1],
                dim_x=X.shape[1],
                hidden=self.config.nuisance_hidden,
                layers=self.config.nuisance_layers,
                weight_decay=self.config.nuisance_weight_decay,
                lr=opts["nuisance_lr"],
                epochs=self.config.nuisance_epochs,
                batch_size=opts["nuisance_batch_size"],
                ode_steps=self.config.nuisance_ode_steps,
                base_kind=self.config.nuisance_base_kind,
                base_noise_std=self.config.nuisance_base_noise_std,
                velocity_kind="mlp",
                y_is_image=False,
            )
        else:
            channels, height, width = Y.shape[1:]
            cfg = ConditionalFlowFMConfig(
                dim_y=1,  # unused in image mode
                dim_x=X.shape[1],
                weight_decay=self.config.nuisance_weight_decay,
                lr=opts["nuisance_lr"],
                epochs=self.config.nuisance_epochs,
                batch_size=opts["nuisance_batch_size"],
                ode_steps=self.config.nuisance_ode_steps,
                base_kind=self.config.nuisance_base_kind,
                base_noise_std=self.config.nuisance_base_noise_std,
                velocity_kind="unetx",
                y_is_image=True,
                y_channels=channels,
                y_height=height,
                y_width=width,
                num_classes=2,
                x_dim=X.shape[1],
                unet_c=self.config.nuisance_unet_channels,
                film_encoder=self.config.nuisance_film_encoder,
                film_hidden=self.config.nuisance_film_hidden,
            )
        model = ConditionalFlowFM(cfg, device=self.device)
        model.fit(X, A, Y, verbose=verbose)
        return model

    def _build_target(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        architecture: str,
        opts: dict[str, Any],
    ) -> DeconfoundingFlow:
        cfg = DeconfoundingFlowConfig(
            dim_y=Y.shape[1] if Y.ndim == 2 else 1,
            hidden=self.config.hidden,
            layers=self.config.layers,
            base_kind=self.config.base_kind,
            batch_size=opts["batch_size"],
            lr=opts["lr"],
            epochs=self.config.epochs,
            iterations=self.config.iterations,
            ode_steps=self.config.ode_steps,
            min_propensity=self.config.min_propensity,
            plugin_reservoir=opts["plugin_reservoir"],
            plugin_batch=opts["plugin_batch"],
            update_plugin_reservoir=opts["update_plugin_reservoir"],
            plugin_reservoir_update_frequency=self.config.plugin_reservoir_update_frequency,
            base_noise_std=self.config.base_noise_std,
            use_ot=self.config.coupling in {"ot", "eot"},
            ot_eps=self.config.eot_epsilon,
            ot_eps_scale=self.config.eot_epsilon_scale,
            ot_iters=self.config.eot_iterations,
            ot_plugin_batch=self.config.eot_plugin_batch,
            ot_src_batch=self.config.eot_source_batch,
        )
        velocity = None
        if architecture == "unet":
            velocity = UNet(
                in_channels=Y.shape[1],
                out_channels=Y.shape[1],
                num_classes=2,
                c=self.config.unet_channels,
            )
        return DeconfoundingFlow(
            cfg,
            nuisance_outcome=self.outcome_model,
            nuisance_pi=self.propensity_model,
            device=self.device,
            velocity=velocity,
        )

    @torch.no_grad()
    def _diagnostics(self, X: torch.Tensor, A: torch.Tensor) -> dict[str, Any]:
        raw = self.propensity_model(X)
        raw = torch.as_tensor(raw, dtype=X.dtype, device=X.device).reshape(-1)
        eps = float(self.config.min_propensity)
        clipped = raw.clamp(eps, 1.0 - eps)
        A_cpu = A.reshape(-1).long()
        result: dict[str, Any] = {
            "n": int(len(A)),
            "architecture": self.architecture_,
            "coupling": self.config.coupling,
            "automatic_cross_fitting": False,
            "arm_counts": {0: int((A_cpu == 0).sum()), 1: int((A_cpu == 1).sum())},
            "propensity": {
                "raw_min": float(raw.min()),
                "raw_max": float(raw.max()),
                "clipped_min": float(clipped.min()),
                "clipped_max": float(clipped.max()),
                "fraction_clipped": float(((raw < eps) | (raw > 1.0 - eps)).float().mean()),
            },
            "effective_sample_size": {},
        }
        for arm in (0, 1):
            p = clipped if arm == 1 else 1.0 - clipped
            mask = A_cpu == arm
            w = 1.0 / p[mask]
            result["effective_sample_size"][arm] = float(
                (w.sum().square() / w.square().sum()).item()
            )
        if self.model_ is not None and hasattr(self.model_, "training_steps_"):
            result["training_iterations"] = int(self.model_.training_steps_)
        if self.model_ is not None and hasattr(self.model_, "_ot_eps"):
            result["eot_epsilon_last"] = float(self.model_._ot_eps)
        return result

    def fit(self, X: Any, A: Any, Y: Any, *, verbose: bool = True) -> "DeconfoundingFM":
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        X, A, Y = self._prepare_data(X, A, Y)
        architecture = self._resolve_architecture(Y)
        opts = self._resolved_training_options(architecture)
        self.architecture_ = architecture
        self.dim_x_ = X.shape[1]
        self.outcome_shape_ = tuple(Y.shape[1:])

        if self.propensity_model is None:
            self.propensity_model = self._fit_default_propensity(X, A)
        if self.outcome_model is None:
            self.outcome_model = self._fit_default_outcome(
                X, A, Y, architecture, opts, verbose=verbose
            )
        if not callable(self.propensity_model):
            raise TypeError("propensity_model must be callable on a torch X tensor.")
        if not hasattr(self.outcome_model, "sample_conditional"):
            raise TypeError("outcome_model must implement sample_conditional(...).")

        self.model_ = self._build_target(X, Y, architecture, opts)
        self.model_.fit(X, A, Y, verbose=verbose)
        self.model_.eval()
        self._fitted = True
        self.diagnostics_ = self._diagnostics(X, A)
        return self

    def _check_fitted(self):
        if not self._fitted or self.model_ is None:
            raise RuntimeError("DeconfoundingFM is not fitted. Call fit(X, A, Y) first.")

    @torch.no_grad()
    def sample(self, a: int, n: int, *, ode_steps: Optional[int] = None) -> torch.Tensor:
        """Draw samples from the fitted counterfactual distribution for arm ``a``."""
        self._check_fitted()
        return self.model_.sample(a, n, ode_steps=ode_steps)

    @torch.no_grad()
    def transform(self, Y: Any, a: int, *, ode_steps: Optional[int] = None) -> torch.Tensor:
        """Apply the learned arm-specific deconfounding flow to supplied source outcomes."""
        self._check_fitted()
        y = self._tensor(Y).to(self.device)
        expected = self.outcome_shape_
        if len(expected) == 1 and y.ndim == 1:
            if expected[0] == 1:
                y = y[:, None]  # scalar outcome: a 1D input is a batch
            elif y.numel() == expected[0]:
                y = y[None, :]  # vector outcome: a matching 1D input is one sample
        elif len(expected) == 3 and y.ndim == 3:
            y = y.unsqueeze(0)  # one image
        if tuple(y.shape[1:]) != expected:
            raise ValueError(f"Expected outcome shape (N,{expected}); got {tuple(y.shape)}.")
        return self.model_.transform(y, a, ode_steps=ode_steps)

    def diagnostics(self) -> dict[str, Any]:
        self._check_fitted()
        return dict(self.diagnostics_)

    @property
    def velocity_(self):
        self._check_fitted()
        return self.model_.velocity
