"""Population-minibatch training utilities for synthetic experiments.

The normal :class:`deconfoundingfm.DeconfoundingFM` API fits from a finite
observational dataset.  For simulation studies it is sometimes useful to train
against the population objective by drawing a fresh observational minibatch at
every optimizer step.  This module provides that separate path without
changing the semantics of the applied finite-data estimator.

The target trainer keeps only a *nuisance context reservoir*: plugin draws are
cached at a renewable set of X values because evaluating the conditional image
flow requires ODE integration.  Fresh X/A/Y and empirical-source Y are still
redrawn at every target step. The plugin draw for each fresh X is taken from
the nearest context in a renewable reservoir, which can be periodically
refreshed.  This mirrors the computational role of the plugin
reservoir in the research code while removing a fixed observational training
sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

import torch

from ..core.target import DeconfoundingFlow
from ..nuisance.outcome import ConditionalFlowFM


class PopulationSource(Protocol):
    """Sampling interface required by :class:`PopulationFlowTrainer`."""

    device: torch.device
    image_shape: tuple[int, ...]

    def sample_x(self, n: int) -> torch.Tensor: ...

    def sample_observational(self, n: int) -> dict[str, torch.Tensor]: ...

    def sample_observational_given_x(self, X: torch.Tensor) -> dict[str, torch.Tensor]: ...

    def sample_source(self, arm: int, n: int) -> dict[str, torch.Tensor]: ...


@dataclass
class PopulationTargetConfig:
    """Controls the renewable plugin reservoir used for population target fitting."""

    context_reservoir_size: int = 2048
    context_refresh_steps: int = 1000
    plugin_context_chunk: int = 32
    plugin_ode_steps: Optional[int] = None
    reservoir_dtype: str = "float16"  # float16 on CUDA, promoted to float32 on CPU
    amp: bool = False


class PopulationFlowTrainer:
    """Train nuisance and target flows from population samplers.

    Notes
    -----
    ``fit_outcome`` draws an entirely fresh observational minibatch every
    optimizer step. ``fit_target`` also redraws A/Y and source outcomes every
    step.  The only reused quantities are conditional-flow draws in a renewable
    X-context reservoir, which is necessary to make image-scale population
    training computationally practical.
    """

    def __init__(
        self,
        source: PopulationSource,
        *,
        device: torch.device | str | None = None,
    ):
        self.source = source
        self.device = torch.device(source.device if device is None else device)
        if torch.device(source.device) != self.device:
            raise ValueError(
                "Population source and trainer must live on the same device. "
                "Construct the source with the requested training device."
            )

    # ------------------------------------------------------------------
    # AMP helpers
    # ------------------------------------------------------------------
    def _autocast(self, enabled: bool):
        use_amp = bool(enabled and self.device.type == "cuda")
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.bfloat16,
            enabled=use_amp,
        )

    def _scaler(self, enabled: bool):
        use_amp = bool(enabled and self.device.type == "cuda")
        # ``torch.cuda.amp.GradScaler`` is available across all supported torch
        # versions in this package. It is a no-op when AMP is disabled.
        return torch.amp.GradScaler("cuda", enabled=use_amp)

    # ------------------------------------------------------------------
    # Population nuisance fitting
    # ------------------------------------------------------------------
    def fit_outcome(
        self,
        model: ConditionalFlowFM,
        *,
        iterations: int,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        amp: bool = False,
        verbose: bool = True,
        log_every: int = 1000,
    ) -> ConditionalFlowFM:
        """Fit ``P(Y|X,A)`` using a fresh population batch at every update."""
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("iterations must be >= 1.")
        batch_size = int(model.cfg.batch_size if batch_size is None else batch_size)
        lr = float(model.cfg.lr if lr is None else lr)
        weight_decay = float(
            model.cfg.weight_decay if weight_decay is None else weight_decay
        )

        model.to(self.device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
        scaler = self._scaler(amp)
        use_amp = bool(amp and self.device.type == "cuda")

        model.train()
        history: list[float] = []
        for step in range(iterations):
            batch = self.source.sample_observational(batch_size)

            # Support empirical nuisance bases too, although the ColorMNIST
            # experiment uses a Gaussian nuisance base.
            if model.cfg.base_kind == "empirical":
                model._base0 = self.source.sample_source(0, batch_size)["Y"]
                model._base1 = self.source.sample_source(1, batch_size)["Y"]

            optimizer.zero_grad(set_to_none=True)
            with self._autocast(use_amp):
                loss = model.fm_step(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite population nuisance loss at iteration {step + 1}."
                )
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            value = float(loss.detach())
            history.append(value)
            if verbose and (
                step == 0 or (step + 1) % int(log_every) == 0 or step + 1 == iterations
            ):
                print(
                    f"Population outcome iteration {step + 1}/{iterations} | "
                    f"loss={value:.6f}"
                )

        model.population_training_steps_ = iterations
        model.training_loss_ = history
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Renewable nuisance reservoir
    # ------------------------------------------------------------------
    def _reservoir_dtype(self, name: str) -> torch.dtype:
        name = str(name).lower()
        if name == "float16" and self.device.type == "cuda":
            return torch.float16
        if name in {"float16", "float32"}:
            return torch.float32
        if name == "bfloat16" and self.device.type == "cuda":
            return torch.bfloat16
        raise ValueError("reservoir_dtype must be 'float16', 'bfloat16', or 'float32'.")

    @torch.no_grad()
    def _build_plugin_reservoir(
        self,
        target: DeconfoundingFlow,
        config: PopulationTargetConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nuisance = target.nuisance_outcome
        if nuisance is None or not hasattr(nuisance, "sample_conditional"):
            raise TypeError("Target nuisance_outcome must implement sample_conditional(...).")

        n_context = int(config.context_reservoir_size)
        chunk = int(config.plugin_context_chunk)
        reservoir = int(target.cfg.plugin_reservoir)
        if n_context < 1 or chunk < 1 or reservoir < 1:
            raise ValueError("Population reservoir sizes must all be >= 1.")

        X = self.source.sample_x(n_context).to(self.device)
        dtype = self._reservoir_dtype(config.reservoir_dtype)
        outcome_shape = tuple(self.source.image_shape)
        y0_chunks: list[torch.Tensor] = []
        y1_chunks: list[torch.Tensor] = []
        ode_steps = config.plugin_ode_steps

        def normalize(draws: torch.Tensor, rows: int) -> torch.Tensor:
            draws = draws.to(self.device)
            if tuple(draws.shape[1:]) == outcome_shape:
                draws = draws.unsqueeze(1)
            expected = (rows, reservoir) + outcome_shape
            if tuple(draws.shape) != expected:
                raise ValueError(
                    f"Plugin reservoir expected {expected}, got {tuple(draws.shape)}."
                )
            return draws.to(dtype=dtype)

        nuisance.eval()
        for start in range(0, n_context, chunk):
            end = min(start + chunk, n_context)
            x_chunk = X[start:end]
            rows = end - start
            a0 = torch.zeros(rows, device=self.device)
            a1 = torch.ones(rows, device=self.device)
            y0 = nuisance.sample_conditional(
                x_chunk,
                a0,
                n_per_context=reservoir,
                ode_steps=ode_steps,
            )
            y1 = nuisance.sample_conditional(
                x_chunk,
                a1,
                n_per_context=reservoir,
                ode_steps=ode_steps,
            )
            y0_chunks.append(normalize(y0, rows))
            y1_chunks.append(normalize(y1, rows))

        return X, torch.cat(y0_chunks, dim=0), torch.cat(y1_chunks, dim=0)

    # ------------------------------------------------------------------
    # Population target fitting
    # ------------------------------------------------------------------
    def fit_target(
        self,
        target: DeconfoundingFlow,
        *,
        iterations: int,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        population: PopulationTargetConfig | None = None,
        checkpoint_steps: Optional[Iterable[int]] = None,
        verbose: bool = True,
        log_every: int = 1000,
    ) -> DeconfoundingFlow:
        """Fit a deconfounding target using fresh population observations.

        Freshness semantics
        -------------------
        * every optimizer step draws fresh ``X,A,Y`` from the DGP;
        * every optimizer step draws new observational source outcomes
          ``Y~P(Y|A=a)`` when ``base_kind='empirical'``;
        * conditional-flow draws live in a renewable X-context reservoir. Fresh
          X values use the nearest cached context for the nuisance draw, and the
          reservoir is replaced every ``context_refresh_steps``.

        The last point is the computational approximation: exact on-the-fly
        plugin ODE sampling at every image target update is typically much more
        expensive than target training. With a dense 1D ColorMNIST context
        reservoir, the nearest-context approximation is very small.
        """
        population = PopulationTargetConfig() if population is None else population
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("iterations must be >= 1.")
        batch_size = int(target.cfg.batch_size if batch_size is None else batch_size)
        lr = float(target.cfg.lr if lr is None else lr)
        refresh = int(population.context_refresh_steps)
        if refresh < 1:
            raise ValueError("context_refresh_steps must be >= 1.")
        if int(target.cfg.plugin_batch) < 1:
            raise ValueError("plugin_batch must be >= 1.")
        if int(target.cfg.plugin_batch) > int(target.cfg.plugin_reservoir):
            raise ValueError("plugin_batch cannot exceed plugin_reservoir in population mode.")

        target.to(self.device)
        target.nuisance_outcome.eval()
        if hasattr(target.nuisance_pi, "eval"):
            target.nuisance_pi.eval()

        # Record the image shape without creating a finite empirical training base.
        if getattr(target.velocity, "is_image", False):
            target._image_shape[:] = torch.tensor(
                self.source.image_shape, device=self.device, dtype=torch.long
            )

        optimizer = torch.optim.Adam(target.velocity.parameters(), lr=lr)
        scaler = self._scaler(population.amp)
        use_amp = bool(population.amp and self.device.type == "cuda")

        checkpoint_set = (
            set() if checkpoint_steps is None else {int(step) for step in checkpoint_steps}
        )
        if any(step < 1 or step > iterations for step in checkpoint_set):
            raise ValueError("checkpoint_steps must lie in [1, iterations].")
        target.checkpoint_state_dicts_ = {}

        X_reservoir = yhat0_reservoir = yhat1_reservoir = None
        history: list[float] = []
        target.train()
        # ``nuisance_outcome`` is registered as a child module of the target, so
        # target.train() would otherwise flip it back into training mode.
        target.nuisance_outcome.eval()
        if hasattr(target.nuisance_pi, "eval"):
            target.nuisance_pi.eval()

        for step in range(iterations):
            if X_reservoir is None or step % refresh == 0:
                X_reservoir, yhat0_reservoir, yhat1_reservoir = (
                    self._build_plugin_reservoir(target, population)
                )
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            # Draw a completely fresh population X minibatch. Expensive plugin
            # ODE samples are approximated using the nearest renewable context.
            X = self.source.sample_x(batch_size).to(self.device)
            distances = (
                X[:, None, :] - X_reservoir[None, :, :]
            ).square().sum(dim=-1)
            idx_res = distances.argmin(dim=1)
            batch = self.source.sample_observational_given_x(X)
            A, Y = batch["A"], batch["Y"]

            # Fresh observational source draws at every target update.
            if target.cfg.base_kind == "empirical":
                target._base0 = self.source.sample_source(0, batch_size)["Y"]
                target._base1 = self.source.sample_source(1, batch_size)["Y"]

            # Stage the renewable plugin reservoir for the selected contexts.
            target._Yhat0_store = yhat0_reservoir.index_select(0, idx_res).to(
                dtype=Y.dtype
            )
            target._Yhat1_store = yhat1_reservoir.index_select(0, idx_res).to(
                dtype=Y.dtype
            )
            target.precompute_pi(X)
            local_idx = torch.arange(batch_size, device=self.device)
            staged = {"X": X, "A": A, "Y": Y, "idx": local_idx}

            optimizer.zero_grad(set_to_none=True)
            # Keep OT in float32 by default. AMP remains available for the
            # independent/Gaussian target paths when explicitly requested.
            amp_this_step = use_amp and not bool(target.cfg.use_ot)
            with self._autocast(amp_this_step):
                loss = target.fm_step(staged)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite population target loss at iteration {step + 1}."
                )
            if amp_this_step:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            value = float(loss.detach())
            history.append(value)
            current_step = step + 1
            if current_step in checkpoint_set:
                target.checkpoint_state_dicts_[current_step] = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in target.velocity.state_dict().items()
                }

            if verbose and (
                step == 0 or current_step % int(log_every) == 0 or current_step == iterations
            ):
                print(
                    f"Population target iteration {current_step}/{iterations} | "
                    f"loss={value:.6f}"
                )

        target.training_steps_ = iterations
        target.training_loss_ = history
        target.eval()
        return target

    # ------------------------------------------------------------------
    # Population sampling from a fitted target
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_target(
        self,
        target: DeconfoundingFlow,
        arm: int,
        n: int,
        *,
        chunk_size: int = 128,
        ode_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample a fitted population target without requiring a finite base cache."""
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1.")
        chunks: list[torch.Tensor] = []
        for start in range(0, n, int(chunk_size)):
            m = min(int(chunk_size), n - start)
            if target.cfg.base_kind == "gaussian":
                z = torch.randn(m, *self.source.image_shape, device=self.device)
            else:
                z = self.source.sample_source(arm, m)["Y"]
            chunks.append(target.transform(z, arm, ode_steps=ode_steps).cpu())
        return torch.cat(chunks, dim=0)
