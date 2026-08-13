"""Canonical binary-treatment DeconfoundingFM target flow.

This module consolidates the research repository's generalized target-flow
implementation. Setting ``use_ot=False`` recovers the independent-coupling
path; ``use_ot=True`` activates the minibatch entropic-OT conditional. Vector
and image outcomes share this implementation.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..nn.velocity import FMVelocityConfig, MLPVelocityField
from ..integrators import integrate_midpoint
from .data import DatasetDict
from ..couplings import sinkhorn_target_dual, sample_from_ot_conditional


# ================================================================
#  Config
# ================================================================
@dataclass
class DeconfoundingFlowConfig:
    dim_y: int
    hidden: int = 64
    layers: int = 2
    base_kind: str = "empirical"
    batch_size: int = 1024
    lr: float = 1e-3
    epochs: int = 1000
    iterations: Optional[int] = None
    ode_steps: int = 100
    min_propensity: float = 1e-2
    plugin_reservoir: int = 1000
    plugin_batch: int = 10
    update_plugin_reservoir: bool = False
    plugin_reservoir_update_frequency: int = 50
    base_noise_std: float = 0.0

    # ----------------------------
    # OT conditioning (DR-OT)
    # ----------------------------
    use_ot: bool = False
    ot_eps: float = 0.1                # entropic regularisation (used if ot_eps_scale is None)
    ot_iters: int = 20                 # Sinkhorn iterations
    ot_plugin_batch: int = 1           # number of plugin samples per row used to form OT source set
    ot_src_batch: Optional[int] = 128   # optional additional subsample of OT source set size (after ot_plugin_batch)
    ot_eps_scale: Optional[float] = 0.1  # if set: eps := ot_eps_scale * mean(cost)
    ot_eps_min: float = 1e-8
    ot_eps_max: float = 1e8


# ================================================================
#  Shared Parameter DR FM (Image-compatible)
# ================================================================
class DeconfoundingFlow(nn.Module):

    def __init__(
        self,
        cfg: DeconfoundingFlowConfig,
        nuisance_outcome,
        nuisance_pi,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        velocity: nn.Module | None = None,
    ):
        super().__init__()

        self.cfg = cfg
        self.dim_y = cfg.dim_y
        self.nuisance_outcome = nuisance_outcome
        self.nuisance_pi = nuisance_pi

        # -------------------------------------------------
        # Empirical bases (shape-agnostic)
        # -------------------------------------------------
        self.register_buffer("_base0", torch.empty(0))
        self.register_buffer("_base1", torch.empty(0))

        # -------------------------------------------------
        # Velocity field
        # -------------------------------------------------
        if velocity is None:
            vel_cfg = FMVelocityConfig(
                dim_y=cfg.dim_y,
                hidden=cfg.hidden,
                layers=cfg.layers,
                context_dim=1,
            )
            self.velocity = MLPVelocityField(vel_cfg)
        else:
            self.velocity = velocity

        # Image shape (if needed)
        if getattr(self.velocity, "is_image", False):
            if not hasattr(self, "_image_shape"):
                self.register_buffer("_image_shape", torch.tensor([-1, -1, -1], dtype=torch.long))

        # Nuisances are fixed while the target flow is trained.  They may be
        # PyTorch modules or lightweight callable/sampler objects.
        for nuisance in (self.nuisance_outcome, self.nuisance_pi):
            if hasattr(nuisance, "parameters"):
                for parameter in nuisance.parameters():
                    parameter.requires_grad_(False)

        if device is not None:
            self.to(device=device, dtype=dtype)

    # ===============================================================
    # Base handling (copied from IPW)
    # ===============================================================
    def set_empirical_base(self, Y: torch.Tensor, A: torch.Tensor):
        device = next(self.velocity.parameters()).device
        Y = Y.to(device)
        A = A.to(device).long()

        if getattr(self.velocity, "is_image", False):
            if Y.ndim != 4:
                raise ValueError("Image model expects Y of shape (N,C,H,W)")
            C, H, W = Y.shape[1:]
            self._image_shape[:] = torch.tensor([C, H, W], device=device)
        else:
            if Y.ndim == 1:
                Y = Y.unsqueeze(-1)

        self._base0 = Y[A.squeeze(-1) == 0]
        self._base1 = Y[A.squeeze(-1) == 1]

        if self._base0.numel() == 0 or self._base1.numel() == 0:
            raise RuntimeError("Empirical base missing samples for an arm.")

    def _sample_base(self, arm: int, n: int, like: torch.Tensor | None = None):
        device = next(self.velocity.parameters()).device

        # ------------------ Image ------------------
        if getattr(self.velocity, "is_image", False):
            C, H, W = self._image_shape.tolist()
            if C <= 0:
                raise RuntimeError("Image shape unknown; call set_empirical_base first.")

            if self.cfg.base_kind == "gaussian":
                return torch.randn(n, C, H, W, device=device)

            base = self._base0 if arm == 0 else self._base1
            idx = torch.randint(base.shape[0], (n,), device=device)
            y = base[idx]
            if float(self.cfg.base_noise_std) > 0:
                y = y + self.cfg.base_noise_std * torch.randn_like(y)
            return y

        # ------------------ Vector ------------------
        if self.cfg.base_kind == "gaussian":
            return torch.randn(n, self.dim_y, device=device)

        base = self._base0 if arm == 0 else self._base1
        idx = torch.randint(base.shape[0], (n,), device=device)
        y = base[idx]
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if self.cfg.base_noise_std > 0:
            y = y + self.cfg.base_noise_std * torch.randn_like(y)
        return y

    def drain_empirical_base(self):
        """
        Remove empirical base samples while preserving shape invariants.
        Safe for serialization and later re-injection via set_empirical_base().
        """
        device = next(self.velocity.parameters()).device

        # -------------------------------------------------
        # Image-valued Y
        # -------------------------------------------------
        if getattr(self.velocity, "is_image", False):
            if not hasattr(self, "_image_shape"):
                raise RuntimeError(
                    "Image shape unknown; cannot drain empirical base safely."
                )

            C, H, W = self._image_shape.tolist()
            if C <= 0 or H <= 0 or W <= 0:
                raise RuntimeError(
                    "Invalid image shape stored in _image_shape."
                )

            empty = torch.empty((0, C, H, W), device=device)
            self._base0 = empty
            self._base1 = empty

        # -------------------------------------------------
        # Vector-valued Y
        # -------------------------------------------------
        else:
            empty = torch.empty((0, self.dim_y), device=device)
            self._base0 = empty
            self._base1 = empty

    def drain_plugin_store(self):
        device = next(self.velocity.parameters()).device
        if hasattr(self, "_Yhat0_store"):
            self._Yhat0_store = torch.empty(0, device=device)
        if hasattr(self, "_Yhat1_store"):
            self._Yhat1_store = torch.empty(0, device=device)

    # ===============================================================
    # Context handling (copied from IPW)
    # ===============================================================
    def _make_context(self, arm: int, B: int, device):
        if getattr(self.velocity, "requires_onehot_context", False):
            ctx = torch.zeros(B, 2, device=device)
            ctx[:, arm] = 1.0
            return ctx
        return torch.full((B, 1), float(arm), device=device)

    # ===============================================================
    # Plugin reservoir (unchanged logic, shape-safe)
    # ===============================================================
    @torch.no_grad()
    def set_plugin_samples(self, X, A=None, chunk_size: int = 128):
        """Cache nuisance draws with an explicit reservoir axis.

        The research sampler returns ``(N, ...)`` when one draw is requested and
        ``(N, M, ...)`` for ``M>1``.  We normalize both cases to ``(N, M, ...)``
        here; this also fixes the legacy ``plugin_reservoir=1`` shape ambiguity.
        """
        if self.nuisance_outcome is None:
            raise RuntimeError("A nuisance outcome sampler is required for fitting.")
        device = next(self.velocity.parameters()).device
        X = X.to(device)
        N = X.shape[0]
        m = int(self.cfg.plugin_reservoir)
        outcome_shape = tuple(self._base0.shape[1:])

        A0 = torch.zeros(N, device=device)
        A1 = torch.ones(N, device=device)
        Yhat0_chunks = []
        Yhat1_chunks = []

        def normalize(draws, n_rows):
            if not isinstance(draws, torch.Tensor):
                draws = torch.as_tensor(draws, dtype=self._base0.dtype, device=device)
            draws = draws.to(device=device, dtype=self._base0.dtype)
            if draws.shape[0] != n_rows:
                raise ValueError("Outcome nuisance returned the wrong number of contexts.")
            if tuple(draws.shape[1:]) == outcome_shape:
                draws = draws.unsqueeze(1)
            expected = (n_rows, m) + outcome_shape
            if tuple(draws.shape) != expected:
                raise ValueError(
                    "Outcome nuisance must return shape (N, ...) for one draw or "
                    f"(N, M, ...) for M draws; expected {expected}, got {tuple(draws.shape)}."
                )
            if not torch.isfinite(draws).all():
                raise ValueError("Outcome nuisance returned non-finite samples.")
            return draws

        for start in range(0, N, int(chunk_size)):
            end = min(start + int(chunk_size), N)
            x_chunk = X[start:end]
            Y0 = self.nuisance_outcome.sample_conditional(
                x=x_chunk, a=A0[start:end], n_per_context=m
            )
            Y1 = self.nuisance_outcome.sample_conditional(
                x=x_chunk, a=A1[start:end], n_per_context=m
            )
            Yhat0_chunks.append(normalize(Y0, end - start))
            Yhat1_chunks.append(normalize(Y1, end - start))
            if device.type == "cuda":
                torch.cuda.empty_cache()

        self._Yhat0_store = torch.cat(Yhat0_chunks, dim=0)
        self._Yhat1_store = torch.cat(Yhat1_chunks, dim=0)

    @torch.no_grad()
    def _sample_plugin(self, idx, M, arm):
        Yhat_store = self._Yhat0_store if arm == 0 else self._Yhat1_store
        B = idx.shape[0]
        m = Yhat_store.shape[1]

        if M == 1:
            j = torch.randint(0, m, (B,), device=idx.device)
            return Yhat_store[idx, j]  # (B, ...)

        j = torch.randint(0, m, (B, M), device=idx.device)
        idx_rows = idx[:, None].expand(B, M)
        return Yhat_store[idx_rows, j]  # (B, M, ...)

    # ===============================================================
    # Propensity handling (cached)
    # ===============================================================
    @torch.no_grad()
    def precompute_pi(self, X):
        """Cache clipped propensity probabilities for the training observations."""
        device = next(self.velocity.parameters()).device
        raw = self.nuisance_pi(X)
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw, dtype=X.dtype, device=device)
        p1 = raw.reshape(-1).to(device=device, dtype=X.dtype)
        if p1.shape[0] != X.shape[0]:
            raise ValueError(
                "Propensity model must return one P(A=1|X) value per observation."
            )
        if not torch.isfinite(p1).all():
            raise ValueError("Propensity model returned non-finite probabilities.")
        eps = float(self.cfg.min_propensity)
        if not (0.0 < eps < 0.5):
            raise ValueError("min_propensity must lie in (0, 0.5).")
        self.pi_raw_cached = p1.clone()
        p1 = p1.clamp(eps, 1.0 - eps)
        p0 = 1.0 - p1
        self.pi_cached = torch.stack([p0, p1], dim=-1)

    @torch.no_grad()
    def _pi_stacked(self, idx):
        return self.pi_cached[idx]

    # ===============================================================
    # FM pair loss (image-safe)
    # ===============================================================
    def _fm_pair_loss(self, Y_target, Y_base, arm: int):
        device = Y_target.device
        B = Y_target.shape[0]

        t = torch.rand(B, device=device)
        t_view = t.view(-1, *([1] * (Y_target.ndim - 1)))

        y_t = (1 - t_view) * Y_base + t_view * Y_target
        u_star = Y_target - Y_base

        context = self._make_context(arm, B, device)
        v = self.velocity(y_t, t_view, context=context)

        return (v - u_star).reshape(B, -1).pow(2).sum(-1)

    # ===============================================================
    # DR loss per arm (with optional OT conditioning)
    # ===============================================================
    def _dr_arm(self, X, A, Y, idx, arm: int):
        device = Y.device
        B = Y.shape[0]

        # -------------------------
        # Base points (target set)
        # -------------------------
        Y_tilde = self._sample_base(arm, B, like=Y)

        # -------------------------
        # Plugin points
        # -------------------------
        M = int(self.cfg.plugin_batch)
        if M < 1:
            raise ValueError("plugin_batch must be >= 1")

        Y_hat = self._sample_plugin(idx, M, arm)

        # -------------------------
        # Non-OT path (unchanged)
        # -------------------------
        if not self.cfg.use_ot:
            if M == 1:
                ell_hat = self._fm_pair_loss(Y_hat, Y_tilde, arm)
            else:
                Y_tilde_exp = Y_tilde.unsqueeze(1).expand_as(Y_hat)
                ell_hat = self._fm_pair_loss(
                    Y_hat.reshape(-1, *Y_hat.shape[2:]),
                    Y_tilde_exp.reshape(-1, *Y_hat.shape[2:]),
                    arm,
                ).view(B, M).mean(1)

            ell_obs = self._fm_pair_loss(Y, Y_tilde, arm)

            pi = self._pi_stacked(idx)[:, arm]
            w = (A.squeeze(-1) == arm).float() / pi

            return ell_hat.mean() + (w * (ell_obs - ell_hat)).mean()

        # =========================================================
        # OT-conditioned path (matches vector DR-OT structure)
        #   - one coupling per minibatch+arm
        #   - target support = current Y_tilde (size B)
        # =========================================================

        # Flatten target support for OT computations: (B, d_flat)
        Y_tilde_flat = Y_tilde.reshape(B, -1)

        # Build OT source set from plugin samples (subsampled per row)
        if M == 1:
            # Y_hat is (B, ...)
            Y_hat_ot = Y_hat
        else:
            K = min(int(self.cfg.ot_plugin_batch), M)
            if K < 1:
                raise ValueError("ot_plugin_batch must be >= 1 when use_ot=True")
            cols = torch.randint(M, (K,), device=device)
            # Y_hat: (B,M,...) -> pick columns -> (B,K,...) -> flatten -> (B*K,...)
            Y_hat_ot = Y_hat[:, cols, ...].reshape(B * K, *Y.shape[1:])

        # Optional additional subsampling of OT source set size
        if self.cfg.ot_src_batch is not None:
            S = int(self.cfg.ot_src_batch)
            if S < 1:
                raise ValueError("ot_src_batch must be >= 1 if provided")
            if Y_hat_ot.shape[0] > S:
                sel = torch.randint(Y_hat_ot.shape[0], (S,), device=device)
                Y_hat_ot = Y_hat_ot[sel]

        Y_hat_ot_flat = Y_hat_ot.reshape(Y_hat_ot.shape[0], -1)

        # Choose eps
        if self.cfg.ot_eps_scale is None:
            eps_val = float(self.cfg.ot_eps)
        else:
            with torch.no_grad():
                C = torch.cdist(Y_hat_ot_flat, Y_tilde_flat, p=2) ** 2
                mean_cost = C.mean()
            eps_val = float(self.cfg.ot_eps_scale) * float(mean_cost.item())

        # Clamp eps for safety
        eps_val = max(float(self.cfg.ot_eps_min), min(float(self.cfg.ot_eps_max), eps_val))
        self._ot_eps = eps_val


        # (a) Compute target dual potentials v (one per target point in Y_tilde)
        v = sinkhorn_target_dual(
            x_src=Y_hat_ot_flat,
            x_tgt=Y_tilde_flat,
            eps=eps_val,
            n_iters=int(self.cfg.ot_iters),
        )  # (B,)

        # (b) Observed term: sample one OT-conditioned base point per observed Y
        Y_tilde_obs_flat = sample_from_ot_conditional(
            y=Y.reshape(B, -1),
            x_tgt=Y_tilde_flat,
            v=v,
            eps=eps_val,
        )  # (B, d_flat)
        Y_tilde_obs = Y_tilde_obs_flat.reshape_as(Y)
        ell_obs = self._fm_pair_loss(Y, Y_tilde_obs, arm)

        # (c) Plugin term: sample one OT-conditioned base point per plugin draw
        if M == 1:
            Y_hat_flat = Y_hat.reshape(B, -1)  # (B, d_flat)
            Y_tilde_hat_flat = sample_from_ot_conditional(
                y=Y_hat_flat,
                x_tgt=Y_tilde_flat,
                v=v,
                eps=eps_val,
            )  # (B, d_flat)
            Y_tilde_hat = Y_tilde_hat_flat.reshape_as(Y_hat)
            ell_hat = self._fm_pair_loss(Y_hat, Y_tilde_hat, arm)  # (B,)
        else:
            Y_hat_flat = Y_hat.reshape(B * M, -1)  # (B*M, d_flat)
            Y_tilde_hat_flat = sample_from_ot_conditional(
                y=Y_hat_flat,
                x_tgt=Y_tilde_flat,
                v=v,
                eps=eps_val,
            )  # (B*M, d_flat)
            Y_tilde_hat = Y_tilde_hat_flat.reshape(B, M, *Y.shape[1:])

            ell_hat_flat = self._fm_pair_loss(
                Y_hat.reshape(B * M, *Y.shape[1:]),
                Y_tilde_hat.reshape(B * M, *Y.shape[1:]),
                arm,
            ).view(B, M)
            ell_hat = ell_hat_flat.mean(1)  # (B,)

        # (d) DR aggregation (unchanged)
        pi = self._pi_stacked(idx)[:, arm]
        w = (A.squeeze(-1) == arm).float() / pi

        return ell_hat.mean() + (w * (ell_obs - ell_hat)).mean()

    # ===============================================================
    # FM step
    # ===============================================================
    def fm_step(self, batch):
        X, A, Y, idx = batch["X"], batch["A"], batch["Y"], batch["idx"]
        idx = idx.to(Y.device)

        return self._dr_arm(X, A, Y, idx, 0) + self._dr_arm(X, A, Y, idx, 1)

    # ===============================================================
    # Training
    # ===============================================================
    def fit(self, X, A, Y, *, verbose: bool = False):
        """Fit the deconfounding target flow using fixed nuisance estimators."""
        device = next(self.velocity.parameters()).device
        X, A, Y = X.to(device), A.to(device), Y.to(device)

        if X.shape[0] != A.shape[0] or X.shape[0] != Y.shape[0]:
            raise ValueError("X, A, and Y must have the same number of observations.")
        if self.cfg.base_kind not in {"empirical", "gaussian"}:
            raise ValueError("base_kind must be 'empirical' or 'gaussian'.")
        if int(self.cfg.plugin_reservoir) < 1:
            raise ValueError("plugin_reservoir must be >= 1.")
        if int(self.cfg.plugin_batch) < 1:
            raise ValueError("plugin_batch must be >= 1.")

        self.set_empirical_base(Y, A)
        self.precompute_pi(X)
        self.set_plugin_samples(X, A)

        dataset = DatasetDict(X, A, Y)
        loader = DataLoader(
            dataset,
            batch_size=min(int(self.cfg.batch_size), len(dataset)),
            shuffle=True,
        )

        # Only the deconfounding velocity is optimized; nuisance models are fixed.
        opt = torch.optim.Adam(self.velocity.parameters(), lr=self.cfg.lr)
        self.train()

        history = []

        # ``iterations`` gives an exact optimizer-step budget, independent of
        # dataset size and batch size.  ``epochs`` is retained as the backward-
        # compatible fallback when no iteration budget is supplied.
        if self.cfg.iterations is not None:
            n_steps = int(self.cfg.iterations)
            if n_steps < 1:
                raise ValueError("iterations must be >= 1 when provided.")
            iterator = iter(loader)
            for step in range(n_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)

                opt.zero_grad(set_to_none=True)
                loss = self.fm_step(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite DeconfoundingFM loss at iteration {step + 1}."
                    )
                loss.backward()
                opt.step()
                loss_value = float(loss.detach())
                history.append(loss_value)

                if verbose and (
                    step == 0 or (step + 1) % 1000 == 0 or step + 1 == n_steps
                ):
                    print(
                        f"DeconfoundingFM iteration {step + 1}/{n_steps} | "
                        f"loss={loss_value:.6f}"
                    )

                if (
                    self.cfg.update_plugin_reservoir
                    and step > 0
                    and step % int(self.cfg.plugin_reservoir_update_frequency) == 0
                ):
                    self.set_plugin_samples(X, A)
            self.training_steps_ = n_steps
        else:
            total_steps = 0
            for ep in range(int(self.cfg.epochs)):
                total = 0.0
                n_batches = 0
                for batch in loader:
                    opt.zero_grad(set_to_none=True)
                    loss = self.fm_step(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite DeconfoundingFM loss at epoch {ep + 1}."
                        )
                    loss.backward()
                    opt.step()
                    total += float(loss.detach())
                    n_batches += 1
                    total_steps += 1

                mean_loss = total / max(n_batches, 1)
                history.append(mean_loss)
                if verbose and (
                    ep == 0 or (ep + 1) % 100 == 0 or ep + 1 == self.cfg.epochs
                ):
                    print(
                        f"DeconfoundingFM epoch {ep + 1}/{self.cfg.epochs} | "
                        f"loss={mean_loss:.6f}"
                    )

                if (
                    self.cfg.update_plugin_reservoir
                    and ep > 0
                    and ep % self.cfg.plugin_reservoir_update_frequency == 0
                ):
                    self.set_plugin_samples(X, A)
            self.training_steps_ = total_steps

        self.training_loss_ = history
        return self

    # ===============================================================
    # Sampling and deterministic application of the learned flow
    # ===============================================================
    @torch.no_grad()
    def transform(self, y: torch.Tensor, a: int, *, ode_steps: Optional[int] = None):
        if a not in (0, 1):
            raise ValueError("a must be 0 or 1.")
        device = next(self.velocity.parameters()).device
        y = y.to(device)
        if getattr(self.velocity, "is_image", False):
            if y.ndim != 4:
                raise ValueError("Image velocity expects y with shape (N,C,H,W).")
        else:
            if y.ndim == 1:
                y = y.unsqueeze(-1)
            if y.ndim != 2 or y.shape[1] != self.dim_y:
                raise ValueError(f"Expected vector outcomes with shape (N,{self.dim_y}).")
        context = self._make_context(a, y.shape[0], y.device)
        steps = int(self.cfg.ode_steps if ode_steps is None else ode_steps)
        return integrate_midpoint(self.velocity, y, context=context, steps=steps)

    @torch.no_grad()
    def sample(self, a: int, n: int, *, ode_steps: Optional[int] = None):
        if n < 1:
            raise ValueError("n must be >= 1.")
        if getattr(self.velocity, "is_image", False):
            base = self._base0 if a == 0 else self._base1
            z = self._sample_base(a, n, like=base[: min(n, len(base))])
        else:
            z = self._sample_base(a, n)
        return self.transform(z, a, ode_steps=ode_steps)

