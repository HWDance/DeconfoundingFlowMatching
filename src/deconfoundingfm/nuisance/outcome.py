"""Conditional flow-matching nuisance for :math:`P(Y|X,A)`.

This is the consolidated vector/image implementation from the research
repository's generalized conditional-flow module.  The nuisance always uses
standard conditional flow matching; OT is reserved for the target
deconfounding coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..nn.velocity import FMVelocityConfig, MLPVelocityField, UNetX
from ..integrators import integrate_midpoint, integrate_euler
from ..core.data import DatasetDict
from ..flow_matching import flow_matching_loss

# ======================================================================
# Config
# ======================================================================

@dataclass
class ConditionalFlowFMConfig:
    """
    Configuration for the conditional flow-matching nuisance model p(Y | X, A).
    """

    # --- core dimensions ---
    dim_y: int                 # vector dim if not image
    dim_x: int                 # continuous X dimension
    dim_a: int = 1             # scalar A for MLP mode

    # --- MLP velocity ---
    hidden: int = 64
    layers: int = 1

    # --- optimisation ---
    weight_decay: float = 0.0
    lr: float = 1e-3
    epochs: int = 1000
    batch_size: int = 1024
    ode_steps: int = 50

    # --- base distribution ---
    base_kind: str = "empirical"   # "gaussian" | "empirical"

    # base mixing + trainability ---
    base_noise_std: float = 0.0     # σ (can be negative/zero)
    base_mix_w: float = 1.0         # w in (0,1), 1=empirical, 0=gaussian
    mix_base: bool = False          # whether to use the mixture base when empirical
    learn_mix: bool = False         # toggle for learning w and σ

    # ============================================================
    # velocity + data mode
    # ============================================================
    velocity_kind: str = "mlp"    # "mlp" | "unetx"

    # --- image-specific (only used if velocity_kind == "unetx") ---
    y_is_image: bool = False
    y_channels: int = 1
    y_height: int = 28
    y_width: int = 28

    # --- UNetX conditioning ---
    num_classes: int = 2          # number of A classes (binary → 2)
    x_dim: int = 1                # continuous X dim passed to UNetX
    
    # --- UNetX architecture ---
    unet_c: int = 64
    film_encoder: bool = False
    film_hidden: int = 64   # optional, but you already expose it in UNetX

# ======================================================================
# Conditional Flow-Matching Nuisance Model
# ======================================================================

class ConditionalFlowFM(nn.Module):
    r"""
    Conditional Flow-Matching model for p_theta(Y | X, A).

    Supports:
      - vector-valued Y with MLP velocity
      - image-valued Y with UNetX velocity
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        cfg: ConditionalFlowFMConfig,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.cfg = cfg
        if cfg.base_kind not in {"empirical", "gaussian"}:
            raise ValueError("base_kind must be 'empirical' or 'gaussian'.")
        if cfg.dim_x < 1:
            raise ValueError("dim_x must be >= 1.")
        if cfg.velocity_kind == "unetx" and not cfg.y_is_image:
            raise ValueError("velocity_kind='unetx' requires y_is_image=True")
        if cfg.velocity_kind == "mlp" and cfg.y_is_image:
            raise ValueError("y_is_image=True is incompatible with velocity_kind='mlp' in this implementation")

        # ------------------------------------------------------------
        # base mixing params (w, σ)
        # ------------------------------------------------------------
        self.base_noise_std = nn.Parameter(torch.tensor(cfg.base_noise_std, dtype=dtype))

        init_w = float(cfg.base_mix_w)
        init_w = min(max(init_w, 1e-6), 1 - 1e-6)  # numerical safety
        init_logit_w = torch.log(torch.tensor(init_w / (1.0 - init_w), dtype=dtype))
        self.base_mix_logit = nn.Parameter(init_logit_w)

        if not cfg.learn_mix:
            self.base_noise_std.requires_grad_(False)
            self.base_mix_logit.requires_grad_(False)

        # ------------------------------------------------------------
        # Velocity instantiation
        # ------------------------------------------------------------
        if cfg.velocity_kind == "mlp":
            context_dim = cfg.dim_x + cfg.dim_a
            vel_cfg = FMVelocityConfig(
                dim_y=cfg.dim_y,
                hidden=cfg.hidden,
                layers=cfg.layers,
                context_dim=context_dim,
            )
            self.velocity = MLPVelocityField(vel_cfg)

        elif cfg.velocity_kind == "unetx":
            self.velocity = UNetX(
                in_channels=cfg.y_channels,
                out_channels=cfg.y_channels,
                num_classes=cfg.num_classes,
                x_dim=cfg.x_dim,
                c=cfg.unet_c,
                film_hidden=cfg.film_hidden,
                film_encoder=cfg.film_encoder,
            )
        else:
            raise ValueError(f"Unknown velocity_kind: {cfg.velocity_kind}")

        self.fm_loss = flow_matching_loss

        # ------------------------------------------------------------
        # Base buffers
        # ------------------------------------------------------------
        if cfg.y_is_image:
            empty = torch.empty(0, cfg.y_channels, cfg.y_height, cfg.y_width)
        else:
            empty = torch.empty(0, cfg.dim_y)

        self.register_buffer("_base0", empty)
        self.register_buffer("_base1", empty)

        if device is not None:
            self.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def _make_context(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Build context tensor.

        - MLP   : [x, a]
        - UNetX : [x_continuous, onehot(a)]
        """
        if self.cfg.velocity_kind == "unetx":
            if a.ndim > 1:
                a = a.squeeze(-1)
            a = a.long()
            a_onehot = F.one_hot(a, num_classes=self.cfg.num_classes).to(x.dtype)
            return torch.cat([x, a_onehot], dim=-1)

        # --- MLP (original behaviour) ---
        if a.ndim == 1:
            a = a.unsqueeze(-1)
        return torch.cat([x, a], dim=-1)

    # ------------------------------------------------------------------
    # Base handling
    # ------------------------------------------------------------------

    def set_empirical_base(self, Y: torch.Tensor, A: torch.Tensor):
        device = next(self.parameters()).device
        Y = Y.to(device)
        A = A.to(device)

        if A.ndim > 1:
            A = A.squeeze(-1)
        A = A.long()

        self._base0 = Y[A == 0]
        self._base1 = Y[A == 1]
        if self._base0.shape[0] == 0 or self._base1.shape[0] == 0:
            raise ValueError("Empirical outcome base requires observations in both treatment arms.")

    def sample_base(self, a: int, n: int) -> torch.Tensor:
        if int(a) not in (0, 1):
            raise ValueError("a must be 0 or 1.")
        if int(n) < 1:
            raise ValueError("n must be >= 1.")
        device = next(self.parameters()).device

        # -------------------------------------------------
        # Gaussian base (unchanged shape; now scaled by σ param)
        # -------------------------------------------------
        if self.cfg.base_kind == "gaussian":
            if self.cfg.y_is_image:
                z = torch.randn(
                    n,
                    self.cfg.y_channels,
                    self.cfg.y_height,
                    self.cfg.y_width,
                    device=device,
                )
            else:
                z = torch.randn(n, self.cfg.dim_y, device=device)

            return z 

        # -------------------------------------------------
        # Empirical base
        # -------------------------------------------------
        base = self._base0 if int(a) == 0 else self._base1
        if base.shape[0] == 0:
            raise RuntimeError(f"Empirical base for arm {a} is empty.")
        idx = torch.randint(base.shape[0], (n,), device=device)
        y_emp = base[idx]

        # -------------------------------------------------
        # empirical–Gaussian mixture
        # -------------------------------------------------
        if self.cfg.mix_base:
            eps = torch.randn_like(y_emp)
            y_gauss = self.base_noise_std * eps
            w = torch.sigmoid(self.base_mix_logit)
            return w * y_emp + (1.0 - w) * y_gauss

        if float(self.base_noise_std.detach()) > 0:
            y_emp = y_emp + self.base_noise_std * torch.randn_like(y_emp)
        return y_emp

    def drain_empirical_base(self):
        """
        Remove empirical base samples while preserving shape invariants.
        This allows safe serialization and later re-injection via set_empirical_base().
        """
        device = next(self.parameters()).device
    
        if self.cfg.y_is_image:
            shape = (0, self.cfg.y_channels, self.cfg.y_height, self.cfg.y_width)
        else:
            shape = (0, self.cfg.dim_y)
    
        self._base0 = torch.empty(*shape, device=device)
        self._base1 = torch.empty(*shape, device=device)


    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def fm_step(self, batch: dict) -> torch.Tensor:
        x = batch["X"]
        a = batch["A"]
        y = batch["Y"]

        if a.ndim > 1:
            a = a.squeeze(-1)
        a = a.long()
        B = y.shape[0]

        z = torch.empty_like(y)
        for arm in (0, 1):
            idx = (a == arm).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                z[idx] = self.sample_base(arm, idx.numel())

        t = torch.rand(B, device=y.device)
        context = self._make_context(x, a)
        return self.fm_loss(self.velocity, z, y, t, context)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        Y: torch.Tensor,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        verbose: bool = True,
    ):
        """Fit the conditional outcome nuisance by standard flow matching."""
        batch_size = self.cfg.batch_size if batch_size is None else int(batch_size)
        epochs = self.cfg.epochs if epochs is None else int(epochs)
        lr = self.cfg.lr if lr is None else float(lr)

        device = next(self.velocity.parameters()).device
        X, A, Y = X.to(device), A.to(device), Y.to(device)
        if X.shape[0] != A.shape[0] or X.shape[0] != Y.shape[0]:
            raise ValueError("X, A, and Y must have the same number of observations.")
        if self.cfg.y_is_image:
            expected = (self.cfg.y_channels, self.cfg.y_height, self.cfg.y_width)
            if Y.ndim != 4 or tuple(Y.shape[1:]) != expected:
                raise ValueError(f"Expected image outcomes of shape (N,{expected}); got {tuple(Y.shape)}.")
        else:
            if Y.ndim == 1:
                Y = Y.unsqueeze(-1)
            if Y.ndim != 2 or Y.shape[1] != self.cfg.dim_y:
                raise ValueError(f"Expected vector outcomes with dim_y={self.cfg.dim_y}.")

        if self.cfg.base_kind == "empirical":
            self.set_empirical_base(Y, A)

        loader = DataLoader(
            DatasetDict(X, A, Y),
            batch_size=min(batch_size, len(X)),
            shuffle=True,
        )
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=lr, weight_decay=self.cfg.weight_decay)

        self.train()
        history = []
        for epoch in range(epochs):
            total = 0.0
            n_batches = 0
            for batch in loader:
                opt.zero_grad(set_to_none=True)
                loss = self.fm_step(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite conditional-flow loss at epoch {epoch + 1}."
                    )
                loss.backward()
                opt.step()
                total += float(loss.detach())
                n_batches += 1
            mean_loss = total / max(n_batches, 1)
            history.append(mean_loss)
            if verbose and (epoch == 0 or (epoch + 1) % 100 == 0 or epoch + 1 == epochs):
                print(f"Outcome nuisance epoch {epoch + 1}/{epochs} | loss={mean_loss:.6f}")

        self.training_loss_ = history
        self.eval()
        return self

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_conditional(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        n_per_context: int = 1,
        ode_steps: Optional[int] = None,
        integrator=integrate_midpoint,
    ):
        if int(n_per_context) < 1:
            raise ValueError("n_per_context must be >= 1.")
        device = next(self.parameters()).device
        ode_steps = self.cfg.ode_steps if ode_steps is None else int(ode_steps)

        x = x.to(device)
        a = a.to(device)
        if a.ndim > 1:
            a = a.squeeze(-1)
        a = a.long()

        N = x.shape[0]
        context = self._make_context(x, a)

        # --------------------------------------------------
        # Single sample
        # --------------------------------------------------
        if n_per_context == 1:
            z = torch.empty(
                (N,) + y_shape(self.cfg),
                device=device,
            )
            for arm in [0, 1]:
                idx = (a == arm).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    z[idx] = self.sample_base(arm, idx.numel())

            return integrator(self.velocity, z, context=context, steps=ode_steps)

        # --------------------------------------------------
        # K samples
        # --------------------------------------------------
        K = n_per_context
        z = torch.empty(
            (N, K) + y_shape(self.cfg),
            device=device,
        )

        for arm in [0, 1]:
            idx = (a == arm).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                base = self.sample_base(arm, idx.numel() * K)
                z[idx] = base.view(idx.numel(), K, *base.shape[1:])

        z_flat = z.reshape(N * K, *z.shape[2:])
        context_flat = context.unsqueeze(1).expand(N, K, context.shape[-1])
        context_flat = context_flat.reshape(N * K, context.shape[-1])

        y_flat = integrator(
            self.velocity,
            z_flat,
            context=context_flat,
            steps=ode_steps,
        )

        return y_flat.view(N, K, *y_flat.shape[1:])

    # ------------------------------------------------------------------
    # Observational / interventional
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_observational(
        self,
        X_reference: torch.Tensor,
        A_reference: torch.Tensor,
        n: int,
        **kwargs,
    ):
        idx = torch.randint(0, X_reference.shape[0], (n,), device=X_reference.device)
        return self.sample_conditional(
            X_reference[idx],
            A_reference[idx],
            **kwargs,
        )

    @torch.no_grad()
    def sample_interventional(
        self,
        a: int,
        X_reference: torch.Tensor,
        n: int,
        **kwargs,
    ):
        idx = torch.randint(0, X_reference.shape[0], (n,), device=X_reference.device)
        x = X_reference[idx]
        a = torch.full((n,), a, device=x.device)
        return self.sample_conditional(x, a, **kwargs)


# ======================================================================
# Utilities
# ======================================================================

def y_shape(cfg: ConditionalFlowFMConfig):
    if cfg.y_is_image:
        return (cfg.y_channels, cfg.y_height, cfg.y_width)
    return (cfg.dim_y,)
