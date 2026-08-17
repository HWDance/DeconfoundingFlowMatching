from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable

import torch
from torch.utils.data import DataLoader

from ..nuisance.outcome import ConditionalFlowFM, ConditionalFlowFMConfig
from ..core.target import DeconfoundingFlow, DeconfoundingFlowConfig
from ..core.data import DatasetDict


class GeneratorConditionalFlowFM(ConditionalFlowFM):
    """Conditional nuisance using a frozen arm-conditional generator as base.

    The source generator must implement ``sample(a: int, n: int, device=None)``.
    This allows the nuisance model to draw fresh observationally-biased samples
    online, while still fitting on a fixed observational dataset ``(X,A,Y)``.
    """

    def __init__(self, cfg: ConditionalFlowFMConfig, source_generator, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.source_generator = source_generator

    def sample_base(self, a: int, n: int) -> torch.Tensor:
        if self.cfg.base_kind == 'gaussian':
            return super().sample_base(a, n)
        device = next(self.parameters()).device
        y = self.source_generator.sample(int(a), int(n), device=device)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=torch.float32, device=device)
        y = y.to(device=device, dtype=next(self.parameters()).dtype)
        if float(self.cfg.base_noise_std) > 0:
            y = y + float(self.cfg.base_noise_std) * torch.randn_like(y)
        return y

    def fit(self, X, A, Y, batch_size: Optional[int]=None, epochs: Optional[int]=None, lr: Optional[float]=None, verbose: bool=True):
        batch_size = self.cfg.batch_size if batch_size is None else int(batch_size)
        epochs = self.cfg.epochs if epochs is None else int(epochs)
        lr = self.cfg.lr if lr is None else float(lr)
        device = next(self.velocity.parameters()).device
        X, A, Y = X.to(device), A.to(device), Y.to(device)
        if self.cfg.y_is_image:
            expected = (self.cfg.y_channels, self.cfg.y_height, self.cfg.y_width)
            if Y.ndim != 4 or tuple(Y.shape[1:]) != expected:
                raise ValueError(f"Expected image outcomes of shape (N,{expected}); got {tuple(Y.shape)}.")
        else:
            if Y.ndim == 1:
                Y = Y.unsqueeze(-1)
        # only retain empirical base if explicitly gaussian not requested and no generator provided
        if self.source_generator is None and self.cfg.base_kind == 'empirical':
            self.set_empirical_base(Y, A)
        loader = DataLoader(DatasetDict(X, A, Y), batch_size=min(batch_size, len(X)), shuffle=True)
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
                    raise FloatingPointError(f"Non-finite conditional-flow loss at epoch {epoch + 1}.")
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

    def fit_iterations(
        self,
        X,
        A,
        Y,
        *,
        iterations: int,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        verbose: bool = True,
    ):
        """Fit for an exact optimizer-update budget while drawing a fresh generator base each batch."""
        batch_size = self.cfg.batch_size if batch_size is None else int(batch_size)
        lr = self.cfg.lr if lr is None else float(lr)
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("iterations must be >= 1.")
        device = next(self.velocity.parameters()).device
        X, A, Y = X.to(device), A.to(device), Y.to(device)
        loader = DataLoader(DatasetDict(X, A, Y), batch_size=min(batch_size, len(X)), shuffle=True)
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=lr, weight_decay=self.cfg.weight_decay)
        self.train()
        iterator = iter(loader)
        history = []
        for step in range(iterations):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            opt.zero_grad(set_to_none=True)
            loss = self.fm_step(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite conditional-flow loss at iteration {step + 1}.")
            loss.backward()
            opt.step()
            loss_value = float(loss.detach())
            history.append(loss_value)
            if verbose and (step == 0 or (step + 1) % 1000 == 0 or step + 1 == iterations):
                print(f"Outcome nuisance iteration {step + 1}/{iterations} | loss={loss_value:.6f}")
        self.training_loss_ = history
        self.training_steps_ = iterations
        self.eval()
        return self


class GeneratorDeconfoundingFlow(DeconfoundingFlow):
    """Target DeconfoundingFM using a frozen arm-conditional generator as base."""

    def __init__(self, cfg: DeconfoundingFlowConfig, nuisance_outcome, nuisance_pi, source_generator, *args, **kwargs):
        super().__init__(cfg, nuisance_outcome, nuisance_pi, *args, **kwargs)
        self.source_generator = source_generator

    def set_source_shape(self, Y: torch.Tensor, A: torch.Tensor):
        device = next(self.velocity.parameters()).device
        Y = Y.to(device)
        A = A.to(device).long()
        if getattr(self.velocity, 'is_image', False):
            if Y.ndim != 4:
                raise ValueError('Image model expects Y of shape (N,C,H,W)')
            C, H, W = Y.shape[1:]
            self._image_shape[:] = torch.tensor([C, H, W], device=device)
        else:
            if Y.ndim == 1:
                Y = Y.unsqueeze(-1)
            self.dim_y = Y.shape[1]
        # keep tiny placeholders for downstream shape logic when needed
        self._base0 = Y[A.squeeze(-1) == 0][:1]
        self._base1 = Y[A.squeeze(-1) == 1][:1]

    def _sample_base(self, arm: int, n: int, like: torch.Tensor | None = None):
        if self.cfg.base_kind == 'gaussian':
            return super()._sample_base(arm, n, like=like)
        device = next(self.velocity.parameters()).device
        y = self.source_generator.sample(int(arm), int(n), device=device)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=torch.float32, device=device)
        y = y.to(device=device, dtype=next(self.velocity.parameters()).dtype)
        if float(self.cfg.base_noise_std) > 0:
            y = y + float(self.cfg.base_noise_std) * torch.randn_like(y)
        return y

    def fit(self, X: torch.Tensor, A: torch.Tensor, Y: torch.Tensor, *, verbose: bool=True, checkpoint_steps: Optional[Iterable[int]] = None):
        device = next(self.velocity.parameters()).device
        X, A, Y = X.to(device), A.to(device), Y.to(device)
        if X.shape[0] != A.shape[0] or X.shape[0] != Y.shape[0]:
            raise ValueError('X, A, and Y must have the same number of observations.')
        if self.cfg.base_kind not in {'empirical', 'gaussian'}:
            raise ValueError("base_kind must be 'empirical' or 'gaussian'.")
        if int(self.cfg.plugin_reservoir) < 1:
            raise ValueError('plugin_reservoir must be >= 1.')
        if int(self.cfg.plugin_batch) < 1:
            raise ValueError('plugin_batch must be >= 1.')
        if self.source_generator is not None and self.cfg.base_kind != 'gaussian':
            self.set_source_shape(Y, A)
        else:
            self.set_empirical_base(Y, A)
        self.precompute_pi(X)
        self.set_plugin_samples(X, A)
        dataset = DatasetDict(X, A, Y)
        loader = DataLoader(dataset, batch_size=min(int(self.cfg.batch_size), len(dataset)), shuffle=True)
        opt = torch.optim.Adam(self.velocity.parameters(), lr=self.cfg.lr)
        self.train()
        history = []
        checkpoint_set = set() if checkpoint_steps is None else {int(s) for s in checkpoint_steps}
        self.checkpoint_state_dicts_ = {}
        if self.cfg.iterations is not None:
            n_steps = int(self.cfg.iterations)
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
                    raise FloatingPointError(f'Non-finite DeconfoundingFM loss at iteration {step + 1}.')
                loss.backward()
                opt.step()
                loss_value = float(loss.detach())
                history.append(loss_value)
                current_step = step + 1
                if current_step in checkpoint_set:
                    self.checkpoint_state_dicts_[current_step] = {k: v.detach().cpu().clone() for k,v in self.velocity.state_dict().items()}
                if verbose and (step == 0 or (step + 1) % 1000 == 0 or step + 1 == n_steps):
                    print(f'DeconfoundingFM iteration {step + 1}/{n_steps} | loss={loss_value:.6f}')
                if (
                    self.cfg.update_plugin_reservoir
                    and current_step < n_steps
                    and current_step % int(self.cfg.plugin_reservoir_update_frequency) == 0
                ):
                    if verbose:
                        print(f"Refreshing plug-in reservoir after iteration {current_step}.")
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
                        raise FloatingPointError(f'Non-finite DeconfoundingFM loss at epoch {ep + 1}.')
                    loss.backward()
                    opt.step()
                    total += float(loss.detach())
                    n_batches += 1
                    total_steps += 1
                mean_loss = total / max(n_batches, 1)
                history.append(mean_loss)
            self.training_steps_ = total_steps
        self.training_loss_ = history
        return self
