"""Propensity-score nuisance estimators used by DeconfoundingFM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold


class EmpiricalPropensityEstimator(nn.Module):
    """Empirical ``P(A=1|X)`` for one-dimensional discrete covariates."""

    def __init__(self, X: torch.Tensor, A: torch.Tensor, smoothing: float = 0.0):
        super().__init__()
        X = torch.as_tensor(X).reshape(-1).long()
        A = torch.as_tensor(A).reshape(-1).float()
        if len(X) != len(A):
            raise ValueError("X and A must have the same length.")
        uniques, inv = torch.unique(X, return_inverse=True)
        total = torch.bincount(inv)
        count1 = torch.bincount(inv, weights=A)
        p1 = (count1 + smoothing) / (total + 2.0 * smoothing)
        self.register_buffer("uniques", uniques)
        self.register_buffer("p1", p1)
        self.register_buffer("global_p1", A.mean().reshape(1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        original_device = X.device
        values = X.reshape(-1).long().to(self.uniques.device)
        mask = values[:, None] == self.uniques[None, :]
        idx = mask.float().argmax(dim=1)
        found = mask.any(dim=1)
        p = self.p1[idx].clone()
        p[~found] = self.global_p1
        return p.to(original_device)


class LogisticMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 32,
        depth: int = 1,
        dropout: float = 0.0,
        norm_in: bool = False,
    ):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim) if norm_in else None
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(d, hidden), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_in is not None:
            x = self.norm_in(x)
        return torch.sigmoid(self.net(x)).reshape(-1).clamp(1e-6, 1 - 1e-6)


@dataclass
class LogisticMLPConfig:
    in_dim: int = 1
    hidden: int = 64
    depth: int = 2
    dropout: float = 0.0
    norm_in: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 1000
    print_every: int = 200


class LogisticMLPPropensityEstimator(nn.Module):
    """Small neural propensity model with the callable interface expected by the target flow."""

    def __init__(self, cfg: LogisticMLPConfig, device: str | torch.device = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = LogisticMLP(
            cfg.in_dim, cfg.hidden, cfg.depth, cfg.dropout, cfg.norm_in
        ).to(self.device)
        self.history: dict[str, list[float]] = {}

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X.to(self.device)).to(X.device)

    def fit(self, X_train, A_train, X_val=None, A_val=None):
        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        A_train = torch.as_tensor(A_train, dtype=torch.float32, device=self.device).reshape(-1)
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        train_loss: list[float] = []
        val_auc: list[float] = []
        for ep in range(int(self.cfg.epochs)):
            self.model.train()
            opt.zero_grad(set_to_none=True)
            p = self.model(X_train)
            loss = F.binary_cross_entropy(p, A_train)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite propensity loss.")
            loss.backward()
            opt.step()
            train_loss.append(float(loss.detach()))
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    pv = self.model(torch.as_tensor(X_val, dtype=torch.float32, device=self.device))
                av = torch.as_tensor(A_val).reshape(-1).cpu().numpy()
                val_auc.append(float(roc_auc_score(av, pv.cpu().numpy())))
        self.history = {"train_loss": train_loss, "val_auc": val_auc}
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        return self


@dataclass
class RandomForestConfig:
    in_dim: int = 1
    n_estimators: int = 500
    max_depth: Optional[int] = 5
    min_samples_leaf: int = 1
    random_state: int = 123
    n_jobs: Optional[int] = None


class RandomForestPropensityEstimator(nn.Module):
    """Random-forest propensity estimator matching the paper implementation family."""

    def __init__(self, cfg: RandomForestConfig, device: str | torch.device = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        self.history: dict[str, float | int | None] = {}

    @staticmethod
    def _numpy_X(X) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("X must have shape (N,d_x).")
        return X

    @staticmethod
    def _numpy_A(A) -> np.ndarray:
        if isinstance(A, torch.Tensor):
            A = A.detach().cpu().numpy()
        A = np.asarray(A).reshape(-1).astype(int)
        if not set(np.unique(A)).issubset({0, 1}):
            raise ValueError("A must be binary in {0,1}.")
        return A

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        p1 = self.model.predict_proba(self._numpy_X(X))[:, 1]
        return torch.as_tensor(p1, dtype=X.dtype, device=X.device)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(self._numpy_X(X))

    def fit(self, X_train, A_train, X_val=None, A_val=None):
        Xtr, Atr = self._numpy_X(X_train), self._numpy_A(A_train)
        self.model.fit(Xtr, Atr)
        val_auc = None
        if X_val is not None:
            Xv, Av = self._numpy_X(X_val), self._numpy_A(A_val)
            val_auc = float(roc_auc_score(Av, self.model.predict_proba(Xv)[:, 1]))
        self.history = {"val_auc": val_auc, "max_depth": self.model.max_depth}
        return self

    def cross_validate(
        self,
        X,
        A,
        max_depths: Sequence[Optional[int]] = (1, 3, 5, 10),
        n_splits: int = 5,
    ):
        X_np, A_np = self._numpy_X(X), self._numpy_A(A)
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.cfg.random_state
        )
        base = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            min_samples_leaf=self.cfg.min_samples_leaf,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
        )
        grid = GridSearchCV(
            base,
            param_grid={"max_depth": list(max_depths)},
            scoring="roc_auc",
            cv=splitter,
            refit=True,
        )
        grid.fit(X_np, A_np)
        self.model = grid.best_estimator_
        self.history = {
            "val_auc": float(grid.best_score_),
            "max_depth": grid.best_params_["max_depth"],
        }
        return self
