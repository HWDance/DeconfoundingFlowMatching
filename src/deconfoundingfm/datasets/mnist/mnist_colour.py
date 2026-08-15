# =========================================================
# Foreground-color-confounded MNIST DGP
# Adapted directly from the paper/research implementation.
# =========================================================

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Dict, Tuple, Literal

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# MNIST loading from IDX (filesystem or package data)
# =========================================================

def load_mnist_idx(
    images_path: str = "t10k-images.idx3-ubyte",
    labels_path: str = "t10k-labels.idx1-ubyte",
    *,
    package: str | None = None,
    device: torch.device | str = "cpu",
) -> Dict[int, torch.Tensor]:
    """Load the exact pre-downloaded MNIST t10k IDX files used in the research repo."""
    if package is not None:
        root = files(package)
        images_path = Path(root / images_path)
        labels_path = Path(root / labels_path)
    else:
        images_path = Path(images_path)
        labels_path = Path(labels_path)

    with open(images_path, "rb") as f:
        magic, n, rows, cols = np.frombuffer(f.read(16), dtype=">i4")
        assert magic == 2051
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)

    with open(labels_path, "rb") as f:
        magic, n_labels = np.frombuffer(f.read(8), dtype=">i4")
        assert magic == 2049
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    if int(n_labels) != int(n):
        raise RuntimeError("MNIST image/label IDX files have inconsistent lengths.")

    images = torch.from_numpy(images.copy()).float() / 255.0
    images = images.unsqueeze(1)
    labels = torch.from_numpy(labels.copy()).long()

    by_digit = {d: [] for d in range(10)}
    for img, lab in zip(images, labels):
        by_digit[int(lab.item())].append(img)

    return {
        d: torch.stack(v).to(device)
        for d, v in by_digit.items()
        if len(v) > 0
    }


# =========================================================
# Foreground / background recoloring (RGB)
# =========================================================

def recolor_foreground_background(
    S: torch.Tensor,
    *,
    fg_rgb,
    bg_rgb,
    tau: float = 0.08,
    k: float = 10.0,
) -> torch.Tensor:
    """Exact scalar-image recoloring function from the original CMNIST DGP."""
    if S.ndim == 3:
        S = S.squeeze(0)

    M = torch.sigmoid(k * (S - tau))
    M = M.unsqueeze(0)

    fg = torch.tensor(fg_rgb, device=S.device, dtype=S.dtype).view(3, 1, 1)
    bg = torch.tensor(bg_rgb, device=S.device, dtype=S.dtype).view(3, 1, 1)

    fg_img = fg * S.unsqueeze(0)
    Y = M * fg_img + (1.0 - M) * bg
    return Y.clamp(0.0, 1.0)


def recolor_foreground_background_batch(
    S: torch.Tensor,
    x: torch.Tensor,
    *,
    fg_alpha: float = 0.0,
    bg_rgb=(0.0, 0.0, 0.0),
    tau: float = 0.08,
    k: float = 10.0,
) -> torch.Tensor:
    """Vectorized equivalent of :func:`recolor_foreground_background`."""
    if S.ndim == 3:
        S = S.unsqueeze(1)
    if S.ndim != 4 or S.shape[1] != 1:
        raise ValueError("S must have shape (N,1,H,W) or (N,H,W).")
    x = x.reshape(-1).to(device=S.device, dtype=S.dtype)
    if len(x) != len(S):
        raise ValueError("x and S must contain the same number of samples.")

    M = torch.sigmoid(float(k) * (S - float(tau)))
    fg = torch.stack(
        [x, torch.full_like(x, float(fg_alpha)), 1.0 - x], dim=1
    ).view(-1, 3, 1, 1)
    bg = torch.tensor(bg_rgb, device=S.device, dtype=S.dtype).view(1, 3, 1, 1)
    fg_img = fg * S
    Y = M * fg_img + (1.0 - M) * bg
    return Y.clamp(0.0, 1.0)


# =========================================================
# Foreground colour mapping for confounder X
# =========================================================

def fg_color_from_x(x: float, alpha: float) -> Tuple[float, float, float]:
    return (float(x), float(alpha), float(1.0 - x))


BG_BLACK = (0.0, 0.0, 0.0)


def true_propensity(x: torch.Tensor, *, w: float, eps: float = 1e-2) -> torch.Tensor:
    return torch.sigmoid(float(w) * (x.reshape(-1) - 0.5)).clamp(float(eps), 1.0 - float(eps))


@torch.no_grad()
def sample_x_given_arm(
    arm: int,
    n: int,
    *,
    w: float,
    x_low: float = 0.0,
    x_high: float = 1.0,
    eps: float = 1e-2,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Draw exactly from X|A=arm for the original Uniform-X sigmoid assignment DGP."""
    arm = int(arm)
    if arm not in (0, 1):
        raise ValueError("arm must be 0 or 1.")
    device = torch.device(device)
    accepted = []
    need = int(n)
    while need > 0:
        m = max(64, 3 * need)
        x = torch.empty(m, device=device).uniform_(float(x_low), float(x_high))
        p1 = true_propensity(x, w=w, eps=eps)
        pa = p1 if arm == 1 else (1.0 - p1)
        keep = torch.rand(m, device=device) < pa
        if keep.any():
            take = x[keep][:need]
            accepted.append(take)
            need -= len(take)
    return torch.cat(accepted, dim=0)


# =========================================================
# Main DGP from the research code
# =========================================================

def generate_color_mnist_confounding(
    *,
    n_obs: int,
    n_ref: int,
    digits: Tuple[int, int] = (1, 9),
    x_low: float = 0.0,
    x_high: float = 1.0,
    w: float = 2.0,
    p_X: Literal["uniform"] = "uniform",
    device: torch.device | str = "cpu",
    seed: int = 0,
    tau: float = 0.08,
    k: float = 10.0,
    fg_alpha: float = 0.00,
    mnist_package: str | None = "deconfoundingfm.datasets.mnist",
) -> Dict[str, torch.Tensor]:
    """Original observational/reference CMNIST generator, namespace-adjusted only."""
    torch.manual_seed(seed)
    if p_X != "uniform":
        raise ValueError("Only the original p_X='uniform' design is supported.")

    mnist_by_digit = load_mnist_idx(package=mnist_package, device=device)
    d0, d1 = digits

    X = torch.empty(n_obs, device=device).uniform_(x_low, x_high)
    pi_true = true_propensity(X, w=w)
    A = torch.bernoulli(pi_true).long()

    Y = torch.empty((n_obs, 3, 28, 28), device=device)
    for i in range(n_obs):
        digit = d0 if A[i] == 0 else d1
        pool = mnist_by_digit[digit]
        S = pool[torch.randint(len(pool), (1,), device=device)].squeeze(0)
        fg_rgb = fg_color_from_x(float(X[i].item()), fg_alpha)
        Y[i] = recolor_foreground_background(
            S, fg_rgb=fg_rgb, bg_rgb=BG_BLACK, tau=tau, k=k
        )

    def sample_Y_ref(a: int):
        digit = d0 if a == 0 else d1
        pool = mnist_by_digit[digit]
        Xr = torch.empty(n_ref, device=device).uniform_(x_low, x_high)
        out = torch.empty((n_ref, 3, 28, 28), device=device)
        for i in range(n_ref):
            S = pool[torch.randint(len(pool), (1,), device=device)].squeeze(0)
            fg_rgb = fg_color_from_x(float(Xr[i].item()), fg_alpha)
            out[i] = recolor_foreground_background(
                S, fg_rgb=fg_rgb, bg_rgb=BG_BLACK, tau=tau, k=k
            )
        return out

    Y0_ref = sample_Y_ref(0)
    Y1_ref = sample_Y_ref(1)

    return dict(
        X=X.view(-1, 1),
        A=A.view(-1, 1),
        Y=Y,
        pi_true=pi_true,
        Y0_ref=Y0_ref,
        Y1_ref=Y1_ref,
        meta=dict(
            digits=digits,
            fg_alpha=float(fg_alpha),
            x_low=float(x_low),
            x_high=float(x_high),
            w=float(w),
            tau=float(tau),
            k=float(k),
            seed=int(seed),
        ),
    )


# =========================================================
# Fixed 20k observational population for the generator demo
# =========================================================

@torch.no_grad()
def generate_two_color_observational_population(
    *,
    n_bw_shapes: int = 10_000,
    color_draws_per_shape: int = 2,
    digits: Tuple[int, int] = (1, 6),
    x_low: float = 0.0,
    x_high: float = 1.0,
    w: float = 5.0,
    device: torch.device | str = "cpu",
    seed: int = 0,
    tau: float = 0.08,
    k: float = 10.0,
    fg_alpha: float = 0.0,
    mnist_package: str = "deconfoundingfm.datasets.mnist",
) -> Dict[str, torch.Tensor]:
    """Create a fixed observational population with repeated color draws per shape.

    The raw t10k IDX file contains all ten digits, while the CMNIST causal problem
    uses only the two treatment digits (1,6 by default). To preserve the original
    binary DGP *and* obtain exactly 20k observations, we first draw ``n_bw_shapes``
    grayscale shapes from the original arm-specific digit pools, then generate
    ``color_draws_per_shape`` independent X|A color draws for each fixed shape.

    With the defaults this is 10,000 grayscale shape draws x 2 colors = 20,000
    labeled observational examples.
    """
    if n_bw_shapes < 1 or color_draws_per_shape < 1:
        raise ValueError("n_bw_shapes and color_draws_per_shape must be positive.")
    torch.manual_seed(int(seed))
    device = torch.device(device)
    pools = load_mnist_idx(package=mnist_package, device=device)
    d0, d1 = map(int, digits)

    # Draw the arm attached to each grayscale shape from the same marginal induced
    # by X~Uniform and A|X~Bernoulli(pi(X)). The anchor X is only used to draw A.
    x_anchor = torch.empty(n_bw_shapes, device=device).uniform_(x_low, x_high)
    a_shape = torch.bernoulli(true_propensity(x_anchor, w=w)).long()

    shapes = torch.empty((n_bw_shapes, 1, 28, 28), device=device)
    source_pool_index = torch.empty(n_bw_shapes, dtype=torch.long, device=device)
    for arm, digit in ((0, d0), (1, d1)):
        pos = (a_shape == arm).nonzero(as_tuple=True)[0]
        pool = pools[digit]
        j = torch.randint(len(pool), (len(pos),), device=device)
        shapes[pos] = pool[j]
        source_pool_index[pos] = j

    r = int(color_draws_per_shape)
    X = torch.empty((n_bw_shapes, r), device=device)
    for arm in (0, 1):
        pos = (a_shape == arm).nonzero(as_tuple=True)[0]
        draws = sample_x_given_arm(
            arm,
            len(pos) * r,
            w=w,
            x_low=x_low,
            x_high=x_high,
            device=device,
        ).view(len(pos), r)
        X[pos] = draws

    S_rep = shapes[:, None].expand(n_bw_shapes, r, 1, 28, 28).reshape(-1, 1, 28, 28)
    X_flat = X.reshape(-1)
    A_flat = a_shape[:, None].expand(n_bw_shapes, r).reshape(-1)
    Y = recolor_foreground_background_batch(
        S_rep,
        X_flat,
        fg_alpha=fg_alpha,
        bg_rgb=BG_BLACK,
        tau=tau,
        k=k,
    )

    return dict(
        X=X_flat.view(-1, 1),
        A=A_flat.view(-1, 1),
        Y=Y,
        shape_id=torch.arange(n_bw_shapes, device=device)[:, None].expand(n_bw_shapes, r).reshape(-1),
        color_draw=torch.arange(r, device=device)[None, :].expand(n_bw_shapes, r).reshape(-1),
        source_pool_index=source_pool_index[:, None].expand(n_bw_shapes, r).reshape(-1),
        meta=dict(
            n_bw_shapes=int(n_bw_shapes),
            color_draws_per_shape=r,
            n_observations=int(n_bw_shapes * r),
            digits=tuple(map(int, digits)),
            x_low=float(x_low),
            x_high=float(x_high),
            w=float(w),
            tau=float(tau),
            k=float(k),
            fg_alpha=float(fg_alpha),
            seed=int(seed),
        ),
    )


class OracleSigmoidPropensity(nn.Module):
    """Retained only for DGP validation; the demo learner does not use it."""
    def __init__(self, *, w: float, eps: float = 1e-2):
        super().__init__()
        self.w = float(w)
        self.eps = float(eps)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return true_propensity(X, w=self.w, eps=self.eps)
