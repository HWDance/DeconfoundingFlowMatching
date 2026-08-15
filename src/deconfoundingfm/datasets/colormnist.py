"""ColorMNIST population sampler used by the image demonstration.

The data-generating process matches the ColorMNIST experiment in the research
repository:

* two MNIST digit classes encode the binary treatment;
* ``X ~ Uniform(x_low, x_high)`` controls the foreground colour;
* ``A | X=x ~ Bernoulli(sigmoid(w * (x - 1/2)))`` (with clipping);
* foreground RGB is ``(x, fg_alpha, 1-x)`` on a black background.

Unlike the finite-sample experiment helper, :class:`ColorMNISTPopulation`
exposes sampling primitives that can generate fresh population minibatches on
demand.  It is intended for simulation/demo use, not as a real-data loader for
the causal estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ColorMNISTConfig:
    digits: tuple[int, int] = (1, 6)
    x_low: float = 0.0
    x_high: float = 1.0
    confounding_w: float = 5.0
    propensity_clip: float = 1e-2
    tau: float = 0.08
    mask_sharpness: float = 10.0
    fg_alpha: float = 0.0


class OracleColorMNISTPropensity(nn.Module):
    """Oracle ``P(A=1|X)`` for the ColorMNIST DGP."""

    def __init__(self, *, w: float, eps: float = 1e-2):
        super().__init__()
        self.w = float(w)
        self.eps = float(eps)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = X.reshape(-1)
        p = torch.sigmoid(self.w * (x - 0.5))
        return p.clamp(self.eps, 1.0 - self.eps)


class ColorMNISTPopulation:
    """Fresh-minibatch population sampler for the paper's ColorMNIST DGP.

    Parameters
    ----------
    digit_pools:
        Mapping from digit label to grayscale tensors of shape ``(N,1,H,W)``
        with values in ``[0,1]``.  Only the two labels in ``config.digits`` are
        required.
    config:
        DGP parameters.
    device:
        Device on which sampling and recolouring are performed.  Keeping the
        digit pools on the GPU makes repeated population minibatch generation
        inexpensive.
    """

    def __init__(
        self,
        digit_pools: Mapping[int, torch.Tensor],
        *,
        config: ColorMNISTConfig | None = None,
        device: torch.device | str = "cpu",
    ):
        self.config = ColorMNISTConfig() if config is None else config
        self.device = torch.device(device)

        d0, d1 = self.config.digits
        missing = [d for d in (d0, d1) if d not in digit_pools]
        if missing:
            raise ValueError(f"Missing MNIST digit pools: {missing}")

        pools: dict[int, torch.Tensor] = {}
        shape = None
        for digit in (d0, d1):
            pool = torch.as_tensor(digit_pools[digit]).float()
            if pool.ndim == 3:
                pool = pool.unsqueeze(1)
            if pool.ndim != 4 or pool.shape[1] != 1:
                raise ValueError(
                    "Each digit pool must have shape (N,1,H,W) or (N,H,W)."
                )
            if pool.shape[0] < 1:
                raise ValueError(f"Digit pool {digit} is empty.")
            this_shape = tuple(pool.shape[1:])
            if shape is None:
                shape = this_shape
            elif this_shape != shape:
                raise ValueError("All digit pools must share the same image shape.")
            pools[digit] = pool.to(self.device)

        self.digit_pools = pools
        _, self.height, self.width = shape  # grayscale pool shape is (1,H,W)
        self.image_shape = (3, self.height, self.width)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_torchvision(
        cls,
        root: str | Path,
        *,
        config: ColorMNISTConfig | None = None,
        train: bool = False,
        download: bool = False,
        device: torch.device | str = "cpu",
    ) -> "ColorMNISTPopulation":
        """Load the same MNIST split through torchvision.

        The research experiment used the raw ``t10k`` IDX files, which
        corresponds to ``torchvision.datasets.MNIST(train=False)``.
        """
        try:
            from torchvision.datasets import MNIST
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "ColorMNIST requires torchvision. Install the demo extra with "
                "`pip install -e '.[cmnist]'`."
            ) from exc

        dataset = MNIST(root=str(root), train=train, download=download)
        data = dataset.data.float().div_(255.0).unsqueeze(1)
        labels = dataset.targets.long()
        cfg = ColorMNISTConfig() if config is None else config
        pools = {digit: data[labels == digit] for digit in cfg.digits}
        return cls(pools, config=cfg, device=device)

    @classmethod
    def from_idx(
        cls,
        images_path: str | Path,
        labels_path: str | Path,
        *,
        config: ColorMNISTConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> "ColorMNISTPopulation":
        """Load MNIST directly from the raw IDX files used by the experiment."""
        images_path = Path(images_path)
        labels_path = Path(labels_path)
        with images_path.open("rb") as handle:
            magic, n, rows, cols = np.frombuffer(handle.read(16), dtype=">i4")
            if int(magic) != 2051:
                raise ValueError("Invalid MNIST image IDX magic number.")
            images = np.frombuffer(handle.read(), dtype=np.uint8).reshape(n, rows, cols)
        with labels_path.open("rb") as handle:
            magic, n_labels = np.frombuffer(handle.read(8), dtype=">i4")
            if int(magic) != 2049 or int(n_labels) != int(n):
                raise ValueError("Invalid MNIST label IDX file.")
            labels = np.frombuffer(handle.read(), dtype=np.uint8)

        data = torch.from_numpy(images.copy()).float().div_(255.0).unsqueeze(1)
        labels_t = torch.from_numpy(labels.copy()).long()
        cfg = ColorMNISTConfig() if config is None else config
        pools = {digit: data[labels_t == digit] for digit in cfg.digits}
        return cls(pools, config=cfg, device=device)

    # ------------------------------------------------------------------
    # DGP pieces
    # ------------------------------------------------------------------
    def oracle_propensity(self) -> OracleColorMNISTPropensity:
        return OracleColorMNISTPropensity(
            w=self.config.confounding_w,
            eps=self.config.propensity_clip,
        ).to(self.device)

    def propensity(self, X: torch.Tensor) -> torch.Tensor:
        x = X.reshape(-1)
        p = torch.sigmoid(self.config.confounding_w * (x - 0.5))
        return p.clamp(
            self.config.propensity_clip,
            1.0 - self.config.propensity_clip,
        )

    def sample_x(self, n: int) -> torch.Tensor:
        if int(n) < 1:
            raise ValueError("n must be >= 1.")
        x = torch.empty(int(n), 1, device=self.device)
        return x.uniform_(self.config.x_low, self.config.x_high)

    def _sample_shapes(self, arm: int, n: int) -> torch.Tensor:
        if int(arm) not in (0, 1):
            raise ValueError("arm must be 0 or 1.")
        digit = self.config.digits[int(arm)]
        pool = self.digit_pools[digit]
        idx = torch.randint(pool.shape[0], (int(n),), device=self.device)
        return pool.index_select(0, idx)

    def _recolor(self, shapes: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # Same smooth foreground mask as the research DGP, vectorized over B.
        if shapes.ndim != 4 or shapes.shape[1] != 1:
            raise ValueError("shapes must have shape (B,1,H,W).")
        x = X.reshape(-1, 1, 1, 1).to(shapes.dtype)
        s = shapes
        mask = torch.sigmoid(
            self.config.mask_sharpness * (s - self.config.tau)
        )
        red = x
        green = torch.full_like(x, float(self.config.fg_alpha))
        blue = 1.0 - x
        fg = torch.cat([red, green, blue], dim=1)
        fg_img = fg * s
        # The experiment uses a black background, so the background term is 0.
        return (mask * fg_img).clamp_(0.0, 1.0)

    def sample_outcome_given_x_arm(self, X: torch.Tensor, arm: int) -> torch.Tensor:
        X = X.to(self.device)
        shapes = self._sample_shapes(arm, X.shape[0])
        return self._recolor(shapes, X)

    def sample_observational_given_x(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Draw fresh ``A,Y`` at supplied population covariates ``X``."""
        X = X.to(self.device)
        p = self.propensity(X)
        A = torch.bernoulli(p).long()
        Y = torch.empty(
            X.shape[0], *self.image_shape, device=self.device, dtype=torch.float32
        )
        for arm in (0, 1):
            mask = A == arm
            n_arm = int(mask.sum().item())
            if n_arm:
                Y[mask] = self.sample_outcome_given_x_arm(X[mask], arm)
        return {"X": X, "A": A.view(-1, 1), "Y": Y, "pi_true": p}

    def sample_observational(self, n: int) -> dict[str, torch.Tensor]:
        return self.sample_observational_given_x(self.sample_x(n))

    def sample_interventional(self, arm: int, n: int) -> dict[str, torch.Tensor]:
        X = self.sample_x(n)
        Y = self.sample_outcome_given_x_arm(X, arm)
        return {"X": X, "Y": Y}

    def sample_source(self, arm: int, n: int) -> dict[str, torch.Tensor]:
        """Draw fresh samples from the observational source ``P(Y|A=arm)``.

        Rejection sampling is exact for the clipped assignment mechanism and is
        efficient here because the symmetric DGP has treatment mass near 1/2.
        """
        if int(arm) not in (0, 1):
            raise ValueError("arm must be 0 or 1.")
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1.")

        accepted: list[torch.Tensor] = []
        total = 0
        # Oversample by ~2x under the symmetric treatment mechanism.
        while total < n:
            remaining = n - total
            m = max(64, 3 * remaining)
            X = self.sample_x(m)
            A = torch.bernoulli(self.propensity(X)).long()
            keep = A == int(arm)
            if keep.any():
                x_keep = X[keep][:remaining]
                accepted.append(x_keep)
                total += x_keep.shape[0]

        X_arm = torch.cat(accepted, dim=0)[:n]
        Y_arm = self.sample_outcome_given_x_arm(X_arm, arm)
        return {"X": X_arm, "Y": Y_arm}
