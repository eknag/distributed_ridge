import datetime
from time import sleep

import numpy as np
import torch

from distributed_ridge.config import *


def make_correlated_X(
    n: int,
    d_in: int,
    corr_factor: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Generate an (n x d_in) matrix X with adjustable correlation among features.

    The correlation is controlled by corr_factor in [0, 1]:

      - corr_factor = 0  --> No correlation (X ~ N(0, I))
      - corr_factor = 1  --> Fully correlated by some random SPD matrix
      - 0 < corr_factor < 1 --> Linear interpolation between I and the random SPD

    Args:
        n: Number of samples (rows)
        d_in: Number of features (columns)
        corr_factor: Correlation factor in [0, 1]
        device: 'cpu' or 'cuda'
        dtype: e.g. torch.float32
        eps: Small diagonal for numerical stability in covariance

    Returns:
        X: torch.Tensor of shape (n, d_in), with correlated columns.
    """
    # Step 1: Generate base standard normal (n x d_in)
    Z = torch.randn(n, d_in, device=device, dtype=dtype)

    # If corr_factor is 0, we can skip everything and return Z directly.
    if corr_factor <= 0:
        return Z

    # Step 2: Create a random SPD matrix SPD of shape (d_in, d_in)
    A = torch.randn(d_in, d_in, device=device, dtype=dtype)
    SPD = A @ A.T  # Symmetric PSD

    # Normalize it so diagonal ~ 1.0 (like a correlation matrix)
    avg_diag = SPD.diagonal().mean()
    SPD /= avg_diag

    # Step 3: Interpolate between identity and SPD
    # Cov = (1 - corr_factor)*I + corr_factor*SPD
    I = torch.eye(d_in, device=device, dtype=dtype)
    Cov = (1 - corr_factor) * I + corr_factor * SPD

    # Add small eps to ensure positive definiteness
    Cov += eps * I

    # Step 4: Factor Cov (Cholesky).
    # L is lower-triangular, so Cov = L @ L.T
    L = torch.linalg.cholesky(Cov)

    # Step 5: Multiply Z by L => correlated data
    X = Z @ L

    return X


def get_data(date: datetime.datetime) -> tuple[torch.Tensor, torch.Tensor]:
    n = SAMPLES_PER_DAY  # number of samples for the given day (adjust as needed)
    d_in = D_IN  # number of input features
    d_out = D_OUT  # number of output features
    device = DEVICE
    dtype = DTYPE
    x_corr = X_CORR_FACTOR
    # For reproducibility, seed based on the date.
    torch.manual_seed(int(date.strftime("%Y%m%d")))

    X = make_correlated_X(n, d_in, x_corr, device=device, dtype=dtype)
    Y = X @ REAL_WEIGHTS + Y_RANDOM_NOISE_COEF * torch.randn(
        n, d_out, device=device, dtype=dtype
    )

    # sleep(5)  # Simulate data loading time
    return X, Y
