import numpy as np
import torch

from distributed_ridge.config import (DEVICE, LAMBDA_MAX, LAMBDA_MIN,
                                      NUM_LAMBDAS)


def solve_ridge(xTx, xTy, lambd, device=DEVICE):
    """
    Solve the ridge regression for a single λ value.

    Computes:
      w = (XᵀX + λI)⁻¹ XᵀY
    using the eigen-decomposition of XTX.

    Args:
      xTx: numpy.ndarray of shape (d_in, d_in)
      xTy: numpy.ndarray of shape (d_in, d_out)
      lambd: regularization parameter (a scalar)
      device: device to use ('cuda' or 'cpu')

    Returns:
      w: torch.Tensor of shape (d_in, d_out)
    """
    xTx_t = torch.tensor(xTx, device=device)
    xTy_t = torch.tensor(xTy, device=device)
    I = torch.eye(xTx_t.shape[0], device=device, dtype=xTx_t.dtype)
    return torch.linalg.solve(xTx_t + lambd * I, xTy_t)


def solve_ridge_batch(xTx, xTy, lambdas, batch_size=1000, device=DEVICE):
    """
    Solve ridge regression for a batch of λ values.

    Args:
      xTx: numpy.ndarray of shape (d_in, d_in)
      xTy: numpy.ndarray of shape (d_in, d_out)
      lambdas: numpy.ndarray of λ values, shape (N,)
      batch_size: number of λ values to process in one batch
      device: device to use ('cuda' or 'cpu')

    Yields:
      (lambdas_batch, weights_batch) where:
        - lambdas_batch: torch.Tensor of shape (B,)
        - weights_batch: torch.Tensor of shape (B, d_in, d_out)
    """
    xTx_t = torch.tensor(xTx, device=device)
    xTy_t = torch.tensor(xTy, device=device)
    lambdas = torch.tensor(lambdas, device=device, dtype=xTx_t.dtype)
    S, Q = torch.linalg.eigh(xTx_t)  # S: (d_in,), Q: (d_in, d_in)
    Z = Q.T @ xTy_t  # (d_in, d_out)

    N = lambdas.shape[0]

    for i in range(0, N, batch_size):
        lambdas_batch = lambdas[i : i + batch_size]  # (B,)
        B = lambdas_batch.shape[0]

        # Compute the diagonal inverse for each λ in the batch
        diag_inv = 1.0 / (S.unsqueeze(0) + lambdas_batch.unsqueeze(1))  # (B, d_in)

        # Scale Z (of shape (d_in, d_out)) by diag_inv for each λ
        scaled_Z = diag_inv.unsqueeze(2) * Z.unsqueeze(0)  # (B, d_in, d_out)

        # Compute the ridge regression solution for each λ
        weights = torch.matmul(
            Q.unsqueeze(0).expand(B, -1, -1), scaled_Z
        )  # (B, d_in, d_out)

        yield lambdas_batch, weights


def get_lambda_grid(
    num_lambdas=NUM_LAMBDAS, lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX
):
    """
    Return a numpy array of λ values (logarithmically spaced).
    """
    return np.logspace(np.log10(lambda_min), np.log10(lambda_max), num=num_lambdas)


def evaluate_ridge_batch(
    xTx_val, xTy_val, yTy_val, n_val, weights_batch, device=DEVICE
):
    """
    Evaluate MSE for a batch of ridge regression weights on validation data.

    MSE = (1 / n_val) * [trace(y^T y) - 2 trace(w^T X^T y) + trace(w^T X^T X w)]
    """
    xTx_val = torch.as_tensor(xTx_val, device=device)
    xTy_val = torch.as_tensor(xTy_val, device=device)
    yTy_val = torch.as_tensor(yTy_val, device=device)
    weights_batch = torch.as_tensor(weights_batch, device=device)

    d_out = xTy_val.shape[1]

    # w' X' y
    wxTy = torch.sum(weights_batch * xTy_val.unsqueeze(0), dim=(1, 2))  # (batch_size,)

    # w' X' X w
    xTxw = torch.matmul(xTx_val.unsqueeze(0), weights_batch)
    wxTxw = torch.sum(weights_batch * xTxw, dim=(1, 2))  # (batch_size,)

    # Only sum the diagonal of y^T y (the total sum of squares across all outputs)
    yTy_trace = torch.trace(yTy_val)

    # MSE per sample (across all output dims)
    mse_batch = (yTy_trace - 2 * wxTy + wxTxw) / (n_val * d_out)

    return mse_batch
