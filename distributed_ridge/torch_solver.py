import numpy as np
import torch
from enum import Enum

from distributed_ridge.config import DEVICE, LAMBDA_MAX, LAMBDA_MIN, NUM_LAMBDAS


class RidgeSolver(Enum):
    BASELINE = "baseline"
    LINEAR_SOLVER = "linear_solver"
    INVERSE = "inverse"
    EIGEN = "eigen"
    SVD = "svd"
    CHOLESKY = "cholesky"


def solve_ridge(xTx, xTy, lambd, device=DEVICE):
    """
    Solve the ridge regression for a single λ value.

    Computes:
      w = (XᵀX + λI)⁻¹ XᵀY
    using the eigen-decomposition of XTX.

    Args:
      xTx: torch.tensor of shape (d_in, d_in)
      xTy: torch.tensor of shape (d_in, d_out)
      lambd: regularization parameter (a scalar)
      device: device to use ('cuda' or 'cpu')

    Returns:
      w: torch.Tensor of shape (d_in, d_out)
    """
    xTx = torch.tensor(xTx, device=device)
    xTy = torch.tensor(xTy, device=device)
    I = torch.eye(xTx.shape[0], device=device, dtype=xTx.dtype)
    return torch.linalg.solve(xTx + lambd * I, xTy)


def solve_ridge_batch_baseline(xTx, xTy, lambdas, batch_size=1000, device=DEVICE):
    with torch.no_grad():
        d_in, d_out = xTy.shape
        I = torch.eye(d_in, device=device, dtype=xTx.dtype)
        N = lambdas.shape[0]

        for i in range(0, N, batch_size):
            lambdas_batch = lambdas[i : i + batch_size]
            B = lambdas_batch.shape[0]

            res = []
            for j in range(B):
                lambdas_t = lambdas_batch[j]
                X = xTx + lambdas_t * I
                Y = xTy
                w = torch.linalg.solve(X, Y)
                res.append(w)

            yield lambdas_batch, torch.stack(res)


def solve_ridge_batch_linear_solver(xTx, xTy, lambdas, batch_size=1000, device=DEVICE):
    with torch.no_grad():
        d_in, d_out = xTy.shape

        xTx = xTx.unsqueeze(0)
        xTy = xTy.unsqueeze(0)

        I = torch.eye(d_in, device=device, dtype=xTx.dtype).unsqueeze(0)
        N = lambdas.shape[0]

        for i in range(0, N, batch_size):
            lambdas_batch = lambdas[i : i + batch_size]
            B = lambdas_batch.shape[0]

            lambdas_t = torch.tensor(
                lambdas_batch, device=device, dtype=xTx.dtype
            ).view(-1, 1, 1)

            X = xTx + lambdas_t * I
            Y = xTy.expand(X.size(0), d_in, d_out)
            w_batch = torch.linalg.solve(X, Y)

            yield lambdas_batch, w_batch


def solve_ridge_batch_inverse(xTx, xTy, lambdas, batch_size=1000, device=DEVICE):
    with torch.no_grad():
        lambdas = torch.tensor(lambdas, device=device, dtype=xTx.dtype)
        I = torch.eye(xTx.shape[0], device=device, dtype=xTx.dtype)
        N = lambdas.shape[0]

        for i in range(0, N, batch_size):
            lambdas_batch = lambdas[i : i + batch_size]
            B = lambdas_batch.shape[0]

            xTx_exp = xTx.unsqueeze(0).expand(B, -1, -1)
            I_exp = I.unsqueeze(0).expand(B, -1, -1)
            invs = torch.linalg.inv(xTx_exp + lambdas_batch.view(-1, 1, 1) * I_exp)

            weights = torch.matmul(invs, xTy.unsqueeze(0).expand(B, -1, -1))

            yield lambdas_batch, weights


def solve_ridge_batch_eigen(xTx, xTy, lambdas, batch_size=1000, device=DEVICE):
    with torch.no_grad():
        lambdas = torch.tensor(lambdas, device=device, dtype=xTx.dtype)
        S, Q = torch.linalg.eigh(xTx)
        Z = Q.T @ xTy

        N = lambdas.shape[0]

        for i in range(0, N, batch_size):
            lambdas_batch = lambdas[i : i + batch_size]
            B = lambdas_batch.shape[0]

            diag_inv = 1.0 / (S.unsqueeze(0) + lambdas_batch.unsqueeze(1))

            scaled_Z = diag_inv.unsqueeze(2) * Z.unsqueeze(0)

            weights = torch.matmul(Q.unsqueeze(0).expand(B, -1, -1), scaled_Z)

            yield lambdas_batch, weights


def solve_ridge_batch_svd(xTx, xTy, lambdas, batch_size=1000, device=DEVICE):
    with torch.no_grad():
        lambdas = torch.tensor(lambdas, device=device, dtype=xTx.dtype)
        U, S, V = torch.linalg.svd(xTx)
        Z = U.T @ xTy

        N = lambdas.shape[0]

        for i in range(0, N, batch_size):
            lambdas_batch = lambdas[i : i + batch_size]
            B = lambdas_batch.shape[0]

            diag_inv = 1.0 / (S.unsqueeze(0) + lambdas_batch.unsqueeze(1))

            scaled_Z = diag_inv.unsqueeze(2) * Z.unsqueeze(0)

            weights = torch.matmul(V.T.unsqueeze(0).expand(B, -1, -1), scaled_Z)

            yield lambdas_batch, weights


def solve_ridge_batch_cholesky(xTx, xTy, lambdas, batch_size=1000, device="cuda"):
    with torch.no_grad():
        lambdas_t = torch.as_tensor(lambdas, device=device, dtype=xTx.dtype)
        I = torch.eye(xTx.shape[0], device=device, dtype=xTx.dtype)

        N = lambdas_t.shape[0]
        for start in range(0, N, batch_size):
            lam_batch = lambdas_t[start : start + batch_size]
            B = lam_batch.size(0)

            mat_batch = xTx.unsqueeze(0).expand(B, -1, -1).clone()
            mat_batch += lam_batch.view(-1, 1, 1) * I

            L = torch.linalg.cholesky(mat_batch)

            rhs = xTy.unsqueeze(0).expand(B, -1, -1)
            w = torch.cholesky_solve(rhs, L, upper=False)

            yield lam_batch, w


def solve_ridge_batch(
    xTx, xTy, lambdas, solver=RidgeSolver.LINEAR_SOLVER, batch_size=1000, device=DEVICE
):
    """
    Solve ridge regression for a batch of λ values.

    Args:
      xTx: torch.tensor of shape (d_in, d_in)
      xTy: torch.tensor of shape (d_in, d_out)
      lambdas: torch.tensor of λ values, shape (N,)
      batch_size: number of λ values to process in one batch
      device: device to use ('cuda' or 'cpu')

    Yields:
      (lambdas_batch, weights_batch) where:
        - lambdas_batch: torch.Tensor of shape (B,)
        - weights_batch: torch.Tensor of shape (B, d_in, d_out)
    """
    with torch.no_grad():
        xTx = torch.as_tensor(xTx, device=device)
        xTy = torch.as_tensor(xTy, device=device)
        match solver:
            case RidgeSolver.BASELINE:
                yield from solve_ridge_batch_baseline(
                    xTx, xTy, lambdas, batch_size, device
                )
            case RidgeSolver.LINEAR_SOLVER:
                yield from solve_ridge_batch_linear_solver(
                    xTx, xTy, lambdas, batch_size, device
                )
            case RidgeSolver.INVERSE:
                yield from solve_ridge_batch_inverse(
                    xTx, xTy, lambdas, batch_size, device
                )
            case RidgeSolver.EIGEN:
                yield from solve_ridge_batch_eigen(
                    xTx, xTy, lambdas, batch_size, device
                )
            case RidgeSolver.SVD:
                yield from solve_ridge_batch_svd(xTx, xTy, lambdas, batch_size, device)
            case RidgeSolver.CHOLESKY:
                yield from solve_ridge_batch_cholesky(
                    xTx, xTy, lambdas, batch_size, device
                )
            case _:
                raise ValueError(f"Invalid solver: {solver}")


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
