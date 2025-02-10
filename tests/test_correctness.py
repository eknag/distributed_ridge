import pytest
import numpy as np
import torch
from distributed_ridge.torch_solver import (
    evaluate_ridge_batch,
    solve_ridge_batch,
    RidgeSolver,
    get_lambda_grid,
)
from distributed_ridge.config import SAMPLES_PER_DAY

# Test configurations
TEST_CONFIGS = [
    {"d_in": 10, "d_out": 5},
    {"d_in": 100, "d_out": 20},
    {"d_in": 50, "d_out": 50},
]

TEST_DEVICES = ["cpu"]
if torch.cuda.is_available():
    TEST_DEVICES.append("cuda")


@pytest.fixture(params=TEST_CONFIGS)
def dimensions(request):
    return request.param


@pytest.fixture(params=TEST_DEVICES)
def device(request):
    return torch.device(request.param)


@pytest.fixture
def sufficient_stats(dimensions, dtype: np.dtype = np.float32):
    """
    Fixture to generate dummy sufficient statistics for testing.
    """
    d_in, d_out = dimensions["d_in"], dimensions["d_out"]
    x = np.random.rand(SAMPLES_PER_DAY, d_in).astype(dtype)
    y = np.random.rand(SAMPLES_PER_DAY, d_out).astype(dtype)
    xTx = x.T @ x
    xTy = x.T @ y
    yTy = y.T @ y
    return xTx, xTy, yTy


@pytest.mark.parametrize(
    "solver", [s for s in RidgeSolver if s != RidgeSolver.BASELINE]
)
def test_solver_matches_baseline(sufficient_stats, solver, device, dimensions):
    """
    Test that each solver's results match the baseline solver within tolerance.
    """
    xTx, xTy, yTy = sufficient_stats
    lambdas = get_lambda_grid(num_lambdas=100)  # Smaller grid for testing

    initial_memory = torch.cuda.memory_allocated(device) if device == "cuda" else 0

    # Get baseline results
    baseline_mses = []
    for lambdas_batch, weights_batch in solve_ridge_batch(
        xTx, xTy, lambdas, solver=RidgeSolver.BASELINE, batch_size=10, device=device
    ):
        mse_batch = evaluate_ridge_batch(xTx, xTy, yTy, SAMPLES_PER_DAY, weights_batch)
        baseline_mses.append(mse_batch)
    baseline_mses = torch.cat(baseline_mses)

    # Get solver results
    solver_mses = []
    for lambdas_batch, weights_batch in solve_ridge_batch(
        xTx, xTy, lambdas, solver=solver, batch_size=10, device=device
    ):
        mse_batch = evaluate_ridge_batch(xTx, xTy, yTy, SAMPLES_PER_DAY, weights_batch)
        solver_mses.append(mse_batch)
    solver_mses = torch.cat(solver_mses)

    # Compare results
    torch.testing.assert_close(
        solver_mses,
        baseline_mses,
        rtol=1e-4,
        atol=1e-4,
        msg=f"{solver.name} results do not match baseline (device={device}, d_in={dimensions['d_in']}, d_out={dimensions['d_out']})",
    )

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated(device)
        memory_diff = final_memory - initial_memory
        assert memory_diff == 0, f"Memory leak detected: {memory_diff} bytes"
