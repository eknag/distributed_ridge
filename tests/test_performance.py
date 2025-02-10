import time
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
import gc

# Test configurations
DIMENSIONS = [(1000, 1000), (2000, 500), (500, 2000)]  # (D_IN, D_OUT) pairs
DTYPES = [torch.float32]
DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.fixture(params=DIMENSIONS)
def dimensions(request):
    return request.param


@pytest.fixture(params=DTYPES)
def dtype(request):
    return request.param


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


@pytest.fixture
def sufficient_stats(dtype, dimensions, device):
    """
    Fixture to generate dummy sufficient statistics for testing using torch.
    """
    d_in, d_out = dimensions
    x = torch.rand(SAMPLES_PER_DAY, d_in, dtype=dtype, device=device)
    y = torch.rand(SAMPLES_PER_DAY, d_out, dtype=dtype, device=device)
    xTx = x.T @ x
    xTy = x.T @ y
    return xTx, xTy, d_in, d_out


@pytest.mark.parametrize("solver", list(RidgeSolver))
def test_solve_ridge_batch_performance(sufficient_stats, solver, device):
    """
    Test the runtime of solve_ridge_batch for each RidgeSolver, dimension, dtype and device.
    """
    xTx, xTy, d_in, d_out = sufficient_stats
    lambdas = get_lambda_grid(num_lambdas=1000)

    if device == "cuda":
        total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
        dtype_size = xTx.element_size()
        xTx_size = xTx.numel() * dtype_size
        xTy_size = xTy.numel() * dtype_size
        total_memory = xTx_size + xTy_size
        batch_size = total_gpu_memory // (total_memory * 4)
    else:
        batch_size = 100

    initial_memory = torch.cuda.memory_allocated(device) if device == "cuda" else 0

    start_time = time.time()
    for lambdas_batch, weights_batch in solve_ridge_batch(
        xTx, xTy, lambdas, solver=solver, batch_size=batch_size, device=device
    ):
        del lambdas_batch, weights_batch

    torch.cuda.synchronize()
    end_time = time.time()
    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated(device) if device == "cuda" else 0
    memory_diff = final_memory - initial_memory

    elapsed_time = end_time - start_time
    print(
        f"Solver.{solver.name:<15} "
        f"Dims[{d_in:>5}, {d_out:>5}], "
        f"{str(xTx.dtype):<13}, "
        f"{device:<4}, "
        f"Time: {elapsed_time:>7.4f} s"
    )

    assert elapsed_time > 0, f"Solver {solver.name} took non-positive time"
    assert memory_diff == 0, f"Memory leak detected: {memory_diff} bytes"
