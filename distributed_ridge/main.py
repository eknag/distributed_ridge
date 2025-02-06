import datetime
import logging
import os
import time
from typing import NamedTuple

import numpy as np
import torch

from distributed_ridge.compute_stats import compute_and_cache_stats
from distributed_ridge.config import *
from distributed_ridge.torch_solver import (evaluate_ridge_batch,
                                            get_lambda_grid, solve_ridge_batch)
from distributed_ridge.utils import load_sufficient_stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SufficientStats(NamedTuple):
    xTx: np.ndarray
    xTy: np.ndarray
    yTy: np.ndarray
    n: int


def load_stats(
    start: datetime.datetime, end: datetime.datetime, path: str
) -> SufficientStats:
    dates = [start + datetime.timedelta(days=i) for i in range((end - start).days + 1)]

    # Compute and cache sufficient statistics if they do not already exist.
    if not os.path.exists(path):
        logger.info("Sufficient statistics not found on disk. Computing...")
        compute_and_cache_stats(dates, path)
    else:
        logger.info("Found cached sufficient statistics.")

    # Load the sufficient statistics
    return load_sufficient_stats(path)


def main():
    try:

        start = time.time()

        xTx, xTy, _, _ = load_stats(TRAIN_START, TRAIN_END, TRAIN_STATS_FILE)
        xTx_val, xTy_val, yTy_val, n_val = load_stats(
            VAL_START, VAL_END, VAL_STATS_FILE
        )

        torch.cuda.synchronize()
        end = time.time()

        logger.info(f"Loaded sufficient statistics in {end - start:.2f} seconds")

        # Generate the λ grid
        lambdas = get_lambda_grid()

        logger.info(f"Evaluating {len(lambdas)} λ values in batches...")

        best_mse = float("inf")
        best_lambda = None
        best_weights = None

        start = time.time()
        for lambdas_batch, weights_batch in solve_ridge_batch(
            xTx, xTy, lambdas, batch_size=BATCH_SIZE
        ):
            mse_batch = evaluate_ridge_batch(
                xTx_val, xTy_val, yTy_val, n_val, weights_batch
            )

            # Update best values
            min_idx = torch.argmin(mse_batch)
            batch_min_mse = mse_batch[min_idx]

            if batch_min_mse < best_mse:
                best_mse = batch_min_mse.item()
                best_lambda = lambdas_batch[min_idx].item()
                best_weights = weights_batch[min_idx]

            logger.info(
                f"batch best λ: {lambdas_batch[min_idx]:.8f}, batch best MSE: {batch_min_mse:.8f}"
            )
            logger.info(f"Current best λ: {best_lambda:.8f}, MSE: {best_mse:.8f}")

        # Wait for GPU computations to complete
        torch.cuda.synchronize()

        end = time.time()
        logger.info(f"Processed λ values in {end - start:.2f} seconds")

    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
