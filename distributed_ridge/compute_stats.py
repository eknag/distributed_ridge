import datetime
import logging
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Tuple

import numpy as np
import torch
from distributed_ridge.data_api import get_data
from distributed_ridge.utils import save_sufficient_stats

from distributed_ridge.config import MAX_WORKERS

logger = logging.getLogger(__name__)


def process_date(
    date: datetime.datetime,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    X, Y = get_data(date)  # I/O
    with torch.no_grad():
        # Then do CPU matmul
        xTx = X.T @ X
        xTy = X.T @ Y
        yTy = Y.T @ Y
    return xTx, xTy, yTy, X.shape[0]


def compute_sufficient_statistics(dates, max_workers=4):
    xTx_total = None
    xTy_total = None
    yTy_total = None
    n_total = 0

    with torch.no_grad():
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_date, d) for d in dates]
            for f in as_completed(futures):
                xTx, xTy, yTy, n = f.result()
                if xTx_total is None:
                    xTx_total = xTx
                    xTy_total = xTy
                    yTy_total = yTy
                else:
                    xTx_total += xTx
                    xTy_total += xTy
                    yTy_total += yTy
                n_total += n

    return xTx_total, xTy_total, yTy_total, n_total


def compute_and_cache_stats(dates: List[datetime.datetime], path: str) -> None:
    """
    Compute the sufficient statistics and save them to disk.
    """
    logger.info("Starting computation of sufficient statistics...")
    try:
        xTx, xTy, yTy, n_total = compute_sufficient_statistics(
            dates, max_workers=MAX_WORKERS
        )
        logger.info("Computation complete. Caching results...")

        dir_ = os.path.dirname(path)
        os.makedirs(dir_, exist_ok=True)
        save_sufficient_stats(
            xTx.cpu().numpy(), xTy.cpu().numpy(), yTy.cpu().numpy(), n_total, path
        )

        logger.info(f"Successfully cached statistics to {path}")

    except Exception as e:
        logger.error(f"Error in compute_and_cache_stats: {str(e)}")
        raise
