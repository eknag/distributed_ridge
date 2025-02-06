import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Callable
import datetime
import logging

logger = logging.getLogger(__name__)


def process_date(date: datetime.datetime) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single date and return its statistics."""
    try:
        from my_api import get_data

        X, Y = get_data(date)
        return X.T @ X, X.T @ Y
    except Exception as e:
        logger.error(f"Error processing date {date}: {str(e)}")
        raise


def compute_sufficient_statistics(
    dates: List[datetime.datetime],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sufficient statistics in parallel using Python's multiprocessing.
    Returns:
    - XTX: The sum over days of XᵀX
    - XTy: The sum over days of XᵀY
    """
    logger.info(
        f"Computing statistics for {len(dates)} dates using {mp.cpu_count()} cores"
    )

    # Create a pool of workers
    with mp.Pool() as pool:
        # Map the process_date function over all dates
        results = pool.map(process_date, dates)

    # Combine results
    xTx_list, xTy_list = zip(*results)

    # Sum up all the statistics
    xTx = sum(xTx_list)
    xTy = sum(xTy_list)

    return xTx, xTy


def compute_and_cache_stats(dates: List[datetime.datetime]) -> None:
    """Compute the sufficient statistics and save them to disk."""
    from config import CACHE_DIR, STATS_FILE
    import os
    from utils import save_sufficient_stats

    logger.info("Starting computation of sufficient statistics...")
    try:
        xTx, xTy = compute_sufficient_statistics(dates)
        logger.info("Computation complete. Caching results...")

        os.makedirs(CACHE_DIR, exist_ok=True)
        save_sufficient_stats(xTx, xTy, STATS_FILE)

        logger.info(f"Successfully cached statistics to {STATS_FILE}")

    except Exception as e:
        logger.error(f"Error in compute_and_cache_stats: {str(e)}")
        raise
