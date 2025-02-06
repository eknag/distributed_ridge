import os

import h5py
import numpy as np


def save_sufficient_stats(
    xTx: np.ndarray, xTy: np.ndarray, yTy: np.ndarray, n: int, filename: str
):
    """Save the sufficient statistics (XTX and XTy) into an HDF5 file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with h5py.File(filename, "w") as f:
        f.create_dataset("xTx", data=xTx, dtype=xTx.dtype)
        f.create_dataset("xTy", data=xTy, dtype=xTy.dtype)
        f.create_dataset("yTy", data=yTy, dtype=yTy.dtype)
        f.create_dataset("n", data=n, dtype=np.int64)
    print(f"Saved sufficient statistics to {filename}")


def load_sufficient_stats(
    filename: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load the sufficient statistics from an HDF5 file."""
    with h5py.File(filename, "r") as f:
        xTx = f["xTx"][:]
        xTy = f["xTy"][:]
        yTy = f["yTy"][:]
        n = f["n"][()]
    print(f"Loaded sufficient statistics from {filename}")
    return xTx, xTy, yTy, n
