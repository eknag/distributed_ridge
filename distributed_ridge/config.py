import datetime
import os

import torch

# --- Data caching configuration ---
CACHE_DIR = ".data_cache"
import hashlib

CONFIG_HASH = (
    int(hashlib.md5(open(__file__, "r").read().encode()).hexdigest(), 16) % 100000
)
# We cache the sufficient statistics (XTX and XTy) in an HDF5 file.
TRAIN_STATS_FILE = os.path.join(CACHE_DIR, f"train_sufficient_stats_{CONFIG_HASH}.h5")
VAL_STATS_FILE = os.path.join(CACHE_DIR, f"val_sufficient_stats_{CONFIG_HASH}.h5")


# --- Regularization grid configuration ---
NUM_LAMBDAS = 500  # Number of candidate λ values
LAMBDA_MIN = 1e-10  # Minimum λ value
LAMBDA_MAX = 1e10  # Maximum λ value
BATCH_SIZE = 100

# --- GPU / PyTorch configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
torch.manual_seed(42)

# just to initialize linalg module
torch.inverse(torch.ones((1, 1), device="cuda:0"))


# --- Data generation configuration ---
MAX_WORKERS = 16  # Maximum number of workers for ThreadPoolExecutor
SAMPLES_PER_DAY = 100000  # Number of samples per day
D_IN = 1000  # Number of input features
D_OUT = 100  # Number of outputs
REAL_WEIGHTS = torch.ones((D_IN, D_OUT), device=DEVICE, dtype=DTYPE) / (D_IN ** 0.5)
X_CORR_FACTOR = 0.0  # Correlation factor for input data
Y_RANDOM_NOISE_COEF = 0.0  # Coefficient for random noise


# --- Train / validation date ranges ---
TRAIN_START = datetime.datetime(2021, 1, 1)
TRAIN_END = datetime.datetime(2021, 4, 10)

VAL_START = datetime.datetime(2022, 1, 1)
VAL_END = datetime.datetime(2022, 1, 10)
