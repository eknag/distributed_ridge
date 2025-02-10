# Distributed Ridge

## About

This library uses GPUS to efficiently solve ridge regression on massive datasets

## Profiling

### batched eigen solve

find optimal lambda seaching over 100,000 candidates.

$160$ GB dataset
$365$ days * $100,000$ samples per day * ($1000$ d_in + $100$ d_out) * $4$ B per fp32
sufficient statistics in $4.73$ seconds
Solve $100,000$ times in closed form to find optimal lambda
Processed λ values in $2.97$ seconds


Profiling results on AMD Ryzen 7 3700X 8-Core Processor with RTX 3080

It shows EIGEN solver is the fastest in every case.

### CUDA Profiling other approaches
Solver | D_OUT | D_OUT | DTYPE | DEVICE | RUNTIME (s) | Speedup
-- | -- | -- | -- | -- | -- | --
Solver.BASELINE | Dims[ 1000 | 1000] | torch.float32 | cuda | 2.96 | 1
.Solver.LINEAR_SOLVER | Dims[ 1000 | 1000] | torch.float32 | cuda | 0.86 | 3
.Solver.INVERSE | Dims[ 1000 | 1000] | torch.float32 | cuda | 1.01 | 3
.Solver.EIGEN | Dims[ 1000 | 1000] | torch.float32 | cuda | 0.17 | **17**
.Solver.SVD | Dims[ 1000 | 1000] | torch.float32 | cuda | 0.20 | 14
.Solver.CHOLESKY | Dims[ 1000 | 1000] | torch.float32 | cuda | 0.73 | 4
.Solver.BASELINE | Dims[ 2000 | 500] | torch.float32 | cuda | 6.49 | 1
.Solver.LINEAR_SOLVER | Dims[ 2000 | 500] | torch.float32 | cuda | 2.45 | 3
.Solver.INVERSE | Dims[ 2000 | 500] | torch.float32 | cuda | 4.71 | 1
.Solver.EIGEN | Dims[ 2000 | 500] | torch.float32 | cuda | 0.30 | **22**
.Solver.SVD | Dims[ 2000 | 500] | torch.float32 | cuda | 0.55 | 12
.Solver.CHOLESKY | Dims[ 2000 | 500] | torch.float32 | cuda | 2.08 | 3
.Solver.BASELINE | Dims[  500 | 2000] | torch.float32 | cuda | 1.47 | 1
.Solver.LINEAR_SOLVER | Dims[  500 | 2000] | torch.float32 | cuda | 0.34 | 4
.Solver.INVERSE | Dims[  500 | 2000] | torch.float32 | cuda | 0.20 | 7
.Solver.EIGEN | Dims[  500 | 2000] | torch.float32 | cuda | 0.09 | **16**
.Solver.SVD | Dims[  500 | 2000] | torch.float32 | cuda | 0.10 | 15
.Solver.CHOLESKY | Dims[  500 | 2000] | torch.float32 | cuda | 0.34 | 4


## Quickstart


set config options in `distributed_ridge/config.py`

```bash
root@C.17345749:/workspace/distributed_ridge$ ./run.sh 
PyTorch version: 2.6.0+cu124
CUDA available: True
INFO:__main__:Found cached sufficient statistics.
Loaded sufficient statistics from .data_cache/train_sufficient_stats_72615.h5
INFO:__main__:Found cached sufficient statistics.
Loaded sufficient statistics from .data_cache/val_sufficient_stats_72615.h5
INFO:__main__:Loaded sufficient statistics in 0.01 seconds
INFO:__main__:Evaluating 500 λ values in batches...
INFO:__main__:batch best λ: 0.00000000, batch best MSE: -0.00000032
INFO:__main__:Current best λ: 0.00000000, MSE: -0.00000032
INFO:__main__:batch best λ: 0.00000102, batch best MSE: -0.00000032
INFO:__main__:Current best λ: 0.00000000, MSE: -0.00000032
INFO:__main__:batch best λ: 0.01037605, batch best MSE: -0.00000032
INFO:__main__:Current best λ: 0.00000000, MSE: -0.00000032
INFO:__main__:batch best λ: 384.74264526, batch best MSE: -0.00000048
INFO:__main__:Current best λ: 384.74264526, MSE: -0.00000048
INFO:__main__:batch best λ: 1076624.12500000, batch best MSE: 0.00943904
INFO:__main__:Current best λ: 384.74264526, MSE: -0.00000048
INFO:__main__:Processed λ values in 0.22 seconds
```

