# Distributed Ridge

## About

This library uses GPUS to efficiently solve ridge regression on massive datasets

## Benchmark

$160$ GB dataset

$365$ days * $100,000$ samples per day * ($1000$ d_in + $100$ d_out) * $4$ B per fp32

sufficient statistics in $4.73$ seconds

Solve $100,000$ times in closed form to find optimal lambda

Processed λ values in $2.97$ seconds

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

