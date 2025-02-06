#!/bin/bash

# Ensure script is run from the script's directory
cd "$(dirname "$0")"

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Set PYTHONPATH to include the project directory
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Print environment info
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Add the distributed_ridge directory to PYTHONPATH
export PYTHONPATH="${PWD}/distributed_ridge:${PYTHONPATH}"
# Run the main script
OMP_NUM_THREADS=1 python distributed_ridge/main.py