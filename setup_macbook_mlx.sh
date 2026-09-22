#!/usr/bin/env bash
# ==============================================================================
# 🍎 Laya Apple Silicon MacBook & MLX Setup Script
# Works on macOS (M1, M2, M3, M4 series MacBooks with Unified Memory)
# ==============================================================================

set -e

echo "======================================================================"
echo " 🍎 LAYA APPLE SILICON MACBOOK & MLX SETUP"
echo "======================================================================"

# 1. Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required for Laya'"

# 2. Create virtual environment if not present
if [ ! -d "macbook_venv" ]; then
    echo "[1/4] Creating Python virtual environment in 'macbook_venv'..."
    python3 -m venv macbook_venv
else
    echo "[1/4] Virtual environment 'macbook_venv' already exists."
fi

source macbook_venv/bin/activate

# 3. Install Laya, Apple MLX, PyTorch (with MPS Metal support)
echo "[2/4] Installing Laya, Apple MLX, PyTorch, and dependencies..."
pip install --upgrade pip setuptools wheel
pip install torch transformers safetensors huggingface_hub numpy mlx

# 4. Install local laya package
pip install -e .

echo "[3/4] Testing Apple Silicon GPU (MPS) and MLX detection..."
python3 -c "
import torch
import mlx.core as mx

print('PyTorch Version       :', torch.__version__)
print('Apple MPS Available   :', torch.backends.mps.is_available())
print('Apple MLX Core Active :', mx.default_device())
"

echo "======================================================================"
echo " ✅ SETUP COMPLETE!"
echo "======================================================================"
echo "To run the notebook on your MacBook:"
echo "  1. Activate virtualenv : source macbook_venv/bin/activate"
echo "  2. Install jupyter    : pip install jupyterlab"
echo "  3. Launch notebook    : jupyter lab notebooks/laya_apple_macbook_mlx_setup.ipynb"
echo "======================================================================"
