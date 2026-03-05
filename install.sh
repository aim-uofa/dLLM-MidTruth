#!/bin/bash
# ==============================================================================
# Installation and Setup Script
#
# This script installs dependencies and sets up the directory structure
# for training and evaluation.
#
# Usage:
#   bash install.sh
# ==============================================================================

set -e

# ===================== Step 1: Install Dependencies =====================
echo "Installing dependencies..."
pip install -r requirements.txt

# Install the custom TRL fork (required for diffusion GRPO training)
if [ -d "third_parties/trl" ]; then
    cd third_parties/trl && pip install -e . && cd ../..
else
    echo "Warning: third_parties/trl not found. Please clone the required TRL fork."
    echo "See README.md for details."
fi

echo "Installation Done!"

# ===================== Step 2: Prepare Directories =====================
# Datasets are expected under dataset/ directory.
# Required datasets (download from HuggingFace):
#   - openai/gsm8k
#   - Jiayi-Pan/Countdown-Tasks-3to4
#   - ankner/math-500
#   - ChilleD/SVAMP
#   - simplescaling/s1K (for SFT)

# Model weights are expected under pretrained_weights/ directory.
# Required models:
#   - GSAI-ML/LLaDA-8B-Instruct
#   Download from: https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct

mkdir -p pretrained_weights

echo "Setup complete! See README.md for training instructions."
