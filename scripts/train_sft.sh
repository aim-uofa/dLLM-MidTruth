#!/bin/bash
# ==============================================================================
# SFT (Supervised Fine-Tuning) Training Script
# 
# This script fine-tunes a diffusion language model (e.g., LLaDA-8B-Instruct)
# using the s1K dataset with LoRA adapters.
#
# Usage:
#   bash scripts/train_sft.sh
#
# Prerequisites:
#   - Set up the environment following the README
#   - Download the model weights (e.g., GSAI-ML/LLaDA-8B-Instruct)
#   - Download the s1K dataset to dataset/simplescaling/s1K
# ==============================================================================

set -e

# ===================== Configuration =====================
# Model
MODEL_PATH="pretrained_weights/GSAI-ML/LLaDA-8B-Instruct"  # Path to the base model

# Training hyperparameters
LR=1e-5
BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NUM_EPOCHS=20
EVAL_STEPS=100
SAVE_STEPS=100

# Experiment name
EXP_NAME="llada-s1_lr${LR}"

# Number of GPUs
NUM_GPUS=8

# ===================== Run Training =====================
cd SFT

accelerate launch \
    --config_file ddp_config.yaml \
    --main_process_port 29500 \
    --num_processes ${NUM_GPUS} \
    sft_train.py \
    --output_dir outputs/sft \
    --job_name ${EXP_NAME} \
    --grad_accum_steps ${GRAD_ACCUM_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --learning_rate ${LR} \
    --model_name ../${MODEL_PATH}
