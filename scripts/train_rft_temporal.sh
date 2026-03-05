#!/bin/bash
# ==============================================================================
# RFT with Temporal Reward Only (Temporal Consistency Reinforcement)
#
# This script trains a diffusion language model using GRPO with only the 
# Temporal Semantic Entropy (TSE) reward signal (no ground-truth accuracy reward).
# This corresponds to the "Temporal Reward Only" setting in the paper.
#
# Usage:
#   bash scripts/train_rft_temporal.sh
#
# Prerequisites:
#   - Set up the environment following the README
#   - Download the model weights (e.g., GSAI-ML/LLaDA-8B-Instruct)
#   - (Optional) Run SFT first and set SFT_PATH to the checkpoint
# ==============================================================================

set -e

# ===================== Configuration =====================
# Model
MODEL_PATH="pretrained_weights/GSAI-ML/LLaDA-8B-Instruct"

# SFT checkpoint (set to the path of your SFT checkpoint, or comment out to train without SFT)
SFT_PATH="outputs/sft/llada-s1_lr1e-5/checkpoint-620"

# Dataset: choose from "countdown", "gsm8k", "math", "svamp", "sudoku"
DATASET="countdown"

# GRPO hyperparameters
NUM_ITERATIONS=12       # Number of policy gradient inner update iterations (μ)
NUM_GENERATIONS=6       # Number of generations per prompt (group size)
LR=3e-6                 # Learning rate

# Generation settings
MAX_LEN=256             # Maximum completion length
DIFFUSION_STEPS=128     # Number of diffusion denoising steps

# Temporal reward settings (TSE-only mode)
TEMPORAL_REWARD_ONLY=1          # Use only temporal reward (no accuracy reward)
TEMPORAL_REWARD_WEIGHT=2.0      # Weight for the temporal reward
TEMPORAL_REWARD_TYPE="exp"      # Temporal weighting: "fixed", "linear", or "exp"
WITH_FORMAT_REWARD=0            # No format reward in temporal-only mode
REMOVE_GT_REWARD=0              # Not applicable in temporal-only mode

# Entropy calculation settings
ENTROPY_INCLUDE_NONE=0          # Whether to include None (unparseable) in entropy calculation
ENTROPY_SKIP_STEPS_RATIO=0.5   # Skip first 50% of denoising steps for entropy
ENTROPY_EXP_ALPHA=12.8          # Exponential decay factor for time-weighted entropy

# Prompt type (0: default, 1: alternative with boxed format)
PROMPT_TYPE=0

# Number of GPUs
NUM_GPUS=8

# ===================== Construct Run Name =====================
RUN_NAME="${DATASET}_rft_temporal_len${MAX_LEN}steps${DIFFUSION_STEPS}_${TEMPORAL_REWARD_TYPE}weight${TEMPORAL_REWARD_WEIGHT}_alpha${ENTROPY_EXP_ALPHA}_skips${ENTROPY_SKIP_STEPS_RATIO}_lr${LR}"

# ===================== Run Training =====================
cd diffu-grpo

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 29501 \
    --num_processes ${NUM_GPUS} \
    diffu_grpo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path ../${MODEL_PATH} \
    --num_generations ${NUM_GENERATIONS} \
    --num_iterations ${NUM_ITERATIONS} \
    --dataset ${DATASET} \
    --run_name ${RUN_NAME} \
    --output_dir outputs/grpo/${RUN_NAME} \
    --pretrained_sft_lora_path ${SFT_PATH} \
    --max_completion_length ${MAX_LEN} \
    --diffusion_steps ${DIFFUSION_STEPS} \
    --temporal_reward_only ${TEMPORAL_REWARD_ONLY} \
    --temporal_reward_weight ${TEMPORAL_REWARD_WEIGHT} \
    --temporal_reward_type ${TEMPORAL_REWARD_TYPE} \
    --with_format_reward ${WITH_FORMAT_REWARD} \
    --prompt_type ${PROMPT_TYPE} \
    --entropy_include_none ${ENTROPY_INCLUDE_NONE} \
    --entropy_skip_steps_ratio ${ENTROPY_SKIP_STEPS_RATIO} \
    --remove_gt_reward ${REMOVE_GT_REWARD} \
    --learning_rate ${LR} \
    --entropy_exp_alpha ${ENTROPY_EXP_ALPHA}
