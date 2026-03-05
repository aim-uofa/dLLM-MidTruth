#!/bin/bash
# ==============================================================================
# RFT with Temporal + Accuracy Reward (Temporal Consistency Reinforcement)
#
# This script trains a diffusion language model using GRPO with both the 
# Temporal Semantic Entropy (TSE) reward and ground-truth accuracy reward,
# combined using a proper scoring rule (e.g., Brier, Spherical).
# This corresponds to the "Temporal + Accuracy Reward" setting in the paper.
#
# Usage:
#   bash scripts/train_rft_temporal_acc.sh
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

# Temporal + Accuracy reward settings
TEMPORAL_REWARD_ONLY=1                  # Start with temporal-only base (accuracy added via temporal_reward_with_gt)
TEMPORAL_REWARD_WEIGHT=2.0              # Weight for the combined temporal+accuracy reward
TEMPORAL_REWARD_TYPE="exp"              # Temporal weighting: "fixed", "linear", or "exp"
WITH_FORMAT_REWARD=0                    # No separate format reward
REMOVE_GT_REWARD=0                      # Not applicable (gt is combined via temporal_reward_with_gt)
TEMPORAL_REWARD_WITH_GT=1               # Enable combined temporal + ground-truth accuracy reward
TEMPORAL_GT_COMBINE_METHOD="spherical_plus"  # Combining method: "fixed", "brier", "logarithmic", "spherical", "logarithmic_plus", "spherical_plus"

# Entropy calculation settings
ENTROPY_INCLUDE_NONE=0          # Whether to include None (unparseable) in entropy calculation
ENTROPY_SKIP_STEPS_RATIO=0.5   # Skip first 50% of denoising steps for entropy
ENTROPY_EXP_ALPHA=12.8          # Exponential decay factor for time-weighted entropy

# Prompt type (0: default, 1: alternative with boxed format)
PROMPT_TYPE=0

# Token sampling (0: random masking, 1: token sampling)
USE_TOKEN_SAMPLING=0

# Number of GPUs
NUM_GPUS=8

# ===================== Construct Run Name =====================
RUN_NAME="${DATASET}_rft_temporal_acc_len${MAX_LEN}steps${DIFFUSION_STEPS}_${TEMPORAL_REWARD_TYPE}weight${TEMPORAL_REWARD_WEIGHT}_alpha${ENTROPY_EXP_ALPHA}_skips${ENTROPY_SKIP_STEPS_RATIO}_gt${TEMPORAL_GT_COMBINE_METHOD}_lr${LR}"

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
    --entropy_exp_alpha ${ENTROPY_EXP_ALPHA} \
    --use_token_sampling ${USE_TOKEN_SAMPLING} \
    --temporal_reward_with_gt ${TEMPORAL_REWARD_WITH_GT} \
    --temporal_gt_combine_method ${TEMPORAL_GT_COMBINE_METHOD}
