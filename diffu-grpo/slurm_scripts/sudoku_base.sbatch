#!/bin/bash
#SBATCH --output=slurm/%A/slurm.out
#SBATCH --job-name=sudoku-base
#SBATCH --time=72:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --open-mode=append

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --qos=storygen_high
#SBATCH --account=storygen
#SBATCH --cpus-per-task=50



export LOGDIR=checkpoints/slurm/${SLURM_JOB_ID}
mkdir -p $LOGDIR
echo $LOGDIR


DATASET="sudoku"
RUN_NAME=${DATASET}_base_bs12
MODEL_PATH=LLaDA-8B-Instruct
NUM_ITER=8
srun --output ${LOGDIR}/grpo_%j.out \
    accelerate launch \
        --config_file ./accelerate_a100.yaml \
        --main_process_port 12346 ../diffu_grpo_train.py \
        --config ./train.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $DATASET \
        --run_name $RUN_NAME \
        --output_dir checkpoints/$RUN_NAME
        --temperature 0.3
