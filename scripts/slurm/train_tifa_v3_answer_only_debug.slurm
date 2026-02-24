#!/bin/bash
#SBATCH --job-name=tifa_v3_ao_debug
#SBATCH --qos=normal
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1600G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/tifa_v3_answer_only_debug_%j.out

# TIFA v3 dh_midpoint answer-only debug training on Tillicum
# 8 GPUs, 100 steps, ~2hr time limit

# Create log directory
mkdir -p slurm_logs

# Load conda environment
module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

# Set working directory
cd /gpfs/home/linjli/source/SpatialReasoning_train

# Wandb config - set your API key or run `wandb login` beforehand
# export WANDB_API_KEY=your_key_here
# Uncomment the next line if compute nodes have no internet:
# export WANDB_MODE=offline

# Redirect wandb data to scrubbed storage to avoid home directory disk quota issues
export WANDB_DIR=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/wandb
export WANDB_CACHE_DIR=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/wandb/.cache
mkdir -p $WANDB_DIR $WANDB_CACHE_DIR

# Run training
bash scripts/train_tifa_v3_answer_only_debug.sh
