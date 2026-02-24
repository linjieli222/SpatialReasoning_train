#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1600G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Generic TIFA v3 training launcher for Tillicum
# Usage: sbatch --job-name=<name> [--time=HH:MM:SS] scripts/slurm/train_tifa_v3.slurm <training_script.sh>
#
# Examples:
#   sbatch --job-name=p1_td_path_ao scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_path_answer_only.sh
#   sbatch --job-name=p2_td_path_l64 --time=24:00:00 scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_path_cot_l64.sh

TRAIN_SCRIPT=${1:?Usage: sbatch train_tifa_v3.slurm <training_script.sh>}

mkdir -p slurm_logs

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

cd /gpfs/home/linjli/source/SpatialReasoning_train

# Redirect wandb to scrubbed storage (home quota too small)
export WANDB_DIR=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/wandb
export WANDB_CACHE_DIR=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/wandb/.cache
mkdir -p $WANDB_DIR $WANDB_CACHE_DIR

echo "=== $(date) Starting: $TRAIN_SCRIPT ==="
bash "$TRAIN_SCRIPT"
echo "=== $(date) Finished: $TRAIN_SCRIPT (exit $?) ==="
