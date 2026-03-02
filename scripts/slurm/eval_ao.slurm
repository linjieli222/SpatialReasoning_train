#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Eval AO checkpoint on a single subset
# Usage: sbatch --job-name=eval_ao_s12000_noema_td_path scripts/slurm/eval_ao.slurm <ckpt_path> <subset>
# Example: sbatch scripts/slurm/eval_ao.slurm .../0012000_full_noema AI2ThorPT2P_td_path

CKPT=${1:?Usage: sbatch eval_ao.slurm <ckpt_path> <subset>}
SUBSET=${2:?Usage: sbatch eval_ao.slurm <ckpt_path> <subset>}

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

cd /gpfs/home/linjli/source/SpatialReasoning_train
source scripts/preflight.sh
preflight_eval_check "${CKPT}" "${SUBSET}" "bagel_mot"

cd /gpfs/home/linjli/source/SpatialReasoning_Eval

export THINKMORPH_MODEL_PATH=${CKPT}
MASTER_PORT=$((29500 + RANDOM % 10000))

echo "=== $(date) Eval: model=${CKPT} subset=${SUBSET} port=${MASTER_PORT} ==="
torchrun --nproc-per-node=2 --master_port=${MASTER_PORT} run.py \
    --model bagel_mot \
    --data ${SUBSET} \
    --work-dir ${CKPT}/eval
echo "=== $(date) Done (exit $?) ==="
