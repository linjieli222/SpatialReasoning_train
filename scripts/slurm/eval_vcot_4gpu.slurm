#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Eval VCoT/MMCoT checkpoint on a single subset with 4 GPUs (faster)
# Usage: sbatch --job-name=eval_vcot4_s9000_noema_td_path scripts/slurm/eval_vcot_4gpu.slurm <ckpt_path> <subset>

CKPT=${1:?Usage: sbatch eval_vcot_4gpu.slurm <ckpt_path> <subset>}
SUBSET=${2:?Usage: sbatch eval_vcot_4gpu.slurm <ckpt_path> <subset>}

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

cd /gpfs/home/linjli/source/SpatialReasoning_train
source scripts/preflight.sh
preflight_eval_check "${CKPT}" "${SUBSET}" "bagel_mot_vcot"

cd /gpfs/home/linjli/source/SpatialReasoning_Eval

export THINKMORPH_MODEL_PATH=${CKPT}
MASTER_PORT=$((29500 + RANDOM % 10000))

echo "=== $(date) Eval: model=${CKPT} subset=${SUBSET} config=bagel_mot_vcot port=${MASTER_PORT} gpus=4 ==="
torchrun --nproc-per-node=4 --master_port=${MASTER_PORT} run.py \
    --model bagel_mot_vcot \
    --data ${SUBSET} \
    --work-dir ${CKPT}/eval
echo "=== $(date) Done (exit $?) ==="
