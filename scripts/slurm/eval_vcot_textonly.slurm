#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Eval VCoT checkpoint with text-only config (bagel_mot) — no image generation
# This intentionally skips the preflight config check since we want to test
# VCoT checkpoints in understanding-only mode.
# Usage: sbatch --job-name=eval_vcot_textonly_s7000_noema scripts/slurm/eval_vcot_textonly.slurm <ckpt_path> <subset>

CKPT=${1:?Usage: sbatch eval_vcot_textonly.slurm <ckpt_path> <subset>}
SUBSET=${2:?Usage: sbatch eval_vcot_textonly.slurm <ckpt_path> <subset>}

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

export HF_HOME=/gpfs/scrubbed/linjli/hf_cache

if [ ! -d "${CKPT}" ]; then
    echo "[FAIL] Checkpoint not found: ${CKPT}"
    exit 1
fi

cd /gpfs/home/linjli/source/SpatialReasoning_Eval

export THINKMORPH_MODEL_PATH=${CKPT}
MASTER_PORT=$((29500 + RANDOM % 10000))

echo "=== $(date) Eval (text-only): model=${CKPT} subset=${SUBSET} config=bagel_mot port=${MASTER_PORT} ==="
torchrun --nproc-per-node=2 --master_port=${MASTER_PORT} run.py \
    --model bagel_mot \
    --data ${SUBSET} \
    --work-dir ${CKPT}/eval
echo "=== $(date) Done (exit $?) ==="
