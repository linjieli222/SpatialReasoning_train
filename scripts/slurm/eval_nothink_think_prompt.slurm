#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Eval with bagel_mot_nothink_think_prompt (VCoT system prompt + answer-only instruction)
# Usage: sbatch --job-name=eval_nttp_s3k_ego scripts/slurm/eval_nothink_think_prompt.slurm <ckpt_path> <subset>

CKPT=${1:?Usage: sbatch eval_nothink_think_prompt.slurm <ckpt_path> <subset>}
SUBSET=${2:?Usage: sbatch eval_nothink_think_prompt.slurm <ckpt_path> <subset>}

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

echo "=== $(date) Eval (nothink_think_prompt): model=${CKPT} subset=${SUBSET} config=bagel_mot_nothink_think_prompt port=${MASTER_PORT} ==="
torchrun --nproc-per-node=2 --master_port=${MASTER_PORT} run.py \
    --model bagel_mot_nothink_think_prompt \
    --data ${SUBSET} \
    --work-dir ${CKPT}/eval
echo "=== $(date) Done (exit $?) ==="
