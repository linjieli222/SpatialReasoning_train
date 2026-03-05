#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Eval with answer-only prompt + visual generation (bagel_mot_nothink_vcot)
# Usage: sbatch --job-name=nothink_vcot_mixed_s1k scripts/slurm/eval_nothink_vcot.slurm <ckpt_path> <subset>

CKPT=${1:?Usage: sbatch eval_nothink_vcot.slurm <ckpt_path> <subset>}
SUBSET=${2:?Usage: sbatch eval_nothink_vcot.slurm <ckpt_path> <subset>}

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

echo "=== $(date) Eval (nothink_vcot): model=${CKPT} subset=${SUBSET} config=bagel_mot_nothink_vcot port=${MASTER_PORT} gpus=4 ==="
torchrun --nproc-per-node=4 --master_port=${MASTER_PORT} run.py \
    --model bagel_mot_nothink_vcot \
    --data ${SUBSET} \
    --work-dir ${CKPT}/eval
echo "=== $(date) Done (exit $?) ==="
