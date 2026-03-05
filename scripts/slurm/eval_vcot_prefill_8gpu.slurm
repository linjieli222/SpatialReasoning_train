#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Eval VCoT prefill with 8 GPUs: GT sideview images injected as visual thoughts
# Usage: sbatch --job-name=eval_prefill_8gpu_s3k scripts/slurm/eval_vcot_prefill_8gpu.slurm <ckpt_path> <subset>

CKPT=${1:?Usage: sbatch eval_vcot_prefill_8gpu.slurm <ckpt_path> <subset>}
SUBSET=${2:?Usage: sbatch eval_vcot_prefill_8gpu.slurm <ckpt_path> <subset>}

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

export HF_HOME=/gpfs/scrubbed/linjli/hf_cache

cd /gpfs/home/linjli/source/SpatialReasoning_train
source scripts/preflight.sh
preflight_eval_check "${CKPT}" "${SUBSET}" "bagel_mot_vcot_prefill"

cd /gpfs/home/linjli/source/SpatialReasoning_Eval

export THINKMORPH_MODEL_PATH=${CKPT}
MASTER_PORT=$((29500 + RANDOM % 10000))

echo "=== $(date) Eval (8GPU prefill): model=${CKPT} subset=${SUBSET} config=bagel_mot_vcot_prefill port=${MASTER_PORT} ==="
torchrun --nproc-per-node=8 --master_port=${MASTER_PORT} run.py \
    --model bagel_mot_vcot_prefill \
    --data ${SUBSET} \
    --work-dir ${CKPT}/eval
echo "=== $(date) Done (exit $?) ==="
