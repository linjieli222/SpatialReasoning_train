#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Generic eval: any model config on any subset
# Usage: sbatch --job-name=<name> scripts/slurm/eval_generic.slurm <ckpt_path> <subset> <model_config>
# Example: sbatch scripts/slurm/eval_generic.slurm .../0001000_full_ema AI2ThorPT2P_td_ego_dir bagel_mot_nothink

CKPT=${1:?Usage: eval_generic.slurm <ckpt_path> <subset> <model_config>}
SUBSET=${2:?Usage: eval_generic.slurm <ckpt_path> <subset> <model_config>}
MODEL_CONFIG=${3:?Usage: eval_generic.slurm <ckpt_path> <subset> <model_config>}

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

export HF_HOME=/gpfs/scrubbed/linjli/hf_cache

cd /gpfs/home/linjli/source/SpatialReasoning_Eval

export THINKMORPH_MODEL_PATH=${CKPT}
MASTER_PORT=$((29500 + RANDOM % 10000))

echo "=== $(date) Eval: model=${CKPT} subset=${SUBSET} config=${MODEL_CONFIG} port=${MASTER_PORT} ==="
torchrun --nproc-per-node=2 --master_port=${MASTER_PORT} run.py \
    --model ${MODEL_CONFIG} \
    --data ${SUBSET} \
    --work-dir ${CKPT}/eval
echo "=== $(date) Done (exit $?) ==="
