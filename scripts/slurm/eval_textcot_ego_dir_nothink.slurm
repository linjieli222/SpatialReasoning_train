#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Eval TextCoT td_ego_dir EMA checkpoints with nothink config (sequentially)
# Usage: sbatch --dependency=afterok:<job_id> --job-name=textcot_ego_nothink scripts/slurm/eval_textcot_ego_dir_nothink.slurm

BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_text_cot/textcot_td_ego_dir_8gpu
SUBSETS="AI2ThorPT2P_td_ego_dir AI2ThorSV_td_ego_dir"

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

export HF_HOME=/gpfs/scrubbed/linjli/hf_cache

cd /gpfs/home/linjli/source/SpatialReasoning_Eval

for STEP in 0001000 0002000 0003000 0004000 0005000; do
    CKPT_EMA=${BASE}/${STEP}_full_ema
    export THINKMORPH_MODEL_PATH=${CKPT_EMA}

    for SUBSET in ${SUBSETS}; do
        MASTER_PORT=$((29500 + RANDOM % 10000))
        echo "=== $(date) Eval nothink: step=${STEP} subset=${SUBSET} port=${MASTER_PORT} ==="
        torchrun --nproc-per-node=2 --master_port=${MASTER_PORT} run.py \
            --model bagel_mot_nothink \
            --data ${SUBSET} \
            --work-dir ${CKPT_EMA}/eval
        echo "=== $(date) Eval done (exit $?) ==="
    done
done

echo "=== $(date) All done ==="
