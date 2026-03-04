#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Eval TextCoT td_ego_dir EMA checkpoints on PT2PV2 (think + nothink, sequentially)
# Usage: sbatch --dependency=afterany:JOBID --job-name=textcot_pt2pv2 scripts/slurm/eval_textcot_ego_dir_pt2pv2.slurm

BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_text_cot/textcot_td_ego_dir_8gpu

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

export HF_HOME=/gpfs/scrubbed/linjli/hf_cache

cd /gpfs/home/linjli/source/SpatialReasoning_Eval

for STEP in 0001000 0002000 0003000 0004000 0005000 0006000; do
    CKPT_EMA=${BASE}/${STEP}_full_ema
    [ -d "${CKPT_EMA}" ] || { echo "=== $(date) ${STEP} not converted, skipping ==="; continue; }
    export THINKMORPH_MODEL_PATH=${CKPT_EMA}

    for MODEL_CONFIG in bagel_mot bagel_mot_nothink; do
        MASTER_PORT=$((29500 + RANDOM % 10000))
        echo "=== $(date) Eval: step=${STEP} subset=AI2ThorPT2PV2_td_ego_dir config=${MODEL_CONFIG} port=${MASTER_PORT} ==="
        torchrun --nproc-per-node=2 --master_port=${MASTER_PORT} run.py \
            --model ${MODEL_CONFIG} \
            --data AI2ThorPT2PV2_td_ego_dir \
            --work-dir ${CKPT_EMA}/eval
        echo "=== $(date) Eval done (exit $?) ==="
    done
done

echo "=== $(date) All done ==="
