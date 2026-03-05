#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Nothink eval on PT2PV2 for VCoT td_ego_dir EMA checkpoints (sequentially)
# Skips checkpoints that already have results.

BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_vcot_l32/vcot_td_ego_dir_8gpu

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

export HF_HOME=/gpfs/scrubbed/linjli/hf_cache

cd /gpfs/home/linjli/source/SpatialReasoning_Eval

for STEP in 0001000 0003000 0004000 0005000 0006000 0008000; do
    CKPT_EMA=${BASE}/${STEP}_full_ema
    [ -d "${CKPT_EMA}" ] || continue

    # Skip if already done
    if ls ${CKPT_EMA}/eval/bagel_mot_nothink/*PT2PV2*_acc.csv 2>/dev/null | grep -q .; then
        echo "=== $(date) ${STEP} already has nothink PT2PV2, skipping ==="
        continue
    fi

    export THINKMORPH_MODEL_PATH=${CKPT_EMA}
    MASTER_PORT=$((29500 + RANDOM % 10000))
    echo "=== $(date) Eval nothink PT2PV2: step=${STEP} port=${MASTER_PORT} ==="
    torchrun --nproc-per-node=2 --master_port=${MASTER_PORT} run.py \
        --model bagel_mot_nothink \
        --data AI2ThorPT2PV2_td_ego_dir \
        --work-dir ${CKPT_EMA}/eval
    echo "=== $(date) Eval done (exit $?) ==="
done

echo "=== $(date) All done ==="
