#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Convert MMCoT EMA + nothink eval on PT2PV2 and SV (sequentially)
# Usage: sbatch --job-name=mmcot_nothink_eval scripts/slurm/convert_eval_mmcot_ego_dir_nothink.slurm

BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_mmcot/mmcot_td_ego_dir_8gpu
MODEL=/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

export HF_HOME=/gpfs/scrubbed/linjli/hf_cache

for STEP in 0001000 0002000 0003000; do
    CKPT_EMA=${BASE}/${STEP}_full_ema

    # Convert EMA if not already done (no --no_visual_gen for MMCoT)
    if [ ! -f "${CKPT_EMA}/ema.safetensors" ]; then
        echo "=== $(date) Converting ${STEP} EMA ==="
        cd /gpfs/home/linjli/source/SpatialReasoning_train
        python scripts/convert_sharded_to_full.py \
            --checkpoint_path ${BASE}/${STEP} \
            --output_path ${CKPT_EMA} \
            --model_path ${MODEL} \
            --convert_ema
        echo "=== $(date) Conversion done (exit $?) ==="
    else
        echo "=== $(date) ${STEP} EMA already converted, skipping ==="
    fi

    export THINKMORPH_MODEL_PATH=${CKPT_EMA}
    cd /gpfs/home/linjli/source/SpatialReasoning_Eval

    for SUBSET in AI2ThorPT2PV2_td_ego_dir AI2ThorSV_td_ego_dir; do
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
