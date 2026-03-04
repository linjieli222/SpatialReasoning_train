#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=16:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Convert EMA + eval TextCoT td_ego_dir checkpoints sequentially
# Usage: sbatch --job-name=textcot_ego_conv_eval scripts/slurm/convert_eval_textcot_ego_dir.slurm

BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_text_cot/textcot_td_ego_dir_8gpu
MODEL=/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT
SUBSETS="AI2ThorPT2P_td_ego_dir AI2ThorSV_td_ego_dir"

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

export HF_HOME=/gpfs/scrubbed/linjli/hf_cache

for STEP in 0001000 0002000 0003000 0004000 0005000; do
    CKPT_EMA=${BASE}/${STEP}_full_ema

    # Convert EMA if not already done
    if [ ! -f "${CKPT_EMA}/ema.safetensors" ]; then
        echo "=== $(date) Converting ${STEP} EMA ==="
        cd /gpfs/home/linjli/source/SpatialReasoning_train
        python scripts/convert_sharded_to_full.py \
            --checkpoint_path ${BASE}/${STEP} \
            --output_path ${CKPT_EMA} \
            --model_path ${MODEL} \
            --convert_ema \
            --no_visual_gen
        echo "=== $(date) Conversion done (exit $?) ==="
    else
        echo "=== $(date) ${STEP} EMA already converted, skipping ==="
    fi

    # Eval each subset
    export THINKMORPH_MODEL_PATH=${CKPT_EMA}
    cd /gpfs/home/linjli/source/SpatialReasoning_Eval

    for SUBSET in ${SUBSETS}; do
        MASTER_PORT=$((29500 + RANDOM % 10000))
        echo "=== $(date) Eval: step=${STEP} subset=${SUBSET} port=${MASTER_PORT} ==="
        torchrun --nproc-per-node=2 --master_port=${MASTER_PORT} run.py \
            --model bagel_mot \
            --data ${SUBSET} \
            --work-dir ${CKPT_EMA}/eval
        echo "=== $(date) Eval done (exit $?) ==="
    done
done

echo "=== $(date) All done ==="
