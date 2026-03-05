#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Convert VCoT l64 checkpoint from FSDP to safetensors
# Usage: sbatch --job-name=conv_vcot_s9000 scripts/slurm/convert_vcot.slurm 0009000

STEP=${1:?Usage: sbatch convert_vcot.slurm <step_dir>}
BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_cot_l64/vcot_td_path_l64_8gpu
MODEL=/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

cd /gpfs/home/linjli/source/SpatialReasoning_train

echo "=== $(date) Converting ${STEP} noEMA ==="
python scripts/convert_sharded_to_full.py \
    --checkpoint_path ${BASE}/${STEP} \
    --output_path ${BASE}/${STEP}_full_noema \
    --model_path ${MODEL}

echo "=== $(date) Converting ${STEP} EMA ==="
python scripts/convert_sharded_to_full.py \
    --checkpoint_path ${BASE}/${STEP} \
    --output_path ${BASE}/${STEP}_full_ema \
    --model_path ${MODEL} \
    --convert_ema

echo "=== $(date) Done ==="
