#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Convert ego_dir checkpoint from FSDP to safetensors
# Usage: sbatch --job-name=conv_ego_ao_s1000 scripts/slurm/convert_ego_dir.slurm <base_dir> <step> [--no_visual_gen]
# Example AO:   sbatch scripts/slurm/convert_ego_dir.slurm /gpfs/.../ao_td_ego_dir_8gpu 0001000 --no_visual_gen
# Example VCoT: sbatch scripts/slurm/convert_ego_dir.slurm /gpfs/.../vcot_td_ego_dir_8gpu 0001000

BASE=${1:?Usage: sbatch convert_ego_dir.slurm <base_dir> <step> [--no_visual_gen]}
STEP=${2:?Usage: sbatch convert_ego_dir.slurm <base_dir> <step> [--no_visual_gen]}
EXTRA_FLAGS=${3:-}
MODEL=/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

cd /gpfs/home/linjli/source/SpatialReasoning_train

echo "=== $(date) Converting ${STEP} noEMA ==="
python scripts/convert_sharded_to_full.py \
    --checkpoint_path ${BASE}/${STEP} \
    --output_path ${BASE}/${STEP}_full_noema \
    --model_path ${MODEL} \
    ${EXTRA_FLAGS}

echo "=== $(date) Converting ${STEP} EMA ==="
python scripts/convert_sharded_to_full.py \
    --checkpoint_path ${BASE}/${STEP} \
    --output_path ${BASE}/${STEP}_full_ema \
    --model_path ${MODEL} \
    --convert_ema \
    ${EXTRA_FLAGS}

echo "=== $(date) Done ==="
