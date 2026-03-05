#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Convert checkpoint from FSDP to safetensors (EMA only)
# Usage: sbatch --job-name=conv_mmcot_s7000 scripts/slurm/convert_ema_only.slurm <base_dir> <step> [extra_flags]

BASE=${1:?Usage: sbatch convert_ema_only.slurm <base_dir> <step> [extra_flags]}
STEP=${2:?Usage: sbatch convert_ema_only.slurm <base_dir> <step> [extra_flags]}
EXTRA_FLAGS=${3:-}
MODEL=/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

cd /gpfs/home/linjli/source/SpatialReasoning_train

echo "=== $(date) Converting ${STEP} EMA ==="
python scripts/convert_sharded_to_full.py \
    --checkpoint_path ${BASE}/${STEP} \
    --output_path ${BASE}/${STEP}_full_ema \
    --model_path ${MODEL} \
    --convert_ema \
    ${EXTRA_FLAGS}

echo "=== $(date) Done ==="
