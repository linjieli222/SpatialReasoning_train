#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Convert AO checkpoint from FSDP to safetensors
# Usage: sbatch --job-name=conv_ao_s12000 scripts/slurm/convert_ao_25k.slurm 0012000

STEP=${1:?Usage: sbatch convert_ao_25k.slurm <step_dir, e.g. 0012000>}
BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_answer_only/ao_td_path_8gpu
MODEL=/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

cd /gpfs/home/linjli/source/SpatialReasoning_train

echo "=== $(date) Converting ${STEP} noEMA ==="
python scripts/convert_sharded_to_full.py \
    --checkpoint_path ${BASE}/${STEP} \
    --output_path ${BASE}/${STEP}_full_noema \
    --model_path ${MODEL} \
    --no_visual_gen

echo "=== $(date) Converting ${STEP} EMA ==="
python scripts/convert_sharded_to_full.py \
    --checkpoint_path ${BASE}/${STEP} \
    --output_path ${BASE}/${STEP}_full_ema \
    --model_path ${MODEL} \
    --convert_ema \
    --no_visual_gen

echo "=== $(date) Done ==="
