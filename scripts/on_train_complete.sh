#!/bin/bash
# on_train_complete.sh - Auto-launch eval when a SpatialReasoning training run finishes.
#
# Called by: trainlog sync (via --on-complete hook)
# Receives env vars from trainlog:
#   TRAINLOG_ID, TRAINLOG_NAME, TRAINLOG_STATUS, TRAINLOG_OUTPUT_DIR,
#   TRAINLOG_CHECKPOINT (last checkpoint path), TRAINLOG_CONFIG_FILE,
#   TRAINLOG_SCRIPT_FILE, TRAINLOG_PROJECT, TRAINLOG_DESCRIPTION
#
# Usage:
#   trainlog add --name "my_run" ... \
#     --on-complete /gpfs/home/linjli/source/SpatialReasoning_train/scripts/on_train_complete.sh
#
# What it does:
#   1. Detects the matching eval subset from the run name/config/output_dir
#   2. Submits a Slurm job to convert FSDP checkpoint → safetensors
#   3. Submits a dependent Slurm job to run eval on the matching subset

set -euo pipefail

# --- Config ---
CONVERT_SCRIPT="/gpfs/home/linjli/source/SpatialReasoning_train/scripts/convert_sharded_to_full.py"
EVAL_DIR="/gpfs/home/linjli/source/SpatialReasoning_Eval"
BASE_MODEL="/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"
EVAL_OUTPUT_BASE="/gpfs/scrubbed/krishna/linjli/bagel_eval"
SLURM_LOG_DIR="/gpfs/home/linjli/slurm_logs"
CONDA_ENV="/gpfs/projects/krishna/envs/thinkmorph"

# --- Validate inputs ---
if [ -z "${TRAINLOG_CHECKPOINT:-}" ]; then
    echo "No checkpoint found for run #${TRAINLOG_ID} ${TRAINLOG_NAME}, skipping eval."
    exit 0
fi

if [ ! -d "${TRAINLOG_CHECKPOINT}" ]; then
    echo "Checkpoint dir does not exist: ${TRAINLOG_CHECKPOINT}"
    exit 1
fi

# --- Detect eval subset ---
# Subset keys ordered longest-first to avoid partial matches
SUBSET_KEYS=(
    td_ego_dir_arrow
    td_ego_side_arrow
    td_path_arrow
    td_ego_dir
    td_ego_side
    td_midpoint
    td_path
    dh_midpoint
)

HAYSTACK="${TRAINLOG_NAME} ${TRAINLOG_CONFIG_FILE:-} ${TRAINLOG_OUTPUT_DIR:-} ${TRAINLOG_DESCRIPTION:-}"
HAYSTACK=$(echo "$HAYSTACK" | tr '[:upper:]' '[:lower:]')

EVAL_SUBSET=""
for key in "${SUBSET_KEYS[@]}"; do
    if [[ "$HAYSTACK" == *"$key"* ]]; then
        EVAL_SUBSET="AI2ThorPT2P_${key}"
        break
    fi
done

if [ -z "$EVAL_SUBSET" ]; then
    echo "No matching eval subset for run #${TRAINLOG_ID} ${TRAINLOG_NAME}, skipping."
    exit 0
fi

# --- Detect answer-only vs CoT ---
CONVERT_EXTRA=""
IS_ANSWER_ONLY=false
if [[ "$HAYSTACK" == *"answer_only"* ]] || [[ "$HAYSTACK" == *"answer-only"* ]] || [[ "$HAYSTACK" == *"_ao"* ]]; then
    CONVERT_EXTRA="--no_visual_gen"
    IS_ANSWER_ONLY=true
fi

# --- Detect output resolution for CoT models ---
# latent 16 -> 256, latent 32 -> 512, latent 64 -> 1024 (default)
EVAL_MODEL="bagel_mot"
OUTPUT_RESOLUTION=1024
if [ "$IS_ANSWER_ONLY" = false ]; then
    EVAL_MODEL="bagel_mot_vcot"
    if [[ "$HAYSTACK" == *"_l16"* ]] || [[ "$HAYSTACK" == *"latent_16"* ]] || [[ "$HAYSTACK" == *"latent16"* ]]; then
        OUTPUT_RESOLUTION=256
    elif [[ "$HAYSTACK" == *"_l32"* ]] || [[ "$HAYSTACK" == *"latent_32"* ]] || [[ "$HAYSTACK" == *"latent32"* ]]; then
        OUTPUT_RESOLUTION=512
    fi
fi

# --- Paths ---
CONVERTED_PATH="${TRAINLOG_CHECKPOINT}_full"
# Prefix CoT eval runs with "vcot_" for clarity
if [ "$IS_ANSWER_ONLY" = true ]; then
    RUN_NAME="${TRAINLOG_NAME}"
else
    RUN_NAME="vcot_${TRAINLOG_NAME}"
fi

mkdir -p "$SLURM_LOG_DIR"

# --- Submit convert job (2 GPUs, 400G) ---
CONVERT_JOB_ID=$(sbatch --parsable <<SBATCH
#!/bin/bash
#SBATCH --job-name=conv_${RUN_NAME}
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=${SLURM_LOG_DIR}/conv_${RUN_NAME}_%j.out

module load conda
conda activate ${CONDA_ENV}

echo "=== Converting checkpoint: ${TRAINLOG_CHECKPOINT} ==="
python3 ${CONVERT_SCRIPT} --checkpoint_path ${TRAINLOG_CHECKPOINT} --output_path ${CONVERTED_PATH} --model_path ${BASE_MODEL} --convert_ema ${CONVERT_EXTRA}
echo "=== Convert exit: \$? ==="
SBATCH
)

echo "  Submitted convert job ${CONVERT_JOB_ID} for ${RUN_NAME}"

# --- Submit eval job (2 GPUs, depends on convert) ---
EVAL_JOB_ID=$(sbatch --parsable --dependency=afterok:${CONVERT_JOB_ID} <<SBATCH
#!/bin/bash
#SBATCH --job-name=eval_${RUN_NAME}
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=${SLURM_LOG_DIR}/eval_${RUN_NAME}_%j.out

module load conda
conda activate ${CONDA_ENV}

cd ${EVAL_DIR}

echo "=== Evaluating: ${RUN_NAME} (model: ${EVAL_MODEL}, checkpoint: ${CONVERTED_PATH}, subset: ${EVAL_SUBSET}, output_res: ${OUTPUT_RESOLUTION}) ==="
export THINKMORPH_MODEL_PATH=${CONVERTED_PATH}
export THINKMORPH_OUTPUT_RESOLUTION=${OUTPUT_RESOLUTION}
export THINKMORPH_SAVE_DIR=${CONVERTED_PATH}/eval/vcot_images
mkdir -p \${THINKMORPH_SAVE_DIR}
MASTER_PORT=\$((29500 + RANDOM % 1000))
torchrun --master-port=\${MASTER_PORT} --nproc-per-node=2 run.py --model ${EVAL_MODEL} --data ${EVAL_SUBSET} --work-dir ${CONVERTED_PATH}/eval --mode all --reuse
echo "=== Eval exit: \$? ==="
SBATCH
)

echo "  Submitted eval job ${EVAL_JOB_ID} (depends on ${CONVERT_JOB_ID}) for ${RUN_NAME} → ${EVAL_SUBSET}"
