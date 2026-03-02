#!/bin/bash
# Watch for new TextCoT EMA999 checkpoints and auto-submit conversion + eval
# Designed to run via cron every 2 minutes.
# Each invocation does a single sweep and exits.
#
# Cron entry (login01):
#   */2 * * * * /gpfs/home/linjli/source/SpatialReasoning_train/scripts/watch_and_eval_textcot_ema999.sh >> /gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_text_cot_ema999/watch.log 2>&1

BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_text_cot_ema999/textcot_td_path_8gpu
TRAIN_DIR=/gpfs/home/linjli/source/SpatialReasoning_train
CONVERT_SCRIPT=scripts/slurm/convert_textcot_ema999.slurm
EVAL_SCRIPT=scripts/slurm/eval_ao.slurm

# Subsets to evaluate
SUBSETS="AI2ThorPT2P_td_path AI2ThorPT2P_dh_midpoint AI2ThorSV_td_path AI2ThorSV_dh_midpoint"

# Track which checkpoints we've already processed
PROCESSED_FILE=${BASE}/watch_processed.txt
touch ${PROCESSED_FILE}

FOUND_NEW=0

# Find checkpoint dirs (7-digit step numbers)
for ckpt_dir in ${BASE}/0*/; do
    [ -d "$ckpt_dir" ] || continue

    step=$(basename $ckpt_dir)
    # Skip non-checkpoint dirs (e.g., 0001000_full_noema)
    [[ $step =~ ^[0-9]{7}$ ]] || continue

    # Skip already processed
    grep -qx "$step" ${PROCESSED_FILE} && continue

    # Check if checkpoint is complete (scheduler.pt is written last)
    [ -f "${ckpt_dir}/scheduler.pt" ] || continue

    FOUND_NEW=1
    echo "=== $(date) New checkpoint: ${step} ==="

    # Submit conversion
    cd ${TRAIN_DIR}
    CONV_JOBID=$(sbatch --parsable --job-name=conv_tc999_s${step#000} ${CONVERT_SCRIPT} ${step})
    echo "  Conversion job: ${CONV_JOBID}"

    # Submit evals for both noema and ema, dependent on conversion
    for variant in noema ema; do
        CKPT=${BASE}/${step}_full_${variant}
        for subset in ${SUBSETS}; do
            short_subset=$(echo $subset | sed 's/AI2ThorPT2P_/pt2p_/; s/AI2ThorSV_/sv_/')
            short_step=${step#000}  # 0002000 -> 2000
            short_step=${short_step#0}  # 02000 -> 2000
            jobname="eval_tc999_s${short_step}_${variant}_${short_subset}"
            EVAL_JOBID=$(sbatch --parsable --dependency=afterok:${CONV_JOBID} --job-name=${jobname} ${EVAL_SCRIPT} ${CKPT} ${subset})
            echo "  Eval job: ${EVAL_JOBID} (${jobname})"
        done
    done

    # Mark as processed
    echo "$step" >> ${PROCESSED_FILE}
done

if [ "$FOUND_NEW" -eq 0 ]; then
    : # Silent when no new checkpoints (keep cron log clean)
fi
