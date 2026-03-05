#!/bin/bash
# Watch for new Mixed from VCoT td_ego_dir checkpoints, auto-submit conversion + nothink/answeronly eval
# Monitors both mixed_from_vcot_s7k and mixed_from_vcot_s2k training runs.
# Designed to run via cron every 2 minutes.
#
# Cron entry (login01):
#   */2 * * * * /gpfs/home/linjli/source/SpatialReasoning_train/scripts/watch_and_eval_mixed_from_vcot.sh >> /gpfs/scrubbed/krishna/linjli/bagel_debug_output/watch_mixed_from_vcot.log 2>&1

TRAIN_DIR=/gpfs/home/linjli/source/SpatialReasoning_train
CONVERT_SLURM=${TRAIN_DIR}/scripts/slurm/convert_ema_only.slurm
EVAL_NOTHINK_SLURM=${TRAIN_DIR}/scripts/slurm/eval_nothink.slurm
EVAL_AO_SLURM=${TRAIN_DIR}/scripts/slurm/eval_ao.slurm

SUBSETS="AI2ThorPT2PV2_td_ego_dir AI2ThorPT2PV2_td_path AI2ThorPT2PV2_td_path_arrow RealPT_td_path RealPT_td_path_arrow"

cd ${TRAIN_DIR}

process_run() {
    local RUN_BASE=$1
    local RUN_TAG=$2  # e.g. mfv7k, mfv2k

    [ -d "${RUN_BASE}" ] || return 0
    local PROCESSED=${RUN_BASE}/watch_processed.txt
    touch ${PROCESSED}

    for ckpt_dir in ${RUN_BASE}/0*/; do
        [ -d "$ckpt_dir" ] || continue
        local step=$(basename $ckpt_dir)
        [[ $step =~ ^[0-9]{7}$ ]] || continue
        grep -qx "$step" ${PROCESSED} && continue
        [ -f "${ckpt_dir}/scheduler.pt" ] || continue

        echo "=== $(date) [${RUN_TAG}] New checkpoint: ${step} ==="

        # Conversion job (EMA only)
        local CONV_JOBID=$(sbatch --parsable --job-name=conv_${RUN_TAG}_s${step#000} \
            ${CONVERT_SLURM} ${RUN_BASE} ${step})
        echo "  Conversion job: ${CONV_JOBID}"

        local CKPT=${RUN_BASE}/${step}_full_ema
        local short_step=${step#000}; short_step=${short_step#0}

        # Nothink evals on all subsets
        for subset in ${SUBSETS}; do
            local short_subset=$(echo $subset | sed 's/AI2ThorPT2PV2_/v2_/; s/RealPT_/real_/')
            local jobname="${RUN_TAG}_s${short_step}_nothink_${short_subset}"
            local EVAL_JOBID=$(sbatch --parsable --dependency=afterok:${CONV_JOBID} --job-name=${jobname} \
                ${EVAL_NOTHINK_SLURM} ${CKPT} ${subset})
            echo "  Nothink eval: ${EVAL_JOBID} (${jobname})"
        done

        # Answeronly evals on all subsets
        for subset in ${SUBSETS}; do
            local short_subset=$(echo $subset | sed 's/AI2ThorPT2PV2_/v2_/; s/RealPT_/real_/')
            local jobname="${RUN_TAG}_s${short_step}_ao_${short_subset}"
            local EVAL_JOBID=$(sbatch --parsable --dependency=afterok:${CONV_JOBID} --job-name=${jobname} \
                ${EVAL_AO_SLURM} ${CKPT} ${subset})
            echo "  Answeronly eval: ${EVAL_JOBID} (${jobname})"
        done

        echo "$step" >> ${PROCESSED}
    done
}

# Mixed from VCoT s7k
process_run \
    /gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_mixed_from_vcot/mixed_from_vcot_s7k_td_ego_dir_8gpu \
    mfv7k

# Mixed from VCoT s2k
process_run \
    /gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_mixed_from_vcot/mixed_from_vcot_s2k_td_ego_dir_8gpu \
    mfv2k
