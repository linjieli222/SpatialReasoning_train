#!/bin/bash
# Watch for new MMCoT td_ego_dir checkpoints, auto-submit conversion + nothink eval (PT2PV2 + SV)
# EMA only. Runs one eval at a time via dependency chains.
#
# Usage (loop mode): nohup scripts/watch_and_eval_mmcot_ego_dir.sh --loop >> /gpfs/scrubbed/krishna/linjli/bagel_debug_output/watch_mmcot_ego_dir.log 2>&1 &
# Usage (single):    scripts/watch_and_eval_mmcot_ego_dir.sh

TRAIN_DIR=/gpfs/home/linjli/source/SpatialReasoning_train
CONVERT_SLURM=${TRAIN_DIR}/scripts/slurm/convert_ego_dir.slurm
EVAL_SLURM=${TRAIN_DIR}/scripts/slurm/eval_generic.slurm

BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_mmcot/mmcot_td_ego_dir_8gpu
PROCESSED=${BASE}/watch_processed.txt
touch ${PROCESSED}

SUBSETS="AI2ThorPT2P_td_ego_dir AI2ThorPT2PV2_td_ego_dir AI2ThorSV_td_ego_dir"

run_once() {
    for ckpt_dir in ${BASE}/0*/; do
        [ -d "$ckpt_dir" ] || continue
        step=$(basename $ckpt_dir)
        [[ $step =~ ^[0-9]{7}$ ]] || continue
        grep -qx "$step" ${PROCESSED} && continue
        [ -f "${ckpt_dir}/scheduler.pt" ] || continue

        echo "=== $(date) [MMCoT] New checkpoint: ${step} ==="
        cd ${TRAIN_DIR}

        # Convert EMA (no --no_visual_gen for MMCoT)
        CONV_JOBID=$(sbatch --parsable --job-name=conv_mmcot_s${step#000} \
            ${CONVERT_SLURM} ${BASE} ${step})
        echo "  Conversion job: ${CONV_JOBID}"

        CKPT_EMA=${BASE}/${step}_full_ema
        LAST_JOBID=${CONV_JOBID}

        # Nothink evals on PT2PV2 and SV
        for subset in ${SUBSETS}; do
            short_subset=$(echo $subset | sed 's/AI2ThorPT2P_/pt2p_/; s/AI2ThorPT2PV2_/pt2pv2_/; s/AI2ThorSV_/sv_/')
            short_step=${step#000}; short_step=${short_step#0}
            jobname="eval_mmcot_s${short_step}_ema_nothink_${short_subset}"
            EVAL_JOBID=$(sbatch --parsable --dependency=afterany:${LAST_JOBID} \
                --job-name=${jobname} ${EVAL_SLURM} ${CKPT_EMA} ${subset} bagel_mot_nothink)
            echo "  Nothink eval: ${EVAL_JOBID} (${jobname})"
            LAST_JOBID=${EVAL_JOBID}
        done

        echo "$step" >> ${PROCESSED}
    done
}

if [ "$1" = "--loop" ]; then
    echo "=== $(date) Starting MMCoT watch loop (PID $$) ==="
    while true; do
        run_once
        sleep 120
    done
else
    run_once
fi
