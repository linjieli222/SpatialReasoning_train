#!/bin/bash
# Watch for new MMCoT td_ego_dir checkpoints, auto-submit conversion + nothink eval
# Designed to run via cron every 2 minutes.
#
# Cron entry (login01):
#   */2 * * * * /gpfs/home/linjli/source/SpatialReasoning_train/scripts/watch_and_eval_mmcot.sh >> /gpfs/scrubbed/krishna/linjli/bagel_debug_output/watch_mmcot.log 2>&1

TRAIN_DIR=/gpfs/home/linjli/source/SpatialReasoning_train
CONVERT_SLURM=${TRAIN_DIR}/scripts/slurm/convert_ego_dir.slurm
EVAL_NOTHINK_SLURM=${TRAIN_DIR}/scripts/slurm/eval_nothink.slurm

SUBSET=AI2ThorPT2PV2_td_ego_dir

MMCOT_BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_mmcot/mmcot_td_ego_dir_8gpu
MMCOT_PROCESSED=${MMCOT_BASE}/watch_processed.txt

[ -d "${MMCOT_BASE}" ] || exit 0
touch ${MMCOT_PROCESSED}

cd ${TRAIN_DIR}

for ckpt_dir in ${MMCOT_BASE}/0*/; do
    [ -d "$ckpt_dir" ] || continue
    step=$(basename $ckpt_dir)
    [[ $step =~ ^[0-9]{7}$ ]] || continue
    grep -qx "$step" ${MMCOT_PROCESSED} && continue
    [ -f "${ckpt_dir}/scheduler.pt" ] || continue

    echo "=== $(date) [MMCoT] New checkpoint: ${step} ==="

    # Conversion job (keeps visual_gen for MMCoT)
    CONV_JOBID=$(sbatch --parsable --job-name=conv_mmcot_s${step#000} \
        ${CONVERT_SLURM} ${MMCOT_BASE} ${step})
    echo "  Conversion job: ${CONV_JOBID}"

    # Nothink eval on EMA only
    CKPT=${MMCOT_BASE}/${step}_full_ema
    short_step=${step#000}; short_step=${short_step#0}
    jobname="eval_mmcot_s${short_step}_ema_nothink_pt2pv2_ego"
    EVAL_JOBID=$(sbatch --parsable --dependency=afterok:${CONV_JOBID} --job-name=${jobname} \
        ${EVAL_NOTHINK_SLURM} ${CKPT} ${SUBSET})
    echo "  Eval job: ${EVAL_JOBID} (${jobname})"

    echo "$step" >> ${MMCOT_PROCESSED}
done
