#!/bin/bash
# Watch for new ego_dir AO and VCoT checkpoints, auto-submit conversion + eval
# Designed to run via cron every 2 minutes.
#
# Cron entry (login01):
#   */2 * * * * /gpfs/home/linjli/source/SpatialReasoning_train/scripts/watch_and_eval_ego_dir.sh >> /gpfs/scrubbed/krishna/linjli/bagel_debug_output/watch_ego_dir.log 2>&1

TRAIN_DIR=/gpfs/home/linjli/source/SpatialReasoning_train
MODEL=/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT
CONVERT_SCRIPT=${TRAIN_DIR}/scripts/convert_sharded_to_full.py

CONVERT_SCRIPT_SLURM=scripts/slurm/convert_ego_dir.slurm
EVAL_AO_SCRIPT=scripts/slurm/eval_ao.slurm
EVAL_VCOT_SCRIPT=scripts/slurm/eval_vcot.slurm

# Subsets to evaluate
SUBSETS="AI2ThorPT2P_td_ego_dir AI2ThorSV_td_ego_dir"

# --- AO ---
AO_BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_ao/ao_td_ego_dir_8gpu
AO_PROCESSED=${AO_BASE}/watch_processed.txt
touch ${AO_PROCESSED}

for ckpt_dir in ${AO_BASE}/0*/; do
    [ -d "$ckpt_dir" ] || continue
    step=$(basename $ckpt_dir)
    [[ $step =~ ^[0-9]{7}$ ]] || continue
    grep -qx "$step" ${AO_PROCESSED} && continue
    [ -f "${ckpt_dir}/scheduler.pt" ] || continue

    echo "=== $(date) [AO] New checkpoint: ${step} ==="

    cd ${TRAIN_DIR}

    # Conversion job (with --no_visual_gen for AO)
    CONV_JOBID=$(sbatch --parsable --job-name=conv_ego_ao_s${step#000} \
        ${CONVERT_SCRIPT_SLURM} ${AO_BASE} ${step} --no_visual_gen)
    echo "  Conversion job: ${CONV_JOBID}"

    for variant in noema ema; do
        CKPT=${AO_BASE}/${step}_full_${variant}
        for subset in ${SUBSETS}; do
            short_subset=$(echo $subset | sed 's/AI2ThorPT2P_/pt2p_/; s/AI2ThorSV_/sv_/')
            short_step=${step#000}; short_step=${short_step#0}
            jobname="eval_ego_ao_s${short_step}_${variant}_${short_subset}"
            EVAL_JOBID=$(sbatch --parsable --dependency=afterok:${CONV_JOBID} --job-name=${jobname} ${EVAL_AO_SCRIPT} ${CKPT} ${subset})
            echo "  Eval job: ${EVAL_JOBID} (${jobname})"
        done
    done

    echo "$step" >> ${AO_PROCESSED}
done

# --- VCoT ---
VCOT_BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_vcot_l32/vcot_td_ego_dir_8gpu
VCOT_PROCESSED=${VCOT_BASE}/watch_processed.txt
# Only create if dir exists (training may not have started yet)
if [ -d "${VCOT_BASE}" ]; then
    touch ${VCOT_PROCESSED}

    for ckpt_dir in ${VCOT_BASE}/0*/; do
        [ -d "$ckpt_dir" ] || continue
        step=$(basename $ckpt_dir)
        [[ $step =~ ^[0-9]{7}$ ]] || continue
        grep -qx "$step" ${VCOT_PROCESSED} && continue
        [ -f "${ckpt_dir}/scheduler.pt" ] || continue

        echo "=== $(date) [VCoT] New checkpoint: ${step} ==="

        cd ${TRAIN_DIR}

        # Conversion job (VCoT keeps visual_gen)
        CONV_JOBID=$(sbatch --parsable --job-name=conv_ego_vcot_s${step#000} \
            ${CONVERT_SCRIPT_SLURM} ${VCOT_BASE} ${step})
        echo "  Conversion job: ${CONV_JOBID}"

        for variant in noema ema; do
            CKPT=${VCOT_BASE}/${step}_full_${variant}
            for subset in ${SUBSETS}; do
                short_subset=$(echo $subset | sed 's/AI2ThorPT2P_/pt2p_/; s/AI2ThorSV_/sv_/')
                short_step=${step#000}; short_step=${short_step#0}
                jobname="eval_ego_vcot_s${short_step}_${variant}_${short_subset}"
                EVAL_JOBID=$(sbatch --parsable --dependency=afterok:${CONV_JOBID} --job-name=${jobname} ${EVAL_VCOT_SCRIPT} ${CKPT} ${subset})
                echo "  Eval job: ${EVAL_JOBID} (${jobname})"
            done
        done

        echo "$step" >> ${VCOT_PROCESSED}
    done
fi
