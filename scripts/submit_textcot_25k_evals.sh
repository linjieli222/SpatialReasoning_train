#!/bin/bash
# Submit all TextCoT 25k eval jobs with dependencies on conversion jobs

BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_text_cot/textcot_td_path_8gpu
SLURM=scripts/slurm/eval_ao.slurm  # same eval script, bagel_mot config

SUBSETS="AI2ThorPT2P_td_path AI2ThorPT2P_td_path_arrow AI2ThorPT2P_dh_midpoint AI2ThorPerspective_Arrow AI2ThorPerspective_NoArrow SAT_perspective"

declare -A CONV_JOBS
CONV_JOBS[0006000]=${1:?Usage: $0 <conv_6k> <conv_9k> <conv_12k> <conv_15k> <conv_18k>}
CONV_JOBS[0009000]=${2}
CONV_JOBS[0012000]=${3}
CONV_JOBS[0015000]=${4}
CONV_JOBS[0018000]=${5}

count=0
for step in 0006000 0009000 0012000 0015000 0018000; do
    conv_id=${CONV_JOBS[$step]}
    step_short=${step#00}
    for variant in noema ema; do
        ckpt=${BASE}/${step}_full_${variant}
        for subset in $SUBSETS; do
            short_subset=${subset#AI2Thor}
            short_subset=${short_subset#PT2P_}
            job_name="eval_tc_s${step_short}_${variant}_${short_subset}"
            sbatch --dependency=afterok:${conv_id} --job-name=${job_name} ${SLURM} ${ckpt} ${subset}
            count=$((count + 1))
        done
    done
done

echo "Submitted ${count} eval jobs"
