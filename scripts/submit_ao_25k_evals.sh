#!/bin/bash
# Submit all AO 25k eval jobs with dependencies on conversion jobs
# Each conversion job produces both noema and ema checkpoints

BASE=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_answer_only/ao_td_path_8gpu
SLURM=scripts/slurm/eval_ao.slurm

SUBSETS="AI2ThorPT2P_td_path AI2ThorPT2P_td_path_arrow AI2ThorPT2P_dh_midpoint AI2ThorPerspective_Arrow AI2ThorPerspective_NoArrow SAT_perspective"

# Map: step -> conversion job ID
declare -A CONV_JOBS
CONV_JOBS[0012000]=${1:?Usage: $0 <conv_job_12k> <conv_job_15k> <conv_job_18k> <conv_job_21k> <conv_job_24k>}
CONV_JOBS[0015000]=${2}
CONV_JOBS[0018000]=${3}
CONV_JOBS[0021000]=${4}
CONV_JOBS[0024000]=${5}

count=0
for step in 0012000 0015000 0018000 0021000 0024000; do
    conv_id=${CONV_JOBS[$step]}
    step_short=${step#00}  # e.g. 12000
    for variant in noema ema; do
        ckpt=${BASE}/${step}_full_${variant}
        for subset in $SUBSETS; do
            short_subset=${subset#AI2Thor}      # e.g. PT2P_td_path
            short_subset=${short_subset#PT2P_}  # e.g. td_path
            job_name="eval_ao_s${step_short}_${variant}_${short_subset}"
            sbatch --dependency=afterok:${conv_id} --job-name=${job_name} ${SLURM} ${ckpt} ${subset}
            count=$((count + 1))
        done
    done
done

echo "Submitted ${count} eval jobs"
