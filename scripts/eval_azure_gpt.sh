#!/bin/bash
# Run Azure GPT (gpt-5) evals on specified subsets — all in parallel
# Usage: bash scripts/eval_azure_gpt.sh

export AZURE_OPENAI_ENDPOINT="https://linjl-ma65uv6u-eastus2.cognitiveservices.azure.com/"
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:?Set AZURE_OPENAI_API_KEY env var}"
export AZURE_OPENAI_DEPLOYMENT="gpt-5"
export HF_HOME=/gpfs/scrubbed/linjli/hf_cache

WORK_DIR=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/azure_gpt_eval
PYTHON=/gpfs/projects/krishna/envs/thinkmorph/bin/python
LOG_DIR=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/azure_gpt_eval/logs
mkdir -p ${LOG_DIR}

cd /gpfs/home/linjli/source/SpatialReasoning_Eval

SUBSETS=(
    AI2ThorPT2PV2_td_ego_dir
    AI2ThorPT2PV2_td_path
    AI2ThorPT2PV2_td_path_arrow
    RealPT_td_path
    RealPT_td_path_arrow
)

PIDS=()
for subset in "${SUBSETS[@]}"; do
    echo "$(date) Launching ${subset}..."
    $PYTHON run.py --model AzureGPT --data ${subset} --work-dir ${WORK_DIR} \
        > ${LOG_DIR}/${subset}.log 2>&1 &
    PIDS+=($!)
done

echo "$(date) All 5 evals launched. PIDs: ${PIDS[*]}"
echo "Waiting for all to finish..."

for i in "${!SUBSETS[@]}"; do
    wait ${PIDS[$i]}
    echo "$(date) ${SUBSETS[$i]} finished (exit $?)"
done

echo "$(date) All done."
