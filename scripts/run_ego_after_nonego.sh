#!/bin/bash
# Watches for non-ego processing to finish (td_path_arrow_answer_only),
# then kicks off ego configs sequentially with num_proc=1.

WATCH_DIR="/gpfs/projects/krishna/linjli/bagel_example/editing/tifa_train_v3_td_path_arrow_answer_only"
WATCH_FILE="$WATCH_DIR/chunk_4.parquet"
SCRIPT="/gpfs/home/linjli/source/SpatialReasoning_train/data_processing/transform_tifa_train_v3.py"
PYTHON="/gpfs/projects/krishna/envs/thinkmorph/bin/python"

export HF_HOME=/gpfs/scrubbed/linjli/hf_cache

echo "$(date) - Waiting for non-ego processing to finish..."
echo "Watching: $WATCH_FILE"

# Wait until the file exists and was modified today (02/19)
while true; do
    if [ -f "$WATCH_FILE" ]; then
        mod_date=$(stat -c %Y "$WATCH_FILE")
        today_start=$(date -d "today 00:00" +%s)
        if [ "$mod_date" -ge "$today_start" ]; then
            echo "$(date) - Non-ego processing complete! td_path_arrow_answer_only updated at $(date -d @$mod_date)"
            # Wait an extra 60s to ensure the process fully exits
            sleep 60
            break
        fi
    fi
    sleep 120
done

echo "$(date) - Starting ego configs processing..."

CONFIGS=(td_ego_dir td_ego_dir_arrow td_ego_side td_ego_side_arrow)
VARIANTS=(default answer_only)

for config in "${CONFIGS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        echo ""
        echo "$(date) - Processing $config ($variant)..."
        $PYTHON $SCRIPT --configs $config --variants $variant --num_proc 1 2>&1 | tail -20
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "$(date) - ERROR: $config ($variant) failed with exit code $exit_code"
        else
            echo "$(date) - DONE: $config ($variant)"
        fi
    done
done

echo ""
echo "$(date) - All ego configs processing complete!"
