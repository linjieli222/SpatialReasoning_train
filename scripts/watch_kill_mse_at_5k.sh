#!/bin/bash
# Watch mse5 (67190) and mse2 (67191) training jobs
# Kill each when they reach step 5000

MSE5_JOB=67190
MSE2_JOB=67191
MSE5_LOG=$(scontrol show job $MSE5_JOB 2>/dev/null | grep -oP 'StdOut=\K\S+')
MSE2_LOG=$(scontrol show job $MSE2_JOB 2>/dev/null | grep -oP 'StdOut=\K\S+')

MSE5_KILLED=false
MSE2_KILLED=false

echo "Watching mse5 job $MSE5_JOB (log: $MSE5_LOG)"
echo "Watching mse2 job $MSE2_JOB (log: $MSE2_LOG)"

while true; do
    # Check mse5
    if [ "$MSE5_KILLED" = false ]; then
        if squeue -j $MSE5_JOB 2>/dev/null | grep -q $MSE5_JOB; then
            MSE5_STEP=$(tail -20 "$MSE5_LOG" 2>/dev/null | grep -oP 'step=\K[0-9]+' | tail -1)
            if [ -n "$MSE5_STEP" ] && [ "$MSE5_STEP" -ge 5000 ] 2>/dev/null; then
                echo "$(date) - mse5 reached step $MSE5_STEP >= 5000, cancelling job $MSE5_JOB"
                scancel $MSE5_JOB
                MSE5_KILLED=true
            else
                echo "$(date) - mse5 at step ${MSE5_STEP:-unknown}"
            fi
        else
            echo "$(date) - mse5 job $MSE5_JOB no longer running"
            MSE5_KILLED=true
        fi
    fi

    # Check mse2
    if [ "$MSE2_KILLED" = false ]; then
        if squeue -j $MSE2_JOB 2>/dev/null | grep -q $MSE2_JOB; then
            MSE2_STEP=$(tail -20 "$MSE2_LOG" 2>/dev/null | grep -oP 'step=\K[0-9]+' | tail -1)
            if [ -n "$MSE2_STEP" ] && [ "$MSE2_STEP" -ge 5000 ] 2>/dev/null; then
                echo "$(date) - mse2 reached step $MSE2_STEP >= 5000, cancelling job $MSE2_JOB"
                scancel $MSE2_JOB
                MSE2_KILLED=true
            else
                echo "$(date) - mse2 at step ${MSE2_STEP:-unknown}"
            fi
        else
            echo "$(date) - mse2 job $MSE2_JOB no longer running"
            MSE2_KILLED=true
        fi
    fi

    # Exit if both killed
    if [ "$MSE5_KILLED" = true ] && [ "$MSE2_KILLED" = true ]; then
        echo "$(date) - Both jobs handled. Exiting watcher."
        break
    fi

    sleep 60
done
