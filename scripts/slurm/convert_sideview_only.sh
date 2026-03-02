#!/bin/bash
#SBATCH --job-name=convert_sideview_only
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/convert_sideview_only_%j.out

source ~/.bashrc

cd /gpfs/home/linjli/source/SpatialReasoning_train

/gpfs/home/linjli/source/spatial-vcot/.venv/bin/python \
    data_processing/transform_sideview_only.py \
    --split td_path \
    --num_shards 5 \
    --num_proc 4
