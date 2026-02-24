#!/bin/bash

# Training script for TIFA v3 (dh_midpoint default variant - visual CoT)
# Based on train_pat.sh, adapted for tifa_train_v3 dataset on Tillicum

resume_from=${resume_from:-"/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"}
run_name=${run_name:-"debug_8gpu"}
output_path=${output_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_dh_midpoint/${run_name}/output"}
ckpt_path=${ckpt_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_dh_midpoint/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/tifa_v3_dh_midpoint_default_debug.yaml \
  --model_path $resume_from \
  --layer_module Qwen2MoTDecoderLayer \
  --finetune_from_hf True \
  --auto_resume True \
  --resume_model_only True \
  --finetune-from-ema True \
  --resume_from $resume_from \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --lr 1e-5 \
  --num_workers 8 \
  --max_latent_size 64 \
  --max_num_tokens 32768 \
  --max_num_tokens_per_sample 24576 \
  --expected_num_tokens 24576 \
  --mse_weight 1 \
  --ce_weight 1 \
  --total_steps 100 \
  --save_every 50 \
  --wandb_project tifa_v3 \
  --wandb_name dh_midpoint_debug
