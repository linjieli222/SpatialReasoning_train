#!/bin/bash
# td_ego_dir Self-Forcing VCoT: initialized from VCoT s4k checkpoint
# 11,204 samples, ~16 samples/step, ~700 steps/epoch
# 5,000 steps ≈ 7 epochs
# NOTE: self-forcing adds N denoising forward passes per step (under no_grad).
# With --self_force_steps 4, expect ~5x slower than normal VCoT.

resume_from=${resume_from:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_vcot_l32/vcot_td_ego_dir_8gpu/0004000_full_ema"}
run_name=${run_name:-"self_force_td_ego_dir_8gpu"}
output_path=${output_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_self_force/${run_name}/output"}
ckpt_path=${ckpt_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_self_force/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/tifa_v3_td_ego_dir_vcot_l32.yaml \
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
  --max_latent_size 32 \
  --max_num_tokens 32768 \
  --max_num_tokens_per_sample 24576 \
  --expected_num_tokens 24576 \
  --mse_weight 1 \
  --ce_weight 1 \
  --ema 0.999 \
  --total_steps 5000 \
  --save_every 1000 \
  --self_force True \
  --self_force_steps 4 \
  --self_force_ratio 0.5 \
  --wandb_project tifa_v3 \
  --wandb_name self_force_td_ego_dir
