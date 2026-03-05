#!/bin/bash
# td_ego_dir Mixed VCoT+AO initialized from VCoT l32 s7k EMA checkpoint
# Same data mix as mixed training: 50% VCoT + 50% AO
# 5,000 steps from VCoT s7k

resume_from=${resume_from:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_vcot_l32/vcot_td_ego_dir_8gpu/0007000_full_ema"}
run_name=${run_name:-"mixed_from_vcot_s7k_td_ego_dir_8gpu"}
output_path=${output_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_mixed_from_vcot/${run_name}/output"}
ckpt_path=${ckpt_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_mixed_from_vcot/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/tifa_v3_td_ego_dir_mixed_vcot_ao.yaml \
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
  --wandb_project tifa_v3 \
  --wandb_name mixed_from_vcot_s7k_td_ego_dir
