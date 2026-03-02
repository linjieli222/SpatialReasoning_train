#!/bin/bash
# td_path text CoT 10k steps (refreshed data, EMA 0.999)

resume_from=${resume_from:-"/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"}
run_name=${run_name:-"textcot_td_path_8gpu"}
output_path=${output_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_text_cot_v2/${run_name}/output"}
ckpt_path=${ckpt_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_text_cot_v2/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/tifa_v3_td_path_text_cot.yaml \
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
  --visual_gen False \
  --mse_weight 0 \
  --ce_weight 1 \
  --ema 0.999 \
  --total_steps 10000 \
  --save_every 1000 \
  --wandb_project tifa_v3 \
  --wandb_name textcot_td_path_10k_v2
