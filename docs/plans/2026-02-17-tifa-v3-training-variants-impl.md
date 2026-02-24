# TIFA v3 Training Variant Experiments — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create all configs, scripts, and slurm files to run the 18-run TIFA v3 training experiment matrix.

**Architecture:** Each experiment has 3 files: a YAML dataset config, a training shell script, and a Slurm submission script. A generic Slurm template is shared across all experiments. The training scripts differ in dataset config, step count, latent size, and visual_gen flag.

**Tech Stack:** Bash, YAML, Slurm, torchrun, FSDP

**Design doc:** `docs/plans/2026-02-17-tifa-v3-training-variants-design.md`

---

### Task 1: Fix stale model path in default training script

The existing `scripts/train_tifa_v3_debug.sh` still uses the wrong model path.

**Files:**
- Modify: `scripts/train_tifa_v3_debug.sh:6`

**Step 1: Fix the path**

Change line 6 from:
```bash
resume_from=${resume_from:-"/gpfs/projects/krishna/hf_cache/BAGEL-7B-MoT"}
```
to:
```bash
resume_from=${resume_from:-"/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"}
```

**Step 2: Also add WANDB_DIR to its slurm script**

The default debug slurm script (`scripts/slurm/train_tifa_v3_debug.slurm`) is missing the WANDB_DIR redirect. Add before the `bash scripts/...` line:
```bash
# Redirect wandb data to scrubbed storage to avoid home directory disk quota issues
export WANDB_DIR=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/wandb
export WANDB_CACHE_DIR=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/wandb/.cache
mkdir -p $WANDB_DIR $WANDB_CACHE_DIR
```

**Step 3: Commit**
```bash
git add scripts/train_tifa_v3_debug.sh scripts/slurm/train_tifa_v3_debug.slurm
git commit -m "fix: correct model path and add WANDB_DIR to default training scripts"
```

---

### Task 2: Create generic Slurm template

A single parameterized Slurm script that all experiments share. Experiment-specific settings (job name, time, script) are passed via sbatch CLI overrides.

**Files:**
- Create: `scripts/slurm/train_tifa_v3.slurm`

**Step 1: Write the template**

```bash
#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1600G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# Generic TIFA v3 training launcher for Tillicum
# Usage: sbatch --job-name=<name> [--time=HH:MM:SS] scripts/slurm/train_tifa_v3.slurm <training_script.sh>
#
# Examples:
#   sbatch --job-name=p1_td_path_ao scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_path_answer_only.sh
#   sbatch --job-name=p2_td_path_l64 --time=24:00:00 scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_path_cot_l64.sh

TRAIN_SCRIPT=${1:?Usage: sbatch train_tifa_v3.slurm <training_script.sh>}

mkdir -p slurm_logs

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

cd /gpfs/home/linjli/source/SpatialReasoning_train

# Redirect wandb to scrubbed storage (home quota too small)
export WANDB_DIR=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/wandb
export WANDB_CACHE_DIR=/gpfs/scrubbed/krishna/linjli/bagel_debug_output/wandb/.cache
mkdir -p $WANDB_DIR $WANDB_CACHE_DIR

echo "=== $(date) Starting: $TRAIN_SCRIPT ==="
bash "$TRAIN_SCRIPT"
echo "=== $(date) Finished: $TRAIN_SCRIPT (exit $?) ==="
```

**Step 2: Commit**
```bash
git add scripts/slurm/train_tifa_v3.slurm
git commit -m "feat: add generic slurm template for tifa v3 experiments"
```

---

### Task 3: Create Phase 1 YAML configs (answer-only annotation sweep)

Need 3 new YAML configs. `tifa_v3_dh_midpoint_answer_only_debug.yaml` already exists and can be reused as-is (the YAML has no debug-specific content).

**Files:**
- Create: `data/configs/tifa_v3_td_midpoint_answer_only.yaml`
- Create: `data/configs/tifa_v3_td_path_answer_only.yaml`
- Create: `data/configs/tifa_v3_td_path_arrow_answer_only.yaml`

**Step 1: Create all 3 configs**

`data/configs/tifa_v3_td_midpoint_answer_only.yaml`:
```yaml
unified_edit:
  dataset_names:
  - tifa_v3_td_midpoint_answer_only
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  is_mandatory: True
  num_used_data:
  - 3752
  weight: 1
```

`data/configs/tifa_v3_td_path_answer_only.yaml`:
```yaml
unified_edit:
  dataset_names:
  - tifa_v3_td_path_answer_only
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  is_mandatory: True
  num_used_data:
  - 9689
  weight: 1
```

`data/configs/tifa_v3_td_path_arrow_answer_only.yaml`:
```yaml
unified_edit:
  dataset_names:
  - tifa_v3_td_path_arrow_answer_only
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  is_mandatory: True
  num_used_data:
  - 9898
  weight: 1
```

**Step 2: Commit**
```bash
git add data/configs/tifa_v3_td_midpoint_answer_only.yaml data/configs/tifa_v3_td_path_answer_only.yaml data/configs/tifa_v3_td_path_arrow_answer_only.yaml
git commit -m "feat: add Phase 1 answer-only YAML configs for td_midpoint, td_path, td_path_arrow"
```

---

### Task 4: Create Phase 1 training scripts (answer-only)

4 training scripts. dh_midpoint reuses existing debug YAML. All target ~1,563 steps (50K samples / ~32 samples per step), save every 300.

**Files:**
- Create: `scripts/train_tifa_v3_dh_midpoint_answer_only.sh`
- Create: `scripts/train_tifa_v3_td_midpoint_answer_only.sh`
- Create: `scripts/train_tifa_v3_td_path_answer_only.sh`
- Create: `scripts/train_tifa_v3_td_path_arrow_answer_only.sh`

**Step 1: Create all 4 scripts**

Each follows this pattern (shown for dh_midpoint, others differ in dataset_config_file, output paths, wandb_name):

`scripts/train_tifa_v3_dh_midpoint_answer_only.sh`:
```bash
#!/bin/bash
# Phase 1: dh_midpoint answer-only (2,020 samples)

resume_from=${resume_from:-"/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"}
run_name=${run_name:-"phase1_8gpu"}
output_path=${output_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_dh_midpoint_answer_only/${run_name}/output"}
ckpt_path=${ckpt_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_dh_midpoint_answer_only/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/tifa_v3_dh_midpoint_answer_only_debug.yaml \
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
  --total_steps 1563 \
  --save_every 300 \
  --wandb_project tifa_v3 \
  --wandb_name p1_dh_midpoint_ao
```

`scripts/train_tifa_v3_td_midpoint_answer_only.sh`:
```bash
#!/bin/bash
# Phase 1: td_midpoint answer-only (3,752 samples)

resume_from=${resume_from:-"/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"}
run_name=${run_name:-"phase1_8gpu"}
output_path=${output_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_midpoint_answer_only/${run_name}/output"}
ckpt_path=${ckpt_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_midpoint_answer_only/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/tifa_v3_td_midpoint_answer_only.yaml \
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
  --total_steps 1563 \
  --save_every 300 \
  --wandb_project tifa_v3 \
  --wandb_name p1_td_midpoint_ao
```

`scripts/train_tifa_v3_td_path_answer_only.sh`:
```bash
#!/bin/bash
# Phase 1: td_path answer-only (9,689 samples)

resume_from=${resume_from:-"/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"}
run_name=${run_name:-"phase1_8gpu"}
output_path=${output_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_answer_only/${run_name}/output"}
ckpt_path=${ckpt_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_answer_only/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/tifa_v3_td_path_answer_only.yaml \
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
  --total_steps 1563 \
  --save_every 300 \
  --wandb_project tifa_v3 \
  --wandb_name p1_td_path_ao
```

`scripts/train_tifa_v3_td_path_arrow_answer_only.sh`:
```bash
#!/bin/bash
# Phase 1: td_path_arrow answer-only (9,898 samples)

resume_from=${resume_from:-"/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"}
run_name=${run_name:-"phase1_8gpu"}
output_path=${output_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_arrow_answer_only/${run_name}/output"}
ckpt_path=${ckpt_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_arrow_answer_only/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/tifa_v3_td_path_arrow_answer_only.yaml \
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
  --total_steps 1563 \
  --save_every 300 \
  --wandb_project tifa_v3 \
  --wandb_name p1_td_path_arrow_ao
```

**Step 2: Commit**
```bash
git add scripts/train_tifa_v3_dh_midpoint_answer_only.sh scripts/train_tifa_v3_td_midpoint_answer_only.sh scripts/train_tifa_v3_td_path_answer_only.sh scripts/train_tifa_v3_td_path_arrow_answer_only.sh
git commit -m "feat: add Phase 1 answer-only training scripts for all 4 annotations"
```

---

### Task 5: Create Phase 2 YAML configs (CoT latent sweep)

3 configs for td_path with different output image resolutions. Latent 64 uses the default transform (no output_image_transform_args needed). Latent 32 and 16 need explicit output transforms.

**Files:**
- Create: `data/configs/tifa_v3_td_path_default.yaml` (latent 64, output = input transform)
- Create: `data/configs/tifa_v3_td_path_default_l32.yaml` (latent 32, output max 512)
- Create: `data/configs/tifa_v3_td_path_default_l16.yaml` (latent 16, output max 256)

**Step 1: Create all 3 configs**

`data/configs/tifa_v3_td_path_default.yaml` (latent 64):
```yaml
unified_edit:
  dataset_names:
  - tifa_v3_td_path
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  is_mandatory: True
  num_used_data:
  - 9689
  weight: 1
```

`data/configs/tifa_v3_td_path_default_l32.yaml` (latent 32):
```yaml
unified_edit:
  dataset_names:
  - tifa_v3_td_path
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  output_image_transform_args:
    image_stride: 16
    max_image_size: 512
    min_image_size: 256
  is_mandatory: True
  num_used_data:
  - 9689
  weight: 1
```

`data/configs/tifa_v3_td_path_default_l16.yaml` (latent 16):
```yaml
unified_edit:
  dataset_names:
  - tifa_v3_td_path
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  output_image_transform_args:
    image_stride: 16
    max_image_size: 256
    min_image_size: 128
  is_mandatory: True
  num_used_data:
  - 9689
  weight: 1
```

**Step 2: Commit**
```bash
git add data/configs/tifa_v3_td_path_default.yaml data/configs/tifa_v3_td_path_default_l32.yaml data/configs/tifa_v3_td_path_default_l16.yaml
git commit -m "feat: add Phase 2 CoT configs for td_path with latent 64/32/16"
```

---

### Task 6: Create Phase 2 training scripts (CoT latent sweep)

3 training scripts for td_path CoT with different latent sizes. Step counts normalized by effective batch size (see design doc).

**Files:**
- Create: `scripts/train_tifa_v3_td_path_cot_l64.sh`
- Create: `scripts/train_tifa_v3_td_path_cot_l32.sh`
- Create: `scripts/train_tifa_v3_td_path_cot_l16.sh`

**Step 1: Create all 3 scripts**

`scripts/train_tifa_v3_td_path_cot_l64.sh`:
```bash
#!/bin/bash
# Phase 2: td_path CoT latent 64 (9,689 samples, ~8 samples/step -> 6,250 steps)

resume_from=${resume_from:-"/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"}
run_name=${run_name:-"phase2_l64_8gpu"}
output_path=${output_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_cot_l64/${run_name}/output"}
ckpt_path=${ckpt_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_cot_l64/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/tifa_v3_td_path_default.yaml \
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
  --total_steps 6250 \
  --save_every 1000 \
  --wandb_project tifa_v3 \
  --wandb_name p2_td_path_cot_l64
```

`scripts/train_tifa_v3_td_path_cot_l32.sh`:
```bash
#!/bin/bash
# Phase 2: td_path CoT latent 32 (9,689 samples, ~20 samples/step -> 2,500 steps)

resume_from=${resume_from:-"/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"}
run_name=${run_name:-"phase2_l32_8gpu"}
output_path=${output_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_cot_l32/${run_name}/output"}
ckpt_path=${ckpt_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_cot_l32/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/tifa_v3_td_path_default_l32.yaml \
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
  --total_steps 2500 \
  --save_every 500 \
  --wandb_project tifa_v3 \
  --wandb_name p2_td_path_cot_l32
```

`scripts/train_tifa_v3_td_path_cot_l16.sh`:
```bash
#!/bin/bash
# Phase 2: td_path CoT latent 16 (9,689 samples, ~24 samples/step -> 2,083 steps)

resume_from=${resume_from:-"/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"}
run_name=${run_name:-"phase2_l16_8gpu"}
output_path=${output_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_cot_l16/${run_name}/output"}
ckpt_path=${ckpt_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_path_cot_l16/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/tifa_v3_td_path_default_l16.yaml \
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
  --max_latent_size 16 \
  --max_num_tokens 32768 \
  --max_num_tokens_per_sample 24576 \
  --expected_num_tokens 24576 \
  --mse_weight 1 \
  --ce_weight 1 \
  --total_steps 2083 \
  --save_every 400 \
  --wandb_project tifa_v3 \
  --wandb_name p2_td_path_cot_l16
```

**Step 2: Commit**
```bash
git add scripts/train_tifa_v3_td_path_cot_l64.sh scripts/train_tifa_v3_td_path_cot_l32.sh scripts/train_tifa_v3_td_path_cot_l16.sh
git commit -m "feat: add Phase 2 CoT latent sweep training scripts for td_path"
```

---

### Task 7: Create Phase 4 combined YAML configs

Combined configs that mix all 4 non-ego annotation datasets.

**Files:**
- Create: `data/configs/tifa_v3_all_answer_only.yaml`
- Create: `data/configs/tifa_v3_all_default.yaml` (latent 64 default; latent 32/16 variants created later based on Phase 2 results)

**Step 1: Create combined configs**

`data/configs/tifa_v3_all_answer_only.yaml`:
```yaml
unified_edit:
  dataset_names:
  - tifa_v3_dh_midpoint_answer_only
  - tifa_v3_td_midpoint_answer_only
  - tifa_v3_td_path_answer_only
  - tifa_v3_td_path_arrow_answer_only
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  is_mandatory: True
  num_used_data:
  - 2020
  - 3752
  - 9689
  - 9898
  weight: 1
```

`data/configs/tifa_v3_all_default.yaml`:
```yaml
unified_edit:
  dataset_names:
  - tifa_v3_dh_midpoint
  - tifa_v3_td_midpoint
  - tifa_v3_td_path
  - tifa_v3_td_path_arrow
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  is_mandatory: True
  num_used_data:
  - 2020
  - 3752
  - 9689
  - 9898
  weight: 1
```

**Step 2: Create training scripts**

`scripts/train_tifa_v3_all_answer_only.sh`:
```bash
#!/bin/bash
# Phase 4: All non-ego answer-only combined (~25,359 samples, ~32 samples/step -> 1,563 steps)

resume_from=${resume_from:-"/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"}
run_name=${run_name:-"phase4_8gpu"}
output_path=${output_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_all_answer_only/${run_name}/output"}
ckpt_path=${ckpt_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_all_answer_only/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/tifa_v3_all_answer_only.yaml \
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
  --total_steps 1563 \
  --save_every 300 \
  --wandb_project tifa_v3 \
  --wandb_name p4_all_ao
```

`scripts/train_tifa_v3_all_cot_l64.sh`:
```bash
#!/bin/bash
# Phase 4: All non-ego CoT combined latent 64 (~25,359 samples, ~8 samples/step -> 6,250 steps)

resume_from=${resume_from:-"/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"}
run_name=${run_name:-"phase4_8gpu"}
output_path=${output_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_all_cot_l64/${run_name}/output"}
ckpt_path=${ckpt_path:-"/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_all_cot_l64/${run_name}"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/tifa_v3_all_default.yaml \
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
  --total_steps 6250 \
  --save_every 1000 \
  --wandb_project tifa_v3 \
  --wandb_name p4_all_cot_l64
```

**Step 3: Commit**
```bash
git add data/configs/tifa_v3_all_answer_only.yaml data/configs/tifa_v3_all_default.yaml scripts/train_tifa_v3_all_answer_only.sh scripts/train_tifa_v3_all_cot_l64.sh
git commit -m "feat: add Phase 4 combined dataset configs and training scripts"
```

---

### Task 8: Register missing default (CoT) datasets in dataset_info.py

The td_midpoint, td_path, and td_path_arrow default (CoT) datasets are already registered, but verify dh_midpoint default is also registered. Also add a note about ego datasets being pending.

**Files:**
- Modify: `data/dataset_info.py` (verify all 8 non-ego datasets are registered)

**Step 1: Verify registry completeness**

Check that all these keys exist in DATASET_INFO['unified_edit']:
- `tifa_v3_dh_midpoint` ✓ (line 103)
- `tifa_v3_dh_midpoint_answer_only` ✓ (line 110)
- `tifa_v3_td_midpoint` ✓ (line 116)
- `tifa_v3_td_midpoint_answer_only` ✓ (line 122)
- `tifa_v3_td_path` ✓ (line 128)
- `tifa_v3_td_path_answer_only` ✓ (line 134)
- `tifa_v3_td_path_arrow` ✓ (line 140)
- `tifa_v3_td_path_arrow_answer_only` ✓ (line 146)

All 8 non-ego datasets are already registered. No changes needed.

---

### Task 9: Validate Phase 1+2 setup with dry runs

Run 10 steps of one answer-only and one CoT experiment to verify configs load correctly and measure actual effective batch size.

**Files:** None (validation only)

**Step 1: Submit a 10-step answer-only validation**

Create a temporary override:
```bash
sbatch --job-name=validate_ao --time=00:30:00 scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_path_answer_only.sh
```

But first override total_steps by setting env vars or editing the script temporarily. Better approach: use the existing debug scripts to validate the configs work, then just change the YAML path.

Actually, simplest: submit with `total_steps` override. But total_steps is a CLI arg in the .sh script, not an env var. So create a quick validation script:

```bash
# Quick validation: just run the real script but override steps
total_steps=10 save_every=5 \
  bash scripts/train_tifa_v3_td_path_answer_only.sh 2>&1 | head -100
```

Wait — the scripts use `${total_steps:-1563}` pattern only for `resume_from`, `run_name`, etc. The `--total_steps` is hardcoded. So we can't override via env var without modifying the scripts.

**Better approach:** After creating all scripts, submit one Phase 1 run as a real experiment. If it fails in the first few steps, we'll see it in the slurm log. The auto_resume flag means we can restart without losing progress.

**Step 1: Submit Phase 1 run #3 (td_path answer_only) as first real experiment**
```bash
cd /gpfs/home/linjli/source/SpatialReasoning_train
sbatch --job-name=p1_td_path_ao scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_path_answer_only.sh
```

**Step 2: Monitor first 50 steps**
```bash
# Wait ~5 min, then check log
tail -50 slurm_logs/p1_td_path_ao_*.out
```

Expected: training starts, CE loss logged, no crashes.

---

### Task 10: Process ego variant data (Phase 5 prep)

Run `transform_tifa_train_v3.py` for the 4 ego configs. This downloads from HuggingFace and saves parquet files to projects storage.

**Files:**
- Modify: `data/dataset_info.py` (add 8 ego dataset entries after processing)

**Step 1: Create a Slurm job for data processing**

Create `scripts/slurm/process_ego_data.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=process_ego
#SBATCH --qos=normal
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/process_ego_%j.out

mkdir -p slurm_logs

module load conda
conda activate /gpfs/projects/krishna/envs/thinkmorph

cd /gpfs/home/linjli/source/SpatialReasoning_train

python data_processing/transform_tifa_train_v3.py \
  --configs td_ego_dir td_ego_dir_arrow td_ego_side td_ego_side_arrow \
  --variants default answer_only \
  --num_shards 5 \
  --num_proc 8
```

**Step 2: Submit the processing job**
```bash
sbatch scripts/slurm/process_ego_data.slurm
```

**Step 3: After processing completes, add ego datasets to registry**

Add to `data/dataset_info.py` in the `unified_edit` section (exact paths and sample counts will come from the processing script output):

```python
        # TIFA v3 ego variants (Tillicum) - processed from linjieli222/tifa_train_v3
        'tifa_v3_td_ego_dir': {
            'data_dir': '/gpfs/projects/krishna/linjli/bagel_example/editing/tifa_train_v3_td_ego_dir',
            'num_files': 5,
            'num_total_samples': 11204,  # verify from processing output
            'parquet_info_path': '/gpfs/projects/krishna/linjli/bagel_example/editing/parquet_info/tifa_train_v3_td_ego_dir.json',
        },
        'tifa_v3_td_ego_dir_answer_only': {
            'data_dir': '/gpfs/projects/krishna/linjli/bagel_example/editing/tifa_train_v3_td_ego_dir_answer_only',
            'num_files': 5,
            'num_total_samples': 11204,
            'parquet_info_path': '/gpfs/projects/krishna/linjli/bagel_example/editing/parquet_info/tifa_train_v3_td_ego_dir_answer_only.json',
        },
        'tifa_v3_td_ego_dir_arrow': {
            'data_dir': '/gpfs/projects/krishna/linjli/bagel_example/editing/tifa_train_v3_td_ego_dir_arrow',
            'num_files': 5,
            'num_total_samples': 11290,
            'parquet_info_path': '/gpfs/projects/krishna/linjli/bagel_example/editing/parquet_info/tifa_train_v3_td_ego_dir_arrow.json',
        },
        'tifa_v3_td_ego_dir_arrow_answer_only': {
            'data_dir': '/gpfs/projects/krishna/linjli/bagel_example/editing/tifa_train_v3_td_ego_dir_arrow_answer_only',
            'num_files': 5,
            'num_total_samples': 11290,
            'parquet_info_path': '/gpfs/projects/krishna/linjli/bagel_example/editing/parquet_info/tifa_train_v3_td_ego_dir_arrow_answer_only.json',
        },
        'tifa_v3_td_ego_side': {
            'data_dir': '/gpfs/projects/krishna/linjli/bagel_example/editing/tifa_train_v3_td_ego_side',
            'num_files': 5,
            'num_total_samples': 13260,
            'parquet_info_path': '/gpfs/projects/krishna/linjli/bagel_example/editing/parquet_info/tifa_train_v3_td_ego_side.json',
        },
        'tifa_v3_td_ego_side_answer_only': {
            'data_dir': '/gpfs/projects/krishna/linjli/bagel_example/editing/tifa_train_v3_td_ego_side_answer_only',
            'num_files': 5,
            'num_total_samples': 13260,
            'parquet_info_path': '/gpfs/projects/krishna/linjli/bagel_example/editing/parquet_info/tifa_train_v3_td_ego_side_answer_only.json',
        },
        'tifa_v3_td_ego_side_arrow': {
            'data_dir': '/gpfs/projects/krishna/linjli/bagel_example/editing/tifa_train_v3_td_ego_side_arrow',
            'num_files': 5,
            'num_total_samples': 13270,
            'parquet_info_path': '/gpfs/projects/krishna/linjli/bagel_example/editing/parquet_info/tifa_train_v3_td_ego_side_arrow.json',
        },
        'tifa_v3_td_ego_side_arrow_answer_only': {
            'data_dir': '/gpfs/projects/krishna/linjli/bagel_example/editing/tifa_train_v3_td_ego_side_arrow_answer_only',
            'num_files': 5,
            'num_total_samples': 13270,
            'parquet_info_path': '/gpfs/projects/krishna/linjli/bagel_example/editing/parquet_info/tifa_train_v3_td_ego_side_arrow_answer_only.json',
        },
```

**Step 4: Commit**
```bash
git add data/dataset_info.py scripts/slurm/process_ego_data.slurm
git commit -m "feat: add ego variant data processing and dataset registry entries"
```

---

### Task 11: Create Phase 5 ego configs and training scripts

After ego data is processed. 4 answer-only configs + 1 CoT config (latent 32 default for ego).

**Files:**
- Create: `data/configs/tifa_v3_td_ego_side_answer_only.yaml`
- Create: `data/configs/tifa_v3_td_ego_side_arrow_answer_only.yaml`
- Create: `data/configs/tifa_v3_td_ego_dir_answer_only.yaml`
- Create: `data/configs/tifa_v3_td_ego_dir_arrow_answer_only.yaml`
- Create: `data/configs/tifa_v3_td_ego_side_default_l32.yaml` (CoT, latent 32)
- Create: 5 corresponding training scripts in `scripts/`

**Step 1: Create answer-only YAML configs**

All ego answer-only configs follow this pattern (example for td_ego_side):

`data/configs/tifa_v3_td_ego_side_answer_only.yaml`:
```yaml
unified_edit:
  dataset_names:
  - tifa_v3_td_ego_side_answer_only
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  is_mandatory: True
  num_used_data:
  - 13260
  weight: 1
```

Repeat for td_ego_side_arrow (13270), td_ego_dir (11204), td_ego_dir_arrow (11290).

**Step 2: Create ego CoT config (latent 32)**

`data/configs/tifa_v3_td_ego_side_default_l32.yaml`:
```yaml
unified_edit:
  dataset_names:
  - tifa_v3_td_ego_side
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  output_image_transform_args:
    image_stride: 16
    max_image_size: 512
    min_image_size: 256
  is_mandatory: True
  num_used_data:
  - 13260
  weight: 1
```

**Step 3: Create training scripts**

Each ego answer-only script: `--visual_gen False --mse_weight 0 --total_steps 2083 --save_every 400`
Ego CoT script: `--max_latent_size 32 --mse_weight 1 --total_steps 3125 --save_every 600`

Follow same pattern as Phase 1/2 scripts with appropriate paths and wandb names (p5_* prefix).

**Step 4: Commit**
```bash
git add data/configs/tifa_v3_td_ego_*.yaml scripts/train_tifa_v3_td_ego_*.sh
git commit -m "feat: add Phase 5 ego variant configs and training scripts"
```

---

### Task 12: Create combined ego+non-ego config (Phase 5 run #18)

**Files:**
- Create: `data/configs/tifa_v3_all_ego_answer_only.yaml`
- Create: `scripts/train_tifa_v3_all_ego_answer_only.sh`

`data/configs/tifa_v3_all_ego_answer_only.yaml`:
```yaml
unified_edit:
  dataset_names:
  - tifa_v3_dh_midpoint_answer_only
  - tifa_v3_td_midpoint_answer_only
  - tifa_v3_td_path_answer_only
  - tifa_v3_td_path_arrow_answer_only
  - tifa_v3_td_ego_dir_answer_only
  - tifa_v3_td_ego_dir_arrow_answer_only
  - tifa_v3_td_ego_side_answer_only
  - tifa_v3_td_ego_side_arrow_answer_only
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  is_mandatory: True
  num_used_data:
  - 2020
  - 3752
  - 9689
  - 9898
  - 11204
  - 11290
  - 13260
  - 13270
  weight: 1
```

Total samples: ~74,383. Training script: `--visual_gen False --mse_weight 0 --total_steps 1563 --save_every 300`

---

### Quick Reference: Submission Commands

```bash
# Phase 1 (all 4 in parallel)
sbatch --job-name=p1_dh_ao scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_dh_midpoint_answer_only.sh
sbatch --job-name=p1_td_mid_ao scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_midpoint_answer_only.sh
sbatch --job-name=p1_td_path_ao scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_path_answer_only.sh
sbatch --job-name=p1_td_parr_ao scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_path_arrow_answer_only.sh

# Phase 2 (can run in parallel with Phase 1)
sbatch --job-name=p2_l64 --time=24:00:00 scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_path_cot_l64.sh
sbatch --job-name=p2_l32 --time=12:00:00 scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_path_cot_l32.sh
sbatch --job-name=p2_l16 --time=12:00:00 scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_path_cot_l16.sh

# Phase 4 (after Phase 1+2 eval)
sbatch --job-name=p4_all_ao scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_all_answer_only.sh
sbatch --job-name=p4_all_cot --time=24:00:00 scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_all_cot_l64.sh

# Phase 5 (after ego data processing)
sbatch --job-name=p5_ego_side_ao scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_ego_side_answer_only.sh
# ... etc
```

### Phase 3 Note

Phase 3 configs and scripts depend on Phase 2 results (which latent size is best). Create them after evaluating Phase 2 checkpoints. They follow the same pattern as Phase 2 but for different datasets (dh_midpoint, td_midpoint, td_path_arrow) using the winning latent size.
