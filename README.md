<p align="center">
    <img src="assets/logo.png" width="40%"> <br>
</p>

# IPT: Imaginative Spatial Token

This repository contains training and evaluation code for **IPT** (Imaginative Spatial Token), which extends [ThinkMorph](https://thinkmorph.github.io/) with spatial Visual Chain-of-Thought (VCoT) reasoning. The model learns to imagine novel viewpoints as intermediate reasoning steps for answering spatial questions about indoor environments.

Built on [BAGEL-7B-MoT](https://github.com/ByteDance-Seed/Bagel) (Qwen2.5-7B LLM + SigLIP ViT + Flux VAE with Mixture-of-Thinking decoder layers).

## Overview

Given a top-down view (and optionally egocentric views) of an indoor scene, the model answers spatial reasoning questions (e.g., "Which object is on your left at waypoint M1?"). With VCoT, the model first *imagines* a sideview perspective as a generated image, then uses that visual thought to answer.

**Training variants explored:**

| Variant | Output Format | Image Gen | Key Flags |
|---------|--------------|-----------|-----------|
| Answer-Only (AO) | `<answer>A</answer>` | No | `--visual_gen False --mse_weight 0` |
| Text CoT | `<think>reasoning</think><answer>A</answer>` | No | `--visual_gen False --mse_weight 0` |
| Visual CoT (VCoT) | `<think>desc</think><image_start>[img]<image_end><answer>A</answer>` | Yes | `--mse_weight 1` |
| Multimodal CoT (MMCoT) | `<think>detailed reasoning</think><image_start>[img]<image_end><answer>A</answer>` | Yes | `--mse_weight 1` |
| Mixed (VCoT+AO) | 50% VCoT + 50% AO samples | Yes | `--mse_weight 1` |

## Setup

### Environment

```bash
git clone <repo-url>
cd SpatialReasoning_train
conda create -n thinkmorph python=3.10 -y
conda activate thinkmorph
pip install -r requirements.txt
```

### Model Checkpoint

Download BAGEL-7B-MoT as the base model:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAGEL-AI/BAGEL-7B-MoT",
    local_dir="models/BAGEL-7B-MoT",
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)
```

## Training

### Data Preparation

1. **Training data** is stored as parquet files with the following schema:

```python
{
    "image_list": [input_image_bytes, ...],          # Top-down + ego images
    "num_input_images": 3,                            # Number of input images
    "instruction_list": [system_prompt, question],    # System prompt + question with <img> tags
    "output_text_list": [model_response],             # <think>...</think><answer>A</answer>
}
```

For VCoT, the output includes image generation tokens and `image_list` contains the target sideview image after the input images.

2. **Register your dataset** in `data/dataset_info.py`:

```python
'my_dataset': {
    'data_dir': '/path/to/parquet/directory',
    'num_files': 5,
    'num_total_samples': 11204,
}
```

3. **Create a YAML config** in `data/configs/`:

```yaml
unified_edit:
  dataset_names:
  - my_dataset
  image_transform_args:        # VAE input encoding
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:    # ViT input encoding
    image_stride: 14
    max_image_size: 518
    min_image_size: 224
  output_image_transform_args: # Output image resolution (VCoT only)
    image_stride: 16
    max_image_size: 512        # 512 = latent-32, 1024 = latent-64
    min_image_size: 256
  is_mandatory: True
  num_used_data:
  - 11204
  weight: 1
```

For mixed training (multiple datasets), list them under `dataset_names` with corresponding `num_used_data` entries:

```yaml
unified_edit:
  dataset_names:
  - my_vcot_dataset          # VCoT samples (sideview generation)
  - my_ao_dataset            # AO samples (answer-only)
  num_used_data:
  - 11204                    # VCoT count
  - 11204                    # AO count
  # ... (same transforms as above)
```

### Running Training

#### Local (multi-GPU)

```bash
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/my_config.yaml \
  --model_path /path/to/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --finetune_from_hf True \
  --auto_resume True \
  --resume_model_only True \
  --finetune-from-ema True \
  --resume_from /path/to/BAGEL-7B-MoT \
  --results_dir /path/to/output \
  --checkpoint_dir /path/to/checkpoints \
  --lr 1e-5 \
  --num_workers 8 \
  --max_latent_size 64 \
  --max_num_tokens 32768 \
  --expected_num_tokens 24576 \
  --mse_weight 1 \
  --ce_weight 1 \
  --ema 0.999 \
  --total_steps 10000 \
  --save_every 1000 \
  --wandb_project my_project \
  --wandb_name my_run
```

#### Slurm (HPC)

Use the provided Slurm template:

```bash
sbatch --job-name=my_run scripts/slurm/train_tifa_v3.slurm scripts/my_training_script.sh
```

Override time limit for longer runs:

```bash
sbatch --job-name=my_run --time=48:00:00 scripts/slurm/train_tifa_v3.slurm scripts/my_training_script.sh
```

The Slurm template (`scripts/slurm/train_tifa_v3.slurm`) requests 8 GPUs, 64 CPUs, 1600G RAM, and handles conda activation and W&B directory setup.

### Training Variant Examples

**Answer-Only** (fastest, ~3x faster than VCoT):
```bash
# Key flags: disable image generation
--visual_gen False --mse_weight 0 --ce_weight 1
```

**Visual CoT** (sideview image generation):
```bash
# Key flags: enable image generation with MSE loss
--mse_weight 1 --ce_weight 1 --max_latent_size 64
```

**Text CoT** (text-only reasoning, no image generation):
```bash
# Same flags as AO, but training data has <think> reasoning
--visual_gen False --mse_weight 0 --ce_weight 1
```

**Mixed VCoT+AO** (prevents VCoT commitment collapse):
```bash
# Uses mixed YAML config with both VCoT and AO datasets
--mse_weight 1 --ce_weight 1 --max_latent_size 64
```

Ready-to-use training scripts are provided in `scripts/`:

| Script | Variant | Dataset |
|--------|---------|---------|
| `train_tifa_v3_td_ego_dir_ao_10k.sh` | Answer-Only | td_ego_dir |
| `train_tifa_v3_td_ego_dir_vcot_15k.sh` | VCoT (latent-32) | td_ego_dir |
| `train_tifa_v3_td_ego_dir_text_cot.sh` | Text CoT | td_ego_dir |
| `train_tifa_v3_td_ego_dir_mmcot.sh` | MMCoT | td_ego_dir |
| `train_tifa_v3_td_ego_dir_mixed.sh` | Mixed VCoT+AO | td_ego_dir |

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-5 | Constant schedule with 2000-step warmup |
| Optimizer | AdamW | beta1=0.9, beta2=0.95, eps=1e-15, weight_decay=0 |
| EMA decay | 0.999 | Exponential moving average of weights |
| Gradient clipping | 1.0 | Max gradient norm |
| Token packing target | 24,576 | Soft target per GPU per step (NaViT-style) |
| Token hard limit | 32,768 | Max tokens per GPU per step |
| Distributed strategy | FSDP HYBRID_SHARD | Across 8 GPUs |
| Loss | CE + MSE | Weighted sum with per-GPU normalization |

### Token Costs (1024x1024 images)

| Component | Tokens |
|-----------|--------|
| Input image (VAE + ViT) | 5,465 |
| Ego image (512x512, VAE + ViT) | 2,393 |
| Output sideview (latent-64, 1024x1024) | 9,561 |
| Output sideview (latent-32, 512x512) | 3,417 |

### Critical: `--max_latent_size`

This parameter controls the position embedding grid for **all** VAE images (input AND output). It must always be >= the latent size of the largest image:

- For 1024x1024 inputs: `--max_latent_size 64` (1024 / 16 = 64)
- Setting it lower causes **CUDA index-out-of-bounds crashes**
- To control output image resolution, use `output_image_transform_args` in the YAML config instead

### Checkpoint Conversion

Training saves FSDP sharded checkpoints. Convert to safetensors for evaluation:

```bash
python scripts/convert_sharded_to_full.py \
  --checkpoint_path /path/to/checkpoint/step_5000 \
  --output_path /path/to/converted \
  --model_path /path/to/BAGEL-7B-MoT \
  --convert_ema
```

This requires ~400GB RAM (use `--gpus=2 --mem=400G` on Slurm).

## Evaluation

Evaluations use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). See the [eval repo](https://github.com/linjieli222/SpatialReasoning_Eval) for details.

```bash
export THINKMORPH_MODEL_PATH=/path/to/converted_checkpoint

# Text-only eval (for AO and TextCoT models)
torchrun --nproc-per-node=2 run.py \
  --model bagel_mot \
  --data AI2ThorPT2P_td_ego_dir \
  --work-dir $THINKMORPH_MODEL_PATH/eval

# VCoT eval (for VCoT and MMCoT models)
export THINKMORPH_SAVE_DIR=$THINKMORPH_MODEL_PATH/eval/vcot_images
torchrun --nproc-per-node=2 run.py \
  --model bagel_mot_vcot \
  --data AI2ThorPT2P_td_ego_dir \
  --work-dir $THINKMORPH_MODEL_PATH/eval
```

**Model config selection is critical:**
- `bagel_mot`: text-only eval (for AO, TextCoT models)
- `bagel_mot_vcot`: image generation eval (for VCoT, MMCoT models)
- Using `bagel_mot` for VCoT/MMCoT models produces invalid (near-0%) results

## Project Structure

```
.
├── train/
│   └── pretrain_unified_navit.py    # Main training script
├── data/
│   ├── configs/                     # YAML dataset configs
│   ├── dataset_info.py              # Dataset registry
│   ├── dataset_base.py              # Base dataset + token packing
│   └── interleave_datasets/         # Multi-dataset interleaving
├── modeling/
│   ├── bagel/                       # BAGEL model architecture
│   ├── qwen2/                       # Qwen2.5 LLM
│   └── siglip/                      # SigLIP Vision Transformer
├── scripts/
│   ├── train_tifa_v3_*.sh           # Training launch scripts
│   ├── slurm/                       # Slurm job templates
│   └── convert_sharded_to_full.py   # Checkpoint conversion
├── docs/
│   ├── plans/                       # Experiment design docs
│   └── eval_results_*.md            # Evaluation results
└── visualization_scripts/           # Result visualization tools
```

## Citation

```bibtex
@article{gu2025thinkmorph,
  title={ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning},
  author={Gu, Jiawei and Hao, Yunzhuo and Wang, Huichen Will and Li, Linjie and Shieh, Michael Qizhe and Choi, Yejin and Krishna, Ranjay and Cheng, Yu},
  journal={arXiv preprint arXiv:2510.27492},
  year={2025}
}
```

## License

See [LICENSE](LICENSE).
