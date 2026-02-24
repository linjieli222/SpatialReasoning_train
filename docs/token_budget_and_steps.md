# Token Budget & Steps-per-Epoch Analysis

**Date:** 2026-02-24
**Dataset:** TIFA v3 td_path (9,689 samples)
**Cluster:** Tillicum, 8× H200 GPUs per run

## NaViT Token Packing Overview

BAGEL uses NaViT-style token packing — there is no fixed batch size. Instead, samples are packed into sequences up to a token budget:

- **`expected_num_tokens = 24,576`** per GPU (soft target: yield when reached)
- **`max_num_tokens = 32,768`** per GPU (hard ceiling: never exceed)
- **`max_num_tokens_per_sample = 24,576`** (skip any single sample exceeding this)
- **8 GPUs** per job → effective batch = `samples_per_gpu × 8`

**Packing rule:** samples are added to a batch until `curr + next_sample > max_num_tokens` (overflow) or `curr >= expected_num_tokens` (target reached). The effective batch size depends entirely on per-sample token count.

**Overhead per sequence plan entry:** Each text segment adds BOS + EOS tokens (+2), each image segment adds `<|startofimage|>` + `<|endofimage|>` (+2). Total overhead = `2 × len(sequence_plan)`.

## Image Token Costs

All TIFA v3 images are 1024×1024.

### Input Images (conditioning, no loss)

Each input image generates tokens for **two** encoders:

| Encoder | Stride | Image Size | Tokens | Seq Entries |
|---------|--------|------------|--------|-------------|
| VAE | 16 | 1024×1024 | 4,096 | 1 (`vae_image`, loss=0) |
| ViT | 14 | 518×518 (resized) | 1,369 | 1 (`vit_image`) |
| **Total** | | | **5,465** | **2** |

For multi-input (ego variants with >1 input image), inputs are resized to 512×512 max:

| Encoder | Stride | Image Size | Tokens | Seq Entries |
|---------|--------|------------|--------|-------------|
| VAE | 16 | 512×512 | 1,024 | 1 |
| ViT | 14 | 512×512 | 1,369 | 1 |
| **Total** | | | **2,393** | **2** |

### Output Images (sideview generation, with loss)

Each output image generates tokens for **three** encoder passes:

| Component | Description | Tokens (l64) | Tokens (l32) | Tokens (l16) |
|-----------|-------------|-------------|-------------|-------------|
| VAE (loss=1) | Denoising target | 4,096 | 1,024 | 256 |
| VAE (loss=0) | CFG conditioning copy | 4,096 | 1,024 | 256 |
| ViT | Visual understanding | 1,369 | 1,369 | 1,369 |
| **Total** | | **9,561** | **3,417** | **1,881** |
| **Seq entries** | | **3** | **3** | **3** |

Output image pixel size: `latent_size × 16` (VAE downsample factor)
- latent 64 → 1024×1024
- latent 32 → 512×512
- latent 16 → 256×256

**Note:** CFG dropout during training may randomly skip the VAE(loss=0) and/or ViT copies for output images, reducing the *actual* sequence length below the *packing budget*. The batch logs show actual lengths of ~20–24k per GPU for settings that budget ~30k, due to this dropout.

## Text Token Costs

Measured with the BAGEL-7B-MoT tokenizer on actual training data:

| Component | Setting | Tokens |
|-----------|---------|--------|
| System prompt (AO) | `"Answer the question..."` | 30 |
| System prompt (CoT) | `"Let's think step by step..."` | 96 |
| Question + choices | Typical td_path MCQ | ~68 |
| AO output | `<answer>C</answer>` | 6 |
| VCoT think text | `<think>short desc</think><image_start>` | ~17 |
| VCoT end text | `<image_end><answer>C</answer>` | ~9 |
| TextCoT output | `<think>detailed reasoning...</think><answer>C</answer>` | ~200 |
| MMCoT part1 | `<think>thought_0</think><image_start>` | ~155 |
| MMCoT part2 | `<image_end><think>thought_1</think><answer>C</answer>` | ~135 |

## Per-Setting Token Breakdown

### Answer-Only (AO)

No output image. `visual_gen=False, mse_weight=0`.

| Component | Raw Tokens | Seq Entries |
|-----------|-----------|-------------|
| System prompt | 30 | 1 |
| Input image (1024²) | 5,465 | 2 |
| Question text | 68 | 1 |
| Output `<answer>` | 6 | 1 |
| **Subtotal** | **5,569** | **5** |
| Overhead (5 × 2) | 10 | |
| **Packing total** | **5,579** | |

### TextCoT

No output image. `visual_gen=False, mse_weight=0`. Longer text reasoning.

| Component | Raw Tokens | Seq Entries |
|-----------|-----------|-------------|
| System prompt | 96 | 1 |
| Input image (1024²) | 5,465 | 2 |
| Question text | 68 | 1 |
| Output (think + answer) | ~200 | 1 |
| **Subtotal** | **~5,829** | **5** |
| Overhead (5 × 2) | 10 | |
| **Packing total** | **~5,839** | |

### VCoT (Visual CoT) — Latent 64

Output sideview at 1024×1024. `mse_weight=1`.

| Component | Raw Tokens | Seq Entries |
|-----------|-----------|-------------|
| System prompt | 96 | 1 |
| Input image (1024²) | 5,465 | 2 |
| Question text | 68 | 1 |
| Output think text | 17 | 1 |
| Output sideview (1024²) | 9,561 | 3 |
| Output end text | 9 | 1 |
| **Subtotal** | **15,216** | **9** |
| Overhead (9 × 2) | 18 | |
| **Packing total** | **15,234** | |

### VCoT — Latent 32

Output sideview at 512×512.

| Component | Raw Tokens | Seq Entries |
|-----------|-----------|-------------|
| System prompt | 96 | 1 |
| Input image (1024²) | 5,465 | 2 |
| Question text | 68 | 1 |
| Output think text | 17 | 1 |
| Output sideview (512²) | 3,417 | 3 |
| Output end text | 9 | 1 |
| **Subtotal** | **9,072** | **9** |
| Overhead (9 × 2) | 18 | |
| **Packing total** | **9,090** | |

### VCoT — Latent 16

Output sideview at 256×256.

| Component | Raw Tokens | Seq Entries |
|-----------|-----------|-------------|
| System prompt | 96 | 1 |
| Input image (1024²) | 5,465 | 2 |
| Question text | 68 | 1 |
| Output think text | 17 | 1 |
| Output sideview (256²) | 1,881 | 3 |
| Output end text | 9 | 1 |
| **Subtotal** | **7,536** | **9** |
| Overhead (9 × 2) | 18 | |
| **Packing total** | **7,554** | |

### MMCoT (Multimodal CoT)

Output sideview at 1024×1024 with longer reasoning text. `mse_weight=1`.

| Component | Raw Tokens | Seq Entries |
|-----------|-----------|-------------|
| System prompt | 96 | 1 |
| Input image (1024²) | 5,465 | 2 |
| Question text | 68 | 1 |
| Output part1 (thought_0) | ~155 | 1 |
| Output sideview (1024²) | 9,561 | 3 |
| Output part2 (thought_1) | ~135 | 1 |
| **Subtotal** | **~15,480** | **9** |
| Overhead (9 × 2) | 18 | |
| **Packing total** | **~15,498** | |

## Packing & Steps-per-Epoch Summary

All settings use `expected_num_tokens=24,576` and `max_num_tokens=32,768` per GPU, with 8 GPUs.

`total_samples` in WandB is accurate — it counts the real number of packed training samples per step (no padding inflation since `use_flex=False`).

| Setting | Tokens/sample | Samples/GPU | Samples/step | Steps/epoch | Total steps | ~Epochs |
|---------|--------------|-------------|-------------|-------------|-------------|---------|
| **AO** | 5,579 | 5 | 40 | 242 | 9,000+ | 37+ |
| **TextCoT** | 5,839 | 5 | 40 | 242 | 3,000 | 12.4 |
| **VCoT l16** | 7,554 | 4 | 32 | 303 | 5,200+ | 17+ |
| **VCoT l32** | 9,090 | 3 | 24 | 404 | 4,500+ | 11+ |
| **VCoT l64** | 15,234 | 2 | 16 | 606 | 7,000+ | 11.6+ |
| **MMCoT** | 15,498 | 2 | 16 | 606 | 7,000+ | 11.6+ |

### How to read this table

- **Tokens/sample**: Total tokens counted for packing (raw + 2 per sequence entry)
- **Samples/GPU**: How many samples fit in one GPU's packing budget before exceeding `expected_num_tokens`
- **Samples/step**: `samples_per_gpu × 8` (total effective batch size)
- **Steps/epoch**: `ceil(9689 / samples_per_step)` (one pass through the dataset)
- **Total steps**: Latest checkpoint available (training may still be running)
- **~Epochs**: `total_steps / steps_per_epoch`

### Epoch Milestones

| Setting | Steps/epoch | 5 epochs (step) | 10 epochs (step) |
|---------|------------|-----------------|-------------------|
| **AO** | 242 | 1,210 | 2,420 |
| **TextCoT** | 242 | 1,210 | 2,420 |
| **VCoT l16** | 303 | 1,515 | 3,030 |
| **VCoT l32** | 404 | 2,020 | 4,040 |
| **VCoT l64** | 606 | 3,030 | 6,060 |
| **MMCoT** | 606 | 3,030 | 6,060 |

### Available Checkpoints

| Setting | Save interval | Checkpoints | Nearest 5-epoch ckpt | Nearest 10-epoch ckpt |
|---------|--------------|-------------|----------------------|-----------------------|
| **AO** | 300 | 300, 600, ..., 9000 | **1,200** (5.0 ep) | **2,400** (9.9 ep) |
| **TextCoT** | 500 → 200 | 500, 1000, 1500, 2000, then 2200, 2400, ..., 3000 | **1,200** (5.0 ep) | **2,400** (9.9 ep) |
| **VCoT l16** | 400 | 400, 800, ..., 5200 | **1,600** (5.3 ep) | **2,800** or **3,200** |
| **VCoT l32** | 500 | 500, 1000, ..., 4500 | **2,000** (5.0 ep) | **4,000** (9.9 ep) |
| **VCoT l64** | 1000 | 1000, 2000, ..., 7000 | **3,000** (5.0 ep) | **6,000** (9.9 ep) |
| **MMCoT** | 1000 | 1000, 2000, ..., 7000 | **3,000** (5.0 ep) | **6,000** (9.9 ep) |

**Note:** TextCoT originally saved every 500 steps up to 2000. Extended run (3k) saves every 200 steps from step 2000 onward.

### Packing verification

The packer adds samples until `curr >= expected_num_tokens (24,576)`:

| Setting | After N-1 samples | After N samples | Fits? |
|---------|-------------------|-----------------|-------|
| AO (N=5) | 4 × 5,579 = 22,316 | 5 × 5,579 = 27,895 ✓ | 27,895 < 32,768 |
| TextCoT (N=5) | 4 × 5,839 = 23,356 | 5 × 5,839 = 29,195 ✓ | 29,195 < 32,768 |
| VCoT l16 (N=4) | 3 × 7,554 = 22,662 | 4 × 7,554 = 30,216 ✓ | 30,216 < 32,768 |
| VCoT l32 (N=3) | 2 × 9,090 = 18,180 | 3 × 9,090 = 27,270 ✓ | 27,270 < 32,768 |
| VCoT l64 (N=2) | 1 × 15,234 = 15,234 | 2 × 15,234 = 30,468 ✓ | 30,468 < 32,768 |
| MMCoT (N=2) | 1 × 15,498 = 15,498 | 2 × 15,498 = 30,996 ✓ | 30,996 < 32,768 |

## Token Budget Dominance

For all settings, image tokens dominate the packing budget:

| Setting | Image tokens | Text tokens | Image % |
|---------|-------------|-------------|---------|
| AO | 5,465 | 104 | 98% |
| TextCoT | 5,465 | 364 | 94% |
| VCoT l64 | 15,026 | 190 | 99% |
| VCoT l32 | 8,882 | 190 | 98% |
| VCoT l16 | 7,346 | 190 | 97% |
| MMCoT | 15,026 | 454 | 97% |

The output image (when present) is the single largest cost:

| Setting | Input image | Output image | Output / Total |
|---------|-------------|-------------|----------------|
| AO | 5,465 | 0 | 0% |
| VCoT l64 | 5,465 | 9,561 | 63% |
| VCoT l32 | 5,465 | 3,417 | 38% |
| VCoT l16 | 5,465 | 1,881 | 25% |

## Formulas

```
tokens_per_sample = sum(text_tokens) + sum(image_tokens) + 2 × num_sequence_entries

samples_per_gpu = N  where  (N-1) × tokens_per_sample < expected_num_tokens
                     and    N × tokens_per_sample < max_num_tokens

samples_per_step = samples_per_gpu × num_gpus

steps_per_epoch = ceil(num_samples / samples_per_step)

total_epochs = total_steps / steps_per_epoch
```

### Image token formulas

```
vae_tokens(image) = (image_size / 16)² = latent_size²
vit_tokens(image) = (min(image_size, 518) / 14)²

input_image_tokens  = vae_tokens + vit_tokens                    (2 seq entries)
output_image_tokens = 2 × vae_tokens + vit_tokens                (3 seq entries)
                      [loss copy]   [CFG copy]  [understanding]
```
