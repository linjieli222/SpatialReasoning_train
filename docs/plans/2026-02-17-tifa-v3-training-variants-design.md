# TIFA v3 Single-Task Training Variant Experiments

**Date:** 2026-02-17
**Budget:** ~600-900 GPU-hours out of 10K total (6-9%)
**Cluster:** Tillicum, 8x H200 GPUs per run
**Eval:** Existing SpatialReasoning eval pipeline

## Research Questions

1. **CoT modality**: Which reasoning format helps most — text-only CoT, visual CoT (image generation), or multimodal CoT (text + image)?
2. **Annotation style**: Which spatial annotation works best (midpoint vs path vs path+arrow, dollhouse vs top-down)?
3. **Latent size**: What VCoT image resolution is needed (latent 64/32/16)?
4. **Egocentric input**: Does adding ego-perspective images improve performance?
5. **Data scaling**: Does combining multiple datasets improve over single-dataset training?

## Datasets

### Available (processed on Tillicum)

| Dataset | Samples | Viewpoint | Annotation | Input Images |
|---------|---------|-----------|------------|-------------|
| dh_midpoint | 2,020 | Dollhouse | Midpoint | 1 (topdown) |
| td_midpoint | 3,752 | Top-down | Midpoint | 1 (topdown) |
| td_path | 9,689 | Top-down | Path | 1 (topdown) |
| td_path_arrow | 9,898 | Top-down | Path+Arrow | 1 (topdown) |

Each has `default` (visual CoT) and `answer_only` variants. Two additional variants are planned:

### Training Variants

| Variant | Output format | Image gen | Training flags | Description |
|---------|--------------|-----------|----------------|-------------|
| `answer_only` | `<answer>A</answer>` | No | `--visual_gen False --mse_weight 0` | Direct answer, no reasoning |
| `text_cot` | `<think>detailed reasoning</think><answer>A</answer>` | No | `--visual_gen False --mse_weight 0` | Text-only CoT with longer, more detailed reasoning steps |
| `default` (vcot) | `<think>desc</think><image_start>[img]<image_end><answer>A</answer>` | Yes | `--mse_weight 1` | Visual CoT — short text + generated sideview image |
| `mm_cot` | `<think>detailed reasoning</think><image_start>[img]<image_end><answer>A</answer>` | Yes | `--mse_weight 1` | Multimodal CoT — longer text reasoning + generated sideview image |

**Data status for text_cot and mm_cot:** PENDING — HuggingFace data not yet generated. These variants require longer, more detailed text reasoning chains compared to the current `default` variant (which only uses the brief sideview description as think text). Once data is available, `transform_tifa_train_v3.py` needs new transform functions (`text_cot`, `mm_cot`) added to `VARIANTS` and `TRANSFORM_FNS`.

### Pending data processing

| Dataset | Samples | Viewpoint | Annotation | Input Images |
|---------|---------|-----------|------------|-------------|
| td_ego_dir | 11,204 | Top-down + Ego | Direction | 3 (topdown + 2 ego) |
| td_ego_dir_arrow | 11,290 | Top-down + Ego | Dir+Arrow | 3 (topdown + 2 ego) |
| td_ego_side | 13,260 | Top-down + Ego | Side | 3 (topdown + 2 ego) |
| td_ego_side_arrow | 13,270 | Top-down + Ego | Side+Arrow | 3 (topdown + 2 ego) |

Requires running `transform_tifa_train_v3.py` for ego configs before Phase 5.

## Token Budget Analysis

All images in TIFA v3 are 1024x1024 RGB.

### Transform parameters

| Transform | Stride | Max Size | Min Size |
|-----------|--------|----------|----------|
| VAE (image_transform) | 16 | 1024 | 512 |
| ViT (vit_image_transform) | 14 | 518 | 224 |

### Image token costs

**Non-ego input image (1024x1024):**
- VAE: 1024x1024 / 16^2 = 4,096 tokens
- ViT: 1024 resized to 518, 518x518 / 14^2 = 1,369 tokens
- **Total: 5,465 tokens/image**

**Ego input image (resized to 512x512 by INPUT_MAX_SIZE when >1 input):**
- VAE: 512x512 / 16^2 = 1,024 tokens
- ViT: 512 resized to 518, 518x518 / 14^2 = 1,369 tokens
- **Total: 2,393 tokens/image**

**Output sideview image (3 entries: VAE-loss + VAE-cond + ViT):**

| Latent Size | Pixel Resolution | VAE-loss | VAE-cond | ViT | Total |
|-------------|-----------------|----------|----------|-----|-------|
| 64 | 1024x1024 | 4,096 | 4,096 | 1,369 | 9,561 |
| 32 | 512x512 | 1,024 | 1,024 | 1,369 | 3,417 |
| 16 | 256x256 | 256 | 256 | 1,369 | 1,881 |

### Per-sample token totals

| Variant | Input tokens | Output img tokens | Text (~) | Total |
|---------|-------------|-------------------|----------|-------|
| Non-ego answer-only | 5,465 (1 img) | 0 | ~100 | ~5,565 |
| Non-ego text_cot | 5,465 (1 img) | 0 | ~400 | ~5,865 |
| Non-ego vcot (default) latent 64 | 5,465 (1 img) | 9,561 | ~200 | ~15,226 |
| Non-ego mm_cot latent 64 | 5,465 (1 img) | 9,561 | ~400 | ~15,426 |
| Non-ego vcot latent 32 | 5,465 (1 img) | 3,417 | ~200 | ~9,082 |
| Non-ego vcot latent 16 | 5,465 (1 img) | 1,881 | ~200 | ~7,546 |
| Ego answer-only | 7,179 (3 imgs) | 0 | ~100 | ~7,279 |
| Ego CoT latent 32 | 7,179 (3 imgs) | 3,417 | ~200 | ~10,796 |

Note: text_cot and mm_cot have ~400 text tokens (vs ~100-200 for answer-only/vcot) due to longer reasoning chains. Exact count depends on data once generated.

### Effective batch size and step normalization

Packing target: `expected_num_tokens = 24,576` per GPU, 8 GPUs.

| Variant | Tokens/sample | Samples/GPU/step | Samples/step (8 GPU) | Steps for 50K samples |
|---------|--------------|------------------|---------------------|----------------------|
| Non-ego answer-only | ~5,565 | ~4 | ~32 | ~1,563 |
| Non-ego CoT latent 64 | ~15,226 | ~1 | ~8 | ~6,250 |
| Non-ego CoT latent 32 | ~9,082 | ~2-3 | ~20 | ~2,500 |
| Non-ego CoT latent 16 | ~7,546 | ~3 | ~24 | ~2,083 |
| Ego answer-only | ~7,279 | ~3 | ~24 | ~2,083 |
| Ego CoT latent 32 | ~10,796 | ~2 | ~16 | ~3,125 |

## Default Latent Sizes

- **Non-ego CoT: latent 64** (1024x1024 sideview matches 1024x1024 input)
- **Ego CoT: latent 32** (512x512 sideview matches 512x512 resized inputs)

**IMPORTANT:** `--max_latent_size` must always be set to 64 (or the max of input latent size),
because it controls the position embedding grid for ALL VAE images (input + output).
To vary the output sideview resolution, use `output_image_transform_args` in the YAML config
to resize the output image before VAE encoding. Do NOT reduce `--max_latent_size` below
the input image's latent dimension or it will cause CUDA index-out-of-bounds errors.

## Experiment Plan

All runs normalize to ~50K effective samples for fair comparison.
Checkpoint every ~1 epoch equivalent. Best checkpoint selected by eval (early stopping).

### Phase 1: Answer-only annotation sweep (4 runs)

Purpose: Which annotation style works best without CoT?

| # | Dataset | Steps | Save every | Est. GPU-hrs |
|---|---------|-------|------------|-------------|
| 1 | dh_midpoint answer_only | 1,563 | 300 | ~14 |
| 2 | td_midpoint answer_only | 1,563 | 300 | ~14 |
| 3 | td_path answer_only | 1,563 | 300 | ~14 |
| 4 | td_path_arrow answer_only | 1,563 | 300 | ~14 |

**Subtotal: ~56 GPU-hours.** All 4 can run in parallel.

### Phase 2: CoT latent size sweep (3 runs)

Purpose: What VCoT image resolution is needed? Uses td_path (largest non-ego dataset).

| # | Dataset | Latent | Steps | Save every | Est. GPU-hrs |
|---|---------|--------|-------|------------|-------------|
| 5 | td_path CoT | 64 | 6,250 | 1,000 | ~153 |
| 6 | td_path CoT | 32 | 2,500 | 500 | ~62 |
| 7 | td_path CoT | 16 | 2,083 | 400 | ~51 |

**Subtotal: ~266 GPU-hours.**

### Phase 2b: CoT modality comparison (blocked on data)

Purpose: Compare text-only CoT, visual CoT, and multimodal CoT. Uses td_path (largest non-ego).
Blocked on: text_cot and mm_cot data generation on HuggingFace.

| # | Dataset | Variant | Steps (5 epochs) | Est. GPU-hrs |
|---|---------|---------|-------|-------------|
| 5b | td_path text_cot | text_cot | ~1,514 | ~14 |
| 5c | td_path mm_cot | mm_cot (best latent) | ~6,056 (l64) | ~153 |

**Subtotal: ~167 GPU-hours.**

text_cot trains at answer-only speed (~4 sec/step, no image gen). mm_cot trains at vcot speed.
Compare against P1 answer-only and P1/P2 vcot results on same dataset.

### Phase 3: Best CoT latent across annotations (3 runs)

Purpose: Does CoT benefit depend on annotation type? Uses best latent from Phase 2.

| # | Dataset | Latent | Steps | Est. GPU-hrs |
|---|---------|--------|-------|-------------|
| 8 | dh_midpoint CoT | (best) | (varies) | ~51-153 |
| 9 | td_midpoint CoT | (best) | (varies) | ~51-153 |
| 10 | td_path_arrow CoT | (best) | (varies) | ~51-153 |

**Subtotal: ~153-459 GPU-hours** (depends on best latent).

### Phase 4: Combined datasets (2 runs)

Purpose: Does more data help?

| # | Dataset | Steps | Est. GPU-hrs |
|---|---------|-------|-------------|
| 11 | All non-ego answer_only (~25K) | 1,563 | ~14 |
| 12 | All non-ego CoT (best latent, ~25K) | (varies) | ~51-153 |

**Subtotal: ~65-167 GPU-hours.**

### Phase 5: Ego variants (blocked on data processing) (4-6 runs)

Purpose: Does egocentric input improve spatial reasoning?

| # | Dataset | Variant | Latent | Steps | Est. GPU-hrs |
|---|---------|---------|--------|-------|-------------|
| 13 | td_ego_side answer_only | answer_only | - | 2,083 | ~17 |
| 14 | td_ego_side_arrow answer_only | answer_only | - | 2,083 | ~17 |
| 15 | td_ego_dir answer_only | answer_only | - | 2,083 | ~17 |
| 16 | td_ego_dir_arrow answer_only | answer_only | - | 2,083 | ~17 |
| 17 | Best ego dataset CoT | CoT | 32 | 3,125 | ~77 |
| 18 | All ego+non-ego answer_only combined | answer_only | - | 1,563 | ~14 |

**Subtotal: ~159 GPU-hours.**

## Budget Summary

| Phase | Runs | GPU-hours | Dependency |
|-------|------|-----------|------------|
| 1: Annotation sweep (AO) | 4 | ~56 | None |
| 2: Latent sweep (vcot) | 3 | ~266 | None |
| 2b: CoT modality (text_cot, mm_cot) | 2 | ~167 | Data generation |
| 3: CoT across annotations | 3 | ~153-459 | Phase 2 results |
| 4: Combined datasets | 2 | ~65-167 | Phase 2 results |
| 5: Ego variants | 4-6 | ~159 | Data processing |
| **Total** | **18-22** | **~866-1,274** | |

Conservative estimate: **~870-1,270 GPU-hours** (9-13% of 10K budget), ~$780-$1,150.

## Implementation Checklist

- [x] Create YAML configs for each run (Phase 1-4: 12 configs)
- [x] Create training shell scripts with correct `--visual_gen`, `--mse_weight`, `--max_latent_size`
- [x] Create Slurm submission scripts with WANDB_DIR redirect
- [x] Fix model path in all scripts (`/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT`)
- [x] Fix critical bug: edit_dataset.py dropping instrs[1] (question+choices)
- [x] Regenerate data with fixed `_build_instruction_list` (2-part format with `<img>` tags)
- [x] Run `transform_tifa_train_v3.py` for ego configs
- [x] Register ego datasets in `data/dataset_info.py`
- [x] Set up auto convert+eval hook (`on_train_complete.sh`)
- [x] Set up trainlog run tracker
- [ ] Fix `--max_latent_size` in l32/l16 scripts (must be 64, use YAML for output size)
- [ ] Regenerate td_path and td_path_arrow data (in progress)
- [ ] Create combined YAML configs (all answer_only, all CoT)
- [ ] Verify vcot actually generates `<image_start>` tokens (P0 debug run in progress)
- [ ] **DATA NEEDED:** Generate text_cot data — longer text reasoning, no sideview image
- [ ] **DATA NEEDED:** Generate mm_cot data — longer text reasoning + sideview image
- [ ] Add `text_cot` and `mm_cot` transform functions to `transform_tifa_train_v3.py`
- [ ] Add `text_cot` system prompt (text thinking instructions without visual thinking)
- [ ] Create YAML configs and training scripts for text_cot and mm_cot variants
- [ ] Set up eval pipeline for text_cot (needs `bagel_mot` model config, text-only output)
- [ ] Set up eval pipeline for mm_cot (needs `bagel_mot_vcot` model config)

## Training Parameters (shared)

```
--model_path /gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT
--layer_module Qwen2MoTDecoderLayer
--finetune_from_hf True
--auto_resume True
--resume_model_only True
--finetune-from-ema True
--lr 1e-5
--num_workers 8
--max_num_tokens 32768
--max_num_tokens_per_sample 24576
--expected_num_tokens 24576
--ce_weight 1
--wandb_project tifa_v3
```

Answer-only specific: `--visual_gen False --mse_weight 0`
CoT specific: `--mse_weight 1 --max_latent_size {64|32|16}`
