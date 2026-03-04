# Self-Forcing Training for VCoT Image Generation

## Problem

VCoT training uses teacher forcing: during training, the model sees ground-truth sideview images as input to generate the answer. During inference, it sees its own generated (noisy/imperfect) sideview image. This train/inference mismatch degrades image generation quality — the model never learns to recover from its own prediction errors.

## Goal

Improve image generation quality by training the model to denoise from its own predicted latents, closing the teacher-forcing gap. Initialize from the best VCoT checkpoint (td_ego_dir, s4k).

## Approach: Two-Phase Forward Pass with Curriculum

### Phase 1: Generate predicted latent (no_grad)

For each VCoT sample with an output image (loss=1):

1. Extract the clean GT latent `x_0` from the VAE encoder output
2. Run a short denoising loop (configurable steps, e.g., 4-8) starting from pure noise:
   - Sample `x_T ~ N(0, I)`
   - For each denoising step `t = T, T-1, ..., 1`: predict velocity, update `x_t`
3. Result: `x_0_pred` — the model's own predicted clean latent

### Phase 2: Train on blended latent

1. Blend: `x_0_blend = ratio * x_0_pred + (1 - ratio) * x_0_gt`
   - `ratio` controlled by `--self_force_ratio` (0.0 = pure teacher forcing, 1.0 = pure self-forcing)
2. Noise `x_0_blend` to `x_t` using the training timestep (same as standard flow)
3. Embed `x_t` and proceed with normal forward pass + MSE loss on velocity prediction

### Why blending?

- Pure self-forcing (ratio=1.0) from the start may be too unstable — predicted latents can be far from GT
- Blending provides a curriculum: start with low ratio (mostly GT), increase as the model improves
- At ratio=0.0, training is identical to standard teacher forcing (backward compatible)

## New Training Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--self_force` | bool | False | Enable self-forcing (all changes gated behind this) |
| `--self_force_steps` | int | 8 | Number of denoising steps for Phase 1 latent prediction |
| `--self_force_ratio` | float | 0.5 | Blend ratio (0=teacher forcing, 1=pure self-forcing) |

## Implementation Location

### `modeling/bagel/bagel.py`

New method `generate_self_force_latent(vae_latent, packed_timesteps, ...)`:
- Extracts clean latent `x_0` positions (where `packed_timesteps != -inf`)
- Runs denoising loop using the model's own weights (with `torch.no_grad()`)
- Returns predicted clean latent `x_0_pred`

### `train/pretrain_unified_navit.py`

Gated block after VAE encode (line ~625), before forward pass:
```python
if args.self_force:
    x_0_pred = model.generate_self_force_latent(...)
    vae_latent = blend(x_0_pred, vae_latent_gt, args.self_force_ratio)
```

### New training script

- Initializes from VCoT s4k checkpoint (best VCoT td_ego_dir)
- Uses same dataset config as VCoT td_ego_dir
- Adds `--self_force --self_force_steps 8 --self_force_ratio 0.5`

## Safety: No Impact on Existing Training

- All self-forcing logic is gated behind `--self_force` (default False)
- When `--self_force` is False, the training loop is completely unchanged
- No modifications to existing functions — only new method and new gated block
- New args have safe defaults that preserve existing behavior

## VAE Latent Flow Reference

Standard teacher forcing:
```
GT image → VAE encode → x_0 → noise to x_t (using packed_timesteps) → embed → forward → predict v_t → MSE(v_t_pred, v_t_target)
```

Self-forcing adds before noising:
```
x_0_gt → [Phase 1: denoise from noise → x_0_pred] → blend(x_0_pred, x_0_gt, ratio) → x_0_blend → noise to x_t → ...
```

## Initialization

Checkpoint: `/gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_vcot_l32/vcot_td_ego_dir_8gpu/0004000_full_ema/`

This is the best VCoT td_ego_dir checkpoint (s4k), already fine-tuned for sideview generation.
