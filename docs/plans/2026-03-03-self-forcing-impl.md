# Self-Forcing Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add self-forcing training to close the teacher-forcing gap for VCoT image generation, gated behind `--self_force` so existing training is unaffected.

**Architecture:** Two-phase forward pass — Phase 1 runs a short no-grad denoising loop inside `forward()` to predict the model's own clean latent, then blends it with GT. Phase 2 is the normal training forward with MSE+CE loss on the blended latent. All logic is gated behind a `self_force_config` parameter.

**Tech Stack:** PyTorch, FSDP, flow matching (velocity prediction)

---

### Task 1: Add self-forcing arguments to TrainingArguments

**Files:**
- Modify: `train/pretrain_unified_navit.py:315-339` (after `use_flex` field, before `def main()`)

**Step 1: Add three new fields to TrainingArguments**

Add these fields at the end of the `TrainingArguments` dataclass (after `use_flex` at line 339, before the blank line at 340):

```python
    # --- self-forcing ---
    self_force: bool = field(
        default=False,
        metadata={"help": "Enable self-forcing: denoise from model's own predictions, blend with GT."}
    )
    self_force_steps: int = field(
        default=4,
        metadata={"help": "Number of denoising steps for self-forcing latent prediction (Phase 1)."}
    )
    self_force_ratio: float = field(
        default=0.5,
        metadata={"help": "Blend ratio: 0.0 = pure teacher-forcing, 1.0 = pure self-forcing."}
    )
```

**Step 2: Commit**

```bash
git add train/pretrain_unified_navit.py
git commit -m "Add self-forcing training arguments"
```

---

### Task 2: Add self-forcing denoising logic to Bagel.forward()

**Files:**
- Modify: `modeling/bagel/bagel.py:101-125` (forward signature) and `modeling/bagel/bagel.py:181-197` (VAE block)

This is the core change. We add `self_force_config` as an optional parameter to `forward()`, and insert a gated denoising block after patchification (line 188) but before noising (line 190).

**Step 1: Add `self_force_config` parameter to forward signature**

Add after `mse_loss_indexes` parameter (line 124):

```python
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
        # for self-forcing
        self_force_config: Optional[dict] = None,
```

**Step 2: Insert self-forcing denoising block**

After line 188 (`packed_latent_clean = torch.cat(packed_latent, dim=0)`), insert the self-forcing block before line 190 (`noise = torch.randn_like(...)`):

```python
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            # --- Self-forcing: predict clean latent via denoising loop ---
            if self_force_config is not None:
                sf_steps = self_force_config['steps']
                sf_ratio = self_force_config['ratio']

                # Identify output image tokens (input images have timestep=-inf)
                is_output = ~torch.isinf(packed_timesteps)

                with torch.no_grad():
                    # Start: clean for input tokens, noise for output tokens
                    x_t = packed_latent_clean.clone()
                    x_t[is_output] = torch.randn(
                        is_output.sum(), packed_latent_clean.shape[1],
                        device=packed_latent_clean.device, dtype=packed_latent_clean.dtype,
                    )

                    # Build timestep schedule (same as generate_image)
                    shift = self.timestep_shift
                    ts = torch.linspace(1, 0, sf_steps + 1, device=x_t.device)
                    ts = shift * ts / (1 + (shift - 1) * ts)
                    dts = ts[:-1] - ts[1:]

                    # Compute MoE routing indexes (needed for language_model call)
                    sf_extra = {}
                    if self.use_moe:
                        sf_und_idx = packed_text_indexes
                        if packed_vit_token_indexes is not None:
                            sf_und_idx = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
                        sf_extra.update(
                            packed_und_token_indexes=sf_und_idx,
                            packed_gen_token_indexes=packed_vae_token_indexes,
                        )

                    # Build attention mask (may already exist from above)
                    if nested_attention_masks is None:
                        sf_sparse = create_sparse_mask(sample_lens, split_lens, attn_modes, x_t.device)
                        sf_seqlen = sum(sample_lens)
                        sf_attention_mask = create_block_mask(
                            sf_sparse, B=1, H=self.num_heads, Q_LEN=sf_seqlen, KV_LEN=sf_seqlen,
                            device=x_t.device, BLOCK_SIZE=128, _compile=True,
                        )
                    else:
                        sf_attention_mask = nested_attention_masks

                    for i, t_val in enumerate(ts[:-1]):
                        # Timestep: t_val for output tokens, 0 for input tokens
                        t_vec = torch.zeros(x_t.shape[0], device=x_t.device)
                        t_vec[is_output] = t_val.item()

                        # Embed VAE tokens at current timestep
                        ts_embed = self.time_embedder(t_vec)
                        pos_embed = self.latent_pos_embed(packed_latent_position_ids)
                        vae_embed = self.vae2llm(x_t) + ts_embed + pos_embed

                        # Build packed sequence with current VAE embeddings
                        sf_seq = packed_sequence.clone()
                        sf_seq[packed_vae_token_indexes] = vae_embed

                        # Forward through language model
                        hidden = self.language_model(
                            packed_sequence=sf_seq,
                            sample_lens=sample_lens,
                            attention_mask=sf_attention_mask,
                            packed_position_ids=packed_position_ids,
                            **sf_extra,
                        )

                        # Extract velocity prediction at output image positions
                        v_t = self.llm2vae(hidden[mse_loss_indexes])
                        # Euler step: x_t -= v_t * dt (velocity points from data to noise)
                        x_t[is_output] = x_t[is_output] - v_t * dts[i]

                    # Blend predicted clean latent with GT (only output tokens)
                    packed_latent_clean[is_output] = (
                        sf_ratio * x_t[is_output] + (1 - sf_ratio) * packed_latent_clean[is_output]
                    )
            # --- End self-forcing block ---

            noise = torch.randn_like(packed_latent_clean)
```

**Key design notes:**
- `attention_mask` variable: at this point in `forward()` (line 162-164), `attention_mask` is already computed and available. But it's a local variable that gets reused later. We use it directly if available, or recompute from `nested_attention_masks` / `split_lens+attn_modes`. To be safe, we recompute inside the block since `attention_mask` is only assigned if `nested_attention_masks is None`.
- `packed_sequence` at this point has text + ViT embeddings filled in but VAE positions are zeros. The self-forcing block fills VAE positions with denoising embeddings.
- `mse_loss_indexes` is a bool tensor over the full sequence. `hidden[mse_loss_indexes]` gives hidden states at output image positions. The count of True values in `mse_loss_indexes` equals the count of True values in `is_output`.

**Step 3: Commit**

```bash
git add modeling/bagel/bagel.py
git commit -m "Add self-forcing denoising block to Bagel forward"
```

---

### Task 3: Pass self_force_config from training loop to forward

**Files:**
- Modify: `train/pretrain_unified_navit.py:612-625` (VAE encode + forward call section)

**Step 1: Add self_force_config to data dict before forward call**

After line 615 (`data['padded_latent'] = vae_model.encode(...)`) and before line 624 (`loss_dict = fsdp_model(**data)`), insert:

```python
            if training_args.visual_gen:
                with torch.no_grad():
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
                # Pass self-forcing config if enabled
                if training_args.self_force:
                    data['self_force_config'] = {
                        'steps': training_args.self_force_steps,
                        'ratio': training_args.self_force_ratio,
                    }
            else:
```

**Step 2: Commit**

```bash
git add train/pretrain_unified_navit.py
git commit -m "Wire self_force_config into training forward pass"
```

---

### Task 4: Create self-forcing training script

**Files:**
- Create: `scripts/train_tifa_v3_td_ego_dir_self_force.sh`

**Step 1: Create the training launch script**

```bash
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
```

**Key differences from standard VCoT script:**
- `resume_from`: VCoT s4k checkpoint (not base model)
- `--self_force True --self_force_steps 4 --self_force_ratio 0.5`: enable self-forcing
- `--total_steps 5000`: shorter since we're fine-tuning from an already-trained checkpoint
- Output dir: `tifa_v3_td_ego_dir_self_force/`
- Same dataset config (`tifa_v3_td_ego_dir_vcot_l32.yaml`) — self-forcing only changes how the latent is processed, not what data is loaded

**Step 2: Commit**

```bash
git add scripts/train_tifa_v3_td_ego_dir_self_force.sh
git commit -m "Add self-forcing training script for td_ego_dir"
```

---

### Task 5: Verify checkpoint exists and submit training job

**Step 1: Verify the VCoT s4k checkpoint path exists**

```bash
ls -la /gpfs/scrubbed/krishna/linjli/bagel_debug_output/tifa_v3_td_ego_dir_vcot_l32/vcot_td_ego_dir_8gpu/0004000_full_ema/
```

Expected: directory with `ema.safetensors` and config files.

**Step 2: Submit the training job**

```bash
sbatch --job-name=sf_ego_dir --time=48:00:00 scripts/slurm/train_tifa_v3.slurm scripts/train_tifa_v3_td_ego_dir_self_force.sh
```

Use 48h timeout since self-forcing is ~5x slower than normal VCoT (4 extra forward passes per step under no_grad).

**Step 3: Verify job starts**

```bash
squeue -u $USER --name=sf_ego_dir
```

Check early wandb logs:
- `mse_loss` should be active (VCoT samples contribute)
- `ce_loss` should be active
- Training speed should be ~5x slower than normal VCoT (~0.02 steps/sec vs ~0.09)

---

## Performance Expectations

| Config | steps/sec | Time per 1000 steps | Notes |
|--------|-----------|---------------------|-------|
| Normal VCoT | ~0.09 | ~3.1h | Baseline |
| Self-force (2 steps) | ~0.03 | ~9.3h | 3x overhead |
| Self-force (4 steps) | ~0.018 | ~15.4h | 5x overhead |
| Self-force (8 steps) | ~0.01 | ~27.8h | 9x overhead |

With 4 denoising steps and 5000 total steps: ~77h (~3.2 days). 48h timeout will reach ~3100 steps. Consider using `--time=72:00:00` or reducing to 2 steps if 48h is insufficient.

## Future Optimization

The current implementation runs the full model forward for each denoising step. A KV-cache approach (prefill text+ViT once, then run lightweight denoising with cached KV) would reduce overhead to ~2x instead of ~5x. This can be added later if self-forcing shows promising results.
