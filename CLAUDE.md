# Project Instructions

## Python Environment
When running Python commands, use the python binary directly:
```
/gpfs/projects/krishna/envs/thinkmorph/bin/python3
```
There is no `activate` script — use the full path to the binary.
In Slurm scripts, use `module load conda && conda activate /gpfs/projects/krishna/envs/thinkmorph`.

## Cluster: Tillicum (UW HPC)
- Home: `/gpfs/home/linjli/` (10GB quota — don't store large files)
- Projects: `/gpfs/projects/krishna/linjli/` (1TB, backed up)
- Scrubbed: `/gpfs/scrubbed/krishna/linjli/` (100TB, 60-day auto-purge)
- Model checkpoint: `/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT`
- Training output: `/gpfs/scrubbed/krishna/linjli/bagel_debug_output/`
- Training data: `/gpfs/projects/krishna/linjli/bagel_example/editing/`
- WANDB_DIR must point to scrubbed storage (home quota too small)
- Azure backup: `~/bin/azcopy`, cron sync every 6h, scripts in `~/scripts/`
- Crontab is on login01 only (2 login nodes, round-robin assignment)

## Training System (BAGEL-7B-MoT)

### Token Packing
- No fixed batch size — uses NaViT-style token packing
- `expected_num_tokens=24576` is the soft packing target per GPU per step
- `max_num_tokens=32768` is the hard upper limit
- Effective batch size varies by sample length (answer-only packs ~32 samples/step, CoT ~8)

### Image Token Costs (all TIFA v3 images are 1024x1024)
- VAE transform (stride 16): 1024x1024 → 4,096 tokens per image
- ViT transform (stride 14, max 518): 1024x1024 → 1,369 tokens per image
- Input image total: 5,465 tokens (ViT + VAE)
- Multi-input images (ego variants, >1 input): resized to 512x512 → 2,393 tokens each

### max_latent_size — CRITICAL
- `--max_latent_size` controls the position embedding grid for ALL VAE images (input AND output)
- It must always be >= the latent size of the LARGEST image (input or output)
- For 1024x1024 inputs: latent size = 1024/16 = 64, so `--max_latent_size 64` always
- Setting it lower (e.g., 32) causes CUDA index-out-of-bounds crashes
- To control OUTPUT image resolution, use `output_image_transform_args` in the YAML config instead
- Example: latent-32 output uses `output_image_transform_args: {max_image_size: 512, ...}` with `--max_latent_size 64`

### Visual CoT vs Answer-Only
- `--visual_gen False --mse_weight 0`: answer-only (no sideview generation, no MSE loss)
- `--mse_weight 1` (without `--visual_gen False`): CoT with sideview image generation
- When `visual_gen=False`, the data loader still generates VAE tokens; they are popped at training time
- Answer-only trains ~3x faster than CoT due to shorter sequences

### Output Image Sizes (latent mapping)
- latent 64 = 1024x1024 (default for non-ego CoT)
- latent 32 = 512x512 (default for ego CoT, matching resized inputs)
- latent 16 = 256x256
- Formula: image_pixels = latent_size × 16 (vae_image_downsample)

### Slurm Job Submission
- Generic template: `scripts/slurm/train_tifa_v3.slurm`
- Usage: `sbatch --job-name=<name> [--time=HH:MM:SS] scripts/slurm/train_tifa_v3.slurm <script.sh>`
- Template handles conda activation, WANDB_DIR, working directory

### Checkpoint Conversion (FSDP → safetensors)
- Training saves FSDP sharded format (`model/`, `ema/` dirs with `.distcp` files)
- Eval needs safetensors format (`ema.safetensors`)
- Script: `scripts/convert_sharded_to_full.py`
- Usage: `python scripts/convert_sharded_to_full.py --checkpoint_path <ckpt> --output_path <out> --model_path /gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT --convert_ema`
- Resource requirements: `--gpus=2 --mem=400G --cpus-per-task=8` (58GB safetensors needs ~65GB RAM)
- Tillicum limits: max 8 CPUs per GPU, max 240GB RAM per GPU, minimum 1 GPU per job

### Evaluation Pipeline
- Eval repo: `/gpfs/home/linjli/source/SpatialReasoning_Eval/`
- Uses `torchrun --nproc-per-node=2` with 2 GPUs per eval
- AI2ThorPT2P subsets use exact MCQ matching — NO GPT judge needed
- Eval takes ~54 min per subset on 2 GPUs (text-only); vcot/mmcot image-gen evals take 2-4h
- **Model config selection** (CRITICAL):
  - `bagel_mot` (understanding_output=True): text-only eval. For **AO** and **textcot** models.
  - `bagel_mot_vcot` (understanding_output=False): image + text generation. For **vcot** and **mmcot** models.
  - Using `bagel_mot` for vcot/mmcot models → predictions truncated at `<image_start>` → near-0% accuracy (INVALID)
- **Work-dir convention**: Save eval results under the checkpoint directory, NOT a shared dir.
  - Pattern: `--work-dir <converted_checkpoint_path>/eval/`
  - This avoids `--reuse` picking up stale predictions from other models.
  - Exception: baseline model evals go to `/gpfs/scrubbed/krishna/linjli/bagel_eval/baseline/`
- Slurm submission pattern:
  ```
  CKPT=<converted_checkpoint_path>
  export THINKMORPH_MODEL_PATH=$CKPT
  # For AO/textcot:
  torchrun --nproc-per-node=2 run.py --model bagel_mot --data AI2ThorPT2P_<subset> --work-dir $CKPT/eval
  # For vcot/mmcot (also set THINKMORPH_SAVE_DIR for generated images):
  export THINKMORPH_SAVE_DIR=$CKPT/eval/vcot_images
  torchrun --nproc-per-node=2 run.py --model bagel_mot_vcot --data AI2ThorPT2P_<subset> --work-dir $CKPT/eval
  ```

## Experiment Plans
- Design doc: `docs/plans/2026-02-17-tifa-v3-training-variants-design.md`
- Implementation plan: `docs/plans/2026-02-17-tifa-v3-training-variants-impl.md`
