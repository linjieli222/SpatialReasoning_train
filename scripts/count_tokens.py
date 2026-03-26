#!/usr/bin/env python3
"""Count exact tokens per sample for TIFA v3 td_path VCoT at different output resolutions.

Image tokens are computed analytically (deterministic for fixed-size images).
Text tokens are computed by tokenizing the actual text from parquet files.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pyarrow.parquet as pq
from pathlib import Path

from transformers import AutoTokenizer
from data.dataset_info import DATASET_INFO


# === ANALYTICAL IMAGE TOKEN COUNTS ===
# All TIFA v3 images are 1024x1024. Token counts are deterministic.

# VAE transform: image_stride=16, so tokens = (w*h) / 16^2
# ViT transform: max_size=518, stride=14 → 1024x1024 resized to 518x518
#   _make_divisible(round(1024*(518/1024)), 14) = _make_divisible(518, 14) = round(518/14)*14 = 37*14 = 518
#   ViT tokens = (518*518) / 14^2 = 268324/196 = 1369

# Input image (1024x1024 topdown): need_loss=False, need_vae=True, need_vit=True
#   VAE cond: 1024*1024/256 = 4096
#   ViT: 518*518/196 = 1369
#   sequence_plan entries: 2 (vae_image + vit_image) → 4 special tokens
INPUT_IMG_TOKENS = 4096 + 1369     # 5465 content tokens
INPUT_IMG_SPECIAL = 2 * 2          # 4 special tokens

# Output image: need_loss=True, need_vae=True, need_vit=True
# VAE uses output_transform; ViT uses vit_transform on ORIGINAL image (always 1024x1024)
# sequence_plan entries: 3 (vae_loss + vae_cond + vit) → 6 special tokens
OUTPUT_VIT_TOKENS = 1369           # ViT always processes original 1024x1024 sideview
OUTPUT_IMG_SPECIAL = 3 * 2         # 6 special tokens

def output_img_tokens(latent_size):
    """VAE tokens for output image at given latent size."""
    vae_tokens = latent_size * latent_size  # VAE loss block
    vae_cond = latent_size * latent_size    # VAE conditioning block
    return vae_tokens + vae_cond + OUTPUT_VIT_TOKENS

# l64: output 1024x1024 → latent 64 → VAE = 64*64 = 4096 per block
# l32: output 512x512 → latent 32 → VAE = 32*32 = 1024 per block
# l16: output 256x256 → latent 16 → VAE = 16*16 = 256 per block
OUTPUT_TOKENS = {
    'l64': output_img_tokens(64),  # 4096+4096+1369 = 9561
    'l32': output_img_tokens(32),  # 1024+1024+1369 = 3417
    'l16': output_img_tokens(16),  # 256+256+1369 = 1881
}


def count_text_tokens(tokenizer):
    """Read all parquet samples, tokenize text, return list of text token counts."""
    ds_info = DATASET_INFO['unified_edit']['tifa_v3_td_path']
    data_dir = ds_info['data_dir']

    with open(ds_info['parquet_info_path'], 'r') as f:
        parquet_info = json.load(f)

    # Collect all parquet files
    parquet_files = sorted(Path(data_dir).glob('*.parquet'))
    print(f"Found {len(parquet_files)} parquet files in {data_dir}")

    text_token_counts = []
    for pf in parquet_files:
        pf_str = str(pf)
        pf_key = pf_str if pf_str in parquet_info else pf.name
        if pf_key not in parquet_info:
            print(f"  Warning: {pf.name} not in parquet_info, reading all row groups")
            table = pq.read_table(pf_str, columns=['instruction_list', 'output_text_list'])
        else:
            num_rg = parquet_info[pf_key]['num_row_groups']
            fr = pq.ParquetFile(pf_str)
            table = fr.read_row_groups(list(range(num_rg)), columns=['instruction_list', 'output_text_list'])

        df = table.to_pandas()
        for _, row in df.iterrows():
            instrs = row.get('instruction_list', [])
            outputs = row.get('output_text_list', [])
            if instrs is None or outputs is None:
                continue
            if hasattr(instrs, 'tolist'):
                instrs = instrs.tolist()
            if hasattr(outputs, 'tolist'):
                outputs = outputs.tolist()
            if not isinstance(instrs, list):
                instrs = [instrs]
            if not isinstance(outputs, list):
                outputs = [outputs]

            # Count text tokens (same tokenization as training)
            n_text = 0
            for text in instrs:
                if text:
                    n_text += len(tokenizer.encode(str(text)))
            for text in outputs:
                if text:
                    n_text += len(tokenizer.encode(str(text)))
            text_token_counts.append(n_text)

        print(f"  {pf.name}: {len(df)} rows (total so far: {len(text_token_counts)})")

    return text_token_counts


def compute_stats(token_counts, expected_num_tokens=24576, max_num_tokens=32768, num_gpus=8):
    """Simulate per-GPU greedy packing."""
    n = len(token_counts)
    avg = sum(token_counts) / n
    mn = min(token_counts)
    mx = max(token_counts)
    median = sorted(token_counts)[n // 2]

    # Simulate packing for one GPU (sees n/num_gpus samples)
    per_gpu_n = n // num_gpus
    per_gpu_steps = 0
    current = 0
    for t in token_counts[:per_gpu_n]:
        if t > max_num_tokens:
            continue
        if current + t > max_num_tokens:
            per_gpu_steps += 1
            current = t
        elif current + t >= expected_num_tokens:
            per_gpu_steps += 1
            current = 0
        else:
            current += t
    if current > 0:
        per_gpu_steps += 1

    samples_per_gpu_per_step = per_gpu_n / per_gpu_steps if per_gpu_steps > 0 else 0

    return {
        'total_samples': n,
        'avg_tokens': avg,
        'min_tokens': mn,
        'max_tokens': mx,
        'median_tokens': median,
        'samples_per_gpu_per_step': samples_per_gpu_per_step,
        'samples_per_step': samples_per_gpu_per_step * num_gpus,
        'steps_per_epoch': per_gpu_steps,
        'steps_5_epochs': per_gpu_steps * 5,
    }


if __name__ == '__main__':
    print("Loading tokenizer...")
    model_path = "/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("\nCounting text tokens from parquet files...")
    text_tokens = count_text_tokens(tokenizer)
    print(f"\nTotal samples with text: {len(text_tokens)}")
    print(f"Text tokens: avg={sum(text_tokens)/len(text_tokens):.1f}, "
          f"min={min(text_tokens)}, max={max(text_tokens)}, "
          f"median={sorted(text_tokens)[len(text_tokens)//2]}")

    print(f"\n=== IMAGE TOKEN COUNTS (analytical) ===")
    print(f"Input image (1024x1024):  {INPUT_IMG_TOKENS} content + {INPUT_IMG_SPECIAL} special = {INPUT_IMG_TOKENS + INPUT_IMG_SPECIAL}")
    for name, out_tok in OUTPUT_TOKENS.items():
        print(f"Output image ({name}):     {out_tok} content + {OUTPUT_IMG_SPECIAL} special = {out_tok + OUTPUT_IMG_SPECIAL}")

    # Compute total tokens per sample for each config
    results = {}
    for name in ['l64', 'l32', 'l16']:
        total_per_sample = [
            t + INPUT_IMG_TOKENS + INPUT_IMG_SPECIAL + OUTPUT_TOKENS[name] + OUTPUT_IMG_SPECIAL
            for t in text_tokens
        ]
        stats = compute_stats(total_per_sample)
        results[name] = stats

    # Print comparison
    print(f"\n\n{'='*62}")
    print(f"{'EXACT COMPARISON TABLE':^62}")
    print(f"{'='*62}")
    print(f"{'Metric':<30} {'l64':>10} {'l32':>10} {'l16':>10}")
    print(f"{'-'*62}")

    for metric in ['total_samples', 'avg_tokens', 'min_tokens', 'max_tokens', 'median_tokens',
                    'samples_per_gpu_per_step', 'samples_per_step', 'steps_per_epoch', 'steps_5_epochs']:
        vals = [results[c][metric] for c in ['l64', 'l32', 'l16']]
        if isinstance(vals[0], float):
            print(f"{metric:<30} {vals[0]:>10.2f} {vals[1]:>10.2f} {vals[2]:>10.2f}")
        else:
            print(f"{metric:<30} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    # Timing estimates based on measured VCoT l64 speed
    print(f"\n{'='*62}")
    print(f"{'TIMING (based on l64 measured 0.09 steps/sec)':^62}")
    print(f"{'='*62}")
    # l32 and l16 are faster per step due to shorter sequences within the same token budget
    # Measured: l64=0.09 steps/sec. Conservatively assume similar for l32/l16
    # (actual might be slightly faster due to shorter attention spans)
    for name in ['l64', 'l32', 'l16']:
        steps = results[name]['steps_5_epochs']
        wall_h = steps / 0.09 / 3600
        gpu_h = wall_h * 8
        sus = wall_h * 64  # billing=64 per hour
        print(f"{name}: {steps} steps × (1/0.09)s = {wall_h:.1f}h wall | {gpu_h:.0f} GPU-hrs | {sus:.0f} SUs")
