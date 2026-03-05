# PT2PV2 Evaluation Results (AI2ThorPT2PV2_td_ego_dir, n=113)

## Eval Configs

| Config | think | understanding_output | Description |
|--------|-------|---------------------|-------------|
| `bagel_mot` (AO) | True | True | Text thinking, no image gen |
| `bagel_mot_vcot` (Imagegen) | True | False | Think prompt + image gen |
| `bagel_mot_nothink` (Nothink) | False | True | Answer-only, no image gen |
| `bagel_mot_nothink_vcot` | False | False | Answer-only + VAE input encoding (no image gen in practice) |

## Mixed (VCoT + AO data)

### td_ego_dir (n=113)

| Step | Nothink | Nothink_VCoT | AO (bagel_mot) | Imagegen (vcot) |
|------|---------|--------------|----------------|-----------------|
| s1k  | 38.94%  | 53.10%       | 45.13%         | 50.44%          |
| s2k  | 64.60%  | 65.49%       | 61.06%         | 43.36%          |
| s3k  | **70.80%** | 69.03%    | 69.03%         | 44.25%          |
| s4k  | 69.03%  | 69.03%       | **70.80%**     | 52.21%          |
| s5k  | **70.80%** | **70.80%** | 69.03%         | **54.87%**      |

### td_path (n=169) — Nothink only

| Step | Accuracy |
|------|----------|
| s1k  | 31.95%   |
| s2k  | 49.70%   |
| s3k  | 60.36%   |
| s4k  | 62.13%   |
| s5k  | **63.31%** |

### td_path_arrow (n=171) — Nothink only

| Step | Accuracy |
|------|----------|
| s1k  | 33.92%   |
| s2k  | 50.88%   |
| s3k  | 60.23%   |
| s4k  | **63.16%** |
| s5k  | 61.40%   |

## TextCoT (text chain-of-thought)

### td_ego_dir

| Step | AO/Think (PT2PV2, n=113) | Nothink (PT2PV2, n=113) | AO/Think (PT2P v1, n=329) | Nothink (PT2P v1, n=329) | AO/Think (SV, n=100) | Nothink (SV, n=100) |
|------|--------------------------|-------------------------|---------------------------|--------------------------|----------------------|---------------------|
| s1k  | 38.05%                   | 38.94%                  | 50.76%                    | 47.72%                   | 13.0%                | 12.0%               |
| s2k  | 45.13%                   | 41.59%                  | 59.57%                    | 56.53%                   | 30.0%                | 28.0%               |
| s3k  | 53.10%                   | 47.79%                  | 61.09%                    | 59.88%                   | 38.0%                | 40.0%               |
| s4k  | **57.52%**               | 46.02%                  | 64.44%                    | 63.53%                   | 44.0%                | 42.0%               |
| s5k  | 53.10%                   | 55.75%                  | **67.78%**                | —                        | 50.0%                | —                   |
| s6k  | 53.98%                   | 47.79%                  | 65.65%                    | 63.22%                   | 49.0%                | 45.0%               |
| s7k  | —                        | —                       | 63.22%                    | 63.53%                   | **53.0%**            | 48.0%               |
| s8k  | 53.98%                   | 50.44%                  | 65.65%                    | 65.35%                   | 53.0%                | 43.0%               |
| s9k  | —                        | —                       | 65.05%                    | 61.09%                   | **55.0%**            | 46.0%               |
| s10k | —                        | —                       | 62.92%                    | —                        | —                    | —                   |

## MMCoT (multi-modal chain-of-thought)

### td_ego_dir — Nothink

| Step | PT2PV2 (n=113) | PT2P v1 (n=329) | SV (n=100) |
|------|----------------|-----------------|------------|
| s1k  | 33.63%         | 48.02%          | 12.0%      |
| s2k  | 39.82%         | 63.22%          | 30.0%      |
| s3k  | 52.21%         | 65.96%          | 41.0%      |
| s4k  | 59.29%         | 69.60%          | 45.0%      |
| s5k  | **62.83%**     | 68.69%          | **51.0%**  |
| s6k  | 59.29%         | **71.43%**      | 48.0%      |

## VCoT (pure VCoT training, mse_weight=1)

### td_ego_dir

| Step | Nothink PT2PV2 (n=113) | Imagegen PT2PV2 (n=113) | Imagegen PT2P v1 (n=329) | Imagegen SV (n=100) |
|------|------------------------|-------------------------|--------------------------|---------------------|
| s1k  | —                      | —                       | 51.37%                   | —                   |
| s2k  | 61.06%                 | —                       | 51.98%                   | —                   |
| s3k  | —                      | —                       | 60.49%                   | —                   |
| s4k  | —                      | —                       | 61.40%                   | —                   |
| s5k  | —                      | —                       | 58.05%                   | —                   |
| s6k  | —                      | —                       | 58.36%                   | —                   |
| s7k  | 0.0%                   | 50.44%                  | **64.44%**               | 77.0%               |
| s8k  | —                      | —                       | 61.70%                   | —                   |

## MSE5 (mse_weight=5)

### td_ego_dir (n=113)

| Step | Nothink | Imagegen (vcot) |
|------|---------|-----------------|
| s1k  | 46.90%  | —               |
| s4k  | **62.83%** | 50.44%       |
| s5k  | 53.98%  | 47.79%          |

## MSE2 (mse_weight=2)

### td_ego_dir (n=113)

| Step | Nothink | Imagegen (vcot) |
|------|---------|-----------------|
| s1k  | 46.90%  | —               |
| s4k  | 6.19%   | **56.64%**      |
| s5k  | 0.88%   | 51.33%          |

## Answer-Only Training Baselines

### AO ego_dir (s6k ema, best) — PT2PV2

| Subset | Accuracy | Count |
|--------|----------|-------|
| td_ego_dir | **73.45%** | 113 |
| td_path | 61.54% | 169 |
| td_path_arrow | 61.99% | 171 |

### AO td_path (s18k ema) — PT2PV2

| Subset | Accuracy | Count |
|--------|----------|-------|
| td_path | **81.07%** | 169 |
| td_path_arrow | **83.63%** | 171 |
| dh_midpoint | 75.93% | 54 |

## GPT-5 Baseline

| Subset | Accuracy | Count |
|--------|----------|-------|
| td_ego_dir | 46.90% | 113 |
| td_path | 56.80% | 169 |
| td_path_arrow | 52.05% | 171 |
| RealPT_td_path | 76.44% | 174 |
| RealPT_td_path_arrow | **88.61%** | 158 |

## Key Observations

1. **Mixed training best**: 70.80% (nothink s3k/s5k) on td_ego_dir — close to AO ego_dir baseline (73.45%) while retaining image generation capability
2. **Nothink_VCoT converges to nothink**: early boost at s1k (53.10% vs 38.94%, +14%), gap closes by s3k, and both reach 70.80% at s5k — VAE input encoding helps early but doesn't matter long-term
3. **Mixed imagegen improves later**: s4k (52.21%) and s5k (54.87%) are best, suggesting image gen quality improves with more training
4. **MSE2 nothink collapses**: drops to 0.88% by s5k — high image loss weight destroys text reasoning
5. **Pure VCoT nothink collapses at s7k**: 0.0% — extended VCoT training loses text-only ability; imagegen drops from 64.44% (PT2P v1) to 50.44% (PT2PV2) showing bias dependence
6. **Mixed prevents collapse**: maintains ~70% nothink through s3k-s5k while imagegen stays ~44-55%
7. **MMCoT steadily improves**: 33.63% → 62.83% nothink over s1k-s5k on PT2PV2, still below mixed (70.80%) but trending up
8. **TextCoT AO/Think > Nothink**: AO (think=True) consistently beats nothink on PT2PV2 (57.52% vs 46.02% at s4k), showing text thinking helps
9. **TextCoT peaks at s4k-s5k then declines**: PT2PV2 AO peaks at 57.52% (s4k) then drops to 53.98% (s8k); PT2P v1 at 67.78% (s5k); SV climbs to 55% at s9k
10. **MMCoT s6k diverges**: PT2PV2 drops to 59.29% (from 62.83%), PT2P v1 peaks at 71.43%, SV drops to 48.0% (from 51.0%) — overfitting to v1 distribution while degrading on debiased/SV
11. **Prefill v2 s7k ema**: 86.73% on PT2PV2 — upper bound when GT sideview is given
