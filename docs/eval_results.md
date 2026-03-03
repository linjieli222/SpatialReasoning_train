# Evaluation Results

Last updated: 2026-03-03

**Results by training regime:**
- [td_path results](eval_results_td_path.md) — AO, TextCoT, VCoT l64, MMCoT
- [td_ego_dir results](eval_results_td_ego_dir.md) — AO, VCoT l32, RealPathTracing

## Scoring Fix (2026-02-24)

All results re-scored with `scripts/rescore_all_evals.py` to fix a critical accuracy bug:

**Bug**: The original eval code (commit `9cee167`) used naive substring matching on full prediction text:
```python
hit = 1 if gt.strip().upper() in pred.strip().upper() else 0
```
When GT is "B", this matches "B" anywhere in `<think>` reasoning, giving false positives.

**Fix**: Extract answer from `<answer>` tags or structured patterns (`The answer is X`, `Answer: X`, etc.) first, then compare letters.

**Impact**: Perspective results were MASSIVELY inflated (e.g., 99.73% -> 47.75%). Some EMA results also changed due to improved answer extraction. PT2P noEMA results are unchanged.

**Note on EMA extraction**: Early EMA checkpoints (s1500-s3000) produce answers in non-standard formats, leading to 81-97% answer extraction rates. Results marked with dagger have <90% extraction (unparsed predictions count as incorrect).

## Convention

- **Model config**: `bagel_mot` for text-only (AO, TextCoT); `bagel_mot_vcot` for image-generating (VCoT, MMCoT)
- **Work-dir**: `<converted_checkpoint>/eval/`
- **Eval repo**: `/gpfs/home/linjli/source/SpatialReasoning_Eval/`
- **Perspective accuracy**: unweighted average across categories (consistent with eval pipeline)

---

## Baseline (pretrained BAGEL-7B-MoT)

| Subset | Accuracy (%) |
|--------|-------------|
| dh_midpoint | 33.95 |
| PathTracing | 6.60* |
| PathTracing_sideview | 4.65 |
| SAT_perspective | 22.73 |
| Perspective_Arrow | -- |
| Perspective_NoArrow | -- |
| HabitatPerspective | FAIL (NCCL) |

*PathTracing 6.6% is artificially low -- `exact_matching` fails on verbose `<think>` output (82% unanswered). Actual performance when extractable: ~37%.

---

## Known Issues

1. **Scoring bug (FIXED)**: Old Perspective eval used substring matching on full prediction text, causing massively inflated accuracy. Fixed by extracting from `<answer>` tags first. See "Scoring Fix" section above.
2. **PathTracing baseline artificially low**: `exact_matching` fails on verbose `<think>` output -- 82% unanswered
3. **td_path baseline contaminated**: Shared work-dir caused AO results to overwrite baseline predictions
4. **GPU allocation bug**: `--gpus=2` can spread across 2 nodes (1 GPU each), causing `nproc-per-node=2` assertion failure. Fix: always use `--gpus-per-node=2 --nodes=1`
5. **HabitatPerspective NCCL crash**: All 6 HabitatPerspective evals failed with NCCL watchdog timeout / SIGABRT after ~1h.
6. **VCoT l64 text-only = 0%**: VCoT l64 model always outputs `<think>desc</think><image_start>` in text-only mode, never producing an answer. Must use `bagel_mot_vcot` for VCoT image-gen eval, or `bagel_mot_nothink` to test text-only reasoning.
7. **VCoT l64 EMA lags noEMA**: s6000 EMA dh_midpoint (11.11%, 25% extraction) vs noEMA (52.47%) -- EMA hasn't converged at s6000
8. **EMA early checkpoints**: s1500-s3000 EMA predictions often lack standard `<answer>` tags, leading to 81-97% extraction rates. True accuracy may be slightly higher than reported.
9. **Port collision (FIXED)**: Multiple eval jobs on the same node defaulting to torchrun port 29500. Fixed with `MASTER_PORT=$((29500 + RANDOM % 10000))`.

## Pending Evals

- **VCoT l32 td_ego_dir image-gen PT2P**: s1k-s5k EMA+noEMA resubmitted with 8 GPUs (2-GPU runs timed out at ~11h)
- **VCoT l32 td_ego_dir image-gen SV**: s6k-s8k running
- **VCoT l32 td_ego_dir prefill**: s3k EMA submitted (job 67101)
- **VCoT l64 td_path prefill**: s7k EMA+noEMA running (~60% done)
- **VCoT mse_weight=5 training**: td_ego_dir l32, 15k steps (job 67093)
- **AO noEMA s12000/s18000**: Perspective_Arrow, Perspective_NoArrow (port collision failures, need resubmission)
- **TextCoT s3000 EMA, s21000, s24000**: td_path, td_path_arrow (were still running at last check)
- **VCoT l64 s9000-s15000**: td_path, td_path_arrow, Perspective_Arrow, Perspective_NoArrow (cancelled)
- **MMCoT s9000-s15000**: td_path, td_path_arrow, Perspective_Arrow, Perspective_NoArrow (cancelled)
- **MMCoT s12000/s15000**: all subsets (not yet evaluated)

## See Also

- [Token Budget & Steps-per-Epoch Analysis](token_budget_and_steps.md)
