# Evaluation Results

Last updated: 2026-02-24 (RESCORED — see scoring fix note below)

## Scoring Fix (2026-02-24)

All results re-scored with `scripts/rescore_all_evals.py` to fix a critical accuracy bug:

**Bug**: The original eval code (commit `9cee167`) used naive substring matching on full prediction text:
```python
hit = 1 if gt.strip().upper() in pred.strip().upper() else 0
```
When GT is "B", this matches "B" anywhere in `<think>` reasoning, giving false positives.

**Fix**: Extract answer from `<answer>` tags or structured patterns (`The answer is X`, `Answer: X`, etc.) first, then compare letters.

**Impact**: Perspective results were MASSIVELY inflated (e.g., 99.73% → 47.75%). Some EMA results also changed due to improved answer extraction. PT2P noEMA results are unchanged.

**Note on EMA extraction**: Early EMA checkpoints (s1500–s3000) produce answers in non-standard formats, leading to 81–97% answer extraction rates. Results marked with † have <90% extraction (unparsed predictions count as incorrect).

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
| Perspective_Arrow | — |
| Perspective_NoArrow | — |
| HabitatPerspective | FAIL (NCCL) |

*PathTracing 6.6% is artificially low — `exact_matching` fails on verbose `<think>` output (82% unanswered). Actual performance when extractable: ~37%.

---

## Answer-Only (AO)

Config: `bagel_mot` | Base dir: `.../tifa_v3_td_path_answer_only/ao_td_path_8gpu/`
Steps/epoch: 242 | s1200 ≈ 5ep, s2400 ≈ 10ep

### AO noEMA

| Subset | s1500 | s2400 | s3000 | s4500 | s6000 | s7500 |
|--------|-------|-------|-------|-------|-------|-------|
| td_path | 70.49 | 74.06 | 77.07 | 78.76 | **82.33** | 80.26 |
| td_path_arrow | 70.02 | 74.07 | 76.90 | 77.60 | **79.01** | 79.01 |
| dh_midpoint | **79.63** | 74.69 | 74.07 | 75.93 | **80.86** | 75.93 |
| td_midpoint | 72.73 | — | 73.14 | **75.62** | 75.21 | 74.79 |
| td_ego_dir | 69.00 | — | 69.91 | 73.56 | **74.16** | 68.69 |
| td_ego_side | 67.65 | — | 78.27 | 79.75 | **80.74** | 78.52 |
| td_ego_dir_arrow | 67.06 | — | 69.14 | 71.22 | **73.59** | 66.47 |
| td_ego_side_arrow | 66.76 | — | **78.21** | 77.37 | 78.21 | 76.82 |
| PathTracing | 68.46 | — | 70.42 | 82.15 | 84.11 | **84.35** |
| Perspective_Arrow | 52.28 | — | **54.25** | 54.72 | 53.51 | 53.82 |
| Perspective_NoArrow | **53.13** | — | 53.40 | **53.76** | 51.41 | 53.20 |
| SAT_perspective | — | — | — | — | 40.91 | — |
| HabitatPerspective_Arrow | — | pending | — | — | FAIL (NCCL) | — |
| HabitatPerspective_NoArrow | — | pending | — | — | FAIL (NCCL) | — |

### AO EMA

| Subset | s1500 | s2400 | s3000 | s4500 | s6000 | s7500 |
|--------|-------|-------|-------|-------|-------|-------|
| td_path | 33.65† | 34.77† | 33.08† | 38.91 | 67.86 | **75.00** |
| td_path_arrow | 35.10† | pending | 35.45† | 45.68 | 70.02 | **76.90** |
| dh_midpoint | 53.70† | 55.56† | 56.17† | 62.35 | 80.86 | **82.10** |
| td_midpoint | 48.76† | — | 52.48† | 56.61 | 74.79 | **78.93** |
| td_ego_dir | 44.68 | — | 45.59 | 55.93 | 66.87 | **71.43** |
| td_ego_side | 41.73 | — | 54.07 | 60.99 | 69.88 | **75.06** |
| td_ego_dir_arrow | 47.18 | — | 52.23 | 61.13 | 67.66 | **71.81** |
| td_ego_side_arrow | 48.60 | — | 59.22 | 67.32 | 72.91 | **78.49** |
| PathTracing | 27.38† | — | 32.76† | 47.92 | 65.53 | **73.11** |
| Perspective_Arrow | 36.05† | pending | 40.18† | 41.61† | 52.23 | **54.78** |
| Perspective_NoArrow | pending | — | 39.66† | 44.79† | 50.54 | **51.00** |
| SAT_perspective | — | pending | — | — | — | — |
| HabitatPerspective_Arrow | — | pending | — | — | — | — |
| HabitatPerspective_NoArrow | — | pending | — | — | — | — |

†Answer extraction rate <90% — some EMA predictions use non-standard formats that couldn't be parsed (counted as incorrect).

---

## Text-CoT

Config: `bagel_mot` | Base dir: `.../tifa_v3_td_path_text_cot/textcot_td_path_8gpu/`
Steps/epoch: 242 | s1200 ≈ 5ep, s2400 ≈ 10ep
Training completed to 3000 steps (checkpoints: 500, 1000, 1500, 2000, 2200, 2400, 2600, 2800, 3000).

### TextCoT noEMA

| Subset | s1500 | s2000 |
|--------|-------|-------|
| td_path | 62.22 | **64.47** |
| td_path_arrow | **63.67** | 61.20 |
| dh_midpoint | **72.22** | 64.81 |
| td_midpoint | **67.36** | 62.40 |
| td_ego_dir | 53.80 | **55.62** |
| td_ego_side | 68.89 | **69.38** |
| td_ego_dir_arrow | 51.34 | **56.97** |
| td_ego_side_arrow | **67.88** | 65.08 |
| PathTracing | 68.22 | **70.42** |
| Perspective_Arrow | 47.75 | **47.67** |
| Perspective_NoArrow | 48.83 | **50.12** |
| SAT_perspective | 59.09 | — |
| HabitatPerspective | — | FAIL (NCCL) |

### TextCoT EMA

| Subset | s1500 | s2000 |
|--------|-------|-------|
| td_path | 31.39† | 32.33† |
| td_path_arrow | 33.33† | **35.27**† |
| dh_midpoint | **56.79**† | 54.94† |
| td_midpoint | 47.11† | FAIL |
| td_ego_dir | 45.90 | **46.81** |
| td_ego_side | **44.94** | 43.70 |
| td_ego_dir_arrow | **48.66** | 47.77 |
| td_ego_side_arrow | **50.84** | 48.88 |
| PathTracing | **33.01**† | 31.54† |
| Perspective_Arrow | pending | 31.04† |
| Perspective_NoArrow | pending | 41.65† |

†Answer extraction rate <90%.

---

## VCoT l64

Config: `bagel_mot_vcot` | Base dir: `.../tifa_v3_td_path_cot_l64/`
Steps/epoch: 606 | s3000 ≈ 5ep, s6000 ≈ 10ep

### VCoT l64 td_path (vcot_td_path_l64_8gpu)

| Subset | s6000 noEMA | s6000 EMA | s7000 noEMA | s7000 EMA |
|--------|-------------|-----------|-------------|-----------|
| td_path | pending | pending | running | running |
| td_path_arrow | pending | pending | running | running |
| dh_midpoint | 52.47 | 11.11†! | running | pending |

> **Note**: VCoT l64 s6000 EMA dh_midpoint has only 25% answer extraction — the model produces mostly unparseable outputs at this EMA checkpoint.
>
> **Note**: VCoT l64 s7000 was previously evaluated with `bagel_mot` (text-only), but the model generates images and needs `bagel_mot_vcot`. Those results (all 0%) are invalid. Proper vcot evaluations are running/pending.

### VCoT dh_midpoint debug (vcot_debug_8gpu)

| Subset | s1000 noEMA | s2000 noEMA | s1000 full | s2000 EMA |
|--------|-------------|-------------|------------|-----------|
| dh_midpoint | **62.96** | **67.28** | 52.47† | 46.91† |

†Answer extraction rate <90%.

---

## MMCoT

Config: `bagel_mot_vcot` | Base dir: `.../tifa_v3_td_path_mmcot/mmcot_td_path_8gpu/`
Steps/epoch: 606 | s3000 ≈ 5ep, s6000 ≈ 10ep

| Subset | s6000 noEMA | s6000 EMA | s7000 noEMA | s7000 EMA |
|--------|-------------|-----------|-------------|-----------|
| td_path | 56.77 | 43.98 | running | running |
| td_path_arrow | 51.50 | 41.80 | running | running |
| dh_midpoint | 61.11 | 67.28 | 55.56 | 67.90 |
| SAT_perspective | pending | 57.58 | — | — |
| Perspective_Arrow | pending | pending | — | — |

> **Note**: MMCoT s6000 dh_midpoint numbers (61.11, 67.28) are from the original eval pipeline and were not re-verified by the rescoring script (files not found in expected location). Since these are PT2P subsets evaluated with `bagel_mot_vcot`, they should not be affected by the substring matching bug.

---

## Cross-Model Comparison: Perspective Taking

### AI2Thor Perspective (Arrow) — category-averaged

| Model | Best noEMA | Best EMA |
|-------|-----------|----------|
| AO | 54.25 (s3000) | 54.78 (s7500) |
| TextCoT | 47.75 (s1500) | 31.04 (s2000)† |
| MMCoT | pending | pending |

### AI2Thor Perspective (NoArrow) — category-averaged

| Model | Best noEMA | Best EMA |
|-------|-----------|----------|
| AO | 53.76 (s4500) | 51.00 (s7500) |
| TextCoT | 50.12 (s2000) | 41.65 (s2000)† |

### SAT Perspective

| Model | Checkpoint | Accuracy (%) |
|-------|-----------|-------------|
| Baseline | — | 22.73 |
| AO | s6000 noEMA | 40.91 |
| TextCoT | s1500 noEMA | 59.09 |
| MMCoT | s6000 EMA | 57.58 |

### Habitat Perspective

All HabitatPerspective evals failed with NCCL watchdog timeout (SIGABRT after ~1h). Resubmitted with GPU allocation fix (62972–62975).

---

## Cross-Model Comparison: Path Tracing (td_path)

| Model | Best noEMA | Best EMA |
|-------|-----------|----------|
| AO | **82.33** (s6000) | 75.00 (s7500) |
| TextCoT | 64.47 (s2000) | 32.33 (s2000)† |
| MMCoT | 56.77 (s6000) | 43.98 (s6000) |
| VCoT l64 | running | running |
| Baseline | — | — |

---

## Visualization HTMLs

Generated by `scripts/visualize_eval.py`.

| File | Model | Contents |
|------|-------|----------|
| `.../bagel_eval/textcot_s2000_noema_td_path_viz.html` | TextCoT s2000 noEMA | 50 samples, input + CoT + answer (48 MB) |
| `.../bagel_eval/ao_s6000_noema_td_path_viz.html` | AO s6000 noEMA | 50 samples, input + answer (48 MB) |
| `.../bagel_eval/vcot_l64_s7000_noema_dh_midpoint_viz.html` | VCoT l64 s7000 noEMA | 50 samples, input + generated sideview (57 MB) |
| `.../bagel_eval/mmcot_s6000_ema_dh_midpoint_viz.html` | MMCoT s6000 EMA | 50 samples, input + generated sideview (56 MB) |

---

## Known Issues

1. **Scoring bug (FIXED)**: Old Perspective eval used substring matching on full prediction text, causing massively inflated accuracy. Fixed by extracting from `<answer>` tags first. See "Scoring Fix" section above.
2. **PathTracing baseline artificially low**: `exact_matching` fails on verbose `<think>` output — 82% unanswered
3. **td_path baseline contaminated**: Shared work-dir caused AO results to overwrite baseline predictions
4. **GPU allocation bug**: `--gpus=2` can spread across 2 nodes (1 GPU each), causing `nproc-per-node=2` assertion failure. 19 jobs failed this way. Fix: always use `--gpus-per-node=2 --nodes=1`
5. **HabitatPerspective NCCL crash**: All 6 HabitatPerspective evals (62886-62891) failed with NCCL watchdog timeout / SIGABRT after ~1h. Resubmitted with GPU fix.
6. **VCoT l64 s7000 invalid results**: Evaluated with `bagel_mot` (text-only) but model generates images → 0% accuracy. Must use `bagel_mot_vcot`.
7. **VCoT l64 EMA lags noEMA**: s6000 EMA dh_midpoint (11.11%, 25% extraction) vs noEMA (52.47%) — EMA hasn't converged at s6000
8. **EMA early checkpoints**: s1500–s3000 EMA predictions often lack standard `<answer>` tags, leading to 81–97% extraction rates. True accuracy may be slightly higher than reported.

## See Also

- [Token Budget & Steps-per-Epoch Analysis](token_budget_and_steps.md)
- [Pending Eval Jobs](pending_eval.md)
