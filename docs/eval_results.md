# Evaluation Results

Last updated: 2026-02-24

## Convention

- **Model config**: `bagel_mot` for text-only (AO, TextCoT); `bagel_mot_vcot` for image-generating (VCoT, MMCoT)
- **Work-dir**: `<converted_checkpoint>/eval/`
- **Eval repo**: `/gpfs/home/linjli/source/SpatialReasoning_Eval/`
- **Result path pattern**: `<base>/<step>_full_<variant>/eval/<subset>/bagel_mot[_vcot]/..._acc.csv`

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
| td_path_arrow | 70.02 | FAIL | 76.90 | 77.60 | **79.01** | 79.01 |
| dh_midpoint | 79.63 | 74.69 | 74.07 | 75.93 | **80.86** | 75.93 |
| td_midpoint | 72.73 | — | 73.14 | **75.62** | 75.21 | 74.79 |
| td_ego_dir | 69.00 | — | 69.91 | 73.56 | **74.16** | 68.69 |
| td_ego_side | 67.65 | — | 78.27 | 79.75 | **80.74** | 78.52 |
| td_ego_dir_arrow | 67.06 | — | 69.14 | 71.22 | **73.59** | 66.47 |
| td_ego_side_arrow | 66.76 | — | **78.21** | 77.37 | 78.21 | 76.82 |
| PathTracing | 68.46 | — | 70.42 | 82.15 | 84.11 | **84.35** |
| Perspective_Arrow | 88.63 | FAIL | 70.87 | 78.16 | 80.46 | **83.76** |
| Perspective_NoArrow | **91.38** | — | 71.07 | 79.48 | 79.98 | 85.84 |
| SAT_perspective | — | FAIL | — | — | 40.91 | — |
| HabitatPerspective_Arrow | — | FAIL | — | — | FAIL (NCCL) | — |
| HabitatPerspective_NoArrow | — | FAIL | — | — | FAIL (NCCL) | — |

### AO EMA

| Subset | s1500 | s2400 | s3000 | s4500 | s6000 | s7500 |
|--------|-------|-------|-------|-------|-------|-------|
| td_path | 34.02 | 34.96 | 33.46 | 40.23 | 67.86 | **75.00** |
| td_path_arrow | 35.45 | FAIL | 35.10 | 45.86 | 70.02 | **76.90** |
| dh_midpoint | 45.06 | 50.00 | 52.47 | 61.73 | 80.86 | **82.10** |
| td_midpoint | 40.08 | — | 48.76 | 55.79 | 74.79 | **78.93** |
| td_ego_dir | 43.47 | — | 45.59 | 55.93 | 66.87 | **71.43** |
| td_ego_side | 41.73 | — | 54.32 | 60.99 | 69.88 | **75.06** |
| td_ego_dir_arrow | 48.37 | — | 52.23 | 61.13 | 67.66 | **71.81** |
| td_ego_side_arrow | 48.60 | — | 59.22 | 67.32 | 72.91 | **78.49** |
| PathTracing | 5.62 | — | 27.38 | 46.70 | 65.53 | **73.11** |
| Perspective_Arrow | 100.00 | FAIL | 16.78 | pending | 74.23 | **74.21** |
| Perspective_NoArrow | pending | — | 21.08 | 38.69 | **75.19** | 74.38 |
| SAT_perspective | — | FAIL | — | — | — | — |
| HabitatPerspective_Arrow | — | FAIL | — | — | — | — |
| HabitatPerspective_NoArrow | — | FAIL | — | — | — | — |

> **Note**: AO s1500 EMA Perspective_Arrow = 100% is suspicious — likely an eval artifact, needs investigation.

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
| PathTracing | 67.73 | **70.42** |
| Perspective_Arrow | **99.73** | 99.71 |
| Perspective_NoArrow | **100.00** | 99.46 |
| SAT_perspective | 59.09 | — |
| HabitatPerspective | — | FAIL (NCCL) |

> **Note**: Perspective noEMA 99.73–100% is suspiciously high — needs verification.

### TextCoT EMA

| Subset | s1500 | s2000 |
|--------|-------|-------|
| td_path | 31.39 | 35.71 |
| td_path_arrow | **34.74** | 31.92 |
| dh_midpoint | 47.53 | **48.77** |
| td_midpoint | 34.71 | FAIL |
| td_ego_dir | **45.90** | 45.90 |
| td_ego_side | **43.70** | 42.96 |
| td_ego_dir_arrow | **48.66** | 46.29 |
| td_ego_side_arrow | **50.84** | 48.32 |
| PathTracing | 6.85 | 5.62 |
| Perspective_Arrow | FAIL | **100.00** |
| Perspective_NoArrow | FAIL | **100.00** |

> **Note**: TextCoT s2000 EMA Perspective = 100% is suspicious — same concern as above.

---

## VCoT l64

Config: `bagel_mot_vcot` | Base dir: `.../tifa_v3_td_path_cot_l64/`
Steps/epoch: 606 | s3000 ≈ 5ep, s6000 ≈ 10ep

### VCoT l64 td_path (phase2_l64_8gpu)

| Subset | s6000 noEMA | s6000 EMA | s7000 noEMA | s7000 EMA |
|--------|-------------|-----------|-------------|-----------|
| td_path | running | FAIL | running | running |
| td_path_arrow | running | FAIL | running | running |
| dh_midpoint | 53.09 | 18.52 | 49.38 | pending |

> **Note**: VCoT l64 s6000 EMA dh_midpoint (18.52%) is very low — may indicate EMA hasn't converged at s6000.

### VCoT dh_midpoint debug (vcot_debug_8gpu)

| Subset | s1000 noEMA | s2000 noEMA | s1000 full | s2000 EMA |
|--------|-------------|-------------|------------|-----------|
| dh_midpoint | **62.96** | **67.28** | 41.36 | 38.89 |

---

## MMCoT

Config: `bagel_mot_vcot` | Base dir: `.../tifa_v3_td_path_mmcot/mmcot_td_path_8gpu/`
Steps/epoch: 606 | s3000 ≈ 5ep, s6000 ≈ 10ep

| Subset | s6000 noEMA | s6000 EMA | s7000 noEMA | s7000 EMA |
|--------|-------------|-----------|-------------|-----------|
| td_path | 56.77 | 44.36 | running | running |
| td_path_arrow | 51.50 | 42.15 | running | running |
| dh_midpoint | 61.11 | 67.28 | 55.56 | 67.90 |
| SAT_perspective | FAIL | 57.58 | — | — |
| Perspective_Arrow | FAIL | FAIL | — | — |

---

## Cross-Model Comparison: Perspective Taking

### AI2Thor Perspective (Arrow)

| Model | Best noEMA | Best EMA |
|-------|-----------|----------|
| AO | 88.63 (s1500) | 100.00 (s1500)* |
| TextCoT | 99.73 (s1500) | 100.00 (s2000)* |
| MMCoT | FAIL | FAIL |

*100% results are suspicious and likely eval artifacts.

### SAT Perspective

| Model | Checkpoint | Accuracy (%) |
|-------|-----------|-------------|
| Baseline | — | 22.73 |
| AO | s6000 noEMA | 40.91 |
| TextCoT | s1500 noEMA | 59.09 |
| MMCoT | s6000 EMA | 57.58 |

### Habitat Perspective

All HabitatPerspective evals failed with NCCL watchdog timeout (SIGABRT after ~1h). Needs investigation.

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

1. **PathTracing baseline artificially low**: `exact_matching` fails on verbose `<think>` output — 82% unanswered
2. **td_path baseline contaminated**: Shared work-dir caused AO results to overwrite baseline predictions
3. **GPU allocation bug**: `--gpus=2` can spread across 2 nodes (1 GPU each), causing `nproc-per-node=2` assertion failure. 19 jobs failed this way. Fix: always use `--gpus-per-node=2 --nodes=1`
4. **HabitatPerspective NCCL crash**: All 6 HabitatPerspective evals (62886-62891) failed with NCCL watchdog timeout / SIGABRT after ~1h. Needs investigation.
5. **Suspicious 100% Perspective results**: AO s1500 EMA, TextCoT s2000 EMA both report 100% on Perspective — likely eval artifact
6. **VCoT l64 EMA lags noEMA**: s6000 EMA dh_midpoint (18.52%) vs noEMA (53.09%) — EMA may need more steps to converge

## See Also

- [Token Budget & Steps-per-Epoch Analysis](token_budget_and_steps.md)
- [Pending Eval Jobs](pending_eval.md)
