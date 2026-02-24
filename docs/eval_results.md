# Evaluation Results

Last updated: 2026-02-24

## Eval Convention

- **Model config**: `bagel_mot` for text-only models (AO, TextCoT); `bagel_mot_vcot` for image-generating models (VCoT, MMCoT)
- **Work-dir**: `<converted_checkpoint>/eval/` — isolates each model's results
- **Eval repo**: `/gpfs/home/linjli/source/SpatialReasoning_Eval/`

---

## Baseline (pretrained BAGEL-7B-MoT, no fine-tuning)

| Subset | Accuracy (%) | Path |
|--------|-------------|------|
| dh_midpoint | 33.95 | `/gpfs/scrubbed/krishna/linjli/bagel_eval/tifa_v3_eval_test/bagel_mot/bagel_mot_AI2ThorPT2P_dh_midpoint_acc.csv` |
| PathTracing | 6.60 | `/gpfs/scrubbed/krishna/linjli/bagel_eval/baseline/bagel_mot/bagel_mot_AI2ThorPathTracing_acc.csv` |
| PathTracing_sideview | 4.65 | `/gpfs/scrubbed/krishna/linjli/bagel_eval/baseline_PathTracing_sideview/bagel_mot/bagel_mot_AI2ThorPathTracing_sideview_acc.csv` |
| SAT_perspective | pending (62874) | `/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT/eval/` |

> **Note**: PathTracing baseline (6.6%) is artificially low — the `exact_matching` extraction policy fails on the baseline model's verbose `<think>` output (82% of predictions get no answer extracted). The actual model performance when answers are extractable is ~37%.

> **Note**: td_path baseline was contaminated by an AO model writing to the shared work-dir. No valid baseline exists for td_path.

---

## Answer-Only (AO)

Config: `bagel_mot` | Base dir: `.../tifa_v3_td_path_answer_only/ao_td_path_8gpu/`

### AO noEMA

| Subset | s1500 | s3000 | s4500 | s6000 | s7500 |
|--------|-------|-------|-------|-------|-------|
| td_path | 70.49 | 77.07 | 78.76 | **82.33** | 80.26 |
| td_path_arrow | 70.02 | 76.90 | 77.60 | **79.01** | 79.01 |
| dh_midpoint | **79.63** | 74.07 | 75.93 | 80.86 | 75.93 |
| td_midpoint | 72.73 | 73.14 | **75.62** | 75.21 | 74.79 |
| td_ego_dir | 69.00 | 69.91 | 73.56 | **74.16** | 68.69 |
| td_ego_side | 67.65 | 78.27 | 79.75 | **80.74** | 78.52 |
| td_ego_dir_arrow | 67.06 | 69.14 | 71.22 | **73.59** | 66.47 |
| td_ego_side_arrow | 66.76 | **78.21** | 77.37 | 78.21 | 76.82 |
| PathTracing | 68.46 | 70.42 | 82.15 | 84.11 | **84.35** |
| Perspective_Arrow | 88.63 | 70.87 | 78.16 | 80.46 | **83.76** |
| Perspective_NoArrow | **91.38** | 71.07 | 79.48 | 79.98 | 85.84 |

### AO EMA

| Subset | s1500 | s3000 | s4500 | s6000 | s7500 |
|--------|-------|-------|-------|-------|-------|
| td_path | 34.02 | 33.46 | 40.23 | 67.86 | **75.00** |
| td_path_arrow | 35.45 | 35.10 | 45.86 | 70.02 | **76.90** |
| dh_midpoint | 45.06 | 52.47 | 61.73 | 80.86 | **82.10** |
| td_midpoint | 40.08 | 48.76 | 55.79 | 74.79 | **78.93** |
| td_ego_dir | 43.47 | 45.59 | 55.93 | 66.87 | **71.43** |
| td_ego_side | 41.73 | 54.32 | 60.99 | 69.88 | **75.06** |
| td_ego_dir_arrow | 48.37 | 52.23 | 61.13 | 67.66 | **71.81** |
| td_ego_side_arrow | 48.60 | 59.22 | 67.32 | 72.91 | **78.49** |
| PathTracing | 5.62 | 27.38 | 46.70 | 65.53 | **73.11** |
| Perspective_Arrow | pending (62852) | pending (62779) | pending (62783) | 74.23 | **74.21** |
| Perspective_NoArrow | pending (62853) | pending (62780) | pending (62784) | **75.19** | 74.38 |

### AO Result Paths

Pattern: `<base>/<step>_full_<variant>/eval/bagel_mot/bagel_mot_<dataset>_acc.csv`

Some s7500 results use nested structure: `<base>/<step>_full_<variant>/eval/AI2ThorPT2P_<subset>/bagel_mot/...`

Missing:
- s1500 EMA Perspective (pending — resubmitted as 62864, prior 62775/62852 failed with NCCL/port conflict)
- s3000 EMA Perspective (running — 62779–62780)
- s4500 EMA Perspective (running — 62783–62784)

---

## Text-CoT

Config: `bagel_mot` | Base dir: `.../tifa_v3_td_path_text_cot/textcot_td_path_8gpu/`

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
| Perspective_Arrow | 99.73 | pending (62858) |
| Perspective_NoArrow | 100.00 | pending (62859) |

### TextCoT EMA

| Subset | s1500 | s2000 |
|--------|-------|-------|
| td_path | 31.39 | FAIL |
| td_path_arrow | 34.74 | FAIL |
| dh_midpoint | 47.53 | **48.77** |
| td_midpoint | 34.71 | FAIL |
| td_ego_dir | **45.90** | 45.90 |
| td_ego_side | **43.70** | 42.96 |
| td_ego_dir_arrow | **48.66** | 46.29 |
| td_ego_side_arrow | **50.84** | 48.32 |
| PathTracing | 6.85 | 5.62 |
| Perspective_Arrow | pending (62854) | pending (62856) |
| Perspective_NoArrow | pending (62855) | pending (62857) |

### TextCoT Result Paths

Pattern: `<base>/<step>_full_<variant>/eval/AI2ThorPT2P_<subset>/bagel_mot/bagel_mot_<dataset>_acc.csv`

Perspective results: `<base>/<step>_full_<variant>/eval/bagel_mot/bagel_mot_AI2Thor<Perspective_variant>_acc.csv`

Missing:
- s2000 EMA: td_path, td_path_arrow, td_midpoint (TIMEOUT/FAIL)
- s1500 EMA Perspective (pending — resubmitted as 62854–62855, prior 62795–96 failed with GPU allocation)
- s2000 EMA Perspective (pending — resubmitted as 62856–62857, prior 62799–800 failed)
- s2000 noEMA Perspective (pending — resubmitted as 62858–62859, prior 62801–802 failed)

> **Note**: TextCoT s1500 noEMA Perspective_Arrow 99.73% and Perspective_NoArrow 100.00% — these numbers are suspiciously high and should be verified.

---

## VCoT dh_midpoint debug

Config: `bagel_mot_vcot` | Base dir: `.../tifa_v3_dh_midpoint_vcot/vcot_debug_8gpu/`

| Subset | s1000 noEMA | s2000 noEMA | s1000 (full) | s2000 EMA |
|--------|-------------|-------------|--------------|-----------|
| dh_midpoint | **62.96** | **67.28** | 41.36 | 38.89 |

### VCoT debug Result Paths

- s1000 noEMA: `.../vcot_debug_8gpu/0001000_full_noema/eval/bagel_mot_vcot/bagel_mot_vcot_AI2ThorPT2P_dh_midpoint_acc.csv`
- s2000 noEMA: `.../vcot_debug_8gpu/0002000_full_noema/eval/bagel_mot_vcot/bagel_mot_vcot_AI2ThorPT2P_dh_midpoint_acc.csv`
- s1000 full: `.../vcot_debug_8gpu/0001000_full/eval/bagel_mot_vcot/bagel_mot_vcot_AI2ThorPT2P_dh_midpoint_acc.csv`
- s2000 EMA: `.../vcot_debug_8gpu/0002000_full_ema/eval/bagel_mot_vcot/bagel_mot_vcot_AI2ThorPT2P_dh_midpoint_acc.csv`

---

## VCoT l64 (s7000)

Config: `bagel_mot_vcot` | Base dir: `.../tifa_v3_td_path_cot_l64/vcot_td_path_l64_8gpu/`

### Valid Results (bagel_mot_vcot)

| Subset | noEMA |  EMA |
|--------|-------|------|
| dh_midpoint | 49.38 | pending (62860) |
| td_path | running (62845) | running (62846) |
| td_path_arrow | running (62847) | running (62848) |

noEMA dh_midpoint path: `/gpfs/scrubbed/krishna/linjli/bagel_eval/vcot_l64_s7000_noema_viz/bagel_mot_vcot/bagel_mot_vcot_AI2ThorPT2P_dh_midpoint_acc.csv`


---

## MMCoT (s6000)

Config: `bagel_mot_vcot` | Base dir: `.../tifa_v3_td_path_mmcot/mmcot_td_path_8gpu/`

### Valid Results (bagel_mot_vcot)

| Subset | noEMA | EMA |
|--------|-------|-----|
| dh_midpoint | 61.11 | 67.28 |
| td_path | pending (62541) | pending (62543) |
| td_path_arrow | pending (62542) | pending (62544) |

dh_midpoint paths:
- EMA: `/gpfs/scrubbed/krishna/linjli/bagel_eval/mmcot_s6000_ema_vcot/bagel_mot_vcot/bagel_mot_vcot_AI2ThorPT2P_dh_midpoint_acc.csv`
- noEMA: `/gpfs/scrubbed/krishna/linjli/bagel_eval/mmcot_s6000_noema_vcot/bagel_mot_vcot/bagel_mot_vcot_AI2ThorPT2P_dh_midpoint_acc.csv`


---

## MMCoT (s7000) — Pending

Conversions submitted: 62803 (EMA), 62804 (noEMA)

Evals chained (16h limit, bagel_mot_vcot):

| Job | Variant | Subset |
|-----|---------|--------|
| 62805 | EMA | td_path |
| 62806 | noEMA | td_path |
| 62807 | EMA | td_path_arrow |
| 62808 | noEMA | td_path_arrow |
| 62809 | EMA | dh_midpoint |
| 62810 | noEMA | dh_midpoint |

---

## SAT Perspective Taking

External benchmark: [SAT_circular](https://huggingface.co/datasets/luckychao/vlmevalkit_tsv) — perspective-taking subset only.

Eval uses `bagel_mot` with `SAT_perspective` dataset. MCQ evaluation via heuristic `exact_matching` (no GPT judge needed).

| Model | Checkpoint | Accuracy (%) | Job | Path |
|-------|-----------|-------------|-----|------|
| Baseline (BAGEL-7B-MoT) | — | pending | 62874 | `/gpfs/scrubbed/linjli/hf_cache/BAGEL-7B-MoT/eval/` |
| AO | s6000 noEMA | pending | 62871 | `.../ao_td_path_8gpu/0006000_full_noema/eval/` |
| TextCoT | s1500 noEMA | pending | 62872 | `.../textcot_td_path_8gpu/0001500_full_noema/eval/` |

---

## Pending Jobs Summary

| Jobs | Model | Subsets | Status |
|------|-------|--------|--------|
| 62541–62544 | MMCoT s6000 vcot | td_path, td_path_arrow (EMA+noEMA) | running (~10h in, 16h limit) |
| 62779–62780 | AO s3000 EMA Perspective | Arrow, NoArrow | running (~2h in) |
| 62783–62784 | AO s4500 EMA Perspective | Arrow, NoArrow | running (~2h in) |
| 62805–62810 | MMCoT s7000 vcot | td_path, td_path_arrow, dh_midpoint (EMA+noEMA) | running (~2h in, 16h limit) |
| 62845–62848 | VCoT l64 s7000 vcot | td_path, td_path_arrow (EMA+noEMA) | running (~1h in, 16h limit) |
| 62853 | AO s1500 EMA Perspective NoArrow | NoArrow | running (~30m in) |
| 62854–62855 | TextCoT s1500 EMA Perspective | Arrow, NoArrow | running (~30m in) |
| 62864 | AO s1500 EMA Perspective Arrow | Arrow | just submitted |
| 62867–62868 | TextCoT s2000 EMA Perspective | Arrow, NoArrow | just submitted |
| 62869–62870 | TextCoT s2000 noEMA Perspective | Arrow, NoArrow | just submitted |
| 62871 | AO s6000 noEMA SAT_perspective | SAT_perspective | just submitted |
| 62872 | TextCoT s1500 noEMA SAT_perspective | SAT_perspective | just submitted |
| 62874 | Baseline SAT_perspective | SAT_perspective | just submitted |

---

## Visualization HTMLs

Generated by `scripts/visualize_eval.py` — reusable script for any eval result.

| File | Model | Contents |
|------|-------|----------|
| `.../bagel_eval/textcot_s2000_noema_td_path_viz.html` | TextCoT s2000 noEMA | 50 samples, input images + question + CoT + answer (48 MB) |
| `.../bagel_eval/ao_s6000_noema_td_path_viz.html` | AO s6000 noEMA | 50 samples, input images + question + answer (48 MB) |
| `.../bagel_eval/vcot_l64_s7000_noema_dh_midpoint_viz.html` | VCoT l64 s7000 noEMA | 50 samples, input images + generated sideview images (57 MB) |
| `.../bagel_eval/mmcot_s6000_ema_dh_midpoint_viz.html` | MMCoT s6000 EMA | 50 samples, input images + generated sideview images (56 MB) |

Older bulk visualizations (all predictions, no input images):

| File | Model | Contents |
|------|-------|----------|
| `.../bagel_eval/textcot_viz.html` | TextCoT | 29 result sets, all steps/variants/subsets (23 MB) |
| `.../bagel_eval/ao_td_path_viz.html` | AO td_path | 10 result sets, s1500–s7500 × EMA/noEMA (11 MB) |
| `.../bagel_eval/ao_perspective_viz.html` | AO Perspective | 14 result sets, Arrow/NoArrow (3 MB) |
| `.../bagel_eval/outputs_dh_midpoint_debug_vcot/viz/viz_dh_midpoint.html` | VCoT debug | Older VCoT debug visualization |

---

## Known Issues

1. **Perspective baseline**: SAT_perspective baseline submitted (62874), no AI2Thor Perspective baseline yet
2. **PathTracing baseline artificially low**: `exact_matching` policy fails on verbose `<think>` output — 82% of predictions get no answer extracted
3. **td_path baseline contaminated**: Shared work-dir had AO results overwrite baseline predictions
4. **TextCoT s2000 EMA missing**: td_path, td_path_arrow, td_midpoint TIMEOUT/FAIL
5. **VCoT l64 incomplete**: Only dh_midpoint (noEMA, 49.38%) has valid results; td_path/td_path_arrow/EMA now running (62845–62848, 62860)
6. **GPU allocation bug**: Jobs submitted with `--gpus=2` can be spread across 2 nodes (1 GPU each), causing `torchrun --nproc-per-node=2` assertion failure. Fix: always use `--gpus-per-node=2 --nodes=1`
