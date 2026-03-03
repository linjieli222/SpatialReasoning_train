# Evaluation Results

Last updated: 2026-03-02

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

## Answer-Only (AO)

Config: `bagel_mot` | Base dir: `.../tifa_v3_td_path_answer_only/ao_td_path_8gpu/`
Steps/epoch: 242 | s1200 = 5ep, s2400 = 10ep
Training extended to 25k steps (save_every=3000 from s7500 onward).

### AO noEMA -- Early Training

| Subset | s1500 | s2400 | s3000 | s4500 | s6000 | s7500 |
|--------|-------|-------|-------|-------|-------|-------|
| td_path | 70.49 | 74.06 | 77.07 | 78.76 | 82.33 | 80.26 |
| td_path_arrow | 70.02 | 74.07 | 76.90 | 77.60 | 79.01 | 79.01 |
| dh_midpoint | 79.63 | 74.69 | 74.07 | 75.93 | 80.86 | 75.93 |
| td_midpoint | 72.73 | -- | 73.14 | 75.62 | 75.21 | 74.79 |
| td_ego_dir | 69.00 | -- | 69.91 | 73.56 | 74.16 | 68.69 |
| td_ego_side | 67.65 | -- | 78.27 | 79.75 | 80.74 | 78.52 |
| td_ego_dir_arrow | 67.06 | -- | 69.14 | 71.22 | 73.59 | 66.47 |
| td_ego_side_arrow | 66.76 | -- | 78.21 | 77.37 | 78.21 | 76.82 |
| PathTracing | 68.46 | -- | 70.42 | 82.15 | 84.11 | 84.35 |
| Perspective_Arrow | 52.28 | -- | 54.25 | 54.72 | 53.51 | 53.82 |
| Perspective_NoArrow | 53.13 | -- | 53.40 | 53.76 | 51.41 | 53.20 |
| SAT_perspective | -- | -- | -- | -- | 40.91 | -- |

### AO EMA -- Early Training

| Subset | s1500 | s2400 | s3000 | s4500 | s6000 | s7500 |
|--------|-------|-------|-------|-------|-------|-------|
| td_path | 33.65+ | 34.77+ | 33.08+ | 38.91 | 67.86 | 75.00 |
| td_path_arrow | 35.10+ | 37.92+ | 35.45+ | 45.68 | 70.02 | 76.90 |
| dh_midpoint | 53.70+ | 55.56+ | 56.17+ | 62.35 | 80.86 | 82.10 |
| td_midpoint | 48.76+ | -- | 52.48+ | 56.61 | 74.79 | 78.93 |
| td_ego_dir | 44.68 | -- | 45.59 | 55.93 | 66.87 | 71.43 |
| td_ego_side | 41.73 | -- | 54.07 | 60.99 | 69.88 | 75.06 |
| td_ego_dir_arrow | 47.18 | -- | 52.23 | 61.13 | 67.66 | 71.81 |
| td_ego_side_arrow | 48.60 | -- | 59.22 | 67.32 | 72.91 | 78.49 |
| PathTracing | 27.38+ | -- | 32.76+ | 47.92 | 65.53 | 73.11 |
| Perspective_Arrow | 36.05+ | -- | 40.18+ | 41.61+ | 52.23 | 54.78 |
| Perspective_NoArrow | -- | -- | 39.66+ | 44.79+ | 50.54 | 51.00 |

+Answer extraction rate <90%.

### AO noEMA -- Extended Training (s12k-s24k)

| Subset | s12000 | s15000 | s18000 | s21000 | s24000 |
|--------|--------|--------|--------|--------|--------|
| td_path | 83.08 | 82.33 | 80.08 | 81.20 | 82.71 |
| td_path_arrow | 83.25 | 82.19 | 80.07 | 79.01 | 82.36 |
| dh_midpoint | 75.31 | 75.93 | 69.14 | 72.22 | 70.37 |
| Perspective_Arrow | -- | 51.21 | -- | 55.84 | 54.07 |
| Perspective_NoArrow | -- | 50.76 | -- | 56.22 | 53.72 |
| SAT_perspective | 39.39 | 37.88 | 46.97 | 43.94 | 48.48 |

> Missing Perspective results at s12000/s18000: those eval jobs failed in the port-collision batch and need resubmission.

### AO EMA -- Extended Training (s12k-s24k)

| Subset | s12000 | s15000 | s18000 | s21000 | s24000 |
|--------|--------|--------|--------|--------|--------|
| td_path | 83.83 | 85.90 | **86.09** | 85.15 | 84.77 |
| td_path_arrow | 83.95 | 84.66 | 85.19 | 84.66 | **85.36** |
| dh_midpoint | 80.86 | **82.72** | 80.86 | 79.01 | 77.16 |
| Perspective_Arrow | 54.77 | 54.65 | 53.40 | 54.07 | 53.34 |
| Perspective_NoArrow | 53.57 | **54.14** | **54.66** | 52.89 | 53.07 |
| SAT_perspective | 39.39 | 39.39 | 40.91 | **45.45** | 42.42 |

**AO key findings**: EMA consistently outperforms noEMA by 3-6pp on core subsets. EMA peaks at s18000 for td_path (86.09%) and s15000 for dh_midpoint (82.72%). Performance plateaus after s15000 with slight degradation on dh_midpoint. SAT_perspective steadily improves with training (22.73% baseline -> 48.48% at s24000 noEMA).

---

## Text-CoT

Config: `bagel_mot` | Base dir: `.../tifa_v3_td_path_text_cot/textcot_td_path_8gpu/`
Steps/epoch: 242 | s1200 = 5ep, s2400 = 10ep
Training extended to 25k steps (save_every=3000 from s3000 onward).

### TextCoT noEMA -- Early Training

| Subset | s1500 | s2000 |
|--------|-------|-------|
| td_path | 62.22 | 64.47 |
| td_path_arrow | 63.67 | 61.20 |
| dh_midpoint | 72.22 | 64.81 |
| td_midpoint | 67.36 | 62.40 |
| td_ego_dir | 53.80 | 55.62 |
| td_ego_side | 68.89 | 69.38 |
| td_ego_dir_arrow | 51.34 | 56.97 |
| td_ego_side_arrow | 67.88 | 65.08 |
| PathTracing | 68.22 | 70.42 |
| Perspective_Arrow | 47.75 | 47.67 |
| Perspective_NoArrow | 48.83 | 50.12 |
| SAT_perspective | 59.09 | -- |

### TextCoT EMA -- Early Training

| Subset | s1500 | s2000 |
|--------|-------|-------|
| td_path | 31.39+ | 32.33+ |
| td_path_arrow | 33.33+ | 35.27+ |
| dh_midpoint | 56.79+ | 54.94+ |
| td_midpoint | 47.11+ | FAIL |
| td_ego_dir | 45.90 | 46.81 |
| td_ego_side | 44.94 | 43.70 |
| td_ego_dir_arrow | 48.66 | 47.77 |
| td_ego_side_arrow | 50.84 | 48.88 |
| PathTracing | 33.01+ | 31.54+ |
| Perspective_Arrow | -- | 31.04+ |
| Perspective_NoArrow | -- | 41.65+ |

+Answer extraction rate <90%.

### TextCoT noEMA -- Extended Training (s3k-s24k)

| Subset | s3000 | s6000 | s9000 | s12000 | s15000 | s18000 | s21000 | s24000 |
|--------|-------|-------|-------|--------|--------|--------|--------|--------|
| td_path | 65.04 | 64.47 | 66.04 | **66.73** | 62.22 | 61.65 | 60.90 | -- |
| td_path_arrow | 61.55 | **65.60** | 60.84 | 63.49 | 59.06 | 64.60 | 62.84 | -- |
| dh_midpoint | **68.52** | 67.28 | 64.81 | 58.64 | 53.70 | 55.56 | 50.62 | 53.70 |
| Perspective_Arrow | 52.51 | 51.80 | 47.14 | 49.28 | 45.71 | 49.28 | 43.93 | 50.00 |
| Perspective_NoArrow | **53.57** | 52.15 | 51.79 | 52.86 | 48.93 | 48.21 | 41.43 | 45.36 |
| SAT_perspective | -- | **56.06** | 39.39 | 34.85 | 50.00 | 45.45 | -- | -- |

### TextCoT EMA -- Extended Training (s3k-s24k)

| Subset | s3000 | s6000 | s9000 | s12000 | s15000 | s18000 | s21000 | s24000 |
|--------|-------|-------|-------|--------|--------|--------|--------|--------|
| td_path | -- | 49.44 | 58.46 | 59.21 | 62.97 | **63.16** | -- | -- |
| td_path_arrow | -- | 44.07 | 57.53 | 59.25 | 60.84 | **63.49** | -- | -- |
| dh_midpoint | 43.83 | **72.84** | 69.75 | 69.75 | 64.81 | 64.81 | 62.35 | 60.49 |
| Perspective_Arrow | -- | **55.00** | **55.00** | 52.15 | 54.00 | 54.64 | 51.79 | **56.47** |
| Perspective_NoArrow | -- | **58.33** | 52.15 | 49.64 | 49.29 | 54.29 | 50.00 | 49.64 |
| SAT_perspective | -- | 50.00 | 50.00 | **51.52** | 48.48 | 40.91 | -- | -- |

> Missing td_path/td_path_arrow at s3000 EMA, s21000, s24000: evals were still running at time of collection.

**TextCoT key findings**: noEMA peaks early (s3000-s12000) then degrades, especially on dh_midpoint (68.5% -> 50.6%). EMA lags noEMA by ~6000 steps but eventually converges (td_path 63.2% EMA s18000 vs 66.7% noEMA s12000). EMA dh_midpoint peaks at s6000 (72.8%) then steadily declines. Overall, TextCoT significantly underperforms AO on td_path (~66% vs ~86%).

---

## VCoT l64

Config: `bagel_mot_vcot` | Base dir: `.../tifa_v3_td_path_cot_l64/`
Steps/epoch: 606 | s3000 = 5ep, s6000 = 10ep
Training extended to 25k steps (still running).

### VCoT l64 noEMA

| Subset | s6000 | s7000 | s9000 | s12000 | s15000 |
|--------|-------|-------|-------|--------|--------|
| td_path | 40.23 | **47.37** | -- | -- | -- |
| td_path_arrow | 38.45 | **45.15** | -- | -- | -- |
| dh_midpoint | 52.47 | 49.38 | 55.56 | **58.02** | 54.94 |
| SAT_perspective | -- | -- | **46.97** | 19.70 | **50.00** |

### VCoT l64 EMA

| Subset | s6000 | s7000 | s9000 | s12000 | s15000 |
|--------|-------|-------|-------|--------|--------|
| td_path | 3.01+! | **40.04** | -- | -- | -- |
| td_path_arrow | 2.47+! | **41.80** | -- | -- | -- |
| dh_midpoint | 11.11+! | 52.47 | 55.56 | **62.35** | 58.64 |
| SAT_perspective | -- | -- | 37.88 | **42.42** | 39.39 |

+! EMA has not converged at s6000 (5-25% answer extraction).

> td_path, td_path_arrow, Perspective_Arrow, Perspective_NoArrow results missing for s9000-s15000: evals were cancelled to free up GPU resources. Will be resubmitted later.

### VCoT l64 — Text-only think (`bagel_mot`)

Text-only eval of VCoT checkpoints using `bagel_mot` config (think=True, understanding_output=True).
The model generates `<think>desc</think><image_start>` but no image is produced — answer extraction fails.

| Subset | s7000 EMA | s7000 noEMA | s9000 EMA | s9000 noEMA |
|--------|-----------|-------------|-----------|-------------|
| td_path | 0.0 | 0.8 | 0.0 | 0.2 |

> All ~0%: VCoT l64 model always outputs `<think>desc</think><image_start>` and never produces an answer letter in text-only mode. The model has completely lost text-only answering ability.

### VCoT l64 — No-think (`bagel_mot_nothink`)

Text-only eval with answer-only system prompt + "Do not think or generate any images." appended to question.

**EMA** — shows clear rise-then-collapse pattern:

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s9000 | s12000 | s15000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|--------|--------|
| td_path | 36.1 | 37.8 | 40.2 | 46.2 | 50.8 | 59.2 | **65.4** | 63.0 | 3.4 | 0.0 |

**noEMA** — all near 0% (s1k-s6k not converted):

| Subset | s7000 | s9000 | s12000 | s15000 |
|--------|-------|-------|--------|--------|
| td_path | 0.4 | 5.1 | 2.1 | 0.4 |

> **Key finding**: EMA nothink accuracy rises steadily from s1k (36.1%) to s7k (65.4%), then collapses at s12k (3.4%) and s15k (0.0%). This suggests VCoT training progressively strengthens image generation commitment, and by s12k the model can no longer suppress it even with explicit "do not generate" instruction. noEMA never learns to suppress image generation.

### VCoT dh_midpoint debug (vcot_debug_8gpu)

| Subset | s1000 noEMA | s2000 noEMA | s1000 full | s2000 EMA |
|--------|-------------|-------------|------------|-----------|
| dh_midpoint | 62.96 | **67.28** | 52.47+ | 46.91+ |

+Answer extraction rate <90%.

---

## MMCoT

Config: `bagel_mot_vcot` | Base dir: `.../tifa_v3_td_path_mmcot/mmcot_td_path_8gpu/`
Steps/epoch: 606 | s3000 = 5ep, s6000 = 10ep
Training extended to 25k steps (still running).

### MMCoT noEMA

| Subset | s6000 | s7000 | s9000 |
|--------|-------|-------|-------|
| td_path | **56.77** | 49.44 | -- |
| td_path_arrow | 51.50 | **51.15** | -- |
| dh_midpoint | 61.11 | 55.56 | **59.88** |
| SAT_perspective | -- | -- | 31.82 |

### MMCoT EMA

| Subset | s6000 | s7000 | s9000 |
|--------|-------|-------|-------|
| td_path | 43.98 | **44.74** | -- |
| td_path_arrow | 41.80 | **46.38** | -- |
| dh_midpoint | -- | **67.90** | -- |
| SAT_perspective | **57.58** | -- | 36.36 |

> td_path, td_path_arrow, Perspective results missing for s9000+: evals were cancelled. s12000/s15000 not yet evaluated.

---

## AO td_ego_dir

Config: `bagel_mot` (text-only, think=True) | Base dir: `.../tifa_v3_td_ego_dir_ao/ao_td_ego_dir_8gpu/`
Training: 5 epochs on td_ego_dir subset. Training still running (s6k+ available).

**Eval subsets:**
- `AI2ThorPT2P_td_ego_dir`: 329 samples, 4-choice MCQ (A/B/C/D), 3 input images (topdown + 2 egocentric views from endpoints)
- `AI2ThorSV_td_ego_dir`: 198 samples, binary (A/B), 3 input images (topdown + 2 egocentric views)

### AO td_ego_dir noEMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 |
|--------|-------|-------|-------|-------|-------|-------|
| PT2P (acc) | 76.3 | 79.0 | 78.1 | 77.5 | 76.9 | 78.1 |
| SV (acc) | 66.7 | 71.2 | 70.7 | 68.2 | 71.7 | 70.2 |
| SV (F1) | 64.5 | 68.2 | 67.0 | 63.6 | 69.9 | 72.6 |

### AO td_ego_dir EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 |
|--------|-------|-------|-------|-------|-------|-------|
| PT2P (acc) | 53.8 | 77.5 | 79.0 | 79.0 | 79.3 | **80.9** |
| SV (acc) | 59.6 | 71.2 | 69.2 | 67.7 | 72.7 | 72.2 |
| SV (F1) | 39.4 | 69.2 | 64.7 | 64.0 | 70.3 | 71.2 |

---

## VCoT l32 td_ego_dir

Config: `bagel_mot_vcot` (image generation, think=True) | Base dir: `.../tifa_v3_td_ego_dir_vcot_l32/vcot_td_ego_dir_8gpu/`
Training: VCoT with 512x512 output images (latent 32). Training still running.

**Eval settings compared:**

| Setting | Config | System Prompt | Image Gen | Description |
|---------|--------|---------------|-----------|-------------|
| VCoT (image gen) | `bagel_mot_vcot` | Think with `<think>` + `<image_start>` tags | Yes (512x512) | Full VCoT pipeline: model generates sideview image then answers |
| Text-only think | `bagel_mot` | Think with `<think>` + `<image_start>` tags | No | Text-only output; model may output `<image_start>` but no image is generated |
| No-think | `bagel_mot_nothink` | "Answer the question" (answer-only) | No | Answer-only prompt + "Do not think or generate any images." appended to question |

### VCoT l32 td_ego_dir noEMA — VCoT image gen (`bagel_mot_vcot`)

| Subset | s1000 | s2000 | s3000 | s4000 |
|--------|-------|-------|-------|-------|
| PT2P (acc) | -- | -- | -- | -- |
| SV (acc) | 67.2 | -- | -- | -- |
| SV (F1) | 61.1 | -- | -- | -- |

### VCoT l32 td_ego_dir EMA — VCoT image gen (`bagel_mot_vcot`)

| Subset | s1000 | s2000 | s3000 | s4000 |
|--------|-------|-------|-------|-------|
| PT2P (acc) | -- | -- | -- | -- |
| SV (acc) | 55.6 | -- | -- | -- |
| SV (F1) | 41.3 | -- | -- | -- |

> Partial results — remaining eval jobs still running (~1h left for s1k-s3k, ~3h for s4k).

### VCoT l32 td_ego_dir noEMA — Text-only think (`bagel_mot`)

| Subset | s1000 | s2000 | s3000 |
|--------|-------|-------|-------|
| PT2P (acc) | 71.7 | -- | -- |
| SV (acc) | 67.2 | 68.2 | -- |
| SV (F1) | 62.0 | 60.8 | -- |

### VCoT l32 td_ego_dir EMA — Text-only think (`bagel_mot`)

| Subset | s1000 | s2000 | s3000 |
|--------|-------|-------|-------|
| PT2P (acc) | 55.3 | 68.1 | -- |
| SV (acc) | 57.6 | 66.2 | 68.7 |
| SV (F1) | 35.4 | 58.9 | 64.8 |

> Partial results — remaining jobs were cancelled before completion (replaced by nothink eval).

### VCoT l32 td_ego_dir noEMA — No-think (`bagel_mot_nothink`)

| Subset | s1000 | s2000 | s3000 |
|--------|-------|-------|-------|
| PT2P (acc) | **72.0** | 71.7 | 0.0 |
| SV (acc) | 67.2 | **68.7** | 0.5 |
| SV (F1) | 62.4 | **67.4** | 0.0 |

### VCoT l32 td_ego_dir EMA — No-think (`bagel_mot_nothink`)

| Subset | s1000 | s2000 | s3000 |
|--------|-------|-------|-------|
| PT2P (acc) | 56.5 | **74.5** | 34.7 |
| SV (acc) | 56.6 | **64.6** | 34.3 |
| SV (F1) | 30.6 | **56.8** | 44.9 |

> **Pattern**: Similar rise-then-collapse as VCoT l64 but at s3k instead of s12k. noEMA collapses completely (0.0% PT2P), EMA partially (34.7% PT2P). VCoT commitment strengthens rapidly between s2k-s3k for ego_dir variant.

---

## Cross-Model Comparison: Path Tracing (td_path)

| Model | Best noEMA | Best EMA |
|-------|-----------|----------|
| AO | 83.08 (s12000) | **86.09** (s18000) |
| TextCoT | 66.73 (s12000) | 63.16 (s18000) |
| MMCoT | 56.77 (s6000) | 44.74 (s7000) |
| VCoT l64 | 47.37 (s7000) | 40.04 (s7000) |

## Cross-Model Comparison: DH Midpoint

| Model | Best noEMA | Best EMA |
|-------|-----------|----------|
| AO | 80.86 (s6000) | **82.72** (s15000) |
| TextCoT | 72.22 (s1500) | 72.84 (s6000) |
| VCoT l64 debug | 67.28 (s2000) | -- |
| VCoT l64 | 58.02 (s12000) | 62.35 (s12000) |
| MMCoT | 61.11 (s6000) | 67.90 (s7000) |

## Cross-Model Comparison: Perspective Taking

### AI2Thor Perspective (Arrow) -- category-averaged

| Model | Best noEMA | Best EMA |
|-------|-----------|----------|
| AO | 55.84 (s21000) | **54.78** (s7500) |
| TextCoT | 52.51 (s3000) | 56.47 (s24000) |

### AI2Thor Perspective (NoArrow) -- category-averaged

| Model | Best noEMA | Best EMA |
|-------|-----------|----------|
| AO | 56.22 (s21000) | 54.66 (s18000) |
| TextCoT | 53.57 (s3000) | **58.33** (s6000) |

### SAT Perspective

| Model | Checkpoint | Accuracy (%) |
|-------|-----------|-------------|
| Baseline | -- | 22.73 |
| TextCoT | s6000 noEMA | **56.06** |
| TextCoT | s1500 noEMA | 59.09 |
| MMCoT | s6000 EMA | 57.58 |
| VCoT l64 | s15000 noEMA | 50.00 |
| AO | s24000 noEMA | 48.48 |

---

## Visualization HTMLs

Generated by `scripts/visualize_eval.py`.

| File | Model | Contents |
|------|-------|----------|
| `.../bagel_eval/textcot_s2000_noema_td_path_viz.html` | TextCoT s2000 noEMA | 50 samples, td_path (48 MB) |
| `.../bagel_eval/ao_s6000_noema_td_path_viz.html` | AO s6000 noEMA | 50 samples, td_path (48 MB) |
| `.../bagel_eval/vcot_l64_s7000_noema_dh_midpoint_viz.html` | VCoT l64 s7000 noEMA | 50 samples, dh_midpoint (31 MB) |
| `.../bagel_eval/mmcot_s7000_ema_dh_midpoint_viz.html` | MMCoT s7000 EMA | 50 samples, dh_midpoint (35 MB) |
| `.../bagel_eval/ao_s6000_noema_perspective_arrow_viz.html` | AO s6000 noEMA | 50 samples, Perspective Arrow (55 MB) |
| `.../bagel_eval/ao_s6000_noema_perspective_noarrow_viz.html` | AO s6000 noEMA | 50 samples, Perspective NoArrow (55 MB) |
| `.../bagel_eval/textcot_s1500_noema_perspective_arrow_viz.html` | TextCoT s1500 noEMA | 50 samples, Perspective Arrow (55 MB) |
| `.../bagel_eval/textcot_s1500_noema_perspective_noarrow_viz.html` | TextCoT s1500 noEMA | 50 samples, Perspective NoArrow (55 MB) |

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

- **VCoT l32 td_ego_dir image-gen**: s1k-s4k EMA+noEMA PT2P+SV (bagel_mot_vcot) — running (jobs 66688-66699, 66782-66785)
- **AO noEMA s12000/s18000**: Perspective_Arrow, Perspective_NoArrow (port collision failures, need resubmission)
- **TextCoT s3000 EMA, s21000, s24000**: td_path, td_path_arrow (were still running at last check)
- **VCoT l64 s9000-s15000**: td_path, td_path_arrow, Perspective_Arrow, Perspective_NoArrow (cancelled)
- **MMCoT s9000-s15000**: td_path, td_path_arrow, Perspective_Arrow, Perspective_NoArrow (cancelled)
- **MMCoT s12000/s15000**: all subsets (not yet evaluated)

## See Also

- [Token Budget & Steps-per-Epoch Analysis](token_budget_and_steps.md)
