# Evaluation Results — td_ego_dir Training Regime

Back to [Eval Results Index](eval_results.md)

> **Data freshness**: Last updated 2026-03-03. VCoT l32 image-gen PT2P evals are mostly **partial results** from timed-out 2-GPU runs (see partial table); only s1000 EMA has a full 8-GPU result. VCoT l32 SV image-gen s6k-s7k still running. VCoT l32 nothink/think evals are complete through s3k but not extended further (collapse makes later steps uninformative). AO td_ego_dir evals are complete through s10000.

## Figures

![AO td_ego_dir training curves — PT2P and SV](figures/ao_ego_dir_training_curves.png)

![VCoT l32 td_ego_dir eval setting comparison at s1k and s2k](figures/vcot_l32_eval_comparison.png)

![RealPathTracing overfitting pattern — AO td_ego_dir](figures/realpt_training_curves.png)

---

## AO td_ego_dir

Config: `bagel_mot` (text-only, think=True) | Base dir: `.../tifa_v3_td_ego_dir_ao/ao_td_ego_dir_8gpu/`
Training: 5 epochs on td_ego_dir subset. Training still running (s6k+ available).

**Eval subsets:**
- `AI2ThorPT2P_td_ego_dir`: 329 samples, 4-choice MCQ (A/B/C/D), 3 input images (topdown + 2 egocentric views from endpoints)
- `AI2ThorSV_td_ego_dir`: 198 samples, binary (A/B), 3 input images (topdown + 2 egocentric views)

### AO td_ego_dir noEMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 | s9000 | s10000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| PT2P (acc) | 76.3 | 79.0 | 78.1 | 77.5 | 76.9 | 78.1 | 78.4 | 75.1 | 79.3 | **80.2** |
| SV (acc) | 66.7 | 71.2 | 70.7 | 68.2 | 71.7 | 70.2 | 66.2 | 69.7 | 71.7 | 72.2 |

### AO td_ego_dir EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 | s9000 | s10000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| PT2P (acc) | 53.8 | 77.5 | 79.0 | 79.0 | 79.3 | **80.9** | 79.6 | 79.9 | 79.0 | 79.6 |
| SV (acc) | 59.6 | 71.2 | 69.2 | 67.7 | 72.7 | 72.2 | 72.2 | 71.7 | **73.7** | **73.7** |

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

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 |
|--------|-------|-------|-------|-------|-------|
| PT2P (acc) | -- | -- | -- | -- | -- |
| SV (acc) | **67.2** | 55.1 | 55.1 | 57.1 | 58.1 |

### VCoT l32 td_ego_dir EMA — VCoT image gen (`bagel_mot_vcot`)

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 |
|--------|-------|-------|-------|-------|-------|
| PT2P (acc) | 51.4 | -- | -- | -- | -- |
| SV (acc) | 55.6 | **60.6** | 57.1 | 57.1 | 59.1 |

> PT2P image-gen evals take ~11h (237s/sample diffusion). Only s1000 EMA PT2P complete (51.4%, 8-GPU re-run Mar 3). SV s6k-s7k still running.

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

### VCoT l32 td_ego_dir — VCoT image gen PT2P partial results

Early accuracy from partial pkl files (timed-out 2-GPU runs). s1k EMA now has full 8-GPU result (51.4%, see table above).

| Checkpoint | noEMA | EMA | Samples |
|------------|-------|-----|---------|
| s1k | 50.0 | **51.4** (full, 8-GPU) | 230/329 (noEMA partial) |
| s2k | 55.5 | 55.5 | 220/329 |
| s3k | **60.5** | **61.8** | 220/329 |
| s4k | 59.5 | **62.3** | 220/329 |
| s5k | 56.2 | 59.1 | 220/329 |
| s6k | 48.3 | 51.7 | 60/329 |
| s7k | 60.0 | 58.3 | 60/329 |

> VCoT image-gen PT2P peaks around s3k-s4k at ~62%, well below GT prefill (87.5%), confirming sideview generation quality is the bottleneck.

---

## RealPathTracing (Real Indoor)

Dataset: `linjieli222/real_indoor_path_tracing` | 2 subsets: `td_path` (174 samples), `td_path_arrow` (158 samples)
Single real-world indoor image per sample (no AI2Thor synthetic images).

### AO td_ego_dir — RealPT

**EMA:**

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| td_path | 39.1 | **49.4** | 47.7 | 48.3 | 48.9 | 46.6 | 45.4 | 43.1 |
| td_path_arrow | 60.1 | **74.1** | 68.4 | 63.9 | 63.9 | 62.7 | 57.6 | 56.3 |

**noEMA:**

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| td_path | **46.6** | 45.4 | 44.3 | 42.0 | 43.1 | 41.4 | 44.8 | 40.2 |
| td_path_arrow | **67.1** | 66.5 | 59.5 | 58.2 | 63.9 | 56.3 | 57.0 | 51.9 |

> AO peaks early (s1k-s2k) on real data then declines — overfits to AI2Thor synthetic images. td_path_arrow consistently 15-20pp easier than td_path.

### VCoT l32 td_ego_dir — RealPT (think, `bagel_mot`)

| Subset | s1k EMA | s1k noEMA | s2k EMA | s2k noEMA | s3k EMA | s3k noEMA | s4k EMA | s4k noEMA |
|--------|---------|-----------|---------|-----------|---------|-----------|---------|-----------|
| td_path | 38.5 | 9.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| td_path_arrow | 57.0 | 11.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

> Same as AI2Thor: VCoT think outputs `<image_start>` and never produces an answer in text-only mode. Only s1k EMA retains partial text answering ability.

### VCoT l32 td_ego_dir — RealPT (nothink, `bagel_mot_nothink`)

| Subset | s1k EMA | s1k noEMA | s2k EMA | s2k noEMA | s3k EMA | s3k noEMA | s4k EMA | s4k noEMA |
|--------|---------|-----------|---------|-----------|---------|-----------|---------|-----------|
| td_path | 45.4 | **47.1** | **46.6** | 46.6 | 43.1 | 0.0 | 6.9 | 0.0 |
| td_path_arrow | 65.8 | **68.4** | **68.4** | 59.5 | -- | 0.0 | 3.8 | 1.3 |

> Nothink matches AO at s1k-s2k, then collapses at s3k (same rise-then-collapse pattern as AI2Thor evals). Real-world transfer peaks at s1k-s2k before AI2Thor overfitting destroys generalization.
