# Evaluation Results — td_ego_dir Training Regime

Back to [Eval Results Index](eval_results.md)

> **Data freshness**: Last updated 2026-03-04. VCoT l32 image-gen evals complete through s8k for both SV (F1) and PT2P (acc). AO td_ego_dir evals complete through s10000. VCoT mse2/mse5 s1k evals complete. Mixed VCoT+AO, self-forcing, TextCoT, and MMCoT experiments launched.

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

The model generates sideview images as visual thoughts (`<think>desc</think><image_start>[generated image]<image_end><answer>X</answer>`), then uses them to answer.

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| PT2P (acc) | -- | -- | -- | -- | -- | -- | -- | -- |
| SV (F1) | 67.5 | 78.8 | 81.7 | 85.7 | 80.2 | 83.7 | **87.0** | 86.4 |

### VCoT l32 td_ego_dir EMA — VCoT image gen (`bagel_mot_vcot`)

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| PT2P (acc) | 51.4 | 52.0 | 60.5 | 61.4 | 58.1 | 58.4 | **64.4** | 61.7 |
| SV (F1) | 47.3 | 70.1 | 85.7 | 83.0 | 84.4 | 83.7 | **87.0** | 85.1 |

> PT2P and SV image-gen evals complete through s8k EMA. Best SV F1: **87.0** at s7k (both ema/noema). PT2P peaks at **s7k (64.4%)**, recovering from the s5k dip. SV and PT2P peaks align at s7k.

### VCoT l32 td_ego_dir — VCoT prefill (`bagel_mot_vcot_prefill`)

VCoT-trained model evaluated with **ground-truth sideview images prefilled** as visual thoughts. The model receives GT images in the `<think>` chain and only predicts the answer. This measures reasoning ability independent of image generation quality.

| Mode | PT2P Accuracy | Samples |
|------|--------------|---------|
| vcot_prefill (s3k EMA) | **90.3%** | 271/300 (91% done) |
| answer-only best (s6k EMA) | 80.9% | 329/329 |
| vcot (s3k EMA) | 60.5% | 329/329 |

> The ~30pp gap between vcot_prefill (90.3%) and end-to-end vcot (60.5%) confirms that **sideview generation quality is the bottleneck**, not the model's reasoning ability. The vcot_prefill result also exceeds the best answer-only result (80.9%) by ~10pp, showing that visual thoughts genuinely help when image quality is high.

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

s1k-s8k EMA now have full official results (see table above). noEMA results are partial estimates from pkl files.

| Checkpoint | noEMA (partial) | EMA (full) |
|------------|-----------------|------------|
| s4k | 59.6 (220/329) | 61.4 |
| s5k | 56.7 (220-240/329) | 58.1 |

> VCoT image-gen PT2P peaks at **s7k (64.4%)**, well below GT prefill (90.3%), confirming sideview generation quality is the bottleneck.

---

## VCoT l32 td_ego_dir — mse_weight variants

Ablation on MSE loss weight. All use same VCoT l32 config, only `--mse_weight` differs.

Base dir: `.../tifa_v3_td_ego_dir_vcot_l32/vcot_td_ego_dir_{mse2,mse5}_8gpu/`

### VCoT mse2 EMA — VCoT image gen (`bagel_mot_vcot`)

| Subset | s1000 |
|--------|-------|
| PT2P (acc) | 47.7 |

### VCoT mse5 EMA — VCoT image gen (`bagel_mot_vcot`)

| Subset | s1000 |
|--------|-------|
| PT2P (acc) | 49.8 |

> Both mse2 and mse5 underperform the default mse1 at s1k (51.4% PT2P). Higher MSE weight slows down the text reasoning improvement early in training. Training stopped at s5k for both variants.

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

### VCoT l32 td_ego_dir — RealPT (vcot image gen, `bagel_mot_vcot`)

| Subset | s4k EMA |
|--------|---------|
| td_path | 34.5 |
| td_path_arrow | 36.7 |

> VCoT image-gen on real data: much lower than AI2Thor (61.4% PT2P) and lower than AO best on real data (49.4% td_path, 74.1% td_path_arrow). Sideview generation quality degrades further on out-of-distribution real images.
