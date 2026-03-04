# Evaluation Results — td_ego_dir Training Regime

Back to [Eval Results Index](eval_results.md)

> **Data freshness**: Last updated 2026-03-04. AO complete through s10k. VCoT l32 image-gen complete through s8k. VCoT nothink PT2PV2 complete (s1k-s8k). TextCoT EMA think through s6k (s5k pending). MMCoT EMA nothink through s4k. Mixed VCoT+AO PT2PV2 through s3k. TextCoT nothink + PT2PV2 evals in progress.

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
Training: VCoT with 512x512 output images (latent 32). 8k steps completed.

**Eval settings:**

| Setting | Config | Image Gen | Description |
|---------|--------|-----------|-------------|
| VCoT (image gen) | `bagel_mot_vcot` | Yes (512x512) | Full VCoT: generates sideview image then answers |
| VCoT prefill | `bagel_mot_vcot_prefill` | Prefilled GT | GT sideview injected, model only answers |
| Think (text-only) | `bagel_mot` | No | Text-only; model may output `<image_start>` but no image generated |
| No-think | `bagel_mot_nothink` | No | Answer-only prompt, no thinking or image generation |

### VCoT image gen (`bagel_mot_vcot`)

#### EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| PT2P (acc) | 51.4 | 52.0 | 60.5 | 61.4 | 58.1 | 58.4 | **64.4** | 61.7 |
| SV (acc) | 55.6 | 60.6 | 57.1 | 57.1 | 59.1 | **60.1** | **60.1** | 58.1 |
| SV (F1) | 52.8 | 60.5 | 55.5 | 56.1 | 58.3 | **59.5** | 58.8 | 56.9 |
| RealPT path (acc) | -- | -- | -- | 34.5 | -- | -- | -- | -- |
| RealPT arrow (acc) | -- | -- | -- | 36.7 | -- | -- | -- | -- |

#### noEMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| SV (acc) | **67.2** | 55.1 | 55.1 | 57.1 | 58.1 | 55.6 | 60.6 | 59.1 |
| SV (F1) | **66.3** | 54.5 | 54.1 | 55.5 | 59.2 | 54.2 | 59.4 | 57.9 |

> PT2P peaks at **s7k EMA (64.4%)**. SV peaks at s6k-s7k EMA (~60%). noEMA SV peaks at s1k then drops. Model skews toward predicting 'A'.

### VCoT prefill (`bagel_mot_vcot_prefill`)

GT sideview images injected as visual thoughts — measures reasoning ability independent of image generation quality.

| Subset | Checkpoint | Accuracy | Samples |
|--------|------------|----------|---------|
| PT2P | s3k EMA | **90.9%** | 329 |
| PT2PV2 | s7k EMA | **86.7%** | 113 (partial) |
| AO best (reference) | s6k EMA | 80.9% | 329 |
| VCoT end-to-end (reference) | s3k EMA | 60.5% | 329 |

> The ~30pp gap between prefill (90.9%) and end-to-end VCoT (60.5%) confirms **sideview generation quality is the bottleneck**. Prefill exceeds AO best (80.9%) by ~10pp, showing visual thoughts genuinely help when image quality is high.

### Think text-only (`bagel_mot`)

#### EMA

| Subset | s1000 | s2000 | s3000 |
|--------|-------|-------|-------|
| PT2P (acc) | 55.3 | 68.1 | -- |
| SV (acc) | 57.6 | 66.2 | 68.7 |
| SV (F1) | 51.9 | 65.1 | 68.3 |
| RealPT path (acc) | 30.5 | 0.0 | 0.0 |
| RealPT arrow (acc) | 47.5 | 0.0 | 0.0 |

#### noEMA

| Subset | s1000 | s2000 |
|--------|-------|-------|
| PT2P (acc) | 71.7 | -- |
| SV (acc) | 67.2 | 68.2 |
| SV (F1) | 66.5 | 67.4 |
| RealPT path (acc) | 9.2 | 0.0 |
| RealPT arrow (acc) | 11.4 | 0.0 |

> Text-only think mode: model outputs `<image_start>` and stops producing answers after s1k-s2k. RealPT drops to 0% as model commits to image generation. Partial results only (replaced by nothink eval).

### No-think (`bagel_mot_nothink`)

#### EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| PT2P (acc) | 56.5 | **74.5** | 34.7 | -- | -- | -- | 0.0 | -- |
| PT2PV2 (acc) | 43.4 | **61.1** | 25.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| SV (acc) | 56.6 | **64.6** | 34.3 | -- | -- | -- | 0.0 | -- |
| SV (F1) | 49.5 | **63.4** | 44.3 | -- | -- | -- | 0.0 | -- |
| RealPT path (acc) | 45.4 | **46.6** | 43.1 | 6.9 | -- | -- | -- | -- |
| RealPT arrow (acc) | **65.8** | **68.4** | -- | 3.8 | -- | -- | -- | -- |

#### noEMA

| Subset | s1000 | s2000 | s3000 |
|--------|-------|-------|-------|
| PT2P (acc) | **72.0** | 71.7 | 0.0 |
| SV (acc) | 67.2 | **68.7** | 0.5 |
| SV (F1) | 66.6 | **69.9** | 1.0 |
| RealPT path (acc) | **47.1** | 46.6 | 0.0 |
| RealPT arrow (acc) | **68.4** | 59.5 | 0.0 |

> **Rise-then-collapse**: Peaks at s2k, collapses by s3k. noEMA collapses completely (0.0%), EMA partially (34.7% PT2P). Same pattern on PT2PV2 and RealPT. VCoT commitment strengthens rapidly between s2k-s3k.

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

## Mixed VCoT+AO td_ego_dir

Config: mixed `bagel_mot_vcot` system prompt for both VCoT and AO samples | Base dir: `.../tifa_v3_td_ego_dir_mixed/mixed_vcot_ao_td_ego_dir_8gpu/`
Training: 50% VCoT (sideview generation) + 50% AO (answer-only with VCoT system prompt). Designed to prevent VCoT commitment collapse.

### Mixed VCoT+AO td_ego_dir EMA — VCoT image gen (`bagel_mot_vcot`)

| Subset | s1000 | s2000 | s3000 |
|--------|-------|-------|-------|
| PT2PV2 (acc) | **50.4** | 43.4 | 44.2 |

> Early results only. PT2P vcot image-gen evals still running for s4k-s5k. Nothink and RealPT evals in progress.

---

## TextCoT td_ego_dir

Config: `bagel_mot` (text-only, think=True) | Base dir: `.../tifa_v3_td_ego_dir_text_cot/textcot_td_ego_dir_8gpu/`
Training: TextCoT with text chain-of-thought reasoning (no image generation). 10k steps on td_ego_dir subset.

### TextCoT td_ego_dir EMA — Think (`bagel_mot`)

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 |
|--------|-------|-------|-------|-------|-------|-------|
| PT2P (acc) | 50.8 | 59.6 | 61.1 | 64.4 | -- | **65.7** |
| SV (acc) | 52.0 | 56.6 | 58.6 | 62.6 | -- | **63.1** |
| SV (F1) | 43.5 | 53.3 | 56.8 | 61.3 | -- | **62.4** |

> Steady improvement through s6k. PT2P reaches 65.7% at s6k. SV improves consistently. s5k eval still running. Nothink and PT2PV2 evals pending.

---

## MMCoT td_ego_dir

Config: `bagel_mot_vcot` (image generation, think=True) | Base dir: `.../tifa_v3_td_ego_dir_mmcot/mmcot_td_ego_dir_8gpu/`
Training: Multimodal CoT with sideview generation + text reasoning. 10k steps, latent 32 (512x512 output).

### MMCoT td_ego_dir EMA — No-think (`bagel_mot_nothink`)

| Subset | s1000 | s2000 | s3000 | s4000 |
|--------|-------|-------|-------|-------|
| PT2PV2 (acc) | 33.6 | 39.8 | 52.2 | **59.3** |
| SV (acc) | 52.0 | 60.1 | 66.2 | **68.7** |
| SV (F1) | 42.9 | 56.2 | 64.0 | **66.9** |

> MMCoT nothink improves steadily through s4k. SV at s4k (68.7%) approaches AO levels. PT2PV2 at 59.3% at s4k. Training ongoing.

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
