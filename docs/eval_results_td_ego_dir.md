# Evaluation Results — td_ego_dir Training Regime

Back to [Eval Results Index](eval_results.md)

> **Data freshness**: Last updated 2026-03-04. AO complete through s10k. VCoT l32 complete through s8k. TextCoT think through s9k, nothink through s8k. MMCoT nothink through s6k. Mixed VCoT+AO through s5k. Baseline PT2PV2 available. First answeronly result (VCoT s2k).

## Figures

![AO td_ego_dir training curves — PT2P and SV](figures/ao_ego_dir_training_curves.png)

![VCoT l32 td_ego_dir eval setting comparison at s1k and s2k](figures/vcot_l32_eval_comparison.png)

![RealPathTracing overfitting pattern — AO td_ego_dir](figures/realpt_training_curves.png)

---

## AO td_ego_dir

Config: `bagel_mot` (text-only, think=True) | Base dir: `.../tifa_v3_td_ego_dir_ao/ao_td_ego_dir_8gpu/`
Training: 5 epochs on td_ego_dir subset. 10k steps completed.

#### EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 | s9000 | s10000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| PT2P (acc) | 53.8 | 77.5 | 79.0 | 79.0 | 79.3 | **80.9** | 79.6 | 79.9 | 79.0 | 79.6 |
| PT2PV2 (acc) | -- | -- | -- | -- | -- | 73.5 | -- | -- | -- | -- |
| SV (acc) | 59.6 | 71.2 | 69.2 | 67.7 | 72.7 | 72.2 | 72.2 | 71.7 | **73.7** | **73.7** |
| SV (F1) | 54.5 | 71.1 | 68.7 | 67.3 | 72.5 | 72.2 | 72.1 | 71.6 | 73.7 | **73.7** |
| RealPT path (acc) | 36.2 | **49.4** | 47.7 | 48.3 | 48.9 | 46.6 | 45.4 | 43.1 | -- | -- |
| RealPT arrow (acc) | 57.0 | **74.1** | 68.4 | 63.9 | 63.9 | 62.7 | 57.6 | 56.3 | -- | -- |

#### noEMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 | s9000 | s10000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| PT2P (acc) | 76.3 | 79.0 | 78.1 | 77.5 | 76.9 | 78.1 | 78.4 | 75.1 | 79.3 | **80.2** |
| SV (acc) | 66.7 | 71.2 | 70.7 | 68.2 | 71.7 | 70.2 | 66.2 | 69.7 | 71.7 | 72.2 |
| SV (F1) | 66.5 | 70.9 | 70.3 | 67.7 | 71.6 | 70.0 | 65.4 | 69.7 | 71.7 | 72.2 |
| RealPT path (acc) | **46.6** | 45.4 | 44.3 | 42.0 | 43.1 | 41.4 | 44.8 | 40.2 | -- | -- |
| RealPT arrow (acc) | **67.1** | 66.5 | 59.5 | 58.2 | 63.9 | 56.3 | 57.0 | 51.9 | -- | -- |

> PT2P peaks at **s6k EMA (80.9%)** and **s10k noEMA (80.2%)**. SV peaks at s9k-s10k EMA (73.7%). RealPT peaks early (s1k-s2k) then declines — overfits to AI2Thor synthetic images. td_path_arrow consistently 15-20pp easier than td_path.

---

## VCoT l32 td_ego_dir

Config: `bagel_mot_vcot` (image generation, think=True) | Base dir: `.../tifa_v3_td_ego_dir_vcot_l32/vcot_td_ego_dir_8gpu/`
Training: VCoT with 512x512 output images (latent 32). 8k steps completed.

**Eval settings:**

| Setting | Config | think | understanding_output | vae_input | Image Gen | System Prompt |
|---------|--------|-------|---------------------|-----------|-----------|---------------|
| VCoT (image gen) | `bagel_mot_vcot` | True | False | False (default) | Yes | VCoT think |
| VCoT prefill | `bagel_mot_vcot_prefill` | True | False | False | Yes (prefilled GT) | VCoT think |
| Think (text-only) | `bagel_mot` | True | True | False | No | VCoT think |
| No-think | `bagel_mot_nothink` | False | True | False | No | Answer-only |
| No-think with VAE | `bagel_mot_nothink_vcot` | False | False | False (default) | Yes | Answer-only |
| Answeronly | `bagel_mot_answeronly` | False | True | True | No | Answer-only |

- **`understanding_output=True`**: Text-only generation. Input images encoded via ViT only (no VAE). No image generation loop.
- **`understanding_output=False`**: Enables image generation loop. Input images encoded via both VAE and ViT. If model outputs `<image_start>`, an image will be generated.
- **`vae_input=True`**: Forces VAE input encoding even with `understanding_output=True`. Fixes train-eval mismatch for models trained with `visual_gen=True`.
- **`think=True`**: Uses VCoT system prompt ("think step by step... visual thinking..."). Model may produce `<think>`, `<image_start>`, `<answer>` tags.
- **`think=False`**: Uses answer-only system prompt ("Answer the question"). Model outputs `<answer>` directly.

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
Training: 50% VCoT (sideview generation) + 50% AO (answer-only with VCoT system prompt). Designed to prevent VCoT commitment collapse. Training ongoing.

### Think (`bagel_mot`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 |
|--------|-------|-------|-------|-------|-------|
| PT2PV2 (acc) | 45.1 | 61.1 | 69.0 | **70.8** | 69.0 |

### No-think (`bagel_mot_nothink`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 |
|--------|-------|-------|-------|-------|-------|
| PT2PV2 ego_dir (acc) | 38.9 | 64.6 | **70.8** | 69.0 | **70.8** |
| PT2PV2 td_path (acc) | 32.0 | 49.7 | 60.4 | 62.1 | **63.3** |
| PT2PV2 td_path_arrow (acc) | 33.9 | 50.9 | 60.2 | **63.2** | 61.4 |

### No-think with VAE input (`bagel_mot_nothink_vcot`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 |
|--------|-------|-------|-------|-------|-------|
| PT2PV2 (acc) | 53.1 | 65.5 | 69.0 | 69.0 | **70.8** |

### VCoT image gen (`bagel_mot_vcot`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 |
|--------|-------|-------|-------|-------|-------|
| PT2PV2 (acc) | 50.4 | 43.4 | 44.2 | 52.2 | **54.9** |

> **Key result: Mixed training prevents VCoT collapse.** Think PT2PV2 peaks at **s4k (70.8%)**, nothink at **s3k-s5k (70.8%)** — no collapse unlike pure VCoT. Nothink generalizes to td_path (63.3%) and td_path_arrow (63.2%). nothink_vcot (VAE input) catches up to nothink at **s5k (70.8%)**. VCoT image-gen improves to **s5k (54.9%)** after initial dip.

---

## TextCoT td_ego_dir

Config: `bagel_mot` (text-only, think=True) | Base dir: `.../tifa_v3_td_ego_dir_text_cot/textcot_td_ego_dir_8gpu/`
Training: TextCoT with text chain-of-thought reasoning (no image generation). Training ongoing.

### Think (`bagel_mot`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 | s9000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| PT2P (acc) | 50.8 | 59.6 | 61.1 | 64.4 | **67.8** | 65.7 | 63.2 | 65.7 | 65.0 |
| PT2PV2 (acc) | 38.1 | 45.1 | 53.1 | **57.5** | 53.1 | 54.0 | -- | 54.0 | -- |
| SV (acc) | 52.0 | 56.6 | 58.6 | 62.6 | 64.6 | 63.1 | **64.6** | 62.1 | 61.1 |
| SV (F1) | 21.5 | 41.1 | 48.1 | 54.3 | 58.8 | 57.3 | **60.2** | 58.6 | 59.1 |

### No-think (`bagel_mot_nothink`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| PT2P (acc) | 47.7 | 56.5 | 59.9 | 63.5 | **64.7** | 63.2 | 63.5 | 65.3 |
| PT2PV2 (acc) | 38.9 | 41.6 | 47.8 | 46.0 | **55.8** | 47.8 | -- | 50.4 |
| SV (acc) | 51.5 | 55.1 | 64.1 | 63.6 | 63.6 | **64.1** | **64.1** | 61.6 |
| SV (F1) | 20.0 | 38.6 | 53.0 | 53.8 | 52.0 | 55.9 | **57.5** | 53.1 |

> PT2P peaks at **s5k (67.8% think)** then dips at s6k-s7k, recovers to 65.7% at s8k-s9k. PT2PV2 peaks at **s4k (57.5% think)**. SV peaks at **s7k (64.6%/F1=60.2 think)**. Nothink PT2P steadily improves to **s5k (64.7%)**, plateaus, then recovers at s8k (65.3%). s9k nothink PT2P dips to 61.1%.

---

## MMCoT td_ego_dir

Config: `bagel_mot_vcot` (image generation, think=True) | Base dir: `.../tifa_v3_td_ego_dir_mmcot/mmcot_td_ego_dir_8gpu/`
Training: Multimodal CoT with sideview generation + text reasoning. 10k steps, latent 32 (512x512 output).

### No-think (`bagel_mot_nothink`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 |
|--------|-------|-------|-------|-------|-------|-------|
| PT2P (acc) | 48.0 | 63.2 | 66.0 | 69.6 | 68.7 | **71.4** |
| PT2PV2 (acc) | 33.6 | 39.8 | 52.2 | 59.3 | **62.8** | 59.3 |
| SV (acc) | 52.0 | 60.1 | 66.2 | 68.7 | **70.2** | 67.2 |
| SV (F1) | 20.2 | 43.2 | 55.0 | 59.2 | **63.4** | 59.6 |

> MMCoT nothink improves steadily. PT2P peaks at **s6k (71.4%)**, approaching AO levels (80.9%). SV peaks at **s5k (70.2%/F1=63.4)**, then dips at s6k. PT2PV2 peaks at **s5k (62.8%)**. Training ongoing.

---

## Baseline (BAGEL-7B-MoT)

Config: `bagel_mot` (text-only, think=True) | Model: base BAGEL-7B-MoT (no fine-tuning)

| Subset | Accuracy |
|--------|----------|
| PT2PV2 td_path | 26.0 |
| PT2PV2 td_ego_dir | 36.3 |
| PT2PV2 td_path_arrow | -- |
| PT2P td_path SV | 42.5 |
| RealPT td_path | 39.7 |
| RealPT td_path_arrow | 45.6 |

> PT2PV2 td_ego_dir (36.3%) significantly higher than td_path (26.0%) at baseline — ego_dir is an easier task format. PT2PV2 td_path_arrow and remaining RealPT evals pending.

---

## Answeronly eval (`bagel_mot_answeronly`)

Config: `understanding_output=True, vae_input=True, think=False` — text-only generation with VAE input encoding matching training.

**Why this config exists:** Models trained with `visual_gen=True` (VCoT, MMCoT, Mixed) encode input images through both VAE and ViT during training. But the standard text-only eval configs (`bagel_mot`, `bagel_mot_nothink`) use `understanding_output=True` which skips VAE input encoding (ViT only). This creates a train-eval mismatch. `bagel_mot_answeronly` fixes this by setting `vae_input=True` to include VAE encoding at eval time, matching the training setup. It still generates text-only output (no image generation loop).

### VCoT l32 s2k EMA

| Subset | Accuracy |
|--------|----------|
| PT2PV2 td_path | 44.4 |

> First answeronly result. Remaining subsets (td_path_arrow, td_ego_dir, RealPT) and Mixed s4k results pending.

