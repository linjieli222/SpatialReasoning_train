# Evaluation Results — td_ego_dir (Averaged)

Back to [Eval Results Index](eval_results.md) | [Full results (per-subset)](eval_results_td_ego_dir.md)

> Synthetic avg = (PT2PV2 ego_dir + td_path + td_path_arrow) / 3. Real avg = (RealPT td_path + td_path_arrow) / 2. Averages only shown where all component values are available.
>
> Instance-weighted averages weight by sample count: **Inst. syn avg** = (113×ego_dir + 169×td_path + 171×td_path_arrow) / 453. **Inst. real avg** = (174×td_path + 158×td_path_arrow) / 332.

---

## AO td_ego_dir

Training: 5 epochs, answer-only. 10k steps completed.

#### EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 | s9000 | s10000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| PT2P (acc) | 53.8 | 77.5 | 79.0 | 79.0 | 79.3 | **80.9** | 79.6 | 79.9 | 79.0 | 79.6 |
| Synthetic avg | -- | -- | -- | -- | -- | **65.7** | -- | -- | -- | -- |
| Inst. syn avg | -- | -- | -- | -- | -- | **64.7** | -- | -- | -- | -- |
| Real avg | 46.6 | **61.8** | 58.1 | 56.1 | 56.4 | 54.7 | 51.5 | 49.7 | -- | -- |
| Inst. real avg | 46.1 | **61.2** | 57.6 | 55.7 | 56.0 | 54.3 | 51.2 | 49.4 | -- | -- |

#### noEMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 | s9000 | s10000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| PT2P (acc) | 76.3 | 79.0 | 78.1 | 77.5 | 76.9 | 78.1 | 78.4 | 75.1 | 79.3 | **80.2** |
| Real avg | **56.9** | 56.0 | 51.9 | 50.1 | 53.5 | 48.9 | 50.9 | 46.1 | -- | -- |
| Inst. real avg | **56.4** | 55.4 | 51.5 | 49.7 | 53.0 | 48.5 | 50.6 | 45.8 | -- | -- |

> PT2P peaks at **s6k EMA (80.9%)**. Synthetic avg only available at s6k (**65.7%**). Real avg peaks at **s2k EMA (61.8%)** then declines — overfits to AI2Thor.

---

## VCoT l32 td_ego_dir

Training: VCoT with 512x512 output images. 8k steps completed.

### VCoT image gen (`bagel_mot_vcot`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| PT2P (acc) | 51.4 | 52.0 | 60.5 | 61.4 | 58.1 | 58.4 | **64.4** | 61.7 |
| Synthetic avg | -- | -- | -- | -- | -- | -- | **42.9** | -- |
| Inst. syn avg | -- | -- | -- | -- | -- | -- | **41.9** | -- |
| Real avg | -- | -- | -- | **35.6** | -- | -- | 26.2 | -- |
| Inst. real avg | -- | -- | -- | **35.5** | -- | -- | 26.2 | -- |

> VCoT image gen: Synthetic avg only at s7k (**42.9%**) — ego_dir (50.4%) stronger than td_path (42.6%) / td_path_arrow (35.7%). Real avg declines from s4k to s7k.

### VCoT prefill (`bagel_mot_vcot_prefill`) — s7k EMA

| Subset | Accuracy |
|--------|----------|
| PT2P | **90.9%** (s3k) |
| Synthetic avg | **55.6%** |
| Inst. syn avg | **51.6%** |
| PT2PV2 ego_dir | 86.7% |
| PT2PV2 td_path | 42.0% |
| PT2PV2 td_path_arrow | 38.0% |

> Prefill synthetic avg (**55.6%**) dragged down by poor td_path generalization. ego_dir (86.7%) vs td_path (42.0%) gap confirms model doesn't generalize sideview reasoning across task formats.

### No-think (`bagel_mot_nothink`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| PT2P (acc) | 56.5 | **74.5** | 34.7 | -- | -- | -- | 0.0 | -- |
| PT2PV2 ego_dir (acc) | 43.4 | **61.1** | 25.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Synthetic avg | -- | **49.0** | -- | -- | -- | -- | -- | -- |
| Inst. syn avg | -- | **47.5** | -- | -- | -- | -- | -- | -- |
| Real avg | **55.6** | **57.5** | -- | 5.4 | -- | -- | -- | -- |
| Inst. real avg | **55.1** | **57.0** | -- | 5.4 | -- | -- | -- | -- |

#### noEMA

| Subset | s1000 | s2000 | s3000 |
|--------|-------|-------|-------|
| PT2P (acc) | **72.0** | 71.7 | 0.0 |
| Real avg | **57.8** | 53.1 | 0.0 |
| Inst. real avg | **57.2** | 52.7 | 0.0 |

> **Rise-then-collapse**: Peaks at s2k (synthetic avg **49.0%**, real avg **57.5%**), collapses by s3k.

### Answeronly (`bagel_mot_answeronly`) — s2k EMA

| Subset | Accuracy |
|--------|----------|
| PT2PV2 ego_dir | 15.9 (invalid) |
| PT2PV2 td_path | 44.4 |
| PT2PV2 td_path_arrow | 46.8 |
| Real avg | **62.3** |
| Inst. real avg | **61.7** |

> ego_dir invalid (truncated predictions). Real avg strong at **62.3%** (inst. **61.7%**).

---

## Mixed VCoT+AO td_ego_dir

Training: 50% VCoT + 50% AO with VCoT system prompt. 10k steps, complete.

### Think (`bagel_mot`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 |
|--------|-------|-------|-------|-------|-------|
| PT2PV2 ego_dir (acc) | 45.1 | 61.1 | 69.0 | **70.8** | 69.0 |

### No-think (`bagel_mot_nothink`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| PT2PV2 ego_dir (acc) | 38.9 | 64.6 | 70.8 | 69.0 | 70.8 | 72.6 | **73.5** | 72.6 |
| Synthetic avg | 34.9 | 55.1 | 63.8 | 64.8 | 65.2 | 66.4 | 65.3 | **67.5** |
| Inst. syn avg | 34.4 | 53.9 | 62.9 | 64.2 | 64.5 | 65.6 | 64.2 | **66.9** |
| Real avg | -- | -- | -- | **56.1** | 53.8 | 52.5 | 51.0 | 52.5 |
| Inst. real avg | -- | -- | -- | **55.5** | 53.3 | 52.1 | 50.6 | 52.1 |

### No-think with VAE (`bagel_mot_nothink_vcot`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|
| PT2PV2 ego_dir (acc) | 53.1 | 65.5 | 69.0 | 69.0 | 70.8 | 69.9 | **71.7** | **71.7** |
| Synthetic avg | -- | -- | -- | 63.8 | -- | **65.5** | **66.5** | **66.7** |
| Inst. syn avg | -- | -- | -- | 63.1 | -- | **64.9** | **65.8** | **66.0** |
| Real avg | -- | -- | -- | 52.3 | **52.4** | 50.4 | -- | 48.6 |
| Inst. real avg | -- | -- | -- | 51.8 | **51.8** | 50.0 | -- | 48.2 |

### VCoT image gen (`bagel_mot_vcot`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 |
|--------|-------|-------|-------|-------|-------|
| PT2PV2 ego_dir (acc) | 50.4 | 43.4 | 44.2 | 52.2 | **54.9** |

### Answeronly (`bagel_mot_answeronly`) — EMA

| Subset | s4000 | s5000 | s6000 | s7000 | s8000 |
|--------|-------|-------|-------|-------|-------|
| PT2PV2 ego_dir (acc) | 58.4 | 69.9 | 69.9 | **70.8** | **70.8** |
| Synthetic avg | 60.5 | 63.7 | **65.7** | 65.2 | **66.5** |
| Inst. syn avg | 60.7 | 62.9 | **65.1** | 64.5 | **66.0** |
| Real avg | **54.2** | 53.4 | 52.2 | 52.8 | 51.2 |
| Inst. real avg | **53.6** | 53.0 | 51.8 | 52.4 | 50.9 |

> **Mixed prevents collapse.** Nothink synthetic avg peaks at **s8k (67.5%)**, real avg at **s4k (56.1%)**. Answeronly synthetic avg peaks at **s8k (66.5%)**. nothink_vcot synthetic avg steady at **66.5-66.7%** (s7k-s8k).

---

## Mixed from VCoT td_ego_dir

### Mixed from VCoT s7k (MFV7k)

Initialized from VCoT l32 s7k EMA (collapsed). 5k steps, complete.

#### No-think (`bagel_mot_nothink`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 |
|--------|-------|-------|-------|-------|-------|
| PT2PV2 ego_dir (acc) | 19.5 | **72.6** | 71.7 | 70.8 | 71.7 |
| Synthetic avg | 22.6 | **65.8** | **65.9** | 64.0 | 65.6 |
| Inst. syn avg | 22.9 | **64.9** | **65.1** | 63.1 | 64.9 |
| Real avg | 20.5 | **48.5** | 47.9 | 47.3 | 46.6 |
| Inst. real avg | 20.2 | **48.2** | 47.6 | 47.0 | 46.4 |

#### Answeronly (`bagel_mot_answeronly`) — s5k EMA

| Subset | Accuracy |
|--------|----------|
| PT2PV2 ego_dir | **74.3** |
| PT2PV2 td_path | 62.1 |
| PT2PV2 td_path_arrow | 64.3 |
| Synthetic avg | **66.9** |
| Inst. syn avg | **66.0** |
| Real avg | **45.5** |
| Inst. real avg | **45.2** |

> MFV7k nothink: recovers from collapse by s2k. Synthetic avg stable at **~65-66%** (s2k-s5k). Real avg peaks at **s2k (48.4%)**. Answeronly s5k synthetic avg **66.9%** — highest among answeronly evals, but real only **45.5%**.

### Mixed from VCoT s2k (MFV2k)

Initialized from VCoT l32 s2k EMA (pre-collapse). 5k steps, training complete.

#### No-think (`bagel_mot_nothink`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 |
|--------|-------|-------|-------|-------|-------|
| PT2PV2 ego_dir (acc) | 63.7 | 67.3 | 71.7 | **73.5** | 72.6 |
| Synthetic avg | 53.2 | 60.9 | 64.5 | **64.7** | 63.8 |
| Inst. syn avg | 51.8 | 60.1 | 63.6 | **63.6** | 62.7 |
| Real avg | **61.0** | **62.1** | 61.1 | 58.6 | 55.1 |
| Inst. real avg | **60.5** | **61.5** | 60.5 | 58.2 | 54.8 |

#### Answeronly (`bagel_mot_answeronly`) — EMA

| Subset | s3000 | s4000 | s5000 |
|--------|-------|-------|-------|
| PT2PV2 ego_dir (acc) | 71.7 | 69.9 | **72.6** |
| Synthetic avg | 64.1 | **65.5** | 63.8 |
| Inst. syn avg | 63.2 | **64.9** | 62.7 |
| Real avg | **56.1** | 53.9 | 54.5 |
| Inst. real avg | **55.5** | 53.6 | 54.2 |

> MFV2k nothink: stronger start than MFV7k. Synthetic avg peaks at **s4k (64.7%)**. Real avg peaks at **s2k (62.1%)** — highest real avg across all models. Real avg declines steadily with training (overfitting). Answeronly synthetic peaks at **s4k (65.5%)**, real at **s3k (56.1%)**.

---

## TextCoT td_ego_dir

Training: text chain-of-thought. 10k steps completed.

### Think (`bagel_mot`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 | s9000 | s10000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| PT2P (acc) | 50.8 | 59.6 | 61.1 | 64.4 | **67.8** | 65.7 | 63.2 | 65.7 | 65.0 | 62.9 |
| PT2PV2 ego_dir (acc) | 38.1 | 45.1 | 53.1 | **57.5** | 53.1 | 54.0 | 51.3 | 54.0 | 56.6 | 49.6 |
| Synthetic avg | -- | -- | -- | **50.2** | 49.7 | -- | -- | -- | -- | 46.6 |
| Inst. syn avg | -- | -- | -- | **49.3** | 49.2 | -- | -- | -- | -- | 46.2 |
| Real avg | -- | -- | -- | **58.3** | 51.8 | -- | -- | -- | -- | 45.6 |
| Inst. real avg | -- | -- | -- | **58.2** | 51.8 | -- | -- | -- | -- | 45.5 |

### No-think (`bagel_mot_nothink`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 | s8000 | s9000 | s10000 |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| PT2P (acc) | 47.7 | 56.5 | 59.9 | 63.5 | 64.7 | 63.2 | 63.5 | 65.3 | -- | **66.0** |
| PT2PV2 ego_dir (acc) | 38.9 | 41.6 | 47.8 | 46.0 | 55.8 | 47.8 | **58.4** | 50.4 | 54.0 | 54.0 |
| Synthetic avg | -- | -- | -- | -- | **49.4** | -- | **51.2** | -- | -- | -- |
| Inst. syn avg | -- | -- | -- | -- | **48.6** | -- | **50.3** | -- | -- | -- |
| Real avg | -- | -- | -- | -- | **52.2** | -- | 46.4 | -- | -- | -- |
| Inst. real avg | -- | -- | -- | -- | **52.1** | -- | 46.4 | -- | -- | -- |

> TextCoT think: synthetic avg peaks at **s4k (50.2%)**, real avg at **s4k (58.3%)**. Nothink: synthetic avg peaks at **s7k (51.2%)**, real avg at **s5k (52.2%)**.

---

## MMCoT td_ego_dir

Training: multimodal CoT with sideview + text reasoning. Stopped at s7k.

### No-think (`bagel_mot_nothink`) — EMA

| Subset | s1000 | s2000 | s3000 | s4000 | s5000 | s6000 | s7000 |
|--------|-------|-------|-------|-------|-------|-------|-------|
| PT2P (acc) | 48.0 | 63.2 | 66.0 | 69.6 | 68.7 | **71.4** | 69.6 |
| PT2PV2 ego_dir (acc) | 33.6 | 39.8 | 52.2 | 59.3 | **62.8** | 59.3 | 59.3 |
| Synthetic avg | -- | -- | -- | -- | **44.7** | -- | -- |
| Inst. syn avg | -- | -- | -- | -- | **42.4** | -- | -- |
| Real avg | -- | -- | -- | -- | **51.0** | -- | -- |
| Inst. real avg | -- | -- | -- | -- | **50.6** | -- | -- |

> MMCoT nothink: PT2P peaks at **s6k (71.4%)**. Synthetic avg at s5k only (**44.7%**) — td_path/arrow much weaker than ego_dir.

---

## Baseline (BAGEL-7B-MoT)

| Subset | Accuracy |
|--------|----------|
| PT2PV2 ego_dir | 36.3 |
| PT2PV2 td_path | 26.0 |
| PT2PV2 td_path_arrow | 27.5 |
| Synthetic avg | **29.9** |
| Inst. syn avg | **29.1** |
| Real avg | **42.7** |
| Inst. real avg | **42.5** |

---

## Cross-Model Comparison (best checkpoints)

| Model | Best Syn Avg | Best Inst. Syn | Best Real Avg | Best Inst. Real | Best PT2P |
|-------|-------------|----------------|---------------|-----------------|-----------|
| Baseline | 29.9 | 29.1 | 42.7 | 42.5 | -- |
| AO | 65.7 (s6k) | 64.7 (s6k) | 61.8 (s2k) | 61.2 (s2k) | 80.9 (s6k) |
| VCoT nothink | 49.0 (s2k) | 47.5 (s2k) | 57.5 (s2k) | 57.0 (s2k) | 74.5 (s2k) |
| VCoT prefill | 55.6 (s7k) | 51.6 (s7k) | -- | -- | 90.9 (s3k) |
| VCoT image gen | 42.9 (s7k) | 41.9 (s7k) | 35.6 (s4k) | 35.5 (s4k) | 64.4 (s7k) |
| Mixed nothink | **67.5** (s8k) | **66.9** (s8k) | 56.1 (s4k) | 55.5 (s4k) | -- |
| Mixed nothink_vcot | 66.7 (s8k) | 66.0 (s8k) | 52.4 (s5k) | 51.8 (s5k) | -- |
| Mixed answeronly | 66.5 (s8k) | 66.0 (s8k) | 54.2 (s4k) | 53.6 (s4k) | -- |
| MFV7k nothink | 65.9 (s3k) | 65.1 (s3k) | 48.5 (s2k) | 48.2 (s2k) | -- |
| MFV7k answeronly | 66.9 (s5k) | 66.0 (s5k) | 45.5 (s5k) | 45.2 (s5k) | -- |
| MFV2k nothink | 64.7 (s4k) | 63.6 (s4k) | **62.1** (s2k) | **61.5** (s2k) | -- |
| MFV2k answeronly | 65.5 (s4k) | 64.9 (s4k) | 56.1 (s3k) | 55.5 (s3k) | -- |
| TextCoT think | 50.2 (s4k) | 49.3 (s4k) | 58.3 (s4k) | 58.2 (s4k) | 67.8 (s5k) |
| TextCoT nothink | 51.2 (s7k) | 50.3 (s7k) | 52.2 (s5k) | 52.1 (s5k) | 66.0 (s10k) |
| MMCoT nothink | 44.7 (s5k) | 42.4 (s5k) | 51.0 (s5k) | 50.6 (s5k) | 71.4 (s6k) |
