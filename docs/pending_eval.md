# Pending Eval & Training Jobs

Last updated: 2026-02-24 09:40

## Running (10 jobs, from earlier)

| Job | Name | Elapsed | Node |
|-----|------|---------|------|
| 62805 | MMCoT s7000 EMA vcot td_path | ~11h | g005 |
| 62806 | MMCoT s7000 noEMA vcot td_path | ~11h | g005 |
| 62807 | MMCoT s7000 EMA vcot td_path_arrow | ~11h | g024 |
| 62808 | MMCoT s7000 noEMA vcot td_path_arrow | ~11h | g010 |
| 62845 | VCoT l64 s7000 noEMA vcot td_path | ~10h | g005 |
| 62846 | VCoT l64 s7000 EMA vcot td_path | ~10h | g016 |
| 62847 | VCoT l64 s7000 noEMA vcot td_path_arrow | ~10h | g022 |
| 62848 | VCoT l64 s7000 EMA vcot td_path_arrow | ~10h | g022 |
| 62906 | VCoT l64 s6000 noEMA vcot td_path | ~7h | g019 |
| 62908 | VCoT l64 s6000 noEMA vcot td_path_arrow | ~7h | g002 |

## Resubmitted (19 jobs, GPU alloc fix: --gpus-per-node=2 --nodes=1)

### AO s2400 (10 jobs)

| Job | Variant | Subset | Time |
|-----|---------|--------|------|
| 62966 | EMA | td_path_arrow | 4h |
| 62967 | noEMA | td_path_arrow | 4h |
| 62968 | EMA | Perspective_Arrow | 4h |
| 62969 | noEMA | Perspective_Arrow | 4h |
| 62970 | EMA | SAT_perspective | 4h |
| 62971 | noEMA | SAT_perspective | 4h |
| 62972 | EMA | HabitatPerspective_Arrow | 4h |
| 62973 | noEMA | HabitatPerspective_Arrow | 4h |
| 62974 | EMA | HabitatPerspective_NoArrow | 4h |
| 62975 | noEMA | HabitatPerspective_NoArrow | 4h |

### AO s1500 EMA (1 job)

| Job | Variant | Subset | Time |
|-----|---------|--------|------|
| 62976 | EMA | Perspective_NoArrow | 4h |

### VCoT l64 s6000 EMA (2 jobs)

| Job | Variant | Subset | Time |
|-----|---------|--------|------|
| 62977 | EMA | td_path | 16h |
| 62978 | EMA | td_path_arrow | 16h |

### VCoT l64 s7000 EMA (1 job)

| Job | Variant | Subset | Time |
|-----|---------|--------|------|
| 62979 | EMA | dh_midpoint | 16h |

### MMCoT s6000 (3 jobs)

| Job | Variant | Subset | Time |
|-----|---------|--------|------|
| 62980 | EMA | Perspective_Arrow | 16h |
| 62981 | noEMA | Perspective_Arrow | 16h |
| 62982 | noEMA | SAT_perspective | 16h |

### TextCoT s1500 EMA (2 jobs)

| Job | Variant | Subset | Time |
|-----|---------|--------|------|
| 62983 | EMA | Perspective_Arrow | 4h |
| 62984 | EMA | Perspective_NoArrow | 4h |

## Known Issue: HabitatPerspective NCCL Crash

All previous HabitatPerspective evals (62886–62891) failed with NCCL watchdog timeout / SIGABRT after ~1h. Root cause unknown. The AO s2400 HabitatPerspective jobs above (62972–62975) will test whether the `--gpus-per-node=2` fix resolves this.

## Completed Successfully

### Conversions

| Job | Name |
|-----|------|
| 62898 | conv_ao_s2400_ema |
| 62899 | conv_ao_s2400_noema |
| 62903 | conv_vcot_l64_s6000_ema |
| 62904 | conv_vcot_l64_s6000_noema |

### Evals

| Job | Name | Result |
|-----|------|--------|
| 62541 | MMCoT s6000 noEMA vcot td_path | 56.77% |
| 62542 | MMCoT s6000 noEMA vcot td_path_arrow | 51.50% |
| 62543 | MMCoT s6000 EMA vcot td_path | 44.36% |
| 62544 | MMCoT s6000 EMA vcot td_path_arrow | 42.15% |
| 62809 | MMCoT s7000 EMA vcot dh_midpoint | 67.90% |
| 62810 | MMCoT s7000 noEMA vcot dh_midpoint | 55.56% |
| 62864 | AO s1500 EMA Perspective Arrow | 100.00%* |
| 62867 | TextCoT s2000 EMA Perspective Arrow | 100.00%* |
| 62868 | TextCoT s2000 EMA Perspective NoArrow | 100.00%* |
| 62869 | TextCoT s2000 noEMA Perspective Arrow | 99.71% |
| 62870 | TextCoT s2000 noEMA Perspective NoArrow | 99.46% |
| 62871 | AO s6000 noEMA SAT_perspective | 40.91% |
| 62872 | TextCoT s1500 noEMA SAT_perspective | 59.09% |
| 62874 | Baseline SAT_perspective | 22.73% |
| 62892 | TextCoT s2000 EMA td_path | 35.71% |
| 62893 | TextCoT s2000 EMA td_path_arrow | 31.92% |
| 62894 | AO s3000 EMA Perspective Arrow | 16.78% |
| 62895 | AO s3000 EMA Perspective NoArrow | 21.08% |
| 62896 | AO s4500 EMA Perspective NoArrow | 38.69% |
| 62900 | AO s2400 EMA td_path | 34.96% |
| 62901 | AO s2400 noEMA td_path | 74.06% |
| 62909 | VCoT l64 s6000 EMA vcot dh_midpoint | 18.52% |
| 62910 | VCoT l64 s6000 noEMA vcot dh_midpoint | 53.09% |
| 62911 | AO s2400 EMA dh_midpoint | 50.00% |
| 62912 | AO s2400 noEMA dh_midpoint | 74.69% |
| 62921 | MMCoT s6000 EMA SAT_perspective | 57.58% |

*100% results are suspicious — likely eval artifacts.

### Training

| Job | Name | Status |
|-----|------|--------|
| 62902 | TextCoT 3k (resumes from s2000) | COMPLETED (ckpts 2200–3000) |

## Previously Failed (Resolved)

| Job | Name | Failure | Resubmitted as |
|-----|------|---------|----------------|
| 62775–62776 | AO s1500 EMA Perspective | NCCL | → 62864 (done) |
| 62779–62780 | AO s3000 EMA Perspective | TIMEOUT 2h | → 62894–62895 (done) |
| 62784 | AO s4500 EMA Perspective NoArrow | TIMEOUT 2h | → 62896 (done) |
| 62799–62802 | TextCoT s2000 Perspective | GPU alloc | → 62867–62868 (done) |
| 62876–62881 | HabitatPerspective (all) | port conflict | → 62886–62891 (NCCL fail) |
| 62886–62891 | HabitatPerspective (all) | NCCL timeout | investigating |
| 62905,62907 | VCoT l64 s6000 EMA td_path/arrow | GPU alloc | → 62977–62978 |
| 62913–62918 | AO s2400 (6 subsets) | GPU alloc | → 62966–62971 |
| 62919–62920 | MMCoT s6000 Perspective Arrow | GPU alloc | → 62980–62981 |
| 62922 | MMCoT s6000 noEMA SAT_perspective | GPU alloc | → 62982 |
| 62923–62926 | AO s2400 HabitatPerspective (4) | GPU alloc | → 62972–62975 |
| 62928 | VCoT l64 s7000 EMA dh_midpoint | GPU alloc | → 62979 |
| 62929 | AO s1500 EMA Perspective NoArrow | GPU alloc | → 62976 |
| 62930–62931 | TextCoT s1500 EMA Perspective | GPU alloc | → 62983–62984 |
