# Pending Eval & Training Jobs

Last updated: 2026-02-24

## Running

| Job | Name | Elapsed | Node |
|-----|------|---------|------|
| 62541 | MMCoT s6000 noEMA vcot td_path | ~12h | g021 |
| 62542 | MMCoT s6000 noEMA vcot td_path_arrow | ~12h | g006 |
| 62543 | MMCoT s6000 EMA vcot td_path | ~12h | g016 |
| 62544 | MMCoT s6000 EMA vcot td_path_arrow | ~12h | g016 |
| 62805 | MMCoT s7000 EMA vcot td_path | ~3.5h | g005 |
| 62806 | MMCoT s7000 noEMA vcot td_path | ~3.5h | g005 |
| 62807 | MMCoT s7000 EMA vcot td_path_arrow | ~3.5h | g024 |
| 62808 | MMCoT s7000 noEMA vcot td_path_arrow | ~3.5h | g010 |
| 62809 | MMCoT s7000 EMA vcot dh_midpoint | ~3.5h | g010 |
| 62810 | MMCoT s7000 noEMA vcot dh_midpoint | ~3.5h | g016 |
| 62845 | VCoT l64 s7000 noEMA vcot td_path | ~2.8h | g005 |
| 62846 | VCoT l64 s7000 EMA vcot td_path | ~2.8h | g016 |
| 62847 | VCoT l64 s7000 noEMA vcot td_path_arrow | ~2.8h | g022 |
| 62848 | VCoT l64 s7000 EMA vcot td_path_arrow | ~2.8h | g022 |
| 62864 | AO s1500 EMA Perspective Arrow | ~1.7h | g004 |
| 62867 | TextCoT s2000 EMA Perspective Arrow | ~1.7h | g019 |
| 62868 | TextCoT s2000 EMA Perspective NoArrow | ~1.7h | g022 |
| 62886 | AO s6000 noEMA HabitatPerspective Arrow | ~40min | g019 |
| 62887 | AO s6000 noEMA HabitatPerspective NoArrow | ~40min | g021 |
| 62888 | TextCoT s1500 noEMA HabitatPerspective Arrow | ~40min | g021 |
| 62889 | TextCoT s1500 noEMA HabitatPerspective NoArrow | ~36min | g006 |
| 62890 | Baseline HabitatPerspective Arrow | ~29min | g002 |
| 62891 | Baseline HabitatPerspective NoArrow | ~29min | g005 |
| 62892 | TextCoT s2000 EMA td_path (4h resubmit) | ~29min | g021 |

## Pending (Priority/Resources)

| Job | Name | Reason |
|-----|------|--------|
| 62893 | TextCoT s2000 EMA td_path_arrow (4h resubmit) | Resources |
| 62894 | AO s3000 EMA Perspective Arrow (4h resubmit) | Priority |
| 62895 | AO s3000 EMA Perspective NoArrow (4h resubmit) | Priority |
| 62896 | AO s4500 EMA Perspective NoArrow (4h resubmit) | Priority |
| 62898 | conv_ao_s2400_ema | Priority |
| 62899 | conv_ao_s2400_noema | Priority |
| 62902 | TextCoT 3k training (resumes from s2000) | Priority |
| 62903 | conv_vcot_l64_s6000_ema | Priority |
| 62904 | conv_vcot_l64_s6000_noema | Priority |
| 62919 | MMCoT s6000 EMA Perspective Arrow | Priority |
| 62920 | MMCoT s6000 noEMA Perspective Arrow | Priority |
| 62921 | MMCoT s6000 EMA SAT_perspective | Priority |
| 62922 | MMCoT s6000 noEMA SAT_perspective | Priority |

## Pending (Dependency)

### AO s2400 evals (waiting on convert 62898/62899)

| Job | Variant | Subset |
|-----|---------|--------|
| 62900 | EMA | td_path |
| 62901 | noEMA | td_path |
| 62911 | EMA | dh_midpoint |
| 62912 | noEMA | dh_midpoint |
| 62913 | EMA | td_path_arrow |
| 62914 | noEMA | td_path_arrow |
| 62915 | EMA | Perspective_Arrow |
| 62916 | noEMA | Perspective_Arrow |
| 62917 | EMA | SAT_perspective |
| 62918 | noEMA | SAT_perspective |
| 62923 | EMA | HabitatPerspective_Arrow |
| 62924 | noEMA | HabitatPerspective_Arrow |
| 62925 | EMA | HabitatPerspective_NoArrow |
| 62926 | noEMA | HabitatPerspective_NoArrow |

### VCoT l64 s6000 evals (waiting on convert 62903/62904)

| Job | Variant | Subset |
|-----|---------|--------|
| 62905 | EMA | td_path |
| 62906 | noEMA | td_path |
| 62907 | EMA | td_path_arrow |
| 62908 | noEMA | td_path_arrow |
| 62909 | EMA | dh_midpoint |
| 62910 | noEMA | dh_midpoint |

## Failed / Timed Out (Need Resubmission)

| Job | Name | Status | Notes |
|-----|------|--------|-------|
| 62849 | VCoT l64 s7000 EMA vcot dh_midpoint | FAILED | needs resubmit |
| 62860 | VCoT l64 s7000 EMA vcot dh_midpoint | FAILED | 2nd attempt, still failed |
| 62853 | AO s1500 EMA Perspective NoArrow | TIMEOUT | 2h limit — resubmitted? |
| 62854 | TextCoT s1500 EMA Perspective Arrow | TIMEOUT | 2h limit |
| 62855 | TextCoT s1500 EMA Perspective NoArrow | TIMEOUT | 2h limit |
| 62856 | TextCoT s2000 EMA Perspective Arrow | FAILED | GPU alloc issue |
| 62857 | TextCoT s2000 EMA Perspective NoArrow | FAILED | GPU alloc issue |

## Previously Failed (Already Resubmitted)

| Job | Name | Status | Resubmitted as |
|-----|------|--------|----------------|
| 62775–62776 | AO s1500 EMA Perspective | FAILED (NCCL) | 62852 → 62864 |
| 62779–62780 | AO s3000 EMA Perspective | TIMEOUT (2h) | 62894–62895 |
| 62784 | AO s4500 EMA Perspective NoArrow | TIMEOUT (2h) | 62896 |
| 62795–62796 | TextCoT s1500 EMA Perspective | FAILED (GPU alloc) | 62854–62855 |
| 62799–62802 | TextCoT s2000 Perspective | FAILED (GPU alloc) | 62856–62857 (still failing) |
| 62876–62881 | HabitatPerspective (all) | FAILED (port conflict) | 62886–62891 |

## Training Jobs

| Job | Name | Total Steps | Save Every | Status |
|-----|------|------------|------------|--------|
| 62902 | TextCoT 3k | 3000 | 200 | pending (resumes from step 2000) |
