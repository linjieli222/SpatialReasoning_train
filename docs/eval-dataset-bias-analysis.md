# AI2ThorPT2P Eval Dataset Bias Analysis

Date: 2026-02-19
Dataset: `AI2ThorPT2P_td_path` (532 samples, 4-choice MCQ)

## Summary

The distractor design has significant biases that allow a model to score well above chance (25%) without spatial reasoning.

## 1. Distractor-Only Objects

64 of 114 unique objects (56%) appear **only as distractors**, never as correct answers.

Top distractor-only objects:
| Object | Distractor count |
|--------|-----------------|
| light switch | 45 |
| plate | 25 |
| credit card | 23 |
| plunger | 21 |
| statue | 20 |
| basketball | 19 |
| key chain | 16 |
| mug | 15 |
| tissue box | 15 |
| cell phone | 15 |

- 62.6% of questions have at least one eliminable option
- Random-guess accuracy after elimination: **37.5%** (vs 25% baseline)
- Only 3 objects appear exclusively as correct answers (washing machine, vacuum cleaner, curtains)

## 2. Object Size Bias

Correct answers strongly favor large furniture; distractors include many small objects.

|  | Large furniture | Small objects | Other |
|--|----------------|--------------|-------|
| Correct answer | 65% | 2% | 32% |
| Distractors | 33% | 21% | 46% |

A model can learn "small object = probably wrong" as a shortcut.

## 3. Per-Object GT Rate

Some objects are disproportionately correct or incorrect:

| Object | GT count | Distractor count | GT rate |
|--------|---------|------------------|---------|
| cabinet | 19 | 12 | 61% |
| side table | 28 | 20 | 58% |
| dining table | 44 | 44 | 50% |
| garbage can | 15 | 73 | 17% |
| television | 11 | 52 | 17% |
| book | 8 | 41 | 16% |
| laptop | 11 | 43 | 20% |

## 4. Other Checks (No Bias Found)

- **Article usage**: 50/50 split between "the" and "a" in correct vs distractors
- **Answer length**: Correct avg 10.4 chars, distractor avg 10.7 chars (negligible)
- **Answer position (A/B/C/D)**: Roughly balanced (140/136/141/115), D slightly underrepresented

## Implications

- Reported accuracy on td_path may be inflated by ~12.5pp from object-prior shortcuts alone
- Cross-subset generalization (e.g., td_path model on dh_midpoint) is a better signal since the model can't memorize subset-specific distractor distributions
- Future eval should use balanced distractor sampling or text-only ablations to isolate spatial reasoning from object priors
