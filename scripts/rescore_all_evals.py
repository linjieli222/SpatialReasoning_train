#!/usr/bin/env python3
"""Re-score all eval results with corrected accuracy computation.

The old scoring used naive substring matching:
    hit = 1 if gt in pred_text else 0
This incorrectly matches when GT letter appears anywhere in <think> reasoning.

The correct scoring extracts the answer from <answer> tags first, then compares letters.
"""
import os
import re
import sys
import pandas as pd
from pathlib import Path


def extract_answer_letter(pred_text):
    """Extract answer letter from model prediction.

    Handles all observed prediction formats:
    - <answer>X</answer> or <answer>X. description</answer>
    - </think><answer>X</answer>
    - <image_end><answer>X</answer>
    - "The answer is (X)" or "The answer is X. description"
    - "**Answer:** X. description"
    - "Answer: X. description" or "ANSWER: X"
    - Bare letter after </think>
    - "X. description" after </think>
    - Single letter prediction
    """
    pred = str(pred_text).strip()

    # 1. <answer>X</answer> tags — most reliable, check first
    #    Handles: <answer>B</answer>, <answer>(B)</answer>, <answer>B. the vase</answer>
    match = re.search(r'<answer>\s*\(?([A-Da-d])\)?[\s.,)]*(?:[^<]*)</answer>', pred)
    if match:
        return match.group(1).upper()

    # 2. After </think> or <image_end>, look for answer patterns in the suffix
    #    This catches models that don't use <answer> tags
    suffix = pred
    for marker in ['</think>', '<image_end>']:
        idx = pred.rfind(marker)
        if idx >= 0:
            suffix = pred[idx + len(marker):]
            break

    # 3. "The answer is (X)" or "The answer is X" patterns
    match = re.search(r'(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?', suffix, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 4. "**Answer:** X" or "Answer: X" patterns
    match = re.search(r'\*{0,2}answer\*{0,2}\s*[:]\s*\(?([A-Da-d])\)?', suffix, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 5. "ANSWER: X" (all caps variant)
    match = re.search(r'ANSWER\s*:\s*\(?([A-Da-d])\)?', suffix)
    if match:
        return match.group(1).upper()

    # 6. Bare letter or "X. description" at start of suffix (after </think>)
    suffix_stripped = suffix.strip()
    match = re.match(r'^\(?([A-Da-d])\)?[\s.,)]', suffix_stripped)
    if match:
        return match.group(1).upper()

    # 7. Single letter (entire suffix is just a letter)
    if suffix_stripped.upper() in ('A', 'B', 'C', 'D'):
        return suffix_stripped.upper()

    # 8. If no </think> marker was found, try the same patterns on the full text
    #    (for models that don't use <think> tags at all)
    if suffix is pred:
        # "The answer is (X)" anywhere in full text
        match = re.search(r'(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?', pred, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # "Answer: X" anywhere
        match = re.search(r'answer\s*[:]\s*\(?([A-Da-d])\)?', pred, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Entire prediction is a single letter
        if pred.upper() in ('A', 'B', 'C', 'D'):
            return pred.upper()

    return None


def extract_answer_text(pred_text):
    """Extract text answer from <answer> tags (for non-MCQ like SAT)."""
    pred = str(pred_text).strip()
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', pred, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def score_sample(pred_text, gt, item):
    """Score a single prediction against ground truth."""
    gt_clean = str(gt).strip().upper()

    # If GT is a letter (MCQ), extract letter and compare
    if gt_clean in ('A', 'B', 'C', 'D'):
        pred_letter = extract_answer_letter(pred_text)
        if pred_letter is None:
            return 0
        return 1 if pred_letter == gt_clean else 0

    # If GT is text (e.g., "yes", "left", "top left" for SAT),
    # first try to extract a letter and resolve to option text
    pred_letter = extract_answer_letter(pred_text)
    if pred_letter is not None:
        pred_answer = str(item.get(pred_letter, '')).strip().upper()
        if pred_answer == gt_clean:
            return 1

    # Also try direct text match from <answer> tags
    answer_text = extract_answer_text(pred_text)
    if answer_text is not None:
        if answer_text.strip().upper() == gt_clean:
            return 1

    return 0


def rescore_file(eval_xlsx_path):
    """Re-score a single eval xlsx file. Returns (accuracy, count, n_extracted) or None."""
    try:
        df = pd.read_excel(eval_xlsx_path)
    except Exception as e:
        return None

    if 'prediction' not in df.columns or 'answer' not in df.columns:
        return None

    hits = []
    n_extracted = 0
    for i in range(len(df)):
        row = df.iloc[i]
        pred = str(row.get('prediction', ''))
        gt = str(row.get('answer', ''))
        gt_clean = str(gt).strip().upper()
        # Track extraction rate for MCQ
        if gt_clean in ('A', 'B', 'C', 'D'):
            if extract_answer_letter(pred) is not None:
                n_extracted += 1
        else:
            # For text GT, count as extracted if we got something
            if extract_answer_letter(pred) is not None or extract_answer_text(pred) is not None:
                n_extracted += 1
        hit = score_sample(pred, gt, row)
        hits.append(hit)

    if len(hits) == 0:
        return None

    acc = sum(hits) / len(hits) * 100

    # For Perspective datasets with categories, compute unweighted avg
    if 'category' in df.columns:
        import numpy as np
        cat_accs = {}
        for cat in df['category'].unique():
            mask = df['category'] == cat
            cat_hits = [hits[i] for i in range(len(df)) if mask.iloc[i]]
            if len(cat_hits) > 0:
                cat_accs[cat] = sum(cat_hits) / len(cat_hits) * 100
        if cat_accs:
            acc = np.mean(list(cat_accs.values()))

    return acc, len(hits), n_extracted


def parse_eval_path(path):
    """Extract model, checkpoint, variant, subset from eval file path."""
    parts = str(path).split('/')
    # Pattern: .../tifa_v3_<setting>/<run_name>/<step>_full_<variant>/eval/<config>/T.../file.xlsx
    result = {}

    # Find the setting
    for p in parts:
        if 'answer_only' in p:
            result['model'] = 'AO'
        elif 'text_cot' in p:
            result['model'] = 'TextCoT'
        elif 'mmcot' in p:
            result['model'] = 'MMCoT'
        elif 'cot_l64' in p:
            result['model'] = 'VCoT_l64'
        elif 'cot_l32' in p:
            result['model'] = 'VCoT_l32'
        elif 'cot_l16' in p:
            result['model'] = 'VCoT_l16'
        elif 'dh_midpoint_vcot' in p:
            result['model'] = 'VCoT_debug'

    # Find step and variant
    for p in parts:
        m = re.match(r'(\d+)_full_(ema|noema)$', p)
        if m:
            result['step'] = int(m.group(1))
            result['variant'] = m.group(2)
        m = re.match(r'(\d+)_full$', p)
        if m:
            result['step'] = int(m.group(1))
            result['variant'] = 'full'

    # Detect eval config (bagel_mot vs bagel_mot_vcot)
    if '/bagel_mot_vcot/' in path:
        result['eval_config'] = 'vcot'
    else:
        result['eval_config'] = 'mot'

    # Find subset from filename
    fname = os.path.basename(path)
    # Remove prefix and suffix
    fname = fname.replace('bagel_mot_vcot_', '').replace('bagel_mot_', '')
    fname = fname.replace('.xlsx', '')
    result['subset'] = fname

    return result


def main():
    base_dir = '/gpfs/scrubbed/krishna/linjli/bagel_debug_output'

    # Find all raw eval xlsx files (not _result)
    # Deduplicate: prefer files NOT in T2026..._G.../  subdirectories
    eval_files_raw = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.xlsx') and '_result' not in f and '/eval/' in root:
                eval_files_raw.append(os.path.join(root, f))

    # Dedup: skip files inside T20*_G*/ subdirs if a copy exists at the parent level
    eval_files = []
    seen = set()
    for path in sorted(eval_files_raw):
        # Skip T-timestamped subdirectory copies
        if re.search(r'/T\d{8}_G[0-9a-f]+/', path):
            continue
        # Deduplicate by (model_dir, step_dir, subset_dir, filename)
        key = os.path.basename(path)
        parts = path.split('/')
        # Find the step dir and subset dir
        step_dir = ''
        subset_dir = ''
        for i, p in enumerate(parts):
            if re.match(r'\d+_full', p):
                step_dir = p
            if p in ('bagel_mot', 'bagel_mot_vcot'):
                subset_dir = parts[i-1] if i > 0 else ''
        dedup_key = (step_dir, subset_dir, key)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        eval_files.append(path)

    eval_files.sort()
    print(f"Found {len(eval_files)} eval files (after dedup)\n")

    results = []
    for path in eval_files:
        info = parse_eval_path(path)
        if 'model' not in info or 'step' not in info:
            continue

        scored = rescore_file(path)
        if scored is None:
            continue

        acc, count, n_extracted = scored
        info['accuracy'] = acc
        info['count'] = count
        info['extracted'] = n_extracted
        info['path'] = path
        results.append(info)

    # Print results grouped by model
    print(f"{'Model':<12} {'Step':>6} {'Variant':<7} {'Cfg':<5} {'Subset':<35} {'Acc':>7} {'N':>5} {'Ext':>5}")
    print('-' * 90)

    results.sort(key=lambda x: (x.get('model', ''), x.get('step', 0), x.get('variant', ''), x.get('eval_config', ''), x.get('subset', '')))
    for r in results:
        ext_pct = r['extracted'] / r['count'] * 100 if r['count'] > 0 else 0
        flag = ' !' if ext_pct < 80 else ''
        cfg = r.get('eval_config', '?')
        print(f"{r.get('model','?'):<12} {r.get('step',0):>6} {r.get('variant','?'):<7} {cfg:<5} {r['subset']:<35} {r['accuracy']:>6.2f}% {r['count']:>5} {ext_pct:>4.0f}%{flag}")


if __name__ == '__main__':
    main()
