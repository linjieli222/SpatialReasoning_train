#!/usr/bin/env python3
"""
Parse PT2P evaluation pkl files and compute accuracy for incomplete checkpoints.

This script:
1. Loads ground truth from a completed evaluation's xlsx file
2. For each target checkpoint, loads and merges pkl shard files (02_ and 12_)
3. Extracts predicted answers using the same logic as the official evaluation
4. Computes accuracy and reports results in a table
"""

import pickle
import re
import glob
import os
import zipfile
import xml.etree.ElementTree as ET
from collections import OrderedDict


BASE_DIR = (
    "/gpfs/scrubbed/krishna/linjli/bagel_debug_output/"
    "tifa_v3_td_ego_dir_vcot_l32/vcot_td_ego_dir_8gpu"
)

# Ground truth source: completed evaluation xlsx from s1k_ema
GT_XLSX = os.path.join(
    BASE_DIR,
    "0001000_full_ema/eval/bagel_mot_vcot/"
    "bagel_mot_vcot_AI2ThorPT2P_td_ego_dir.xlsx",
)

# Target checkpoints to evaluate
TARGET_CHECKPOINTS = [
    "0004000_full_noema",
    "0005000_full_ema",
    "0005000_full_noema",
    "0006000_full_ema",
    "0006000_full_noema",
    "0007000_full_ema",
    "0007000_full_noema",
]

# Also include completed checkpoints for reference (auto-detected below)
COMPLETED_CHECKPOINTS = []  # Will be populated dynamically


def parse_xlsx_ground_truth(xlsx_path):
    """Parse ground truth answers from xlsx file without openpyxl dependency.

    Returns dict {index: answer_letter} e.g. {0: 'B', 1: 'D', ...}
    """
    z = zipfile.ZipFile(xlsx_path)
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

    # Get shared strings
    ss = z.open("xl/sharedStrings.xml")
    ss_tree = ET.parse(ss)
    ss_root = ss_tree.getroot()
    strings = []
    for si in ss_root.findall(f"{{{ns}}}si"):
        texts = si.findall(f".//{{{ns}}}t")
        strings.append("".join(t.text or "" for t in texts))

    # Parse sheet
    sheet = z.open("xl/worksheets/sheet1.xml")
    tree = ET.parse(sheet)
    root = tree.getroot()
    rows = root.findall(f".//{{{ns}}}row")

    def col_letter_to_idx(col_str):
        result = 0
        for c in col_str:
            result = result * 26 + (ord(c) - ord("A") + 1)
        return result - 1

    gt = {}
    for i, row in enumerate(rows):
        if i == 0:
            continue  # skip header
        cells = row.findall(f"{{{ns}}}c")
        row_data = {}
        for cell in cells:
            ref = cell.get("r")
            col_str = re.match(r"([A-Z]+)", ref).group(1)
            col_idx = col_letter_to_idx(col_str)
            cell_type = cell.get("t")
            val_elem = cell.find(f"{{{ns}}}v")
            if val_elem is not None:
                if cell_type == "s":
                    row_data[col_idx] = strings[int(val_elem.text)]
                else:
                    row_data[col_idx] = val_elem.text
        try:
            idx = int(float(row_data.get(0, "")))
            answer = row_data.get(6, "").strip().upper()
            if answer in ("A", "B", "C", "D"):
                gt[idx] = answer
        except (ValueError, TypeError):
            pass

    return gt


def extract_answer(pred):
    """Extract answer letter from model prediction.

    Uses the exact same logic as AI2ThorPathTracing2Point._extract_answer
    in SpatialReasoning_Eval/vlmeval/dataset/ai2thor_spatial.py
    """
    pred = str(pred).strip()

    # Try to extract from <answer>...</answer> tags
    match = re.search(r"<answer>\s*([A-Da-d])\s*</answer>", pred)
    if match:
        return match.group(1).upper()

    # Try to extract from <answer>...(A)...</answer> or answer tag with text
    match = re.search(r"<answer>\s*\(?([A-Da-d])\)?\s*", pred)
    if match:
        return match.group(1).upper()

    # Fallback: exact single letter
    if pred.upper() in ("A", "B", "C", "D"):
        return pred.upper()

    # Fallback: first letter match in the prediction
    match = re.search(r"\b([A-D])\b", pred)
    if match:
        return match.group(1)

    return pred.strip().upper()


def load_and_merge_pkls(ckpt_dir):
    """Load and merge all pkl shard files for a checkpoint.

    Finds all T*/02_AI2ThorPT2P_td_ego_dir.pkl and T*/12_AI2ThorPT2P_td_ego_dir.pkl
    files, and merges them (later timestamps override earlier ones for duplicate keys).

    Returns merged dict {index: prediction_string}
    """
    merged = {}
    eval_dir = os.path.join(ckpt_dir, "eval", "bagel_mot_vcot")

    # Find all T* directories, sorted by name (timestamp order)
    t_dirs = sorted(glob.glob(os.path.join(eval_dir, "T*")))

    for t_dir in t_dirs:
        for shard_prefix in ["02", "12"]:
            pkl_path = os.path.join(
                t_dir, f"{shard_prefix}_AI2ThorPT2P_td_ego_dir.pkl"
            )
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                # Merge: later files override earlier for same keys
                merged.update(data)

    return merged


def compute_accuracy(predictions, ground_truth):
    """Compute accuracy comparing predictions to ground truth.

    Returns (accuracy_pct, correct_count, evaluated_count, total_gt_count)
    """
    correct = 0
    evaluated = 0

    for idx, pred_str in predictions.items():
        gt = ground_truth.get(int(idx))
        if gt is None:
            continue  # no ground truth for this index
        pred_answer = extract_answer(pred_str)
        if pred_answer == gt:
            correct += 1
        evaluated += 1

    accuracy = (correct / evaluated * 100) if evaluated > 0 else 0.0
    return accuracy, correct, evaluated, len(ground_truth)


def get_official_accuracy(ckpt_name):
    """Read official accuracy from completed eval acc.csv if available."""
    acc_path = os.path.join(
        BASE_DIR,
        ckpt_name,
        "eval",
        "bagel_mot_vcot",
        "bagel_mot_vcot_AI2ThorPT2P_td_ego_dir_acc.csv",
    )
    if os.path.exists(acc_path):
        with open(acc_path, "r") as f:
            lines = f.readlines()
        if len(lines) >= 2:
            # CSV: Category,Accuracy,Count
            parts = lines[1].strip().replace('"', "").split(",")
            return float(parts[1]), int(parts[2])
    return None, None


def main():
    print("=" * 80)
    print("PT2P (AI2ThorPT2P_td_ego_dir) Evaluation Results")
    print("=" * 80)
    print()

    # Load ground truth
    print(f"Loading ground truth from: {GT_XLSX}")
    gt = parse_xlsx_ground_truth(GT_XLSX)
    print(f"Ground truth: {len(gt)} samples (indices {min(gt.keys())}-{max(gt.keys())})")
    print()

    # Auto-detect completed evaluations
    all_ckpt_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "*_full_*")))
    completed = []
    for d in all_ckpt_dirs:
        ckpt_name = os.path.basename(d)
        acc, count = get_official_accuracy(ckpt_name)
        if acc is not None:
            completed.append((ckpt_name, acc, count))

    # Print completed evaluations for reference
    print("-" * 80)
    print("COMPLETED EVALUATIONS (official results for reference)")
    print("-" * 80)
    print(f"{'Checkpoint':<25} {'Accuracy':>10} {'Correct':>10} {'Count':>10}")
    print("-" * 80)
    for ckpt_name, acc, count in completed:
        correct = round(acc / 100 * count)
        print(f"{ckpt_name:<25} {acc:>9.2f}% {correct:>10} {count:>10}")
    print()

    # Now process incomplete checkpoints
    print("-" * 80)
    print("INCOMPLETE EVALUATIONS (computed from pkl files)")
    print("-" * 80)
    print(
        f"{'Checkpoint':<25} {'Accuracy':>10} {'Correct':>10} "
        f"{'Evaluated':>10} {'Total':>10} {'Coverage':>10}"
    )
    print("-" * 80)

    results = []
    for ckpt in TARGET_CHECKPOINTS:
        ckpt_dir = os.path.join(BASE_DIR, ckpt)
        if not os.path.exists(ckpt_dir):
            print(f"{ckpt:<25} {'N/A':>10} {'---':>10} {'---':>10} {'---':>10}")
            continue

        predictions = load_and_merge_pkls(ckpt_dir)
        if not predictions:
            print(
                f"{ckpt:<25} {'N/A':>10} {'---':>10} {'---':>10} "
                f"{'---':>10} {'---':>10}"
            )
            continue

        accuracy, correct, evaluated, total = compute_accuracy(predictions, gt)
        coverage = evaluated / total * 100 if total > 0 else 0
        results.append((ckpt, accuracy, correct, evaluated, total, coverage))
        print(
            f"{ckpt:<25} {accuracy:>9.2f}% {correct:>10} "
            f"{evaluated:>10} {total:>10} {coverage:>9.1f}%"
        )

    print("-" * 80)
    print()

    # Summary by step
    print("=" * 80)
    print("SUMMARY BY STEP")
    print("=" * 80)
    print(f"{'Step':<10} {'EMA Acc':>12} {'NoEMA Acc':>12} {'EMA Cov':>10} {'NoEMA Cov':>10}")
    print("-" * 80)

    steps = ["0001000", "0002000", "0003000", "0004000", "0005000", "0006000", "0007000"]
    for step in steps:
        ema_key = f"{step}_full_ema"
        noema_key = f"{step}_full_noema"

        # Get ema result
        ema_acc_str = "---"
        ema_cov_str = "---"
        official_ema, official_count = get_official_accuracy(ema_key)
        if official_ema is not None:
            ema_acc_str = f"{official_ema:.2f}%"
            ema_cov_str = f"{official_count}/{len(gt)}"
        else:
            for r in results:
                if r[0] == ema_key:
                    ema_acc_str = f"{r[1]:.2f}%"
                    ema_cov_str = f"{r[3]}/{r[4]}"

        # Get noema result
        noema_acc_str = "---"
        noema_cov_str = "---"
        official_noema, official_count_noema = get_official_accuracy(noema_key)
        if official_noema is not None:
            noema_acc_str = f"{official_noema:.2f}%"
            noema_cov_str = f"{official_count_noema}/{len(gt)}"
        else:
            for r in results:
                if r[0] == noema_key:
                    noema_acc_str = f"{r[1]:.2f}%"
                    noema_cov_str = f"{r[3]}/{r[4]}"

        print(
            f"{step:<10} {ema_acc_str:>12} {noema_acc_str:>12} "
            f"{ema_cov_str:>10} {noema_cov_str:>10}"
        )

    print("-" * 80)
    print()
    print("Notes:")
    print("- Accuracy is computed using the same extraction logic as the official eval")
    print("  (AI2ThorPathTracing2Point._extract_answer in ai2thor_spatial.py)")
    print("- Coverage shows how many of the 329 total samples have been evaluated")
    print("- PKL files from multiple T* directories and GPU shards (02_, 12_) are merged")
    print("- Completed evaluations use official acc.csv values (auto-detected)")


if __name__ == "__main__":
    main()
