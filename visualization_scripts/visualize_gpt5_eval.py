"""Visualize GPT-5 evaluation results across all 5 subsets as interactive HTML."""
import base64
import io
import os
import re
import sys

import pandas as pd
from datasets import load_dataset
from PIL import Image

HF_CACHE = '/gpfs/scrubbed/linjli/hf_cache'
os.environ['HF_HOME'] = HF_CACHE

PRED_BASE = '/gpfs/scrubbed/krishna/linjli/bagel_debug_output/azure_gpt_eval/AzureGPT/T20260304_G341a106a'

SUBSETS = {
    'AI2ThorPT2PV2_td_ego_dir': {
        'hf_repo': 'linjieli222/ai2thor_path_tracing_2point_tifa_filtered_val_v3',
        'hf_subset': 'td_ego_dir',
        'hf_split': 'train',
        'img_keys': ['topdown_image', 'ego_images'],
        'pred_file': f'{PRED_BASE}/AzureGPT_AI2ThorPT2PV2_td_ego_dir.xlsx',
    },
    'AI2ThorPT2PV2_td_path': {
        'hf_repo': 'linjieli222/ai2thor_path_tracing_2point_tifa_filtered_val_v3',
        'hf_subset': 'td_path',
        'hf_split': 'train',
        'img_keys': ['topdown_image'],
        'pred_file': f'{PRED_BASE}/AzureGPT_AI2ThorPT2PV2_td_path.xlsx',
    },
    'AI2ThorPT2PV2_td_path_arrow': {
        'hf_repo': 'linjieli222/ai2thor_path_tracing_2point_tifa_filtered_val_v3',
        'hf_subset': 'td_path_arrow',
        'hf_split': 'train',
        'img_keys': ['topdown_image'],
        'pred_file': f'{PRED_BASE}/AzureGPT_AI2ThorPT2PV2_td_path_arrow.xlsx',
    },
    'RealPT_td_path': {
        'hf_repo': 'linjieli222/real_indoor_path_tracing',
        'hf_subset': None,
        'hf_split': 'td_path',
        'img_keys': ['image'],
        'pred_file': f'{PRED_BASE}/AzureGPT_RealPT_td_path.xlsx',
    },
    'RealPT_td_path_arrow': {
        'hf_repo': 'linjieli222/real_indoor_path_tracing',
        'hf_subset': None,
        'hf_split': 'td_path_arrow',
        'img_keys': ['image'],
        'pred_file': f'{PRED_BASE}/AzureGPT_RealPT_td_path_arrow.xlsx',
    },
}


def pil_to_b64(img, max_size=400):
    # Resize to keep file size manageable
    if max(img.size) > max_size:
        img = img.copy()
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=75)
    return base64.b64encode(buf.getvalue()).decode()


def extract_pred_letter(pred):
    if pd.isna(pred):
        return None
    pred = str(pred).strip()
    m = re.match(r'^([A-D])[\.\s\)]', pred)
    if m:
        return m.group(1)
    if pred in ['A', 'B', 'C', 'D']:
        return pred
    # Try to find any A-D letter
    m2 = re.search(r'\b([A-D])\b', pred)
    if m2:
        return m2.group(1)
    return None


def load_subset(name, cfg):
    print(f'Loading {name}...')
    if cfg['hf_subset']:
        hf_ds = load_dataset(cfg['hf_repo'], cfg['hf_subset'], split=cfg['hf_split'])
    else:
        hf_ds = load_dataset(cfg['hf_repo'], split=cfg['hf_split'])

    pred_df = pd.read_excel(cfg['pred_file'])
    print(f'  HF samples: {len(hf_ds)}, Predictions: {len(pred_df)}')

    samples = []
    for idx in range(min(len(hf_ds), len(pred_df))):
        ex = hf_ds[idx]
        row = pred_df.iloc[idx]

        # Get images
        images_b64 = []
        for key in cfg['img_keys']:
            val = ex.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                for img in val:
                    images_b64.append(pil_to_b64(img))
            elif isinstance(val, Image.Image):
                images_b64.append(pil_to_b64(val))

        pred_letter = extract_pred_letter(row['prediction'])
        gt_letter = str(row['answer']).strip()
        correct = pred_letter == gt_letter

        samples.append({
            'index': idx,
            'images_b64': images_b64,
            'question': str(row['question']),
            'A': str(row['A']),
            'B': str(row['B']),
            'C': str(row['C']),
            'D': str(row['D']),
            'answer': gt_letter,
            'prediction': str(row['prediction']),
            'pred_letter': pred_letter,
            'correct': correct,
        })

    acc = sum(s['correct'] for s in samples) / len(samples) * 100
    print(f'  Accuracy: {acc:.2f}% ({sum(s["correct"] for s in samples)}/{len(samples)})')
    return samples, acc


def generate_html(all_data, output_path):
    """Generate a single HTML file with all 5 subsets."""

    subset_names = list(all_data.keys())

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GPT-5 Evaluation Results</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; padding: 20px; }
h1 { text-align: center; margin-bottom: 10px; color: #333; }
.summary-table { margin: 0 auto 20px; border-collapse: collapse; }
.summary-table th, .summary-table td { padding: 8px 16px; border: 1px solid #ddd; text-align: center; }
.summary-table th { background: #4a90d9; color: white; }
.summary-table tr:nth-child(even) { background: #f9f9f9; }
.tabs { display: flex; gap: 4px; margin-bottom: 15px; flex-wrap: wrap; justify-content: center; }
.tab-btn { padding: 8px 16px; border: 1px solid #ccc; background: #fff; cursor: pointer; border-radius: 4px 4px 0 0; font-size: 13px; }
.tab-btn.active { background: #4a90d9; color: white; border-color: #4a90d9; }
.tab-content { display: none; }
.tab-content.active { display: block; }
.filter-bar { margin-bottom: 15px; text-align: center; }
.filter-bar button { padding: 6px 14px; margin: 0 4px; border: 1px solid #ccc; background: #fff; cursor: pointer; border-radius: 4px; }
.filter-bar button.active { background: #4a90d9; color: white; }
.sample { background: white; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.sample.correct { border-left: 5px solid #4CAF50; }
.sample.wrong { border-left: 5px solid #f44336; }
.sample-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.sample-idx { font-weight: bold; color: #666; }
.badge { padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; color: white; }
.badge.correct { background: #4CAF50; }
.badge.wrong { background: #f44336; }
.images { display: flex; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
.images img { max-height: 280px; border-radius: 4px; border: 1px solid #eee; }
.question { font-size: 14px; margin-bottom: 10px; color: #333; line-height: 1.5; }
.choices { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 8px; }
.choice { padding: 8px 12px; border-radius: 4px; font-size: 13px; border: 1px solid #e0e0e0; }
.choice.gt { background: #e8f5e9; border-color: #4CAF50; font-weight: bold; }
.choice.pred-wrong { background: #ffebee; border-color: #f44336; }
.prediction-text { font-size: 12px; color: #888; margin-top: 6px; }
.counter { text-align: center; margin-bottom: 10px; color: #666; font-size: 14px; }
</style>
</head>
<body>
<h1>GPT-5 Evaluation Results</h1>
<table class="summary-table">
<tr><th>Subset</th><th>Accuracy</th><th>Correct</th><th>Total</th></tr>
"""
    for name in subset_names:
        samples, acc = all_data[name]
        n_correct = sum(s['correct'] for s in samples)
        html += f'<tr><td>{name}</td><td><b>{acc:.2f}%</b></td><td>{n_correct}</td><td>{len(samples)}</td></tr>\n'

    html += '</table>\n<div class="tabs">\n'
    for i, name in enumerate(subset_names):
        active = ' active' if i == 0 else ''
        _, acc = all_data[name]
        html += f'<button class="tab-btn{active}" onclick="switchTab(\'{name}\')">{name} ({acc:.1f}%)</button>\n'
    html += '</div>\n'

    for i, name in enumerate(subset_names):
        samples, acc = all_data[name]
        active = ' active' if i == 0 else ''
        n_correct = sum(s['correct'] for s in samples)
        n_wrong = len(samples) - n_correct

        html += f'<div class="tab-content{active}" id="tab-{name}">\n'
        html += f'<div class="filter-bar">'
        html += f'<button class="active" onclick="filterSamples(\'{name}\', \'all\', this)">All ({len(samples)})</button>'
        html += f'<button onclick="filterSamples(\'{name}\', \'wrong\', this)">Wrong ({n_wrong})</button>'
        html += f'<button onclick="filterSamples(\'{name}\', \'correct\', this)">Correct ({n_correct})</button>'
        html += f'</div>\n'
        html += f'<div class="counter" id="counter-{name}">Showing {len(samples)} samples</div>\n'

        for s in samples:
            cls = 'correct' if s['correct'] else 'wrong'
            badge_text = 'Correct' if s['correct'] else 'Wrong'

            html += f'<div class="sample {cls}" data-correct="{cls}">\n'
            html += f'<div class="sample-header"><span class="sample-idx">#{s["index"]}</span>'
            html += f'<span class="badge {cls}">{badge_text}</span></div>\n'

            # Images
            html += '<div class="images">\n'
            for j, img_b64 in enumerate(s['images_b64']):
                label = 'Top-down' if j == 0 else f'Ego view {j}'
                html += f'<div><img src="data:image/jpeg;base64,{img_b64}" title="{label}"><div style="text-align:center;font-size:11px;color:#888">{label}</div></div>\n'
            html += '</div>\n'

            # Question (clean up image tags)
            q_clean = re.sub(r'<image_\d+>', '[IMG]', s['question'])
            html += f'<div class="question"><b>Q:</b> {q_clean}</div>\n'

            # Choices
            html += '<div class="choices">\n'
            for letter in ['A', 'B', 'C', 'D']:
                choice_cls = ''
                if letter == s['answer']:
                    choice_cls = ' gt'
                if letter == s['pred_letter'] and not s['correct']:
                    choice_cls = ' pred-wrong'
                marker = ''
                if letter == s['answer']:
                    marker = ' [GT]'
                if letter == s['pred_letter'] and letter != s['answer']:
                    marker = ' [GPT-5]'
                html += f'<div class="choice{choice_cls}">{letter}. {s[letter]}{marker}</div>\n'
            html += '</div>\n'

            # Raw prediction
            pred_short = s['prediction'][:100]
            html += f'<div class="prediction-text">GPT-5 raw: {pred_short}</div>\n'
            html += '</div>\n'

        html += '</div>\n'

    html += """
<script>
function switchTab(name) {
    document.querySelectorAll('.tab-content').forEach(e => e.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(e => e.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    document.querySelectorAll('.tab-btn').forEach(e => {
        if (e.textContent.includes(name)) e.classList.add('active');
    });
}
function filterSamples(name, filter, btn) {
    var tab = document.getElementById('tab-' + name);
    var samples = tab.querySelectorAll('.sample');
    var shown = 0;
    samples.forEach(s => {
        if (filter === 'all' || s.dataset.correct === filter) {
            s.style.display = '';
            shown++;
        } else {
            s.style.display = 'none';
        }
    });
    tab.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('counter-' + name).textContent = 'Showing ' + shown + ' samples';
}
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'\nWritten to {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)')


def main():
    all_data = {}
    for name, cfg in SUBSETS.items():
        samples, acc = load_subset(name, cfg)
        all_data[name] = (samples, acc)

    out = '/gpfs/home/linjli/source/SpatialReasoning_train/visualization_scripts/viz_gpt5_eval.html'
    generate_html(all_data, out)


if __name__ == '__main__':
    main()
