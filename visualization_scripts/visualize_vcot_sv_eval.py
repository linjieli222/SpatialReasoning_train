"""
Visualize VCoT SV evaluation results as interactive HTML.
Shows input images (topdown + ego views), GT sideview, generated sideview thought,
prediction, ground truth, and correctness.
"""
import argparse
import base64
import re
import html as html_lib
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image


def image_to_data_uri(img, max_size=384):
    """Convert PIL Image to base64 data URI, resized for HTML."""
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def file_to_data_uri(path, max_size=384):
    """Load image file and convert to data URI."""
    img = Image.open(path).convert('RGB')
    return image_to_data_uri(img, max_size)


def extract_answer(pred):
    pred = str(pred)
    m = re.search(r'<answer>\s*([A-D])\s*</answer>', pred)
    if m:
        return m.group(1)
    m = re.search(r'\b([A-D])\b', pred)
    if m:
        return m.group(1)
    return '?'


def extract_gen_image_path(pred):
    """Extract generated image path from prediction string."""
    m = re.search(r'\[Image:\s*(.+?)\]', str(pred))
    return m.group(1).strip() if m else None


def extract_think_text(pred):
    """Extract think text from prediction."""
    m = re.search(r'<think>(.*?)</think>', str(pred))
    return m.group(1).strip() if m else ''


def build_html(result_xlsx, dataset, output_path, max_samples=None):
    # Load eval results
    df = pd.read_excel(result_xlsx)
    if 'hit' not in df.columns:
        df['pred_letter'] = df['prediction'].apply(extract_answer)
        df['hit'] = (df['pred_letter'] == df['answer']).astype(int)

    # Load source dataset for input images
    from datasets import load_dataset
    ds = load_dataset(
        'linjieli222/ai2thor_path_tracing_2point_tifa_filtered_val_v2',
        dataset.replace('AI2ThorSV_', ''),
        split='train'
    )

    # Build question_id lookup if available
    ds_lookup = {}
    for i, item in enumerate(ds):
        ds_lookup[i] = item

    n_total = len(df)
    n_correct = int(df['hit'].sum())
    accuracy = n_correct / n_total * 100 if n_total else 0

    # Compute F1
    if 'category' in df.columns:
        tp = len(df[(df['category'] == 'positive') & (df['hit'] == 1)])
        fp = len(df[(df['category'] == 'negative') & (df['hit'] == 0)])
        fn = len(df[(df['category'] == 'positive') & (df['hit'] == 0)])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_str = f"F1: {f1*100:.1f}% | Precision: {precision*100:.1f}% | Recall: {recall*100:.1f}%"
    else:
        f1_str = ""

    if max_samples:
        df = df.head(max_samples)

    samples_html = []
    for idx, row in df.iterrows():
        pred_letter = row.get('pred_letter', extract_answer(row['prediction']))
        gt_letter = row['answer']
        is_correct = row['hit'] == 1
        category = row.get('category', '')
        think_text = extract_think_text(row['prediction'])
        gen_img_path = extract_gen_image_path(row['prediction'])

        # Status
        status_class = 'correct' if is_correct else 'wrong'
        status_emoji = '&#10004;' if is_correct else '&#10008;'

        # Input images from dataset
        src_item = ds_lookup.get(row['index'], None)
        input_imgs_html = ''
        gt_sv_html = ''
        if src_item:
            # Topdown
            td_uri = image_to_data_uri(src_item['topdown_image'])
            input_imgs_html += f'<div class="img-box"><div class="img-label">Top-down</div><img src="{td_uri}"></div>'
            # Ego views
            for i, ego_img in enumerate(src_item.get('ego_images', [])):
                ego_uri = image_to_data_uri(ego_img)
                input_imgs_html += f'<div class="img-box"><div class="img-label">Ego {i+1}</div><img src="{ego_uri}"></div>'
            # GT sideview
            if src_item.get('sideview_image'):
                gt_sv_uri = image_to_data_uri(src_item['sideview_image'])
                gt_sv_html = f'<div class="img-box"><div class="img-label">GT Sideview</div><img src="{gt_sv_uri}"></div>'

        # Generated sideview image
        gen_sv_html = ''
        if gen_img_path and Path(gen_img_path).exists():
            gen_sv_uri = file_to_data_uri(gen_img_path)
            gen_sv_html = f'<div class="img-box"><div class="img-label">Generated Thought</div><img src="{gen_sv_uri}"></div>'

        question_text = html_lib.escape(str(row['question'])[:500])
        choice_a = html_lib.escape(str(row['A']))
        choice_b = html_lib.escape(str(row['B']))

        sample_html = f"""
        <div class="sample {status_class}">
            <div class="sample-header" onclick="this.parentElement.classList.toggle('expanded')">
                <span class="status">{status_emoji}</span>
                <span class="sample-id">#{row['index']}</span>
                <span class="category">[{category}]</span>
                <span class="pred-info">Pred: <b>{pred_letter}</b> | GT: <b>{gt_letter}</b></span>
                <span class="think-preview">{html_lib.escape(think_text[:80])}</span>
                <span class="expand-icon">&#9660;</span>
            </div>
            <div class="sample-body">
                <div class="question"><b>Question:</b> {question_text}</div>
                <div class="choices">
                    <span class="choice {'choice-gt' if gt_letter=='A' else ''} {'choice-pred' if pred_letter=='A' else ''}">A: {choice_a}</span>
                    <span class="choice {'choice-gt' if gt_letter=='B' else ''} {'choice-pred' if pred_letter=='B' else ''}">B: {choice_b}</span>
                </div>
                <div class="images-section">
                    <div class="img-row">
                        <div class="img-group"><div class="group-label">Input Images</div><div class="img-row">{input_imgs_html}</div></div>
                    </div>
                    <div class="img-row">
                        <div class="img-group"><div class="group-label">Sideview Comparison</div><div class="img-row">{gt_sv_html}{gen_sv_html}</div></div>
                    </div>
                </div>
            </div>
        </div>
        """
        samples_html.append(sample_html)

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>VCoT SV Eval — {Path(result_xlsx).parent.parent.parent.name}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ font-size: 1.4em; }}
.stats {{ background: #fff; padding: 12px 18px; border-radius: 8px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.stats b {{ color: #333; }}
.filter-bar {{ margin-bottom: 12px; }}
.filter-bar button {{ padding: 6px 14px; margin-right: 6px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; background: #fff; }}
.filter-bar button.active {{ background: #4a90d9; color: white; border-color: #4a90d9; }}
.sample {{ background: #fff; margin-bottom: 8px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #ccc; overflow: hidden; }}
.sample.correct {{ border-left-color: #4caf50; }}
.sample.wrong {{ border-left-color: #f44336; }}
.sample-header {{ display: flex; align-items: center; padding: 10px 16px; cursor: pointer; gap: 10px; }}
.sample-header:hover {{ background: #f9f9f9; }}
.status {{ font-size: 1.2em; }}
.correct .status {{ color: #4caf50; }}
.wrong .status {{ color: #f44336; }}
.sample-id {{ font-weight: 600; color: #666; min-width: 40px; }}
.category {{ color: #888; font-size: 0.85em; min-width: 70px; }}
.pred-info {{ font-size: 0.9em; min-width: 120px; }}
.think-preview {{ color: #888; font-size: 0.85em; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.expand-icon {{ color: #aaa; transition: transform 0.2s; }}
.sample.expanded .expand-icon {{ transform: rotate(180deg); }}
.sample-body {{ display: none; padding: 12px 16px; border-top: 1px solid #eee; }}
.sample.expanded .sample-body {{ display: block; }}
.question {{ margin-bottom: 8px; line-height: 1.5; font-size: 0.9em; }}
.choices {{ margin-bottom: 12px; }}
.choice {{ display: inline-block; padding: 4px 12px; margin-right: 8px; border-radius: 4px; background: #f0f0f0; font-size: 0.9em; }}
.choice-gt {{ border: 2px solid #4caf50; font-weight: 600; }}
.choice-pred {{ background: #fff3cd; }}
.choice-gt.choice-pred {{ background: #c8e6c9; }}
.images-section {{ margin-top: 8px; }}
.img-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }}
.img-group {{ margin-bottom: 8px; }}
.group-label {{ font-weight: 600; font-size: 0.85em; color: #555; margin-bottom: 6px; }}
.img-box {{ text-align: center; }}
.img-box img {{ max-width: 280px; max-height: 280px; border-radius: 6px; border: 1px solid #ddd; }}
.img-label {{ font-size: 0.8em; color: #666; margin-bottom: 4px; }}
</style>
</head><body>
<h1>VCoT SV Evaluation — Best Model (s7k EMA, F1={f1*100:.1f}%)</h1>
<div class="stats">
    <b>Accuracy:</b> {accuracy:.1f}% ({n_correct}/{n_total}) |
    <b>{f1_str}</b> |
    <b>Showing:</b> {len(df)} samples
</div>
<div class="filter-bar">
    <button class="active" onclick="filterSamples('all', this)">All ({n_total})</button>
    <button onclick="filterSamples('correct', this)">Correct ({n_correct})</button>
    <button onclick="filterSamples('wrong', this)">Wrong ({n_total - n_correct})</button>
</div>
<div id="samples">
{''.join(samples_html)}
</div>
<script>
function filterSamples(filter, btn) {{
    document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.sample').forEach(s => {{
        if (filter === 'all') s.style.display = '';
        else if (filter === 'correct') s.style.display = s.classList.contains('correct') ? '' : 'none';
        else s.style.display = s.classList.contains('wrong') ? '' : 'none';
    }});
}}
</script>
</body></html>"""

    Path(output_path).write_text(full_html)
    print(f"Written {output_path} ({Path(output_path).stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_xlsx', required=True)
    parser.add_argument('--dataset', default='AI2ThorSV_td_ego_dir')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    build_html(args.result_xlsx, args.dataset, args.output, args.max_samples)
