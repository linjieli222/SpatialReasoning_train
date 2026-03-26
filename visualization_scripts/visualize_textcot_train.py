"""
Visualize TextCoT training examples as interactive HTML.
Shows input images (topdown + ego views), question, text chain-of-thought
reasoning, and ground-truth answer.
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
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def bytes_to_data_uri(img_bytes, max_size=384):
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    return image_to_data_uri(img, max_size)


def parse_think_and_answer(output_text):
    think = ''
    answer = '?'
    m = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
    if m:
        think = m.group(1).strip()
    m = re.search(r'<answer>\s*([A-D])\s*</answer>', output_text)
    if m:
        answer = m.group(1)
    return think, answer


def parse_question_and_choices(instruction):
    lines = instruction.strip().split('\n')
    question_lines = []
    choices = {}
    for line in lines:
        m = re.match(r'^\(([A-D])\)\s*(.+)', line.strip())
        if m:
            choices[m.group(1)] = m.group(2).strip()
        else:
            question_lines.append(line)
    question = '\n'.join(question_lines).strip()
    # Remove image tags for display
    question = re.sub(r'<img><\|image_\d+\|></img>', '[IMAGE]', question)
    return question, choices


def build_html(data_dir, output_path, max_samples=None):
    data_dir = Path(data_dir)
    parquet_files = sorted(data_dir.glob('chunk_*.parquet'))
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    n_total = len(df)
    if max_samples:
        df = df.head(max_samples)

    samples_html = []
    for idx, row in df.iterrows():
        instructions = row['instruction_list']
        outputs = row['output_text_list']
        images = row['image_list']
        num_input = row['num_input_images']

        # Parse question (instruction[1]) and CoT output
        question_text, choices = parse_question_and_choices(instructions[1])
        think_text, gt_answer = parse_think_and_answer(outputs[0])

        # Input images
        input_imgs_html = ''
        if num_input >= 1:
            td_uri = bytes_to_data_uri(images[0])
            input_imgs_html += f'<div class="img-box"><div class="img-label">Top-down</div><img src="{td_uri}"></div>'
        for i in range(1, num_input):
            ego_uri = bytes_to_data_uri(images[i])
            input_imgs_html += f'<div class="img-box"><div class="img-label">Ego {i}</div><img src="{ego_uri}"></div>'

        question_short = html_lib.escape(question_text[:400])

        # Build choices HTML
        choices_html = ''
        for letter in ['A', 'B', 'C', 'D']:
            if letter in choices:
                cls = 'choice-gt' if letter == gt_answer else ''
                choice_text = html_lib.escape(choices[letter])
                choices_html += f'<span class="choice {cls}">{letter}: {choice_text}</span>'

        # Think text for preview and expanded view
        think_preview = html_lib.escape(think_text[:100])
        think_full = html_lib.escape(think_text)

        sample_html = f"""
        <div class="sample">
            <div class="sample-header" onclick="this.parentElement.classList.toggle('expanded')">
                <span class="sample-id">#{idx}</span>
                <span class="pred-info">GT: <b>{gt_answer}</b></span>
                <span class="think-preview">{think_preview}</span>
                <span class="expand-icon">&#9660;</span>
            </div>
            <div class="sample-body">
                <div class="question"><b>Question:</b> {question_short}</div>
                <div class="choices">{choices_html}</div>
                <div class="images-section">
                    <div class="img-row">
                        <div class="img-group"><div class="group-label">Input Images</div><div class="img-row">{input_imgs_html}</div></div>
                    </div>
                </div>
                <div class="cot-section">
                    <div class="group-label">Text Chain-of-Thought</div>
                    <div class="cot-text">{think_full}</div>
                </div>
            </div>
        </div>
        """
        samples_html.append(sample_html)

    dataset_name = data_dir.name

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>TextCoT Training Data — {dataset_name}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ font-size: 1.4em; }}
.stats {{ background: #fff; padding: 12px 18px; border-radius: 8px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.sample {{ background: #fff; margin-bottom: 8px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #4a90d9; overflow: hidden; }}
.sample-header {{ display: flex; align-items: center; padding: 10px 16px; cursor: pointer; gap: 10px; }}
.sample-header:hover {{ background: #f9f9f9; }}
.sample-id {{ font-weight: 600; color: #666; min-width: 50px; }}
.pred-info {{ font-size: 0.9em; min-width: 60px; }}
.think-preview {{ color: #888; font-size: 0.85em; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.expand-icon {{ color: #aaa; transition: transform 0.2s; }}
.sample.expanded .expand-icon {{ transform: rotate(180deg); }}
.sample-body {{ display: none; padding: 12px 16px; border-top: 1px solid #eee; }}
.sample.expanded .sample-body {{ display: block; }}
.question {{ margin-bottom: 8px; line-height: 1.5; font-size: 0.9em; }}
.choices {{ margin-bottom: 12px; }}
.choice {{ display: inline-block; padding: 4px 12px; margin-right: 8px; border-radius: 4px; background: #f0f0f0; font-size: 0.9em; }}
.choice-gt {{ border: 2px solid #4caf50; font-weight: 600; background: #c8e6c9; }}
.images-section {{ margin-top: 8px; }}
.img-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }}
.img-group {{ margin-bottom: 8px; }}
.group-label {{ font-weight: 600; font-size: 0.85em; color: #555; margin-bottom: 6px; }}
.img-box {{ text-align: center; }}
.img-box img {{ max-width: 280px; max-height: 280px; border-radius: 6px; border: 1px solid #ddd; }}
.img-label {{ font-size: 0.8em; color: #666; margin-bottom: 4px; }}
.cot-section {{ margin-top: 12px; }}
.cot-text {{ background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px; font-size: 0.88em; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; }}
</style>
</head><body>
<h1>TextCoT Training Data — {dataset_name}</h1>
<div class="stats">
    <b>Total samples:</b> {n_total} |
    <b>Showing:</b> {len(df)} samples
</div>
<div id="samples">
{''.join(samples_html)}
</div>
</body></html>"""

    Path(output_path).write_text(full_html)
    print(f"Written {output_path} ({Path(output_path).stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Path to parquet training data directory')
    parser.add_argument('--output', required=True,
                        help='Output HTML file path')
    parser.add_argument('--max_samples', type=int, default=50,
                        help='Max samples to visualize (default: 50)')
    args = parser.parse_args()
    build_html(args.data_dir, args.output, args.max_samples)
