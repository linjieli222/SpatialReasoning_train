#!/usr/bin/env python3
"""
Generate an HTML visualization of textcot training datasets (bagel parquet format).
Shows: input images, system prompt, question, and output (thinking + answer).
"""
import os
import io
import re
import base64
import argparse
import pandas as pd
from PIL import Image
import html as html_module


def decode_image(image_bytes):
    """Decode image bytes to base64 data URI."""
    img = Image.open(io.BytesIO(image_bytes))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def format_output_text(text):
    """Format output text with colored think/answer tags."""
    text = html_module.escape(text)
    # Highlight <think>...</think>
    text = re.sub(
        r'&lt;think&gt;(.*?)&lt;/think&gt;',
        r'<span class="think-tag">&lt;think&gt;</span><span class="think-content">\1</span><span class="think-tag">&lt;/think&gt;</span>',
        text, flags=re.DOTALL
    )
    # Highlight <answer>...</answer>
    text = re.sub(
        r'&lt;answer&gt;(.*?)&lt;/answer&gt;',
        r'<span class="answer-tag">&lt;answer&gt;</span><span class="answer-content">\1</span><span class="answer-tag">&lt;/answer&gt;</span>',
        text, flags=re.DOTALL
    )
    return text


def generate_html(parquet_dir, output_path, max_samples=100, samples_per_chunk=None):
    """Generate HTML visualization from parquet files."""
    # Find all chunk files
    chunks = sorted([f for f in os.listdir(parquet_dir) if f.endswith('.parquet')])
    print(f"Found {len(chunks)} parquet chunks")

    samples = []
    for chunk_file in chunks:
        chunk_path = os.path.join(parquet_dir, chunk_file)
        df = pd.read_parquet(chunk_path)
        n_to_take = samples_per_chunk if samples_per_chunk else len(df)
        for idx in range(min(n_to_take, len(df))):
            row = df.iloc[idx]
            samples.append((chunk_file, idx, row))
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break

    print(f"Processing {len(samples)} samples...")

    # Build HTML
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TextCoT Dataset Visualization ({len(samples)} samples)</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0; padding: 20px;
    background: #f5f5f5;
}}
h1 {{
    text-align: center; color: #333;
    margin-bottom: 5px;
}}
.subtitle {{
    text-align: center; color: #666;
    margin-bottom: 20px; font-size: 14px;
}}
.controls {{
    text-align: center; margin-bottom: 20px;
    position: sticky; top: 0; background: #f5f5f5;
    padding: 10px 0; z-index: 100;
    border-bottom: 1px solid #ddd;
}}
.controls button {{
    padding: 8px 16px; margin: 0 5px;
    border: 1px solid #ccc; border-radius: 4px;
    cursor: pointer; background: white;
    font-size: 14px;
}}
.controls button:hover {{ background: #e0e0e0; }}
.controls button.active {{ background: #4CAF50; color: white; border-color: #4CAF50; }}
.sample {{
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    overflow: hidden;
}}
.sample-header {{
    background: #2196F3; color: white;
    padding: 10px 15px; font-weight: bold;
    display: flex; justify-content: space-between;
    cursor: pointer;
}}
.sample-header:hover {{ background: #1976D2; }}
.sample-body {{
    display: flex; flex-wrap: wrap;
    padding: 15px;
}}
.image-section {{
    flex: 0 0 auto;
    margin-right: 20px;
    margin-bottom: 10px;
}}
.image-section img {{
    max-width: 400px;
    max-height: 400px;
    border: 1px solid #ddd;
    border-radius: 4px;
}}
.image-label {{
    font-size: 12px; color: #666;
    margin-top: 4px; text-align: center;
}}
.text-section {{
    flex: 1; min-width: 300px;
}}
.field {{
    margin-bottom: 12px;
}}
.field-label {{
    font-weight: bold; color: #555;
    font-size: 13px; margin-bottom: 4px;
    text-transform: uppercase;
}}
.field-content {{
    background: #fafafa;
    border: 1px solid #eee;
    border-radius: 4px;
    padding: 10px;
    font-size: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
    max-height: 300px;
    overflow-y: auto;
}}
.think-tag {{ color: #9C27B0; font-weight: bold; }}
.think-content {{ color: #6A1B9A; }}
.answer-tag {{ color: #E65100; font-weight: bold; }}
.answer-content {{ color: #BF360C; font-weight: bold; font-size: 16px; }}
.collapsed .sample-body {{ display: none; }}
.stats {{
    display: flex; gap: 15px;
    justify-content: center;
    margin-bottom: 15px;
    flex-wrap: wrap;
}}
.stat-box {{
    background: white;
    padding: 10px 20px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    text-align: center;
}}
.stat-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
.stat-label {{ font-size: 12px; color: #666; }}
</style>
</head>
<body>
<h1>TextCoT Dataset Visualization</h1>
<div class="subtitle">Source: {parquet_dir}</div>

<div class="stats">
    <div class="stat-box">
        <div class="stat-value">{len(samples)}</div>
        <div class="stat-label">Samples Shown</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">{len(chunks)}</div>
        <div class="stat-label">Parquet Chunks</div>
    </div>
</div>

<div class="controls">
    <button onclick="expandAll()">Expand All</button>
    <button onclick="collapseAll()">Collapse All</button>
    <button onclick="toggleImages()" id="imgToggle">Hide Images</button>
    <span style="margin-left: 15px; color: #666;">Click headers to toggle individual samples</span>
</div>
""")

    for i, (chunk_name, chunk_idx, row) in enumerate(samples):
        # Extract data
        image_list = row['image_list']
        instruction_list = row['instruction_list']
        output_text_list = row['output_text_list']
        num_input_images = row['num_input_images']

        # Decode images
        image_data_uris = []
        for img_bytes in image_list:
            try:
                uri = decode_image(img_bytes)
                image_data_uris.append(uri)
            except Exception as e:
                image_data_uris.append(None)

        # Parse instructions
        system_prompt = instruction_list[0] if len(instruction_list) > 0 else ''
        question = instruction_list[1] if len(instruction_list) > 1 else ''

        # Parse output
        output_text = output_text_list[0] if len(output_text_list) > 0 else ''

        # Extract answer
        answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
        answer_short = answer_match.group(1).strip()[:50] if answer_match else 'N/A'

        # Build sample HTML
        html_parts.append(f'''
<div class="sample" id="sample-{i}">
    <div class="sample-header" onclick="toggleSample({i})">
        <span>Sample {i} (chunk: {chunk_name}, idx: {chunk_idx})</span>
        <span>Answer: {html_module.escape(answer_short)} | Images: {num_input_images}</span>
    </div>
    <div class="sample-body">
        <div class="image-section">
''')

        for j, uri in enumerate(image_data_uris):
            if uri:
                label = f"Input Image {j+1}" if j < num_input_images else f"Output Image {j+1}"
                html_parts.append(f'            <img src="{uri}" class="sample-img" alt="{label}"><div class="image-label">{label}</div>\n')

        html_parts.append('        </div>\n        <div class="text-section">\n')

        # System prompt (collapsed by default since they're all the same)
        html_parts.append(f'''            <div class="field">
                <div class="field-label">System Prompt</div>
                <div class="field-content" style="max-height:60px;">{html_module.escape(system_prompt)}</div>
            </div>
''')

        # Question
        html_parts.append(f'''            <div class="field">
                <div class="field-label">Question</div>
                <div class="field-content">{html_module.escape(question)}</div>
            </div>
''')

        # Output with formatting
        formatted_output = format_output_text(output_text)
        html_parts.append(f'''            <div class="field">
                <div class="field-label">Output (Thinking + Answer)</div>
                <div class="field-content">{formatted_output}</div>
            </div>
''')

        html_parts.append('        </div>\n    </div>\n</div>\n')

    html_parts.append("""
<script>
function toggleSample(i) {
    document.getElementById('sample-' + i).classList.toggle('collapsed');
}
function expandAll() {
    document.querySelectorAll('.sample').forEach(s => s.classList.remove('collapsed'));
}
function collapseAll() {
    document.querySelectorAll('.sample').forEach(s => s.classList.add('collapsed'));
}
function toggleImages() {
    var btn = document.getElementById('imgToggle');
    var imgs = document.querySelectorAll('.sample-img');
    if (btn.textContent === 'Hide Images') {
        imgs.forEach(img => img.style.display = 'none');
        btn.textContent = 'Show Images';
    } else {
        imgs.forEach(img => img.style.display = '');
        btn.textContent = 'Hide Images';
    }
}
</script>
</body>
</html>
""")

    html_content = ''.join(html_parts)
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"Saved HTML visualization to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize textcot training dataset as HTML')
    parser.add_argument('--parquet_dir', type=str, required=True,
                        help='Directory containing parquet chunks')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file path (default: auto-generated)')
    parser.add_argument('--max_samples', type=int, default=50,
                        help='Maximum total samples to visualize')
    parser.add_argument('--samples_per_chunk', type=int, default=None,
                        help='Max samples per chunk (default: no limit)')
    args = parser.parse_args()

    if args.output is None:
        dirname = os.path.basename(args.parquet_dir.rstrip('/'))
        args.output = os.path.join('visualization_scripts', f'viz_{dirname}.html')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_html(args.parquet_dir, args.output, args.max_samples, args.samples_per_chunk)
