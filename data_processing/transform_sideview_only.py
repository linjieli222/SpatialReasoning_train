"""
Convert linjieli222/ai2thor-sideview-only HuggingFace dataset into bagel format.

This dataset is sideview-only image generation (no MCQ answer).
Each item has:
  - topdown_image: Top-down view (input image)
  - sideview_images: list of sideview images (output images to generate)
  - input_images: list of input images
  - question: question text with <image_N> placeholders
  - answer: empty string (no MCQ answer)
  - sideview_desc: description of sideview (used for think text)

Output format (bagel):
  - image_list: [topdown, sideview]
  - instruction_list: [system_prompt, question_with_tags]
  - output_text_list: ["<think>desc</think><image_start>", "<image_end>"]
  - num_input_images: 1
"""
import os
import io
import re
import json
import argparse
import pyarrow.parquet as pq
from PIL import ImageFile
from datasets import load_dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''


def image_to_bytes(img):
    """Convert a PIL image to PNG bytes."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    b_io = io.BytesIO()
    img.save(b_io, format="PNG")
    return b_io.getvalue()


def _build_instruction_list(item, system_prompt):
    """Build instruction_list matching eval-time token sequence."""
    question = item['question'].strip()

    # Replace <image_N> with Bagel's <img><|image_N|></img> format
    question_with_tags = re.sub(
        r'<image_(\d+)>',
        r'<img><|image_\1|></img>',
        question,
    )

    return [system_prompt, question_with_tags]


def transform_item(item):
    """Transform a sideview-only item to bagel format."""
    # Input: topdown image
    topdown = item['topdown_image']
    input_bytes = [image_to_bytes(topdown)]
    num_input_images = 1

    # Output: sideview image
    sideview_images = item.get('sideview_images', [])
    if not sideview_images or len(sideview_images) == 0:
        return None
    sideview = sideview_images[0]
    image_list = input_bytes + [image_to_bytes(sideview)]

    # Instruction
    instruction_list = _build_instruction_list(item, VLM_THINK_SYSTEM_PROMPT)

    # Output text: think + image, no answer block
    sideview_desc = item.get('sideview_desc', '')
    clean_desc = re.sub(r'\s*<image_\d+>\s*', ' ', sideview_desc)
    # Change "Your" to "my" for first-person perspective
    clean_desc = re.sub(r'\bYour\b', 'my', clean_desc)
    clean_desc = (
        clean_desc
        .replace(' ..', '.')
        .replace(' .', '.')
        .replace(':.', ':')
        .replace('  ', ' ')
        .strip()
    )

    part1 = f"<think>{clean_desc}</think><image_start>"
    part2 = "<image_end>"
    output_text_list = [part1, part2]

    return {
        "image_list": image_list,
        "num_input_images": num_input_images,
        "instruction_list": instruction_list,
        "output_text_list": output_text_list,
    }


def process(dataset_name, split, output_base_dir, num_shards=5, num_proc=4):
    """Process the dataset into bagel format."""
    dir_name = f"sideview_only_{split}"

    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name} split={split}")
    print(f"{'='*60}")

    print(f"Loading dataset...")
    ds = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(ds)} samples")

    print(f"Transforming to bagel format...")
    new_ds = ds.map(
        transform_item,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        writer_batch_size=100,
    )

    # Filter out None results
    before = len(new_ds)
    new_ds = new_ds.filter(lambda x: x['image_list'] is not None)
    after = len(new_ds)
    if before != after:
        print(f"Filtered out {before - after} samples without sideview images")

    # Output directory
    data_dir = os.path.join(output_base_dir, dir_name)
    info_dir = os.path.join(output_base_dir, "parquet_info")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    # Shard and save
    metadata_info = {}
    total_samples = 0

    print(f"Saving to {num_shards} shards in {data_dir}...")
    for i in range(num_shards):
        shard = new_ds.shard(num_shards=num_shards, index=i)
        filename = f"chunk_{i}.parquet"
        filepath = os.path.join(data_dir, filename)

        shard.to_parquet(filepath)

        pf = pq.ParquetFile(filepath)
        metadata_info[filepath] = {
            "num_row_groups": pf.num_row_groups,
            "num_rows": shard.num_rows,
        }
        total_samples += shard.num_rows
        print(f"  Saved {filename}: {shard.num_rows} rows")

    # Save metadata JSON
    json_path = os.path.join(info_dir, f"{dir_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata_info, f, indent=4)

    print(f"Saved metadata to {json_path}")
    print(f"Total samples: {total_samples}")

    # Verify first item
    print(f"\n--- Verification ---")
    sample = new_ds[0]
    print(f"  image_list length: {len(sample['image_list'])}")
    print(f"  num_input_images: {sample['num_input_images']}")
    print(f"  instruction_list length: {len(sample['instruction_list'])}")
    for i, txt in enumerate(sample['instruction_list']):
        print(f"  instruction_list[{i}][:200]: {txt[:200]}...")
    for i, txt in enumerate(sample['output_text_list']):
        print(f"  output_text_list[{i}][:200]: {txt[:200]}...")

    return {
        "data_dir": data_dir,
        "num_files": num_shards,
        "num_total_samples": total_samples,
        "parquet_info_path": json_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert ai2thor-sideview-only dataset to bagel format"
    )
    parser.add_argument(
        "--dataset_name", type=str,
        default="linjieli222/ai2thor-sideview-only",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split", type=str, default="td_path",
        help="Dataset split to process (default: td_path)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output base directory (default: /gpfs/projects/krishna/linjli/bagel_example/editing)"
    )
    parser.add_argument(
        "--num_shards", type=int, default=5,
        help="Number of parquet shards (default: 5)"
    )
    parser.add_argument(
        "--num_proc", type=int, default=4,
        help="Number of parallel processes (default: 4)"
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "/gpfs/projects/krishna/linjli/bagel_example/editing"

    info = process(
        args.dataset_name,
        args.split,
        args.output_dir,
        num_shards=args.num_shards,
        num_proc=args.num_proc,
    )

    print(f"\n{'='*60}")
    print("Dataset registry entry for data/dataset_info.py:")
    print(f"{'='*60}")
    key = f"sideview_only_{args.split}"
    print(f"        '{key}': {{")
    print(f"            'data_dir': '{info['data_dir']}',")
    print(f"            'num_files': {info['num_files']},")
    print(f"            'num_total_samples': {info['num_total_samples']},")
    print(f"            'parquet_info_path': '{info['parquet_info_path']}',")
    print(f"        }},")

    print("\nDone!")


if __name__ == "__main__":
    main()
