"""
Convert linjieli222/tifa_train_v3 HuggingFace dataset into bagel format.

The tifa_train_v3 dataset has 8 configurations:
  dh_midpoint, td_ego_dir, td_ego_dir_arrow, td_ego_side,
  td_ego_side_arrow, td_midpoint, td_path, td_path_arrow

Each item has:
  - topdown_image: Top-down view (input image)
  - sideview_image: Side view (output image for visual thinking)
  - ego_images: Egocentric views (additional input images for ego configs)
  - question, answer, choices: VQA fields
  - sideview_desc: Description of side view (used for think text)
  - is_egocentric: Whether ego images are used

Variants:
  "default" - Visual CoT (vcot): short think text + generated sideview image + answer
    - image_list: [topdown, (ego...,) sideview]
    - output: <think>desc</think><image_start> [image] <image_end><answer>A</answer>

  "answer_only" - Answer only: no thinking, no image generation
    - image_list: [topdown, (ego...)]  (input images only)
    - output: <answer>A</answer>

  "text_cot" - Text-only CoT: longer detailed reasoning, no image generation
    - Requires textcot jsonl with reasoning_text
    - image_list: [topdown, (ego...)]  (input images only)
    - output: <think>detailed reasoning steps</think><answer>A</answer>
    - Training: --visual_gen False --mse_weight 0

  "mmcot" - Multimodal CoT: reasoning before + after sideview image
    - Requires mmcot jsonl with reasoning_thought_0, reasoning_thought_1
    - image_list: [topdown, (ego...,) sideview]
    - output: <think>thought_0</think><image_start> [image] <image_end><think>thought_1</think><answer>A</answer>
"""
import os
import io
import re
import json
import argparse
import pyarrow.parquet as pq
from PIL import ImageFile
from datasets import load_dataset

# Allow loading truncated images instead of raising OSError
ImageFile.LOAD_TRUNCATED_IMAGES = True


VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''

VLM_ANSWER_ONLY_SYSTEM_PROMPT = '''
Answer the question. Conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''

VARIANTS = ["default", "answer_only", "answer_only_think", "mmcot", "text_cot"]

ALL_CONFIGS = [
    "dh_midpoint",
    "td_ego_dir",
    "td_ego_dir_arrow",
    "td_ego_side",
    "td_ego_side_arrow",
    "td_midpoint",
    "td_path",
    "td_path_arrow",
]


def image_to_bytes(img):
    """Convert a PIL image to PNG bytes."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    b_io = io.BytesIO()
    img.save(b_io, format="PNG")
    return b_io.getvalue()


def _format_choices(item):
    """Format answer choices as lettered list."""
    choices = item.get('choices', [])
    if isinstance(choices, str):
        import ast
        try:
            choices = ast.literal_eval(choices)
        except Exception:
            choices = [choices]

    if not choices:
        return ""

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices_str = ""
    for i, choice in enumerate(choices):
        label = letters[i] if i < len(letters) else str(i)
        choices_str += f"\n({label}) {choice}"
    return choices_str


def _build_instruction_list(item, system_prompt):
    """Build instruction_list matching eval-time token sequence.

    Returns 2-element list:
      instruction_list[0] = system_prompt (added before images in KV cache)
      instruction_list[1] = full question text with <img><|image_N|></img> tags + choices
                            (added after images in KV cache)

    This matches the eval inferencer which processes:
      system_prompt → image tokens → text with <img> placeholders
    """
    question = item['question'].strip()
    choices_str = _format_choices(item)

    # Replace <image_N> with Bagel's <img><|image_N|></img> format
    question_with_tags = re.sub(
        r'<image_(\d+)>',
        r'<img><|image_\1|></img>',
        question,
    )

    return [system_prompt, question_with_tags + choices_str]


def _get_input_images(item):
    """Get input images (topdown + ego if present) as bytes."""
    topdown = item['topdown_image']
    raw_images = [topdown]

    ego_images = item.get('ego_images', None)
    if ego_images and isinstance(ego_images, list) and len(ego_images) > 0:
        valid_ego = [img for img in ego_images if img is not None]
        if valid_ego:
            raw_images.extend(valid_ego)

    return [image_to_bytes(img) for img in raw_images]


def transform_item_default(item):
    """Default variant: think text + generated sideview image + answer."""
    # Input images + sideview as output image
    input_bytes = _get_input_images(item)
    num_input_images = len(input_bytes)

    sideview = item['sideview_image']
    image_list = input_bytes + [image_to_bytes(sideview)]

    # Instruction: split around <image_1> so topdown image is interleaved in text
    instruction_list = _build_instruction_list(item, VLM_THINK_SYSTEM_PROMPT)

    # Output: think + image + answer
    answer = item.get('answer', '')
    answer_block = f"<answer>{answer}</answer>" if answer else ""

    sideview_desc = item.get('sideview_desc', '')
    # Remove all <image_N> tags from sideview description
    clean_desc = re.sub(r'\s*<image_\d+>\s*', ' ', sideview_desc)
    clean_desc = (
        clean_desc
        .replace(' ..', '.')
        .replace(' .', '.')
        .replace(':.', ':')
        .replace('  ', ' ')
        .strip()
    )

    part1 = f"<think>{clean_desc}</think><image_start>"
    part2 = f"<image_end>{answer_block}"
    output_text_list = [part1, part2]

    result = {
        "image_list": image_list,
        "instruction_list": instruction_list,
        "output_text_list": output_text_list,
    }
    if num_input_images > 1:
        result["num_input_images"] = num_input_images

    return result


def transform_item_answer_only(item):
    """Answer-only variant: no thinking, no image generation."""
    # Input images only (no sideview output)
    image_list = _get_input_images(item)
    num_input_images = len(image_list)

    # Instruction: split around <image_1> so topdown image is interleaved in text
    instruction_list = _build_instruction_list(item, VLM_ANSWER_ONLY_SYSTEM_PROMPT)

    # Output: just the answer
    answer = item.get('answer', '')
    output_text_list = [f"<answer>{answer}</answer>"]

    result = {
        "image_list": image_list,
        "num_input_images": num_input_images,
        "instruction_list": instruction_list,
        "output_text_list": output_text_list,
    }

    return result


def transform_item_answer_only_think(item):
    """Answer-only with VCoT system prompt — no thinking, no image gen."""
    image_list = _get_input_images(item)
    num_input_images = len(image_list)
    instruction_list = _build_instruction_list(item, VLM_THINK_SYSTEM_PROMPT)
    answer = item.get('answer', '')
    output_text_list = [f"<answer>{answer}</answer>"]
    return {
        "image_list": image_list,
        "num_input_images": num_input_images,
        "instruction_list": instruction_list,
        "output_text_list": output_text_list,
    }


def make_transform_item_mmcot(mmcot_lookup):
    """Create mmcot transform with pre-loaded lookup dict."""
    def transform_item_mmcot(item):
        """MMCoT variant: reasoning_thought_0 + sideview image + reasoning_thought_1 + answer."""
        qid = item['question_id']
        mmcot = mmcot_lookup.get(qid)
        if mmcot is None:
            return None

        # Input images + sideview as output image
        input_bytes = _get_input_images(item)
        num_input_images = len(input_bytes)

        sideview = item['sideview_image']
        image_list = input_bytes + [image_to_bytes(sideview)]

        # Instruction
        instruction_list = _build_instruction_list(item, VLM_THINK_SYSTEM_PROMPT)

        # Output: think_0 + image + think_1 + answer
        answer = item.get('answer', '')
        answer_block = f"<answer>{answer}</answer>" if answer else ""

        thought_0 = mmcot['reasoning_thought_0'].strip()
        thought_1 = mmcot['reasoning_thought_1'].strip()

        part1 = f"<think>{thought_0}</think><image_start>"
        part2 = f"<image_end><think>{thought_1}</think>{answer_block}"
        output_text_list = [part1, part2]

        result = {
            "image_list": image_list,
            "instruction_list": instruction_list,
            "output_text_list": output_text_list,
        }
        if num_input_images > 1:
            result["num_input_images"] = num_input_images

        return result
    return transform_item_mmcot


def make_transform_item_text_cot(textcot_lookup):
    """Create text_cot transform with pre-loaded lookup dict."""
    def transform_item_text_cot(item):
        """Text CoT variant: text-only reasoning + answer, no image generation."""
        qid = item['question_id']
        textcot = textcot_lookup.get(qid)
        if textcot is None:
            return None

        # Input images only (no sideview)
        image_list = _get_input_images(item)
        num_input_images = len(image_list)

        # Instruction
        instruction_list = _build_instruction_list(item, VLM_THINK_SYSTEM_PROMPT)

        # Output: think + answer (no image generation)
        answer = item.get('answer', '')
        answer_block = f"<answer>{answer}</answer>" if answer else ""

        reasoning = textcot['reasoning_text'].strip()
        output_text_list = [f"<think>{reasoning}</think>{answer_block}"]

        result = {
            "image_list": image_list,
            "num_input_images": num_input_images,
            "instruction_list": instruction_list,
            "output_text_list": output_text_list,
        }

        return result
    return transform_item_text_cot


TRANSFORM_FNS = {
    "default": transform_item_default,
    "answer_only": transform_item_answer_only,
    "answer_only_think": transform_item_answer_only_think,
    # "mmcot" and "text_cot" are handled specially in process_config
}


def _load_mmcot(mmcot_dir, config_name):
    """Load mmcot jsonl as a dict keyed by question_id."""
    jsonl_path = os.path.join(mmcot_dir, f"{config_name}.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"MMCoT file not found: {jsonl_path}")

    lookup = {}
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            lookup[item['question_id']] = item
    print(f"Loaded {len(lookup)} mmcot entries from {jsonl_path}")
    return lookup


def process_config(config_name, variant="default",
                   dataset_name="linjieli222/tifa_train_v3",
                   output_base_dir=None, num_shards=5, num_proc=4,
                   mmcot_dir=None, textcot_dir=None):
    """Process a single dataset configuration into bagel format."""
    variant_suffix = "" if variant == "default" else f"_{variant}"
    dir_name = f"tifa_train_v3_{config_name}{variant_suffix}"

    print(f"\n{'='*60}")
    print(f"Processing config: {config_name} (variant: {variant})")
    print(f"{'='*60}")

    # Load dataset
    print(f"Loading {dataset_name} [{config_name}]...")
    ds = load_dataset(dataset_name, config_name, split="train")
    print(f"Loaded {len(ds)} samples")

    # Get transform function
    if variant == "mmcot":
        if mmcot_dir is None:
            raise ValueError("--mmcot_dir is required for mmcot variant")
        mmcot_lookup = _load_mmcot(mmcot_dir, config_name)
        # Use select with pre-computed indices (memory-efficient, avoids filter OOM)
        mmcot_qids = set(mmcot_lookup.keys())
        all_qids = ds['question_id']
        indices = [i for i, qid in enumerate(all_qids) if qid in mmcot_qids]
        ds = ds.select(indices)
        print(f"Selected {len(ds)} samples with mmcot entries")
        transform_fn = make_transform_item_mmcot(mmcot_lookup)
    elif variant == "text_cot":
        if textcot_dir is None:
            raise ValueError("--textcot_dir is required for text_cot variant")
        textcot_lookup = _load_mmcot(textcot_dir, config_name)  # same jsonl loader
        textcot_qids = set(textcot_lookup.keys())
        all_qids = ds['question_id']
        indices = [i for i, qid in enumerate(all_qids) if qid in textcot_qids]
        ds = ds.select(indices)
        print(f"Selected {len(ds)} samples with text_cot entries")
        transform_fn = make_transform_item_text_cot(textcot_lookup)
    else:
        transform_fn = TRANSFORM_FNS[variant]

    # Transform
    print(f"Transforming to bagel format (variant={variant})...")
    new_ds = ds.map(
        transform_fn,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        features=None,
        writer_batch_size=100,
    )

    # Output directory
    if output_base_dir is None:
        output_base_dir = os.path.join("/gpfs/projects/krishna/linjli", "bagel_example", "editing")

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
    print(f"Total samples for {config_name} ({variant}): {total_samples}")

    # Verify first item
    print(f"\n--- Verification ({config_name}, {variant}) ---")
    sample = new_ds[0]
    print(f"  image_list length: {len(sample['image_list'])}")
    print(f"  instruction_list length: {len(sample['instruction_list'])}")
    for i, txt in enumerate(sample['instruction_list']):
        print(f"  instruction_list[{i}][:200]: {txt[:200]}...")
    for i, txt in enumerate(sample['output_text_list']):
        print(f"  output_text_list[{i}][:200]: {txt[:200]}...")
    if 'num_input_images' in sample:
        print(f"  num_input_images: {sample['num_input_images']}")

    return {
        "config": config_name,
        "variant": variant,
        "data_dir": data_dir,
        "num_files": num_shards,
        "num_total_samples": total_samples,
        "parquet_info_path": json_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert tifa_train_v3 dataset to bagel format"
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help=f"Config names to process. Default: all configs. Options: {ALL_CONFIGS}"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output base directory. Default: /gpfs/projects/krishna/linjli/bagel_example/editing"
    )
    parser.add_argument(
        "--num_shards", type=int, default=5,
        help="Number of parquet shards per config (default: 5)"
    )
    parser.add_argument(
        "--variants", nargs="*", default=None,
        help=f"Variants to generate. Default: all. Options: {VARIANTS}"
    )
    parser.add_argument(
        "--num_proc", type=int, default=4,
        help="Number of parallel processes for dataset.map (default: 4)"
    )
    parser.add_argument(
        "--mmcot_dir", type=str, default=None,
        help="Directory containing mmcot jsonl files (required for mmcot variant). "
             "Each file should be named {config_name}.jsonl"
    )
    parser.add_argument(
        "--textcot_dir", type=str, default=None,
        help="Directory containing text_cot jsonl files (required for text_cot variant). "
             "Each file should be named {config_name}.jsonl"
    )
    args = parser.parse_args()

    configs = args.configs if args.configs else ALL_CONFIGS
    variants = args.variants if args.variants else VARIANTS

    print(f"Will process configs: {configs}")
    print(f"Variants: {variants}")
    print(f"Output dir: {args.output_dir or '/gpfs/projects/krishna/linjli/bagel_example/editing'}")

    registry_entries = {}
    for config_name in configs:
        if config_name not in ALL_CONFIGS:
            print(f"WARNING: Unknown config '{config_name}', skipping.")
            continue
        for variant in variants:
            if variant not in VARIANTS:
                print(f"WARNING: Unknown variant '{variant}', skipping.")
                continue
            info = process_config(
                config_name,
                variant=variant,
                output_base_dir=args.output_dir,
                num_shards=args.num_shards,
                num_proc=args.num_proc,
                mmcot_dir=args.mmcot_dir,
                textcot_dir=args.textcot_dir,
            )
            key = f"{config_name}_{variant}" if variant != "default" else config_name
            registry_entries[key] = info

    # Print dataset_info.py registry entries
    print(f"\n{'='*60}")
    print("Dataset registry entries for data/dataset_info.py:")
    print(f"{'='*60}")
    for key, info in registry_entries.items():
        reg_key = f"tifa_v3_{key}"
        print(f"        '{reg_key}': {{")
        print(f"            'data_dir': '{info['data_dir']}',")
        print(f"            'num_files': {info['num_files']},")
        print(f"            'num_total_samples': {info['num_total_samples']},")
        print(f"            'parquet_info_path': '{info['parquet_info_path']}',")
        print(f"        }},")

    print("\nDone!")


if __name__ == "__main__":
    main()
