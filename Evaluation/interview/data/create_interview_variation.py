#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def split_examples(examples):
    total = len(examples)
    train_count = math.ceil(total * 0.8)
    train_part = examples[:train_count]
    val_part = examples[train_count:]
    return train_part, val_part


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create interview variation with 80/20 train/val split per speaker."
    )
    parser.add_argument(
        "--input_dir",
        default="Evaluation/interview/data/processed",
        help="Directory with one JSON file per speaker.",
    )
    parser.add_argument(
        "--output_dir",
        default="Evaluation/interview/data/processed_val2",
        help="Directory to write split JSON files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    total_train = 0
    total_val = 0
    count_distribution = {}

    for file_path in json_files:
        with file_path.open("r", encoding="utf-8") as f:
            examples = json.load(f)

        if not isinstance(examples, list):
            raise ValueError(f"Expected top-level JSON array in {file_path}")

        count_distribution[len(examples)] = count_distribution.get(len(examples), 0) + 1
        train_part, val_part = split_examples(examples)

        transformed = []
        for item in train_part:
            new_item = dict(item)
            new_item["split"] = "train"
            transformed.append(new_item)

        for item in val_part:
            new_item = dict(item)
            new_item["split"] = "val"
            transformed.append(new_item)

        total_train += len(train_part)
        total_val += len(val_part)

        output_path = output_dir / file_path.name
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(transformed, f, indent=2, ensure_ascii=False)

    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"speaker_files: {len(json_files)}")
    print(f"examples_per_file: {count_distribution}")
    print(f"total_train_examples: {total_train}")
    print(f"total_val_examples: {total_val}")


if __name__ == "__main__":
    main()
