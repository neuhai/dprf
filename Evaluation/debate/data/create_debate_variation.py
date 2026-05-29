#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def split_utterances(utterances):
    total = len(utterances)
    train_count = math.ceil(total * 0.8)
    train_part = utterances[:train_count]
    val_part = utterances[train_count:]
    return train_part, val_part


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create debate variation with 80/20 utterance split."
    )
    parser.add_argument(
        "--input",
        default="Evaluation/debate/data/processed/debate_examples.json",
        help="Path to source debate examples file.",
    )
    parser.add_argument(
        "--output",
        default="Evaluation/debate/data/processed/debate_variation.json",
        help="Path to output variation file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array in input file.")

    transformed = []
    for item in data:
        if not isinstance(item, dict):
            continue

        new_item = dict(item)
        utterances = new_item.get("individual_utterances", [])
        if not isinstance(utterances, list):
            utterances = []

        train_part, val_part = split_utterances(utterances)
        new_item["individual_utterances_train"] = train_part
        new_item["individual_utterances_val"] = val_part
        new_item.pop("individual_utterances", None)
        transformed.append(new_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(transformed, f, indent=2, ensure_ascii=False)

    print(f"input_file: {input_path}")
    print(f"output_file: {output_path}")
    print(f"total_records: {len(transformed)}")


if __name__ == "__main__":
    main()
