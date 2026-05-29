#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute statistics for individual_utterances in debate examples."
    )
    parser.add_argument(
        "--input",
        default="Evaluation/debate/data/processed/debate_examples.json",
        help="Path to debate_examples.json",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array.")

    utterance_counts = []
    missing_field_count = 0
    invalid_field_count = 0

    for item in data:
        utterances = item.get("individual_utterances")
        if utterances is None:
            missing_field_count += 1
            continue
        if not isinstance(utterances, list):
            invalid_field_count += 1
            continue
        utterance_counts.append(len(utterances))

    if not utterance_counts:
        print("No valid individual_utterances found.")
        print(f"missing_field_count: {missing_field_count}")
        print(f"invalid_field_count: {invalid_field_count}")
        print(f"total_records: {len(data)}")
        return

    total_valid = len(utterance_counts)
    total_records = len(data)
    average_count = sum(utterance_counts) / total_valid

    print(f"input_file: {input_path}")
    print(f"total_records: {total_records}")
    print(f"valid_records: {total_valid}")
    print(f"missing_field_count: {missing_field_count}")
    print(f"invalid_field_count: {invalid_field_count}")
    print(f"avg_individual_utterances: {average_count:.4f}")
    print(f"min_individual_utterances: {min(utterance_counts)}")
    print(f"max_individual_utterances: {max(utterance_counts)}")


if __name__ == "__main__":
    main()
